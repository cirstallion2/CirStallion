"""
Signal detector — the core edge logic.

Pipeline per tick
─────────────────
1. Compute theoretical Polymarket probability from spot BTC momentum.
2. Compare against actual CLOB mid-price → lag %.
3. If lag > 0.3 %: pass through MiroFish graph filter.
4. If graph confirms direction AND TradingView / CryptoQuant don't conflict:
   emit a Signal.

Theoretical probability model
──────────────────────────────
   base          = 0.50  (BTC 5-min direction is ~coin-flip)
   momentum_adj  = recent_30s_pct_change × 8.0   (capped ±0.30)
   theo          = clamp(base + momentum_adj, 0.10, 0.90)

The "lag" is then:
   lag = (theo - actual_clob_mid)   for an UP signal
   lag = (actual_clob_mid - theo)   for a DOWN signal
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
import numpy as np

from .kraken_feed import KrakenFeed
from .graph_engine import MiroFishEngine, NodeSignals
from .polymarket_client import ActiveMarket, PolymarketClient

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    active: bool
    direction: str              # "UP" | "DOWN"
    lag_pct: float              # e.g. 0.0041 = 0.41 %
    confidence: float           # graph convergence score 0–1
    graph_direction: str        # "BULL" | "BEAR" | "NEUTRAL"
    theo_price: float           # estimated fair probability
    clob_price: float           # actual CLOB mid for the relevant token
    token_id: str               # token to buy
    market: Optional[object] = None
    timestamp: float = field(default_factory=time.monotonic)

    def __bool__(self) -> bool:
        return self.active


_INACTIVE = Signal(
    active=False, direction="", lag_pct=0.0, confidence=0.0,
    graph_direction="NEUTRAL", theo_price=0.0, clob_price=0.0, token_id="",
)


class SignalDetector:
    """
    Combines Kraken spot data, MiroFish graph state, and Polymarket CLOB
    prices to detect exploitable CLOB-lag opportunities.

    Optional signal layers
    ──────────────────────
    TradingView : polled every ~60 s via tradingview-ta.
    CryptoQuant : polled every ~300 s via REST API.
    Both layers are skipped gracefully if credentials are missing or
    the library is unavailable.
    """

    _MOMENTUM_SENSITIVITY = 8.0    # how aggressively spot change shifts theo prob
    _TV_REFRESH_S = 60.0
    _CQ_REFRESH_S = 300.0

    def __init__(
        self,
        feed: KrakenFeed,
        graph: MiroFishEngine,
        poly: PolymarketClient,
        config,
    ):
        self._feed = feed
        self._graph = graph
        self._poly = poly
        self._cfg = config
        self._risk_cfg = config.risk

        # Cached slow signals
        self._tv_signal: float = 0.0           # TradingView [-1, +1]
        self._cq_signal: float = 0.0           # CryptoQuant [-1, +1]
        self._fear_greed: float = 0.0          # [-1, +1]

        self._tv_last_refresh: float = 0.0
        self._cq_last_refresh: float = 0.0
        self._fg_last_refresh: float = 0.0

        self._session: Optional[aiohttp.ClientSession] = None

    # ─── Public API ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    async def detect(self) -> Signal:
        """Run one detection pass. Returns Signal(active=True) if edge found."""
        if not self._feed.is_ready():
            return _INACTIVE
        if not self._poly.active_markets:
            return _INACTIVE

        # ── Refresh slow signal layers ────────────────────────────────────
        now = time.monotonic()
        if now - self._tv_last_refresh > self._TV_REFRESH_S:
            asyncio.create_task(self._refresh_tradingview())
        if now - self._cq_last_refresh > self._CQ_REFRESH_S:
            asyncio.create_task(self._refresh_cryptoquant())
        if now - self._fg_last_refresh > 60.0:
            asyncio.create_task(self._refresh_fear_greed())

        # ── Compute theoretical probability ───────────────────────────────
        raw_change = self._feed.get_raw_change_pct(window_s=30.0)
        momentum_adj = float(np.clip(raw_change * self._MOMENTUM_SENSITIVITY, -0.30, 0.30))
        theo_up = float(np.clip(0.50 + momentum_adj, 0.10, 0.90))
        theo_down = 1.0 - theo_up

        # ── Find the market with the biggest lag ──────────────────────────
        best_signal: Optional[Signal] = None

        for mkt in self._poly.active_markets:
            # Check UP lag
            yes_mid = self._poly.get_mid_price(mkt.yes_token_id)
            no_mid  = self._poly.get_mid_price(mkt.no_token_id)

            if yes_mid <= 0 or no_mid <= 0:
                continue

            lag_up   = theo_up - yes_mid     # positive → CLOB under-pricing UP
            lag_down = theo_down - no_mid    # positive → CLOB under-pricing DOWN

            if abs(lag_up) >= abs(lag_down) and lag_up > self._risk_cfg.min_edge_pct:
                candidate = self._build_signal("UP", lag_up, theo_up, yes_mid, mkt.yes_token_id, mkt)
            elif lag_down > self._risk_cfg.min_edge_pct:
                candidate = self._build_signal("DOWN", lag_down, theo_down, no_mid, mkt.no_token_id, mkt)
            else:
                continue

            # Liquidity gate
            liq = self._poly.get_liquidity(candidate.token_id, depth_usdc=self._cfg.min_liquidity_usdc)
            if liq < self._cfg.min_liquidity_usdc:
                logger.debug("Thin liquidity (%.0f USDC) on %s — skip", liq, candidate.token_id[:12])
                continue

            if best_signal is None or candidate.lag_pct > best_signal.lag_pct:
                best_signal = candidate

        if best_signal is None:
            return _INACTIVE

        # ── MiroFish graph confirmation ───────────────────────────────────
        signals = self._build_node_signals(raw_change, best_signal.lag_pct)
        self._graph.update(signals)
        self._graph.simulate()
        confidence, graph_dir = self._graph.get_convergence()

        best_signal.confidence = confidence
        best_signal.graph_direction = graph_dir

        # Graph must agree with direction (or be NEUTRAL — we allow that)
        required_graph = "BULL" if best_signal.direction == "UP" else "BEAR"
        if graph_dir not in (required_graph, "NEUTRAL"):
            logger.debug(
                "Graph vetoed: direction=%s graph=%s conf=%.2f",
                best_signal.direction, graph_dir, confidence,
            )
            return _INACTIVE

        # ── TradingView / CryptoQuant conflict check ──────────────────────
        if not self._passes_external_filters(best_signal.direction):
            return _INACTIVE

        best_signal.active = True
        logger.info(
            "SIGNAL | dir=%-4s lag=+%.4f theo=%.4f clob=%.4f conf=%.2f graph=%s",
            best_signal.direction, best_signal.lag_pct,
            best_signal.theo_price, best_signal.clob_price,
            confidence, graph_dir,
        )
        return best_signal

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _build_signal(
        self,
        direction: str,
        lag: float,
        theo: float,
        clob_mid: float,
        token_id: str,
        mkt: ActiveMarket,
    ) -> Signal:
        return Signal(
            active=False,   # set to True only after all filters pass
            direction=direction,
            lag_pct=lag,
            confidence=0.0,
            graph_direction="NEUTRAL",
            theo_price=theo,
            clob_price=clob_mid,
            token_id=token_id,
            market=mkt,
        )

    def _build_node_signals(self, raw_change: float, clob_lag: float) -> NodeSignals:
        vol_raw = self._feed.get_volume_ratio() - 1.0
        return NodeSignals(
            price_momentum=self._feed.get_momentum(30.0),
            volume_ratio=float(np.clip(vol_raw * 0.5, -1.0, 1.0)),
            rsi=self._feed.get_rsi_signal(),
            macd=self._feed.get_macd_signal(),
            exchange_flows=self._cq_signal,
            tradingview=self._tv_signal,
            fear_greed=self._fear_greed,
            clob_lag=float(np.clip(clob_lag * 30.0, -1.0, 1.0)),
        )

    def _passes_external_filters(self, direction: str) -> bool:
        """
        Returns False only if TradingView or CryptoQuant give a *strong*
        conflicting signal (value < -0.5 or > +0.5 in the wrong direction).
        Neutral readings always pass.
        """
        target = 1.0 if direction == "UP" else -1.0

        if self._tv_signal != 0.0 and abs(self._tv_signal) > 0.5:
            if self._tv_signal * target < 0:
                logger.debug("TradingView conflicts (%.2f) — skip", self._tv_signal)
                return False

        if self._cq_signal != 0.0 and abs(self._cq_signal) > 0.5:
            if self._cq_signal * target < 0:
                logger.debug("CryptoQuant conflicts (%.2f) — skip", self._cq_signal)
                return False

        return True

    # ─── Slow signal refreshers ───────────────────────────────────────────────

    async def _refresh_tradingview(self) -> None:
        self._tv_last_refresh = time.monotonic()
        try:
            from tradingview_ta import TA_Handler, Interval, Exchange  # type: ignore
            handler = TA_Handler(
                symbol=self._cfg.tradingview.symbol,
                exchange=self._cfg.tradingview.exchange,
                screener=self._cfg.tradingview.screener,
                interval=Interval.INTERVAL_5_MINUTES,
            )
            loop = asyncio.get_event_loop()
            analysis = await loop.run_in_executor(None, handler.get_analysis)
            rec = analysis.summary.get("RECOMMENDATION", "NEUTRAL")
            self._tv_signal = {
                "STRONG_BUY": 1.0, "BUY": 0.6,
                "NEUTRAL": 0.0,
                "SELL": -0.6, "STRONG_SELL": -1.0,
            }.get(rec, 0.0)
            logger.debug("TradingView signal: %s → %.1f", rec, self._tv_signal)
        except Exception as exc:
            logger.debug("TradingView refresh failed: %s", exc)

    async def _refresh_cryptoquant(self) -> None:
        self._cq_last_refresh = time.monotonic()
        api_key = self._cfg.cryptoquant.api_key
        if not api_key:
            return
        url = f"{self._cfg.cryptoquant.base_url}/btc/exchange-flows/all-exchange/netflow"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with self._session.get(url, headers=headers, params={"window": "hour", "limit": 1}) as resp:
                data = await resp.json()
                # Netflow: negative = outflow (bullish), positive = inflow (bearish)
                rows = data.get("result", {}).get("data", [])
                if rows:
                    netflow = float(rows[-1].get("value", 0))
                    # Normalise: ±5,000 BTC/hour → ±1
                    self._cq_signal = float(np.clip(-netflow / 5000.0, -1.0, 1.0))
                    logger.debug("CryptoQuant netflow: %.0f → signal=%.2f", netflow, self._cq_signal)
        except Exception as exc:
            logger.debug("CryptoQuant refresh failed: %s", exc)

    async def _refresh_fear_greed(self) -> None:
        self._fg_last_refresh = time.monotonic()
        url = "https://api.alternative.me/fng/?limit=1"
        try:
            async with self._session.get(url) as resp:
                data = await resp.json()
                val = int(data["data"][0]["value"])
                self._fear_greed = float(np.clip((val - 50) / 50.0, -1.0, 1.0))
                logger.debug("Fear & Greed: %d → signal=%.2f", val, self._fear_greed)
        except Exception as exc:
            logger.debug("Fear & Greed refresh failed: %s", exc)
