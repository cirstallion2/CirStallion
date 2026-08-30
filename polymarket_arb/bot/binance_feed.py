"""
Binance data feed — real-time BTC price via WebSocket + 5M klines via REST.
Maintains a rolling price history used by the signal detector to measure
short-term momentum before comparing against Polymarket CLOB prices.
"""

import asyncio
import json
import logging
import time
from collections import deque
from typing import List, Optional, Tuple

import aiohttp
import numpy as np
import websockets

logger = logging.getLogger(__name__)


class BinanceFeed:
    """
    Streams BTC/USDT trade ticks and 5-minute klines from Binance.

    Attributes
    ----------
    latest_price   : most recent trade price
    price_history  : deque of (timestamp, price) tuples, last 2,000 ticks
    klines         : last 200 closed 5-minute candles (raw Binance format)
    """

    def __init__(self, config):
        self._symbol = config.symbol                   # "BTCUSDT"
        self._interval = config.kline_interval         # "5m"
        self._rest_url = config.rest_url
        self._reconnect_delay = config.reconnect_delay_s
        self._kline_limit = config.kline_limit

        self.latest_price: float = 0.0
        self.price_history: deque = deque(maxlen=2000)
        self.klines: List[list] = []

        self._running = False
        self._ws_stream = (
            f"wss://stream.binance.com:9443/ws/"
            f"{self._symbol.lower()}@trade/"
            f"{self._symbol.lower()}@kline_{self._interval}"
        )

    # ─── Public API ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the feed (call with asyncio.create_task)."""
        self._running = True
        await self._fetch_klines()          # warm up klines before WS starts
        logger.info("BinanceFeed: klines loaded, connecting WebSocket…")

        while self._running:
            try:
                await self._run_ws()
            except Exception as exc:
                logger.warning("BinanceFeed WS error: %s — reconnecting in %.1fs", exc, self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)

    def stop(self) -> None:
        self._running = False

    def get_momentum(self, window_s: float = 30.0) -> float:
        """
        Price-change % over the last *window_s* seconds (normalised to ±1).
        Returns 0.0 if history is too thin.
        """
        if not self.price_history or self.latest_price == 0:
            return 0.0
        now = time.monotonic()
        cutoff = now - window_s
        anchor = next(
            (p for t, p in reversed(self.price_history) if t <= cutoff), None
        )
        if anchor is None or anchor == 0:
            return 0.0
        raw_pct = (self.latest_price - anchor) / anchor
        return float(np.clip(raw_pct * 20.0, -1.0, 1.0))   # ±5% → ±1

    def get_raw_change_pct(self, window_s: float = 30.0) -> float:
        """Raw % change (not normalised) for lag-detection arithmetic."""
        if not self.price_history or self.latest_price == 0:
            return 0.0
        now = time.monotonic()
        cutoff = now - window_s
        anchor = next(
            (p for t, p in reversed(self.price_history) if t <= cutoff), None
        )
        if anchor is None or anchor == 0:
            return 0.0
        return (self.latest_price - anchor) / anchor

    def get_rsi_signal(self) -> float:
        """RSI-based signal from 5M klines, normalised to [-1, +1]."""
        closes = self._kline_closes()
        if len(closes) < 15:
            return 0.0
        rsi = self._calc_rsi(closes, period=14)
        return float(np.clip((rsi - 50.0) / 50.0, -1.0, 1.0))

    def get_macd_signal(self) -> float:
        """MACD histogram normalised to [-1, +1]."""
        closes = self._kline_closes()
        if len(closes) < 30:
            return 0.0
        ema12 = self._ema(closes, 12)
        ema26 = self._ema(closes, 26)
        macd_line = ema12 - ema26
        signal_line = self._ema(np.array([macd_line] * 9), 9)  # simplified
        hist = macd_line - signal_line[-1] if hasattr(signal_line, '__len__') else macd_line - signal_line
        # Normalise by recent price to get a dimensionless ratio
        scale = self.latest_price * 0.001 if self.latest_price else 1.0
        return float(np.clip(hist / scale, -1.0, 1.0))

    def get_volume_ratio(self, lookback: int = 20) -> float:
        """Current 5M volume vs *lookback*-period average."""
        if len(self.klines) < lookback + 1:
            return 1.0
        recent_vol = float(self.klines[-1][5])
        avg_vol = float(np.mean([float(k[5]) for k in self.klines[-lookback - 1:-1]]))
        return recent_vol / avg_vol if avg_vol > 0 else 1.0

    def is_ready(self) -> bool:
        """True once we have live price AND at least one kline."""
        return self.latest_price > 0 and len(self.klines) > 0

    # ─── Internal helpers ────────────────────────────────────────────────────

    async def _run_ws(self) -> None:
        async with websockets.connect(self._ws_stream, ping_interval=20) as ws:
            logger.info("BinanceFeed: WebSocket connected")
            async for raw in ws:
                if not self._running:
                    break
                self._handle_message(raw)

    def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        evt = msg.get("e")
        if evt == "trade":
            price = float(msg["p"])
            self.latest_price = price
            self.price_history.append((time.monotonic(), price))
        elif evt == "kline" and msg["k"]["x"]:   # closed candle
            asyncio.create_task(self._fetch_klines())

    async def _fetch_klines(self) -> None:
        url = f"{self._rest_url}/api/v3/klines"
        params = {
            "symbol": self._symbol,
            "interval": self._interval,
            "limit": self._kline_limit,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    data = await resp.json()
                    if isinstance(data, list) and data:
                        self.klines = data
        except Exception as exc:
            logger.warning("BinanceFeed: kline fetch failed: %s", exc)

    def _kline_closes(self) -> np.ndarray:
        return np.array([float(k[4]) for k in self.klines])

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> float:
        if len(values) < period:
            return float(values.mean())
        k = 2.0 / (period + 1)
        ema = float(values[0])
        for v in values[1:]:
            ema = v * k + ema * (1 - k)
        return ema

    @staticmethod
    def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains[-period:].mean()
        avg_loss = losses[-period:].mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1 + rs))
