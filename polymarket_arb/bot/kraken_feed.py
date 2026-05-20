"""
Kraken data feed — real-time BTC price via WebSocket + 5M OHLC via REST.

Kraken is US-compliant and used by the existing CirStallion ticker.
Public interface is identical to the old BinanceFeed so no downstream
changes are needed.

WebSocket streams
─────────────────
  wss://ws.kraken.com
  subscribe → "trade"  for tick-by-tick price
  subscribe → "ohlc-5" for closed 5-minute candles

REST
────
  GET https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=5
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import List

import aiohttp
import numpy as np
import websockets

logger = logging.getLogger(__name__)

_PAIR   = "XBT/USD"
_SYMBOL = "XBTUSD"


class KrakenFeed:
    """
    Streams BTC/USD trade ticks and 5-minute OHLC from Kraken.

    Attributes
    ----------
    latest_price   : most recent trade price
    price_history  : deque of (monotonic_timestamp, price) — last 2,000 ticks
    klines         : last 200 closed 5-minute candles
                     internal format: [time, open, high, low, close, volume]
    """

    _WS_URL   = "wss://ws.kraken.com"
    _REST_URL = "https://api.kraken.com/0/public/OHLC"

    def __init__(self, config):
        self._reconnect_delay = config.reconnect_delay_s
        self._kline_limit     = config.kline_limit

        self.latest_price: float = 0.0
        self.price_history: deque = deque(maxlen=2000)
        self.klines: List[list] = []

        self._running = False

    # ─── Public API (identical to BinanceFeed) ───────────────────────────────

    async def start(self) -> None:
        self._running = True
        await self._fetch_klines()
        logger.info("KrakenFeed: klines loaded (%d candles), connecting WebSocket…", len(self.klines))

        while self._running:
            try:
                await self._run_ws()
            except Exception as exc:
                logger.warning("KrakenFeed WS error: %s — reconnecting in %.1fs", exc, self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)

    def stop(self) -> None:
        self._running = False

    def get_momentum(self, window_s: float = 30.0) -> float:
        """Price-change % over last *window_s* seconds, normalised to ±1."""
        if not self.price_history or self.latest_price == 0:
            return 0.0
        now    = time.monotonic()
        cutoff = now - window_s
        anchor = next(
            (p for t, p in reversed(self.price_history) if t <= cutoff), None
        )
        if anchor is None or anchor == 0:
            return 0.0
        raw = (self.latest_price - anchor) / anchor
        return float(np.clip(raw * 20.0, -1.0, 1.0))

    def get_raw_change_pct(self, window_s: float = 30.0) -> float:
        """Unnormalised % change — used by SignalDetector for lag arithmetic."""
        if not self.price_history or self.latest_price == 0:
            return 0.0
        now    = time.monotonic()
        cutoff = now - window_s
        anchor = next(
            (p for t, p in reversed(self.price_history) if t <= cutoff), None
        )
        if anchor is None or anchor == 0:
            return 0.0
        return (self.latest_price - anchor) / anchor

    def get_rsi_signal(self) -> float:
        """RSI-14 on 5M closes, normalised (RSI-50)/50 → [-1, +1]."""
        closes = self._closes()
        if len(closes) < 15:
            return 0.0
        rsi = _calc_rsi(closes, period=14)
        return float(np.clip((rsi - 50.0) / 50.0, -1.0, 1.0))

    def get_macd_signal(self) -> float:
        """MACD histogram on 5M closes, normalised to [-1, +1]."""
        closes = self._closes()
        if len(closes) < 30:
            return 0.0
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        hist  = ema12 - ema26
        scale = self.latest_price * 0.001 if self.latest_price else 1.0
        return float(np.clip(hist / scale, -1.0, 1.0))

    def get_volume_ratio(self, lookback: int = 20) -> float:
        """Current candle volume vs *lookback*-period average."""
        if len(self.klines) < lookback + 1:
            return 1.0
        recent = float(self.klines[-1][5])
        avg    = float(np.mean([float(k[5]) for k in self.klines[-lookback - 1:-1]]))
        return recent / avg if avg > 0 else 1.0

    def is_ready(self) -> bool:
        return self.latest_price > 0 and len(self.klines) > 0

    # ─── Internal ────────────────────────────────────────────────────────────

    async def _run_ws(self) -> None:
        sub = json.dumps({
            "event": "subscribe",
            "pair":  [_PAIR],
            "subscription": {"name": "trade"},
        })
        async with websockets.connect(self._WS_URL, ping_interval=20) as ws:
            await ws.send(sub)
            logger.info("KrakenFeed: WebSocket connected, subscribed to %s trades", _PAIR)
            async for raw in ws:
                if not self._running:
                    break
                self._handle(raw)

    def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Kraken trade message: [channelID, [[price, vol, time, side, ...], ...], "trade", "XBT/USD"]
        if not isinstance(msg, list) or len(msg) < 4:
            return
        if msg[2] != "trade":
            return

        for tick in msg[1]:
            try:
                price = float(tick[0])
                self.latest_price = price
                self.price_history.append((time.monotonic(), price))
            except (IndexError, ValueError):
                pass

    async def _fetch_klines(self) -> None:
        params = {"pair": _SYMBOL, "interval": 5}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    self._REST_URL,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=8),
                ) as resp:
                    data = await resp.json()

            result = data.get("result", {})
            # Kraken returns the pair under various key names; pick the first non-"last" key
            pair_key = next((k for k in result if k != "last"), None)
            if not pair_key:
                return

            raw_candles = result[pair_key]
            # Kraken OHLC: [time, open, high, low, close, vwap, volume, count]
            # Normalise to:  [time, open, high, low, close, volume]
            self.klines = [
                [c[0], c[1], c[2], c[3], c[4], c[6]]
                for c in raw_candles[-self._kline_limit:]
            ]
        except Exception as exc:
            logger.warning("KrakenFeed: kline fetch failed: %s", exc)

    def _closes(self) -> np.ndarray:
        return np.array([float(k[4]) for k in self.klines])


# ─── Pure functions (shared by both signal types) ────────────────────────────

def _ema(values: np.ndarray, period: int) -> float:
    if len(values) < period:
        return float(values.mean())
    k   = 2.0 / (period + 1)
    ema = float(values[0])
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains[-period:].mean()
    avg_loss = losses[-period:].mean()
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1 + avg_gain / avg_loss))
