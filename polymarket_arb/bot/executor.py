"""
Order executor — submits FOK orders to the Polymarket CLOB with
sub-100 ms latency and tracks fill outcomes for the risk manager.

Execution path
──────────────
detect() → risk_manager.check() → executor.execute()
             ↳ compute size
             ↳ choose limit price (best-ask - 0.5 tick for taker fills)
             ↳ submit via PolymarketClient.place_order()
             ↳ record result in risk_manager
             ↳ log outcome

Throughput note
───────────────
The executor is async and non-blocking.  Multiple concurrent
signals (different markets) can be in-flight simultaneously.
A semaphore caps concurrent live submissions to avoid rate-limit errors
while still allowing 1,000+ queued intents per second in dry-run mode.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .polymarket_client import OrderResult, PolymarketClient
from .risk_manager import RiskManager
from .signal_detector import Signal

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    direction: str
    token_id: str
    size_usdc: float
    entry_price: float
    filled_size: float
    filled_price: float
    pnl_est_usdc: float     # estimated immediate edge captured
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    order_id: str = ""
    success: bool = False


class Executor:
    """
    Converts confirmed signals into CLOB orders.

    Parameters
    ----------
    poly     : PolymarketClient instance (already started)
    risk     : RiskManager instance
    config   : BotConfig
    """

    _TICK_SIZE = 0.001        # Polymarket min price increment
    _MAX_CONCURRENT = 4       # live submissions in-flight at once

    def __init__(self, poly: PolymarketClient, risk: RiskManager, config):
        self._poly = poly
        self._risk = risk
        self._cfg = config
        self._semaphore = asyncio.Semaphore(self._MAX_CONCURRENT)
        self.trade_log: List[TradeRecord] = []
        self._order_count = 0
        self._session_start = time.monotonic()

    # ─── Public API ──────────────────────────────────────────────────────────

    async def execute(self, signal: Signal) -> Optional[TradeRecord]:
        """
        Attempt to execute the signal.  Returns TradeRecord on success,
        None if the risk manager blocks or the order fails.

        Target: from entry to order-submitted in <100 ms.
        """
        t0 = time.monotonic()

        # ── Risk gate ─────────────────────────────────────────────────────
        allowed, reason = self._risk.check(signal)
        if not allowed:
            logger.debug("Risk blocked: %s", reason)
            return None

        size_usdc = self._risk.compute_size(signal)
        if size_usdc < self._cfg.risk.min_position_usdc:
            logger.debug("Size %.2f USDC below minimum — skip", size_usdc)
            return None

        # ── Limit price: best ask + 0.5 tick (aggressive taker) ──────────
        best_ask = self._poly.get_best_ask(signal.token_id)
        if best_ask <= 0:
            logger.debug("No ask quote for %s — skip", signal.token_id[:12])
            return None

        limit_price = round(best_ask + self._TICK_SIZE * 0.5, 4)
        limit_price = min(limit_price, 0.99)    # never buy a certainty

        # ── Submit order with concurrency cap ─────────────────────────────
        async with self._semaphore:
            result: OrderResult = await self._poly.place_order(
                token_id=signal.token_id,
                side="BUY",
                price=limit_price,
                size_usdc=size_usdc,
            )

        elapsed_ms = (time.monotonic() - t0) * 1000

        if elapsed_ms > self._cfg.target_execution_ms:
            logger.warning("Execution latency %.0f ms exceeded %d ms target", elapsed_ms, self._cfg.target_execution_ms)

        if not result.success:
            logger.warning("Order failed: %s (%.0f ms)", result.error, elapsed_ms)
            return None

        # Estimated edge captured = (theo_price - fill_price) × shares
        shares = result.filled_size
        pnl_est = (signal.theo_price - result.filled_price) * shares

        record = TradeRecord(
            direction=signal.direction,
            token_id=signal.token_id,
            size_usdc=size_usdc,
            entry_price=limit_price,
            filled_size=result.filled_size,
            filled_price=result.filled_price,
            pnl_est_usdc=pnl_est,
            latency_ms=elapsed_ms,
            order_id=result.order_id,
            success=True,
        )

        self._risk.record_fill(record)
        self.trade_log.append(record)
        self._order_count += 1

        logger.info(
            "FILLED | dir=%-4s price=%.4f size=%.2fU pnl_est=+%.3fU lat=%.0fms [#%d]",
            record.direction, record.filled_price, record.size_usdc,
            record.pnl_est_usdc, record.latency_ms, self._order_count,
        )
        return record

    def stats(self) -> dict:
        elapsed_s = max(time.monotonic() - self._session_start, 1.0)
        total_pnl = sum(r.pnl_est_usdc for r in self.trade_log)
        fills = [r for r in self.trade_log if r.success]
        avg_lat = sum(r.latency_ms for r in fills) / len(fills) if fills else 0.0
        return {
            "orders": self._order_count,
            "orders_per_sec": round(self._order_count / elapsed_s, 1),
            "fills": len(fills),
            "total_pnl_est_usdc": round(total_pnl, 4),
            "avg_latency_ms": round(avg_lat, 1),
            "uptime_s": round(elapsed_s, 0),
        }
