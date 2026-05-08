"""
Risk manager — enforces all pre-trade and post-trade risk rules.

Rules (hard-coded defaults; overridden via RiskConfig)
──────────────────────────────────────────────────────
  Per-trade risk  : ≤ 0.5 % of portfolio size (USDC)
  Daily cap       : stop trading once cumulative P&L reaches +2 %
  Hard stop       : emergency halt if daily P&L ≤ -0.4 %
  Thin liquidity  : enforced upstream in SignalDetector

The risk manager is intentionally stateful and single-threaded — only
the main async loop calls it, so no locking is needed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from .executor import TradeRecord
    from .signal_detector import Signal

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    date: str = ""
    trades: int = 0
    pnl_usdc: float = 0.0
    gross_exposure_usdc: float = 0.0
    hard_stop_triggered: bool = False
    daily_cap_reached: bool = False


class RiskManager:
    """
    Stateful risk gate.

    Usage
    ─────
    allowed, reason = risk.check(signal)
    size = risk.compute_size(signal)
    # … execute …
    risk.record_fill(trade_record)
    """

    def __init__(self, config):
        self._cfg = config
        self._portfolio = config.portfolio_size_usdc
        self._per_trade_pct = config.per_trade_risk_pct
        self._daily_cap_pct = config.daily_cap_pct
        self._hard_stop_pct = config.hard_stop_pct
        self._max_pos = config.max_position_usdc

        self.daily = DailyStats(date=self._today())
        self._last_reset_check = time.monotonic()

    # ─── Public API ──────────────────────────────────────────────────────────

    def check(self, signal: "Signal") -> Tuple[bool, str]:
        """
        Returns (True, "") if the trade is allowed, else (False, reason).
        Call this *before* computing size or submitting an order.
        """
        self._maybe_reset_daily()

        if self.daily.hard_stop_triggered:
            return False, "hard-stop active — trading halted for the day"

        if self.daily.daily_cap_reached:
            return False, "daily profit cap reached"

        drawdown_pct = self.daily.pnl_usdc / self._portfolio
        if drawdown_pct <= -self._hard_stop_pct:
            self.daily.hard_stop_triggered = True
            logger.critical(
                "HARD STOP triggered: daily P&L = %.2f USDC (%.2f %%)",
                self.daily.pnl_usdc, drawdown_pct * 100,
            )
            return False, "hard-stop triggered now"

        if drawdown_pct >= self._daily_cap_pct:
            self.daily.daily_cap_reached = True
            logger.info(
                "Daily cap reached: P&L = +%.2f USDC (+%.2f %%) — pausing",
                self.daily.pnl_usdc, drawdown_pct * 100,
            )
            return False, "daily cap reached now"

        return True, ""

    def compute_size(self, signal: "Signal") -> float:
        """
        Returns the position size in USDC for this trade.

        Scales down as we approach the daily cap to avoid overshooting.
        """
        base_size = self._per_trade_pct * self._portfolio

        # Remaining room before daily cap
        cap_usdc = self._daily_cap_pct * self._portfolio
        remaining = cap_usdc - self.daily.pnl_usdc
        if remaining <= 0:
            return 0.0

        # Scale by how confident the signal is (graph confirmation)
        conf_factor = float(max(signal.confidence, 0.65))  # floor at threshold

        size = min(base_size * conf_factor, remaining, self._max_pos)
        return max(size, 0.0)

    def record_fill(self, record: "TradeRecord") -> None:
        """Update daily stats after a confirmed fill."""
        self.daily.trades += 1
        self.daily.pnl_usdc += record.pnl_est_usdc
        self.daily.gross_exposure_usdc += record.size_usdc

        logger.debug(
            "RiskManager: fill recorded | daily_pnl=%.3f USDC (%.2f %%) trades=%d",
            self.daily.pnl_usdc,
            (self.daily.pnl_usdc / self._portfolio) * 100,
            self.daily.trades,
        )

    def summary(self) -> dict:
        return {
            "date": self.daily.date,
            "trades": self.daily.trades,
            "daily_pnl_usdc": round(self.daily.pnl_usdc, 4),
            "daily_pnl_pct": round((self.daily.pnl_usdc / self._portfolio) * 100, 3),
            "gross_exposure_usdc": round(self.daily.gross_exposure_usdc, 2),
            "hard_stop": self.daily.hard_stop_triggered,
            "daily_cap": self.daily.daily_cap_reached,
            "portfolio_usdc": self._portfolio,
        }

    # ─── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _today() -> str:
        from datetime import date
        return date.today().isoformat()

    def _maybe_reset_daily(self) -> None:
        now_day = self._today()
        if self.daily.date != now_day:
            logger.info(
                "New trading day — resetting daily stats (prev P&L: %.2f USDC, trades: %d)",
                self.daily.pnl_usdc, self.daily.trades,
            )
            self.daily = DailyStats(date=now_day)
