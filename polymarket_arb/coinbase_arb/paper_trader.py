"""
Paper trader for triangle gaps. Simulated equity only; places no orders.

Model
-----
1. A net-positive path is seen at time t. Nothing is filled yet.
2. After ``latency_s`` seconds, the same path is re-priced on the books at
   that moment. It fills only if it is still net-positive then. Otherwise it
   counts as missed.
3. A fill is sized at min(equity, depth at fill time). P&L is size times the
   net edge at fill time.
4. Trading halts if equity falls to ``halt_fraction`` of the starting value.

Limits: the model ignores partial fills and leg risk, so it has no losing
trades by construction. Treat its results as an upper bound.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .book import Top
from .triangle import evaluate


@dataclass
class PaperTrader:
    starting_equity: float = 11.0
    taker_fee: float = 0.0
    latency_s: float = 0.3
    halt_fraction: float = 0.5
    cooldown_s: float = 10.0       # no new fill on a path for this long after one fills
    log_path: str = "paper_trades.jsonl"

    equity: float = field(init=False)
    fills: int = field(init=False, default=0)
    missed: int = field(init=False, default=0)
    halted: bool = field(init=False, default=False)
    _pending: Dict[str, float] = field(init=False, default_factory=dict)   # path -> due time
    _cooldown_until: Dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.equity = self.starting_equity

    def on_eval(self, now: float, tops: Dict[str, Optional[Top]]) -> None:
        if self.halted:
            return

        # Resolve gaps whose latency has elapsed.
        for path, due in list(self._pending.items()):
            if now < due:
                continue
            del self._pending[path]
            opp = next((o for o in evaluate(tops, self.taker_fee) if o.path == path), None)
            if opp and opp.net_edge > 0 and opp.capacity_usd > 0:
                size = min(self.equity, opp.capacity_usd)
                pnl = size * opp.net_edge
                self.equity += pnl
                self.fills += 1
                self._cooldown_until[path] = now + self.cooldown_s
                self._log({"ts": time.time(), "event": "fill", "path": path,
                           "size_usd": round(size, 4), "net_pct": round(opp.net_edge * 100, 5),
                           "pnl_usd": round(pnl, 6), "equity": round(self.equity, 6)})
            else:
                self.missed += 1

        # Register new gaps.
        for o in evaluate(tops, self.taker_fee):
            if now < self._cooldown_until.get(o.path, 0.0):
                continue                          # same gap, already filled recently
            if o.net_edge > 0 and o.path not in self._pending:
                self._pending[o.path] = now + self.latency_s

        if self.equity <= self.starting_equity * self.halt_fraction:
            self.halted = True
            self._log({"ts": time.time(), "event": "halt", "equity": round(self.equity, 6)})

    HEADER = (f"{'TIME':<9} {'BALANCE $':>12} {'P&L $':>12} {'P&L %':>10} "
              f"{'FILLS':>7} {'MISSED':>8}  STATUS")

    def row(self, clock: str) -> str:
        """One fixed-width table row matching HEADER."""
        pnl = self.equity - self.starting_equity
        pct = pnl / self.starting_equity * 100
        status = "HALTED" if self.halted else "RUNNING"
        return (f"{clock:<9} {self.equity:>12,.4f} {pnl:>+12,.4f} {pct:>+9.4f}% "
                f"{self.fills:>7} {self.missed:>8}  {status}")

    def summary(self) -> str:
        ret = (self.equity / self.starting_equity - 1.0) * 100
        return (f"paper equity=${self.equity:,.4f} (start ${self.starting_equity:,.2f}, {ret:+.4f}%) "
                f"fills={self.fills} missed={self.missed} halted={self.halted}")

    def _log(self, row: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
