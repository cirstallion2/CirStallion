"""
Paper trader for triangle gaps. Simulated equity only; places no orders.

Model
-----
1. A net-positive path is seen at time t. Nothing is filled yet.
2. After ``latency_s`` seconds, the same path is re-priced on the books at
   that moment. It fills only if it is still net-positive then. Otherwise it
   counts as missed.
3. Slippage: each of the three legs fills ``slippage_pct`` worse than the
   displayed price, so the edge is reduced by 3 x slippage_pct.
4. Depth is used up: notional traded on a path in the last ``depth_recovery_s``
   seconds is subtracted from the capacity the book shows.
5. Failed legs: with probability ``fail_prob`` a fill breaks partway and is
   unwound at a loss of ``unwind_loss_pct`` of its size instead of earning.
6. Trading halts if equity falls to ``halt_fraction`` of the starting value.

Setting slippage_pct, fail_prob and unwind_loss_pct to zero gives the old,
optimistic model. Even with them set, the model is still an approximation, so
treat its results as an estimate, not proof.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .book import Top
from .triangle import evaluate


@dataclass
class PaperTrader:
    starting_equity: float = 11.0
    taker_fee: float = 0.0
    latency_s: float = 0.3
    halt_fraction: float = 0.5
    cooldown_s: float = 10.0        # no new fill on a path for this long after one fills
    slippage_pct: float = 0.0002    # per leg, as a fraction (0.0002 = 0.02%)
    fail_prob: float = 0.05         # chance a fill breaks a leg
    unwind_loss_pct: float = 0.003  # loss as a fraction of size when a fill fails
    depth_recovery_s: float = 30.0  # seconds before consumed depth is available again
    seed: int = 1
    log_path: str = "paper_trades.jsonl"

    equity: float = field(init=False)
    fills: int = field(init=False, default=0)
    failed: int = field(init=False, default=0)
    missed: int = field(init=False, default=0)
    halted: bool = field(init=False, default=False)
    _pending: Dict[str, float] = field(init=False, default_factory=dict)   # path -> due time
    _cooldown_until: Dict[str, float] = field(init=False, default_factory=dict)
    _consumed: Dict[str, List[Tuple[float, float]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.equity = self.starting_equity
        self._rng = random.Random(self.seed)

    def _available_depth(self, path: str, capacity: float, now: float) -> float:
        recent = [(t, s) for t, s in self._consumed.get(path, []) if now - t < self.depth_recovery_s]
        self._consumed[path] = recent
        return capacity - sum(s for _, s in recent)

    def on_eval(self, now: float, tops: Dict[str, Optional[Top]]) -> None:
        if self.halted:
            return

        # Resolve gaps whose latency has elapsed.
        for path, due in list(self._pending.items()):
            if now < due:
                continue
            del self._pending[path]
            opp = next((o for o in evaluate(tops, self.taker_fee) if o.path == path), None)
            edge = (opp.net_edge - 3 * self.slippage_pct) if opp else 0.0
            avail = self._available_depth(path, opp.capacity_usd, now) if opp else 0.0
            if not opp or edge <= 0 or avail <= 0:
                self.missed += 1
                continue

            size = min(self.equity, avail)
            self._consumed.setdefault(path, []).append((now, size))
            self._cooldown_until[path] = now + self.cooldown_s

            if self._rng.random() < self.fail_prob:
                pnl = -size * self.unwind_loss_pct
                self.failed += 1
                event = "fail"
            else:
                pnl = size * edge
                self.fills += 1
                event = "fill"
            self.equity += pnl
            self._log({"ts": time.time(), "event": event, "path": path,
                       "size_usd": round(size, 4), "net_pct": round(edge * 100, 5),
                       "pnl_usd": round(pnl, 6), "equity": round(self.equity, 6)})

        # Register new gaps (slippage is applied when the gap is resolved).
        for o in evaluate(tops, self.taker_fee):
            if now < self._cooldown_until.get(o.path, 0.0):
                continue                          # same gap, already traded recently
            if o.net_edge > 0 and o.path not in self._pending:
                self._pending[o.path] = now + self.latency_s

        if self.equity <= self.starting_equity * self.halt_fraction:
            self.halted = True
            self._log({"ts": time.time(), "event": "halt", "equity": round(self.equity, 6)})

    HEADER = (f"{'TIME':<9} {'BALANCE $':>12} {'P&L $':>12} {'P&L %':>10} "
              f"{'FILLS':>7} {'FAILED':>7} {'MISSED':>8}  STATUS")

    def row(self, clock: str) -> str:
        """One fixed-width table row matching HEADER."""
        pnl = self.equity - self.starting_equity
        pct = pnl / self.starting_equity * 100
        status = "HALTED" if self.halted else "RUNNING"
        return (f"{clock:<9} {self.equity:>12,.4f} {pnl:>+12,.4f} {pct:>+9.4f}% "
                f"{self.fills:>7} {self.failed:>7} {self.missed:>8}  {status}")

    def summary(self) -> str:
        ret = (self.equity / self.starting_equity - 1.0) * 100
        return (f"paper equity=${self.equity:,.4f} (start ${self.starting_equity:,.2f}, {ret:+.4f}%) "
                f"fills={self.fills} failed={self.failed} missed={self.missed} halted={self.halted}")

    def _log(self, row: dict) -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
