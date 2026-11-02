"""
Local level-2 order book for one Coinbase Exchange product.

Fed by the public ``level2_batch`` channel: a ``snapshot`` message seeds the
book, then ``l2update`` messages apply size changes (size 0 removes a level).
"""

from __future__ import annotations

from typing import Optional, Tuple

# (best_bid, size_at_bid, best_ask, size_at_ask)
Top = Tuple[float, float, float, float]


class OrderBook:
    def __init__(self, product_id: str):
        self.product_id = product_id
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.ready = False
        self.last_update = 0.0   # monotonic seconds of last applied message

    def apply_snapshot(self, msg: dict, now: float) -> None:
        self.bids = {float(p): float(s) for p, s in msg["bids"] if float(s) > 0}
        self.asks = {float(p): float(s) for p, s in msg["asks"] if float(s) > 0}
        self.ready = True
        self.last_update = now

    def apply_update(self, msg: dict, now: float) -> None:
        if not self.ready:
            return                       # wait for the snapshot first
        for side, price, size in msg["changes"]:
            book = self.bids if side == "buy" else self.asks
            p, s = float(price), float(size)
            if s == 0:
                book.pop(p, None)
            else:
                book[p] = s
        self.last_update = now

    def reset(self) -> None:
        """Drop state after a reconnect; the next snapshot reseeds it."""
        self.bids.clear()
        self.asks.clear()
        self.ready = False

    def top(self) -> Optional[Top]:
        if not self.ready or not self.bids or not self.asks:
            return None
        bid = max(self.bids)
        ask = min(self.asks)
        return bid, self.bids[bid], ask, self.asks[ask]
