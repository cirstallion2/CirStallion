"""
Coinbase triangular-arbitrage scanner — DRY RUN ONLY.

Subscribes to the public Coinbase Exchange ``level2_batch`` channel for
BTC-USD, ETH-USD and ETH-BTC, keeps a local book per product, and prices
both triangle paths on every book update (throttled). Nothing is traded:
each opportunity with a positive gross edge is appended to a JSONL log so
you can see how often a gap appears and whether it survives fees.

Run from the polymarket_arb directory:

    python -m coinbase_arb.scanner
    python -m coinbase_arb.scanner --taker-fee 0.004 --log opportunities.jsonl

Market data is public; no API keys are needed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import websockets

from .book import OrderBook, Top
from .triangle import DEFAULT_TAKER_FEE, PRODUCTS, Opportunity, breakeven_gross_edge, evaluate

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-feed.exchange.coinbase.com"
DEFAULT_LOG = "coinbase_arb_opportunities.jsonl"


@dataclass
class Stats:
    evaluations: int = 0
    stale_skips: int = 0
    gross_positive: int = 0
    net_positive: int = 0
    max_gross: float = 0.0
    max_net: float = 0.0
    best_net_capacity_usd: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def record(self, opps: list[Opportunity]) -> None:
        self.evaluations += 1
        for o in opps:
            if o.gross_edge > 0:
                self.gross_positive += 1
                self.max_gross = max(self.max_gross, o.gross_edge)
            if o.net_edge > 0:
                self.net_positive += 1
                self.max_net = max(self.max_net, o.net_edge)
                self.best_net_capacity_usd = max(self.best_net_capacity_usd, o.capacity_usd)

    def summary(self) -> str:
        elapsed = max(time.monotonic() - self.started, 1e-9)
        return (
            f"uptime={elapsed:,.0f}s evals={self.evaluations} ({self.evaluations / elapsed:.1f}/s) "
            f"stale_skips={self.stale_skips} gross>0={self.gross_positive} net>0={self.net_positive} "
            f"max_gross={self.max_gross * 100:.3f}% max_net={self.max_net * 100:.3f}% "
            f"best_net_capacity=${self.best_net_capacity_usd:,.0f}"
        )


class CoinbaseScanner:
    def __init__(
        self,
        taker_fee: float = DEFAULT_TAKER_FEE,
        log_path: str = DEFAULT_LOG,
        max_staleness_s: float = 1.0,
        eval_interval_s: float = 0.1,
        reconnect_delay_s: float = 1.0,
    ):
        self.taker_fee = taker_fee
        self.log_path = log_path
        self.max_staleness_s = max_staleness_s
        self.eval_interval_s = eval_interval_s
        self.reconnect_delay_s = reconnect_delay_s

        self.books: Dict[str, OrderBook] = {p: OrderBook(p) for p in PRODUCTS}
        self.stats = Stats()
        self._last_eval = 0.0
        self._running = False

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._running = True
        delay = self.reconnect_delay_s
        while self._running:
            try:
                await self._run_ws()
                delay = self.reconnect_delay_s     # reset only after a clean session
            except Exception as exc:
                logger.warning("Coinbase WS error: %s — reconnecting in %.0fs", exc, delay)
                for b in self.books.values():
                    b.reset()                       # never trade on pre-disconnect state
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    def stop(self) -> None:
        self._running = False

    async def report_stats(self, every_s: float = 30.0) -> None:
        while self._running:
            await asyncio.sleep(every_s)
            logger.info("Coinbase scanner: %s", self.stats.summary())

    # ─── Internals ───────────────────────────────────────────────────────────

    async def _run_ws(self) -> None:
        sub = json.dumps({
            "type": "subscribe",
            "product_ids": list(PRODUCTS),
            "channels": ["level2_batch"],
        })
        async with websockets.connect(WS_URL, ping_interval=20, max_size=2**22) as ws:
            await ws.send(sub)
            logger.info("Coinbase WebSocket connected, subscribed to %s", ", ".join(PRODUCTS))
            async for raw in ws:
                if not self._running:
                    break
                self._handle(raw)

    def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        mtype = msg.get("type")
        if mtype == "error":
            logger.error("Coinbase feed error: %s", msg.get("message", msg))
            return

        book = self.books.get(msg.get("product_id", ""))
        if book is None:
            return

        now = time.monotonic()
        if mtype == "snapshot":
            book.apply_snapshot(msg, now)
        elif mtype == "l2update":
            book.apply_update(msg, now)
            self._maybe_evaluate(now)

    def _maybe_evaluate(self, now: float) -> None:
        if now - self._last_eval < self.eval_interval_s:
            return
        self._last_eval = now

        tops: Dict[str, Optional[Top]] = {}
        for pid, book in self.books.items():
            if not book.ready or now - book.last_update > self.max_staleness_s:
                self.stats.stale_skips += 1
                return
            tops[pid] = book.top()

        opps = evaluate(tops, self.taker_fee)
        self.stats.record(opps)
        for o in opps:
            if o.gross_edge > 0:
                self._log(o, tops)

    def _log(self, opp: Opportunity, tops: Dict[str, Optional[Top]]) -> None:
        row = {
            "ts": time.time(),
            "path": opp.path,
            "gross_pct": round(opp.gross_edge * 100, 4),
            "net_pct": round(opp.net_edge * 100, 4),
            "capacity_usd": round(opp.capacity_usd, 2),
            "taker_fee": self.taker_fee,
            "tops": {p: tops[p] for p in PRODUCTS},
        }
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if opp.net_edge > 0:
            logger.info(
                "NET-POSITIVE %s gross=%.3f%% net=%.3f%% capacity=$%.0f (dry run, not traded)",
                opp.path, opp.gross_edge * 100, opp.net_edge * 100, opp.capacity_usd,
            )


async def _main(args: argparse.Namespace) -> None:
    scanner = CoinbaseScanner(
        taker_fee=args.taker_fee,
        log_path=args.log,
        max_staleness_s=args.max_staleness,
        eval_interval_s=args.eval_interval,
    )
    logger.info(
        "Coinbase triangle scanner (DRY RUN): taker_fee=%.2f%%, breakeven gross edge=%.3f%%, log=%s",
        args.taker_fee * 100, breakeven_gross_edge(args.taker_fee) * 100, args.log,
    )
    tasks = [asyncio.create_task(scanner.run()), asyncio.create_task(scanner.report_stats())]
    try:
        await asyncio.gather(*tasks)
    finally:
        scanner.stop()
        for t in tasks:
            t.cancel()
        logger.info("Final: %s", scanner.stats.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description="Coinbase triangular-arb scanner (dry run)")
    parser.add_argument("--taker-fee", type=float, default=DEFAULT_TAKER_FEE,
                        help="Taker fee as a fraction per leg (0.006 = 0.60%%). Check your tier.")
    parser.add_argument("--log", default=DEFAULT_LOG, help="JSONL file for opportunities")
    parser.add_argument("--max-staleness", type=float, default=1.0,
                        help="Skip evaluation if any book is older than this many seconds")
    parser.add_argument("--eval-interval", type=float, default=0.1,
                        help="Minimum seconds between evaluations")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
