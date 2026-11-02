"""
Triangular-arbitrage scanner — DRY RUN ONLY.

Subscribes to a public order-book feed for three pairs (BTC/USD, ETH/USD and
ETH priced in BTC), keeps a local book per pair, and prices both triangle
paths on every book update (throttled). Nothing is traded: each opportunity
with a positive gross edge is appended to a JSONL log so you can see how
often a gap appears and whether it survives fees.

Venues
------
  coinbase  wss://ws-feed.exchange.coinbase.com  (level2_batch)
  kraken    wss://ws.kraken.com                  (book, depth 10)

Run from the polymarket_arb directory:

    python -m coinbase_arb.scanner
    python -m coinbase_arb.scanner --venue kraken --log kraken_arb_opportunities.jsonl

Market data is public on both venues; no API keys are needed.
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
from .paper_trader import PaperTrader
from .triangle import BTC_USD, ETH_BTC, ETH_USD, PRODUCTS, Opportunity, breakeven_gross_edge, evaluate

logger = logging.getLogger(__name__)

# Taker fees per leg. These are placeholders: fee tiers change and the sources
# disagree, so pass your real rate with --taker-fee.
DEFAULT_TAKER_FEES = {"coinbase": 0.006, "kraken": 0.004}


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


class TriangleScanner:
    """
    Shared book, pricing, logging and reconnect logic.

    Subclasses supply the feed: ``_run_ws`` opens and streams the connection,
    and ``_handle`` turns one raw message into ``_apply`` calls.
    """

    venue = "generic"
    book_depth: Optional[int] = None   # set by venues whose feed uses a fixed depth window

    def __init__(
        self,
        taker_fee: float,
        log_path: str,
        max_staleness_s: float = 2.0,
        eval_interval_s: float = 0.1,
        reconnect_delay_s: float = 1.0,
        paper: Optional[PaperTrader] = None,
    ):
        self.taker_fee = taker_fee
        self.log_path = log_path
        self.max_staleness_s = max_staleness_s
        self.eval_interval_s = eval_interval_s
        self.reconnect_delay_s = reconnect_delay_s
        self.paper = paper

        self.books: Dict[str, OrderBook] = {
            p: OrderBook(p, max_depth=self.book_depth) for p in PRODUCTS
        }
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
                logger.warning("%s WS error: %s — reconnecting in %.0fs", self.venue, exc, delay)
                for b in self.books.values():
                    b.reset()                       # never trade on pre-disconnect state
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    def stop(self) -> None:
        self._running = False

    async def report_stats(self, every_s: float = 30.0) -> None:
        if self.paper:
            print(PaperTrader.HEADER)
        while self._running:
            await asyncio.sleep(every_s)
            logger.info("%s scanner: %s", self.venue, self.stats.summary())
            if self.paper:
                print(self.paper.row(time.strftime("%H:%M:%S")))

    # ─── Subclass hooks ──────────────────────────────────────────────────────

    async def _run_ws(self) -> None:
        raise NotImplementedError

    def _handle(self, raw: str) -> None:
        raise NotImplementedError

    # ─── Shared internals ────────────────────────────────────────────────────

    def _apply(self, internal: str, kind: str, payload: dict, now: float) -> None:
        """kind is 'snapshot' (payload has bids/asks) or 'update' (payload has changes)."""
        book = self.books[internal]
        if kind == "snapshot":
            book.apply_snapshot(payload, now)
        else:
            book.apply_update(payload, now)
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
        if self.paper:
            self.paper.on_eval(now, tops)
        for o in opps:
            if o.gross_edge > 0:
                self._log(o, tops)

    def _log(self, opp: Opportunity, tops: Dict[str, Optional[Top]]) -> None:
        row = {
            "ts": time.time(),
            "venue": self.venue,
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
                "NET-POSITIVE [%s] %s gross=%.3f%% net=%.3f%% capacity=$%.0f (dry run, not traded)",
                self.venue, opp.path, opp.gross_edge * 100, opp.net_edge * 100, opp.capacity_usd,
            )


class CoinbaseScanner(TriangleScanner):
    """Coinbase Exchange public level2_batch feed. Pair names match the internal keys."""

    venue = "coinbase"
    WS_URL = "wss://ws-feed.exchange.coinbase.com"

    async def _run_ws(self) -> None:
        sub = json.dumps({
            "type": "subscribe",
            "product_ids": list(PRODUCTS),
            "channels": ["level2_batch"],
        })
        async with websockets.connect(self.WS_URL, ping_interval=20, max_size=2**22) as ws:
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

        pid = msg.get("product_id", "")
        if pid not in self.books:
            return

        now = time.monotonic()
        if mtype == "snapshot":
            self._apply(pid, "snapshot", msg, now)
        elif mtype == "l2update":
            self._apply(pid, "update", msg, now)


class KrakenScanner(TriangleScanner):
    """
    Kraken public book feed (v1). Pairs arrive as XBT/USD, ETH/USD and ETH/XBT,
    which is ETH priced in BTC, matching ETH-BTC on Coinbase.

    Book messages are arrays: [channelID, data, ..., "book-10", "PAIR"].
    A snapshot dict has "as" (asks) and "bs" (bids). An update dict has "a"
    and/or "b", where volume "0" removes a level.
    """

    venue = "kraken"
    WS_URL = "wss://ws.kraken.com"
    PAIR_TO_INTERNAL = {"XBT/USD": BTC_USD, "ETH/USD": ETH_USD, "ETH/XBT": ETH_BTC}
    BOOK_DEPTH = 10
    book_depth = BOOK_DEPTH            # drop levels that fall out of the subscribed window

    async def _run_ws(self) -> None:
        sub = json.dumps({
            "event": "subscribe",
            "pair": list(self.PAIR_TO_INTERNAL),
            "subscription": {"name": "book", "depth": self.BOOK_DEPTH},
        })
        async with websockets.connect(self.WS_URL, ping_interval=20, max_size=2**22) as ws:
            await ws.send(sub)
            logger.info("Kraken WebSocket connected, subscribed to %s", ", ".join(self.PAIR_TO_INTERNAL))
            async for raw in ws:
                if not self._running:
                    break
                self._handle(raw)

    def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        if isinstance(msg, dict):          # systemStatus, subscriptionStatus, heartbeat
            if msg.get("event") == "subscriptionStatus" and msg.get("status") == "error":
                logger.error("Kraken subscription error: %s", msg.get("errorMessage", msg))
            return

        if not isinstance(msg, list) or len(msg) < 4:
            return
        channel, pair = str(msg[-2]), msg[-1]
        internal = self.PAIR_TO_INTERNAL.get(pair)
        if internal is None or not channel.startswith("book"):
            return

        now = time.monotonic()
        parts = [p for p in msg[1:-2] if isinstance(p, dict)]
        if any("as" in p or "bs" in p for p in parts):
            bids = [[e[0], e[1]] for p in parts for e in p.get("bs", [])]
            asks = [[e[0], e[1]] for p in parts for e in p.get("as", [])]
            self._apply(internal, "snapshot", {"bids": bids, "asks": asks}, now)
        else:
            changes = []
            for p in parts:
                changes += [["sell", e[0], e[1]] for e in p.get("a", [])]
                changes += [["buy", e[0], e[1]] for e in p.get("b", [])]
            if changes:
                self._apply(internal, "update", {"changes": changes}, now)


async def _main(args: argparse.Namespace) -> None:
    cls = KrakenScanner if args.venue == "kraken" else CoinbaseScanner
    taker_fee = args.taker_fee if args.taker_fee is not None else DEFAULT_TAKER_FEES[args.venue]
    log_path = args.log or f"{args.venue}_arb_opportunities.jsonl"

    paper = None
    if args.paper:
        paper = PaperTrader(
            starting_equity=args.paper_equity,
            taker_fee=taker_fee,
            latency_s=args.paper_latency,
            cooldown_s=args.paper_cooldown,
            log_path=f"{args.venue}_paper_trades.jsonl",
        )

    scanner = cls(
        taker_fee=taker_fee,
        log_path=log_path,
        max_staleness_s=args.max_staleness,
        eval_interval_s=args.eval_interval,
        paper=paper,
    )
    logger.info(
        "%s triangle scanner (DRY RUN): taker_fee=%.2f%%, breakeven gross edge=%.3f%%, log=%s",
        args.venue, taker_fee * 100, breakeven_gross_edge(taker_fee) * 100, log_path,
    )
    tasks = [asyncio.create_task(scanner.run()), asyncio.create_task(scanner.report_stats())]
    try:
        await asyncio.gather(*tasks)
    finally:
        scanner.stop()
        for t in tasks:
            t.cancel()
        logger.info("Final: %s", scanner.stats.summary())
        if scanner.paper:
            print(scanner.paper.row(time.strftime("%H:%M:%S")))
            logger.info("Final paper: %s", scanner.paper.summary())


def main() -> None:
    parser = argparse.ArgumentParser(description="Triangular-arb scanner (dry run)")
    parser.add_argument("--venue", choices=["coinbase", "kraken"], default="coinbase")
    parser.add_argument("--taker-fee", type=float, default=None,
                        help="Taker fee per leg as a fraction (0.006 = 0.60%%). "
                             "Default: coinbase 0.60%%, kraken 0.40%%. Check your tier.")
    parser.add_argument("--log", default=None,
                        help="JSONL file for opportunities (default: <venue>_arb_opportunities.jsonl)")
    parser.add_argument("--max-staleness", type=float, default=2.0,
                        help="Skip evaluation if any book is older than this many seconds")
    parser.add_argument("--eval-interval", type=float, default=0.1,
                        help="Minimum seconds between evaluations")
    parser.add_argument("--paper", action="store_true",
                        help="Simulate fills with paper equity (no orders are placed)")
    parser.add_argument("--paper-equity", type=float, default=11.0, help="Starting paper equity in USD")
    parser.add_argument("--paper-latency", type=float, default=0.3,
                        help="Seconds a gap must persist before a simulated fill")
    parser.add_argument("--paper-cooldown", type=float, default=10.0,
                        help="Seconds before the same path can fill again")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
