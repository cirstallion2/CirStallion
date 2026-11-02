"""
Compare the local Kraken WebSocket book with fresh REST books. Read-only.

    python -m coinbase_arb.ws_compare --seconds 120 --interval 3

Every interval it reads the local book's top of book for each pair and, right
after, fetches the REST top of book. It reports how often the best bid and best
ask agree and the largest difference. Because the two reads are a few hundred
ms apart, small differences are normal; persistent large ones mean the local
book is wrong.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Optional

from .book import Top
from .rest_check import DEPTH_URL, REST_PAIRS, parse_depth
from .scanner import KrakenScanner

TOLERANCE = 0.0001   # 0.01%: differences below this count as a match


def compare_tops(ws: Top, rest: Top) -> tuple[float, float]:
    """Relative difference of best bid and best ask, WebSocket vs REST."""
    return (abs(ws[0] - rest[0]) / rest[0], abs(ws[2] - rest[2]) / rest[2])


def _fetch_one(pair: str) -> Top:
    query = urllib.parse.urlencode({"pair": pair, "count": 1})
    with urllib.request.urlopen(f"{DEPTH_URL}?{query}", timeout=5) as resp:
        return parse_depth(json.loads(resp.read()))


async def _fetch_all() -> dict[str, Top]:
    """Fetch the three pairs at the same time to keep the REST snapshot tight."""
    pairs = list(REST_PAIRS)
    results = await asyncio.gather(*(asyncio.to_thread(_fetch_one, p) for p in pairs))
    return {REST_PAIRS[p]: r for p, r in zip(pairs, results)}


async def _main(args: argparse.Namespace) -> None:
    # eval interval is huge so the scanner never evaluates or logs; we only read its books
    scanner = KrakenScanner(taker_fee=0.0, log_path=os.devnull, eval_interval_s=1e9)
    task = asyncio.create_task(scanner.run())

    samples = 0
    matches = {k: 0 for k in REST_PAIRS.values()}
    worst: dict[str, float] = {k: 0.0 for k in REST_PAIRS.values()}
    end = time.monotonic() + args.seconds
    await asyncio.sleep(5.0)                                # let snapshots arrive

    print(f"{'PAIR':<8} {'BID DIFF %':>11} {'ASK DIFF %':>11}   WS bid/ask            REST bid/ask")
    try:
        while time.monotonic() < end:
            started = time.monotonic()
            ws_tops: dict[str, Optional[Top]] = {p: b.top() for p, b in scanner.books.items()}
            try:
                rest_tops = await _fetch_all()
            except Exception as exc:
                print(f"REST error: {exc}")
            else:
                samples += 1
                for pid, rest in rest_tops.items():
                    ws = ws_tops.get(pid)
                    if ws is None:
                        print(f"{pid:<8} local book not ready")
                        continue
                    bd, ad = compare_tops(ws, rest)
                    worst[pid] = max(worst[pid], bd, ad)
                    if bd <= TOLERANCE and ad <= TOLERANCE:
                        matches[pid] += 1
                    print(f"{pid:<8} {bd * 100:>11.4f} {ad * 100:>11.4f}   "
                          f"{ws[0]:<10g}/{ws[2]:<10g}  {rest[0]:<10g}/{rest[2]:<10g}")
            await asyncio.sleep(max(0.0, args.interval - (time.monotonic() - started)))
    finally:
        scanner.stop()
        task.cancel()

    print(f"\nSummary over {samples} samples (match = both sides within {TOLERANCE * 100:.2f}%):")
    for pid in REST_PAIRS.values():
        rate = matches[pid] / samples * 100 if samples else 0.0
        print(f"  {pid:<8} match {rate:5.1f}%   worst difference {worst[pid] * 100:.4f}%")
    if samples and all(matches[p] / samples >= 0.8 for p in matches):
        print("Local book agrees with REST most of the time.")
    else:
        print("Local book disagrees with REST often: gaps seen on it cannot be trusted.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Kraken WebSocket vs REST book comparison (read-only)")
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()
    logging.basicConfig(level="WARNING")
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
