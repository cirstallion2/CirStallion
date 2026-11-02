"""
Read-only check of Kraken triangle gaps against fresh REST order books.

The WebSocket scanner can show gaps that are stale in the local book. This
script polls Kraken's public REST Depth endpoint for the top of book on
XBT/USD, ETH/USD and ETH/XBT, prices both triangle paths on each snapshot,
and logs what it finds. It never places orders.

    python -m coinbase_arb.rest_check --seconds 300 --interval 2 --taker-fee 0

Keep --interval at 2 seconds or more. Each poll makes three public requests,
and Kraken rate-limits public calls.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request

from .book import Top
from .triangle import BTC_USD, ETH_BTC, ETH_USD, DEFAULT_TAKER_FEE, evaluate

DEPTH_URL = "https://api.kraken.com/0/public/Depth"
REST_PAIRS = {"XBTUSD": BTC_USD, "ETHUSD": ETH_USD, "ETHXBT": ETH_BTC}


def parse_depth(payload: dict) -> Top:
    """Top of book from one Kraken Depth response (count=1)."""
    if payload.get("error"):
        raise ValueError(payload["error"])
    data = next(iter(payload["result"].values()))
    bid = data["bids"][0]
    ask = data["asks"][0]
    return float(bid[0]), float(bid[1]), float(ask[0]), float(ask[1])


def fetch_tops(timeout: float = 5.0) -> dict:
    tops = {}
    for pair, internal in REST_PAIRS.items():
        query = urllib.parse.urlencode({"pair": pair, "count": 1})
        with urllib.request.urlopen(f"{DEPTH_URL}?{query}", timeout=timeout) as resp:
            tops[internal] = parse_depth(json.loads(resp.read()))
    return tops


def main() -> None:
    parser = argparse.ArgumentParser(description="Kraken REST triangle check (read-only)")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--taker-fee", type=float, default=DEFAULT_TAKER_FEE)
    parser.add_argument("--log", default="kraken_rest_check.jsonl")
    args = parser.parse_args()

    polls = errors = gross_pos = net_pos = 0
    max_gross = max_net = 0.0
    best_net_capacity = 0.0
    end = time.monotonic() + args.seconds

    print(f"REST check: {args.seconds:.0f}s, every {args.interval:.1f}s, taker_fee={args.taker_fee * 100:.2f}%, "
          f"no orders placed")
    while time.monotonic() < end:
        started = time.monotonic()
        try:
            tops = fetch_tops()
        except Exception as exc:
            errors += 1
            print(f"poll error: {exc}")
        else:
            polls += 1
            opps = evaluate(tops, args.taker_fee)
            for o in opps:
                if o.gross_edge > 0:
                    gross_pos += 1
                    max_gross = max(max_gross, o.gross_edge)
                if o.net_edge > 0:
                    net_pos += 1
                    max_net = max(max_net, o.net_edge)
                    best_net_capacity = max(best_net_capacity, o.capacity_usd)
                row = {"ts": time.time(), "path": o.path, "gross_pct": round(o.gross_edge * 100, 4),
                       "net_pct": round(o.net_edge * 100, 4), "capacity_usd": round(o.capacity_usd, 2),
                       "taker_fee": args.taker_fee, "tops": tops}
                with open(args.log, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
            best = max(opps, key=lambda o: o.gross_edge)
            print(f"{time.strftime('%H:%M:%S')} best={best.path} gross={best.gross_edge * 100:.4f}% "
                  f"net={best.net_edge * 100:.4f}% capacity=${best.capacity_usd:,.0f}")

        time.sleep(max(0.0, args.interval - (time.monotonic() - started)))

    print(f"\nSummary: polls={polls} errors={errors} gross>0={gross_pos} net>0={net_pos} "
          f"max_gross={max_gross * 100:.4f}% max_net={max_net * 100:.4f}% "
          f"best_net_capacity=${best_net_capacity:,.0f}")
    if gross_pos == 0:
        print("No gap appeared on fresh REST books. The WebSocket gaps were likely stale.")


if __name__ == "__main__":
    main()
