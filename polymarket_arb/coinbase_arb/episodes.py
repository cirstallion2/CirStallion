"""
Group logged gaps into episodes and report how long each one lasted.

    python -m coinbase_arb.episodes kraken_zero_fee.jsonl

Consecutive rows for the same path within --merge-gap seconds form one
episode. Durations are only as fine as the scanner's evaluation interval
(0.1 s by default), so a single-observation episode shows as 0 ms.
"""

from __future__ import annotations

import argparse
import json
import statistics


def episodes(rows: list[dict], merge_gap_s: float = 0.5) -> list[dict]:
    out: list[dict] = []
    last: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: (r["path"], r["ts"])):
        usd = r["capacity_usd"] * r["net_pct"] / 100
        cur = last.get(r["path"])
        if cur and r["ts"] - cur["end"] <= merge_gap_s:
            cur["end"] = r["ts"]
            cur["count"] += 1
            cur["best_usd"] = max(cur["best_usd"], usd)
            cur["max_net_pct"] = max(cur["max_net_pct"], r["net_pct"])
        else:
            cur = {"path": r["path"], "start": r["ts"], "end": r["ts"], "count": 1,
                   "best_usd": usd, "max_net_pct": r["net_pct"]}
            out.append(cur)
            last[r["path"]] = cur
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Episode durations for logged net-positive gaps")
    parser.add_argument("log")
    parser.add_argument("--merge-gap", type=float, default=0.5,
                        help="Seconds of silence that end an episode")
    args = parser.parse_args()

    with open(args.log, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    rows = [r for r in rows if r["net_pct"] > 0]
    if not rows:
        print("No net-positive rows in this log.")
        return

    eps = episodes(rows, args.merge_gap)
    durs_ms = [(e["end"] - e["start"]) * 1000 for e in eps]
    best = [e["best_usd"] for e in eps]
    longest = max(eps, key=lambda e: e["end"] - e["start"])

    print(f"rows={len(rows)} episodes={len(eps)}")
    print(f"duration ms: median={statistics.median(durs_ms):.0f} "
          f"p90={sorted(durs_ms)[int(0.9 * (len(durs_ms) - 1))]:.0f} max={max(durs_ms):.0f}")
    print(f"episodes at 0 ms (seen once): {sum(d == 0 for d in durs_ms)} of {len(eps)}")
    print(f"sum of best $ per episode: {sum(best):,.2f}")
    print(f"median best $ per episode: {statistics.median(best):,.2f}")
    print(f"longest episode: {longest['path']} lasted {(longest['end'] - longest['start']) * 1000:.0f} ms, "
          f"max net {longest['max_net_pct']:.4f}%")


if __name__ == "__main__":
    main()
