"""
Verdict on a paper-trading run, read from its JSONL log.

    python -m coinbase_arb.paper_report kraken_paper_trades.jsonl --start 11

The verdict compares final equity with starting equity. Because the paper
model fills only persistent gaps, it cannot show losing trades, so a
PROFITABLE result here is an upper bound, not proof.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime


def build_report(rows: list[dict], starting_equity: float) -> str:
    fills = [r for r in rows if r.get("event") == "fill"]
    halts = [r for r in rows if r.get("event") == "halt"]
    equity = fills[-1]["equity"] if fills else starting_equity
    pnl = equity - starting_equity
    pct = pnl / starting_equity * 100
    traded = sum(f["size_usd"] for f in fills)

    lines = [
        f"fills:             {len(fills)}",
        f"total traded:      ${traded:,.2f}",
        f"starting equity:   ${starting_equity:,.4f}",
        f"ending equity:     ${equity:,.4f}",
        f"net P&L:           ${pnl:+,.4f} ({pct:+.4f}%)",
    ]
    if fills:
        first, last = fills[0]["ts"], fills[-1]["ts"]
        hours = max((last - first) / 3600, 1e-9)
        lines.append(f"span:              {datetime.fromtimestamp(first):%Y-%m-%d %H:%M} to "
                     f"{datetime.fromtimestamp(last):%H:%M} ({hours:.2f} h)")
        lines.append(f"fills per hour:    {len(fills) / hours:,.1f}")
        lines.append(f"avg P&L per fill:  ${pnl / len(fills):+,.6f}")
    if halts:
        lines.append("HALTED: drawdown limit reached during the run")

    if pnl > 0:
        verdict = "PROFITABLE in simulation"
    elif pnl == 0:
        verdict = "BREAKEVEN (no fills)"
    else:
        verdict = "NOT PROFITABLE"
    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    lines.append("Caveat: the simulation cannot show losing trades. Treat PROFITABLE as an upper bound.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verdict for a paper-trading run")
    parser.add_argument("log", help="paper trades JSONL, e.g. kraken_paper_trades.jsonl")
    parser.add_argument("--start", type=float, default=11.0, help="Starting equity used in the run")
    args = parser.parse_args()

    with open(args.log, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    print(build_report(rows, args.start))


if __name__ == "__main__":
    main()
