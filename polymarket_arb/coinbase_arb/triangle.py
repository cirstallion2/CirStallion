"""
Triangular arbitrage on Coinbase: BTC-USD, ETH-USD and ETH-BTC.

ETH-BTC is quoted in BTC per 1 ETH. Two round-trip paths start and end in USD:

  Path A  USD → ETH (buy ETH-USD at ask)
          → BTC (sell ETH-BTC at bid)
          → USD (sell BTC-USD at bid)

  Path B  USD → BTC (buy BTC-USD at ask)
          → ETH (buy ETH-BTC at ask)
          → USD (sell ETH-USD at bid)

Every leg is a taker fill at top of book, so ``net_edge`` deducts the taker
fee three times. ``capacity_usd`` is the largest starting notional the
top-of-book sizes on all three legs can absorb.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .book import Top

BTC_USD = "BTC-USD"
ETH_USD = "ETH-USD"
ETH_BTC = "ETH-BTC"
PRODUCTS = (BTC_USD, ETH_USD, ETH_BTC)

DEFAULT_TAKER_FEE = 0.006   # 0.60% — placeholder, check your Coinbase fee tier


@dataclass(frozen=True)
class Opportunity:
    path: str            # "USD>ETH>BTC>USD" or "USD>BTC>ETH>USD"
    gross_edge: float    # round-trip return before fees, e.g. 0.002 = 0.2%
    net_edge: float      # after taker fees on all three legs
    capacity_usd: float  # max starting notional the top-of-book sizes allow


def breakeven_gross_edge(taker_fee: float = DEFAULT_TAKER_FEE) -> float:
    """Gross edge a path must exceed before taker fees are covered."""
    return 1.0 / (1.0 - taker_fee) ** 3 - 1.0


def evaluate(tops: Dict[str, Optional[Top]], taker_fee: float = DEFAULT_TAKER_FEE) -> List[Opportunity]:
    """
    Price both triangle paths from top-of-book quotes.

    Returns an empty list if any of the three books has no quote yet.
    """
    if any(tops.get(p) is None for p in PRODUCTS):
        return []

    bid_btc, bsz_btc, ask_btc, asz_btc = tops[BTC_USD]
    bid_eth, bsz_eth, ask_eth, asz_eth = tops[ETH_USD]
    bid_x, bsz_x, ask_x, asz_x = tops[ETH_BTC]

    fee_mult = (1.0 - taker_fee) ** 3

    # Path A: buy ETH with USD, sell ETH for BTC, sell BTC for USD.
    ratio_a = bid_btc * bid_x / ask_eth
    cap_a = min(
        asz_eth * ask_eth,                 # leg 1: ETH ask depth
        bsz_x * ask_eth,                   # leg 2: ETH sold into ETH-BTC bid depth
        bsz_btc * ask_eth / bid_x,         # leg 3: BTC sold into BTC-USD bid depth
    )

    # Path B: buy BTC with USD, buy ETH with BTC, sell ETH for USD.
    ratio_b = bid_eth / (ask_btc * ask_x)
    cap_b = min(
        asz_btc * ask_btc,                 # leg 1: BTC ask depth
        asz_x * ask_x * ask_btc,           # leg 2: ETH bought from ETH-BTC ask depth
        bsz_eth * ask_btc * ask_x,         # leg 3: ETH sold into ETH-USD bid depth
    )

    return [
        Opportunity("USD>ETH>BTC>USD", ratio_a - 1.0, ratio_a * fee_mult - 1.0, cap_a),
        Opportunity("USD>BTC>ETH>USD", ratio_b - 1.0, ratio_b * fee_mult - 1.0, cap_b),
    ]
