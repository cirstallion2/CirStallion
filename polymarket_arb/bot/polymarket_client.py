"""
Polymarket CLOB client — market discovery, real-time order-book polling,
and order submission for BTC UP/DOWN 5-minute markets.

Authentication
--------------
Uses py-clob-client with an EOA private key (signature_type=0).
Credentials are loaded from .env / environment variables.

Dry-run
-------
When CONFIG.dry_run is True (default), place_order() logs the intent
but never submits to the CLOB.  Set DRY_RUN=false in .env + fund your
wallet before going live.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# Lazy import — py-clob-client is optional at import time so the rest of the
# codebase can be linted / tested without it installed.
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.constants import POLYGON
    from py_clob_client.order_builder.constants import BUY, SELL

    _CLOB_AVAILABLE = True
except ImportError:
    _CLOB_AVAILABLE = False
    logger.warning("py-clob-client not installed — Polymarket orders disabled")


@dataclass
class OrderBook:
    token_id: str
    bids: List[Dict]    # [{"price": float, "size": float}, …]  best bid first
    asks: List[Dict]    # best ask first
    mid: float = 0.0
    spread: float = 0.0
    bid_depth_usdc: float = 0.0
    ask_depth_usdc: float = 0.0
    fetched_at: float = field(default_factory=time.monotonic)

    def is_stale(self, max_age_s: float = 1.0) -> bool:
        return time.monotonic() - self.fetched_at > max_age_s


@dataclass
class ActiveMarket:
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    yes_price: float     # current best ask for YES (= prob UP)
    no_price: float
    active: bool = True


@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    filled_size: float = 0.0
    filled_price: float = 0.0
    error: str = ""
    latency_ms: float = 0.0


class PolymarketClient:
    """
    Wraps the Polymarket CLOB for the arb bot.

    Key responsibilities
    --------------------
    • Discover and cache active BTC UP/DOWN 5-min markets.
    • Maintain a fresh order-book snapshot (polled via REST).
    • Submit FOK orders with latency tracking.
    """

    _BOOK_TTL_S = 0.2    # refresh order book every 200 ms

    def __init__(self, config):
        self._cfg = config
        self._clob_url = config.clob_url
        self._dry_run = True   # overridden in start() via CONFIG.dry_run

        self._client: Optional[object] = None   # ClobClient if available
        self._session: Optional[aiohttp.ClientSession] = None

        self.active_markets: List[ActiveMarket] = []
        self._books: Dict[str, OrderBook] = {}       # token_id → OrderBook
        self._running = False

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self, dry_run: bool = True) -> None:
        self._dry_run = dry_run
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3.0)
        )
        self._init_clob_client()
        await self._discover_markets()
        logger.info(
            "PolymarketClient ready | dry_run=%s | markets found: %d",
            self._dry_run, len(self.active_markets),
        )
        self._running = True
        asyncio.create_task(self._book_refresh_loop())

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()

    # ─── Market data ─────────────────────────────────────────────────────────

    def get_book(self, token_id: str) -> Optional[OrderBook]:
        return self._books.get(token_id)

    def get_mid_price(self, token_id: str) -> float:
        book = self._books.get(token_id)
        return book.mid if book else 0.0

    def get_best_ask(self, token_id: str) -> float:
        book = self._books.get(token_id)
        if not book or not book.asks:
            return 0.0
        return book.asks[0]["price"]

    def get_best_bid(self, token_id: str) -> float:
        book = self._books.get(token_id)
        if not book or not book.bids:
            return 0.0
        return book.bids[0]["price"]

    def get_liquidity(self, token_id: str, depth_usdc: float = 500.0) -> float:
        """Return available liquidity on the ask side up to *depth_usdc*."""
        book = self._books.get(token_id)
        if not book:
            return 0.0
        total = 0.0
        for ask in book.asks:
            lvl_usdc = ask["price"] * ask["size"]
            total += lvl_usdc
            if total >= depth_usdc:
                return total
        return total

    # ─── Order placement ─────────────────────────────────────────────────────

    async def place_order(
        self,
        token_id: str,
        side: str,          # "BUY" or "SELL"
        price: float,       # limit price in USDC (0–1 range)
        size_usdc: float,   # notional in USDC
    ) -> OrderResult:
        """
        Submit a FOK limit order.  Returns OrderResult with latency_ms filled.

        In dry_run mode logs the order intent and returns a mock success.
        """
        t0 = time.monotonic()
        size_shares = round(size_usdc / price, 2) if price > 0 else 0.0

        if self._dry_run:
            latency = (time.monotonic() - t0) * 1000
            logger.info(
                "[DRY-RUN] ORDER %s token=%s price=%.4f size=%.2f shares "
                "notional=%.2f USDC | latency=%.2fms",
                side, token_id[:12], price, size_shares, size_usdc, latency,
            )
            return OrderResult(
                success=True,
                order_id="DRY-RUN",
                filled_size=size_shares,
                filled_price=price,
                latency_ms=latency,
            )

        if not _CLOB_AVAILABLE or self._client is None:
            return OrderResult(success=False, error="CLOB client not initialised")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._submit_order_sync,
                token_id, side, price, size_shares,
            )
            result.latency_ms = (time.monotonic() - t0) * 1000
            return result
        except Exception as exc:
            return OrderResult(
                success=False,
                error=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _init_clob_client(self) -> None:
        if not _CLOB_AVAILABLE:
            return
        if not self._cfg.private_key or self._dry_run:
            logger.info("PolymarketClient: skipping auth (dry_run or no key)")
            return
        try:
            self._client = ClobClient(
                host=self._clob_url,
                key=self._cfg.private_key,
                chain_id=POLYGON,
                signature_type=0,
                funder=self._cfg.funder_address or None,
            )
            logger.info("PolymarketClient: CLOB client authenticated")
        except Exception as exc:
            logger.error("PolymarketClient: auth failed: %s", exc)

    async def _discover_markets(self) -> None:
        """Pull all markets and filter for active BTC 5-min UP/DOWN contracts."""
        url = f"{self._clob_url}/markets"
        try:
            async with self._session.get(url) as resp:
                data = await resp.json()
                markets_raw = data.get("data", data) if isinstance(data, dict) else data
        except Exception as exc:
            logger.warning("PolymarketClient: market discovery failed: %s", exc)
            return

        found = []
        for m in markets_raw:
            question: str = m.get("question", "")
            if not self._is_btc_5min(question):
                continue
            tokens = m.get("tokens", [])
            if len(tokens) < 2:
                continue

            yes_tok = next((t for t in tokens if t.get("outcome", "").upper() in ("YES", "UP")), None)
            no_tok  = next((t for t in tokens if t.get("outcome", "").upper() in ("NO", "DOWN")), None)
            if not yes_tok or not no_tok:
                continue

            found.append(ActiveMarket(
                condition_id=m.get("condition_id", ""),
                question=question,
                yes_token_id=yes_tok["token_id"],
                no_token_id=no_tok["token_id"],
                yes_price=float(yes_tok.get("price", 0.5)),
                no_price=float(no_tok.get("price", 0.5)),
            ))

        self.active_markets = found
        logger.debug("Market discovery: %d BTC 5-min markets", len(found))

    def _is_btc_5min(self, question: str) -> bool:
        q = question.upper()
        has_btc = "BTC" in q or "BITCOIN" in q
        has_5min = "5 MIN" in q or "5MIN" in q or "5-MIN" in q or "5 MINUTE" in q
        return has_btc and has_5min

    async def _book_refresh_loop(self) -> None:
        """Poll order books for all tracked token IDs at _BOOK_TTL_S cadence."""
        while self._running:
            token_ids: List[str] = []
            for mkt in self.active_markets:
                token_ids.append(mkt.yes_token_id)
                token_ids.append(mkt.no_token_id)

            await asyncio.gather(
                *[self._refresh_book(tid) for tid in token_ids],
                return_exceptions=True,
            )

            # Also update best prices on ActiveMarket objects
            for mkt in self.active_markets:
                mkt.yes_price = self.get_best_ask(mkt.yes_token_id)
                mkt.no_price  = self.get_best_ask(mkt.no_token_id)

            await asyncio.sleep(self._BOOK_TTL_S)

    async def _refresh_book(self, token_id: str) -> None:
        url = f"{self._clob_url}/book"
        params = {"token_id": token_id}
        try:
            async with self._session.get(url, params=params) as resp:
                data = await resp.json()
        except Exception:
            return

        bids = [{"price": float(b["price"]), "size": float(b["size"])}
                for b in data.get("bids", [])]
        asks = [{"price": float(a["price"]), "size": float(a["size"])}
                for a in data.get("asks", [])]

        best_bid = bids[0]["price"] if bids else 0.0
        best_ask = asks[0]["price"] if asks else 0.0
        mid = (best_bid + best_ask) / 2.0 if (best_bid and best_ask) else best_bid or best_ask
        spread = best_ask - best_bid if (best_bid and best_ask) else 0.0

        bid_depth = sum(b["price"] * b["size"] for b in bids[:10])
        ask_depth = sum(a["price"] * a["size"] for a in asks[:10])

        self._books[token_id] = OrderBook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            mid=mid,
            spread=spread,
            bid_depth_usdc=bid_depth,
            ask_depth_usdc=ask_depth,
        )

    def _submit_order_sync(
        self, token_id: str, side: str, price: float, size: float
    ) -> OrderResult:
        """Blocking call — run via executor to avoid blocking the event loop."""
        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=BUY if side == "BUY" else SELL,
                order_type=OrderType.FOK,
            )
            resp = self._client.create_and_post_order(order_args)
            return OrderResult(
                success=True,
                order_id=getattr(resp, "order_id", ""),
                filled_size=float(getattr(resp, "size_matched", size)),
                filled_price=price,
            )
        except Exception as exc:
            return OrderResult(success=False, error=str(exc))
