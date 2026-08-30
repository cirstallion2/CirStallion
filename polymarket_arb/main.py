"""
CirStallion — Polymarket BTC Arb Bot
─────────────────────────────────────
Exploits the ~100 ms CLOB-repricing lag between Kraken spot and
Polymarket BTC UP/DOWN 5-minute markets.

Quick start
───────────
  cp .env.example .env
  # fill in credentials (or leave DRY_RUN=true to paper-trade)
  pip install -r requirements.txt
  python main.py

Architecture
────────────
  KrakenFeed      ──► real-time spot price + 5M OHLC (WebSocket, US-compliant)
  MiroFishEngine  ──► 100-node / 180-edge force-graph cluster detector
  PolymarketClient ─► live CLOB order-book poller + order submission
  SignalDetector  ──► detects lag > 0.3 %, applies graph + TV/CQ filters
  Executor        ──► FOK orders in < 100 ms
  RiskManager     ──► 0.5 % / trade · 2 % daily · −0.4 % hard-stop
"""

from __future__ import annotations

import asyncio
import logging
import signal as unix_signal
import sys
import time
from pathlib import Path

import colorlog  # type: ignore

from bot.kraken_feed import KrakenFeed
from bot.executor import Executor
from bot.graph_engine import MiroFishEngine
from bot.polymarket_client import PolymarketClient
from bot.risk_manager import RiskManager
from bot.signal_detector import SignalDetector
from config import CONFIG


# ─── Logging setup ───────────────────────────────────────────────────────────

def _setup_logging() -> None:
    level = getattr(logging, CONFIG.log_level.upper(), logging.INFO)

    console = colorlog.StreamHandler()
    console.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(white)s%(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        },
    ))

    file_handler = logging.FileHandler(CONFIG.log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)


logger = logging.getLogger(__name__)


# ─── Startup banner ───────────────────────────────────────────────────────────

_BANNER = r"""
   ██████╗██╗██████╗      █████╗ ██████╗ ██████╗      ██████╗  ██████╗ ████████╗
  ██╔════╝██║██╔══██╗    ██╔══██╗██╔══██╗██╔══██╗     ██╔══██╗██╔═══██╗╚══██╔══╝
  ██║     ██║██████╔╝    ███████║██████╔╝██████╔╝     ██████╔╝██║   ██║   ██║
  ██║     ██║██╔══██╗    ██╔══██║██╔══██╗██╔══██╗     ██╔══██╗██║   ██║   ██║
  ╚██████╗██║██║  ██║    ██║  ██║██║  ██║██████╔╝     ██████╔╝╚██████╔╝   ██║
   ╚═════╝╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝      ╚═════╝  ╚═════╝    ╚═╝
  Polymarket BTC 5-MIN Arb  |  MiroFish v1  |  <100ms execution
"""


def _print_config_summary() -> None:
    r = CONFIG.risk
    logger.info("─── Risk Parameters ─────────────────────────────────────────")
    logger.info("  Portfolio      : $%.0f USDC", r.portfolio_size_usdc)
    logger.info("  Per-trade risk : %.1f%%", r.per_trade_risk_pct * 100)
    logger.info("  Daily cap      : +%.1f%%", r.daily_cap_pct * 100)
    logger.info("  Hard stop      : -%.1f%%", r.hard_stop_pct * 100)
    logger.info("  Min edge       : +%.1f%% CLOB lag", r.min_edge_pct * 100)
    logger.info("─── Mode ────────────────────────────────────────────────────")
    logger.info("  DRY RUN        : %s", CONFIG.dry_run)
    logger.info("─────────────────────────────────────────────────────────────")


# ─── Main loop ───────────────────────────────────────────────────────────────

async def run() -> None:
    print(_BANNER)
    _setup_logging()
    _print_config_summary()

    # ── Initialise components ────────────────────────────────────────────
    feed = KrakenFeed(CONFIG.kraken)
    graph = MiroFishEngine(CONFIG.graph)
    poly = PolymarketClient(CONFIG.polymarket)
    risk = RiskManager(CONFIG.risk)
    detector = SignalDetector(feed, graph, poly, CONFIG)
    executor = Executor(poly, risk, CONFIG)

    # ── Wire graceful shutdown ────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _shutdown(*_):
        logger.info("Shutdown signal received — stopping…")
        stop_event.set()

    for sig in (unix_signal.SIGINT, unix_signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    # ── Start background tasks ────────────────────────────────────────────
    feed_task = asyncio.create_task(feed.start(), name="kraken-feed")
    poly_task = asyncio.create_task(poly.start(dry_run=CONFIG.dry_run), name="poly-client")
    detector_task = asyncio.create_task(detector.start(), name="detector-init")

    # Wait until the feed has warmed up
    logger.info("Waiting for Kraken feed to warm up…")
    for _ in range(100):
        if feed.is_ready():
            break
        await asyncio.sleep(0.1)

    if not feed.is_ready():
        logger.warning("Feed not ready after 10 s — continuing anyway")

    logger.info("BTC price: $%.2f  |  klines loaded: %d", feed.latest_price, len(feed.klines))

    # ── Stats printer ─────────────────────────────────────────────────────
    stats_interval = 30.0
    last_stats = time.monotonic()

    # ── Trading loop ──────────────────────────────────────────────────────
    logger.info("Trading loop started (interval: %.0f ms)", CONFIG.loop_interval_s * 1000)

    while not stop_event.is_set():
        try:
            signal = await detector.detect()

            if signal:
                await executor.execute(signal)

            # Periodic stats
            now = time.monotonic()
            if now - last_stats >= stats_interval:
                last_stats = now
                ex_stats = executor.stats()
                rm_stats = risk.summary()
                graph_stats = graph.cluster_summary()
                logger.info(
                    "STATS | orders=%d rate=%.1f/s fills=%d pnl=+%.3fU "
                    "lat=%.0fms | daily_pnl=%.3fU (%.2f%%) trades=%d | "
                    "graph=%s conf=%.2f",
                    ex_stats["orders"], ex_stats["orders_per_sec"],
                    ex_stats["fills"], ex_stats["total_pnl_est_usdc"],
                    ex_stats["avg_latency_ms"],
                    rm_stats["daily_pnl_usdc"], rm_stats["daily_pnl_pct"],
                    rm_stats["trades"],
                    graph_stats["direction"], graph_stats["confidence"],
                )

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Loop error: %s", exc, exc_info=True)

        await asyncio.sleep(CONFIG.loop_interval_s)

    # ── Teardown ─────────────────────────────────────────────────────────
    logger.info("Shutting down…")
    feed.stop()
    await poly.stop()
    await detector.stop()

    feed_task.cancel()
    poly_task.cancel()
    detector_task.cancel()

    for t in (feed_task, poly_task, detector_task):
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    # Final summary
    ex_stats = executor.stats()
    rm_stats = risk.summary()
    logger.info("─── Session Summary ─────────────────────────────────────────")
    logger.info("  Total orders   : %d (%.1f/s avg)", ex_stats["orders"], ex_stats["orders_per_sec"])
    logger.info("  Fills          : %d", ex_stats["fills"])
    logger.info("  Est. P&L       : +%.4f USDC", ex_stats["total_pnl_est_usdc"])
    logger.info("  Avg latency    : %.1f ms", ex_stats["avg_latency_ms"])
    logger.info("  Daily P&L      : %.4f USDC (%.3f%%)", rm_stats["daily_pnl_usdc"], rm_stats["daily_pnl_pct"])
    logger.info("─────────────────────────────────────────────────────────────")


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
