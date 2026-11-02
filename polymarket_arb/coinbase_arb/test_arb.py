"""
Offline checks for the Coinbase triangle scanner. No network needed.

    python -m unittest coinbase_arb.test_arb -v
"""

import json
import os
import tempfile
import time
import unittest

from .book import OrderBook
from .scanner import CoinbaseScanner, KrakenScanner
from .triangle import BTC_USD, ETH_BTC, ETH_USD, evaluate, breakeven_gross_edge


def _tops(btc=(50000.0, 1.0, 50000.0, 1.0), eth_usd=(2500.0, 10.0, 2500.0, 10.0),
          eth_btc=(0.05, 100.0, 0.05, 100.0)):
    return {BTC_USD: btc, ETH_USD: eth_usd, ETH_BTC: eth_btc}


class OrderBookTests(unittest.TestCase):
    def test_snapshot_then_updates(self):
        b = OrderBook("BTC-USD")
        b.apply_update({"changes": [["buy", "1", "1"]]}, 0.0)   # ignored before snapshot
        self.assertFalse(b.ready)

        b.apply_snapshot({"bids": [["100", "1"], ["99", "2"]], "asks": [["101", "1"]]}, 1.0)
        self.assertEqual(b.top(), (100.0, 1.0, 101.0, 1.0))

        b.apply_update({"changes": [["buy", "100", "0"], ["sell", "100.5", "3"]]}, 2.0)
        self.assertEqual(b.top(), (99.0, 2.0, 100.5, 3.0))
        self.assertEqual(b.last_update, 2.0)

    def test_reset_requires_new_snapshot(self):
        b = OrderBook("ETH-BTC")
        b.apply_snapshot({"bids": [["1", "1"]], "asks": [["2", "1"]]}, 0.0)
        b.reset()
        self.assertIsNone(b.top())
        b.apply_update({"changes": [["buy", "1", "5"]]}, 1.0)
        self.assertIsNone(b.top())


class TriangleTests(unittest.TestCase):
    def test_consistent_prices_have_no_edge(self):
        opps = evaluate(_tops())
        self.assertEqual(len(opps), 2)
        for o in opps:
            self.assertAlmostEqual(o.gross_edge, 0.0, places=12)
            self.assertLess(o.net_edge, 0)

    def test_path_a_mispricing_is_detected(self):
        # ETH-BTC bid lifted to 0.051 BTC/ETH: sell ETH into BTC, buy back via USD.
        opps = evaluate(_tops(eth_btc=(0.051, 100.0, 0.051, 100.0)))
        a = opps[0]
        self.assertEqual(a.path, "USD>ETH>BTC>USD")
        self.assertAlmostEqual(a.gross_edge, 0.02, places=9)
        self.assertAlmostEqual(a.net_edge, 1.02 * (1 - 0.006) ** 3 - 1, places=9)
        self.assertGreater(a.gross_edge, 0)
        self.assertEqual(opps[1].path, "USD>BTC>ETH>USD")
        self.assertLess(opps[1].gross_edge, 0)

    def test_path_b_mispricing_is_detected(self):
        # ETH-USD bid lifted to 2600: buy BTC, buy ETH with BTC, sell ETH for USD.
        opps = evaluate(_tops(eth_usd=(2600.0, 10.0, 2600.0, 10.0)))
        b = opps[1]
        self.assertAlmostEqual(b.gross_edge, 2600 / 2500 - 1, places=9)
        self.assertGreater(b.gross_edge, 0)

    def test_fees_kill_small_gaps(self):
        # 0.5% gross is below the ~1.8% break-even at a 0.60% taker fee.
        opps = evaluate(_tops(eth_btc=(0.05 * 1.005, 100.0, 0.05 * 1.005, 100.0)))
        self.assertGreater(opps[0].gross_edge, 0)
        self.assertLess(opps[0].net_edge, 0)
        self.assertLess(opps[0].gross_edge, breakeven_gross_edge(0.006))

    def test_breakeven_matches_fee_math(self):
        self.assertAlmostEqual(breakeven_gross_edge(0.006), 1 / 0.994 ** 3 - 1, places=12)
        self.assertAlmostEqual(breakeven_gross_edge(0.0), 0.0, places=12)

    def test_capacity_limited_by_thinnest_leg(self):
        # Path A leg 2 has only 0.5 ETH on the bid of ETH-BTC -> caps notional at 0.5 * 2500 = $1,250.
        opps = evaluate(_tops(eth_btc=(0.051, 0.5, 0.051, 100.0)))
        self.assertAlmostEqual(opps[0].capacity_usd, 0.5 * 2500.0, places=6)

    def test_missing_book_returns_nothing(self):
        tops = _tops()
        tops[ETH_BTC] = None
        self.assertEqual(evaluate(tops), [])


class ScannerTests(unittest.TestCase):
    def _feed(self, scanner, msg):
        scanner._handle(json.dumps(msg))

    def test_end_to_end_logs_positive_gross_only(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "opps.jsonl")
            s = CoinbaseScanner(taker_fee=0.006, log_path=log, eval_interval_s=0.0)

            self._feed(s, {"type": "snapshot", "product_id": BTC_USD,
                           "bids": [["50000", "1"]], "asks": [["50000", "1"]]})
            self._feed(s, {"type": "snapshot", "product_id": ETH_USD,
                           "bids": [["2500", "10"]], "asks": [["2500", "10"]]})
            # Mispriced ETH-BTC bid -> Path A has positive gross edge.
            self._feed(s, {"type": "snapshot", "product_id": ETH_BTC,
                           "bids": [["0.051", "100"]], "asks": [["0.0515", "100"]]})
            # Trigger an evaluation via an update.
            self._feed(s, {"type": "l2update", "product_id": ETH_BTC,
                           "changes": [["buy", "0.051", "100"]], "time": "t"})

            with open(log, encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh]
            self.assertGreaterEqual(len(rows), 1)
            self.assertEqual(rows[0]["path"], "USD>ETH>BTC>USD")
            self.assertGreater(rows[0]["gross_pct"], 0)
            self.assertGreater(s.stats.evaluations, 0)

    def test_stale_book_blocks_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            s = CoinbaseScanner(taker_fee=0.006, log_path=os.path.join(d, "x.jsonl"), eval_interval_s=0.0,
                                max_staleness_s=1.0)
            for pid in (BTC_USD, ETH_USD, ETH_BTC):
                s.books[pid].apply_snapshot({"bids": [["1", "1"]], "asks": [["2", "1"]]},
                                            time.monotonic() - 10.0)
            s._maybe_evaluate(time.monotonic())
            self.assertEqual(s.stats.evaluations, 0)
            self.assertGreater(s.stats.stale_skips, 0)


class KrakenScannerTests(unittest.TestCase):
    def _feed(self, scanner, msg):
        scanner._handle(json.dumps(msg))

    def _snapshot(self, scanner, pair, bid, ask, size="10"):
        self._feed(scanner, [0, {"as": [[ask, size, "1"]], "bs": [[bid, size, "1"]]},
                             "book-10", pair])

    def test_snapshot_then_update_triggers_path_a(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "k.jsonl")
            s = KrakenScanner(taker_fee=0.004, log_path=log, eval_interval_s=0.0)
            self._snapshot(s, "XBT/USD", "50000.0", "50000.0", "1")
            self._snapshot(s, "ETH/USD", "2500.0", "2500.0", "10")
            self._snapshot(s, "ETH/XBT", "0.0510", "0.0515", "100")
            # Update the ETH/XBT bid to the same level with new size -> evaluation fires.
            self._feed(s, [0, {"b": [["0.0510", "100.00000000", "2"]]}, "book-10", "ETH/XBT"])

            with open(log, encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh]
            self.assertTrue(rows)
            self.assertEqual(rows[0]["venue"], "kraken")
            self.assertEqual(rows[0]["path"], "USD>ETH>BTC>USD")
            self.assertAlmostEqual(rows[0]["gross_pct"], 2.0, places=3)

    def test_zero_volume_removes_level(self):
        with tempfile.TemporaryDirectory() as d:
            s = KrakenScanner(taker_fee=0.004, log_path=os.path.join(d, "k.jsonl"), eval_interval_s=0.0)
            self._snapshot(s, "ETH/XBT", "0.0510", "0.0515", "100")
            s._handle(json.dumps([0, {"b": [["0.0510", "0.00000000", "3"]]}, "book-10", "ETH/XBT"]))
            top = s.books["ETH-BTC"].top()
            self.assertIsNone(top)   # the only bid was removed

    def test_non_book_and_unknown_messages_are_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            s = KrakenScanner(taker_fee=0.004, log_path=os.path.join(d, "k.jsonl"))
            s._handle(json.dumps({"event": "heartbeat"}))
            s._handle(json.dumps({"event": "systemStatus", "status": "online"}))
            s._handle(json.dumps([0, {"a": [["1", "1", "1"]]}, "trade", "XBT/USD"]))
            s._handle(json.dumps([0, {"as": [["1", "1", "1"]], "bs": []}, "book-10", "DOGE/USD"]))
            s._handle("not json")
            self.assertFalse(any(b.ready for b in s.books.values()))


class RestCheckTests(unittest.TestCase):
    def test_parse_depth_top_of_book(self):
        from .rest_check import parse_depth
        payload = {"error": [], "result": {"XXBTZUSD": {
            "asks": [["50001.0", "0.5", "1"]], "bids": [["50000.0", "1.2", "1"]]}}}
        self.assertEqual(parse_depth(payload), (50000.0, 1.2, 50001.0, 0.5))

    def test_parse_depth_raises_on_api_error(self):
        from .rest_check import parse_depth
        with self.assertRaises(ValueError):
            parse_depth({"error": ["EQuery:Unknown asset pair"], "result": {}})


class PaperTraderTests(unittest.TestCase):
    def _gap_tops(self):
        # ETH-BTC bid lifted to 0.051: path A is +2% gross and net at zero fee.
        return _tops(eth_btc=(0.051, 100.0, 0.051, 100.0))

    def test_gap_that_persists_fills_at_capped_size(self):
        from .paper_trader import PaperTrader
        with tempfile.TemporaryDirectory() as d:
            p = PaperTrader(starting_equity=11.0, taker_fee=0.0, latency_s=0.3,
                            log_path=os.path.join(d, "p.jsonl"))
            p.on_eval(0.0, self._gap_tops())          # seen
            p.on_eval(0.5, self._gap_tops())          # still there after latency -> fill
            self.assertEqual(p.fills, 1)
            self.assertAlmostEqual(p.equity, 11.0 + 11.0 * 0.02, places=9)

    def test_gap_that_closes_is_missed(self):
        from .paper_trader import PaperTrader
        with tempfile.TemporaryDirectory() as d:
            p = PaperTrader(starting_equity=11.0, taker_fee=0.0, latency_s=0.3,
                            log_path=os.path.join(d, "p.jsonl"))
            p.on_eval(0.0, self._gap_tops())
            p.on_eval(0.5, _tops())                   # gap closed before latency elapsed
            self.assertEqual(p.fills, 0)
            self.assertEqual(p.missed, 1)
            self.assertEqual(p.equity, 11.0)

    def test_halts_at_drawdown_limit(self):
        from .paper_trader import PaperTrader
        with tempfile.TemporaryDirectory() as d:
            p = PaperTrader(starting_equity=11.0, taker_fee=0.0, latency_s=0.0,
                            halt_fraction=0.5, log_path=os.path.join(d, "p.jsonl"))
            p.equity = 5.0                            # below the 50% line
            p.on_eval(0.0, self._gap_tops())
            self.assertTrue(p.halted)


class BookDepthTests(unittest.TestCase):
    def test_levels_outside_window_are_dropped(self):
        b = OrderBook("ETH-BTC", max_depth=2)
        b.apply_snapshot({"bids": [["10", "1"], ["9", "1"]], "asks": [["11", "1"], ["12", "1"]]}, 0.0)
        # A better bid and a better ask arrive: the worst level on each side falls out of the window.
        b.apply_update({"changes": [["buy", "10.5", "1"], ["sell", "10.9", "1"]]}, 1.0)
        self.assertEqual(sorted(b.bids), [10.0, 10.5])
        self.assertEqual(sorted(b.asks), [10.9, 11.0])

    def test_no_pruning_without_max_depth(self):
        b = OrderBook("BTC-USD")
        b.apply_snapshot({"bids": [["1", "1"], ["2", "1"], ["3", "1"]], "asks": [["4", "1"]]}, 0.0)
        self.assertEqual(len(b.bids), 3)

    def test_kraken_scanner_prunes_to_subscribed_depth(self):
        self.assertEqual(KrakenScanner(taker_fee=0.0, log_path=os.devnull).books["BTC-USD"].max_depth, 10)
        self.assertIsNone(CoinbaseScanner(taker_fee=0.0, log_path=os.devnull).books["BTC-USD"].max_depth)


class CompareTests(unittest.TestCase):
    def test_compare_tops_relative_difference(self):
        from .ws_compare import compare_tops
        bd, ad = compare_tops((100.0, 1, 101.0, 1), (100.0, 1, 100.0, 1))
        self.assertAlmostEqual(bd, 0.0)
        self.assertAlmostEqual(ad, 0.01)


class PaperCooldownTests(unittest.TestCase):
    def test_cooldown_blocks_repeat_fill_on_same_gap(self):
        from .paper_trader import PaperTrader
        gap = _tops(eth_btc=(0.051, 100.0, 0.051, 100.0))   # path A is +2% gross and net at zero fee
        with tempfile.TemporaryDirectory() as d:
            p = PaperTrader(starting_equity=11.0, taker_fee=0.0, latency_s=0.3, cooldown_s=10.0,
                            log_path=os.path.join(d, "p.jsonl"))
            p.on_eval(0.0, gap)          # seen
            p.on_eval(0.5, gap)          # fill
            p.on_eval(1.0, gap)          # re-registration blocked by cooldown
            p.on_eval(2.0, gap)
            p.on_eval(2.5, gap)
            self.assertEqual(p.fills, 1)


class PaperReportTests(unittest.TestCase):
    def test_profitable_verdict(self):
        from .paper_report import build_report
        rows = [{"event": "fill", "ts": 1000.0, "size_usd": 11.0, "equity": 11.5}]
        out = build_report(rows, 11.0)
        self.assertIn("VERDICT: PROFITABLE in simulation", out)
        self.assertIn("net P&L:           $+0.5000", out)

    def test_no_fills_is_breakeven(self):
        from .paper_report import build_report
        self.assertIn("VERDICT: BREAKEVEN", build_report([], 11.0))

    def test_implausible_fill_rate_is_invalid(self):
        from .paper_report import build_report
        # 3 fills over 60 s = 180 fills/hour, above the 120/hour limit
        rows = [{"event": "fill", "ts": 1000.0 + t, "size_usd": 11.0, "equity": 11.5}
                for t in (0.0, 30.0, 60.0)]
        out = build_report(rows, 11.0)
        self.assertIn("INVALID", out)
        self.assertIn("WARNING", out)

    def test_halt_is_reported(self):
        from .paper_report import build_report
        rows = [{"event": "fill", "ts": 1000.0, "size_usd": 11.0, "equity": 4.0},
                {"event": "halt", "ts": 1001.0, "equity": 4.0}]
        out = build_report(rows, 11.0)
        self.assertIn("NOT PROFITABLE", out)
        self.assertIn("HALTED", out)


if __name__ == "__main__":
    unittest.main()
