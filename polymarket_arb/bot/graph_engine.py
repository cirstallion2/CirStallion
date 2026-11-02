"""
MiroFish Force-Graph Engine — 100 nodes / 180 edges.

Maps five categories of market signal onto a force-directed topology.
After each simulation pass, nodes with similar bullish/bearish polarity
cluster together.  The separation between the BULL and BEAR clusters
produces a convergence score and a directional call used to confirm
(or veto) trade signals from the CLOB lag detector.

Node layout
-----------
  0 – 19   : price-momentum nodes
 20 – 39   : volume-confirmation nodes
 40 – 59   : technical-indicator nodes (RSI / MACD)
 60 – 79   : exchange-flow nodes (CryptoQuant)
 80 – 99   : sentiment nodes (TradingView / Fear-Greed)

Edge set
--------
180 edges wired at construction time (seed 42 for reproducibility):
  – 60 intra-group edges  (12 per group × 5 groups)
  – 120 inter-group edges (24 per causal pair × 5 pairs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

_SEED = 42
_GROUP = 20          # nodes per group
_GROUPS = 5
_INTRA_PER_GROUP = 12
_INTER_PER_PAIR = 24


@dataclass
class NodeSignals:
    """Normalised market signals fed into the graph each tick (all in [-1, +1])."""
    price_momentum: float = 0.0      # from BinanceFeed.get_momentum()
    volume_ratio: float = 0.0        # (vol_ratio - 1) * 0.5, clipped
    rsi: float = 0.0                 # (RSI - 50) / 50
    macd: float = 0.0                # from BinanceFeed.get_macd_signal()
    exchange_flows: float = 0.0      # from CryptoQuant; negative outflow = bullish
    tradingview: float = 0.0         # TV recommendation in [-1, +1]
    fear_greed: float = 0.0          # (FG - 50) / 50
    clob_lag: float = 0.0            # normalised spot-vs-poly lag


class MiroFishEngine:
    """
    Force-directed graph that detects BULL/BEAR cluster convergence.

    Call sequence per tick
    ----------------------
    1. engine.update(signals)   — refresh node values
    2. engine.simulate()        — run spring/repulsion passes
    3. conf, dir = engine.get_convergence()
    """

    def __init__(self, config):
        self._n = config.num_nodes                        # 100
        self._threshold = config.convergence_threshold   # 0.65
        self._k_spring = config.spring_k
        self._k_rep = config.repulsion_k
        self._damping = config.damping
        self._iters = config.force_iterations
        self._noise = config.noise_sigma

        rng = np.random.default_rng(_SEED)

        # Node state
        self.positions = rng.standard_normal((_GROUPS * _GROUP, 2)) * 5.0
        self.velocities = np.zeros((_GROUPS * _GROUP, 2))
        self.values = np.zeros(_GROUPS * _GROUP)         # [-1 BEAR … +1 BULL]

        # Edge list shape (180, 2) — fixed topology
        self._edges = self._build_edges(rng)
        self._src = self._edges[:, 0]
        self._dst = self._edges[:, 1]

        logger.debug(
            "MiroFishEngine ready: %d nodes, %d edges",
            self._n, len(self._edges),
        )

    # ─── Public API ──────────────────────────────────────────────────────────

    def update(self, signals: NodeSignals) -> None:
        """Push latest market signals into node values."""
        rng = np.random.default_rng()   # fresh RNG each tick for noise

        # Price-momentum  (nodes 0-19)
        self.values[0:20] = signals.price_momentum + rng.standard_normal(20) * self._noise

        # Volume          (nodes 20-39)
        vol_sig = float(np.clip((signals.volume_ratio) * 0.5, -1.0, 1.0))
        # Volume confirms momentum direction
        signed_vol = vol_sig * np.sign(signals.price_momentum) if signals.price_momentum != 0 else vol_sig
        self.values[20:40] = signed_vol + rng.standard_normal(20) * self._noise

        # Technical       (nodes 40-59)
        tech = signals.rsi * 0.5 + signals.macd * 0.5
        self.values[40:60] = tech + rng.standard_normal(20) * self._noise

        # Exchange flows  (nodes 60-79)
        self.values[60:80] = signals.exchange_flows + rng.standard_normal(20) * self._noise

        # Sentiment       (nodes 80-99)
        sentiment = signals.tradingview * 0.55 + signals.fear_greed * 0.35 + signals.clob_lag * 0.10
        self.values[80:100] = sentiment + rng.standard_normal(20) * self._noise

        self.values = np.clip(self.values, -1.0, 1.0)

    def simulate(self) -> None:
        """Run force-directed simulation (vectorised, ~1–3 ms for 100 nodes)."""
        for _ in range(self._iters):
            forces = np.zeros_like(self.positions)

            # ── Spring forces along edges ──────────────────────────────────
            src_pos = self.positions[self._src]           # (E, 2)
            dst_pos = self.positions[self._dst]           # (E, 2)
            diff = dst_pos - src_pos                      # (E, 2)
            dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8  # (E,1)

            # Attraction is strongest when nodes share the same polarity
            similarity = (
                1.0 - np.abs(self.values[self._src] - self.values[self._dst])[:, None] / 2.0
            )                                             # (E, 1)
            spring = self._k_spring * similarity * diff / dist  # (E, 2)

            np.add.at(forces, self._src, spring)
            np.add.at(forces, self._dst, -spring)

            # ── Repulsion between all node pairs (vectorised) ──────────────
            # diff_mat[i,j] = positions[i] - positions[j]
            diff_mat = self.positions[:, None, :] - self.positions[None, :, :]  # (N,N,2)
            dist_sq = (diff_mat ** 2).sum(axis=2) + 1e-8                        # (N,N)

            val_diff = np.abs(self.values[:, None] - self.values[None, :]) / 2.0  # (N,N)
            rep_strength = self._k_rep * (1.0 + val_diff) / dist_sq              # (N,N)
            np.fill_diagonal(rep_strength, 0.0)

            dist_mat = np.sqrt(dist_sq)[:, :, None]                             # (N,N,1)
            rep_forces = (rep_strength[:, :, None] * diff_mat / dist_mat).sum(axis=1)  # (N,2)

            forces += rep_forces

            # ── Integrate ─────────────────────────────────────────────────
            self.velocities = self.velocities * self._damping + forces * 0.01
            self.positions += self.velocities

    def get_convergence(self) -> Tuple[float, str]:
        """
        Returns
        -------
        confidence : float in [0, 1]
            How strongly one cluster dominates.
        direction  : str
            'BULL', 'BEAR', or 'NEUTRAL' (below threshold).
        """
        bull_mask = self.values > 0
        bear_mask = self.values < 0

        n_bull = bull_mask.sum()
        n_bear = bear_mask.sum()

        if n_bull == 0 and n_bear == 0:
            return 0.0, "NEUTRAL"

        # Edge case: all nodes agree
        if n_bull == 0:
            return 0.80, "BEAR"
        if n_bear == 0:
            return 0.80, "BULL"

        bull_centroid = self.positions[bull_mask].mean(axis=0)
        bear_centroid = self.positions[bear_mask].mean(axis=0)

        # Geometric separation between cluster centroids
        separation = np.linalg.norm(bull_centroid - bear_centroid)

        # Signal-strength dominance
        bull_str = float(np.abs(self.values[bull_mask]).mean())
        bear_str = float(np.abs(self.values[bear_mask]).mean())
        total = bull_str + bear_str + 1e-8

        bull_dom = bull_str / total
        bear_dom = bear_str / total
        dominance = max(bull_dom, bear_dom)       # how one-sided the signal is

        sep_factor = float(np.clip(separation / 12.0, 0.0, 1.0))
        confidence = dominance * 0.60 + sep_factor * 0.40

        direction = "BULL" if bull_dom >= bear_dom else "BEAR"
        if confidence < self._threshold:
            direction = "NEUTRAL"

        return float(confidence), direction

    def cluster_summary(self) -> dict:
        """Diagnostic snapshot — useful for logging."""
        bull_mask = self.values > 0
        bear_mask = self.values < 0
        conf, direction = self.get_convergence()
        return {
            "bull_nodes": int(bull_mask.sum()),
            "bear_nodes": int(bear_mask.sum()),
            "neutral_nodes": int((self.values == 0).sum()),
            "confidence": round(conf, 3),
            "direction": direction,
            "mean_bull_value": round(float(self.values[bull_mask].mean()), 3) if bull_mask.any() else 0.0,
            "mean_bear_value": round(float(self.values[bear_mask].mean()), 3) if bear_mask.any() else 0.0,
        }

    # ─── Edge construction ───────────────────────────────────────────────────

    @staticmethod
    def _build_edges(rng: np.random.Generator) -> np.ndarray:
        """Wire 180 edges: 60 intra-group + 120 inter-group (deterministic)."""
        edges: set[tuple[int, int]] = set()

        # Intra-group: 12 random edges per group
        for g in range(_GROUPS):
            base = g * _GROUP
            attempts = 0
            while sum(1 for e in edges if base <= e[0] < base + _GROUP) < _INTRA_PER_GROUP:
                i = int(rng.integers(base, base + _GROUP))
                j = int(rng.integers(base, base + _GROUP))
                if i != j:
                    edges.add((min(i, j), max(i, j)))
                attempts += 1
                if attempts > 500:
                    break

        # Inter-group pairs: price↔volume, price↔tech, tech↔flows,
        #                    price↔sentiment, flows↔sentiment
        inter_pairs = [(0, 1), (0, 2), (2, 3), (0, 4), (3, 4)]
        for ga, gb in inter_pairs:
            base_a = ga * _GROUP
            base_b = gb * _GROUP
            added = 0
            attempts = 0
            while added < _INTER_PER_PAIR:
                i = int(rng.integers(base_a, base_a + _GROUP))
                j = int(rng.integers(base_b, base_b + _GROUP))
                key = (min(i, j), max(i, j))
                if key not in edges:
                    edges.add(key)
                    added += 1
                attempts += 1
                if attempts > 1000:
                    break

        # Trim / pad to exactly 180
        edge_list = list(edges)
        while len(edge_list) < 180:
            i = int(rng.integers(0, _GROUPS * _GROUP))
            j = int(rng.integers(0, _GROUPS * _GROUP))
            if i != j:
                key = (min(i, j), max(i, j))
                if key not in set(map(tuple, edge_list)):
                    edge_list.append(key)

        return np.array(edge_list[:180], dtype=np.int32)
