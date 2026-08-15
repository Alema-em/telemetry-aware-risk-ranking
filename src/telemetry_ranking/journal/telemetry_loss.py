"""Telemetry impairment channels for TNSM-oriented journal experiments.

Two loss models (matched mean loss rate p_loss):
  - independent: Bernoulli packet drops
  - bursty: Gilbert–Elliott two-state Markov (loss only in Bad)

On a drop the ranker consumes the last successfully delivered belief
(stale monitoring), while the environment continues to evolve.

Writes nowhere; journal runners consume this module. Reference artifacts are not modified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from telemetry_ranking.rank.env_utils import belief_probs, snapshot, restore


ChannelKind = Literal["independent", "bursty"]


@dataclass
class GilbertElliott:
    """Classic GE: lossless Good, always-loss Bad; mean loss ≈ p_loss."""

    p_loss: float
    mean_burst_len: float = 3.0

    def __post_init__(self) -> None:
        p = float(np.clip(self.p_loss, 0.0, 0.999))
        r = 1.0 / max(float(self.mean_burst_len), 1.0)  # P(Bad -> Good)
        # Steady-state P(Bad) = h / (h + r) = p  =>  h = p r / (1-p)
        h = 0.0 if p <= 0 else (p * r) / max(1.0 - p, 1e-12)
        self.p = p
        self.r = float(np.clip(r, 0.0, 1.0))
        self.h = float(np.clip(h, 0.0, 1.0))


class TelemetryChannel:
    """Per-episode packet-delivery process."""

    def __init__(
        self,
        kind: ChannelKind,
        p_loss: float,
        rng: np.random.Generator,
        mean_burst_len: float = 3.0,
    ) -> None:
        self.kind = kind
        self.p_loss = float(np.clip(p_loss, 0.0, 1.0))
        self.rng = rng
        self.ge = GilbertElliott(self.p_loss, mean_burst_len=mean_burst_len)
        self.in_bad = False
        self.n_steps = 0
        self.n_drops = 0

    def reset(self) -> None:
        self.in_bad = False
        self.n_steps = 0
        self.n_drops = 0
        if self.kind == "bursty" and self.p_loss > 0:
            # Start in steady-state occupancy of Bad
            self.in_bad = bool(self.rng.random() < self.p_loss)

    def delivered(self) -> bool:
        """Return True if this step's monitoring packet is delivered."""
        self.n_steps += 1
        if self.p_loss <= 0:
            return True
        if self.kind == "independent":
            drop = bool(self.rng.random() < self.p_loss)
        else:
            # Transition then emit
            if self.in_bad:
                if self.rng.random() < self.ge.r:
                    self.in_bad = False
            else:
                if self.rng.random() < self.ge.h:
                    self.in_bad = True
            drop = self.in_bad
        if drop:
            self.n_drops += 1
            return False
        return True

    @property
    def empirical_loss(self) -> float:
        return float(self.n_drops / max(self.n_steps, 1))


def cache_belief(env) -> np.ndarray:
    return belief_probs(env).copy()


def inject_belief(env, probs: np.ndarray) -> None:
    """Overwrite belief mass used by the ranker; leave other latent fields."""
    d = env.unpack_obs_array(env.current_state)
    p = np.asarray(probs, dtype=float).copy()
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0:
        p = np.ones_like(p) / max(len(p), 1)
    else:
        p = p / s
    env.current_state = env.create_obs_array(
        int(d["c_true"]),
        list(p),
        int(d["c_infer"]),
        float(d["h"]),
        float(d["d"]),
        int(d["z"]),
        int(d["w"]),
        float(d["s"]),
    )
    env.obs = env.extract_obs(env.chosen_obs_names, env.b_add_time_inhomogeneous)


def rank_with_telemetry(env, choose_fn, channel: TelemetryChannel, cached_probs: np.ndarray | None):
    """Rank under live or stale belief according to the channel.

    Returns (action, updated_cached_probs, dropped: bool).
    """
    live = belief_probs(env).copy()
    if cached_probs is None:
        cached_probs = live
    dropped = not channel.delivered()
    if dropped:
        snap = snapshot(env)
        try:
            inject_belief(env, cached_probs)
            a = int(choose_fn(env))
        finally:
            restore(env, snap)
        return a, cached_probs, True
    a = int(choose_fn(env))
    return a, live, False
