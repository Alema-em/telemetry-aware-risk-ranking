from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np

class GateDecision(str, Enum):
    ACT = "act"
    WAIT = "wait"
    DEFER = "defer"

PolicyName = Literal[
    "always_act",
    "age_wait",
    "peak_and_age",
    "always_act_frozen",
    "entropy_gate",
]

@dataclass
class GateConfig:
    policy: PolicyName = "peak_and_age"
    delta_star: int = 1
    pi_star: float = 0.7
    entropy_star: float = 0.6
    defer_action: int = 1
    wait_action: int = 0

def belief_peak(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    s = float(p.sum())
    if s <= 0:
        return 0.5
    return float(np.max(p / s))

def belief_entropy_norm(probs: np.ndarray) -> float:
    """Normalised Shannon entropy H(p)/log(|C|) in [0, 1] (1 = uniform)."""
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0 or len(p) < 2:
        return 1.0
    p = p / s
    p = np.clip(p, 1e-12, 1.0)
    h = float(-np.sum(p * np.log(p)))
    return float(h / np.log(len(p)))

def entropy_norm_from_peak(pi: float) -> float:
    p = float(np.clip(pi, 1e-9, 1.0 - 1e-9))
    h = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
    return float(h / np.log(2.0))

def sticky_transition(stay: float = 0.9) -> np.ndarray:
    """Simple 2-state context transition for belief diffusion."""
    stay = float(np.clip(stay, 0.5, 0.999))
    flip = 1.0 - stay
    return np.array([[stay, flip], [flip, stay]], dtype=float)

def diffuse_belief(probs: np.ndarray, delta: int, M: np.ndarray | None = None) -> np.ndarray:
    """Diffuse stored belief forward Δ steps: p̃ = (M^T)^Δ p (row-stochastic M)."""
    p = np.asarray(probs, dtype=float).copy()
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    p = np.ones_like(p) / len(p) if s <= 0 else p / s
    if delta <= 0:
        return p
    if M is None:
        M = sticky_transition(0.9)
    # row-stochastic: next = p @ M
    out = p.copy()
    for _ in range(int(delta)):
        out = out @ M
    out = np.clip(out, 0.0, None)
    out = out / max(float(out.sum()), 1e-12)
    return out

def decide_gate(delta: int, pi: float, cfg: GateConfig) -> GateDecision:
    """Receiver-side intervene scheduler under belief age."""
    if cfg.policy in ("always_act", "always_act_frozen"):
        return GateDecision.ACT
    # Act only when belief is confident (low normalised entropy), else Defer.
    if cfg.policy == "entropy_gate":
        h = entropy_norm_from_peak(float(pi))
        return GateDecision.ACT if h < float(cfg.entropy_star) else GateDecision.DEFER
    aged = int(delta) >= int(cfg.delta_star)
    if cfg.policy == "age_wait":
        return GateDecision.WAIT if aged else GateDecision.ACT
    # peak_and_age: aged + uncertain -> Defer; aged + sharp -> Wait; fresh -> Act
    if not aged:
        return GateDecision.ACT
    if float(pi) < float(cfg.pi_star):
        return GateDecision.DEFER
    return GateDecision.WAIT

def composite_cost(
    mean_regret: float,
    frac_act: float,
    frac_wait: float = 0.0,
    frac_defer: float = 0.0,
    lambda_act: float = 10.0,
    lambda_wait: float = 8.0,
    lambda_defer: float = 2.0,
) -> float:
    """Operator cost: regret + Act budget + Wait/Defer annoyance weights.

    Default: Wait more expensive than Defer (missed care vs mild prompt).
    """
    return (
        float(mean_regret)
        + float(lambda_act) * float(frac_act)
        + float(lambda_wait) * float(frac_wait)
        + float(lambda_defer) * float(frac_defer)
    )

@dataclass
class BeliefAgeTracker:
    """Tracks steps since last successful monitoring update."""

    delta: int = 0
    last_probs: np.ndarray | None = None
    n_updates: int = 0
    n_stale: int = 0

    def on_update(self, probs: np.ndarray) -> None:
        self.delta = 0
        self.last_probs = np.asarray(probs, dtype=float).copy()
        self.n_updates += 1

    def on_stale(self) -> None:
        self.delta += 1
        self.n_stale += 1

    def aged_belief(self, live_probs: np.ndarray, diffuse: bool = True) -> np.ndarray:
        if self.last_probs is None or self.delta <= 0:
            return np.asarray(live_probs, dtype=float).copy()
        if diffuse:
            return diffuse_belief(self.last_probs, self.delta)
        return np.asarray(self.last_probs, dtype=float).copy()
