"""Synthetic specialize-vs-robust ranking environment (paper study).

Structural twin of the geometry found in StepCountJITAI:
  action 0: noop  ~ L
  action 1: robust ~ M under both contexts
  action 2: specialize c0 (H under c0, L under c1)
  action 3: specialize c1 (L under c0, H under c1)

Uncertainty level u in [0,1] controls belief peak concentration
(higher u => more mass near pi=0.5). Not StepCountJITAI.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from telemetry_ranking.journal.cvar_rankers import score_cvar, score_map


@dataclass
class SynthConfig:
    H: float = 50.0
    M: float = 12.5
    L: float = 0.1
                                                          
    noise_std: float = 0.5


def rho_of(cfg: SynthConfig) -> float:
    return float((cfg.H - cfg.M) / max(cfg.H - cfg.L, 1e-12))


def pi_star(cfg: SynthConfig, alpha: float) -> float:
    return float(1.0 - float(alpha) * rho_of(cfg))


def build_U(cfg: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    """2 x 4 utility matrix with specialize-vs-robust structure."""
    n = rng.normal(0.0, cfg.noise_std, size=(2, 4))
    U = np.array(
        [
            [cfg.L, cfg.M, cfg.H, cfg.L],
            [cfg.L, cfg.M, cfg.L, cfg.H],
        ],
        dtype=float,
    )
    return U + n


def sample_belief(u: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample belief peak concentration from uncertainty u in [0, 1].
    u=0 => near-deterministic peak; u=1 => near-uniform.
    """
    u = float(np.clip(u, 0.0, 1.0))
                                         
    kappa = 1.0 + 40.0 * (1.0 - u) ** 2
                                     
                                        
    mean_peak = 0.5 + 0.5 * (1.0 - u)
                                                                      
    a = mean_peak * kappa
    b = (1.0 - mean_peak) * kappa
    z = float(rng.beta(max(a, 1e-3), max(b, 1e-3)))
    pi = 0.5 + 0.5 * z               
                          
    if rng.random() < 0.5:
        return np.array([pi, 1.0 - pi], dtype=float)
    return np.array([1.0 - pi, pi], dtype=float)


def sample_true_context(probs: np.ndarray, rng: np.random.Generator, calibrated: bool = True) -> int:
    """If calibrated, draw true c ~ belief; else draw flip with belief-mismatch stress."""
    p = probs / max(float(probs.sum()), 1e-12)
    if calibrated:
        return int(rng.choice(len(p), p=p))
                                                                 
    map_c = int(np.argmax(p))
    if rng.random() < 0.2:
        return 1 - map_c
    return map_c


@dataclass
class SynthStep:
    probs: np.ndarray
    U: np.ndarray
    true_c: int
    pi: float
    u: float


def sample_step(
    u: float,
    cfg: SynthConfig,
    rng: np.random.Generator,
    calibrated: bool = True,
) -> SynthStep:
    probs = sample_belief(u, rng)
    U = build_U(cfg, rng)
    true_c = sample_true_context(probs, rng, calibrated=calibrated)
    return SynthStep(probs=probs, U=U, true_c=true_c, pi=float(probs.max()), u=float(u))


def oracle_action(U: np.ndarray, true_c: int) -> int:
    return int(np.argmax(U[true_c]))


def regret(U: np.ndarray, true_c: int, action: int) -> float:
    best = float(np.max(U[true_c]))
    got = float(U[true_c, int(action)])
    return best - got


def choose_map(probs: np.ndarray, U: np.ndarray) -> int:
    return int(np.argmax(score_map(probs, U)))


def choose_cvar(probs: np.ndarray, U: np.ndarray, alpha: float = 0.3) -> int:
    return int(np.argmax(score_cvar(probs, U, alpha=alpha)))


def choose_hybrid(
    probs: np.ndarray,
    U: np.ndarray,
    alpha: float = 0.3,
    pi_star_thr: float | None = None,
    cfg: SynthConfig | None = None,
) -> int:
    pi = float(probs.max() / max(float(probs.sum()), 1e-12))
    if pi_star_thr is None:
        assert cfg is not None
        thr = pi_star(cfg, alpha)
    else:
        thr = float(pi_star_thr)
    if pi >= thr:
        return choose_map(probs, U)
    return choose_cvar(probs, U, alpha=alpha)
