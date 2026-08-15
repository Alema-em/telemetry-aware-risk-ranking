"""Risk-sensitive ranking over belief-simulated utilities (paper study).

Scores belief-conditioned utilities without modifying the environment.
"""
from __future__ import annotations

import numpy as np

from telemetry_ranking.rank.env_utils import belief_probs, utilities_all_actions


def utilities_by_context(env) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      probs: shape (C,)
      U: shape (C, A) where U[c, a] = one-step utility under assumed context c
    """
    probs = belief_probs(env)
    C = len(probs)
    n = int(env.action_space.n)
    U = np.zeros((C, n), dtype=float)
    for c in range(C):
        U[c] = utilities_all_actions(env, assumed_c=c)
    return probs, U


def score_mean(probs: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Belief-averaged utility (BASR-style exact expectation over discrete contexts)."""
    p = probs / max(float(probs.sum()), 1e-12)
    return p @ U


def score_map(probs: np.ndarray, U: np.ndarray) -> np.ndarray:
    c_hat = int(np.argmax(probs))
    return U[c_hat]


def score_worst(probs: np.ndarray, U: np.ndarray, mass_floor: float = 1e-8) -> np.ndarray:
    """Min utility over contexts with non-negligible belief mass."""
    mask = probs >= mass_floor
    if not np.any(mask):
        mask = np.ones_like(probs, dtype=bool)
    return U[mask].min(axis=0)


def score_cvar(probs: np.ndarray, U: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Discrete CVaR over contexts for each action.

    Interprets belief probs as a distribution over context-conditional utilities.
    For each action a, CVaR_alpha is the expected utility in the worst alpha-mass
    of the belief (lower tail). Higher CVaR score is better (less bad tail).

    alpha in (0, 1]: smaller alpha = more pessimistic.
    """
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    p = probs / max(float(probs.sum()), 1e-12)
    C, A = U.shape
    scores = np.zeros(A, dtype=float)

    for a in range(A):
        utils = U[:, a]
        # Sort contexts from worst to best utility for this action
        order = np.argsort(utils)  # ascending = worst first
        u_sorted = utils[order]
        p_sorted = p[order]
        # Accumulate worst alpha probability mass
        remaining = alpha
        cvar_mass = 0.0
        cvar_sum = 0.0
        for u_i, p_i in zip(u_sorted, p_sorted):
            if remaining <= 0:
                break
            take = min(p_i, remaining)
            cvar_sum += take * u_i
            cvar_mass += take
            remaining -= take
        scores[a] = cvar_sum / max(cvar_mass, 1e-12)
    return scores


def rank_from_scores(scores: np.ndarray) -> int:
    return int(np.argmax(scores))


def rank_map_sim(env) -> int:
    probs, U = utilities_by_context(env)
    return rank_from_scores(score_map(probs, U))


def rank_basr_exact(env) -> int:
    probs, U = utilities_by_context(env)
    return rank_from_scores(score_mean(probs, U))


def rank_worst(env) -> int:
    probs, U = utilities_by_context(env)
    return rank_from_scores(score_worst(probs, U))


def rank_cvar(env, alpha: float = 0.3) -> int:
    probs, U = utilities_by_context(env)
    return rank_from_scores(score_cvar(probs, U, alpha=alpha))


def _geometry_HML(U: np.ndarray) -> tuple[float, float, float]:
    """Specialize-vs-robust geometry from 2 x A utilities."""
    a_spec0 = int(np.argmax(U[0]))
    a_spec1 = int(np.argmax(U[1]))
    a_rob = int(np.argmax(U.min(axis=0)))
    H = 0.5 * (float(U[0, a_spec0]) + float(U[1, a_spec1]))
    L = 0.5 * (float(U[1, a_spec0]) + float(U[0, a_spec1]))
    M = float(U[:, a_rob].mean())
    return H, M, L


def pi_star_from_geometry(U: np.ndarray, alpha: float) -> float:
    """π★(α) = 1 - α (H-M)/(H-L)."""
    H, M, L = _geometry_HML(U)
    rho = (H - M) / max(H - L, 1e-12)
    return float(1.0 - float(alpha) * rho)


def rank_hybrid_pi_gate(
    env,
    alpha: float = 0.3,
    pi_star: float | None = None,
    adaptive_pi_star: bool = False,
) -> int:
    """
    Theory-gated hybrid:
      if belief peak π >= π★: MAP-Sim
      else: CVaR_α

    pi_star=None and adaptive_pi_star=False -> operational default 0.7
    (validation sweep; theory from (H,M,L) at α=0.3 is ≈0.775).
    adaptive_pi_star=True -> compute π★ from current U each step.
    """
    probs, U = utilities_by_context(env)
    p = probs / max(float(probs.sum()), 1e-12)
    pi = float(p.max())
    if adaptive_pi_star:
        thr = pi_star_from_geometry(U, alpha=alpha)
    elif pi_star is None:
        thr = 0.7  # operational Hybrid gate (held-out sweep; theory ≈0.775)
    else:
        thr = float(pi_star)
    if pi >= thr:
        return rank_from_scores(score_map(probs, U))
    return rank_from_scores(score_cvar(probs, U, alpha=alpha))


def decision_profile(env, alpha: float = 0.3) -> dict:
    """Diagnostics: when CVaR diverges from MAP."""
    probs, U = utilities_by_context(env)
    a_map = rank_from_scores(score_map(probs, U))
    a_mean = rank_from_scores(score_mean(probs, U))
    a_cvar = rank_from_scores(score_cvar(probs, U, alpha=alpha))
    a_worst = rank_from_scores(score_worst(probs, U))
    return {
        "probs": probs.tolist(),
        "U": U.tolist(),
        "a_map": a_map,
        "a_mean": a_mean,
        "a_cvar": a_cvar,
        "a_worst": a_worst,
        "cvar_neq_map": int(a_cvar != a_map),
        "mean_neq_map": int(a_mean != a_map),
        "utility_spread": float(U.max(axis=0).max() - U.min(axis=0).min()),
        "per_action_util_range": (U.max(axis=0) - U.min(axis=0)).tolist(),
    }
