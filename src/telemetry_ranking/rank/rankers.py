"""Baseline and learned ranking rules for intervention selection."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from telemetry_ranking.rank.env_utils import belief_probs, utilities_all_actions

ROOT = Path(__file__).resolve().parents[3]
BELIEF_LTR_PATH = ROOT / "models" / "belief_ltr.txt"
MAP_LTR_PATH = ROOT / "models" / "map_ltr.txt"
RESIDUAL_LTR_PATH = ROOT / "models" / "residual_ltr.txt"


def rank_rule_based(env) -> int:
    """B0: always no-intervention."""
    return 0


def rank_map(env) -> int:
    """
    B1: MAP simulation ranking.
    Collapse belief to argmax context, simulate each action once, pick best.
    """
    probs = belief_probs(env)
    c_hat = int(np.argmax(probs))
    utils = utilities_all_actions(env, assumed_c=c_hat)
    return int(np.argmax(utils))


def rank_oracle(env) -> int:
    """Oracle: simulate with true latent context (evaluation only)."""
    utils = utilities_all_actions(env, assumed_c=None)
    return int(np.argmax(utils))


def rank_basr(env, n_samples: int = 5, rng: np.random.Generator | None = None) -> int:
    """
    B3 ablation: Belief-Averaged Simulation Ranking.
    Average one-step utilities over contexts sampled from the belief.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    probs = belief_probs(env)
    n = int(env.action_space.n)
    scores = np.zeros(n, dtype=float)
    for _ in range(n_samples):
        c = int(rng.choice(len(probs), p=probs / max(probs.sum(), 1e-12)))
        scores += utilities_all_actions(env, assumed_c=c)
    scores /= float(n_samples)
    return int(np.argmax(scores))


@lru_cache(maxsize=4)
def _load_ltr(path_str: str):
    from telemetry_ranking.rank.ltr_lambdarank import load_booster

    return load_booster(Path(path_str))


def rank_belief_ltr(env, model_path: Path | None = None) -> int:
    """Belief-aware LambdaRank (primary learned ranker)."""
    from telemetry_ranking.rank.ltr_lambdarank import rank_with_booster

    path = Path(model_path) if model_path is not None else BELIEF_LTR_PATH
    booster = _load_ltr(str(path.resolve()))
    return rank_with_booster(env, booster, "belief")


def rank_map_ltr(env, model_path: Path | None = None) -> int:
    """MAP-feature LambdaRank ablation (no soft belief probabilities)."""
    from telemetry_ranking.rank.ltr_lambdarank import rank_with_booster

    path = Path(model_path) if model_path is not None else MAP_LTR_PATH
    booster = _load_ltr(str(path.resolve()))
    return rank_with_booster(env, booster, "map")


def rank_residual_ltr(env, model_path: Path | None = None) -> int:
    """MAP-sim utility + learned belief residual (simulation-augmented LTR tweak)."""
    from telemetry_ranking.rank.ltr_lambdarank import rank_with_booster

    path = Path(model_path) if model_path is not None else RESIDUAL_LTR_PATH
    booster = _load_ltr(str(path.resolve()))
    return rank_with_booster(env, booster, "residual")


def rank_abstain_ltr(env, threshold: float | None = None, model_path: Path | None = None) -> int:
    """
    BALToR-inspired abstention: Belief-LTR if confident, else fall back to MAP-Sim.
    Frozen Belief-LTR weights are unchanged.
    """
    from telemetry_ranking.rank.abstain_ltr import load_threshold
    from telemetry_ranking.rank.abstain_ltr import rank_abstain_ltr as _rank_abstain

    tau = load_threshold() if threshold is None else float(threshold)
    return _rank_abstain(env, threshold=tau, model_path=model_path)


def policy_regret(env, chosen_action: int) -> float:
    """Gap between oracle-best utility and chosen action utility (true context)."""
    utils = utilities_all_actions(env, assumed_c=None)
    best = float(np.max(utils))
    got = float(utils[int(chosen_action)])
    return best - got


def top1_hit(env, chosen_action: int) -> int:
    utils = utilities_all_actions(env, assumed_c=None)
    return int(int(chosen_action) == int(np.argmax(utils)))
