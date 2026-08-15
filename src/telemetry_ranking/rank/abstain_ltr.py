"""BALToR-inspired abstention ranking: Belief-LTR when confident, else MAP-Sim.

Does not retrain models. Uses frozen Belief-LTR + existing MAP-Sim.
Preserves RQ/architecture/eval protocol; only adds a ranking decision rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from telemetry_ranking.rank.env_utils import belief_probs, make_env
from telemetry_ranking.rank.ltr_lambdarank import score_actions
from telemetry_ranking.rank.rankers import (
    BELIEF_LTR_PATH,
    _load_ltr,
    policy_regret,
    rank_belief_ltr,
    rank_map,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
DEFAULT_THRESHOLD_PATH = CONFIGS / "abstain_threshold.json"


def belief_entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    p = p / max(float(p.sum()), 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))


def ltr_margin(env, model_path: Path | None = None) -> float:
    """Top1 - top2 score gap from frozen Belief-LTR (higher => more confident)."""
    path = Path(model_path) if model_path is not None else BELIEF_LTR_PATH
    booster = _load_ltr(str(path.resolve()))
    scores = score_actions(env, booster, "belief")
    ordered = np.sort(scores)[::-1]
    if len(ordered) < 2:
        return float(ordered[0]) if len(ordered) else 0.0
    return float(ordered[0] - ordered[1])


def confidence_features(env, model_path: Path | None = None) -> dict[str, float]:
    probs = belief_probs(env)
    ent = belief_entropy(probs)
    # Normalize entropy to [0,1] for binary belief (max ln2)
    ent_norm = ent / max(np.log(2.0), 1e-12)
    margin = ltr_margin(env, model_path=model_path)
    # Combined confidence in [0,1]-ish: low entropy and larger margin => higher
    # Use sigmoid on margin for scale-robustness
    margin_sig = float(1.0 / (1.0 + np.exp(-margin)))
    conf = float((1.0 - min(max(ent_norm, 0.0), 1.0)) * 0.5 + margin_sig * 0.5)
    return {
        "entropy": ent,
        "entropy_norm": float(ent_norm),
        "ltr_margin": margin,
        "confidence": conf,
    }


def rank_abstain_ltr(
    env,
    threshold: float = 0.55,
    model_path: Path | None = None,
) -> int:
    """
    If Belief-LTR confidence >= threshold, use Belief-LTR; else abstain to MAP-Sim.
    threshold -> +inf recovers MAP; threshold -> -inf recovers Belief-LTR.
    """
    feats = confidence_features(env, model_path=model_path)
    if feats["confidence"] >= float(threshold):
        return rank_belief_ltr(env, model_path=model_path)
    return rank_map(env)


def load_threshold(path: Path | None = None) -> float:
    p = path or DEFAULT_THRESHOLD_PATH
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return float(data.get("threshold", 0.55))
    return 0.55


def tune_threshold(
    thresholds: list[float] | None = None,
    sigmas: list[float] | None = None,
    episodes: int = 10,
    seed: int = 0,
) -> dict:
    """Pick τ minimizing mean policy regret across σ on a small validation budget."""
    if thresholds is None:
        thresholds = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    if sigmas is None:
        sigmas = [0.0, 0.4, 1.0, 2.0]

    # Ensure booster is loadable
    if not BELIEF_LTR_PATH.exists():
        raise FileNotFoundError(f"Frozen Belief-LTR model missing: {BELIEF_LTR_PATH}")

    rows = []
    best_tau = thresholds[0]
    best_score = float("inf")

    for tau in thresholds:
        regrets = []
        for sigma in sigmas:
            env = make_env(sigma=float(sigma), seed=seed, n_version=0)
            for ep in range(episodes):
                env.reset()
                env.rng0 = np.random.default_rng(seed + 1000 * ep + 1)
                env.rng1 = np.random.default_rng(seed + 1000 * ep + 2)
                done = False
                ep_reg = 0.0
                steps = 0
                while not done:
                    a = rank_abstain_ltr(env, threshold=float(tau))
                    ep_reg += policy_regret(env, a)
                    _, _, term, trunc, _ = env.step(a)
                    done = bool(term or trunc)
                    steps += 1
                regrets.append(ep_reg / max(steps, 1))
        mean_r = float(np.mean(regrets))
        rows.append({"threshold": float(tau), "mean_step_regret": mean_r})
        print(f"tau={tau:.2f} val_regret={mean_r:.4f}", flush=True)
        if mean_r < best_score:
            best_score = mean_r
            best_tau = float(tau)

    out = {
        "threshold": best_tau,
        "val_mean_step_regret": best_score,
        "grid": rows,
        "sigmas": sigmas,
        "episodes": episodes,
        "seed": seed,
        "rule": "use Belief-LTR if confidence>=tau else MAP-Sim",
        "confidence": "0.5*(1-H_norm) + 0.5*sigmoid(ltr_top1_top2_margin)",
    }
    CONFIGS.mkdir(parents=True, exist_ok=True)
    DEFAULT_THRESHOLD_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Selected tau={best_tau:.2f} -> {DEFAULT_THRESHOLD_PATH}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Tune abstention threshold for Abstain_LTR")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tune_threshold(episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
