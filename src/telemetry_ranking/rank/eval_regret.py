"""Evaluate ranking policies under partial observability."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from telemetry_ranking.rank.env_utils import make_env
from telemetry_ranking.rank.rankers import (
    policy_regret,
    rank_abstain_ltr,
    rank_basr,
    rank_belief_ltr,
    rank_map,
    rank_map_ltr,
    rank_oracle,
    rank_residual_ltr,
    rank_rule_based,
    top1_hit,
)

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def _build_rankers(include_ltr: bool = True) -> dict:
    rankers = {
        "B0_rule": lambda env, rng: rank_rule_based(env),
        "B1_MAP": lambda env, rng: rank_map(env),
        "B3_BASR": lambda env, rng: rank_basr(env, n_samples=5, rng=rng),
        "Oracle": lambda env, rng: rank_oracle(env),
    }
    if include_ltr:
        belief_path = ROOT / "models" / "belief_ltr.txt"
        map_path = ROOT / "models" / "map_ltr.txt"
        residual_path = ROOT / "models" / "residual_ltr.txt"
        if belief_path.exists():
            rankers["Belief_LTR"] = lambda env, rng: rank_belief_ltr(env, belief_path)
        if map_path.exists():
            rankers["MAP_LTR"] = lambda env, rng: rank_map_ltr(env, map_path)
        if residual_path.exists():
            rankers["Residual_LTR"] = lambda env, rng: rank_residual_ltr(env, residual_path)
        if belief_path.exists():
            rankers["Abstain_LTR"] = lambda env, rng: rank_abstain_ltr(env)
    return rankers


def run_episodes(
    ranker_name: str,
    sigma: float,
    n_episodes: int = 50,
    seed: int = 0,
    n_version: int = 0,
    rankers: dict | None = None,
) -> dict:
    if rankers is None:
        rankers = _build_rankers(include_ltr=True)
    rng = np.random.default_rng(seed)
    env = make_env(sigma=sigma, seed=seed, n_version=n_version)
    choose = rankers[ranker_name]

    regrets = []
    rewards = []
    hits = []
    lengths = []
    latencies = []

    for ep in range(n_episodes):
        env.reset()
        env.rng0 = np.random.default_rng(seed + 1000 * ep + 1)
        env.rng1 = np.random.default_rng(seed + 1000 * ep + 2)
        done = False
        ep_regret = 0.0
        ep_reward = 0.0
        ep_hits = 0
        steps = 0
        while not done:
            t0 = time.perf_counter()
            action = choose(env, rng)
            latencies.append(time.perf_counter() - t0)
            rgt = policy_regret(env, action)
            hit = top1_hit(env, action)
            _, reward, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            ep_regret += rgt
            ep_reward += float(reward)
            ep_hits += hit
            steps += 1
        regrets.append(ep_regret / max(steps, 1))
        rewards.append(ep_reward)
        hits.append(ep_hits / max(steps, 1))
        lengths.append(steps)

    return {
        "ranker": ranker_name,
        "sigma": sigma,
        "seed": seed,
        "n_version": n_version,
        "n_episodes": n_episodes,
        "mean_step_regret": float(np.mean(regrets)),
        "std_step_regret": float(np.std(regrets)),
        "mean_episode_reward": float(np.mean(rewards)),
        "std_episode_reward": float(np.std(rewards)),
        "mean_top1_accuracy": float(np.mean(hits)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_decision_latency_ms": float(np.mean(latencies) * 1000.0),
    }


def plot_metric(rows: list[dict], rankers: list[str], metric: str, ylabel: str, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in rankers:
        xs = sorted({r["sigma"] for r in rows if r["ranker"] == name})
        ys = []
        for s in xs:
            vals = [r[metric] for r in rows if r["ranker"] == name and r["sigma"] == s]
            ys.append(float(np.mean(vals)))
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Context uncertainty σ")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ltr", action="store_true")
    args = parser.parse_args()

    sigmas = [0.0, 0.4, 1.0, 2.0]
    rankers_map = _build_rankers(include_ltr=not args.no_ltr)
    ranker_names = list(rankers_map.keys())

    rows = []
    for sigma in sigmas:
        for name in ranker_names:
            print(f"Running {name} sigma={sigma} ...")
            row = run_episodes(
                name,
                sigma=sigma,
                n_episodes=args.episodes,
                seed=args.seed,
                rankers=rankers_map,
            )
            rows.append(row)
            print(
                f"  regret={row['mean_step_regret']:.3f} "
                f"reward={row['mean_episode_reward']:.1f} "
                f"top1={row['mean_top1_accuracy']:.3f} "
                f"lat={row['mean_decision_latency_ms']:.2f}ms"
            )

    out_json = RESULTS / "day1_regret_baseline.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"episodes": args.episodes, "rows": rows}, f, indent=2)

    plot_metric(
        rows,
        ranker_names,
        "mean_step_regret",
        "Mean per-step policy regret",
        "Ranking under partial observability",
        RESULTS / "regret_vs_sigma.png",
    )
    plot_metric(
        rows,
        ranker_names,
        "mean_top1_accuracy",
        "Mean top-1 accuracy vs oracle",
        "Top-1 intervention accuracy",
        RESULTS / "top1_vs_sigma.png",
    )

    print(f"\nSaved {out_json}")
    print(f"Saved {RESULTS / 'regret_vs_sigma.png'}")
    print(f"Saved {RESULTS / 'top1_vs_sigma.png'}")


if __name__ == "__main__":
    main()
