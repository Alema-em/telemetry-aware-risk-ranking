"""Run the 30-seed risk-neutral comparison."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telemetry_ranking.rank.eval_regret import _build_rankers, run_episodes

from telemetry_ranking.paths import run_output_dir

OUT = run_output_dir()

SIGMAS = [0.0, 0.4, 1.0, 2.0]
N_SEEDS = 30
EPISODES = 15
RANKERS = [
    "B1_MAP",
    "B3_BASR",
    "Belief_LTR",
    "MAP_LTR",
    "Residual_LTR",
    "Abstain_LTR",
    "Oracle",
]


def aggregate_rows(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["ranker"], r["sigma"])].append(r)
    agg = []
    for (ranker, sigma), items in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        regrets = np.asarray([i["mean_step_regret"] for i in items], dtype=float)
        n = len(regrets)
        std = float(np.std(regrets, ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
                                          
        ci = 1.96 * sem
        agg.append(
            {
                "ranker": ranker,
                "sigma": float(sigma),
                "n_seeds": n,
                "mean_step_regret": float(np.mean(regrets)),
                "std_step_regret": std,
                "sem_step_regret": sem,
                "ci95_low": float(np.mean(regrets) - ci),
                "ci95_high": float(np.mean(regrets) + ci),
                "mean_top1_accuracy": float(np.mean([i["mean_top1_accuracy"] for i in items])),
                "mean_episode_reward": float(np.mean([i["mean_episode_reward"] for i in items])),
                "mean_decision_latency_ms": float(
                    np.mean([i["mean_decision_latency_ms"] for i in items])
                ),
            }
        )
    return agg


def belief_ablation(agg: list[dict]) -> list[dict]:
    out = []
    for sigma in SIGMAS:
        belief = next((r for r in agg if r["ranker"] == "Belief_LTR" and r["sigma"] == sigma), None)
        map_ltr = next((r for r in agg if r["ranker"] == "MAP_LTR" and r["sigma"] == sigma), None)
        mapa = next((r for r in agg if r["ranker"] == "B1_MAP" and r["sigma"] == sigma), None)
        if not (belief and map_ltr and mapa):
            continue
        out.append(
            {
                "sigma": sigma,
                "belief_ltr": belief["mean_step_regret"],
                "map_ltr": map_ltr["mean_step_regret"],
                "map_sim": mapa["mean_step_regret"],
                "delta_belief_minus_mapltr": belief["mean_step_regret"] - map_ltr["mean_step_regret"],
                "delta_belief_minus_mapsim": belief["mean_step_regret"] - mapa["mean_step_regret"],
            }
        )
    return out


def plot_regime(agg: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name in RANKERS:
        xs = [r["sigma"] for r in agg if r["ranker"] == name]
        ys = [r["mean_step_regret"] for r in agg if r["ranker"] == name]
        es = [r["sem_step_regret"] for r in agg if r["ranker"] == name]
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=name)
    ax.set_xlabel("Context uncertainty σ")
    ax.set_ylabel("Mean per-step policy regret")
    ax.set_title(f"Risk-neutral comparison ({N_SEEDS} seeds; error bars = SEM)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    rankers_map = _build_rankers(include_ltr=True)
    names = [n for n in RANKERS if n in rankers_map]
    raw_rows: list[dict] = []
    total = N_SEEDS * len(SIGMAS) * len(names)
    k = 0
    for seed in range(N_SEEDS):
        for sigma in SIGMAS:
            for name in names:
                k += 1
                print(f"[{k}/{total}] seed={seed} {name} sigma={sigma}", flush=True)
                row = run_episodes(
                    name,
                    sigma=float(sigma),
                    n_episodes=EPISODES,
                    seed=int(seed),
                    n_version=0,
                    rankers=rankers_map,
                )
                row["seed"] = int(seed)
                raw_rows.append(row)

    agg = aggregate_rows(raw_rows)
    payload = {
        "config": {
            "n_seeds": N_SEEDS,
            "episodes": EPISODES,
            "sigmas": SIGMAS,
            "rankers": names,
            "note": "Thirty-seed paper protocol.",
        },
        "raw_rows": raw_rows,
        "aggregated": agg,
        "belief_ablation": belief_ablation(agg),
    }
    out_json = OUT / "regime_map_30seed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    plot_regime(agg, OUT / "regime_map_30seed.png")
    print(f"Saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
