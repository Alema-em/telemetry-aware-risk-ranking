"""Habituation / disengagement trajectory ablation (journal-only).

Compares MAP vs CVaR under:
  - baseline dynamics (default H/D)
  - frozen H/D (εh=δh=εd=δd=0, D_threshold=2 so episodes are not cut by D)

Prediction: if CVaR's episode regret win is trajectory-mediated, the MAP−CVaR
gap should shrink under frozen H/D.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telemetry_ranking.journal.cvar_rankers import rank_cvar, rank_map_sim
from telemetry_ranking.rank.env_utils import make_env
from telemetry_ranking.rank.rankers import policy_regret, top1_hit

from telemetry_ranking.paths import run_output_dir

OUT = run_output_dir()

N_SEEDS = 20
EPISODES = 15
ALPHA = 0.3
SIGMAS = [1.0, 2.0]

BASELINE = {}
FROZEN_HD = {
    "δh": 0.0,
    "εh": 0.0,
    "δd": 0.0,
    "εd": 0.0,
    "D_threshold": 2.0,
}


def run_cell(method: str, sigma: float, seed: int, env_kw: dict) -> dict:
    fn = rank_map_sim if method == "MAP" else (lambda e: rank_cvar(e, alpha=ALPHA))
    env = make_env(sigma=float(sigma), seed=int(seed), n_version=0, **env_kw)
    regrets, tops, lengths = [], [], []
    specialized = 0
    steps_total = 0
    for ep in range(EPISODES):
        env.reset()
        env.rng0 = np.random.default_rng(seed + 1000 * ep + 1)
        env.rng1 = np.random.default_rng(seed + 1000 * ep + 2)
        done = False
        ep_r, ep_hit, steps = 0.0, 0, 0
        while not done:
            a = int(fn(env))
            if a in (2, 3):
                specialized += 1
            ep_r += policy_regret(env, a)
            ep_hit += top1_hit(env, a)
            _, _, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            steps += 1
        regrets.append(ep_r / max(steps, 1))
        tops.append(ep_hit / max(steps, 1))
        lengths.append(steps)
        steps_total += steps
    return {
        "method": method,
        "sigma": float(sigma),
        "seed": int(seed),
        "mean_step_regret": float(np.mean(regrets)),
        "mean_top1": float(np.mean(tops)),
        "mean_episode_length": float(np.mean(lengths)),
        "specialized_rate": float(specialized / max(steps_total, 1)),
    }


def summarize(raw: list[dict], condition: str) -> tuple[list[dict], list[dict]]:
    out = []
    for sigma in SIGMAS:
        for method in ("MAP", "CVaR_0.3"):
            vals = [
                r["mean_step_regret"]
                for r in raw
                if r["condition"] == condition
                and r["method"] == method
                and r["sigma"] == sigma
            ]
            if not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            out.append(
                {
                    "condition": condition,
                    "method": method,
                    "sigma": sigma,
                    "n_seeds": len(arr),
                    "mean_step_regret": float(np.mean(arr)),
                    "sem": float(np.std(arr, ddof=1) / np.sqrt(len(arr))),
                }
            )
                   
    deltas = []
    for sigma in SIGMAS:
        map_by = {
            r["seed"]: r["mean_step_regret"]
            for r in raw
            if r["condition"] == condition and r["method"] == "MAP" and r["sigma"] == sigma
        }
        cvar_by = {
            r["seed"]: r["mean_step_regret"]
            for r in raw
            if r["condition"] == condition
            and r["method"] == "CVaR_0.3"
            and r["sigma"] == sigma
        }
        seeds = sorted(set(map_by) & set(cvar_by))
        d = np.asarray([cvar_by[s] - map_by[s] for s in seeds], dtype=float)
        deltas.append(
            {
                "condition": condition,
                "sigma": sigma,
                "mean_delta_cvar_minus_map": float(np.mean(d)),
                "sem_delta": float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0,
                "n_pairs": len(d),
            }
        )
    return out, deltas


def main() -> None:
    conditions = {
        "baseline": BASELINE,
        "frozen_HD": FROZEN_HD,
    }
    raw = []
    total = N_SEEDS * len(SIGMAS) * 2 * len(conditions)
    k = 0
    for cond, kw in conditions.items():
        for seed in range(N_SEEDS):
            for sigma in SIGMAS:
                for method in ("MAP", "CVaR_0.3"):
                    k += 1
                    print(f"[{k}/{total}] {cond} seed={seed} sigma={sigma} {method}", flush=True)
                    row = run_cell(method, sigma, seed, kw)
                    row["condition"] = cond
                    raw.append(row)

    agg_base, delta_base = summarize(raw, "baseline")
    agg_fr, delta_fr = summarize(raw, "frozen_HD")
    payload = {
        "config": {
            "n_seeds": N_SEEDS,
            "episodes": EPISODES,
            "sigmas": SIGMAS,
            "frozen": FROZEN_HD,
        },
        "raw": raw,
        "aggregated": agg_base + agg_fr,
        "deltas": delta_base + delta_fr,
    }
    out = OUT / "hd_ablation.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md = OUT / "HD_ABLATION_FINDINGS.md"
    lines = ["# H/D trajectory ablation", ""]
    lines.append("| Condition | σ | mean Δ (CVaR−MAP) | SEM |")
    lines.append("|-----------|---|-------------------|-----|")
    for d in delta_base + delta_fr:
        lines.append(
            f"| {d['condition']} | {d['sigma']} | {d['mean_delta_cvar_minus_map']:.3f} | {d['sem_delta']:.3f} |"
        )
    lines.append("")
    lines.append(
        "Negative Δ favours CVaR. If |Δ| shrinks under `frozen_HD`, the episode win is "
        "consistent with trajectory mediation via H/D."
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
