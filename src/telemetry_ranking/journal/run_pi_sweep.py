"""Hybrid gate π★ sweep: episode reward and per-step regret vs threshold.

Default operational gate is π★=0.7 (validation sweep); theory point estimate
from (H,M,L) at α=0.3 is ≈0.775 — both reported.

Writes results to the configured run directory.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telemetry_ranking.journal.cvar_rankers import rank_cvar, rank_hybrid_pi_gate, rank_map_sim
from telemetry_ranking.paths import figure_output_dir, run_output_dir
from telemetry_ranking.rank.env_utils import make_env
from telemetry_ranking.rank.rankers import policy_regret

OUT = run_output_dir()
FIG = figure_output_dir()

SIGMAS = [0.0, 0.4, 1.0, 2.0]
PI_GRID = [0.50, 0.60, 0.70, 0.775, 0.85, 0.90]
N_SEEDS = 12
EPISODES = 12
ALPHA = 0.3
PI_OPERATIONAL = 0.7
PI_THEORY = 0.775


def _run(args: tuple) -> dict:
    kind, sigma, pi_star, seed = args
    env = make_env(sigma=float(sigma), seed=int(seed), n_version=0)
    if kind == "MAP":
        choose = rank_map_sim
    elif kind == "CVaR":
        choose = lambda e: rank_cvar(e, alpha=ALPHA)
    else:
        choose = lambda e, thr=float(pi_star): rank_hybrid_pi_gate(
            e, alpha=ALPHA, pi_star=thr, adaptive_pi_star=False
        )

    step_regs = []
    ep_rews = []
    gate_cvar = 0
    n_steps = 0
    for ep in range(EPISODES):
        env.reset()
        env.rng0 = np.random.default_rng(seed + 1000 * ep + 1)
        env.rng1 = np.random.default_rng(seed + 1000 * ep + 2)
        done = False
        ep_reg = 0.0
        ep_rew = 0.0
        steps = 0
        while not done:
            from telemetry_ranking.rank.env_utils import belief_probs

            p = belief_probs(env)
            pi = float(np.max(p) / max(float(np.sum(p)), 1e-12))
            if kind == "Hybrid" and pi < float(pi_star):
                gate_cvar += 1
            a = int(choose(env))
            ep_reg += policy_regret(env, a)
            _o, rew, term, trunc, _i = env.step(a)
            ep_rew += float(rew)
            done = bool(term or trunc)
            steps += 1
            n_steps += 1
        step_regs.append(ep_reg / max(steps, 1))
        ep_rews.append(ep_rew)
    return {
        "kind": kind,
        "sigma": float(sigma),
        "pi_star": None if pi_star is None else float(pi_star),
        "seed": int(seed),
        "mean_step_regret": float(np.mean(step_regs)),
        "mean_episode_reward": float(np.mean(ep_rews)),
        "gate_cvar_frac": float(gate_cvar / max(n_steps, 1)) if kind == "Hybrid" else None,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        buckets[(r["kind"], r["sigma"], r["pi_star"])].append(r)
    out = []
    for (kind, sigma, pi_star), items in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1] or -1, x[0][2] or -1)):
        mr = np.asarray([i["mean_step_regret"] for i in items], dtype=float)
        rw = np.asarray([i["mean_episode_reward"] for i in items], dtype=float)
        row = {
            "kind": kind,
            "sigma": sigma,
            "pi_star": pi_star,
            "n_seeds": len(items),
            "mean_step_regret": float(np.mean(mr)),
            "sem_step_regret": float(np.std(mr, ddof=1) / np.sqrt(len(mr))) if len(mr) > 1 else 0.0,
            "mean_episode_reward": float(np.mean(rw)),
            "sem_episode_reward": float(np.std(rw, ddof=1) / np.sqrt(len(rw))) if len(rw) > 1 else 0.0,
        }
        if kind == "Hybrid":
            g = np.asarray([i["gate_cvar_frac"] for i in items], dtype=float)
            row["gate_cvar_frac"] = float(np.mean(g))
        out.append(row)
    return out


def plot(agg: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
                                                      
    for sigma in (1.0, 2.0):
        hyb = [r for r in agg if r["kind"] == "Hybrid" and r["sigma"] == sigma]
        hyb = sorted(hyb, key=lambda r: r["pi_star"])
        xs = [r["pi_star"] for r in hyb]
        map_r = next(r for r in agg if r["kind"] == "MAP" and r["sigma"] == sigma)
        cvar_r = next(r for r in agg if r["kind"] == "CVaR" and r["sigma"] == sigma)

        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), dpi=180)
        ax = axes[0]
        ax.errorbar(xs, [r["mean_step_regret"] for r in hyb], yerr=[r["sem_step_regret"] for r in hyb],
                    marker="o", color="#1F4E79", lw=1.5, capsize=3, label="Hybrid")
        ax.axhline(map_r["mean_step_regret"], color="#5A5A5A", ls="--", lw=1.2, label="MAP")
        ax.axhline(cvar_r["mean_step_regret"], color="#C45C26", ls="--", lw=1.2, label=r"CVaR$_{0.3}$")
        ax.axvline(PI_OPERATIONAL, color="#2F6F4E", ls=":", lw=1.3, label=r"operational $\pi^\star{=}0.7$")
        ax.axvline(PI_THEORY, color="#8B4513", ls=":", lw=1.0, label=r"theory $\approx 0.775$")
        ax.set_xlabel(r"Hybrid gate $\pi^\star$")
        ax.set_ylabel("Mean per-step policy regret")
        ax.set_title(rf"Regret vs $\pi^\star$ ($\sigma={sigma:g}$)")
        ax.legend(frameon=False, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = axes[1]
        ax.errorbar(xs, [r["mean_episode_reward"] for r in hyb], yerr=[r["sem_episode_reward"] for r in hyb],
                    marker="o", color="#1F4E79", lw=1.5, capsize=3, label="Hybrid")
        ax.axhline(map_r["mean_episode_reward"], color="#5A5A5A", ls="--", lw=1.2, label="MAP")
        ax.axhline(cvar_r["mean_episode_reward"], color="#C45C26", ls="--", lw=1.2, label=r"CVaR$_{0.3}$")
        ax.axvline(PI_OPERATIONAL, color="#2F6F4E", ls=":", lw=1.3, label=r"operational $\pi^\star{=}0.7$")
        ax.axvline(PI_THEORY, color="#8B4513", ls=":", lw=1.0, label=r"theory $\approx 0.775$")
        ax.set_xlabel(r"Hybrid gate $\pi^\star$")
        ax.set_ylabel("Mean episode reward")
        ax.set_title(rf"Reward vs $\pi^\star$ ($\sigma={sigma:g}$)")
        ax.legend(frameon=False, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        stem = f"pi_sweep_sigma_{sigma:g}".replace(".", "p")
        fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
        fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
        plt.close(fig)

                                                          
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), dpi=180)
    for col, sigma in enumerate((1.0, 2.0)):
        hyb = sorted([r for r in agg if r["kind"] == "Hybrid" and r["sigma"] == sigma], key=lambda r: r["pi_star"])
        xs = [r["pi_star"] for r in hyb]
        map_r = next(r for r in agg if r["kind"] == "MAP" and r["sigma"] == sigma)
        cvar_r = next(r for r in agg if r["kind"] == "CVaR" and r["sigma"] == sigma)
        for row_i, (metric, ylab) in enumerate(
            (("mean_step_regret", "Mean per-step policy regret"), ("mean_episode_reward", "Mean episode reward"))
        ):
            ax = axes[row_i, col]
            sem_key = "sem_step_regret" if "regret" in metric else "sem_episode_reward"
            ax.errorbar(xs, [r[metric] for r in hyb], yerr=[r[sem_key] for r in hyb],
                        marker="o", color="#1F4E79", lw=1.5, capsize=3, label="Hybrid")
            ax.axhline(map_r[metric], color="#5A5A5A", ls="--", lw=1.1, label="MAP")
            ax.axhline(cvar_r[metric], color="#C45C26", ls="--", lw=1.1, label=r"CVaR$_{0.3}$")
            ax.axvline(PI_OPERATIONAL, color="#2F6F4E", ls=":", lw=1.2)
            ax.axvline(PI_THEORY, color="#8B4513", ls=":", lw=1.0)
            ax.set_xlabel(r"Hybrid gate $\pi^\star$")
            ax.set_ylabel(ylab)
            ax.set_title(rf"$\sigma={sigma:g}$")
            if row_i == 0 and col == 0:
                ax.legend(frameon=False, fontsize=7.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "pi_sweep_regret_reward.pdf", bbox_inches="tight")
    fig.savefig(FIG / "pi_sweep_regret_reward.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    jobs = []
    for sigma in SIGMAS:
        for seed in range(N_SEEDS):
            jobs.append(("MAP", sigma, None, seed))
            jobs.append(("CVaR", sigma, None, seed))
            for thr in PI_GRID:
                jobs.append(("Hybrid", sigma, thr, seed))

    print(f"Running {len(jobs)} seed-cells for pi* sweep...", flush=True)
    rows: list[dict] = []
    import os

    with ProcessPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 4))) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 30 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    agg = aggregate(rows)
    payload = {
        "config": {
            "sigmas": SIGMAS,
            "pi_grid": PI_GRID,
            "n_seeds": N_SEEDS,
            "episodes": EPISODES,
            "alpha": ALPHA,
            "pi_operational": PI_OPERATIONAL,
            "pi_theory": PI_THEORY,
            "note": "Operational Hybrid uses pi*=0.7; theory from H,M,L gives ~0.775 at alpha=0.3.",
        },
        "seed_rows": rows,
        "aggregated": agg,
    }
    out = OUT / "pi_star_sweep.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot(agg)

    lines = [
        "# Hybrid π★ sweep",
        "",
        f"Operational gate **π★=0.7**; theory point estimate **≈0.775** (α={ALPHA}).",
        "",
        "| σ | π★ | regret | reward | gate CVaR frac |",
        "|---|-----|--------|--------|----------------|",
    ]
    for r in agg:
        if r["kind"] != "Hybrid":
            continue
        lines.append(
            f"| {r['sigma']:g} | {r['pi_star']:.3f} | {r['mean_step_regret']:.2f}±{r['sem_step_regret']:.2f} | "
            f"{r['mean_episode_reward']:.1f}±{r['sem_episode_reward']:.1f} | {r.get('gate_cvar_frac', float('nan')):.2f} |"
        )
    (OUT / "PI_STAR_SWEEP_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
