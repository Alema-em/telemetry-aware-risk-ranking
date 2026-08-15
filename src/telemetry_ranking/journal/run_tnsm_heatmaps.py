"""TNSM telemetry heatmaps: α × p_loss under independent and bursty loss.

Supports multiple σ values. Metrics: mean per-step policy regret, mean episode
reward, SLA-proxy violation = P(specialised ∧ π < 0.7).

Writes results to the configured run directory.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telemetry_ranking.journal.cvar_rankers import rank_cvar, rank_map_sim
from telemetry_ranking.journal.telemetry_loss import TelemetryChannel, rank_with_telemetry
from telemetry_ranking.paths import figure_output_dir, run_output_dir
from telemetry_ranking.rank.env_utils import belief_probs, make_env
from telemetry_ranking.rank.rankers import policy_regret

OUT = run_output_dir()
FIG = figure_output_dir()

SIGMAS = [1.0, 2.0]
ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.75]
P_LOSS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
CHANNELS = ("independent", "bursty")
N_SEEDS = 10
EPISODES = 10
PI_SLA = 0.7
SPEC_ACTIONS = (2, 3)


def _run_cell(args: tuple) -> dict:
    sigma, channel_kind, p_loss, alpha, method, seed = args
    env = make_env(sigma=float(sigma), seed=int(seed), n_version=0)
    if method == "MAP":
        choose = rank_map_sim
    else:
        choose = lambda e, a=float(alpha): rank_cvar(e, alpha=a)

    step_regrets = []
    ep_rewards = []
    n_steps = 0
    n_sla_viol = 0
    emp_loss = []

    for ep in range(EPISODES):
        env.reset()
        env.rng0 = np.random.default_rng(seed + 1000 * ep + 1)
        env.rng1 = np.random.default_rng(seed + 1000 * ep + 2)
        ch = TelemetryChannel(channel_kind, p_loss, np.random.default_rng(seed * 17 + ep * 31 + int(10 * sigma)))
        ch.reset()
        cached = None
        done = False
        ep_reg = 0.0
        ep_rew = 0.0
        steps = 0
        while not done:
            a, cached, _dropped = rank_with_telemetry(env, choose, ch, cached)
            p = np.asarray(cached if cached is not None else belief_probs(env), dtype=float)
            pi = float(p.max() / max(float(p.sum()), 1e-12))
            if int(a) in SPEC_ACTIONS and pi < PI_SLA:
                n_sla_viol += 1
            ep_reg += policy_regret(env, a)
            _obs, rew, term, trunc, _info = env.step(a)
            ep_rew += float(rew)
            done = bool(term or trunc)
            steps += 1
            n_steps += 1
        step_regrets.append(ep_reg / max(steps, 1))
        ep_rewards.append(ep_rew)
        emp_loss.append(ch.empirical_loss)

    return {
        "sigma": float(sigma),
        "channel": channel_kind,
        "p_loss": float(p_loss),
        "alpha": float(alpha),
        "method": method,
        "seed": int(seed),
        "mean_step_regret": float(np.mean(step_regrets)),
        "mean_episode_reward": float(np.mean(ep_rewards)),
        "sla_violation_rate": float(n_sla_viol / max(n_steps, 1)),
        "empirical_loss": float(np.mean(emp_loss)),
        "n_episodes": EPISODES,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["sigma"], r["channel"], r["p_loss"], r["alpha"], r["method"])
        buckets[key].append(r)
    out = []
    for (sig, ch, p, a, m), items in sorted(buckets.items()):
        def _mean_sem(key: str) -> tuple[float, float]:
            arr = np.asarray([i[key] for i in items], dtype=float)
            mean = float(np.mean(arr))
            sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            return mean, sem

        mr, sr = _mean_sem("mean_step_regret")
        rw, sw = _mean_sem("mean_episode_reward")
        sv, ss = _mean_sem("sla_violation_rate")
        el, _ = _mean_sem("empirical_loss")
        out.append(
            {
                "sigma": sig,
                "channel": ch,
                "p_loss": p,
                "alpha": a,
                "method": m,
                "n_seeds": len(items),
                "mean_step_regret": mr,
                "sem_step_regret": sr,
                "mean_episode_reward": rw,
                "sem_episode_reward": sw,
                "sla_violation_rate": sv,
                "sem_sla_violation": ss,
                "empirical_loss": el,
            }
        )
    return out


def _grid(agg: list[dict], sigma: float, channel: str, method: str, metric: str):
    Z = np.full((len(ALPHAS), len(P_LOSS)), np.nan)
    by = {
        (r["alpha"], r["p_loss"]): r
        for r in agg
        if abs(r["sigma"] - sigma) < 1e-9 and r["channel"] == channel and r["method"] == method
    }
    for i, a in enumerate(ALPHAS):
        for j, p in enumerate(P_LOSS):
            row = by.get((a, p))
            if row is not None:
                Z[i, j] = row[metric]
    return Z


def plot_heatmaps(agg: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    metrics = [
        ("mean_step_regret", "Mean per-step policy regret", "regret"),
        ("mean_episode_reward", "Mean episode reward", "reward"),
        ("sla_violation_rate", r"SLA-proxy violation $P(a_{\mathrm{spec}}\wedge\pi<0.7)$", "sla"),
    ]
    for sigma in SIGMAS:
        for channel in CHANNELS:
            fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), dpi=180)
            for ax, (metric, title, tag) in zip(axes, metrics):
                Z = _grid(agg, sigma, channel, "CVaR", metric)
                im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis" if tag != "sla" else "YlOrRd")
                ax.set_xticks(range(len(P_LOSS)))
                ax.set_xticklabels([f"{p:g}" for p in P_LOSS])
                ax.set_yticks(range(len(ALPHAS)))
                ax.set_yticklabels([f"{a:g}" for a in ALPHAS])
                ax.set_xlabel(r"$p_{\mathrm{loss}}$")
                ax.set_ylabel(r"CVaR $\alpha$")
                ax.set_title(title, fontsize=9)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
                for i in range(Z.shape[0]):
                    for j in range(Z.shape[1]):
                        v = Z[i, j]
                        if np.isnan(v):
                            continue
                        txt = f"{v:.0f}" if tag == "reward" else f"{v:.2f}"
                        ax.text(
                            j,
                            i,
                            txt,
                            ha="center",
                            va="center",
                            fontsize=6.5,
                            color="white" if tag != "sla" or v > 0.35 else "black",
                        )
            fig.suptitle(
                f"{'Independent' if channel == 'independent' else 'Bursty (Gilbert–Elliott)'} "
                rf"telemetry loss ($\sigma={sigma:g}$)",
                fontsize=11,
                y=1.02,
            )
            fig.tight_layout()
            stem = f"heatmap_{channel}_alpha_ploss_sigma{sigma:g}".replace(".", "p")
            # also keep legacy names for σ=1
            fig.savefig(FIG / f"{stem}.png", bbox_inches="tight")
            fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
            if abs(sigma - 1.0) < 1e-9:
                legacy = f"heatmap_{channel}_alpha_ploss"
                fig.savefig(FIG / f"{legacy}.png", bbox_inches="tight")
                fig.savefig(FIG / f"{legacy}.pdf", bbox_inches="tight")
            plt.close(fig)

    # Combined delta plot: σ=1 and σ=2, both channels
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), dpi=180)
    for ax, channel in zip(axes, CHANNELS):
        for sigma, style, color in ((1.0, "-o", "#1F4E79"), (2.0, "-s", "#C45C26")):
            deltas = []
            for p in P_LOSS:
                c = next(
                    r
                    for r in agg
                    if abs(r["sigma"] - sigma) < 1e-9
                    and r["channel"] == channel
                    and r["method"] == "CVaR"
                    and abs(r["alpha"] - 0.3) < 1e-9
                    and r["p_loss"] == p
                )
                m = next(
                    r
                    for r in agg
                    if abs(r["sigma"] - sigma) < 1e-9
                    and r["channel"] == channel
                    and r["method"] == "MAP"
                    and abs(r["alpha"] - 0.3) < 1e-9
                    and r["p_loss"] == p
                )
                deltas.append(c["mean_step_regret"] - m["mean_step_regret"])
            ax.plot(P_LOSS, deltas, style, color=color, lw=1.5, label=rf"$\sigma={sigma:g}$")
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_xlabel(r"$p_{\mathrm{loss}}$")
        ax.set_ylabel(r"$\Delta$ regret (CVaR$_{0.3}-$MAP)")
        ax.set_title("Independent" if channel == "independent" else "Bursty", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "delta_regret_vs_ploss.pdf", bbox_inches="tight")
    fig.savefig(FIG / "delta_regret_vs_ploss.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    # Optionally reuse existing σ=1 seed rows to save time
    prev_path = OUT / "tnsm_telemetry_heatmaps.json"
    reused: list[dict] = []
    run_sigmas = list(SIGMAS)
    if prev_path.exists():
        prev = json.loads(prev_path.read_text(encoding="utf-8"))
        old_rows = prev.get("seed_rows", [])
        # Only reuse if they already carry a sigma field or were the σ=1-only run
        for r in old_rows:
            if "sigma" not in r:
                r = dict(r)
                r["sigma"] = float(prev.get("config", {}).get("sigma", 1.0))
            if abs(float(r["sigma"]) - 1.0) < 1e-9:
                reused.append(r)
        if reused:
            run_sigmas = [s for s in SIGMAS if abs(s - 1.0) > 1e-9]
            print(f"Reusing {len(reused)} sigma=1 seed-rows; running sigma={run_sigmas}", flush=True)

    jobs = []
    for sigma in run_sigmas:
        for ch in CHANNELS:
            for p in P_LOSS:
                for seed in range(N_SEEDS):
                    jobs.append((sigma, ch, p, 0.3, "MAP", seed))
                    for a in ALPHAS:
                        jobs.append((sigma, ch, p, a, "CVaR", seed))

    rows: list[dict] = list(reused)
    if jobs:
        print(f"Running {len(jobs)} new seed-cells...", flush=True)
        with ProcessPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 4))) as ex:
            futs = {ex.submit(_run_cell, j): j for j in jobs}
            done = 0
            for fut in as_completed(futs):
                rows.append(fut.result())
                done += 1
                if done % 40 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)}", flush=True)

    # Replicate MAP across α
    map_rows = [r for r in rows if r["method"] == "MAP"]
    have = {(r["sigma"], r["channel"], r["p_loss"], r["alpha"], r["seed"]) for r in rows if r["method"] == "MAP"}
    extra = []
    for r in map_rows:
        for a in ALPHAS:
            key = (r["sigma"], r["channel"], r["p_loss"], float(a), r["seed"])
            if key in have:
                continue
            q = dict(r)
            q["alpha"] = float(a)
            extra.append(q)
            have.add(key)
    rows.extend(extra)
    agg = aggregate(rows)

    payload = {
        "config": {
            "sigmas": SIGMAS,
            "alphas": ALPHAS,
            "p_loss": P_LOSS,
            "channels": list(CHANNELS),
            "n_seeds": N_SEEDS,
            "episodes": EPISODES,
            "pi_sla": PI_SLA,
            "sla_definition": "P(a in {2,3} and belief_peak < 0.7) under belief seen by ranker",
            "independent": "Bernoulli(p_loss)",
            "bursty": "Gilbert-Elliott Good/Bad; lossless Good, always-loss Bad; mean_burst_len=3; E[loss]=p_loss",
        },
        "seed_rows": rows,
        "aggregated": agg,
    }
    out = OUT / "tnsm_telemetry_heatmaps.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_heatmaps(agg)

    lines = [
        "# TNSM telemetry heatmaps",
        "",
        f"σ∈{SIGMAS}, seeds={N_SEEDS}, episodes={EPISODES}.",
        "SLA proxy: P(specialised ∧ π<0.7).",
        "",
        "## CVaR₀.₃ vs MAP Δregret by p_loss",
        "",
        "| σ | channel | p_loss | MAP | CVaR | Δ | SLA(MAP) | SLA(CVaR) |",
        "|---|---------|--------|-----|------|---|----------|-----------|",
    ]
    for sigma in SIGMAS:
        for ch in CHANNELS:
            for p in P_LOSS:
                c = next(
                    r
                    for r in agg
                    if abs(r["sigma"] - sigma) < 1e-9
                    and r["channel"] == ch
                    and r["method"] == "CVaR"
                    and abs(r["alpha"] - 0.3) < 1e-9
                    and r["p_loss"] == p
                )
                m = next(
                    r
                    for r in agg
                    if abs(r["sigma"] - sigma) < 1e-9
                    and r["channel"] == ch
                    and r["method"] == "MAP"
                    and abs(r["alpha"] - 0.3) < 1e-9
                    and r["p_loss"] == p
                )
                lines.append(
                    f"| {sigma:g} | {ch} | {p:.2f} | {m['mean_step_regret']:.2f} | {c['mean_step_regret']:.2f} | "
                    f"{c['mean_step_regret']-m['mean_step_regret']:.2f} | "
                    f"{m['sla_violation_rate']:.3f} | {c['sla_violation_rate']:.3f} |"
                )
    (OUT / "TNSM_TELEMETRY_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
