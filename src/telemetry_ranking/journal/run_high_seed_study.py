"""High-seed journal study: StepCountJITAI + synthetic structural twin.

Increases statistical power (default 30 seeds) and validates that the
specialize-vs-robust crossover is structural, not StepCount-specific.

Writes results to the configured run directory.
"""
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

from telemetry_ranking.journal.cvar_rankers import rank_cvar, rank_hybrid_pi_gate, rank_map_sim
from telemetry_ranking.journal.synthetic_env import (
    SynthConfig,
    choose_cvar,
    choose_hybrid,
    choose_map,
    pi_star,
    regret,
    rho_of,
    sample_step,
)
from telemetry_ranking.rank.env_utils import make_env
from telemetry_ranking.rank.rankers import policy_regret, top1_hit

from telemetry_ranking.paths import run_output_dir

OUT = run_output_dir()

N_SEEDS = 30
EPISODES = 15
ALPHA = 0.3
SIGMAS = [0.0, 0.4, 1.0, 2.0]
# synthetic uncertainty grid (analogous to sigma)
US = [0.0, 0.25, 0.5, 0.75, 1.0]
STEPS_PER_SEED = 200  # i.i.d. synthetic steps per seed per u


def _agg(raw: list[dict], key_method: str = "method", key_x: str = "sigma") -> list[dict]:
    buckets: dict[tuple, list] = defaultdict(list)
    for r in raw:
        buckets[(r[key_method], r[key_x])].append(r)
    # map baselines
    map_ref = {}
    for (m, x), items in buckets.items():
        if m == "MAP":
            map_ref[x] = float(np.mean([i["mean_step_regret"] for i in items]))
    out = []
    for (m, x), items in sorted(buckets.items(), key=lambda t: (t[0][0], t[0][1])):
        regs = [i["mean_step_regret"] for i in items]
        out.append(
            {
                key_method: m,
                key_x: x,
                "n_seeds": len(items),
                "mean_step_regret": float(np.mean(regs)),
                "std_step_regret": float(np.std(regs, ddof=1) if len(regs) > 1 else 0.0),
                "sem_step_regret": float(
                    (np.std(regs, ddof=1) / np.sqrt(len(regs))) if len(regs) > 1 else 0.0
                ),
                "delta_vs_map": float(np.mean(regs) - map_ref.get(x, np.mean(regs))),
            }
        )
    return out


def run_stepcount_high_seed() -> dict:
    print("=== StepCountJITAI high-seed regime ===", flush=True)
    methods = {
        "MAP": rank_map_sim,
        "CVaR_0.3": lambda env: rank_cvar(env, alpha=ALPHA),
        "Hybrid_0.7": lambda env: rank_hybrid_pi_gate(
            env, alpha=ALPHA, pi_star=0.7, adaptive_pi_star=False
        ),
    }
    raw = []
    for seed in range(N_SEEDS):
        for sigma in SIGMAS:
            for name, fn in methods.items():
                print(f"[SC] seed={seed}/{N_SEEDS-1} sigma={sigma} {name}", flush=True)
                env = make_env(sigma=float(sigma), seed=int(seed), n_version=0)
                regrets, tops = [], []
                for ep in range(EPISODES):
                    env.reset()
                    env.rng0 = np.random.default_rng(seed + 1000 * ep + 1)
                    env.rng1 = np.random.default_rng(seed + 1000 * ep + 2)
                    done = False
                    ep_r, ep_hit, steps = 0.0, 0, 0
                    while not done:
                        a = int(fn(env))
                        ep_r += policy_regret(env, a)
                        ep_hit += top1_hit(env, a)
                        _, _, term, trunc, _ = env.step(a)
                        done = bool(term or trunc)
                        steps += 1
                    regrets.append(ep_r / max(steps, 1))
                    tops.append(ep_hit / max(steps, 1))
                raw.append(
                    {
                        "method": name,
                        "sigma": float(sigma),
                        "seed": int(seed),
                        "mean_step_regret": float(np.mean(regrets)),
                        "std_within": float(np.std(regrets)),
                        "mean_top1": float(np.mean(tops)),
                    }
                )
    agg = _agg(raw, "method", "sigma")
    return {"config": {"n_seeds": N_SEEDS, "episodes": EPISODES, "sigmas": SIGMAS}, "raw": raw, "aggregated": agg}


def run_synthetic_study() -> dict:
    print("=== Synthetic structural twin ===", flush=True)
    cfg = SynthConfig()
    thr = pi_star(cfg, ALPHA)
    rho = rho_of(cfg)
    print(f"synthetic H={cfg.H} M={cfg.M} L={cfg.L} rho={rho:.3f} pi*={thr:.3f}", flush=True)

    # Decision-rule accuracy across all u
    decision_rows = []
    regret_raw = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(10_000 + seed)
        for u in US:
            step_reg_map, step_reg_cvar, step_reg_hyb = [], [], []
            match = 0
            n = 0
            for _ in range(STEPS_PER_SEED):
                s = sample_step(u, cfg, rng, calibrated=True)
                a_map = choose_map(s.probs, s.U)
                a_cvar = choose_cvar(s.probs, s.U, ALPHA)
                a_hyb = choose_hybrid(s.probs, s.U, ALPHA, pi_star_thr=thr, cfg=cfg)
                theory_rob = int(s.pi < thr)
                # robust action id is 1 in construction
                cvar_rob = int(a_cvar == 1)
                match += int(theory_rob == cvar_rob)
                n += 1
                step_reg_map.append(regret(s.U, s.true_c, a_map))
                step_reg_cvar.append(regret(s.U, s.true_c, a_cvar))
                step_reg_hyb.append(regret(s.U, s.true_c, a_hyb))
                decision_rows.append(
                    {
                        "u": u,
                        "seed": seed,
                        "pi": s.pi,
                        "theory_rob": theory_rob,
                        "cvar_rob": cvar_rob,
                        "match": int(theory_rob == cvar_rob),
                        "disagree": int(a_map != a_cvar),
                    }
                )
            for method, arr in (
                ("MAP", step_reg_map),
                ("CVaR_0.3", step_reg_cvar),
                ("Hybrid_pi_star", step_reg_hyb),
            ):
                regret_raw.append(
                    {
                        "method": method,
                        "u": float(u),
                        "seed": int(seed),
                        "mean_step_regret": float(np.mean(arr)),
                    }
                )
            print(
                f"[SYN] seed={seed}/{N_SEEDS-1} u={u} "
                f"acc={match / max(n, 1):.3f} "
                f"rMAP={np.mean(step_reg_map):.2f} rCVaR={np.mean(step_reg_cvar):.2f}",
                flush=True,
            )

    # bin decision accuracy by pi
    bins = np.linspace(0.5, 1.0, 11)
    binned = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            sub = [r for r in decision_rows if lo <= r["pi"] <= hi]
        else:
            sub = [r for r in decision_rows if lo <= r["pi"] < hi]
        if not sub:
            continue
        binned.append(
            {
                "pi_mid": float(0.5 * (lo + hi)),
                "n": len(sub),
                "frac_cvar_rob": float(np.mean([r["cvar_rob"] for r in sub])),
                "frac_theory_rob": float(np.mean([r["theory_rob"] for r in sub])),
                "accuracy": float(np.mean([r["match"] for r in sub])),
                "frac_disagree": float(np.mean([r["disagree"] for r in sub])),
            }
        )

    agg = _agg(regret_raw, "method", "u")
    overall_acc = float(np.mean([r["match"] for r in decision_rows]))
    return {
        "config": {
            "n_seeds": N_SEEDS,
            "steps_per_seed": STEPS_PER_SEED,
            "us": US,
            "alpha": ALPHA,
            "H": cfg.H,
            "M": cfg.M,
            "L": cfg.L,
            "rho": rho,
            "pi_star": thr,
        },
        "overall_decision_accuracy": overall_acc,
        "regret_raw": regret_raw,
        "aggregated": agg,
        "decision_binned": binned,
    }


def plot_sc(agg: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for name in ["MAP", "CVaR_0.3", "Hybrid_0.7"]:
        xs = [a["sigma"] for a in agg if a["method"] == name]
        ys = [a["mean_step_regret"] for a in agg if a["method"] == name]
        es = [a["sem_step_regret"] for a in agg if a["method"] == name]
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=name)
    ax.set_xlabel("sigma")
    ax.set_ylabel("mean step regret (+/- SEM over seeds)")
    ax.set_title(f"StepCountJITAI high-seed (n={N_SEEDS})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_syn_regret(agg: list[dict], thr: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for name in ["MAP", "CVaR_0.3", "Hybrid_pi_star"]:
        xs = [a["u"] for a in agg if a["method"] == name]
        ys = [a["mean_step_regret"] for a in agg if a["method"] == name]
        es = [a["sem_step_regret"] for a in agg if a["method"] == name]
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=name)
    ax.set_xlabel("synthetic uncertainty u")
    ax.set_ylabel("mean step regret (+/- SEM)")
    ax.set_title(f"Synthetic specialize-vs-robust (pi*={thr:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_syn_decision(binned: list[dict], thr: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xs = [b["pi_mid"] for b in binned]
    ax.plot(xs, [b["frac_cvar_rob"] for b in binned], marker="o", label="Empirical CVaR->robust")
    ax.plot(xs, [b["frac_theory_rob"] for b in binned], marker="x", linestyle="--", label="Theory pi<pi*")
    ax.axvline(thr, color="black", linestyle="--", label=f"pi*={thr:.3f}")
    ax.set_xlabel("belief peak pi")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Synthetic decision crossover")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    sc = run_stepcount_high_seed()
    (OUT / "highseed_stepcount.json").write_text(json.dumps(sc, indent=2), encoding="utf-8")
    plot_sc(sc["aggregated"], OUT / "highseed_stepcount_regret.png")

    syn = run_synthetic_study()
    (OUT / "highseed_synthetic.json").write_text(json.dumps(syn, indent=2), encoding="utf-8")
    plot_syn_regret(syn["aggregated"], syn["config"]["pi_star"], OUT / "highseed_synthetic_regret.png")
    plot_syn_decision(syn["decision_binned"], syn["config"]["pi_star"], OUT / "highseed_synthetic_decision.png")

    # concise summary tables
    summary = {
        "n_seeds": N_SEEDS,
        "stepcount": sc["aggregated"],
        "synthetic_decision_accuracy": syn["overall_decision_accuracy"],
        "synthetic": syn["aggregated"],
        "synthetic_pi_star": syn["config"]["pi_star"],
        "synthetic_rho": syn["config"]["rho"],
    }
    (OUT / "highseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# High-seed + synthetic validation\n\n",
        f"Seeds = **{N_SEEDS}**. Reference artifacts are not modified.\n\n",
        "## StepCountJITAI (mean +/- SEM)\n\n",
        "| sigma | MAP | CVaR_0.3 | Hybrid_0.7 | CVaR delta |\n",
        "|-------|-----|----------|------------|------------|\n",
    ]
    for s in SIGMAS:
        row = {}
        for m in ["MAP", "CVaR_0.3", "Hybrid_0.7"]:
            hit = next(a for a in sc["aggregated"] if a["method"] == m and a["sigma"] == s)
            row[m] = hit
        lines.append(
            f"| {s} | {row['MAP']['mean_step_regret']:.2f}+/-{row['MAP']['sem_step_regret']:.2f} | "
            f"{row['CVaR_0.3']['mean_step_regret']:.2f}+/-{row['CVaR_0.3']['sem_step_regret']:.2f} | "
            f"{row['Hybrid_0.7']['mean_step_regret']:.2f}+/-{row['Hybrid_0.7']['sem_step_regret']:.2f} | "
            f"{row['CVaR_0.3']['delta_vs_map']:+.2f} |\n"
        )
    lines.append(
        f"\n## Synthetic twin\n\n"
        f"- Geometry fixed: H={syn['config']['H']}, M={syn['config']['M']}, L={syn['config']['L']}\n"
        f"- rho={syn['config']['rho']:.3f}, pi*={syn['config']['pi_star']:.3f}\n"
        f"- Decision-rule accuracy: **{syn['overall_decision_accuracy']:.3f}**\n\n"
        "| u | MAP | CVaR | Hybrid | CVaR delta |\n"
        "|---|-----|------|--------|------------|\n"
    )
    for u in US:
        row = {}
        for m in ["MAP", "CVaR_0.3", "Hybrid_pi_star"]:
            hit = next(a for a in syn["aggregated"] if a["method"] == m and a["u"] == u)
            row[m] = hit
        lines.append(
            f"| {u} | {row['MAP']['mean_step_regret']:.2f}+/-{row['MAP']['sem_step_regret']:.2f} | "
            f"{row['CVaR_0.3']['mean_step_regret']:.2f}+/-{row['CVaR_0.3']['sem_step_regret']:.2f} | "
            f"{row['Hybrid_pi_star']['mean_step_regret']:.2f}+/-{row['Hybrid_pi_star']['sem_step_regret']:.2f} | "
            f"{row['CVaR_0.3']['delta_vs_map']:+.2f} |\n"
        )
    lines.append(
        "\n## Takeaway\n\n"
        "If the synthetic study recovers the same qualitative crossover "
        "(CVaR/Hybrid better at high uncertainty; MAP better when peaked), "
        "the mechanism is structural.\n"
    )
    (OUT / "HIGHSEED_FINDINGS.md").write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {OUT / 'HIGHSEED_FINDINGS.md'}", flush=True)
    print(f"synthetic decision accuracy={syn['overall_decision_accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
