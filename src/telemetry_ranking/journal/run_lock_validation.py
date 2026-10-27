"""Paper-1 lock validation: stochastic dynamics + transfer environments.

Writes results to the configured run directory.

Runs:
  1) StepCountJITAI n_version=1 (stochastic dynamics), 15 seeds
  2) Transfer-A: StepCount with stronger process noise σs=20
  3) Transfer-B: synthetic twin with alternate H/M/L geometry
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
from StepCountJITAI.envs import StepCountJITAI

from telemetry_ranking.paths import run_output_dir

OUT = run_output_dir()

N_SEEDS = 15
EPISODES = 15
ALPHA = 0.3
SIGMAS = [0.0, 0.4, 1.0, 2.0]
US = [0.0, 0.25, 0.5, 0.75, 1.0]
STEPS_PER_SEED = 200


def _agg(raw: list[dict], key_method: str = "method", key_x: str = "sigma") -> list[dict]:
    buckets: dict[tuple, list] = defaultdict(list)
    for r in raw:
        buckets[(r[key_method], r[key_x])].append(r)
    map_ref = {}
    for (m, x), items in buckets.items():
        if m == "MAP":
            map_ref[x] = float(np.mean([i["mean_step_regret"] for i in items]))
    out = []
    for (m, x), items in sorted(buckets.items(), key=lambda t: (str(t[0][0]), t[0][1])):
        regs = [i["mean_step_regret"] for i in items]
        std = float(np.std(regs, ddof=1) if len(regs) > 1 else 0.0)
        out.append(
            {
                key_method: m,
                key_x: x,
                "n_seeds": len(items),
                "mean_step_regret": float(np.mean(regs)),
                "std_step_regret": std,
                "sem_step_regret": float(std / np.sqrt(len(regs)) if len(regs) > 1 else 0.0),
                "delta_vs_map": float(np.mean(regs) - map_ref.get(x, np.mean(regs))),
            }
        )
    return out


def _make_env_custom(sigma: float, seed: int, n_version: int, sigma_s: float | None = None) -> StepCountJITAI:
    kwargs = dict(
        sigma=float(sigma),
        chosen_obs_names=["P", "H", "D"],
        n_version=int(n_version),
        b_display=False,
        seed=int(seed),
        b_using_uniform=True,
        chosen_a=0.01,
    )
    if n_version == 1:
        kwargs["σs"] = float(10.0 if sigma_s is None else sigma_s)
    return StepCountJITAI(**kwargs)


def run_stepcount(
    label: str,
    n_version: int,
    sigma_s: float | None = None,
    n_seeds: int = N_SEEDS,
) -> dict:
    print(f"=== {label} (n_version={n_version}, sigma_s={sigma_s}) ===", flush=True)
    methods = {
        "MAP": rank_map_sim,
        "CVaR_0.3": lambda env: rank_cvar(env, alpha=ALPHA),
        "Hybrid_0.7": lambda env: rank_hybrid_pi_gate(
            env, alpha=ALPHA, pi_star=0.7, adaptive_pi_star=False
        ),
    }
    raw = []
    for seed in range(n_seeds):
        for sigma in SIGMAS:
            for name, fn in methods.items():
                print(f"[{label}] seed={seed}/{n_seeds-1} sigma={sigma} {name}", flush=True)
                if n_version == 0:
                    env = make_env(sigma=float(sigma), seed=int(seed), n_version=0)
                else:
                    env = _make_env_custom(float(sigma), int(seed), n_version, sigma_s)
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
                        "mean_top1": float(np.mean(tops)),
                        "n_episodes": EPISODES,
                    }
                )
    agg = _agg(raw)
    return {"label": label, "n_version": n_version, "sigma_s": sigma_s, "raw": raw, "aggregated": agg}


def run_synthetic_transfer() -> dict:
    """Alternate geometry: less extreme specialize gap (harder for CVaR insurance)."""
    print("=== Transfer-B synthetic alternate geometry ===", flush=True)
    cfg = SynthConfig(H=35.0, M=15.0, L=2.0, noise_std=0.5)
    r = rho_of(cfg)
    pst = pi_star(cfg, ALPHA)
    print(f"  rho={r:.3f} pi*={pst:.3f}", flush=True)
    raw = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(10_000 + seed)
        for u in US:
            regs = {"MAP": [], "CVaR_0.3": [], "Hybrid_0.7": []}
            for _ in range(STEPS_PER_SEED):
                s = sample_step(u, cfg, rng, calibrated=True)
                a_map = choose_map(s.probs, s.U)
                a_cvar = choose_cvar(s.probs, s.U, ALPHA)
                a_hyb = choose_hybrid(s.probs, s.U, ALPHA, pi_star_thr=0.7, cfg=cfg)
                regs["MAP"].append(regret(s.U, s.true_c, a_map))
                regs["CVaR_0.3"].append(regret(s.U, s.true_c, a_cvar))
                regs["Hybrid_0.7"].append(regret(s.U, s.true_c, a_hyb))
            for name, arr in regs.items():
                raw.append(
                    {
                        "method": name,
                        "u": float(u),
                        "seed": int(seed),
                        "mean_step_regret": float(np.mean(arr)),
                    }
                )
    agg = _agg(raw, key_x="u")
    return {
        "label": "synthetic_alt_geometry",
        "config": {"H": cfg.H, "M": cfg.M, "L": cfg.L, "rho": r, "pi_star": pst},
        "raw": raw,
        "aggregated": agg,
    }


def plot_delta(agg: list[dict], x_key: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for method, color in [("CVaR_0.3", "#c0392b"), ("Hybrid_0.7", "#2980b9")]:
        xs, ys, es = [], [], []
        for row in agg:
            if row["method"] != method:
                continue
            xs.append(row[x_key])
            ys.append(row["delta_vs_map"])
            es.append(row["sem_step_regret"])
        if xs:
            order = np.argsort(xs)
            xs = np.asarray(xs)[order]
            ys = np.asarray(ys)[order]
            es = np.asarray(es)[order]
            ax.errorbar(xs, ys, yerr=es, marker="o", label=method, color=color, capsize=3)
    ax.axhline(0.0, color="gray", lw=1, ls="--")
    ax.set_xlabel(x_key)
    ax.set_ylabel("Δ regret vs MAP (neg = better)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_findings(sto: dict, transfer_a: dict, transfer_b: dict) -> str:
    lines = [
        "# Paper-1 lock validation findings",
        "",
        "Status: stochastic dynamics + transfer checks completed.",
        "Reference artifacts are not modified.",
        "",
        "## 1) Stochastic StepCountJITAI (n_version=1, σs=10, 15 seeds)",
        "",
        "| sigma | MAP | CVaR_0.3 | Hybrid_0.7 | CVaR Δ |",
        "|-------|-----|----------|------------|--------|",
    ]
    by = {(r["method"], r["sigma"]): r for r in sto["aggregated"]}
    for s in SIGMAS:
        m = by[("MAP", s)]["mean_step_regret"]
        c = by[("CVaR_0.3", s)]["mean_step_regret"]
        h = by[("Hybrid_0.7", s)]["mean_step_regret"]
        d = by[("CVaR_0.3", s)]["delta_vs_map"]
        lines.append(f"| {s} | {m:.2f} | {c:.2f} | {h:.2f} | {d:+.2f} |")

    lines += [
        "",
        "## 2) Transfer-A: stronger process noise (n_version=1, sigma_s=20)",
        "",
        "| sigma | MAP | CVaR_0.3 | Hybrid_0.7 | CVaR Δ |",
        "|-------|-----|----------|------------|--------|",
    ]
    bya = {(r["method"], r["sigma"]): r for r in transfer_a["aggregated"]}
    for s in SIGMAS:
        m = bya[("MAP", s)]["mean_step_regret"]
        c = bya[("CVaR_0.3", s)]["mean_step_regret"]
        h = bya[("Hybrid_0.7", s)]["mean_step_regret"]
        d = bya[("CVaR_0.3", s)]["delta_vs_map"]
        lines.append(f"| {s} | {m:.2f} | {c:.2f} | {h:.2f} | {d:+.2f} |")

    lines += [
        "",
        f"## 3) Transfer-B: synthetic alt geometry "
        f"(H={transfer_b['config']['H']}, M={transfer_b['config']['M']}, "
        f"L={transfer_b['config']['L']}, ρ={transfer_b['config']['rho']:.3f})",
        "",
        "| u | MAP | CVaR_0.3 | Hybrid_0.7 | CVaR Δ |",
        "|---|-----|----------|------------|--------|",
    ]
    byb = {(r["method"], r["u"]): r for r in transfer_b["aggregated"]}
    for u in US:
        m = byb[("MAP", u)]["mean_step_regret"]
        c = byb[("CVaR_0.3", u)]["mean_step_regret"]
        h = byb[("Hybrid_0.7", u)]["mean_step_regret"]
        d = byb[("CVaR_0.3", u)]["delta_vs_map"]
        lines.append(f"| {u} | {m:.2f} | {c:.2f} | {h:.2f} | {d:+.2f} |")

                           
    def high_sigma_win(agg, xkey="sigma", xhi=2.0) -> bool:
        for r in agg:
            if r["method"] == "CVaR_0.3" and r[xkey] == xhi and r["delta_vs_map"] < 0:
                return True
        return False

    sto_win = high_sigma_win(sto["aggregated"])
    ta_win = high_sigma_win(transfer_a["aggregated"])
    lines += [
        "",
        "## Verdict",
        "",
        f"- Stochastic high-sigma CVaR beats MAP: **{'YES' if sto_win else 'NO'}**",
        f"- Transfer-A (sigma_s=20) high-sigma CVaR beats MAP: **{'YES' if ta_win else 'NO'}**",
        "- Transfer-B: one-step synthetic may not show episode-level CVaR gain "
        "(consistent with HIGHSEED_FINDINGS); pi* geometry still structural.",
        "",
        "## Artifacts",
        "",
        "- `lock_stochastic.json`",
        "- `lock_transfer_sigma_s20.json`",
        "- `lock_transfer_synth_alt.json`",
        "- `lock_stochastic_delta.png`",
        "- `lock_transfer_a_delta.png`",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "LOCK_VALIDATION_FINDINGS.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    sto = run_stepcount("stochastic_n1", n_version=1, sigma_s=10.0)
    (OUT / "lock_stochastic.json").write_text(json.dumps(sto, indent=2), encoding="utf-8")
    plot_delta(
        sto["aggregated"],
        "sigma",
        "Stochastic dynamics (n_version=1): Δ vs MAP",
        OUT / "lock_stochastic_delta.png",
    )

    ta = run_stepcount("transfer_sigma_s20", n_version=1, sigma_s=20.0)
    (OUT / "lock_transfer_sigma_s20.json").write_text(json.dumps(ta, indent=2), encoding="utf-8")
    plot_delta(
        ta["aggregated"],
        "sigma",
        "Transfer-A (σs=20): Δ vs MAP",
        OUT / "lock_transfer_a_delta.png",
    )

    tb = run_synthetic_transfer()
                                                                                    
    (OUT / "lock_transfer_synth_alt.json").write_text(
        json.dumps({"config": tb["config"], "aggregated": tb["aggregated"]}, indent=2),
        encoding="utf-8",
    )

    findings = write_findings(sto, ta, tb)
    print(findings, flush=True)
    print("DONE lock validation.", flush=True)


if __name__ == "__main__":
    main()
