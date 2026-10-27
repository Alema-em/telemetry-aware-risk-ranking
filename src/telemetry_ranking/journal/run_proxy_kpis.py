"""Low-confidence specialised-prompt rate. The true context is not required.

Reports, per method × σ:
  - specialized-prompt rate (a in {2,3})
  - robust-prompt rate (a == 1)
  - mean belief peak π
  - low-confidence specialised-prompt rate (π < 0.7 and a in {2,3})
  - mean decision latency (ms)
  - hybrid gate-CVaR fraction (when applicable)

Writes results to the configured run directory.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telemetry_ranking.journal.cvar_rankers import rank_cvar, rank_hybrid_pi_gate, rank_map_sim
from telemetry_ranking.rank.env_utils import belief_probs, make_env
from telemetry_ranking.rank.rankers import policy_regret

from telemetry_ranking.paths import run_output_dir

OUT = run_output_dir()

N_SEEDS = 20
EPISODES = 15
SIGMAS = [0.0, 0.4, 1.0, 2.0]
ALPHA = 0.3
PI_GATE = 0.7


def run_method(name: str, choose, sigma: float, seed: int) -> dict:
    env = make_env(sigma=float(sigma), seed=int(seed), n_version=0)
    n_steps = 0
    n_spec = n_rob = n_noop = 0
    n_low_conf_spec = 0
    peaks = []
    latencies = []
    gate_cvar = 0
    regrets = []
    for ep in range(EPISODES):
        env.reset()
        env.rng0 = np.random.default_rng(seed + 1000 * ep + 1)
        env.rng1 = np.random.default_rng(seed + 1000 * ep + 2)
        done = False
        ep_r, steps = 0.0, 0
        while not done:
            p = belief_probs(env)
            pi = float(np.max(p))
            peaks.append(pi)
            t0 = time.perf_counter()
            a = int(choose(env))
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if name.startswith("Hybrid"):
                                                         
                if pi < PI_GATE:
                    gate_cvar += 1
            if a in (2, 3):
                n_spec += 1
                if pi < PI_GATE:
                    n_low_conf_spec += 1
            elif a == 1:
                n_rob += 1
            elif a == 0:
                n_noop += 1
            ep_r += policy_regret(env, a)
            _, _, term, trunc, _ = env.step(a)
            done = bool(term or trunc)
            steps += 1
            n_steps += 1
        regrets.append(ep_r / max(steps, 1))
    return {
        "method": name,
        "sigma": float(sigma),
        "seed": int(seed),
        "mean_step_regret": float(np.mean(regrets)),
        "specialized_rate": n_spec / max(n_steps, 1),
        "robust_rate": n_rob / max(n_steps, 1),
        "noop_rate": n_noop / max(n_steps, 1),
        "mean_belief_peak": float(np.mean(peaks)),
        "low_conf_specialized_rate": n_low_conf_spec / max(n_steps, 1),
        "mean_latency_ms": float(np.mean(latencies)),
        "frac_gate_cvar": gate_cvar / max(n_steps, 1) if name.startswith("Hybrid") else None,
    }


def main() -> None:
    methods = {
        "MAP": rank_map_sim,
        "CVaR_0.3": lambda e: rank_cvar(e, alpha=ALPHA),
        "Hybrid_0.7": lambda e: rank_hybrid_pi_gate(
            e, alpha=ALPHA, pi_star=PI_GATE, adaptive_pi_star=False
        ),
    }
    raw = []
    total = N_SEEDS * len(SIGMAS) * len(methods)
    k = 0
    for seed in range(N_SEEDS):
        for sigma in SIGMAS:
            for name, fn in methods.items():
                k += 1
                print(f"[{k}/{total}] seed={seed} sigma={sigma} {name}", flush=True)
                raw.append(run_method(name, fn, sigma, seed))

               
    agg = []
    for sigma in SIGMAS:
        for name in methods:
            rows = [r for r in raw if r["method"] == name and r["sigma"] == sigma]
            def m(key):
                return float(np.mean([r[key] for r in rows]))

            agg.append(
                {
                    "method": name,
                    "sigma": sigma,
                    "n_seeds": len(rows),
                    "mean_step_regret": m("mean_step_regret"),
                    "specialized_rate": m("specialized_rate"),
                    "robust_rate": m("robust_rate"),
                    "low_conf_specialized_rate": m("low_conf_specialized_rate"),
                    "mean_belief_peak": m("mean_belief_peak"),
                    "mean_latency_ms": m("mean_latency_ms"),
                }
            )

    payload = {
        "config": {
            "n_seeds": N_SEEDS,
            "episodes": EPISODES,
            "proxy_definition": "low_conf_specialized_rate = P(a in {2,3} and max belief < 0.7)",
        },
        "raw": raw,
        "aggregated": agg,
    }
    out = OUT / "proxy_kpis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md = OUT / "PROXY_KPI_FINDINGS.md"
    lines = [
        "# Low-confidence specialised-prompt rate",
        "",
        "Primary research metric remains oracle regret. Proxies below need no true context.",
        "",
        "| Method | σ | specialized | low-conf specialized | robust | latency ms |",
        "|--------|---|-------------|----------------------|--------|------------|",
    ]
    for r in agg:
        lines.append(
            f"| {r['method']} | {r['sigma']} | {r['specialized_rate']:.3f} | "
            f"{r['low_conf_specialized_rate']:.3f} | {r['robust_rate']:.3f} | "
            f"{r['mean_latency_ms']:.3f} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
