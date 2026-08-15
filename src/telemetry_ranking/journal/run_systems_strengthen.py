"""Strengthen systems evidence: telemetry-loss sweep + higher multi-tenant load."""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telemetry_ranking.journal.cvar_rankers import rank_cvar, rank_map_sim
from telemetry_ranking.journal.run_systems_package import _choose, bench_multitenant
from telemetry_ranking.rank.env_utils import make_env
from telemetry_ranking.rank.rankers import policy_regret

from telemetry_ranking.paths import run_output_dir

OUT = run_output_dir()
RNG = np.random.default_rng(20260807)


def loss_sweep(
    sigma: float = 1.0,
    drop_ps: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    seeds: int = 15,
    episodes: int = 10,
) -> list[dict]:
    rows = []
    for p_drop in drop_ps:
        for method in ("MAP", "CVaR_0.3"):
            regs = []
            for seed in range(seeds):
                env = make_env(sigma=sigma, seed=seed, n_version=0)
                ep_regs = []
                for ep in range(episodes):
                    env.reset()
                    env.rng0 = np.random.default_rng(seed + 1000 * ep + 1)
                    env.rng1 = np.random.default_rng(seed + 1000 * ep + 2)
                    done = False
                    ep_r, steps = 0.0, 0
                    stale_action = None
                    while not done:
                        use_stale = (p_drop > 0) and (RNG.random() < p_drop) and stale_action is not None
                        if use_stale:
                            a = int(stale_action)
                        else:
                            a = _choose(method, env)
                            stale_action = a
                        ep_r += policy_regret(env, a)
                        _, _, term, trunc, _ = env.step(a)
                        done = bool(term or trunc)
                        steps += 1
                    ep_regs.append(ep_r / max(steps, 1))
                regs.append(float(np.mean(ep_regs)))
            arr = np.asarray(regs, dtype=float)
            rows.append(
                {
                    "sigma": sigma,
                    "drop_p": float(p_drop),
                    "method": method,
                    "n_seeds": seeds,
                    "mean_step_regret": float(np.mean(arr)),
                    "sem": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                }
            )
            print(
                f"drop={p_drop:.2f} {method}: {np.mean(arr):.2f} +/- "
                f"{(np.std(arr, ddof=1)/np.sqrt(len(arr))):.2f}",
                flush=True,
            )
    return rows


def main() -> None:
    print("=== telemetry loss sweep ===", flush=True)
    sweep = loss_sweep()
    print("=== multi-tenant scale ===", flush=True)
    mt = []
    for n in (8, 32, 64):
        print(f"users={n}", flush=True)
        mt.extend(bench_multitenant(n_users=n, steps_each=30, sigma=1.0)["multitenant"])

    payload = {
        "telemetry_loss_sweep": sweep,
        "multitenant_scale": mt,
    }
    out = OUT / "systems_strengthen.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = ["# Strengthened systems findings", "", "## Telemetry loss sweep (sigma=1)", ""]
    lines.append("| drop_p | MAP regret | CVaR regret |")
    lines.append("|--------|------------|-------------|")
    by = {}
    for r in sweep:
        by.setdefault(r["drop_p"], {})[r["method"]] = r
    for p in sorted(by):
        m, c = by[p]["MAP"], by[p]["CVaR_0.3"]
        lines.append(
            f"| {p:.2f} | {m['mean_step_regret']:.2f}+/-{m['sem']:.2f} | "
            f"{c['mean_step_regret']:.2f}+/-{c['sem']:.2f} |"
        )
    lines += ["", "## Multi-tenant scale-up", ""]
    lines.append("| users | method | p95 ms | throughput dec/s |")
    lines.append("|-------|--------|--------|------------------|")
    for r in mt:
        lines.append(
            f"| {r['n_users']} | {r['method']} | {r['rank_p95_ms']:.1f} | "
            f"{r['throughput_decisions_per_s']:.1f} |"
        )
    (OUT / "SYSTEMS_STRENGTHEN_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
