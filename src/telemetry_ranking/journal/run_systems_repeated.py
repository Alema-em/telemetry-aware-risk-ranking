"""Repeat multi-tenant SLA benches and the sigma-2 loss check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telemetry_ranking.journal.run_systems_package import _choose, bench_multitenant
from telemetry_ranking.journal.run_systems_strengthen import loss_sweep

from telemetry_ranking.paths import run_output_dir

OUT = run_output_dir()


def repeated_multitenant(
    users: tuple[int, ...] = (8, 32, 64),
    repeats: int = 8,
    steps_each: int = 30,
) -> list[dict]:
    rows = []
    for n in users:
        for method in ("MAP", "CVaR_0.3"):
            p95s, thrus = [], []
            for r in range(repeats):
                out = bench_multitenant(
                    n_users=n,
                    steps_each=steps_each,
                    sigma=1.0,
                    methods=(method,),
                    seed_offset=1000 * r + n,
                )
                mt = out["multitenant"][0]
                p95s.append(float(mt["rank_p95_ms"]))
                thrus.append(float(mt["throughput_decisions_per_s"]))
                print(
                    f"n={n} {method} rep={r+1}/{repeats}: "
                    f"p95={mt['rank_p95_ms']:.1f} thr={mt['throughput_decisions_per_s']:.1f}",
                    flush=True,
                )
            p95 = np.asarray(p95s, dtype=float)
            thr = np.asarray(thrus, dtype=float)
            rows.append(
                {
                    "n_users": n,
                    "method": method,
                    "repeats": repeats,
                    "rank_p95_mean_ms": float(np.mean(p95)),
                    "rank_p95_std_ms": float(np.std(p95, ddof=1)) if len(p95) > 1 else 0.0,
                    "throughput_mean": float(np.mean(thr)),
                    "throughput_std": float(np.std(thr, ddof=1)) if len(thr) > 1 else 0.0,
                    "rank_p95_reps": p95s,
                    "throughput_reps": thrus,
                }
            )
    return rows


def main() -> None:
    print("=== repeated multi-tenant ===", flush=True)
    mt = repeated_multitenant(repeats=8)

    print("=== stale sweep at sigma=2 (spot confirm) ===", flush=True)
    sweep2 = loss_sweep(sigma=2.0, drop_ps=(0.0, 0.10, 0.20), seeds=10, episodes=8)

    payload = {
        "multitenant_repeated": mt,
        "telemetry_loss_sweep_sigma2": sweep2,
        "note": "8 repeats for SLA cells; sigma=2 stale sweep is confirmatory mid-check.",
    }
    out = OUT / "systems_repeated.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Repeated systems benchmarks", "", "## Multi-tenant (8 repeats)", ""]
    lines.append("| users | method | p95 ms (mean±SD) | thr dec/s (mean±SD) |")
    lines.append("|-------|--------|------------------|---------------------|")
    for r in mt:
        lines.append(
            f"| {r['n_users']} | {r['method']} | "
            f"{r['rank_p95_mean_ms']:.1f}±{r['rank_p95_std_ms']:.1f} | "
            f"{r['throughput_mean']:.1f}±{r['throughput_std']:.1f} |"
        )
    lines += ["", "## Stale telemetry at sigma=2", ""]
    lines.append("| drop_p | MAP | CVaR |")
    lines.append("|--------|-----|------|")
    by = {}
    for r in sweep2:
        by.setdefault(r["drop_p"], {})[r["method"]] = r
    for p in sorted(by):
        m, c = by[p]["MAP"], by[p]["CVaR_0.3"]
        lines.append(
            f"| {p:.2f} | {m['mean_step_regret']:.2f}±{m['sem']:.2f} | "
            f"{c['mean_step_regret']:.2f}±{c['sem']:.2f} |"
        )
    (OUT / "SYSTEMS_REPEATED_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
