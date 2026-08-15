"""Lightweight TNSM/TSC-facing service bench: multi-request ranking latency.

Not a full multi-tenant deployment claim — a desktop stress microbench that
reports p50/p95 latency under concurrent ranking workers, for venue framing.

Writes results to the configured run directory.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telemetry_ranking.journal.cvar_rankers import rank_cvar, rank_map_sim
from telemetry_ranking.rank.env_utils import make_env

from telemetry_ranking.paths import run_output_dir

OUT = run_output_dir()


def one_decision(method: str, sigma: float, seed: int) -> float:
    env = make_env(sigma=sigma, seed=seed, n_version=0)
    env.reset()
    fn = rank_map_sim if method == "MAP" else (lambda e: rank_cvar(e, alpha=0.3))
    t0 = time.perf_counter()
    _ = int(fn(env))
    return (time.perf_counter() - t0) * 1000.0


def bench(method: str, n_requests: int = 200, workers: int = 8, sigma: float = 1.0) -> dict:
    lats = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(one_decision, method, sigma, i)
            for i in range(n_requests)
        ]
        for f in as_completed(futs):
            lats.append(float(f.result()))
    arr = np.asarray(lats, dtype=float)
    return {
        "method": method,
        "sigma": sigma,
        "n_requests": n_requests,
        "workers": workers,
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
    }


def main() -> None:
    rows = []
    for workers in (1, 4, 8):
        for method in ("MAP", "CVaR_0.3"):
            print(f"bench {method} workers={workers}", flush=True)
            rows.append(bench(method, n_requests=200, workers=workers, sigma=1.0))
    payload = {
        "note": "Desktop multi-request microbench for TNSM/TSC framing; "
        "not a multi-tenant production claim.",
        "rows": rows,
    }
    out = OUT / "service_latency_bench.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out}", flush=True)
    for r in rows:
        print(
            f"  {r['method']} w={r['workers']}: "
            f"p50={r['p50_ms']:.3f} p95={r['p95_ms']:.3f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
