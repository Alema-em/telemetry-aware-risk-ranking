"""TNSM systems package: queueing, jitter/loss, multi-tenant ranking load.

Secondary service-feasibility evidence for journal. Writes results to the configured run directory.
Does not claim live multi-tenant production deployment.
"""
from __future__ import annotations

import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telemetry_ranking.journal.cvar_rankers import rank_cvar, rank_map_sim
from telemetry_ranking.rank.env_utils import belief_probs, make_env
from telemetry_ranking.rank.rankers import policy_regret

from telemetry_ranking.paths import run_output_dir

OUT = run_output_dir()

RNG = np.random.default_rng(20260806)


def _choose(method: str, env):
    if method == "MAP":
        return int(rank_map_sim(env))
    return int(rank_cvar(env, alpha=0.3))


def bench_e2e_breakdown(n: int = 200, sigma: float = 1.0) -> dict:
    """Time twin step (sense/monitor proxy) vs ranking alone."""
    rows = []
    for method in ("MAP", "CVaR_0.3"):
        sense_ms, rank_ms, total_ms = [], [], []
        for i in range(n):
            env = make_env(sigma=sigma, seed=i, n_version=0)
            env.reset()
            t0 = time.perf_counter()
            env.step(0)  # sense/monitor proxy: advance twin
            t1 = time.perf_counter()
            _ = _choose(method, env)
            t2 = time.perf_counter()
            sense_ms.append((t1 - t0) * 1000.0)
            rank_ms.append((t2 - t1) * 1000.0)
            total_ms.append((t2 - t0) * 1000.0)
        rows.append(
            {
                "method": method,
                "n": n,
                "sigma": sigma,
                "sense_monitor_p50_ms": float(np.percentile(sense_ms, 50)),
                "rank_p50_ms": float(np.percentile(rank_ms, 50)),
                "rank_p95_ms": float(np.percentile(rank_ms, 95)),
                "e2e_p50_ms": float(np.percentile(total_ms, 50)),
                "e2e_p95_ms": float(np.percentile(total_ms, 95)),
            }
        )
    return {"e2e_breakdown": rows}


def bench_queue(arrival_hz: float, n_jobs: int = 300, workers: int = 1, sigma: float = 1.0) -> dict:
    """
    Simple single-queue multi-worker ranking service.
    Jobs arrive as Poisson; waiting time = queue delay + service time.
    """
    methods = {}
    for method in ("MAP", "CVaR_0.3"):
        # pre-generate service times by measuring
        svc = []
        for i in range(80):
            env = make_env(sigma=sigma, seed=10_000 + i, n_version=0)
            env.reset()
            t0 = time.perf_counter()
            _ = _choose(method, env)
            svc.append(time.perf_counter() - t0)
        mean_svc = float(np.mean(svc))

        # event simulation in simulated time (not wall clock) using measured svc dist
        svc_arr = np.asarray(svc)
        arrivals = RNG.exponential(1.0 / arrival_hz, size=n_jobs)
        t_arr = np.cumsum(arrivals)
        # each worker next-free time
        free = np.zeros(workers, dtype=float)
        waits, sojourns = [], []
        for ta in t_arr:
            w = int(np.argmin(free))
            start = max(ta, free[w])
            s = float(RNG.choice(svc_arr))
            end = start + s
            free[w] = end
            waits.append(start - ta)
            sojourns.append(end - ta)
        methods[method] = {
            "arrival_hz": arrival_hz,
            "workers": workers,
            "n_jobs": n_jobs,
            "mean_service_ms": mean_svc * 1000.0,
            "wait_p50_ms": float(np.percentile(waits, 50) * 1000.0),
            "wait_p95_ms": float(np.percentile(waits, 95) * 1000.0),
            "sojourn_p50_ms": float(np.percentile(sojourns, 50) * 1000.0),
            "sojourn_p95_ms": float(np.percentile(sojourns, 95) * 1000.0),
        }
    return methods


def bench_jitter_loss(
    sigma: float = 1.0,
    seeds: int = 15,
    episodes: int = 10,
    drop_p: float = 0.1,
    jitter_ms: float = 50.0,
) -> dict:
    """
    Packet-loss / belief-staleness stress: with prob drop_p, reuse previous belief
    action context by skipping env observation refresh (approximate stale belief).
    Reports MAP vs CVaR regret under clean vs lossy monitoring.
    """
    out = []
    for condition, p_drop in (("clean", 0.0), ("lossy", drop_p)):
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
                        # optional network delay (busy wait tiny / recorded only)
                        if jitter_ms > 0 and condition == "lossy":
                            time.sleep(min(jitter_ms, 5.0) / 1000.0 * 0.0)  # no real sleep; keep fast
                        use_stale = condition == "lossy" and (RNG.random() < p_drop) and stale_action is not None
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
            out.append(
                {
                    "condition": condition,
                    "method": method,
                    "sigma": sigma,
                    "drop_p": p_drop if condition == "lossy" else 0.0,
                    "n_seeds": seeds,
                    "mean_step_regret": float(np.mean(arr)),
                    "sem": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                }
            )
    return {"jitter_loss": out}


def bench_multitenant(
    n_users: int = 8,
    steps_each: int = 40,
    sigma: float = 1.0,
    methods: tuple[str, ...] = ("MAP", "CVaR_0.3"),
    seed_offset: int = 0,
) -> dict:
    """Concurrent twin+rank loops (thread pool) as a crude multi-tenant stress."""
    def user_loop(uid: int, method: str):
        env = make_env(sigma=sigma, seed=seed_offset + uid, n_version=0)
        env.reset()
        lats = []
        for s in range(steps_each):
            t0 = time.perf_counter()
            a = _choose(method, env)
            lats.append((time.perf_counter() - t0) * 1000.0)
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                env.reset()
        return lats

    rows = []
    for method in methods:
        all_lats = []
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_users) as ex:
            futs = [ex.submit(user_loop, u, method) for u in range(n_users)]
            for f in as_completed(futs):
                all_lats.extend(f.result())
        wall = time.perf_counter() - t0
        arr = np.asarray(all_lats, dtype=float)
        rows.append(
            {
                "method": method,
                "n_users": n_users,
                "steps_each": steps_each,
                "seed_offset": seed_offset,
                "wall_s": float(wall),
                "rank_p50_ms": float(np.percentile(arr, 50)),
                "rank_p95_ms": float(np.percentile(arr, 95)),
                "throughput_decisions_per_s": float(len(arr) / wall),
            }
        )
    return {"multitenant": rows}


def main() -> None:
    print("=== e2e breakdown ===", flush=True)
    e2e = bench_e2e_breakdown()
    print("=== queueing ===", flush=True)
    queue_rows = []
    for hz in (20.0, 50.0, 100.0):
        for workers in (1, 4):
            print(f"queue hz={hz} workers={workers}", flush=True)
            q = bench_queue(arrival_hz=hz, workers=workers)
            for method, row in q.items():
                queue_rows.append(row | {"method": method})
    print("=== jitter/loss ===", flush=True)
    jl = bench_jitter_loss()
    print("=== multitenant ===", flush=True)
    mt = bench_multitenant()

    # reuse prior concurrency microbench if present
    prior = OUT / "service_latency_bench.json"
    prior_data = json.loads(prior.read_text(encoding="utf-8")) if prior.exists() else None

    payload = {
        "note": "Desktop systems package for TNSM framing; not a live multi-tenant field deployment.",
        **e2e,
        "queueing": queue_rows,
        **jl,
        **mt,
        "prior_concurrency_microbench": prior_data,
    }
    out = OUT / "systems_package.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md = OUT / "SYSTEMS_PACKAGE_FINDINGS.md"
    lines = [
        "# TNSM systems package findings",
        "",
        "## E2E breakdown (sense/monitor proxy vs rank)",
        "",
    ]
    for r in e2e["e2e_breakdown"]:
        lines.append(
            f"- **{r['method']}**: rank p50={r['rank_p50_ms']:.3f} ms, "
            f"e2e p95={r['e2e_p95_ms']:.3f} ms"
        )
    lines += ["", "## Queueing (simulated Poisson arrivals)", ""]
    for r in queue_rows:
        if r["arrival_hz"] in (50.0, 100.0) and r["workers"] in (1, 4):
            lines.append(
                f"- {r['method']} λ={r['arrival_hz']}/s w={r['workers']}: "
                f"wait p95={r['wait_p95_ms']:.2f} ms, sojourn p95={r['sojourn_p95_ms']:.2f} ms"
            )
    lines += ["", "## Monitoring loss (stale-action approx)", ""]
    for r in jl["jitter_loss"]:
        lines.append(
            f"- {r['condition']} {r['method']}: regret={r['mean_step_regret']:.2f} ± {r['sem']:.2f}"
        )
    lines += ["", "## Multi-tenant concurrent twins", ""]
    for r in mt["multitenant"]:
        lines.append(
            f"- {r['method']} users={r['n_users']}: p95={r['rank_p95_ms']:.2f} ms, "
            f"throughput={r['throughput_decisions_per_s']:.1f} dec/s"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}", flush=True)
    print(f"Saved {md}", flush=True)


if __name__ == "__main__":
    main()
