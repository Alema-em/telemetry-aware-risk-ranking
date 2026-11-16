from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from belief_age.age_gate import BeliefAgeTracker, GateConfig, belief_peak, decide_gate
from belief_age.sticky_world import StickyWorld, ingest_telemetry_proxy

def bench_direct(n: int = 500) -> dict:
    world = StickyWorld(sigma=1.0, seed=0)
    world.reset(0)
    tracker = BeliefAgeTracker()
    cfg = GateConfig(policy="peak_and_age")
    times_ms: list[float] = []
    for i in range(n):
        tel = {
            "device_id": "bench",
            "seq": i,
            "t_ms": i * 1000,
            "steps": float(i),
            "hr": 70.0,
            "schema": "jitai.telemetry.v1",
        }
        t0 = time.perf_counter()
        ingest_telemetry_proxy(world, tel)
        if i % 3 == 0:
            tracker.on_update(world.belief_probs())
        else:
            tracker.on_stale()
            world.predict_belief(1)
        p = world.belief_probs()
        pi = belief_peak(p)
        gate = decide_gate(tracker.delta, pi, cfg)
        _ = world.choose_map(p) if gate.value == "act" else (1 if gate.value == "defer" else 0)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return _summarize("sticky_twin_direct", times_ms)

def _summarize(name: str, times_ms: list[float]) -> dict:
    arr = sorted(times_ms)
    def pct(p: float) -> float:
        if not arr:
            return 0.0
        k = int(round((p / 100.0) * (len(arr) - 1)))
        return float(arr[k])
    return {
        "runtime": name,
        "available": True,
        "n": len(arr),
        "mean_ms": float(statistics.mean(arr)) if arr else 0.0,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "max_ms": float(max(arr)) if arr else 0.0,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-direct", type=int, default=500)
    parser.add_argument("--n-lg", type=int, default=150)
    args = parser.parse_args()

    direct = bench_direct(args.n_direct)
    payload = {
        "host_note": "Measured on development host; Pi-class ARM is expected within ~2–5× for Python paths.",
        "direct": direct,
        "langgraph": lg,
    }
    out = ROOT / "results" / "gateway_overhead.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
