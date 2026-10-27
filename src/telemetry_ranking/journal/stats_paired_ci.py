"""Paired bootstrap intervals for the journal results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from telemetry_ranking.paths import run_output_dir

OUT = run_output_dir()

RNG = np.random.default_rng(20260805)
N_BOOT = 5000


def paired_delta_stats(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    """a,b: paired per-seed values. Delta = b - a (negative favours b if lower-is-better)."""
    assert len(a) == len(b) and len(a) > 1
    d = b - a
    n = len(d)
    mean_d = float(np.mean(d))
                                                  
    boots = []
    idx = np.arange(n)
    for _ in range(N_BOOT):
        take = RNG.choice(idx, size=n, replace=True)
        boots.append(float(np.mean(d[take])))
    boots_arr = np.asarray(boots)
    ci_low, ci_high = np.percentile(boots_arr, [2.5, 97.5])
                               
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    t_stat = mean_d / se if se > 0 else float("inf")
                                                                                       
    from math import erfc, sqrt

    z = abs(t_stat)
    p_approx = float(erfc(z / sqrt(2.0)))                           
                                                                            
    wilcoxon_p = None
    try:
        from scipy.stats import t as student_t
        from scipy.stats import wilcoxon

        p_t = float(student_t.sf(abs(t_stat), df=n - 1) * 2.0)
                                                 
        if np.any(d != 0):
            wilcoxon_p = float(wilcoxon(d, alternative="two-sided").pvalue)
        p_approx = p_t
    except Exception:
        p_t = p_approx

    return {
        "contrast": label,
        "n_pairs": n,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_delta_b_minus_a": mean_d,
        "bootstrap_ci95": [float(ci_low), float(ci_high)],
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(p_approx),
        "wilcoxon_pvalue": wilcoxon_p,
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
    }


def _seed_vectors(raw: list[dict], method_a: str, method_b: str, sigma: float, key="method"):
    by_seed_a, by_seed_b = {}, {}
    for r in raw:
        if float(r["sigma"]) != float(sigma):
            continue
        m = r.get(key) or r.get("ranker")
        if m == method_a:
            by_seed_a[int(r["seed"])] = float(r["mean_step_regret"])
        elif m == method_b:
            by_seed_b[int(r["seed"])] = float(r["mean_step_regret"])
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    a = np.asarray([by_seed_a[s] for s in seeds], dtype=float)
    b = np.asarray([by_seed_b[s] for s in seeds], dtype=float)
    return a, b, seeds


def stats_highseed_cvar() -> list[dict]:
    path = OUT / "highseed_stepcount.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data["raw"]
    rows = []
    for sigma in sorted({float(r["sigma"]) for r in raw}):
        for other in ("CVaR_0.3", "Hybrid_0.7"):
            a, b, seeds = _seed_vectors(raw, "MAP", other, sigma, key="method")
            if len(seeds) < 2:
                continue
                                                                        
            st = paired_delta_stats(a, b, f"MAP vs {other} @ σ={sigma}")
            st["sigma"] = sigma
            st["method_a"] = "MAP"
            st["method_b"] = other
            rows.append(st)
    return rows


def stats_regime_map() -> list[dict]:
    path = OUT / "regime_map_30seed.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
                                                                                  
    raw = []
    for r in data.get("raw_rows", []):
                                                                                  
        raw.append(r)
                                                          
    if raw and "seed" not in raw[0]:
                                                                        
        return [
            {
                "note": "regime_map_30seed.json raw_rows lack per-seed id for pairing; "
                "re-run with seed stamped or use aggregated SEM CIs only.",
                "n_aggregated": len(data.get("aggregated", [])),
            }
        ]
    rows = []
    for sigma in sorted({float(r["sigma"]) for r in raw}):
        a, b, seeds = _seed_vectors(raw, "B1_MAP", "Belief_LTR", sigma, key="ranker")
        if len(seeds) >= 2:
            st = paired_delta_stats(a, b, f"MAP-Sim vs Belief-LTR @ σ={sigma}")
            st["sigma"] = sigma
            rows.append(st)
        a, b, seeds = _seed_vectors(raw, "B1_MAP", "B3_BASR", sigma, key="ranker")
        if len(seeds) >= 2:
            st = paired_delta_stats(a, b, f"MAP-Sim vs BASR @ σ={sigma}")
            st["sigma"] = sigma
            rows.append(st)
    return rows


def main() -> None:
    report = {
        "config": {"n_bootstrap": N_BOOT, "alpha": 0.05},
        "cvar_vs_map": stats_highseed_cvar(),
        "regime_map_contrasts": stats_regime_map(),
    }
    out = OUT / "stats_paired_ci.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = OUT / "STATS_FINDINGS.md"
    lines = ["# Paired statistics (journal)", ""]
    lines.append("## CVaR / Hybrid vs MAP (30 seeds)")
    lines.append("")
    for r in report["cvar_vs_map"]:
        lines.append(
            f"- **{r['contrast']}**: Δ={r['mean_delta_b_minus_a']:.3f} "
            f"(95% boot CI [{r['bootstrap_ci95'][0]:.3f}, {r['bootstrap_ci95'][1]:.3f}]), "
            f"paired t p≈{r['paired_t_pvalue']:.4g}, "
            f"CI excludes 0: {r['ci_excludes_zero']}"
        )
    lines.append("")
    lines.append("## Risk-neutral comparison")
    lines.append("")
    for r in report["regime_map_contrasts"]:
        if "note" in r:
            lines.append(f"- {r['note']}")
        else:
            lines.append(
                f"- **{r['contrast']}**: Δ={r['mean_delta_b_minus_a']:.3f} "
                f"CI [{r['bootstrap_ci95'][0]:.3f}, {r['bootstrap_ci95'][1]:.3f}] "
                f"p≈{r['paired_t_pvalue']:.4g}"
            )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out}", flush=True)
    print(f"Saved {md}", flush=True)


if __name__ == "__main__":
    main()
