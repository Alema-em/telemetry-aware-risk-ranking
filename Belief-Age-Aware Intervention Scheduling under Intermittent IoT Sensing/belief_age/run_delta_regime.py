from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from belief_age.telemetry_channel import TelemetryChannel
from belief_age.age_gate import BeliefAgeTracker, GateConfig, GateDecision, belief_peak, decide_gate
from belief_age.sticky_world import StickyWorld

ACTION_WAIT = 0
ACTION_DEFER = 1
BOOT_RNG = np.random.default_rng(20260919)
N_BOOT = 4000

def _choose_act(world: StickyWorld, ranker: str, probs: np.ndarray) -> int:
    if ranker == "cvar":
        return world.choose_cvar(probs, alpha=0.3)
    return world.choose_map(probs)

def bootstrap_mean_ci(values: list[float] | np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    if n == 1:
        return mean, mean, mean
    boots = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for i in range(n_boot):
        take = BOOT_RNG.choice(idx, size=n, replace=True)
        boots[i] = float(np.mean(arr[take]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)

def paired_delta_ci(a: list[float], b: list[float], n_boot: int = N_BOOT) -> dict:
    """Delta = b - a (negative => b lower regret than a)."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    d = bb - aa
    mean_d = float(np.mean(d))
    n = len(d)
    boots = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for i in range(n_boot):
        take = BOOT_RNG.choice(idx, size=n, replace=True)
        boots[i] = float(np.mean(d[take]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    t_stat = mean_d / se if se > 0 else 0.0
    # normal approx two-sided p
    from math import erfc, sqrt

    p_approx = float(erfc(abs(t_stat) / sqrt(2.0))) if se > 0 else 1.0
    try:
        from scipy.stats import t as student_t

        p_approx = float(student_t.sf(abs(t_stat), df=n - 1) * 2.0)
    except Exception:
        pass
    return {
        "mean_delta": mean_d,
        "ci95": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "paired_t_pvalue": p_approx,
    }

def run_episode(
    sigma: float,
    p_loss: float,
    seed: int,
    gate_cfg: GateConfig,
    ranker: str = "map",
    max_steps: int = 50,
    diffuse: bool = True,
    channel_kind: str = "independent",
) -> dict:
    world = StickyWorld(sigma=sigma, seed=seed)
    world.reset(seed=seed)
    rng = np.random.default_rng(seed + 17)
    channel = TelemetryChannel(kind=channel_kind, p_loss=p_loss, rng=rng)  # type: ignore[arg-type]
    channel.reset()
    tracker = BeliefAgeTracker()

    step_regrets: list[float] = []
    rewards: list[float] = []
    gate_counts = {g.value: 0 for g in GateDecision}
    ages: list[int] = []

    for _ in range(max_steps):
        delivered = channel.delivered()
        if delivered:
            tracker.on_update(world.belief_probs())
        else:
            tracker.on_stale()

        live = world.belief_probs()
        if (not delivered) and (not diffuse) and tracker.last_probs is not None:
            belief_for_gate = np.asarray(tracker.last_probs, dtype=float).copy()
        else:
            belief_for_gate = live

        pi = belief_peak(belief_for_gate)
        gate = decide_gate(tracker.delta, pi, gate_cfg)
        gate_counts[gate.value] += 1
        ages.append(int(tracker.delta))

        if gate == GateDecision.WAIT:
            action = ACTION_WAIT
        elif gate == GateDecision.DEFER:
            action = ACTION_DEFER
        else:
            action = _choose_act(world, ranker, belief_for_gate)

        out = world.step(int(action), monitoring_delivered=delivered)
        if delivered:
            tracker.last_probs = world.belief_probs()
        step_regrets.append(float(out["regret"]))
        rewards.append(float(out["reward"]))

    n = max(len(step_regrets), 1)
    return {
        "sigma": sigma,
        "p_loss": p_loss,
        "seed": seed,
        "policy": gate_cfg.policy,
        "delta_star": int(gate_cfg.delta_star),
        "ranker": ranker,
        "diffuse": diffuse,
        "channel": channel_kind,
        "mean_step_regret": float(np.mean(step_regrets)),
        "mean_episode_reward": float(np.sum(rewards)),
        "mean_age": float(np.mean(ages)) if ages else 0.0,
        "frac_act": gate_counts["act"] / n,
        "frac_wait": gate_counts["wait"] / n,
        "frac_defer": gate_counts["defer"] / n,
        "empirical_loss": float(channel.empirical_loss),
        "steps": len(step_regrets),
        "world": "sticky_context",
    }

def policy_key(pname: str, cfg: GateConfig) -> str:
    if pname.startswith("peak") and cfg.delta_star != 1:
        return f"peak_and_age_d{cfg.delta_star}"
    return pname

def aggregate(
    ep_rows: list[dict],
    sigma: float,
    p_loss: float,
    pname: str,
    ranker: str,
    n_seeds: int,
    channel: str,
    extra: dict | None = None,
    baseline_regrets: list[float] | None = None,
) -> dict:
    regrets = [r["mean_step_regret"] for r in ep_rows]
    mean, lo, hi = bootstrap_mean_ci(regrets)
    row = {
        "sigma": sigma,
        "p_loss": p_loss,
        "policy": pname,
        "ranker": ranker,
        "channel": channel,
        "n_seeds": n_seeds,
        "mean_step_regret": mean,
        "sem_step_regret": float(np.std(regrets) / max(np.sqrt(n_seeds), 1)),
        "ci95_low": lo,
        "ci95_high": hi,
        "mean_episode_reward": float(np.mean([r["mean_episode_reward"] for r in ep_rows])),
        "mean_age": float(np.mean([r["mean_age"] for r in ep_rows])),
        "frac_act": float(np.mean([r["frac_act"] for r in ep_rows])),
        "frac_wait": float(np.mean([r["frac_wait"] for r in ep_rows])),
        "frac_defer": float(np.mean([r["frac_defer"] for r in ep_rows])),
        "world": "sticky_context",
        "seed_regrets": regrets,
    }
    if baseline_regrets is not None and len(baseline_regrets) == len(regrets):
        row["vs_always_act"] = paired_delta_ci(baseline_regrets, regrets)
    if extra:
        row.update(extra)
    return row

def main() -> None:
    parser = argparse.ArgumentParser(description="Paper2 sticky-world intermittency regime")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--out", type=str, default=str(ROOT / "results" / "delta_regime.json"))
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--bursty", action="store_true", help="Also run Gilbert-Elliott bursty channel")
    args = parser.parse_args()

    sigmas = [0.4, 1.0, 2.0]
    p_losses = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    policies: list[tuple[str, GateConfig]] = [
        ("always_act", GateConfig(policy="always_act")),
        ("age_wait", GateConfig(policy="age_wait", delta_star=1)),
        ("peak_and_age", GateConfig(policy="peak_and_age", delta_star=1, pi_star=0.7)),
        ("peak_and_age", GateConfig(policy="peak_and_age", delta_star=2, pi_star=0.7)),
    ]
    rankers = ["map", "cvar"]
    channels = ["independent"]
    if args.bursty:
        channels.append("bursty")

    rows: list[dict] = []
    for channel in channels:
        for sigma in sigmas:
            for p_loss in p_losses:
                # Collect Always-Act MAP seed vector first for paired contrasts
                baseline_map: dict[str, list[float]] = {}
                for pname, cfg in policies:
                    key = policy_key(pname, cfg)
                    for ranker in rankers:
                        ep_rows = [
                            run_episode(
                                sigma=sigma,
                                p_loss=p_loss,
                                seed=s,
                                gate_cfg=cfg,
                                ranker=ranker,
                                max_steps=args.steps,
                                diffuse=True,
                                channel_kind=channel,
                            )
                            for s in range(args.seeds)
                        ]
                        base = None
                        if ranker == "map":
                            if key == "always_act":
                                baseline_map[f"{sigma}_{p_loss}_{channel}"] = [
                                    r["mean_step_regret"] for r in ep_rows
                                ]
                            else:
                                base = baseline_map.get(f"{sigma}_{p_loss}_{channel}")
                        row = aggregate(
                            ep_rows,
                            sigma,
                            p_loss,
                            key,
                            ranker,
                            args.seeds,
                            channel,
                            baseline_regrets=base,
                        )
                        # drop bulky seed list from printed row copy in JSON? keep for stats
                        rows.append(row)
                        vs = ""
                        if "vs_always_act" in row:
                            d = row["vs_always_act"]
                            vs = (
                                f" Δ={d['mean_delta']:+.2f}"
                                f" CI[{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]"
                                f"{'*' if d['ci_excludes_zero'] else ''}"
                            )
                        print(
                            f"{channel[:5]} σ={sigma:g} p={p_loss:g} {key:16s} {ranker:4s} "
                            f"regret={row['mean_step_regret']:.2f} "
                            f"CI[{row['ci95_low']:.2f},{row['ci95_high']:.2f}] "
                            f"act={row['frac_act']:.2f}{vs}",
                            flush=True,
                        )

    ablation_rows: list[dict] = []
    if args.ablations:
        for diffuse in (True, False):
            ep_rows = [
                run_episode(
                    1.0,
                    0.3,
                    s,
                    GateConfig(policy="peak_and_age", delta_star=1, pi_star=0.7),
                    "map",
                    args.steps,
                    diffuse,
                    "independent",
                )
                for s in range(args.seeds)
            ]
            ablation_rows.append(
                aggregate(
                    ep_rows,
                    1.0,
                    0.3,
                    "peak_and_age",
                    "map",
                    args.seeds,
                    "independent",
                    extra={"ablation": "diffusion", "diffuse": diffuse},
                )
            )
        for dstar in (1, 2, 3, 5):
            ep_rows = [
                run_episode(
                    1.0,
                    0.3,
                    s,
                    GateConfig(policy="peak_and_age", delta_star=dstar, pi_star=0.7),
                    "map",
                    args.steps,
                    True,
                    "independent",
                )
                for s in range(args.seeds)
            ]
            ablation_rows.append(
                aggregate(
                    ep_rows,
                    1.0,
                    0.3,
                    f"peak_and_age_d{dstar}",
                    "map",
                    args.seeds,
                    "independent",
                    extra={"ablation": "delta_star", "delta_star": dstar},
                )
            )
            print(
                f"ablation Δ★={dstar} regret={ablation_rows[-1]['mean_step_regret']:.2f} "
                f"CI[{ablation_rows[-1]['ci95_low']:.2f},{ablation_rows[-1]['ci95_high']:.2f}]",
                flush=True,
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Slim seed vectors in ablations only; keep in main for reproducibility but cap file size
    payload = {
        "meta": {
            "seeds": args.seeds,
            "steps": args.steps,
            "n_boot": N_BOOT,
            "pi_star": 0.7,
            "world": "sticky_context",
            "channels": channels,
            "note": "Paper-2 sticky world with bootstrap CIs and optional bursty loss.",
        },
        "rows": rows,
        "ablations": ablation_rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)

    # Markdown summaries
    md = ROOT / "results" / "delta_regime_sigma1.md"
    lines = [
        "
        "",
        f"Seeds={args.seeds}, steps={args.steps}, n_boot={N_BOOT}. "
        "Δ vs Always-Act: positive = worse than Always-Act; * = 95% CI excludes 0.",
        "",
    ]
    for channel in channels:
        lines += [
            f"## Channel: {channel}",
            "",
            "| p_loss | policy | ranker | regret | CI95 | act | wait | defer | Δ vs AA |",
            "|-------:|:-------|:-------|-------:|-----:|----:|-----:|------:|--------:|",
        ]
        for r in rows:
            if abs(r["sigma"] - 1.0) > 1e-9 or r["channel"] != channel:
                continue
            vs = "—"
            if "vs_always_act" in r:
                d = r["vs_always_act"]
                star = "*" if d["ci_excludes_zero"] else ""
                vs = f"{d['mean_delta']:+.2f}{star}"
            lines.append(
                f"| {r['p_loss']:.2f} | {r['policy']} | {r['ranker']} | "
                f"{r['mean_step_regret']:.2f} | [{r['ci95_low']:.2f},{r['ci95_high']:.2f}] | "
                f"{r['frac_act']:.2f} | {r['frac_wait']:.2f} | {r['frac_defer']:.2f} | {vs} |"
            )
        lines.append("")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md}", flush=True)

    # Compact stats for paper tables
    stats_md = ROOT / "results" / "stats_package.md"
    focus = [
        r
        for r in rows
        if r["ranker"] == "map"
        and abs(r["sigma"] - 1.0) < 1e-9
        and r["p_loss"] in (0.3, 0.7)
        and r["channel"] == "independent"
    ]
    focus04 = [
        r
        for r in rows
        if r["ranker"] == "map"
        and abs(r["sigma"] - 0.4) < 1e-9
        and r["p_loss"] in (0.0, 0.7)
        and r["policy"] == "always_act"
        and r["channel"] == "independent"
    ]
    s_lines = [
        "
        "",
        "## σ=1 independent MAP (headline)",
        "",
    ]
    for r in focus:
        vs = r.get("vs_always_act", {})
        s_lines.append(
            f"- p={r['p_loss']:.2f} {r['policy']}: "
            f"{r['mean_step_regret']:.2f} [{r['ci95_low']:.2f},{r['ci95_high']:.2f}] "
            f"act={r['frac_act']:.2f}"
            + (
                f" ΔAA={vs.get('mean_delta', 0):+.2f} CI{vs.get('ci95')} excl0={vs.get('ci_excludes_zero')}"
                if vs
                else ""
            )
        )
    s_lines += ["", "## σ=0.4 Always-Act MAP (sharp-belief degradation)", ""]
    for r in focus04:
        s_lines.append(
            f"- p={r['p_loss']:.2f}: {r['mean_step_regret']:.2f} "
            f"[{r['ci95_low']:.2f},{r['ci95_high']:.2f}]"
        )
    if any(r["channel"] == "bursty" for r in rows):
        s_lines += ["", "## Bursty (Gilbert–Elliott) σ=1 MAP p=0.7", ""]
        for r in rows:
            if (
                r["channel"] == "bursty"
                and abs(r["sigma"] - 1.0) < 1e-9
                and r["ranker"] == "map"
                and abs(r["p_loss"] - 0.7) < 1e-9
            ):
                s_lines.append(
                    f"- {r['policy']}: {r['mean_step_regret']:.2f} "
                    f"[{r['ci95_low']:.2f},{r['ci95_high']:.2f}] act={r['frac_act']:.2f}"
                )
    stats_md.write_text("\n".join(s_lines) + "\n", encoding="utf-8")
    print(f"Wrote {stats_md}", flush=True)

if __name__ == "__main__":
    main()
