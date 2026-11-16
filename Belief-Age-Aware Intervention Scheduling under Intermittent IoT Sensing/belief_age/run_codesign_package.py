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
from belief_age.age_gate import (
    BeliefAgeTracker,
    GateConfig,
    GateDecision,
    belief_peak,
    composite_cost,
    decide_gate,
)
from belief_age.sticky_world import StickyWorld

ACTION_WAIT = 0
ACTION_DEFER = 1

def _choose_act(world: StickyWorld, ranker: str, probs: np.ndarray) -> int:
    if ranker == "cvar":
        return world.choose_cvar(probs, alpha=0.3)
    return world.choose_map(probs)

def run_episode(
    sigma: float,
    p_loss: float,
    seed: int,
    gate_cfg: GateConfig,
    ranker: str = "map",
    max_steps: int = 80,
    channel_kind: str = "independent",
    predict_on_miss: bool | None = None,
    delivery_mask: list[bool] | None = None,
) -> dict:
    if predict_on_miss is None:
        predict_on_miss = gate_cfg.policy != "always_act_frozen"

    world = StickyWorld(sigma=sigma, seed=seed)
    world.reset(seed=seed)
    rng = np.random.default_rng(seed + 17)
    channel = TelemetryChannel(kind=channel_kind, p_loss=p_loss, rng=rng)  # type: ignore[arg-type]
    channel.reset()
    tracker = BeliefAgeTracker()

    step_regrets: list[float] = []
    gate_counts = {g.value: 0 for g in GateDecision}
    ages: list[int] = []

    for t in range(max_steps):
        if delivery_mask is not None:
            delivered = bool(delivery_mask[t % len(delivery_mask)])
        else:
            delivered = channel.delivered()

        if delivered:
            tracker.on_update(world.belief_probs())
        else:
            tracker.on_stale()

        belief_for_gate = world.belief_probs()
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

        out = world.step(
            int(action),
            monitoring_delivered=delivered,
            predict_on_miss=predict_on_miss,
        )
        if delivered:
            tracker.last_probs = world.belief_probs()
        step_regrets.append(float(out["regret"]))

    n = max(len(step_regrets), 1)
    mean_regret = float(np.mean(step_regrets))
    frac_act = gate_counts["act"] / n
    frac_wait = gate_counts["wait"] / n
    frac_defer = gate_counts["defer"] / n
    if gate_cfg.policy == "peak_and_age" and gate_cfg.delta_star != 1:
        policy_label = f"peak_and_age_d{gate_cfg.delta_star}"
    else:
        policy_label = gate_cfg.policy
    return {
        "sigma": sigma,
        "p_loss": p_loss,
        "seed": seed,
        "policy": policy_label,
        "delta_star": int(gate_cfg.delta_star),
        "ranker": ranker,
        "predict_on_miss": predict_on_miss,
        "mean_step_regret": mean_regret,
        "mean_age": float(np.mean(ages)) if ages else 0.0,
        "frac_act": frac_act,
        "frac_wait": frac_wait,
        "frac_defer": frac_defer,
        "cost_default": composite_cost(mean_regret, frac_act, frac_wait, frac_defer),
        "pareto_cost_l10": composite_cost(
            mean_regret, frac_act, frac_wait, frac_defer, lambda_act=10.0, lambda_wait=0.0, lambda_defer=0.0
        ),
        "pareto_cost_l20": composite_cost(
            mean_regret, frac_act, frac_wait, frac_defer, lambda_act=20.0, lambda_wait=0.0, lambda_defer=0.0
        ),
        "steps": len(step_regrets),
    }

def load_delivery_mask_from_jsonl(path: Path, max_steps: int = 200) -> list[bool]:
    """Build a delivery mask from logged seq gaps (True = packet delivered)."""
    if not path.exists():
        return []
    seqs: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        tel = row.get("telemetry") or {}
        if "seq" in tel:
            seqs.append(int(tel["seq"]))
    if len(seqs) < 2:
        return []
    seqs = sorted(set(seqs))
    lo, hi = seqs[0], seqs[-1]
    delivered_set = set(seqs)
    mask = [s in delivered_set for s in range(lo, hi + 1)]
    if len(mask) < max_steps:
        # tile
        reps = (max_steps // len(mask)) + 1
        mask = (mask * reps)[:max_steps]
    return mask[:max_steps]

def synthetic_duty_mask(skip_every: int = 3, length: int = 200) -> list[bool]:
    """Software duty-cycle mask matching ESP MODE_DUTY (publish 1 of N)."""
    return [(i % skip_every) == 0 for i in range(length)]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--p-loss", type=float, default=0.5)
    parser.add_argument(
        "--jsonl",
        type=str,
        default=str(ROOT / "results" / "iot_live_decisions.jsonl"),
    )
    parser.add_argument("--out", type=str, default=str(ROOT / "results" / "codesign.json"))
    args = parser.parse_args()

    policies: list[tuple[str, GateConfig]] = [
        ("always_act", GateConfig(policy="always_act")),
        ("always_act_frozen", GateConfig(policy="always_act_frozen")),
        ("age_wait", GateConfig(policy="age_wait", delta_star=1)),
        ("peak_and_age", GateConfig(policy="peak_and_age", delta_star=1, pi_star=0.7)),
        ("peak_and_age_d2", GateConfig(policy="peak_and_age", delta_star=2, pi_star=0.7)),
    ]

    rows = []
    for label, cfg in policies:
        eps = [
            run_episode(args.sigma, args.p_loss, s, cfg, "map", args.steps)
            for s in range(args.seeds)
        ]
        regrets = [e["mean_step_regret"] for e in eps]
        acts = [e["frac_act"] for e in eps]
        waits = [e["frac_wait"] for e in eps]
        defers = [e["frac_defer"] for e in eps]
        costs = [e["cost_default"] for e in eps]
        p10 = [e["pareto_cost_l10"] for e in eps]
        p20 = [e["pareto_cost_l20"] for e in eps]
        row = {
            "label": label,
            "sigma": args.sigma,
            "p_loss": args.p_loss,
            "n_seeds": args.seeds,
            "mean_regret": float(np.mean(regrets)),
            "sem_regret": float(np.std(regrets) / np.sqrt(args.seeds)),
            "mean_act": float(np.mean(acts)),
            "mean_wait": float(np.mean(waits)),
            "mean_defer": float(np.mean(defers)),
            "mean_cost_annoyance": float(np.mean(costs)),
            "mean_pareto_l10": float(np.mean(p10)),
            "mean_pareto_l20": float(np.mean(p20)),
            "channel": "independent",
        }
        rows.append(row)
        print(
            f"{label:18s} regret={row['mean_regret']:.2f} act={row['mean_act']:.2f} "
            f"cost={row['mean_cost_annoyance']:.2f} pareto10={row['mean_pareto_l10']:.2f} "
            f"pareto20={row['mean_pareto_l20']:.2f}",
            flush=True,
        )

    # Pareto sweep over lambda for peak vs always_act
    lambdas = [0, 5, 10, 15, 20, 30, 40]
    pareto = []
    for lam in lambdas:
        for label, cfg in [
            ("always_act", GateConfig(policy="always_act")),
            ("peak_and_age", GateConfig(policy="peak_and_age", delta_star=1, pi_star=0.7)),
            ("peak_and_age_d2", GateConfig(policy="peak_and_age", delta_star=2, pi_star=0.7)),
            ("age_wait", GateConfig(policy="age_wait", delta_star=1)),
        ]:
            eps = [
                run_episode(args.sigma, args.p_loss, s, cfg, "map", args.steps)
                for s in range(min(args.seeds, 30))
            ]
            j = [
                composite_cost(
                    e["mean_step_regret"],
                    e["frac_act"],
                    e["frac_wait"],
                    e["frac_defer"],
                    lambda_act=lam,
                    lambda_wait=0.0,
                    lambda_defer=0.0,
                )
                for e in eps
            ]
            pareto.append(
                {
                    "lambda_act": lam,
                    "policy": label,
                    "mean_J": float(np.mean(j)),
                    "mean_regret": float(np.mean([e["mean_step_regret"] for e in eps])),
                    "mean_act": float(np.mean([e["frac_act"] for e in eps])),
                }
            )
        print(f"pareto λ={lam}: done", flush=True)

    # Hardware / duty replay
    jsonl = Path(args.jsonl)
    mask = load_delivery_mask_from_jsonl(jsonl, max_steps=args.steps)
    if not mask:
        mask = synthetic_duty_mask(skip_every=3, length=args.steps)
        replay_source = "synthetic_duty_skip3"
    else:
        replay_source = str(jsonl)

    replay_rows = []
    for label, cfg in policies:
        eps = [
            run_episode(
                args.sigma,
                0.0,
                s,
                cfg,
                "map",
                args.steps,
                delivery_mask=mask,
            )
            for s in range(args.seeds)
        ]
        replay_rows.append(
            {
                "label": label,
                "replay_source": replay_source,
                "empirical_delivery_rate": float(np.mean(mask)),
                "mean_regret": float(np.mean([e["mean_step_regret"] for e in eps])),
                "mean_act": float(np.mean([e["frac_act"] for e in eps])),
                "mean_cost_annoyance": float(np.mean([e["cost_default"] for e in eps])),
                "mean_pareto_l10": float(np.mean([e["pareto_cost_l10"] for e in eps])),
            }
        )
        print(
            f"replay {label:18s} regret={replay_rows[-1]['mean_regret']:.2f} "
            f"act={replay_rows[-1]['mean_act']:.2f} cost={replay_rows[-1]['mean_cost_annoyance']:.2f}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "framing": "co-design / operator guideline under Act budget",
            "sigma": args.sigma,
            "p_loss": args.p_loss,
            "seeds": args.seeds,
            "steps": args.steps,
            "cost": "regret + λ_act·Act + λ_wait·Wait + λ_defer·Defer (defaults 10/8/2)",
            "baselines": "always_act (predictive age); always_act_frozen (no predict on miss)",
        },
        "rows": rows,
        "pareto": pareto,
        "replay": replay_rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)

    # Markdown + Pareto figure
    md = ROOT / "results" / "codesign.md"
    lines = [
        "
        "",
        f"σ={args.sigma}, p_loss={args.p_loss}, seeds={args.seeds}.",
        "",
        "## Headline (independent loss)",
        "",
        "| policy | regret | act | wait | defer | cost(annoy) | J(λ=10) | J(λ=20) |",
        "|:-------|-------:|----:|-----:|------:|------------:|--------:|--------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['mean_regret']:.2f} | {r['mean_act']:.2f} | "
            f"{r['mean_wait']:.2f} | {r['mean_defer']:.2f} | {r['mean_cost_annoyance']:.2f} | "
            f"{r['mean_pareto_l10']:.2f} | {r['mean_pareto_l20']:.2f} |"
        )
    lines += ["", f"## Replay ({replay_source})", ""]
    for r in replay_rows:
        lines.append(
            f"- {r['label']}: regret={r['mean_regret']:.2f} act={r['mean_act']:.2f} "
            f"cost={r['mean_cost_annoyance']:.2f} J10={r['mean_pareto_l10']:.2f}"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {md}", flush=True)

    # Plot Pareto
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=200)
        styles = {
            "always_act": ("#1F4E79", "o"),
            "peak_and_age": ("#2F6F4E", "D"),
            "peak_and_age_d2": ("#6B4C9A", "^"),
            "age_wait": ("#C45C26", "s"),
        }
        for pol, (color, marker) in styles.items():
            sub = [p for p in pareto if p["policy"] == pol]
            if not sub:
                continue
            ax.plot(
                [p["lambda_act"] for p in sub],
                [p["mean_J"] for p in sub],
                marker=marker,
                color=color,
                lw=2,
                label=pol,
            )
        ax.set_xlabel(r"Act-budget weight $\lambda$ in $J=$regret$+\lambda\cdot$Act-rate")
        ax.set_ylabel(r"Mean operator cost $J$")
        ax.set_title(rf"Co-design Pareto at $\sigma={args.sigma}$, $p_{{\mathrm{{loss}}}}={args.p_loss}$")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        out_dir = ROOT / "figures" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in ("paper2_pareto_cost.pdf", "paper2_pareto_cost.png"):
            fig.savefig(out_dir / name, bbox_inches="tight")
            fig.savefig(ROOT / "results" / name, bbox_inches="tight")
            print(f"Wrote {out_dir / name}")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"plot skipped: {exc}")

if __name__ == "__main__":
    main()
