from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from belief_age.age_gate import GateConfig
from belief_age.run_codesign_package import run_episode

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--p-loss", type=float, default=0.5)
    parser.add_argument("--lambda-act", type=float, default=20.0)
    args = parser.parse_args()

    delta_stars = [1, 2, 3, 4]
    pi_stars = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    grid_regret = np.zeros((len(delta_stars), len(pi_stars)))
    grid_act = np.zeros_like(grid_regret)
    grid_j = np.zeros_like(grid_regret)

    rows = []
    for i, dstar in enumerate(delta_stars):
        for j, pistar in enumerate(pi_stars):
            regrets, acts = [], []
            cfg = GateConfig(policy="peak_and_age", delta_star=dstar, pi_star=pistar)
            for seed in range(args.seeds):
                out = run_episode(
                    sigma=args.sigma,
                    p_loss=args.p_loss,
                    seed=seed,
                    gate_cfg=cfg,
                    ranker="map",
                    max_steps=args.steps,
                )
                regrets.append(out["mean_step_regret"])
                acts.append(out["frac_act"])
            mean_r = float(np.mean(regrets))
            mean_a = float(np.mean(acts))
            jcost = mean_r + args.lambda_act * mean_a
            grid_regret[i, j] = mean_r
            grid_act[i, j] = mean_a
            grid_j[i, j] = jcost
            rows.append(
                {
                    "delta_star": dstar,
                    "pi_star": pistar,
                    "mean_regret": mean_r,
                    "mean_act": mean_a,
                    "J_l20": jcost,
                }
            )
            print(f"Δ*={dstar} π*={pistar:.2f}  R={mean_r:.2f}  Act={mean_a:.2f}  J={jcost:.2f}")

    best = min(rows, key=lambda r: r["J_l20"])
    payload = {
        "sigma": args.sigma,
        "p_loss": args.p_loss,
        "seeds": args.seeds,
        "lambda_act": args.lambda_act,
        "delta_stars": delta_stars,
        "pi_stars": pi_stars,
        "grid_J": grid_j.tolist(),
        "grid_regret": grid_regret.tolist(),
        "grid_act": grid_act.tolist(),
        "rows": rows,
        "best": best,
        "default_J": next(
            r["J_l20"] for r in rows if r["delta_star"] == 1 and abs(r["pi_star"] - 0.7) < 1e-9
        ),
    }
    out_json = ROOT / "results" / "threshold_sensitivity.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"best={best}  default(Δ*=1,π*=0.7) J={payload['default_J']:.2f}")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    im = ax.imshow(grid_j, aspect="auto", cmap="viridis_r", origin="upper")
    ax.set_xticks(range(len(pi_stars)))
    ax.set_xticklabels([f"{p:.2f}".lstrip("0") for p in pi_stars], rotation=40, ha="right")
    ax.set_yticks(range(len(delta_stars)))
    ax.set_yticklabels([str(d) for d in delta_stars])
    ax.set_xlabel(r"Belief peak threshold $\pi^\star$")
    ax.set_ylabel(r"Age threshold $\Delta^\star$")
    ax.set_title(rf"Operator cost $J(\lambda={int(args.lambda_act)})$ — Peak-and-age")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.ax.tick_params(labelsize=9)
    # mark default
    di = delta_stars.index(1)
    pj = pi_stars.index(0.70)
    ax.plot(pj, di, marker="o", markersize=9, markerfacecolor="none", markeredgecolor="white", markeredgewidth=2)
    ax.annotate("default", (pj, di), textcoords="offset points", xytext=(6, -10), color="white", fontsize=11)
    fig.tight_layout()
    fig_dir = ROOT / "figures" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf = fig_dir / "paper2_threshold_heatmap.pdf"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(fig_dir / "paper2_threshold_heatmap.png", dpi=160, bbox_inches="tight", pad_inches=0.08)
    print(f"wrote {pdf}")

if __name__ == "__main__":
    main()
