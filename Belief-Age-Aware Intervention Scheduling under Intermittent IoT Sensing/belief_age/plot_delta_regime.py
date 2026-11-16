from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "results" / "delta_regime.json"
OUT_DIR = ROOT / "figures"

def _plot(rows: list[dict], sigma: float, channel: str, outfile: str, title: str) -> None:
    sub0 = [
        r
        for r in rows
        if abs(r["sigma"] - sigma) < 1e-9
        and r["ranker"] == "map"
        and r.get("channel", "independent") == channel
    ]
    policies = [
        ("always_act", "#1F4E79", "o", "Always-Act"),
        ("age_wait", "#C45C26", "s", "Age-Wait"),
        ("peak_and_age", "#2F6F4E", "D", r"Peak-and-age ($\Delta^\star{=}1$)"),
        ("peak_and_age_d2", "#6B4C9A", "^", r"Peak-and-age ($\Delta^\star{=}2$)"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=200)
    for pol, color, marker, label in policies:
        sub = sorted([r for r in sub0 if r["policy"] == pol], key=lambda r: r["p_loss"])
        if not sub:
            continue
        xs = [r["p_loss"] for r in sub]
        ys = [r["mean_step_regret"] for r in sub]
        yerr = [
            [r["mean_step_regret"] - r["ci95_low"] for r in sub],
            [r["ci95_high"] - r["mean_step_regret"] for r in sub],
        ]
        ax.errorbar(
            xs, ys, yerr=yerr, marker=marker, color=color, lw=2.0, markersize=7, capsize=3, label=label
        )
    ax.set_xlabel(r"Monitoring packet-loss probability $p_{\mathrm{loss}}$")
    ax.set_ylabel("Mean per-step policy regret")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        path = OUT_DIR / (outfile + ext)
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(ROOT / "results" / (outfile + ext), bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)

def main() -> None:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = data["rows"]
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "pdf.fonttype": 42,
            "font.family": "DejaVu Sans",
        }
    )
    _plot(rows, 1.0, "independent", "paper2_regret_vs_ploss", r"$\sigma=1$, independent loss (MAP)")
    _plot(rows, 0.4, "independent", "paper2_regret_vs_ploss_sigma04", r"$\sigma=0.4$, independent loss (MAP)")
    _plot(rows, 1.0, "bursty", "paper2_regret_vs_ploss_bursty", r"$\sigma=1$, bursty loss (MAP)")

if __name__ == "__main__":
    main()
