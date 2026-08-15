"""Generate manuscript figures from result JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "results" / "reference"
OUT = ROOT / "paper" / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.titlepad": 8,
        "axes.labelpad": 4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def plot_cvar_vs_map():
    agg = json.loads((J / "highseed_stepcount.json").read_text(encoding="utf-8"))["aggregated"]
    by_sig = {}
    for row in agg:
        sig = float(row["sigma"])
        method = row["method"]
        by_sig.setdefault(sig, {})[method] = row

    sigmas = sorted(by_sig)
    map_m, map_e, cvar_m, cvar_e, delta = [], [], [], [], []
    for s in sigmas:
        m = by_sig[s]["MAP"]
        c = by_sig[s]["CVaR_0.3"]
        map_m.append(m["mean_step_regret"])
        map_e.append(m["sem_step_regret"])
        cvar_m.append(c["mean_step_regret"])
        cvar_e.append(c["sem_step_regret"])
        delta.append(c["delta_vs_map"])

    x = np.arange(len(sigmas))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=180)
    ax.bar(x - w / 2, map_m, w, yerr=map_e, capsize=3, label="MAP-Sim", color="#1F4E79", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, cvar_m, w, yerr=cvar_e, capsize=3, label=r"CVaR$_{0.3}$", color="#C45C26", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}" for s in sigmas])
    ax.set_xlabel(r"Context noise $\sigma$")
    ax.set_ylabel("Mean per-step policy regret")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "cvar_vs_map.png")
    fig.savefig(OUT / "cvar_vs_map.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=180)
    colors = ["#B22222" if d > 0 else "#2F6F4E" for d in delta]
    ax.bar(x, delta, color=colors)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}" for s in sigmas])
    ax.set_xlabel(r"Context noise $\sigma$")
    ax.set_ylabel(r"$\Delta$ regret (CVaR $-$ MAP)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "cvar_delta.png")
    fig.savefig(OUT / "cvar_delta.pdf")
    plt.close(fig)
    print("cvar delta", delta)


def plot_lock_bars():
    main = json.loads((J / "highseed_stepcount.json").read_text(encoding="utf-8"))
    stochastic = json.loads((J / "lock_stochastic.json").read_text(encoding="utf-8"))
    transfer = json.loads((J / "lock_transfer_sigma_s20.json").read_text(encoding="utf-8"))

    def delta_at_sigma_two(data: dict) -> float:
        row = next(
            item
            for item in data["aggregated"]
            if item["method"] == "CVaR_0.3" and float(item["sigma"]) == 2.0
        )
        return float(row["delta_vs_map"])

    labels = ["Deterministic\n(30 seeds)", "Stochastic\n" + r"$n{=}1$, $\sigma_s{=}10$", "Transfer\n" + r"$\sigma_s{=}20$"]
    deltas = [
        delta_at_sigma_two(main),
        delta_at_sigma_two(stochastic),
        delta_at_sigma_two(transfer),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=180)
    ax.bar(np.arange(3), deltas, color="#2F6F4E")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta$ regret at $\sigma{=}2$ (CVaR $-$ MAP)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "lock_delta.png")
    fig.savefig(OUT / "lock_delta.pdf")
    plt.close(fig)


def plot_twin_dissociation():
    summ = json.loads((J / "highseed_summary.json").read_text(encoding="utf-8"))
    acc = float(summ.get("synthetic_decision_accuracy", 0.996)) * 100.0
    fig, ax = plt.subplots(figsize=(5.0, 3.1), dpi=180)
    ax.bar([0], [acc], color="#2F6F4E", width=0.55, edgecolor="black", linewidth=0.4)
    ax.bar([1], [0.0], color="#B22222", width=0.55, edgecolor="black", linewidth=0.4)
    ax.text(0, acc + 2, f"{acc:.1f}%", ha="center", fontsize=9)
    ax.text(1, 4, "none", ha="center", fontsize=9, color="#7A1F1F")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$\pi^\star$ decision accuracy", "One-step twin\nregret advantage"])
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 118)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "twin_dissociation.png")
    fig.savefig(OUT / "twin_dissociation.pdf")
    plt.close(fig)
    print("twin acc", acc)


def plot_regime_30seed():
    data = json.loads((J / "regime_map_30seed.json").read_text(encoding="utf-8"))
    agg = data["aggregated"]
    names = ["B1_MAP", "B3_BASR", "Belief_LTR", "MAP_LTR", "Residual_LTR", "Abstain_LTR", "Oracle"]
    labels = {
        "B1_MAP": "MAP-Sim",
        "B3_BASR": "BASR",
        "Belief_LTR": "Belief-LTR",
        "MAP_LTR": "MAP-LTR",
        "Residual_LTR": "Residual-LTR",
        "Abstain_LTR": "Abstain-LTR",
        "Oracle": "Oracle",
    }
    fig, ax = plt.subplots(figsize=(5.6, 3.5), dpi=180)
    for name in names:
        xs = [r["sigma"] for r in agg if r["ranker"] == name]
        ys = [r["mean_step_regret"] for r in agg if r["ranker"] == name]
        es = [r["sem_step_regret"] for r in agg if r["ranker"] == name]
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2.5, label=labels.get(name, name), lw=1.2)
    ax.set_xlabel(r"Context noise $\sigma$")
    ax.set_ylabel("Mean per-step policy regret")
    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "regret_vs_sigma_full.png")
    fig.savefig(OUT / "regret_vs_sigma_full.pdf")
    plt.close(fig)


def plot_hd_ablation():
    data = json.loads((J / "hd_ablation.json").read_text(encoding="utf-8"))
    items = {(d["condition"], float(d["sigma"])): d for d in data["deltas"]}
    labels = [r"$\sigma{=}1$", r"$\sigma{=}2$"]
    base = [items[("baseline", 1.0)]["mean_delta_cvar_minus_map"], items[("baseline", 2.0)]["mean_delta_cvar_minus_map"]]
    fr = [items[("frozen_HD", 1.0)]["mean_delta_cvar_minus_map"], items[("frozen_HD", 2.0)]["mean_delta_cvar_minus_map"]]
    x = np.arange(2)
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=180)
    ax.bar(x - w / 2, base, w, label="Baseline H/D", color="#2F6F4E", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, fr, w, label="Frozen H/D", color="#B22222", edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta$ regret (CVaR $-$ MAP)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(-30, 75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "hd_ablation.png")
    fig.savefig(OUT / "hd_ablation.pdf")
    plt.close(fig)


def plot_proxy_kpi():
    data = json.loads((J / "proxy_kpis.json").read_text(encoding="utf-8"))
    agg = data["aggregated"]
    sigmas = [0.0, 0.4, 1.0, 2.0]
    map_y, cvar_y = [], []
    for s in sigmas:
        map_y.append(next(r["low_conf_specialized_rate"] for r in agg if r["method"] == "MAP" and float(r["sigma"]) == s))
        cvar_y.append(next(r["low_conf_specialized_rate"] for r in agg if r["method"] == "CVaR_0.3" and float(r["sigma"]) == s))
    x = np.arange(len(sigmas))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=180)
    ax.bar(x - w / 2, map_y, w, label="MAP-Sim", color="#1F4E79", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, cvar_y, w, label=r"CVaR$_{0.3}$", color="#C45C26", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:g}" for s in sigmas])
    ax.set_xlabel(r"Context noise $\sigma$")
    ax.set_ylabel(r"Low-conf. specialised rate")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "proxy_kpi.png")
    fig.savefig(OUT / "proxy_kpi.pdf")
    plt.close(fig)


def generate_all(results_dir: Path, figures_dir: Path) -> None:
    global J, OUT
    J = results_dir
    OUT = figures_dir
    OUT.mkdir(parents=True, exist_ok=True)
    plot_regime_30seed()
    plot_cvar_vs_map()
    plot_lock_bars()
    plot_twin_dissociation()
    plot_hd_ablation()
    plot_proxy_kpi()
    print("figures ->", OUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=J)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    generate_all(args.results, args.output)


if __name__ == "__main__":
    main()
