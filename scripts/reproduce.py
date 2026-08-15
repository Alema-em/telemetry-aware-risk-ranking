from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from telemetry_ranking.figures import generate_all
from telemetry_ranking.journal.cvar_rankers import rank_cvar, rank_map_sim
from telemetry_ranking.rank.env_utils import make_env
from telemetry_ranking.rank.rankers import policy_regret


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"
PAPER = ROOT / "paper"
PACKAGES = [
    "StepCountJITAI",
    "gymnasium",
    "lightgbm",
    "matplotlib",
    "numpy",
    "PyYAML",
    "scipy",
]
FULL_MODULES = [
    "telemetry_ranking.journal.run_regime_map_highseed",
    "telemetry_ranking.journal.run_high_seed_study",
    "telemetry_ranking.journal.run_lock_validation",
    "telemetry_ranking.journal.run_hd_ablation",
    "telemetry_ranking.journal.run_proxy_kpis",
    "telemetry_ranking.journal.stats_paired_ci",
    "telemetry_ranking.journal.run_pi_sweep",
    "telemetry_ranking.journal.run_tnsm_heatmaps",
    "telemetry_ranking.journal.run_service_latency_bench",
    "telemetry_ranking.journal.run_systems_package",
    "telemetry_ranking.journal.run_systems_strengthen",
    "telemetry_ranking.journal.run_systems_repeated",
]


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def run_smoke(output: Path) -> None:
    rows = []
    methods = {
        "MAP": rank_map_sim,
        "CVaR_0.3": lambda env: rank_cvar(env, alpha=0.3),
    }
    for sigma in (1.0, 2.0):
        for name, choose in methods.items():
            env = make_env(sigma=sigma, seed=0, n_version=0)
            regrets = []
            rewards = []
            for episode in range(2):
                env.reset()
                env.rng0 = np.random.default_rng(1000 * episode + 1)
                env.rng1 = np.random.default_rng(1000 * episode + 2)
                done = False
                regret = 0.0
                reward = 0.0
                steps = 0
                while not done:
                    action = int(choose(env))
                    regret += policy_regret(env, action)
                    _, value, terminated, truncated, _ = env.step(action)
                    reward += float(value)
                    done = bool(terminated or truncated)
                    steps += 1
                regrets.append(regret / max(steps, 1))
                rewards.append(reward)
            rows.append(
                {
                    "method": name,
                    "sigma": sigma,
                    "seed": 0,
                    "episodes": 2,
                    "mean_step_regret": float(np.mean(regrets)),
                    "mean_episode_reward": float(np.mean(rewards)),
                }
            )
    (output / "smoke_metrics.json").write_text(
        json.dumps({"rows": rows}, indent=2),
        encoding="utf-8",
    )


def run_modules(output: Path, figures: Path) -> None:
    env = os.environ.copy()
    env["TELEMETRY_RANKING_OUTPUT_DIR"] = str(output)
    env["TELEMETRY_RANKING_FIGURE_DIR"] = str(figures)
    for module in FULL_MODULES:
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=ROOT,
            env=env,
            check=True,
        )


def prepare_figures(results: Path, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    for source in (PAPER / "figures").glob("*.pdf"):
        shutil.copy2(source, figures / source.name)
    generate_all(results, figures)


def compile_paper(output: Path, figures: Path) -> None:
    if shutil.which("pdflatex") is None or shutil.which("bibtex") is None:
        raise RuntimeError("pdflatex and bibtex are required for the paper profile")
    build = output / "paper"
    build.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PAPER / "manuscript.tex", build / "manuscript.tex")
    shutil.copy2(PAPER / "references.bib", build / "references.bib")
    target_figures = build / "figures"
    target_figures.mkdir(exist_ok=True)
    for source in figures.glob("*.pdf"):
        shutil.copy2(source, target_figures / source.name)
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
        ["bibtex", "manuscript"],
        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
        ["pdflatex", "-interaction=nonstopmode", "manuscript.tex"],
    ]
    for command in commands:
        subprocess.run(command, cwd=build, check=True, capture_output=True)


def write_manifest(
    output: Path,
    profile: str,
    started: datetime,
    status: str,
) -> None:
    finished = datetime.now(timezone.utc)
    try:
        output_label = str(output.resolve().relative_to(ROOT))
    except ValueError:
        output_label = str(output.resolve())
    manifest = {
        "profile": profile,
        "status": status,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "git_revision": git_revision(),
        "output_dir": output_label,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("smoke", "paper", "full"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "results" / "runs" / f"{args.profile}-{stamp}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    started = datetime.now(timezone.utc)

    try:
        if args.profile == "smoke":
            run_smoke(output)
        elif args.profile == "paper":
            prepare_figures(REFERENCE, figures)
            compile_paper(output, figures)
        else:
            run_modules(output, figures)
            prepare_figures(output, figures)
            compile_paper(output, figures)
    except Exception:
        write_manifest(output, args.profile, started, "failed")
        raise
    write_manifest(output, args.profile, started, "completed")
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
