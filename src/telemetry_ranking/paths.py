from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_output_dir() -> Path:
    path = Path(
        os.environ.get(
            "TELEMETRY_RANKING_OUTPUT_DIR",
            REPO_ROOT / "results" / "runs" / "latest",
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def figure_output_dir() -> Path:
    path = Path(
        os.environ.get(
            "TELEMETRY_RANKING_FIGURE_DIR",
            run_output_dir() / "figures",
        )
    )
    path.mkdir(parents=True, exist_ok=True)
    return path
