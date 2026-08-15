from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "results" / "reference"
SPECS = {
    "regime_map_30seed.json": {
        "rows": "aggregated",
        "keys": ("ranker", "sigma"),
        "metrics": ("mean_step_regret", "mean_episode_reward"),
        "atol": 0.1,
    },
    "highseed_stepcount.json": {
        "rows": "aggregated",
        "keys": ("method", "sigma"),
        "metrics": ("mean_step_regret",),
        "atol": 0.1,
    },
    "pi_star_sweep.json": {
        "rows": "aggregated",
        "keys": ("kind", "sigma", "pi_star"),
        "metrics": ("mean_step_regret", "mean_episode_reward"),
        "atol": 0.2,
    },
    "tnsm_telemetry_heatmaps.json": {
        "rows": "aggregated",
        "keys": ("sigma", "channel", "p_loss", "alpha", "method"),
        "metrics": ("mean_step_regret", "mean_episode_reward", "sla_violation_rate"),
        "atol": 0.2,
    },
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def index(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, dict]:
    return {tuple(row.get(key) for key in keys): row for row in rows}


def verify_file(actual_path: Path, reference_path: Path, spec: dict) -> list[str]:
    actual = index(load(actual_path)[spec["rows"]], spec["keys"])
    expected = index(load(reference_path)[spec["rows"]], spec["keys"])
    errors = []
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys(), key=str)
        extra = sorted(actual.keys() - expected.keys(), key=str)
        errors.append(f"{actual_path.name}: missing={missing} extra={extra}")
        return errors
    for key in expected:
        for metric in spec["metrics"]:
            delta = abs(float(actual[key][metric]) - float(expected[key][metric]))
            if delta > spec["atol"]:
                errors.append(
                    f"{actual_path.name} {key} {metric}: "
                    f"delta={delta:.6g} tolerance={spec['atol']}"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()

    errors = []
    checked = 0
    for name, spec in SPECS.items():
        actual = args.results / name
        reference = args.reference / name
        if not actual.exists():
            errors.append(f"missing result: {actual}")
            continue
        errors.extend(verify_file(actual, reference, spec))
        checked += 1
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"verified {checked} result files")


if __name__ == "__main__":
    main()
