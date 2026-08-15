import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"


def read(name: str) -> dict:
    return json.loads((REFERENCE / name).read_text(encoding="utf-8"))


def test_primary_reference_artifacts_have_expected_shape() -> None:
    regime = read("regime_map_30seed.json")
    highseed = read("highseed_stepcount.json")
    heatmaps = read("tnsm_telemetry_heatmaps.json")
    sweep = read("pi_star_sweep.json")

    assert regime["aggregated"]
    assert highseed["aggregated"]
    assert heatmaps["aggregated"]
    assert sweep["aggregated"]


def test_reference_config_records_seed_counts() -> None:
    assert read("regime_map_30seed.json")["config"]["n_seeds"] == 30
    assert read("highseed_stepcount.json")["config"]["n_seeds"] == 30
    assert read("tnsm_telemetry_heatmaps.json")["config"]["n_seeds"] == 10
