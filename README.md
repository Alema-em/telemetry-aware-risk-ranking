# Telemetry-aware risk-sensitive ranking

This repository contains the code, reference results and manuscript sources for a simulation-first study of intervention ranking under unreliable telemetry.

The study holds the intervention candidates and simulator utility fixed. It compares MAP collapse, soft-belief ranking, CVaR ranking and a belief-peak-gated Hybrid. Monitoring stress includes context noise, independent packet loss and Gilbert–Elliott bursty loss.

## Main result

Risk-neutral soft belief and learning-to-rank do not improve on MAP in this ranking interface. At high context noise, CVaR and the Hybrid reduce policy regret by preferring robust interventions when the belief is diffuse. The effect reverses when habituation and disengagement are frozen, which is consistent with trajectory mediation.

## Repository layout

```text
configs/              Frozen experiment settings
models/               Trained LightGBM rankers
paper/                Manuscript, bibliography and final figures
results/reference/    Reference JSON artifacts used in the paper
scripts/              Reproduction and verification commands
src/telemetry_ranking Core rankers and journal experiments
tests/                Deterministic unit and schema tests
```

## Quick start

Python 3.12 is required.

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts\reproduce.py smoke
.\.venv\Scripts\python.exe -m pytest
```

Linux or macOS:

```bash
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python scripts/reproduce.py smoke
./.venv/bin/python -m pytest
```

See [REPRODUCE.md](REPRODUCE.md) for the paper and full profiles.

## Data statement

All oracle training data and experiment traces are simulator-generated. The ESP32 records discussed in the manuscript use synthetic device telemetry for gateway timing. This repository contains no participant data, protected health information or device credentials.

## Reproducibility scope

Policy metrics are deterministic under the recorded seeds and software versions. Wall-clock latency depends on the host, operating system and scheduler. Latency artifacts are provided as reference measurements rather than cross-machine equality targets.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). The manuscript is under review.

## License

Code in this repository is released under the MIT License. Third-party notices are listed in [NOTICE.md](NOTICE.md).
