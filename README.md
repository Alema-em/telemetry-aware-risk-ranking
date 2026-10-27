# Telemetry-aware risk-sensitive ranking

This repository contains the code, reference results and manuscript for a simulation study of intervention ranking under unreliable telemetry.

The intervention candidates and the one-step utility stay fixed. The comparison covers MAP collapse, soft-belief ranking, CVaR ranking and a Hybrid gate on the belief peak. The monitoring conditions are context noise, independent packet loss and Gilbert–Elliott bursty loss. Context noise is the StepCountJITAI observation noise. Packet loss drops an already formed belief update.

## Main result

Risk-neutral soft belief and learning-to-rank do not improve on MAP in this ranking interface. At high context noise, CVaR and Hybrid lower policy regret by choosing the robust prompt when the belief is spread out. Freezing habituation and disengagement reverses that gain. The episode-level change follows the closed-loop trajectory.

At the highest noise level and over 30 seeds, CVaR lowers mean per-step regret by 24.89 units and Hybrid at a belief-peak gate of 0.7 lowers it by 26.09 units.

## Repository layout

```text
configs/              Frozen experiment settings
models/               Trained LightGBM rankers
paper/                Manuscript, bibliography and figures
results/reference/    Reference JSON used in the paper
scripts/              Reproduction and verification commands
src/telemetry_ranking Core rankers and experiments
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

Oracle training rows and experiment traces are produced by the simulator. The ESP32 records discussed in the manuscript use synthetic device telemetry for gateway timing. This repository contains no participant data, protected health information or device credentials.

## Reproducibility scope

Policy metrics are deterministic under the recorded seeds and software versions. Wall-clock latency depends on the host, the operating system and the scheduler. Latency files are reference measurements. They are not cross-machine equality targets.

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff). The manuscript is under review. The title is Telemetry-Aware Risk-Sensitive Ranking for Intervention Service Management.

## License

Code in this repository is released under the MIT License. Third-party notices are in [NOTICE.md](NOTICE.md).
