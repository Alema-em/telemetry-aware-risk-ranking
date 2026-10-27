# Reproducing the paper

## Environment

The reference environment used Python 3.12.4 with the versions in `requirements-lock.txt`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The package installs StepCountJITAI 0.0.1 from PyPI. Its upstream source and license are listed in `NOTICE.md`.

## Profiles

### Smoke

Runs two episodes for MAP and CVaR at `sigma=1` and `sigma=2`.

```powershell
.\.venv\Scripts\python.exe scripts\reproduce.py smoke
```

Expected runtime: under one minute.

### Paper

Regenerates JSON-driven figures from the committed reference results and compiles the manuscript.

```powershell
.\.venv\Scripts\python.exe scripts\reproduce.py paper
```

This profile requires `pdflatex` and `bibtex`. Expected runtime: under five minutes.

### Full

Re-runs the scientific experiments, telemetry-loss sweeps and systems benchmarks before rebuilding the manuscript.

```powershell
.\.venv\Scripts\python.exe scripts\reproduce.py full
```

Expected runtime on an eight-core desktop: two to four hours. Telemetry heatmaps use multiprocessing. Latency measurements will vary across machines.

Each run writes to a timestamped folder under `results/runs/`. Reference results are never overwritten. Use `--output PATH` to select another output folder.

## Verification

After a full run:

```powershell
.\.venv\Scripts\python.exe scripts\verify_results.py results\runs\full-TIMESTAMP
```

The verifier checks the risk-neutral comparison, the high-seed study, the Hybrid sweep and the telemetry heatmaps against the committed reference metrics. Deterministic policy metrics use small numeric tolerances. Timing results are excluded.

## Seeds

Primary experiments use seeds `0` through `N-1`. Episode streams are reset with:

```text
rng0 = seed + 1000 * episode + 1
rng1 = seed + 1000 * episode + 2
```

The synthetic twin uses `10000 + seed`. Bootstrap and systems studies use fixed seeds recorded in their scripts.

## Outputs

Every profile writes a `manifest.json` with:

- profile and status
- start time, finish time and elapsed time
- Python and dependency versions
- operating system details
- git revision when available
- relative output path

The figure and table provenance map is in `docs/ARTIFACTS.md`.
