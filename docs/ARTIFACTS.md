# Experiment artifact map

Committed reference JSON under `results/reference/` lets reported values be checked without repeating the longest runs.

| Artifact | Generator | Reference input |
|---|---|---|
| Regret vs context noise | `telemetry_ranking.figures` | `regime_map_30seed.json` |
| Hybrid gate sweep | `telemetry_ranking.journal.run_pi_sweep` | `pi_star_sweep.json` |
| Synthetic twin dissociation | `telemetry_ranking.figures` | `highseed_summary.json` |
| Independent loss heatmaps | `telemetry_ranking.journal.run_tnsm_heatmaps` | `tnsm_telemetry_heatmaps.json` |
| Bursty loss heatmaps | `telemetry_ranking.journal.run_tnsm_heatmaps` | `tnsm_telemetry_heatmaps.json` |
| Risk-neutral comparison table | `telemetry_ranking.journal.run_regime_map_highseed` | `regime_map_30seed.json` |
| CVaR and Hybrid table | `telemetry_ranking.journal.run_high_seed_study` | `highseed_stepcount.json` |
| H/D ablation table | `telemetry_ranking.journal.run_hd_ablation` | `hd_ablation.json` |
| Low-confidence specialised-prompt table | `telemetry_ranking.journal.run_proxy_kpis` | `proxy_kpis.json` |
| Independent and bursty loss table | `telemetry_ranking.journal.run_tnsm_heatmaps` | `tnsm_telemetry_heatmaps.json` |
| SLA latency table | `telemetry_ranking.journal.run_systems_repeated` | `systems_repeated.json` |

## Reference result policy

Files under `results/reference/` are immutable experiment artifacts. New runs belong under `results/runs/`. The reproduction runner sets output paths for each experiment so reference files are not replaced.

`scripts/verify_results.py` checks the corresponding regenerated JSON before a release.
