# Paper artifact map

The manuscript uses committed reference JSON so that reported values can be checked without repeating the longest runs.

| Paper artifact | Generator | Reference input |
|---|---|---|
| Closed-loop architecture | `paper/manuscript.tex` | Inline TikZ |
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

Files under `results/reference/` are immutable paper artifacts. New runs belong under `results/runs/`. The reproduction runner sets output paths for each experiment so reference files are not replaced.

The LaTeX tables are currently typeset in `paper/manuscript.tex`. `scripts/verify_results.py` checks the corresponding regenerated JSON before a release.
