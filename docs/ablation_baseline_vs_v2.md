# Ablation: baseline vs v2 decision rules

Samme datasett/freeze:
- master_valued: `G:\Min disk\NEW DEECON\data\processed\master_valued.parquet`
- prices_panel: `G:\Min disk\NEW DEECON\data\raw\prices\prices_panel.parquet`

Endringer:
- #1 Kvalitetsgate: >=2 svekkede kvalitetsindikatorer => ikke kvalifisert
- #2 Verdiskapingsgate: ROIC > WACC i 3-ar konservativ bane

## Metrics
| Metric | Baseline | #1 only | #2 only | #1+#2 |
| --- | ---: | ---: | ---: | ---: |
| periods | 175.0000 | 175.0000 | 175.0000 | 175.0000 |
| hit_rate | 56.63% | 56.36% | 57.89% | 57.89% |
| max_drawdown | -66.50% | -66.50% | -67.92% | -67.92% |
| sharpe_ann | 0.7058 | 0.6942 | 0.6910 | 0.6910 |
| deflated_sharpe_prob | 100.00% | 100.00% | 100.00% | 100.00% |
| turnover | 25.86% | 25.86% | 24.71% | 24.71% |
| share_cash | 5.14% | 5.71% | 13.14% | 13.14% |
| selection_stability | 74.14% | 74.14% | 75.29% | 75.29% |
| terminal_nav | 19.2007 | 17.9771 | 17.5060 | 17.5060 |

Artefakter:
- `G:\Min disk\NEW DEECON\reports\ablation_baseline_vs_v2_metrics.csv`
- `G:\Min disk\NEW DEECON\reports\ablation_baseline_vs_v2_periods.csv`
