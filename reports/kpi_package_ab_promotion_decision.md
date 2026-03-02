# KPI Package A/B Decision (2026-03-01)

## Inputs
- WFT current_21: `runs/wft_sweep_kpi_current_21_20260301/sweep_summary.csv`
- WFT core_26: `runs/wft_sweep_kpi_core_26_20260301/sweep_summary.csv`
- Coverage: `tmp/kpi_package_ab/variant_coverage_summary.csv`
- Ablation summary: `reports/ablation_kpi_package_summary.csv`

## Best Seed Metrics
| set_name | cagr_net | excess_cagr | info_ratio | max_dd | turnover | cash_share | go_no_go | go_no_go_reasons | folds | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_21 | 0.2294782473261409 | 0.2038576022329055 | 1.052248216399802 | -0.1499999999999999 | 0.1 | 0.0 | NO_GO | maxdd_vs_benchmark | 3 | 11 |
| core_26 | 0.2294782473261409 | 0.2038576022329055 | 1.052248216399802 | -0.1499999999999999 | 0.1 | 0.0 | NO_GO | maxdd_vs_benchmark | 3 | 11 |

## Promotion Rule
- Rule: promote only if OOS improves and drawdown/turnover are not worse.
- OOS improved (cagr_net): False (0.229478247326 vs 0.229478247326)
- Drawdown not worse (abs max_dd): True (0.150000000000 vs 0.150000000000)
- Turnover not worse: True (0.100000000000 vs 0.100000000000)
- Final decision: DO_NOT_PROMOTE

## Coverage Snapshot
| set_name | rows | mapped_ins_rows | roic_cov | ev_ebit_cov | nd_ebitda_cov | fcf_yield_cov | market_cap_cov | out_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_21 | 1741 | 1713 | 0.8897185525560023 | 0.0 | 0.8156232050545663 | 0.9149913842619184 | 0.9798966111430212 | g:\Min disk\NEW DEECON\tmp\kpi_package_ab\current_21\master_valued.parquet |
| core_26 | 1741 | 1713 | 0.8897185525560023 | 0.8925904652498564 | 0.8156232050545663 | 0.9149913842619184 | 0.9798966111430212 | g:\Min disk\NEW DEECON\tmp\kpi_package_ab\core_26\master_valued.parquet |

## Ablation Snapshot
| set_name | baseline_terminal_nav | baseline_sharpe_ann | baseline_max_drawdown | baseline_turnover | baseline_share_cash | baseline_hit_rate | change1_quality_terminal_nav | change1_quality_sharpe_ann | change1_quality_max_drawdown | change1_quality_turnover | change1_quality_share_cash | change1_quality_hit_rate | change2_value_terminal_nav | change2_value_sharpe_ann | change2_value_max_drawdown | change2_value_turnover | change2_value_share_cash | change2_value_hit_rate | v2_terminal_nav | v2_sharpe_ann | v2_max_drawdown | v2_turnover | v2_share_cash | v2_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_21 | 18.839592341946798 | 0.7050515624249831 | -0.5991974629800043 | 0.2471264367816092 | 0.0514285714285714 | 0.5662650602409639 | 17.639024868850314 | 0.693379001243574 | -0.5991974629800043 | 0.2471264367816092 | 0.0571428571428571 | 0.5636363636363636 | 17.50602981560321 | 0.6910238163743025 | -0.6792499762796166 | 0.2471264367816092 | 0.1314285714285714 | 0.5789473684210527 | 17.50602981560321 | 0.6910238163743025 | -0.6792499762796166 | 0.2471264367816092 | 0.1314285714285714 | 0.5789473684210527 |
| core_26 | 18.839592341946798 | 0.7050515624249831 | -0.5991974629800043 | 0.2471264367816092 | 0.0514285714285714 | 0.5662650602409639 | 17.639024868850314 | 0.693379001243574 | -0.5991974629800043 | 0.2471264367816092 | 0.0571428571428571 | 0.5636363636363636 | 17.50602981560321 | 0.6910238163743025 | -0.6792499762796166 | 0.2471264367816092 | 0.1314285714285714 | 0.5789473684210527 | 17.50602981560321 | 0.6910238163743025 | -0.6792499762796166 | 0.2471264367816092 | 0.1314285714285714 | 0.5789473684210527 |
