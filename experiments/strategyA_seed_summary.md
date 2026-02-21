# Strategy A Seed Summary

## Filters
- mode: `strategyA`
- holdout_start/holdout_end: `2021` / `2022`
- seeds: `1,2,3,4,5`
- latest: `10`

## Baseline
- baseline_run_id: `wft_opt_baseline_20260219_223428`
- cumulative_return_net: 0.000000
- holdout_excess_return_net: 0.000000
- holdout_max_dd: 0.000000
- holdout_turnover: 0.000000
- holdout_pct_cash: 1.000000

## Ranked Table
| rank | seed | run_id | holdout_excess_return_net | holdout_max_dd | holdout_turnover | holdout_pct_cash | delta_holdout_excess_return_net | delta_holdout_max_dd | delta_holdout_turnover | delta_holdout_pct_cash | delta_utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | wft_opt_20260219_202532 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | 2 | wft_opt_20260219_203426 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | 3 | wft_opt_20260219_204305 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | 4 | wft_opt_20260219_205204 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | 5 | wft_opt_20260219_210109 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Notes
- `delta_*` values are `StrategyA - Baseline`.
- Ranking uses holdout utility = excess - 0.50*abs(max_dd) - 0.10*turnover - 0.05*pct_cash.
