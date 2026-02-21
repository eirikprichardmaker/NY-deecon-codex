# Strategy A Recommendation

## Decision Rule
1. Prefer higher holdout_excess_return_net (fallback: cumulative_return_net).
2. Penalize worse drawdown, turnover, and pct_cash vs baseline.
3. Require stability: at least 3 seeds with the same direction vs baseline.

## Result
- decision: **tighten regularization/penalties**

## Why
- seeds_evaluated=5, improve=0, degrade=0, flat=5
- stability_count=0 (must be >=3)
- median_delta_excess=0.000000
- mean_delta_excess=0.000000

## Seed-Level Table
| seed | run_id | holdout_excess_return_net | holdout_max_dd | holdout_turnover | holdout_pct_cash | delta_holdout_excess_return_net | delta_holdout_max_dd | delta_holdout_turnover | delta_holdout_pct_cash | delta_utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | wft_opt_20260219_202532 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | wft_opt_20260219_203426 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | wft_opt_20260219_204305 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | wft_opt_20260219_205204 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | wft_opt_20260219_210109 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
