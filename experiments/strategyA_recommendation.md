# Strategy A Recommendation

## Decision Rule
1. Prefer higher holdout_excess_return_net (fallback: cumulative_return_net).
2. Penalize worse drawdown, turnover, and pct_cash vs baseline.
3. Require stability: at least 3 seeds with the same direction vs baseline.

## Result
- decision: **tighten regularization/penalties**

## Why
- No baseline run available, so deltas are unavailable.
- Defaulting to tighten regularization/penalties until baseline comparison is completed.

## Seed-Level Table
_(empty)_
