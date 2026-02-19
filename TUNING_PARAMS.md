# Tunable Parameters in Current Deecon WFT Pipeline

## Where `chosen_params` is created
- `src/wft.py` in `run_wft(...)` writes per-fold rows with `chosen_params` (JSON from `WFTParams`).
- `WFTParams` currently contains exactly:
  - `mos_threshold`
  - `mad_min`
  - `weakness_rule_variant`

## Parameters currently tunable in WFT

| Name | Type | Valid values / range (today) | Used in code |
| --- | --- | --- | --- |
| `mos_threshold` | `float` | Grid values used in WFT: `0.30, 0.35, 0.40, 0.45` (sweep can use subset) | `src/wft.py` `_apply_filters(...)`: base MoS requirement |
| `mad_min` | `float` | Grid values used in WFT: `-0.05, -0.02, 0.00, 0.02` (sweep can use subset) | `src/wft.py` `_apply_filters(...)`: stock/index MAD filter |
| `weakness_rule_variant` | `str` | Implemented variants: `baseline`, `stricter` | `src/wft.py` `_apply_filters(...)`: `weak_fail_min = 2` for `baseline`, otherwise `1` |

## Rules that decide `CASH` vs `CANDIDATE` in WFT
- `src/wft.py` `_pick_ticker(...)`: returns `CASH` if no eligible ticker after filters.
- `src/wft.py` `_apply_filters(...)` eligibility requires:
  - `mos >= mos_req`
  - `value_creation_ok_base == True`
  - `quality_weak_count < weak_fail_min`
  - stock `above_ma200 == True`
  - stock `mad >= mad_min`
  - index data present + index `above_ma200 == True`
  - index `mad >= mad_min`
- `src/wft.py` enforces higher MoS floor for high-risk names:
  - `mos_req = max(0.40, mos_threshold)` when `high_risk_flag == True`.

## Related thresholds in `decision.py` (config-driven, not in WFT `chosen_params`)

These affect weekly decision output, and mirror/extend WFT concepts:
- `decision.mos_min` (default `0.30`) and `decision.mos_high_uncertainty` (default `0.40`) in `src/decision.py`.
- Technical filters:
  - `decision.mad_min` (default `-0.05`)
  - `decision.require_above_ma200` (default `True`)
  - `decision.require_index_ma200` (default `True`)
  - `decision.require_index_mad` (default `True`)
- Quality gate:
  - `decision.quality_weak_fail_min` (default `2`)
  - plus component thresholds: `quality_roic_min`, `quality_fcf_yield_min`, `quality_nd_ebitda_max`, `quality_ev_ebit_max`
- Value creation persistence:
  - `decision.value_creation_spread_decay_per_year` (default `0.01`)

## Notes for sweep scope (conservative)
- The only parameters that are both implemented in WFT tuning and exported in `chosen_params` are:
  - `mos_threshold`, `mad_min`, `weakness_rule_variant`.
- Literal variant `"strict"` is not implemented; implemented stricter mode is `"stricter"`.
