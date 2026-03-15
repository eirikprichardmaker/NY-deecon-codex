# DEECON — Tuning Parameters & Known Risks

Reference for WFT tuning parameters, DQ rule thresholds, and known edge-cases.

---

## DQ Rules (decision.data_quality)

### DQ-EVEBIT-GLOBAL
| Parameter | Default | Config key |
|-----------|---------|-----------|
| EV/EBIT floor | 6.0 | `ev_ebit_floor_global` |
| Severity (positive, below floor) | FAIL | — |
| Severity (negative EBIT) | WARN | — |
| Exception | Banks (`is_bank_proxy=True`) | — |

**Motivation**: Boom-inflated FCF can give high MoS (>80%) even at low EV/EBIT. This
signals mean-reversion risk, not genuine value. Root cause: DEDI.ST (2024, EV/EBIT=5.96,
MoS=94%, actual return -48.5%) and DEVP-B.ST (EV/EBIT=5.88, -24.7%).

Extends the existing cyclical EV/EBIT guard to all non-bank companies.

---

### DQ-MCAP-MIN
| Parameter | Default | Config key |
|-----------|---------|-----------|
| Hard floor (FAIL) | 50,000,000 | `mcap_min_hard` |
| Soft floor (WARN) | 100,000,000 | `mcap_min_warn` |

**Motivation**: Micro-caps (<50M EUR) have unreliable fundamentals, thin liquidity, and
large price swings on single bad quarters. CONSTI.HE (89M EUR, 2024: -12.8%) illustrated
the soft-warn zone.

**Currency assumption**: `market_cap` is stored in local currency (EUR for Finnish/Danish,
SEK for Swedish, NOK for Norwegian). The thresholds are EUR-calibrated. For SEK/NOK stocks,
50M local currency ≈ 4–5M EUR — much lower than intended. This limits the rule's
effectiveness for Nordic non-EUR stocks and is a known limitation.

---

### DQ-MOS-SUSPECT
| Parameter | Default | Config key |
|-----------|---------|-----------|
| MoS threshold | 0.80 | `mos_suspect_threshold` |
| Required EV/EBIT when MoS > threshold | 8.0 | `mos_suspect_evebit_min` |
| Severity | WARN (not blocking alone) | — |

**Motivation**: High MoS (>80%) combined with low EV/EBIT (<8) is the signature of a
3-year normalized FCF including one or more boom years being fed into a perpetuity DCF.
The model sees large upside; the market prices the company cheaply on earnings — suggesting
the FCF level is not sustainable at steady state.

---

## WFT Backtesting — Known Cases

### 2018 Option D Regression vs Option C: AIK-B.ST

**Observation**: Option C 2018 = +23.2%, Option D 2018 = +6.5% (delta −16.7pp). Root cause:
Option C's #1 pick (by quality_score) was **AIK-B.ST** (Consumer Discretionary, Swedish),
which returned +36.1% in 2018. Option D blocked it via two independent DQ rules.

| Field | Value |
|-------|-------|
| Ticker | AIK-B.ST |
| Sector | Consumer Discretionary |
| quality_score (Dec-2017) | 2.59 (rank #1 among eligible) |
| ROIC | 85.6% |
| FCF yield | 20.1% |
| EV/EBIT | 0.261 (near-zero EBIT) |
| market_cap | 41.7M (local SEK — below 50M hard floor) |
| 2018 actual return | +36.1% |
| DQ rules triggered | DQ-EVEBIT-GLOBAL (FAIL), DQ-MCAP-MIN (FAIL) |

**Classification**: **CORRECT BLOCK**. Both rules fired independently and correctly:
1. DQ-MCAP-MIN: 41.7M is below the 50M hard floor — micro-cap with thin liquidity.
2. DQ-EVEBIT-GLOBAL: EV/EBIT=0.26 is near zero — high ROIC + FCF but negligible EBIT
   suggests earnings quality mismatch (e.g. large non-recurring cash flows).

The +36.1% return is a fortunate outcome in a high-risk micro-cap; it does not invalidate
the rule. Expected-value reasoning: one in three such positions may win big while two lose.
No threshold adjustment required.

**Note on benchmark mismatch (Option C vs D 2018)**: AIK-B.ST is Swedish (^OMXS). Blocking
it caused ^OMXS to disappear from Option D's benchmark symbols (^HEX,^OMXS,^OSEAX → ^HEX,^OSEAX).
The WFT uses a position-dependent benchmark (tracks country index of actual holdings), so
benchmark CAGRs are not directly comparable across options with different DQ rules.

---

## WFT Tuner — Known Risks

### `mad_min` Overfit Risk

**Rule**: `mad_min` should not be set above `0.0` if the training set contains fewer
than two complete market cycles (typically < 8–10 years).

**Why**: WFT tuner can overfit to strict MAD values observed during bull years. With a
short training window dominated by bull-market data, the tuner learns that `mad_min=0.02`
(require stocks to be solidly above MA200) produces good in-sample returns. In the
out-of-sample recovery year immediately following a correction (e.g. 2019 after Q4-2018
sell-off), a strict MAD floor blocks the most attractive recovery picks when they are
still in the 0–2% above MA200 range.

**Observed**: Option B 2019 (1-year train window) → tuned `mad_min=-0.05`, return -8.7%.
Option C 2019 (8-year train window) → tuned `mad_min=0.02`, return -19.0%. The stricter
MAD parameter caused 10pp underperformance by blocking recovery picks.

**Mitigation**: Clip `mad_min` at `0.0` in the WFT parameter grid, or only allow
negative values (e.g. `[-0.05, -0.02, 0.0]`). Do not include positive values unless
the training set spans at least 2 full bull-bear cycles.

---

## Cyclical Guards (cyclicality config)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `mos_threshold` | 0.40 | MoS floor for cyclical sectors (vs 0.35 standard) |
| `ev_ebit_floor` | 6.0 | EV/EBIT floor for cyclicals |
| `fcf_window_years` | 5 | FCF normalisation window for cyclicals |
| `default_fcf_window_years` | 3 | FCF window for non-cyclicals |

Cyclical sectors: Materials, Oil & Gas, Shipping, Seafood (per Damodaran classification).
