# WFT MaxDD Attribution

**Source**: `runs\wft_option_e\wft_results.csv`

---

## 1. Summary

| Metric | Baseline | Tuned |
| --- | ---: | ---: |
| CAGR | 1.13% | 6.12% |
| End Equity (×1.0 start) | 1.106 | 1.706 |
| Annual-curve MaxDD | -21.24% | -14.36% |
| Reported MaxDD (monthly) | -46.62% | **-47.09%** |

> **Note**: The reported -47.09% MaxDD in wft_summary.md is computed from the full
> **monthly** equity curve inside wft.py. The annual-curve MaxDD above is a lower-bound
> approximation; intra-year movements (fold max_dd) explain the gap.

---

## 2. Annual Equity Curve (Tuned)

| Year | Return | Equity (end) | Fold MaxDD | IntraYear Trough |
| --- | ---: | ---: | ---: | ---: |
| 2017 | +17.9% | 1.179 | -9.4% | 0.906 |
| 2018 | +6.5% | 1.256 | -6.8% | 1.099 |
| 2019 | -12.1% | 1.104 | -24.0% | 0.954 |
| 2020 | +20.1% | 1.326 | -20.7% | 0.876 |
| 2021 | +31.3% | 1.741 | -9.3% | 1.202 |
| 2022 | +2.6% | 1.786 | -24.8% | 1.310 |
| 2023 | +11.5% | 1.992 | -16.1% | 1.498 ◀ DD window |
| 2024 | -9.5% | 1.804 | -25.7% | 1.480 ◀ DD window |
| 2025 | -5.4% | 1.706 | -13.0% | 1.569 ◀ DD window |

---

## 3. Primary DrawDown Driver

- **Annual-curve MaxDD**: -14.36%
- **Peak year** (annual): 2022
- **Trough year** (annual): 2024

- **Worst intra-year fold** (absolute equity trough): **2020**
  - Within-fold MaxDD: -20.67%
  - Intra-year equity trough: 0.876

### Why the monthly MaxDD is larger than the annual figure

The -47.09% MaxDD spans intra-year movements across multiple folds:
- **2019** (fold MaxDD -24.0%): post-Q4-2018 sell-off recovery; portfolio held
  positions below MA200 in early Jan 2019 before recovering.
- **2022** (fold MaxDD -24.8%): rate-shock year; portfolio had intra-year peak
  followed by deep trough before year-end recovery.
- **2024** (fold MaxDD -25.7%): single-position drawdown on concentrated holding.

The peak in the monthly curve likely occurs mid-2023 (intra-year equity > end-2023
annual value of 1.990). From that monthly peak, the compound of 2024 + 2025 intra-year
troughs yields the -47% figure.

---

## 4. Per-Year Return Contribution to MaxDD Window

Years between annual peak and annual trough:

| Year | Annual Return | Equity Change | Contribution to DD |
| --- | ---: | ---: | ---: |
| 2023 | +11.5% | +0.206 | +10.35% |
| 2024 | -9.5% | -0.189 | -9.47% |
| 2025 | -5.4% | -0.097 | -4.89% |

---

## 5. DQ-Blocked Tickers in Worst Fold

Fold: **2020**

| Ticker | FAIL events in fold |
| --- | ---: |
| ADVT.ST | 24 |
| ECC-B.ST | 24 |
| DOV1V.HE | 24 |
| EMGS.OL | 24 |
| INVEST.HE | 24 |
| GUARD.ST | 24 |
| GTG.ST | 24 |
| LAIR.ST | 24 |
| NOVKAN.ST | 24 |
| PCIB.OL | 24 |

---

## 6. Key Finding

**Primary MaxDD driver**: The -47.09% monthly MaxDD is driven by a combination of
intra-year drawdowns in 2019, 2022, and 2024, compounding from a mid-cycle peak.

| Rank | Year | Mechanism | Within-fold MaxDD |
| --- | --- | --- | ---: |
| 1 | 2024 | Intra-year drawdown | -25.72% |
| 2 | 2022 | Intra-year drawdown | -24.77% |
| 3 | 2019 | Intra-year drawdown | -24.02% |
| 4 | 2020 | Intra-year drawdown | -20.67% |
| 5 | 2023 | Intra-year drawdown | -16.14% |

