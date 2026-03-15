# 2022 Fold Diagnosis — Option E

**Fold result**: tuned return = **+2.6%**, benchmark = -3.2%, fold MaxDD = **-24.8%**
**Benchmark symbol**: `^OSEAX`
**Note**: The fold ended *positive* (+2.6%). The -24.8% MaxDD is purely intra-year
(peak Feb, trough Dec). This is structurally different from 2024 which ended negative.

---

## 1. What Was Selected

Parameters chosen for 2022 fold: `mad_min=-0.02, mos_threshold=0.30, baseline`.

| Month (decision→holding) | Position | Weight (approx.) | Port. Ret |
|--------------------------|----------|:----------------:|----------:|
| Jan 2022 | REACH.OL + NKR.OL + KOG.OL | 79% / 14% / 7% | **+12.8%** |
| Feb 2022 | REACH.OL + NKR.OL + NAS.OL | 80% / 14% / 6% | **+20.9%** |
| Mar 2022 | REACH.OL + NAS.OL + KOG.OL | 86% / 7% / 8% | -4.9% |
| Apr 2022 | REACH.OL + NAS.OL + KOG.OL | 86% / 7% / 8% | -6.9% |
| May 2022 | REACH.OL + NAS.OL + KOG.OL | 86% / 7% / 8% | **-8.8%** |
| Jun 2022 | SCANA.OL + REACH.OL + KOG.OL | **51% / 45% / 4%** | +7.2% |
| Jul 2022 | SCANA.OL + REACH.OL + KOG.OL | 51% / 45% / 4% | -4.0% |
| Aug 2022 | SCANA.OL + REACH.OL + KOG.OL | 51% / 45% / 4% | -6.9% |
| Sep 2022 | CASH | — | 0.0% |
| Oct 2022 | CASH | — | 0.0% |
| Nov 2022 | REACH.OL + DEVP-B.ST + TIETO.HE | 83% / 10% / 7% | -2.7% |

*Weights are MoS-proportional. REACH.OL (mos=4.86) dominates Jan–Jun. SCANA.OL
(mos=5.59) displaces REACH to 45% when it enters in Jun.*

### Equity curve

| Point | Equity | Note |
|-------|-------:|------|
| Jan start | 1.000 | |
| After Feb (peak) | **1.364** | ← PEAK: oil rally + defense + shipping |
| After May | 1.101 | Oil correction; NAS Jun crash incoming |
| After Aug | 1.055 | SCANA dragging Jul–Sep |
| After Nov (CASH) | 1.055 | Hedged |
| After Dec (trough) | **1.026** | ← TROUGH (year end) |

**MaxDD**: (1.026 – 1.364) / 1.364 = **−24.8%** ✓
**Total return**: +2.6% ✓

---

## 2. Root Cause: Two distinct phases

### Phase 1 — REACH.OL concentration (Jan–Jun)

REACH.OL (Oil & Gas, mos=4.86) dominated the portfolio at **79–86% weight** due to its
extreme MoS. This is by design (MoS-proportional weighting = high conviction → high weight).

| Metric | Value |
|--------|-------|
| Sector | Oil & Gas |
| MoS (Dec-2021 snapshot) | 4.86 (486%) |
| ROIC | 36.4% |
| FCF yield | 39.7% ← flagged as outlier (WARN) |
| ev_ebit | 7.57 |
| Annual return 2022 | **+39.1%** |

REACH's intra-year monthly returns:

| Month | REACH ret | Port impact (86% weight) |
|-------|----------:|------------------------:|
| Jan→Feb | +16.2% | +12.8% |
| Feb→Mar | **+27.6%** | **+22.1%** → peak |
| Mar→Apr | -8.9% | -7.7% |
| Apr→May | -5.4% | -4.6% |
| May→Jun | **-8.5%** | **-7.3%** ← worst month |
| Jun→Jul | -2.0% | -0.9% |
| Aug→Sep | -1.0% | -0.5% |

**Mechanism**: Oil prices peaked in March 2022 (Ukraine invasion premium), then fell
back through June. REACH's 86% weight meant the portfolio tracked the oil price
peak-to-trough closely. This is a **market-wide energy cycle event**, not a selection error.
REACH ended the year at +39.1%.

### Phase 2 — SCANA.OL concentration (Jul–Sep)

When SCANA.OL entered in July, it was ranked #1 by quality_score *and* had mos=5.59,
displacing REACH to 45% weight and taking 51% itself.

| Metric | Value |
|--------|-------|
| Sector | Financial Services |
| MoS (2022 snapshot) | **5.59** (559%) |
| ROIC | 31.9% |
| FCF yield | **43.7%** ← extreme |
| ev_ebit | 17.74 |
| market_cap | 132M NOK |
| Annual return 2022 | **-25.2%** |

SCANA's Jul–Sep monthly returns with 51% weight:

| Month | SCANA ret | Port impact (51% weight) |
|-------|----------:|------------------------:|
| Jun→Jul | +15.6% | +7.9% |
| Jul→Aug | **-16.2%** | **-8.3%** |
| Aug→Sep | -12.4% | -6.3% |

**Mechanism**: SCANA's FCF yield of 43.7% is extreme for a Financial Services company.
This inflated its intrinsic_value → mos=5.59 → 51% portfolio weight. When SCANA fell
-16.2% in August and -12.4% in September, the portfolio absorbed the full impact.
This mirrors the TITAN.HE pattern: a Financial Services company with an artefactual
quality metric that inflated its ranking.

### Secondary holdings (minor impact)

| Ticker | Period held | Annual ret | Portfolio weight | Est. contribution |
|--------|------------|----------:|:---------------:|------------------:|
| KOG.OL | Jan–Sep | **+54.8%** | 4–8% | +positive (offset) |
| NKR.OL | Jan–Feb | -38.2% | 14% | ~-2.5% to Feb months |
| NAS.OL | Mar–Jun | -38.9% | 7% | ~-2.7% to Jun month |

NAS.OL had **ev_ebit = -4.45** (loss-making company, post-bankruptcy restructuring).
This generated only WARN under DQ-EVEBIT-GLOBAL. Due to 7% weight, NAS contributed
≈ 7% × (−28.5% Jun return) = −2.0% to the worst month — marginal but not zero.

NAS's ROIC of 36.1% is a post-restructuring artefact: Norwegian Air Shuttle went through
debt cancellation and equity dilution in 2020–2021, leaving near-zero book equity.
ROIC on near-zero equity appears high but is not indicative of operational quality.

---

## 3. DQ Gates That Triggered in 2022

### Correctly Blocked

DQ-EVEBIT-GLOBAL (0 < ev_ebit < 6.0 → FAIL) blocked multiple tickers with boom-FCF
artefacts. No detail on which ones would have ranked highly is available from the audit
(global audit, not per-fold selection log), but the rule was active.

### WARN-only (not blocking) in 2022 simulation

| Ticker | Rule | ev_ebit (2022 snap) | ev_ebit (global audit) | Impact |
|--------|------|-------------------:|----------------------:|--------|
| REACH.OL | DQ_OUTLIER_ROBUST_fcf_yield | +7.57 | n/a | WARN, fcf_yield=39.7% |
| NAS.OL | DQ-EVEBIT-GLOBAL (WARN) | -4.45 | no event | Negative EBIT → WARN |
| NKR.OL | DQ-EVEBIT-GLOBAL (WARN) | +6.12 (2022 snap) | ev_ebit=-113 (current) | Only WARN in global audit |
| SCANA.OL | DQ-EVEBIT-GLOBAL (WARN) | +17.74 (2022 snap) | ev_ebit=-23.4 (current) | Only WARN in global audit |

**Important note on the DQ audit**: `wft_data_quality_audit.csv` is built from the
*global* panel (latest snapshot), not per-fold snapshots. NKR and SCANA show negative
ev_ebit in the global audit because their fundamentals deteriorated *after* 2022 — in
the actual 2022 simulation they had positive ev_ebit (6.12 and 17.74 respectively).

---

## 4. Classification: C — Combination

The -24.8% MaxDD falls into **Category C** (market-wide + portfolio construction factor):

**Market component (A)**: Oil price peaked March 2022 then normalized by June. This was
a broad commodity cycle event unrelated to REACH's fundamental quality. REACH ended
+39.1% — it was a correct pick. There was no selection error on REACH.

**Portfolio construction component (B-adjacent)**: SCANA.OL (Financial Services,
FCF yield=43.7%) was ranked #1 in quality_score AND had the highest MoS (5.59) in the
universe for the June decision. This gave it 51% portfolio weight. When SCANA fell -25%
from July onward, the concentrated exposure amplified the drawdown.

SCANA's FCF yield of 43.7% for a 132M-NOK Financial Services company is structurally
similar to TITAN.HE's ROIC of 111% — both are artefactual quality signals for Financial
Services companies that inflated their ranking and portfolio weight.

**Split of the -24.8% MaxDD**:
- Phase 1 (Feb peak → Jun trough through REACH): Oil cycle, −13–15% from peak
- Phase 2 (Jun → Dec trough through SCANA): FCF artefact + sector decline, −7–9% additional
- The final trough at year-end combines both phases

---

## 5. Proposed Rule (if B/C element)

The SCANA.OL case mirrors TITAN.HE: a **Financial Services company with artefactually
high quality metrics** (FCF yield in SCANA's case, ROIC in TITAN's case) dominated the
portfolio and delivered significant losses.

**Candidate rule: DQ-FINSERV-FCF**

| Field | Value |
|-------|-------|
| Rule ID | `DQ-FINSERV-FCF` |
| Field | `fcf_yield` + `sector` |
| Condition | `fcf_yield > 0.30` AND `sector == "Financial Services"` AND NOT `is_bank_proxy` |
| Severity | **WARN** (escalate to FAIL after validation) |
| Threshold | `fcf_yield > 0.30` (30%) |
| Scope | Financial Services non-bank |
| Rationale | FCF in financial companies includes loan proceeds, bond issuance, and working capital movements — not the same as operating free cash flow for industrial companies. A 30%+ FCF yield for a small FinServ company likely reflects capital structure artefacts, not sustainable cash generation. |
| Expected impact | Would have WARNed on SCANA (43.7%) and TITAN (if measured similarly). Paired with DQ-MOS-SUSPECT, could catch the broader "high-MoS Financial Services" pattern. |

**Note**: The existing `DQ_OUTLIER_ROBUST_fcf_yield` rule already WARNs on statistical
outliers, but uses universe-wide z-score normalization — it is sector-agnostic. A
sector-specific FCF floor is a more targeted fix for the Financial Services artefact.

**NAS.OL secondary rule consideration** (minor impact):

The current DQ-EVEBIT-GLOBAL treats `ev_ebit < 0` as WARN only, while `0 < ev_ebit < 6.0`
is FAIL. A loss-making company (negative EBIT) should arguably be at least as risky as
a boom-cycle cheap company. However, given NAS's 7% weight in 2022, its impact was small.
This is worth tracking but not a priority fix.

---

## 6. Comparison with TITAN.HE (2024)

| Dimension | TITAN.HE (2024) | SCANA.OL (2022) |
|-----------|----------------|-----------------|
| Sector | Financial Services | Financial Services |
| Artefactual metric | ROIC=111% (leverage) | FCF yield=44% (capital structure) |
| Portfolio weight | 18% (limited by max_positions=3) | **51%** (dominant via MoS) |
| Annual return | -43.2% | -25.2% |
| EV/EBIT | 9.6 (positive, passed rules) | 17.74 (positive, passed rules) |
| DQ gate that failed | No rule caps Financial Services ROIC | No rule caps Financial Services FCF |
| Annual fold result | -9.5% loss | +2.6% gain |

**Key difference**: In 2024, TITAN.HE was a selection error (bad pick drove annual loss).
In 2022, SCANA was selected correctly by existing rules but was one contributor to
intra-year MaxDD; the year ended positive because REACH and KOG were offsetting winners.

**Same underlying pattern**: Financial Services companies with leverage- or structure-
inflated quality metrics escape existing DQ rules and reach high portfolio weights.

---

## 7. Summary

| Item | Value |
|------|-------|
| Primary MaxDD driver | REACH.OL oil-price intra-year cycle (86% weight) |
| Secondary MaxDD driver | SCANA.OL FCF artefact (51% weight Jul–Sep) |
| Fold annual return | **+2.6%** (positive — MaxDD is intra-year only) |
| Fold MaxDD | -24.8% (peak Feb→trough Dec) |
| DQ gate that would have helped | `DQ-FINSERV-FCF`: FCF yield cap for Financial Services |
| Classification | **C — Combination** (market cycle + FinServ artefact) |
| Same pattern as TITAN.HE? | **Yes** (FinServ quality artefact, different metric) |
| Rule implementation | Not implemented — analysis only |

The 2022 fold is the inverse of 2024: a market-driven energy rally saved the portfolio
despite poor intra-year positioning. In 2024, energy/defense tailwinds were absent and
the TITAN.HE ROIC artefact caused an annual loss. Both share the same root cause:
Financial Services companies with non-operational quality metrics inflating their rank.
