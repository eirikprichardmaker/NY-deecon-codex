# 2024 Fold Diagnosis — Option E

**Fold result**: tuned return = **-9.5%**, benchmark = +0.8%, fold MaxDD = **-25.7%**
**Benchmark symbols**: `^HEX, ^OSEAX`

---

## 1. What Was Selected

Portfolio ranked by quality_score from December-2023 snapshot (asof 2023-12-29):

| Rank | Ticker | Sector | MoS | EV/EBIT | ROIC | FCF Yield | quality_score | 2024 Return |
|------|--------|--------|----:|--------:|-----:|----------:|--------------:|------------:|
| 1 | **TITAN.HE** | Financial Services | 47.9% | 9.6 | 111.2% | 9.5% | 2.60 | **-43.2%** |
| 2 | NAS.OL | Industrials | 141.6% | 6.6 | 14.7% | 18.8% | 0.53 | 0.0% |
| 3 | NOHO.HE | Consumer Disc | 70.4% | 13.1 | 2.1% | 26.4% | 0.47 | -10.0% |
| 4 | KJELL.ST | Consumer Disc | 164.3% | 73.6 | -1.4% | 26.4% | 0.35 | -67.3% |
| 5 | DRIL.ST | Materials | 65.0% | 9.9 | 9.6% | 12.8% | 0.32 | -28.9% |

With max_positions=3 and MoS-proportional weights:
- TITAN.HE: weight ≈ 18% (mos=0.479)
- NAS.OL: weight ≈ 55% (mos=1.416)
- NOHO.HE: weight ≈ 27% (mos=0.704)

**Estimated 2024 return**: 0.18×(−43.2%) + 0.55×(0.0%) + 0.27×(−10.0%) ≈ **−10.5%**
(Actual: -9.5% with monthly rebalancing)

---

## 2. Root Cause: TITAN.HE

TITAN.HE (Finnish Financial Services) was ranked #1 by quality_score thanks to
exceptionally high ROIC (111%) and positive FCF yield (9.5%). However, it collapsed
-43.2% in 2024, concentrated primarily in April (-22.6%) with no recovery.

### TITAN.HE monthly returns 2024:

| Month | Return |
|-------|-------:|
| Jan 2024 | -1.4% |
| Feb 2024 | -11.7% |
| Mar 2024 | -7.7% |
| **Apr 2024** | **-22.6%** |
| May 2024 | +23.2% |
| Jun 2024 | -7.5% |
| Jul 2024 | -3.4% |
| Aug 2024 | -8.7% |
| Sep 2024 | -1.4% |
| Oct 2024 | -5.4% |
| Nov 2024 | -0.8% |
| Dec 2024 | -2.9% |

**Key insight**: TITAN.HE's ROIC=111% reflects Financial Services accounting (high
returns on equity capital, not operational efficiency). The quality_score z-score
heavily rewards this, creating a systematic bias toward financial firms with
leverage-inflated ROIC. The MA200 gate (MAD=+0.046 in Dec-2023) gave no warning.

---

## 3. DQ Gates That Triggered in 2024

### Correctly Blocked (DQ-EVEBIT-GLOBAL saved the portfolio):

| Ticker | ev_ebit | 2024 return | Verdict |
|--------|--------:|------------:|---------|
| DEDI.ST | 5.96 | **-48.5%** | CORRECT BLOCK |
| DEVP-B.ST | 5.88 | **-24.7%** | CORRECT BLOCK |
| SSAB-A.ST | 3.71 | **-42.1%** | CORRECT BLOCK |
| SSAB-B.ST | 3.59 | **-43.6%** | CORRECT BLOCK |

Without DQ-EVEBIT-GLOBAL, DEDI.ST (quality_score=high, MoS=94%) would have been
the #1 pick — returning -48.5% vs TITAN.HE's -43.2%. The rule avoided the worst
outcome but did not prevent the 2024 loss entirely.

### Missed Opportunities (DQ-EVEBIT-GLOBAL also blocked winners):

| Ticker | ev_ebit | 2024 return | Verdict |
|--------|--------:|------------:|---------|
| SHOT.ST | 5.35 | **+50.5%** | FALSE POSITIVE cost |
| MTHH.CO | 2.70 | **+141.8%** | FALSE POSITIVE cost |
| SPKSJF.CO | 4.38 | **+24.1%** | FALSE POSITIVE cost |

These were blocked because DQ-EVEBIT-GLOBAL uses a single threshold (6.0) without
distinguishing between value traps (DEDI, SSAB) and genuinely cheap quality companies
(SHOT, MTHH). This is the **fundamental precision/recall tradeoff** of the rule.

---

## 4. Was There a Near-Miss? Could a Different Ticker Have Been Selected?

Yes. If TITAN.HE had been blocked (or if it started dropping earlier in the year
and crossed below MA200), the next candidate would have been **NAS.OL** (0.0% in 2024).

NAS.OL at full portfolio weight would have given ~0.0% return instead of -9.5%.

**Could NAS.OL have been the sole pick?** Only if TITAN.HE was blocked before Jan 2024.
TITAN.HE's MAD in December 2023 was +4.6% (above MA200). No existing gate blocked it.

**Candidate rule improvement** (not implemented): Add `roic_cap_financial_services`
to quality score computation — financial firms with ROIC > 50% should have quality
score capped to avoid ROIC-leverage bias inflating their rank.

---

## 5. Summary

| Item | Value |
|------|-------|
| Primary driver of -9.5% loss | **TITAN.HE** (-43.2%, financial ROIC bias) |
| DQ-EVEBIT-GLOBAL savings | Avoided DEDI.ST (-48.5%) and SSAB losses |
| DQ-EVEBIT-GLOBAL cost | Missed SHOT.ST (+50.5%) and MTHH.CO (+141.8%) |
| Gate that would have helped | ROIC cap for Financial Services in quality score |
| Fold MaxDD -25.7% | Driven by TITAN.HE's April crash (-22.6% in one month) |

The 2024 -9.5% loss is smaller than Option C's -20.0% (DQ-EVEBIT-GLOBAL blocked
the worst value traps), but TITAN.HE was a new failure mode not captured by existing
rules — high-quality financial firm with leverage-inflated ROIC.
