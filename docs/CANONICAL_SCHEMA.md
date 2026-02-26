# Canonical Schema v1 (Nordic ESEF)

Canonical Schema v1 normalizes ESEF iXBRL facts into stable, valuation-ready raw fields for Nordic listed issuers (NO/SE/DK/FI), across industrials, banks, and insurers.

## Design Rules

1. **Raw-first schema**
   - `config/canonical_schema.yaml` contains only raw fields. Computed metrics are defined separately in `config/computed_metrics.yaml`.
2. **Sector-aware fields**
   - Each field has `sector` = `all`, `bank`, or `insurance`.
3. **Statement grounding**
   - Every field is tied to `IS`, `BS`, `CF`, `NOTE`, `SHARES`, or `DEBT`.
4. **Sign convention is explicit**
   - Outflows (e.g., capex, dividends) use negative expectation.
5. **Stable IDs**
   - `field_id` is immutable snake_case.

## IFRS16 Leases Treatment

- Keep both debt views for valuation consistency:
  - **Including leases**: use `lease_liabilities_current` + `lease_liabilities_noncurrent` in net debt.
  - **Excluding leases**: omit lease liabilities for legacy comparability.
- Right-of-use assets are mapped separately via `right_of_use_assets`.

## Goodwill and Intangibles

- Keep `goodwill` and `intangible_assets` separate.
- Do not net impairment into carrying values in canonical raw fields.
- Tangible-capital ROIC variants should be computed downstream via computed metrics.

## Banks-specific Guidance

Use bank-only fields when entity is classified as bank:
- Core P&L: `bank_net_interest_income`, `bank_fee_commission_income`, `bank_loan_loss_provisions`.
- Balance sheet: `bank_total_loans_gross`, `bank_ecl_allowance`, `bank_total_loans_net`, `bank_customer_deposits`.
- Regulatory: `bank_cet1_capital`, `bank_cet1_ratio`, `bank_rwa`.
- NIM and credit metrics should be computed downstream, not stored as raw fields.

## Insurance-specific Guidance

Use insurance-only fields for insurers:
- Premium flow: `ins_gross_written_premiums`, `ins_earned_premiums`.
- Claims/costs: `ins_claims_incurred`, `ins_acquisition_costs`, `ins_operating_expenses`.
- Result: `ins_underwriting_result`, `ins_investment_result`.
- Solvency and combined ratio can be ingested if explicitly reported (`ins_solvency_ratio`, `ins_combined_ratio_reported`).

## Currency Handling

- Preserve original reported currency in `reporting_currency`.
- Store raw values exactly as reported before FX translation.
- FX normalization/conversion should happen in downstream transformation layers with explicit as-of rates.

## Period Types and `period_end`

- `period_end` is the canonical end date (`YYYY-MM-DD`) from XBRL context.
- `reporting_period_type` should encode period class: `FY`, `H1`, `Q1`, `Q3`, etc.
- Prefer annual (`FY`) values for long-horizon valuation metrics; allow interim for monitoring.

## Mapping Examples (XBRL fact -> field_id)

- `ifrs-full:Revenue` -> `revenue_total`
- `ifrs-full:OperatingProfitLoss` -> `operating_income_ebit`
- `ifrs-full:LeaseLiabilitiesNoncurrent` -> `lease_liabilities_noncurrent`
- `ifrs-full:NetCashFlowsFromUsedInOperatingActivities` -> `cash_flow_from_operations`
- `ifrs-full:WeightedAverageNumberOfSharesOutstandingBasic` -> `shares_outstanding_basic`
- `ifrs-full:InsuranceRevenue` -> `ins_earned_premiums` (insurance sector only)

## Minimum Viable Coverage Target

For listed ESEF filers, minimum viable coverage target is:
- **>= 95% coverage** on priority-1 core valuation-ready raw fields
- measured at issuer-period level for annual reports.

## Field Index (summary)

| Segment | Approx fields | Core focus |
|---|---:|---|
| Common (`sector=all`) | 71 | Revenue, EBIT, taxes, cash flow, debt, equity, shares |
| Bank (`sector=bank`) | 21 | NII, ECL, loans/deposits, CET1, RWA |
| Insurance (`sector=insurance`) | 15 | Premiums, claims, underwriting result, insurance liabilities |
| **Total** | **107** | Supports DCF/FCFF, RIM/ROE, and factor research |

