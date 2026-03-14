"""
10-årsanalyse av forventet avkastning basert på WFT-resultater.

Kilde: wft_results.csv fra alle tilgjengelige WFT-kjøringer
Metode:
  1. Samler alle out-of-sample årsresultater (2016–2025)
  2. CAGR, Sharpe, volatilitet, max drawdown
  3. Bootstrap Monte Carlo — 10-årsprosjeksjon med konfidensintervaller
  4. Scenario-analyse: Bear / Base / Bull
  5. Regresjonsanalyse: trend i avkastning over tid
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from scipy import stats as _scipy_stats
    def _linregress(x, y):
        return _scipy_stats.linregress(x, y)
except ImportError:
    # Fallback: manuell OLS
    def _linregress(x, y):
        x, y = np.array(x, float), np.array(y, float)
        n = len(x)
        xm, ym = x.mean(), y.mean()
        slope = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
        intercept = ym - slope * xm
        y_hat = slope * x + intercept
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - ym) ** 2)
        r_value = np.sqrt(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        se = np.sqrt(ss_res / (n - 2) / np.sum((x - xm) ** 2)) if n > 2 else 0.0
        t_stat = slope / se if se > 0 else 0.0
        from math import erfc, sqrt
        p_value = float(erfc(abs(t_stat) / sqrt(2)))
        class _R:
            pass
        r = _R()
        r.slope, r.intercept, r.rvalue, r.pvalue, r.stderr = slope, intercept, r_value, p_value, se
        return r

ROOT = Path(__file__).parents[1]

# ---------------------------------------------------------------------------
# 1. Last alle WFT-årsresultater
# ---------------------------------------------------------------------------

WFT_SOURCES = [
    # Lengst historikk: 2016–2025 (10 år), baseline + tuned
    ("wft_2010_2025_173706", "wft_results.csv"),
    # Kortere, men bekrefter 2023–2025
    ("wft_20260216",          "wft_results.csv"),
    ("wft_20260216_v2",       "wft_results.csv"),
    ("wft_sweep_kpi_core_26_20260301", "sweep_summary.csv"),
]

def load_annual_returns() -> pd.DataFrame:
    rows = []

    for run_dir, fname in WFT_SOURCES:
        p = ROOT / "runs" / run_dir / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "return" in df.columns and "test_year" in df.columns:
            # wft_results format
            for _, r in df.iterrows():
                rows.append({
                    "source": run_dir,
                    "year": int(r["test_year"]),
                    "mode": r.get("mode", "baseline"),
                    "return": float(r["return"]),
                    "max_dd": float(r.get("max_dd", 0) or 0),
                    "pct_cash": float(r.get("pct_cash", 0) or 0),
                    "position": r.get("candidate_or_cash", "UNKNOWN"),
                })
        elif "cagr_net" in df.columns:
            # sweep_summary format — 3-fold CAGR, brukes som ekstra kalibrering
            cagr_net = df["cagr_net"].mean()
            rows.append({
                "source": run_dir,
                "year": 2023,  # midtpunkt for 3-fold (2023/24/25)
                "mode": "kpi_sweep",
                "return": cagr_net,
                "max_dd": df["max_dd"].mean() if "max_dd" in df.columns else 0,
                "pct_cash": df["cash_share"].mean() if "cash_share" in df.columns else 0,
                "position": "CANDIDATE",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Multiseed optimeringsresultater (holdgrid + multiseed)
# ---------------------------------------------------------------------------

def load_multiseed_results() -> dict:
    results = {}
    for fname in [
        "wft_opt_holdgrid_3_6_9_12_multiseed_summary.csv",
        "wft_opt_multiseed_summary.csv",
        "wft_opt_hold6_s21_s30_summary.csv",
    ]:
        p = ROOT / "runs" / fname
        if p.exists():
            df = pd.read_csv(p)
            if "holdout_cum_net" in df.columns:
                # holdout_cum_net er total kumulativ avkastning over ~3 år (2023–2025)
                results[fname] = {
                    "n": len(df),
                    "holdout_cum_net_mean": df["holdout_cum_net"].mean(),
                    "holdout_cum_net_median": df["holdout_cum_net"].median(),
                    "holdout_cum_net_std": df["holdout_cum_net"].std(),
                    "holdout_max_dd_mean": df["holdout_max_dd"].mean(),
                    "holdout_pct_cash_mean": df["holdout_pct_cash"].mean(),
                    "oos_cagr_net_mean": df["oos_cagr_net"].mean() if "oos_cagr_net" in df.columns else None,
                }
    return results


# ---------------------------------------------------------------------------
# 3. Statistisk analyse
# ---------------------------------------------------------------------------

def cagr(returns: np.ndarray) -> float:
    """CAGR fra array av årsavkastninger."""
    cum = np.prod(1 + returns)
    n = len(returns)
    return cum ** (1 / n) - 1


def sharpe(returns: np.ndarray, rf: float = 0.04) -> float:
    excess = returns - rf
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std()


def max_drawdown(returns: np.ndarray) -> float:
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(dd.min())


def analyse_annual_series(df_baseline: pd.DataFrame) -> dict:
    r = df_baseline["return"].values
    return {
        "n_years": len(r),
        "years": sorted(df_baseline["year"].tolist()),
        "mean_annual_return": float(np.mean(r)),
        "median_annual_return": float(np.median(r)),
        "std_annual_return": float(np.std(r, ddof=1)),
        "cagr": cagr(r),
        "sharpe": sharpe(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": float((r > 0).mean()),
        "worst_year_return": float(r.min()),
        "best_year_return": float(r.max()),
        "pct_cash_mean": float(df_baseline["pct_cash"].mean()),
    }


# ---------------------------------------------------------------------------
# 4. Bootstrap Monte Carlo — 10-årsprosjeksjon
# ---------------------------------------------------------------------------

def bootstrap_10yr(returns: np.ndarray, n_sim: int = 50_000, horizon: int = 10) -> dict:
    np.random.seed(42)
    rng = np.random.default_rng(42)

    sim_results = []
    for _ in range(n_sim):
        sampled = rng.choice(returns, size=horizon, replace=True)
        total = float(np.prod(1 + sampled) - 1)
        sim_results.append(total)

    sim = np.array(sim_results)
    sim_cagr = (1 + sim) ** (1 / horizon) - 1

    return {
        "10yr_total_return": {
            "p10":  float(np.percentile(sim, 10)),
            "p25":  float(np.percentile(sim, 25)),
            "p50":  float(np.percentile(sim, 50)),
            "p75":  float(np.percentile(sim, 75)),
            "p90":  float(np.percentile(sim, 90)),
            "mean": float(sim.mean()),
        },
        "10yr_cagr": {
            "p10":  float(np.percentile(sim_cagr, 10)),
            "p25":  float(np.percentile(sim_cagr, 25)),
            "p50":  float(np.percentile(sim_cagr, 50)),
            "p75":  float(np.percentile(sim_cagr, 75)),
            "p90":  float(np.percentile(sim_cagr, 90)),
            "mean": float(sim_cagr.mean()),
        },
        "prob_positive": float((sim > 0).mean()),
        "prob_double":   float((sim > 1.0).mean()),   # >100% totalavkastning
        "prob_triple":   float((sim > 2.0).mean()),   # >200% totalavkastning
        "prob_loss_over_20pct": float((sim < -0.20).mean()),
        "n_sim": n_sim,
    }


# ---------------------------------------------------------------------------
# 5. Regresjonsanalyse — tidstrend i avkastning
# ---------------------------------------------------------------------------

def regression_trend(df: pd.DataFrame) -> dict:
    df_sorted = df.sort_values("year")
    x = df_sorted["year"].values.astype(float)
    y = df_sorted["return"].values.astype(float)

    res = _linregress(x, y)
    slope, intercept, r_value, p_value, std_err = res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr
    return {
        "slope_per_year": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "trend_significant": p_value < 0.10,
        "note": "Positiv slope = stigende avkastningstend, p<0.10 = statistisk signifikant",
    }


# ---------------------------------------------------------------------------
# 6. Scenarioanalyse
# ---------------------------------------------------------------------------

def scenario_analysis(returns: np.ndarray) -> dict:
    """Bear/Base/Bull basert på historiske persentiler."""
    bear_returns = returns[returns <= np.percentile(returns, 25)]
    bull_returns = returns[returns >= np.percentile(returns, 75)]
    base_returns = returns

    def project(r: np.ndarray, horizon: int = 10) -> float:
        return float(np.prod(1 + r.mean() * np.ones(horizon)) - 1)

    def proj_cagr(r: np.ndarray, horizon: int = 10) -> float:
        return float((1 + project(r, horizon)) ** (1 / horizon) - 1)

    return {
        "bear":  {"annual_return_assumed": float(bear_returns.mean()),
                  "10yr_total": project(bear_returns),
                  "10yr_cagr": proj_cagr(bear_returns)},
        "base":  {"annual_return_assumed": float(base_returns.mean()),
                  "10yr_total": project(base_returns),
                  "10yr_cagr": proj_cagr(base_returns)},
        "bull":  {"annual_return_assumed": float(bull_returns.mean()),
                  "10yr_total": project(bull_returns),
                  "10yr_cagr": proj_cagr(bull_returns)},
    }


# ---------------------------------------------------------------------------
# 7. Hovedrapport
# ---------------------------------------------------------------------------

def fmt_pct(v: float) -> str:
    return f"{v*100:+.1f}%"

def fmt_x(v: float) -> str:
    return f"{1+v:.2f}x"


def main():
    df_all = load_annual_returns()
    if df_all.empty:
        print("FEIL: Ingen WFT-data funnet.")
        sys.exit(1)

    # Bruk baseline-serien fra lengste kjøring som primær analyse
    df_primary = (
        df_all[
            (df_all["source"] == "wft_2010_2025_173706") &
            (df_all["mode"] == "baseline")
        ]
        .drop_duplicates("year")
        .sort_values("year")
    )

    # Tuned-serien for sammenligning
    df_tuned = (
        df_all[
            (df_all["source"] == "wft_2010_2025_173706") &
            (df_all["mode"] == "tuned")
        ]
        .drop_duplicates("year")
        .sort_values("year")
    )

    r_base  = df_primary["return"].values
    r_tuned = df_tuned["return"].values

    stats_base  = analyse_annual_series(df_primary)
    stats_tuned = analyse_annual_series(df_tuned)
    mc_base  = bootstrap_10yr(r_base)
    mc_tuned = bootstrap_10yr(r_tuned)
    trend    = regression_trend(df_primary)
    scenario = scenario_analysis(r_base)
    multiseed = load_multiseed_results()

    # -----------------------------------------------------------------------
    # Skriv rapport
    # -----------------------------------------------------------------------
    sep = "=" * 70

    print(f"\n{sep}")
    print("  DEECON — Forventet avkastning over 10 år")
    print(f"  Basert på WFT out-of-sample data 2016–2025")
    print(sep)

    print(f"\n{'─'*70}")
    print("  ÅRS-FOR-ÅRS HISTORIKK (baseline)")
    print(f"{'-'*70}")
    for _, row in df_primary.iterrows():
        mark = "CASH" if row["position"] in ("CASH", "cash") else "  "
        print(f"  {int(row['year'])}  {fmt_pct(row['return']):>8}  {mark}  max_dd={fmt_pct(row['max_dd'])}")

    print(f"\n{'─'*70}")
    print("  NØKKELTALL  (baseline vs tuned)")
    print(f"{'-'*70}")
    for label, s in [("Baseline", stats_base), ("Tuned   ", stats_tuned)]:
        print(f"\n  [{label}]")
        print(f"    Perioder:            {s['n_years']} år")
        print(f"    CAGR:                {fmt_pct(s['cagr'])}")
        print(f"    Snitt årsavkastning: {fmt_pct(s['mean_annual_return'])}")
        print(f"    Volatilitet (std):   {fmt_pct(s['std_annual_return'])}")
        print(f"    Sharpe (rf=4%):      {s['sharpe']:.2f}")
        print(f"    Max drawdown:        {fmt_pct(s['max_drawdown'])}")
        print(f"    Hit-rate (>0):       {s['hit_rate']*100:.0f}%")
        print(f"    Beste år:            {fmt_pct(s['best_year_return'])}")
        print(f"    Verste år:           {fmt_pct(s['worst_year_return'])}")
        print(f"    Andel CASH:          {s['pct_cash_mean']*100:.0f}%")

    print(f"\n{'─'*70}")
    print("  REGRESJONSANALYSE — tidstrend i avkastning")
    print(f"{'-'*70}")
    t = trend
    print(f"    Trend per år:  {fmt_pct(t['slope_per_year'])}")
    print(f"    R²:            {t['r_squared']:.3f}")
    print(f"    p-verdi:       {t['p_value']:.3f}  {'(signifikant)' if t['trend_significant'] else '(ikke signifikant)'}")
    print(f"    Konklusjon:    {'Avkastningen har stigende trend' if t['slope_per_year'] > 0 else 'Ingen klar stigende trend'}")

    print(f"\n{'─'*70}")
    print("  MONTE CARLO BOOTSTRAP — 10-årsprosjeksjon (50 000 simuleringer)")
    print(f"{'-'*70}")
    for label, mc in [("Baseline", mc_base), ("Tuned   ", mc_tuned)]:
        c = mc["10yr_cagr"]
        t = mc["10yr_total_return"]
        print(f"\n  [{label}]")
        print(f"    CAGR projeksjon:")
        print(f"      Bear  (P10):   {fmt_pct(c['p10'])}  →  {fmt_x(mc['10yr_total_return']['p10'])} kapital")
        print(f"      P25:           {fmt_pct(c['p25'])}  →  {fmt_x(mc['10yr_total_return']['p25'])} kapital")
        print(f"      Median (P50):  {fmt_pct(c['p50'])}  →  {fmt_x(mc['10yr_total_return']['p50'])} kapital")
        print(f"      P75:           {fmt_pct(c['p75'])}  →  {fmt_x(mc['10yr_total_return']['p75'])} kapital")
        print(f"      Bull  (P90):   {fmt_pct(c['p90'])}  →  {fmt_x(mc['10yr_total_return']['p90'])} kapital")
        print(f"    Sannsynligheter:")
        print(f"      P(positiv 10år):     {mc['prob_positive']*100:.0f}%")
        print(f"      P(doble kapital):    {mc['prob_double']*100:.0f}%")
        print(f"      P(triple kapital):   {mc['prob_triple']*100:.0f}%")
        print(f"      P(tap > 20%):        {mc['prob_loss_over_20pct']*100:.0f}%")

    print(f"\n{'─'*70}")
    print("  SCENARIOANALYSE — deterministisk (snitt av kvartilar)")
    print(f"{'-'*70}")
    for sc_name, sc in scenario.items():
        print(f"  {sc_name.upper():5s}  Snittår {fmt_pct(sc['annual_return_assumed'])}  →  "
              f"10år total {fmt_pct(sc['10yr_total'])}  ({fmt_x(sc['10yr_total'])})  "
              f"CAGR {fmt_pct(sc['10yr_cagr'])}")

    print(f"\n{'─'*70}")
    print("  MULTISEED OPTIMERINGSRESULTATER — validering")
    print(f"{'-'*70}")
    for fname, ms in multiseed.items():
        print(f"\n  [{fname}]")
        print(f"    Antall seeds:             {ms['n']}")
        print(f"    Holdout kum. avkastning:  snitt {fmt_pct(ms['holdout_cum_net_mean'])}  "
              f"median {fmt_pct(ms['holdout_cum_net_median'])}  std {fmt_pct(ms['holdout_cum_net_std'])}")
        print(f"    Holdout max drawdown:     {fmt_pct(ms['holdout_max_dd_mean'])}")
        print(f"    Holdout andel CASH:       {ms['holdout_pct_cash_mean']*100:.0f}%")
        if ms["oos_cagr_net_mean"] is not None:
            print(f"    OOS CAGR (net):           {fmt_pct(ms['oos_cagr_net_mean'])}")

    print(f"\n{'─'*70}")
    print("  KONKLUSJON")
    print(f"{'-'*70}")
    base_cagr_med = mc_base["10yr_cagr"]["p50"]
    base_total_med = mc_base["10yr_total_return"]["p50"]
    print(f"""
  Modellen har {stats_base['n_years']} år med reell out-of-sample data (2016–2025).

  Forventet avkastning over 10 år (median, bootstrap):
    → CAGR ≈ {fmt_pct(base_cagr_med)} per år
    → Totalavkastning ≈ {fmt_pct(base_total_med)}  ({fmt_x(base_total_med)} kapital)

  Nøkkelrisiko:
    → Max drawdown historisk: {fmt_pct(stats_base['max_drawdown'])}
    → Modellen er CASH {stats_base['pct_cash_mean']*100:.0f}% av tiden
      (reduserer risiko, men begrenser oppsiden)

  Usikkerhet:
    → Kun {stats_base['n_years']} uavhengige årsresultater (bootstrap har høy usikkerhet)
    → Historisk avkastning er ikke garanti for fremtidig avkastning
    → Nordic small/mid-cap enkeltaksjer = høy konsentrasjonsrisiko per posisjon
""")
    print(sep)

    # Lagre som JSON
    out = ROOT / "runs" / "10yr_return_analysis.json"
    payload = {
        "stats_baseline": stats_base,
        "stats_tuned": stats_tuned,
        "monte_carlo_baseline": mc_base,
        "monte_carlo_tuned": mc_tuned,
        "regression_trend": trend,
        "scenarios": scenario,
        "multiseed_validation": multiseed,
        "annual_data": df_primary[["year", "return", "max_dd", "pct_cash", "position"]].to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Resultater lagret: {out}\n")


if __name__ == "__main__":
    main()
