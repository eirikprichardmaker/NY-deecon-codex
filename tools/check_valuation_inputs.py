"""
Vis DCF-inngangsverdier per kandidat fra valuation.csv og shortlist.csv.

Brukes til å kvalitetssikre at FCF, WACC og ROIC er rimelige
FØR du tar en investeringsbeslutning.

Bruk:
  python -m tools.check_valuation_inputs            (siste run)
  python -m tools.check_valuation_inputs --run runs/20260312_073204
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path


def _find_latest_run(repo: Path) -> Path:
    runs = list((repo / "runs").glob("*/shortlist.csv"))
    if not runs:
        raise FileNotFoundError("Ingen runs med shortlist.csv funnet")
    # Sort by modification time, newest last
    runs.sort(key=lambda p: p.stat().st_mtime)
    return runs[-1].parent


def _fmt(val, pct=False, decimals=2) -> str:
    try:
        v = float(val)
        if not math.isfinite(v):
            return "n/a"
        if pct:
            return f"{v:.{decimals}%}"
        if abs(v) >= 1e6:
            return f"{v/1e6:.1f}M"
        return f"{v:.{decimals}f}"
    except Exception:
        return str(val) if val else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sjekk DCF-inngangsverdier per kandidat")
    parser.add_argument("--run", help="Sti til run-mappe, f.eks. runs/20260312_073204")
    parser.add_argument("--repo", default=".", help="Rot-mappe for prosjektet")
    args = parser.parse_args()

    import pandas as pd

    repo = Path(args.repo).resolve()
    run_dir = Path(args.run).resolve() if args.run else _find_latest_run(repo)
    print(f"Run: {run_dir.name}\n")

    # Les shortlist
    shortlist_path = run_dir / "shortlist.csv"
    if not shortlist_path.exists():
        print("Ingen shortlist.csv i denne run-mappen.")
        return
    shortlist = pd.read_csv(shortlist_path)
    candidates = shortlist["ticker"].tolist()

    # Finn valuation.csv — kan ligge i en annen run
    val_path = run_dir / "valuation.csv"
    if not val_path.exists():
        # Prøv å finne siste valuation.csv i andre runs
        val_files = sorted((repo / "runs").glob("*/valuation.csv"))
        if not val_files:
            print("Ingen valuation.csv funnet.")
            return
        val_path = val_files[-1]
        print(f"(Bruker valuation fra: {val_path.parent.name})\n")
    val = pd.read_csv(val_path)

    # Les fundamentals_summary.parquet for EBITDA og normalisert FCF
    summary = None
    for d in sorted((repo / "data" / "raw").iterdir(), reverse=True):
        p = d / "fundamentals_summary.parquet"
        if p.exists():
            try:
                summary = pd.read_parquet(p)
                summary_id = "yahoo_ticker" if "yahoo_ticker" in summary.columns else "ticker"
                summary[summary_id] = summary[summary_id].astype(str).str.upper().str.strip()
                print(f"(Fundamentals summary fra: {d.name})\n")
            except Exception:
                summary = None
            break

    # Filtrer til kandidatene i shortlist
    ticker_col = "ticker" if "ticker" in val.columns else "yahoo_ticker"
    val_cands = val[val[ticker_col].isin(candidates)].copy()
    if val_cands.empty:
        # Prøv via yahoo_ticker
        if "yahoo_ticker" in val.columns:
            # Match på prefix (f.eks. TALK.ST → TALK)
            val["_base"] = val["yahoo_ticker"].str.split(".").str[0]
            val_cands = val[val["_base"].isin(candidates)].copy()

    # Beregn implisert FCF fra intrinsic_equity
    # DCF med WACC=9%, g=2%, tg=2%, n=5 → multiplier ≈ 14.57 per million FCF
    def dcf_multiplier(wacc=0.09, g=0.02, tg=0.02, n=5):
        pv = sum(((1+g)**t) / ((1+wacc)**t) for t in range(1, n+1))
        fn = (1+g)**n
        tv = fn * (1+tg) / (wacc - tg)
        return pv + tv / ((1+wacc)**n)

    W = 140
    print("=" * W)
    print(f"{'Ticker':<8} {'FCF brukt':>10} {'FCF 3y med':>10} {'EBITDA':>9} {'F/E':>5} "
          f"{'FCF-kilde':<13} {'WACC':>6} {'MoS':>7} {'ROIC':>8} {'NetGjeld':>10} {'Mkt Cap':>9}  Advarsel")
    print("-" * W)

    for ticker in candidates:
        sl_row = shortlist[shortlist["ticker"] == ticker].iloc[0] if ticker in shortlist["ticker"].values else None

        # Finn i valuation
        if ticker_col in val_cands.columns:
            vrows = val_cands[val_cands[ticker_col] == ticker]
        else:
            vrows = val_cands[val_cands.get("_base", pd.Series()) == ticker]
        vrow = vrows.iloc[0] if not vrows.empty else None

        # Finn i fundamentals_summary via yahoo_ticker
        srow = None
        if summary is not None and vrow is not None:
            yt = str(vrow.get("yahoo_ticker", "")).upper().strip()
            smatch = summary[summary[summary_id] == yt]
            if smatch.empty:
                # Fallback: match på base ticker
                smatch = summary[summary[summary_id].str.split(".").str[0] == ticker]
            srow = smatch.iloc[0] if not smatch.empty else None

        fcf_val = float(vrow["fcf_used_millions"]) if vrow is not None and "fcf_used_millions" in val.columns and math.isfinite(float(vrow.get("fcf_used_millions", float("nan")))) else float("nan")
        fcf_str = _fmt(fcf_val) + "M" if math.isfinite(fcf_val) else "n/a"
        fcf_src = str(vrow.get("fcf_source", "?"))[:11] if vrow is not None else "n/a"

        fcf_3y = float(srow["fcf_m_median_3y"]) if srow is not None and "fcf_m_median_3y" in srow and math.isfinite(float(srow.get("fcf_m_median_3y", float("nan")))) else float("nan")
        fcf_cv = float(srow.get("fcf_m_cv_3y", float("nan"))) if srow is not None else float("nan")
        nd_ebitda = float(srow.get("netdebt_ebitda_latest", float("nan"))) if srow is not None else float("nan")
        fcf_3y_str = _fmt(fcf_3y) + "M" if math.isfinite(fcf_3y) else "n/a"

        ebitda = float(srow["ebitda_m_latest"]) if srow is not None and "ebitda_m_latest" in srow and math.isfinite(float(srow.get("ebitda_m_latest", float("nan")))) else float("nan")
        ebitda_str = _fmt(ebitda) + "M" if math.isfinite(ebitda) else "n/a"

        fe_ratio = fcf_val / ebitda if math.isfinite(fcf_val) and math.isfinite(ebitda) and ebitda != 0 else float("nan")
        fe_str = _fmt(fe_ratio, decimals=1) if math.isfinite(fe_ratio) else "n/a"

        wacc_val = float(sl_row["wacc_used"]) if sl_row is not None and "wacc_used" in sl_row else 0.09
        mos = _fmt(sl_row["mos"] if sl_row is not None else None, pct=True)
        roic_spread = float(sl_row["roic_wacc_spread"]) if sl_row is not None and "roic_wacc_spread" in sl_row else float("nan")
        roic_val = roic_spread + wacc_val
        roic_str = _fmt(roic_val, pct=True)

        # Net debt and market cap (in millions)
        net_debt_raw = float(vrow.get("net_debt_used", float("nan"))) if vrow is not None else float("nan")
        net_debt_m = net_debt_raw / 1e6 if math.isfinite(net_debt_raw) else float("nan")
        net_debt_str = _fmt(net_debt_m) + "M" if math.isfinite(net_debt_m) else "n/a"

        mcap_raw = float(sl_row.get("market_cap", float("nan"))) if sl_row is not None else float("nan")
        mcap_m = mcap_raw / 1e6 if math.isfinite(mcap_raw) and mcap_raw > 1e6 else mcap_raw  # normalise
        mcap_str = _fmt(mcap_m) + "M" if math.isfinite(mcap_m) else "n/a"

        # Advarsler (samles, NetGjeld=0 legges til etter at net_debt_m er kjent)
        warnings = []
        if math.isfinite(roic_val) and abs(roic_val) > 1.0:
            warnings.append(f"ROIC={roic_val:.0%} skala?")
        if math.isfinite(fe_ratio) and abs(fe_ratio) > 3.0:
            warnings.append(f"FCF/EBITDA={fe_ratio:.1f}x valuta?")
        if math.isfinite(fcf_val) and math.isfinite(fcf_3y) and fcf_3y > 0 and fcf_val > 3 * fcf_3y:
            warnings.append(f"FCF={fcf_val:.0f}M >> 3y-median={fcf_3y:.0f}M (peak?)")

        # Net debt implausibility check
        if math.isfinite(net_debt_m) and net_debt_m == 0.0:
            warnings.append("NetGjeld=0 (mangler?)")
        # Leasing-gjeld: stor selskap med nesten ingen finansiell gjeld — kan mangle IFRS16
        if math.isfinite(net_debt_m) and math.isfinite(mcap_m) and mcap_m > 5000 and abs(net_debt_m) < 200:
            warnings.append(f"Stor selskap, lav finans.gjeld ({net_debt_m:.0f}M) — sjekk IFRS16 leasing")
        # Syklisk FCF-risiko: høy variasjonskoeffisient
        if math.isfinite(fcf_cv) and fcf_cv > 0.75:
            warnings.append(f"FCF syklisk (CV={fcf_cv:.1f})")
        warn_str = " | ".join(f"⚠️ {w}" for w in warnings) if warnings else "OK"

        print(f"{ticker:<8} {fcf_str:>10} {fcf_3y_str:>10} {ebitda_str:>9} {fe_str:>5} "
              f"{fcf_src:<13} {wacc_val:.0%}:>6 {mos:>7} {roic_str:>8} {net_debt_str:>10} {mcap_str:>9}  {warn_str}")

    print("=" * W)
    print("\nForklaring:")
    print("  FCF brukt   = siste R12 FCF brukt i DCF (millioner, lokal valuta)")
    print("  FCF 3y med  = median FCF siste 3 år (normalisert, fra fundamentals_summary)")
    print("  EBITDA      = siste årsverdi EBITDA (fra fundamentals_summary)")
    print("  F/E         = FCF / EBITDA — bør normalt være < 1.5")
    print("  ⚠️ ROIC skala = ROIC > 100% etter normalisering (sannsynlig datafeil)")
    print("  ⚠️ FCF/EBITDA = FCF > 3× EBITDA (sannsynlig valuta- eller dataproblem)")
    print("  ⚠️ FCF >> 3y  = R12 FCF er mer enn 3× høyere enn historisk median (peak-earnings?)")
    print("  ⚠️ NetGjeld=0 = Ingen netto gjeld funnet — intrinsic_equity = intrinsic_EV (mulig feil)")
    print("  ⚠️ Stor/lav gjeld = Stor selskap med <200M finansiell gjeld — kan mangle IFRS16 leasing")
    print("  ⚠️ FCF syklisk  = Variasjonskoeffisient > 0.75 over 3 år — inntjeningen er ustabil")
    print("  NetGjeld        = netto rentebærende gjeld brukt i DCF (trekkes fra EV → equity)")
    print("  Mkt Cap         = markedsverdi fra shortlist (millioner, lokal valuta)")


if __name__ == "__main__":
    main()
