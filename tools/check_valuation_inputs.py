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
    runs = sorted((repo / "runs").glob("*/shortlist.csv"))
    if not runs:
        raise FileNotFoundError("Ingen runs med shortlist.csv funnet")
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

    print("=" * 95)
    print(f"{'Ticker':<8} {'FCF brukt':>10} {'FCF-kilde':<14} {'WACC':>6} {'Net gjeld':>10} "
          f"{'Intrinsic':>12} {'MoS':>7} {'ROIC':>8} {'Advarsel'}")
    print("-" * 95)

    for ticker in candidates:
        sl_row = shortlist[shortlist["ticker"] == ticker].iloc[0] if ticker in shortlist["ticker"].values else None

        # Finn i valuation
        if ticker_col in val_cands.columns:
            vrows = val_cands[val_cands[ticker_col] == ticker]
        else:
            vrows = val_cands[val_cands.get("_base", pd.Series()) == ticker]

        vrow = vrows.iloc[0] if not vrows.empty else None

        fcf = _fmt(vrow["fcf_used_millions"] if vrow is not None else None) + "M" if vrow is not None and "fcf_used_millions" in val.columns else "n/a"
        fcf_src = str(vrow.get("fcf_source", "?"))[:12] if vrow is not None else "n/a"
        wacc = _fmt(vrow["wacc_used"] if vrow is not None else None, pct=True) if vrow is not None else "n/a"
        net_debt = _fmt(vrow["net_debt_used"] if vrow is not None else None) if vrow is not None else "n/a"
        intrinsic = _fmt(sl_row["intrinsic_value"] if sl_row is not None else None)
        mos = _fmt(sl_row["mos"] if sl_row is not None else None, pct=True)
        roic_spread = float(sl_row["roic_wacc_spread"]) if sl_row is not None and "roic_wacc_spread" in sl_row else float("nan")
        wacc_val = float(sl_row["wacc_used"]) if sl_row is not None and "wacc_used" in sl_row else 0.09
        roic_val = roic_spread + wacc_val
        roic_str = _fmt(roic_val, pct=True)

        # Advarsler
        warnings = []
        if math.isfinite(roic_val) and abs(roic_val) > 1.0:
            warnings.append(f"⚠️ ROIC={roic_val:.0%} sannsynlig feil")
        if vrow is not None and "fcf_used_millions" in vrow:
            try:
                fcf_m = float(vrow["fcf_used_millions"])
                mult = dcf_multiplier(wacc=wacc_val)
                implied_iv = fcf_m * mult * 1e6
                mcap = float(sl_row["market_cap"]) if sl_row is not None else float("nan")
                if math.isfinite(implied_iv) and math.isfinite(mcap) and mcap > 0:
                    ratio = implied_iv / mcap
                    if ratio > 10:
                        warnings.append(f"⚠️ FCF={fcf_m:.1f}M gir {ratio:.0f}x mcap")
            except Exception:
                pass

        warn_str = " | ".join(warnings) if warnings else "OK"

        print(f"{ticker:<8} {fcf:>10} {fcf_src:<14} {wacc:>6} {net_debt:>10} "
              f"{intrinsic:>12} {mos:>7} {roic_str:>8}  {warn_str}")

    print("=" * 95)
    print("\nForklaring:")
    print("  FCF brukt   = grunnlag for DCF (millioner, lokal valuta)")
    print("  ROIC        = ROIC etter normalisering (> 100% = sannsynlig datafeil)")
    print("  ⚠️ ROIC-feil = skalaproblemer i KPI-data fra Borsdata")
    print("  ⚠️ FCF-ratio = FCF impliserer urealistisk høy intrinsic value vs markedsverdi")


if __name__ == "__main__":
    main()
