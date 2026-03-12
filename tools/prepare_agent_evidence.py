"""
Prepare text evidence for Agent A (Business Quality Evaluator).

Leser Borsdata freeze-data og skriver lesbare tekstsammendrag til
data/raw/ir/{ticker}/{date}/ slik at evidence_builder kan bruke dem.

Inkluderer:
  - Kvartalstall (inntekt, EBIT, EPS, gjeld) fra reports_per_instrument
  - Holdings: insider-handler, short-posisjoner, buyback fra borsdata_proplus_freeze

Bruk:
  python -m tools.prepare_agent_evidence --asof 2026-03-12
"""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path


def _load_gz(path: Path) -> dict | list:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl_gz(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _build_ticker_to_insid(master_path: Path) -> dict[str, int]:
    """Bygg ticker -> ins_id mapping fra Nordic instrument master."""
    data = _load_gz(master_path)
    payload = data.get("payload", data) if isinstance(data, dict) else data
    ins_list = None
    if isinstance(payload, dict):
        for key in ("instruments", "instrumentList", "instrumentsList"):
            if isinstance(payload.get(key), list):
                ins_list = payload[key]
                break
        if ins_list is None:
            for v in payload.values():
                if isinstance(v, list):
                    ins_list = v
                    break
    if ins_list is None:
        return {}
    return {str(x.get("ticker", "")): int(x["insId"]) for x in ins_list if x.get("insId") and x.get("ticker")}


def _find_reports_per_instrument_dir(freeze_root: Path) -> Path | None:
    """Finn nyeste reports_per_instrument-mappe."""
    best = None
    for date_dir in sorted((freeze_root / "borsdata").iterdir(), reverse=True):
        candidate = date_dir / "raw" / "reports_per_instrument"
        if candidate.exists():
            best = candidate
            break
    return best


def _find_holdings_files(freeze_root: Path, asof: str) -> list[Path]:
    """Finn holdings-filer fra borsdata_proplus_freeze."""
    results = []
    proplus_dir = freeze_root / "borsdata_proplus_freeze"
    if not proplus_dir.exists():
        return results
    # Bruk nærmeste dato til asof
    for date_dir in sorted(proplus_dir.iterdir(), reverse=True):
        holdings_dir = date_dir / "normalized"
        if holdings_dir.exists():
            for f in holdings_dir.glob("*.jsonl.gz"):
                results.append(f)
            if results:
                break
    return results


def _format_number(val) -> str:
    try:
        v = float(val)
        if abs(v) >= 1e9:
            return f"{v/1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"{v/1e6:.1f}M"
        return f"{v:.1f}"
    except Exception:
        return str(val)


def _build_financial_summary(ins_id: int, reports_dir: Path) -> str | None:
    """Les kvartalstall for ett instrument og lag tekstsammendrag."""
    q_file = reports_dir / f"ins_{ins_id}__quarter.json.gz"
    y_file = reports_dir / f"ins_{ins_id}__year.json.gz"

    lines = []

    for label, fpath in [("Kvartal", q_file), ("Årsrapport", y_file)]:
        if not fpath.exists():
            continue
        try:
            data = _load_gz(fpath)
            payload = data.get("payload", data) if isinstance(data, dict) else data
            reports = None
            if isinstance(payload, dict):
                for key in ("reports", "reportList", "list"):
                    if isinstance(payload.get(key), list):
                        reports = payload[key]
                        break
                if reports is None:
                    for v in payload.values():
                        if isinstance(v, list) and len(v) > 0:
                            reports = v
                            break
            elif isinstance(payload, list):
                reports = payload
            if not reports:
                continue

            # Sorter etter report_End_Date (Borsdata felt), fallback til year+period
            def get_date(r):
                return (
                    r.get("report_End_Date", "")
                    or r.get("reportEndDate", "")
                    or f"{r.get('year', 0):04d}-{r.get('period', 0):02d}"
                )

            reports_sorted = sorted(reports, key=get_date, reverse=True)[:4]

            lines.append(f"\n## {label}stall (siste perioder)")
            for r in reports_sorted:
                end_date = (
                    r.get("report_End_Date", r.get("reportEndDate", ""))
                    or f"{r.get('year', '?')}-Q{r.get('period', '?')}"
                )
                # Borsdata field names use mixed_Case_With_Underscores
                revenues = r.get("revenues")
                ebit = r.get("operating_Income", r.get("operatingIncome"))
                net_income = r.get("profit_Before_Tax", r.get("profitBeforeTax", r.get("profit_To_Equity_Holders")))
                eps = r.get("earnings_Per_Share", r.get("earningsPerShare"))

                line = f"  {end_date}: "
                parts = []
                if revenues is not None:
                    parts.append(f"Omsetning={_format_number(revenues)}")
                if ebit is not None:
                    parts.append(f"EBIT={_format_number(ebit)}")
                if net_income is not None:
                    parts.append(f"Nettoresultat={_format_number(net_income)}")
                if eps is not None:
                    parts.append(f"EPS={eps}")
                line += ", ".join(parts) if parts else "(ingen tall)"
                lines.append(line)
        except Exception as e:
            lines.append(f"  (feil ved lesing av {label}: {e})")

    return "\n".join(lines) if lines else None


def _build_holdings_summary(ticker: str, ins_id: int, holdings_files: list[Path]) -> str | None:
    """Les insider-handler, shorts og buyback for ett instrument."""
    lines = []

    for fpath in holdings_files:
        try:
            rows = _load_jsonl_gz(fpath)
            fname = fpath.stem.replace(".jsonl", "")

            # Filtrer på ins_id eller ticker
            relevant = [r for r in rows if r.get("insId") == ins_id or str(r.get("ticker", "")).upper() == ticker.upper()]
            if not relevant:
                continue

            relevant_sorted = sorted(relevant, key=lambda r: r.get("date", r.get("transactionDate", "")), reverse=True)[:5]

            lines.append(f"\n## {fname.replace('_', ' ').title()} (siste transaksjoner)")
            for r in relevant_sorted:
                date = r.get("date", r.get("transactionDate", "ukjent"))
                name = r.get("name", r.get("ownerName", ""))
                shares = r.get("shares", r.get("numberOfShares", ""))
                value = r.get("value", r.get("transactionValue", ""))
                ttype = r.get("type", r.get("transactionType", ""))

                line = f"  {date}: {name} — {ttype}"
                if shares:
                    line += f" {_format_number(shares)} aksjer"
                if value:
                    line += f" ({_format_number(value)} NOK)"
                lines.append(line)
        except Exception:
            pass

    return "\n".join(lines) if lines else None


def prepare_evidence(asof: str, repo_root: Path, explicit_tickers: list[str] | None = None) -> None:
    freeze_root = repo_root / "data" / "freeze"
    out_root = repo_root / "data" / "raw" / "ir"

    # Last Nordic instrument master
    master_path = (
        repo_root / "data" / "freeze" / "borsdata_proplus" / asof / "meta_global_capture" / "instrument_master_nordic.json.gz"
    )
    # Fallback: finn nyeste
    if not master_path.exists():
        for d in sorted((freeze_root / "borsdata_proplus").iterdir(), reverse=True):
            candidate = d / "meta_global_capture" / "instrument_master_nordic.json.gz"
            if candidate.exists():
                master_path = candidate
                break

    if not master_path.exists():
        print("FEIL: Finner ikke instrument_master_nordic.json.gz")
        return

    ticker_to_insid = _build_ticker_to_insid(master_path)
    print(f"Instrument master: {len(ticker_to_insid)} Nordic tickers")

    # Finn reports_per_instrument
    reports_dir = _find_reports_per_instrument_dir(freeze_root)
    if reports_dir:
        print(f"Reports: {reports_dir}")
    else:
        print("ADVARSEL: Ingen reports_per_instrument funnet")

    # Finn holdings
    holdings_files = _find_holdings_files(freeze_root, asof)
    print(f"Holdings-filer: {len(holdings_files)}")

    # Finn hvilke tickers vi skal generere evidence for
    if explicit_tickers:
        # Eksplisitt liste overstyrer alt
        tickers = [t for t in explicit_tickers if t in ticker_to_insid]
        missing = [t for t in explicit_tickers if t not in ticker_to_insid]
        if missing:
            print(f"ADVARSEL: Disse tickerne ble ikke funnet i master: {missing}")
        print(f"Eksplisitte tickers: {tickers}")
    else:
        import csv
        tickers = None
        runs_root = repo_root / "runs"
        if runs_root.exists():
            # Finn nyeste shortlist der minst én ticker finnes i master
            for run_dir in sorted(runs_root.iterdir(), reverse=True):
                candidate = run_dir / "shortlist.csv"
                if candidate.exists():
                    with open(candidate, encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    candidate_tickers = [r["ticker"] for r in rows if r.get("ticker") and r["ticker"] in ticker_to_insid]
                    if candidate_tickers:
                        tickers = candidate_tickers
                        print(f"Shortlist ({candidate}): {tickers}")
                        break
        if not tickers:
            tickers = list(ticker_to_insid.keys())[:20]
            print(f"Ingen shortlist funnet — bruker topp {len(tickers)} tickers fra master")

    # Generer evidence for hver ticker
    for ticker in tickers:
        ins_id = ticker_to_insid.get(ticker)
        if not ins_id:
            print(f"  {ticker}: ingen ins_id funnet — hopper over")
            continue

        out_dir = out_root / ticker / asof
        out_dir.mkdir(parents=True, exist_ok=True)

        sections = [f"# Borsdata-data for {ticker} (ins_id={ins_id}, asof={asof})\n"]

        if reports_dir:
            fin = _build_financial_summary(ins_id, reports_dir)
            if fin:
                sections.append(fin)
            else:
                sections.append(f"\n## Finansdata\n  (ingen data funnet for ins_id={ins_id})")

        if holdings_files:
            hold = _build_holdings_summary(ticker, ins_id, holdings_files)
            if hold:
                sections.append(hold)

        text = "\n".join(sections)
        out_file = out_dir / "borsdata_summary.txt"
        out_file.write_text(text, encoding="utf-8")
        print(f"  {ticker}: skrevet {out_file} ({len(text)} tegn)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forbered tekstbevis for Agent A fra Borsdata freeze-data")
    ap.add_argument("--asof", default=datetime.today().strftime("%Y-%m-%d"))
    ap.add_argument("--repo", default=".")
    ap.add_argument("--tickers", default="", help="Kommaseparert liste med tickers, f.eks. TALK,VEI,CAMBI")
    args = ap.parse_args()

    explicit = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    prepare_evidence(asof=args.asof, repo_root=Path(args.repo), explicit_tickers=explicit)


if __name__ == "__main__":
    main()
