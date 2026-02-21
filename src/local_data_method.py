from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

ITEM_ORDER = [
    "instrument_universliste",
    "daglige_priser_aksjer",
    "daglige_priser_indekser",
    "rapportdata_kvartal_ar",
    "ttm_derivat_grunnlag",
    "nokkeldata_kpi",
    "utbytte_corporate_actions",
    "metadata_reporting_lag",
]


@dataclass(frozen=True)
class SemanticRequirement:
    name: str
    aliases: Tuple[str, ...]


def _norm_col(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_")


def _column_set(df: pd.DataFrame) -> set[str]:
    return {_norm_col(c) for c in df.columns}


def _missing_semantics(df: pd.DataFrame, reqs: Sequence[SemanticRequirement]) -> list[str]:
    cols = _column_set(df)
    missing: list[str] = []
    for req in reqs:
        if not any(alias in cols for alias in req.aliases):
            missing.append(req.name)
    return missing


def _resolve_asof_dir(data_dir: Path, asof: Optional[str]) -> Path:
    raw_root = data_dir / "raw"
    if asof:
        target = raw_root / asof
        if not target.exists():
            raise FileNotFoundError(f"Mangler data snapshot: {target}")
        return target

    dated: list[Tuple[datetime, Path]] = []
    if raw_root.exists():
        for p in raw_root.iterdir():
            if not p.is_dir():
                continue
            try:
                dt = datetime.strptime(p.name, "%Y-%m-%d")
            except ValueError:
                continue
            dated.append((dt, p))
    if not dated:
        raise FileNotFoundError(f"Fant ingen daterte snapshots under {raw_root}")
    return sorted(dated, key=lambda x: x[0])[-1][1]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _first_non_empty(paths: Iterable[Path]) -> tuple[Optional[Path], Optional[pd.DataFrame], str]:
    last_error = ""
    for path in paths:
        if not path.exists():
            continue
        try:
            df = _read_table(path)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = str(exc)
            continue
        if len(df) > 0:
            return path, df, ""
        last_error = "empty table"
    return None, None, last_error


def _manifest_frame(raw_asof_dir: Path) -> pd.DataFrame:
    manifest_path = raw_asof_dir / "manifest.parquet"
    if not manifest_path.exists():
        return pd.DataFrame(columns=["dataset", "status", "rows", "file_path", "source"])
    try:
        df = pd.read_parquet(manifest_path)
    except Exception:
        return pd.DataFrame(columns=["dataset", "status", "rows", "file_path", "source"])

    out = df.copy()
    for col, default in {
        "dataset": "",
        "status": "",
        "rows": 0,
        "file_path": "",
        "source": "",
    }.items():
        if col not in out.columns:
            out[col] = default
    out["dataset"] = out["dataset"].fillna("").astype(str)
    out["status"] = out["status"].fillna("").astype(str)
    out["file_path"] = out["file_path"].fillna("").astype(str)
    out["source"] = out["source"].fillna("").astype(str)
    out["rows"] = pd.to_numeric(out["rows"], errors="coerce").fillna(0).astype(int)
    return out


def _manifest_stats(manifest: pd.DataFrame, datasets: Sequence[str]) -> tuple[int, int, str]:
    if manifest.empty:
        return 0, 0, ""
    mask = manifest["dataset"].isin(datasets)
    subset = manifest.loc[mask].copy()
    if subset.empty:
        return 0, 0, ""
    ok = subset["status"].isin(["ok", "cached"])  # completed
    n_files = int(ok.sum())
    n_rows = int(subset.loc[ok, "rows"].sum())
    sample = subset.loc[ok, "file_path"].head(3).tolist()
    source = "manifest:" + (", ".join(sample) if sample else "(ingen file_path)")
    return n_files, n_rows, source


def _inventory_row(
    freeze_item: str,
    status: str,
    source: str,
    rows: int,
    required_fields: Sequence[str],
    missing_fields: Sequence[str],
    comment: str,
) -> dict:
    return {
        "freeze_item": freeze_item,
        "status": status,
        "source": source,
        "rows": int(rows),
        "required_fields": ",".join(required_fields),
        "missing_fields": ",".join(missing_fields),
        "comment": comment,
    }


def _source_row(freeze_item: str, path_or_pattern: str, read_hint: str, note: str) -> dict:
    return {
        "freeze_item": freeze_item,
        "path_or_pattern": path_or_pattern,
        "read_hint": read_hint,
        "note": note,
    }


def build_local_inventory(data_dir: Path, asof: Optional[str] = None) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    data_dir = data_dir.resolve()
    raw_asof_dir = _resolve_asof_dir(data_dir, asof)
    asof_value = raw_asof_dir.name
    manifest = _manifest_frame(raw_asof_dir)

    rows: list[dict] = []
    sources: list[dict] = []

    # 1) Instrument/universe
    req_universe = [
        SemanticRequirement("instrument_id", ("instrument_id", "ins_id", "insid", "id")),
        SemanticRequirement("ticker", ("ticker", "symbol", "yahoo_ticker")),
        SemanticRequirement("market", ("market", "market_id", "marketid", "country", "country_id", "countryid", "exchange")),
        SemanticRequirement("currency", ("currency", "stockpricecurrency", "reportcurrency", "stock_price_currency", "report_currency")),
        SemanticRequirement("sector_or_bransje", ("sector", "sectorid", "bransje", "branch", "branchid", "industry")),
    ]
    repo_root = data_dir.parent if data_dir.name.lower() == "data" else data_dir
    uni_candidates = [
        raw_asof_dir / "instruments_dump.csv",
        raw_asof_dir / "ticker_map.csv",
        repo_root / "config" / "tickers_with_insid_clean.csv",
        repo_root / "config" / "tickers_with_insid.csv",
    ]
    uni_path, uni_df, uni_err = _first_non_empty(uni_candidates)
    if uni_df is not None and uni_path is not None:
        missing = _missing_semantics(uni_df, req_universe)
        status = "ready" if not missing else "partial"
        rows.append(
            _inventory_row(
                "instrument_universliste",
                status,
                str(uni_path),
                len(uni_df),
                [r.name for r in req_universe],
                missing,
                "Direkte lokal tabell funnet.",
            )
        )
        sources.append(_source_row("instrument_universliste", str(uni_path), "pd.read_csv", "Primarkilde for univers."))
    else:
        rows.append(
            _inventory_row(
                "instrument_universliste",
                "missing",
                "",
                0,
                [r.name for r in req_universe],
                [r.name for r in req_universe],
                f"Ingen lokal universfil funnet ({uni_err}).",
            )
        )

    # 2) Daily stock prices
    req_stock_px = [
        SemanticRequirement("date", ("date", "pricedate", "price_date")),
        SemanticRequirement("close_or_adj_close", ("close", "adj_close", "adjusted_close", "price", "last")),
        SemanticRequirement("volume", ("volume",)),
    ]
    stock_candidates = [
        raw_asof_dir / "prices.parquet",
        data_dir / "raw" / "prices" / "prices_panel.parquet",
        data_dir / "processed" / "prices.parquet",
    ]
    px_path, px_df, px_err = _first_non_empty(stock_candidates)
    if px_df is not None and px_path is not None:
        missing = _missing_semantics(px_df, req_stock_px)
        status = "ready" if not missing else "partial"
        rows.append(
            _inventory_row(
                "daglige_priser_aksjer",
                status,
                str(px_path),
                len(px_df),
                [r.name for r in req_stock_px],
                missing,
                "Bruk denne tabellen som lokal kilde for teknisk gating/beta/retur.",
            )
        )
        sources.append(_source_row("daglige_priser_aksjer", str(px_path), "pd.read_parquet", "Anbefalt lokal pris-kilde."))
    else:
        n_files, n_rows, manifest_source = _manifest_stats(manifest, ["prices"])
        status = "partial" if n_files > 0 else "missing"
        comment = "Prisfiler ligger i manifest (partitionert per instrument)." if n_files > 0 else f"Ingen pris-kilde funnet ({px_err})."
        rows.append(
            _inventory_row(
                "daglige_priser_aksjer",
                status,
                manifest_source,
                n_rows,
                [r.name for r in req_stock_px],
                [r.name for r in req_stock_px] if status == "missing" else [],
                comment,
            )
        )
        if n_files > 0:
            sources.append(
                _source_row(
                    "daglige_priser_aksjer",
                    str(raw_asof_dir / "prices*"),
                    "pd.read_parquet + concat",
                    "Manifest peker til filbaner i raw snapshot.",
                )
            )

    # 3) Daily index prices
    req_idx_px = [
        SemanticRequirement("index_symbol", ("index_symbol", "symbol", "ticker", "index")),
        SemanticRequirement("date", ("date", "pricedate", "price_date")),
        SemanticRequirement("close", ("close", "adj_close", "price", "last")),
    ]
    idx_candidates = [
        raw_asof_dir / "index_prices.parquet",
        data_dir / "processed" / "index_prices.parquet",
        data_dir / "raw" / "indices" / "index_prices.parquet",
    ]
    idx_path, idx_df, idx_err = _first_non_empty(idx_candidates)
    if idx_df is not None and idx_path is not None:
        missing = _missing_semantics(idx_df, req_idx_px)
        status = "ready" if not missing else "partial"
        rows.append(_inventory_row("daglige_priser_indekser", status, str(idx_path), len(idx_df), [r.name for r in req_idx_px], missing, "Lokal indeksserie funnet."))
        sources.append(_source_row("daglige_priser_indekser", str(idx_path), "pd.read_parquet", "Indeksregime (MA200/MAD)."))
    else:
        rows.append(
            _inventory_row(
                "daglige_priser_indekser",
                "missing",
                "",
                0,
                [r.name for r in req_idx_px],
                [r.name for r in req_idx_px],
                f"Ingen lokal indeks-parquet funnet ({idx_err}).",
            )
        )

    # 4) As-reported reports (quarter/year)
    req_reports = [
        SemanticRequirement("period_end", ("period_end", "date", "report_date")),
        SemanticRequirement("report_period", ("report_period", "report_type", "period")),
        SemanticRequirement("available_date", ("available_date", "published_date", "update_date")),
        SemanticRequirement("regnskapsposter", ("metric", "item", "value", "revenue", "ebit")),
    ]
    reports_candidates = [
        raw_asof_dir / "fundamentals_history.parquet",
        data_dir / "golden" / "fundamentals_history.parquet",
    ]
    rep_path, rep_df, rep_err = _first_non_empty(reports_candidates)
    if rep_df is not None and rep_path is not None:
        missing = _missing_semantics(rep_df, req_reports)
        status = "ready" if not missing else "partial"
        rows.append(_inventory_row("rapportdata_kvartal_ar", status, str(rep_path), len(rep_df), [r.name for r in req_reports], missing, "As-reported grunnlag funnet."))
        sources.append(_source_row("rapportdata_kvartal_ar", str(rep_path), "pd.read_parquet", "Historisk regnskapsgrunnlag."))
    else:
        n_files, n_rows, manifest_source = _manifest_stats(manifest, ["reports_y", "reports_q"])
        status = "partial" if n_files > 0 else "missing"
        rows.append(
            _inventory_row(
                "rapportdata_kvartal_ar",
                status,
                manifest_source,
                n_rows,
                [r.name for r in req_reports],
                [r.name for r in req_reports] if status == "missing" else [],
                "Finnes i manifest per instrument." if status == "partial" else f"Mangler lokal rapporttabell ({rep_err}).",
            )
        )
        if n_files > 0:
            sources.append(_source_row("rapportdata_kvartal_ar", str(raw_asof_dir / "reports_*"), "pd.read_parquet + concat", "Bruk manifest.file_path."))

    # 5) TTM derivat / buildable basis
    req_ttm = [
        SemanticRequirement("as_of", ("as_of", "asof", "available_date", "date")),
        SemanticRequirement("quarter_basis", ("quarter", "report_type", "period", "metric")),
    ]
    ttm_candidates = [
        raw_asof_dir / "reports_r12.parquet",
        data_dir / "processed" / "fundamentals_r12.parquet",
    ]
    ttm_path, ttm_df, _ = _first_non_empty(ttm_candidates)
    if ttm_df is not None and ttm_path is not None:
        missing = _missing_semantics(ttm_df, req_ttm)
        status = "ready" if not missing else "partial"
        rows.append(_inventory_row("ttm_derivat_grunnlag", status, str(ttm_path), len(ttm_df), [r.name for r in req_ttm], missing, "Direkte TTM-grunnlag funnet."))
        sources.append(_source_row("ttm_derivat_grunnlag", str(ttm_path), "pd.read_parquet", "Brukes for as-of TTM."))
    else:
        n_files, n_rows, manifest_source = _manifest_stats(manifest, ["reports_r12", "reports_q"])
        status = "partial" if n_files > 0 else "missing"
        rows.append(
            _inventory_row(
                "ttm_derivat_grunnlag",
                status,
                manifest_source,
                n_rows,
                [r.name for r in req_ttm],
                [r.name for r in req_ttm] if status == "missing" else [],
                "Kan bygges fra manifest-filer per instrument." if status == "partial" else "Ingen lokal TTM-kilde oppdaget.",
            )
        )
        if n_files > 0:
            sources.append(_source_row("ttm_derivat_grunnlag", str(raw_asof_dir / "reports_r12*"), "pd.read_parquet + concat", "Alternativt bygg fra reports_q."))

    # 6) KPIs (optional)
    req_kpi = [
        SemanticRequirement("kpi_identifier", ("kpi_id", "metric", "name")),
        SemanticRequirement("value", ("value", "val", "kpi")),
    ]
    kpi_candidates = [
        raw_asof_dir / "kpis.parquet",
        data_dir / "processed" / "kpis.parquet",
    ]
    kpi_path, kpi_df, _ = _first_non_empty(kpi_candidates)
    if kpi_df is not None and kpi_path is not None:
        missing = _missing_semantics(kpi_df, req_kpi)
        status = "ready" if not missing else "partial"
        rows.append(_inventory_row("nokkeldata_kpi", status, str(kpi_path), len(kpi_df), [r.name for r in req_kpi], missing, "Valgfri KPI-tabell er tilgjengelig."))
        sources.append(_source_row("nokkeldata_kpi", str(kpi_path), "pd.read_parquet", "Rask feature-bygging."))
    else:
        n_files, n_rows, manifest_source = _manifest_stats(manifest, ["kpis"])
        status = "partial" if n_files > 0 else "missing"
        rows.append(
            _inventory_row(
                "nokkeldata_kpi",
                status,
                manifest_source,
                n_rows,
                [r.name for r in req_kpi],
                [r.name for r in req_kpi] if status == "missing" else [],
                "KPI finnes i manifest per instrument." if status == "partial" else "Ingen KPI-data funnet lokalt.",
            )
        )
        if n_files > 0:
            sources.append(_source_row("nokkeldata_kpi", str(raw_asof_dir / "kpis*"), "pd.read_parquet + concat", "Bruk manifest.file_path."))

    # 7) Dividends + corporate actions
    req_actions = [
        SemanticRequirement("ex_date", ("ex_date", "date")),
        SemanticRequirement("amount", ("amount", "value", "dividend")),
        SemanticRequirement("currency", ("currency",)),
        SemanticRequirement("type", ("type", "action_type", "event")),
    ]
    actions_candidates = [
        raw_asof_dir / "actions.parquet",
        raw_asof_dir / "dividends.parquet",
        raw_asof_dir / "splits.parquet",
    ]
    act_path, act_df, _ = _first_non_empty(actions_candidates)
    if act_df is not None and act_path is not None:
        missing = _missing_semantics(act_df, req_actions)
        status = "ready" if not missing else "partial"
        rows.append(_inventory_row("utbytte_corporate_actions", status, str(act_path), len(act_df), [r.name for r in req_actions], missing, "Corporate actions-tabell funnet."))
        sources.append(_source_row("utbytte_corporate_actions", str(act_path), "pd.read_parquet", "Total-return og justeringer."))
    else:
        n_files, n_rows, manifest_source = _manifest_stats(manifest, ["dividends", "splits"])
        status = "partial" if n_files > 0 else "missing"
        rows.append(
            _inventory_row(
                "utbytte_corporate_actions",
                status,
                manifest_source,
                n_rows,
                [r.name for r in req_actions],
                [r.name for r in req_actions] if status == "missing" else [],
                "Finnes i manifest per instrument." if status == "partial" else "Ingen actions-data funnet lokalt.",
            )
        )
        if n_files > 0:
            sources.append(_source_row("utbytte_corporate_actions", str(raw_asof_dir / "dividends*"), "pd.read_parquet + concat", "Kombiner med splits* ved behov."))

    # 8) Reporting lag metadata
    req_lag = [
        SemanticRequirement("available_or_update_date", ("available_date", "published_date", "update_date", "fetched_at")),
    ]
    lag_rows = 0
    lag_source = ""
    lag_missing = [r.name for r in req_lag]
    lag_status = "missing"

    if rep_df is not None and rep_path is not None:
        lag_missing = _missing_semantics(rep_df, req_lag)
        lag_rows = len(rep_df)
        lag_source = str(rep_path)
        lag_status = "ready" if not lag_missing else "partial"
    else:
        n_files, n_rows, manifest_source = _manifest_stats(manifest, ["reports_y", "reports_q", "reports_r12"])
        if n_files > 0:
            lag_status = "partial"
            lag_rows = n_rows
            lag_source = manifest_source
            lag_missing = []

    rows.append(
        _inventory_row(
            "metadata_reporting_lag",
            lag_status,
            lag_source,
            lag_rows,
            [r.name for r in req_lag],
            lag_missing,
            "Brukes til realistisk as-of (delay/available_date).",
        )
    )
    if lag_source:
        hint = "pd.read_parquet" if lag_source.endswith(".parquet") else "pd.read_parquet + concat"
        sources.append(_source_row("metadata_reporting_lag", lag_source, hint, "Leses fra rapportkilden."))

    inventory = pd.DataFrame(rows)
    inventory["_order"] = inventory["freeze_item"].map({k: i for i, k in enumerate(ITEM_ORDER)})
    inventory = inventory.sort_values(["_order", "freeze_item"]).drop(columns=["_order"]).reset_index(drop=True)

    source_df = pd.DataFrame(sources)
    if not source_df.empty:
        source_df = source_df.drop_duplicates(subset=["freeze_item", "path_or_pattern"]).reset_index(drop=True)

    return asof_value, inventory, source_df


def _build_inventory_md(asof: str, inventory: pd.DataFrame, source_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append(f"# Lokal data-inventory ({asof})")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("| freeze_item | status | rows | source | missing_fields |")
    lines.append("|---|---|---:|---|---|")
    for _, r in inventory.iterrows():
        lines.append(
            "| {freeze_item} | {status} | {rows} | {source} | {missing_fields} |".format(
                freeze_item=r["freeze_item"],
                status=r["status"],
                rows=int(r["rows"]),
                source=str(r["source"]).replace("|", "\\|"),
                missing_fields=str(r["missing_fields"]),
            )
        )

    lines.append("")
    lines.append("## Lokal hentemetode")
    lines.append("")
    lines.append("1. Oppdater/frys data lokalt (inkrementelt):")
    lines.append("```powershell")
    lines.append(f"python -m src.freeze_golden_fundamentals_history --asof {asof} --datasets prices,reports_y,reports_q,reports_r12,kpis,dividends,splits --markets NO,SE,DK,FI --skip-existing true --refetch-invalid-cache true")
    lines.append("```")
    lines.append("2. Regenerer lokal inventory + source map:")
    lines.append("```powershell")
    lines.append(f"python -m src.local_data_method --asof {asof} --data-dir data --out-dir runs/{asof}")
    lines.append("```")
    lines.append("3. Les lokale kilder fra `local_data_sources.csv` (path_or_pattern + read_hint).")

    if not source_df.empty:
        lines.append("")
        lines.append("## Kildekart")
        lines.append("")
        lines.append("| freeze_item | path_or_pattern | read_hint | note |")
        lines.append("|---|---|---|---|")
        for _, r in source_df.iterrows():
            lines.append(
                "| {freeze_item} | {path} | {hint} | {note} |".format(
                    freeze_item=r["freeze_item"],
                    path=str(r["path_or_pattern"]).replace("|", "\\|"),
                    hint=r["read_hint"],
                    note=str(r["note"]).replace("|", "\\|"),
                )
            )

    lines.append("")
    return "\n".join(lines)


def write_local_inventory(asof: str, out_dir: Path, inventory: pd.DataFrame, source_df: pd.DataFrame) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "local_data_inventory.csv"
    sources_path = out_dir / "local_data_sources.csv"
    md_path = out_dir / "local_data_inventory.md"

    inventory.to_csv(csv_path, index=False)
    source_df.to_csv(sources_path, index=False)
    md_path.write_text(_build_inventory_md(asof, inventory, source_df), encoding="utf-8")

    return {
        "inventory_csv": csv_path,
        "sources_csv": sources_path,
        "inventory_md": md_path,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bygg lokal data-inventory + source-map for freeze-itemene.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--asof", default=None, help="YYYY-MM-DD. Default: nyeste snapshot under data/raw/")
    p.add_argument("--out-dir", default=None, help="Default: runs/<asof>")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    asof, inventory, sources = build_local_inventory(data_dir=data_dir, asof=args.asof)

    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / asof
    outputs = write_local_inventory(asof=asof, out_dir=out_dir, inventory=inventory, source_df=sources)

    ready = int((inventory["status"] == "ready").sum())
    partial = int((inventory["status"] == "partial").sum())
    missing = int((inventory["status"] == "missing").sum())
    print(f"OK: asof={asof}")
    print(f"OK: ready={ready} partial={partial} missing={missing}")
    print(f"OK: inventory -> {outputs['inventory_csv']}")
    print(f"OK: sources   -> {outputs['sources_csv']}")
    print(f"OK: report    -> {outputs['inventory_md']}")


if __name__ == "__main__":
    main()
