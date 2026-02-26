from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)

RAW_FACT_COLUMNS = [
    "source_doc_id",
    "company_id",
    "ticker",
    "period_end",
    "concept",
    "value",
    "decimals",
    "unit",
    "currency",
    "context_ref",
    "entity",
    "dimensions",
    "taxonomy_ref",
    "raw_label",
    "source_type",
]

LONG_COLUMNS = [
    "company_id",
    "period_end",
    "statement",
    "field_id",
    "value",
    "unit",
    "currency",
    "source_doc_id",
    "confidence",
    "raw_tag",
    "raw_label",
    "period_type",
    "original_value",
    "original_unit",
    "normalized_value",
    "normalized_unit",
]

ARELLE_VALIDATION_COLUMNS = ["severity", "code", "message", "location", "source_doc_id"]
SOURCE_PRIORITY = {"esef_zip": 3, "esef": 2, "esef_xhtml": 2, "pdf": 1}


@dataclass
class AgentConfig:
    report_sources_csv: Path
    canonical_schema: Path
    mapping_default: Path
    mapping_overrides_dir: Path
    tolerance_balance: float
    tolerance_cashflow: float
    timeout_sec: int
    user_agent: str


def load_agent_config(path: Path) -> AgentConfig:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent

    def _p(key: str, default: str) -> Path:
        return (root / cfg.get(key, default)).resolve()

    return AgentConfig(
        report_sources_csv=_p("report_sources_csv", "report_sources.csv"),
        canonical_schema=_p("canonical_schema", "canonical_schema.yaml"),
        mapping_default=_p("mapping_default", "mappings/esef_ifrs_default.yaml"),
        mapping_overrides_dir=_p("mapping_overrides_dir", "mappings/company_overrides"),
        tolerance_balance=float(cfg.get("tolerance_balance", 0.02)),
        tolerance_cashflow=float(cfg.get("tolerance_cashflow", 0.08)),
        timeout_sec=int(cfg.get("timeout_sec", 45)),
        user_agent=str(cfg.get("user_agent", "financials-agent/1.0")),
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_filename(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        return "report.bin"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def read_report_sources(csv_path: Path, max_companies: int | None = None) -> pd.DataFrame:
    src = pd.read_csv(csv_path)
    required = ["company_id", "ticker", "year", "report_url", "source_type"]
    missing = [c for c in required if c not in src.columns]
    if missing:
        raise ValueError(f"report_sources.csv missing columns: {missing}")
    src["year"] = src["year"].astype(int)
    src = src.sort_values(["company_id", "year", "ticker"], kind="mergesort").reset_index(drop=True)
    if max_companies:
        keep = src["company_id"].drop_duplicates().head(max_companies)
        src = src[src["company_id"].isin(keep)].copy()
    return src


def download_reports(
    sources: pd.DataFrame,
    raw_dir: Path,
    dry_run: bool,
    force: bool,
    timeout_sec: int,
    user_agent: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    docs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for row in sources.itertuples(index=False):
        doc_id = f"{row.company_id}_{row.year}_{hashlib.md5(str(row.report_url).encode()).hexdigest()[:10]}"
        doc_dir = raw_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        target_name = _safe_filename(str(row.report_url))
        target_file = doc_dir / target_name

        metadata = {
            "doc_id": doc_id,
            "company_id": row.company_id,
            "ticker": row.ticker,
            "year": int(row.year),
            "report_url": row.report_url,
            "source_type": row.source_type,
            "downloaded_at": pd.Timestamp.utcnow().isoformat(),
            "status": "skipped" if dry_run else "pending",
            "file_name": target_name,
            "file_path": str(target_file),
            "sha256": None,
            "bytes": None,
        }

        if dry_run:
            (doc_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            docs.append(metadata)
            continue

        if target_file.exists() and not force:
            metadata["sha256"] = _sha256_file(target_file)
            metadata["bytes"] = target_file.stat().st_size
            metadata["status"] = "cached"
            (doc_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            docs.append(metadata)
            continue

        try:
            req = Request(row.report_url, headers={"User-Agent": user_agent})
            with urlopen(req, timeout=timeout_sec) as resp:
                payload = resp.read()
            target_file.write_bytes(payload)
            metadata["sha256"] = _sha256_file(target_file)
            metadata["bytes"] = target_file.stat().st_size
            metadata["status"] = "downloaded"
        except Exception as exc:  # pragma: no cover - network-dependent
            metadata["status"] = "failed"
            metadata["error"] = str(exc)
            errors.append({"severity": "error", "issue_type": "download_failed", "doc_id": doc_id, "details": str(exc)})

        (doc_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        docs.append(metadata)

    docs_df = pd.DataFrame(docs)
    if not docs_df.empty:
        docs_df.to_parquet(raw_dir / "download_index.parquet", index=False)
    return docs_df, errors


def _extract_ixbrl_facts_from_text(text: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"<ix:(?:nonFraction|nonNumeric)\b([^>]*)>(.*?)</ix:(?:nonFraction|nonNumeric)>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    attrs_pattern = re.compile(r"([\w:.-]+)=\"(.*?)\"")
    facts: list[dict[str, Any]] = []
    for attrs_blob, inner in pattern.findall(text):
        attrs = {k: v for k, v in attrs_pattern.findall(attrs_blob)}
        concept = attrs.get("name", "")
        value_str = re.sub(r"<[^>]+>", "", inner).strip()
        try:
            value = float(value_str.replace(" ", "").replace(",", ""))
        except ValueError:
            value = np.nan
        facts.append(
            {
                "concept": concept,
                "value": value,
                "decimals": attrs.get("decimals"),
                "unit": attrs.get("unitRef"),
                "context_ref": attrs.get("contextRef"),
                "raw_label": concept.split(":")[-1],
            }
        )
    return facts


def _parse_esef_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".zip":
        out: list[dict[str, Any]] = []
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                lname = name.lower()
                if lname.endswith((".xhtml", ".html", ".htm")):
                    out.extend(_extract_ixbrl_facts_from_text(zf.read(name).decode("utf-8", errors="ignore")))
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    return _extract_ixbrl_facts_from_text(text)


def _infer_currency_from_unit(unit: Any) -> str:
    token = str(unit or "").upper()
    for code in ("EUR", "USD", "NOK", "SEK", "DKK", "ISK", "GBP"):
        if code in token:
            return code
    return ""


def _arelle_available() -> bool:
    return importlib.util.find_spec("arelle") is not None


def parse_esef_with_arelle(file_path: Path, source_doc_id: str, validate: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not _arelle_available():
        raise RuntimeError("Arelle dependency is not installed. Install `arelle` or run with --arelle-mode off.")

    if file_path.suffix.lower() in {".json", ".jsonl"}:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        facts = payload.get("facts", payload if isinstance(payload, list) else [])
        validation = payload.get("validation", [])
        return facts, validation

    from arelle import Cntlr, FileSource, ModelDocument, ModelManager, ValidateXbrl

    cntlr = Cntlr.Cntlr(logFileName=None)
    model_manager = ModelManager.initialize(cntlr)
    model_manager.validateDisclosureSystem = True
    file_source = FileSource.openFileSource(str(file_path), cntlr)
    model_xbrl = model_manager.load(file_source, "loading")
    if model_xbrl is None:
        raise RuntimeError(f"Arelle failed to load source {file_path}")

    if getattr(model_xbrl.modelDocument, "type", None) == ModelDocument.Type.INLINEXBRLDOCUMENTSET:
        model_xbrl.createContexts = True

    facts_out: list[dict[str, Any]] = []
    for fact in model_xbrl.factsInInstance:
        context = getattr(fact, "context", None)
        unit = getattr(fact, "unit", None)
        measures = [str(m) for side in getattr(unit, "measures", ()) for m in side] if unit is not None else []
        period_end = getattr(context, "endDatetime", None) if context else None
        instant = getattr(context, "instantDatetime", None) if context else None
        dims = {}
        if context is not None:
            for dim_qn, mem in context.qnameDims.items():
                dims[str(dim_qn)] = str(getattr(mem, "memberQname", mem))
        unit_str = "*".join(measures)
        facts_out.append(
            {
                "concept": str(getattr(fact, "qname", "")),
                "value": pd.to_numeric(getattr(fact, "value", np.nan), errors="coerce"),
                "decimals": getattr(fact, "decimals", None),
                "unit": unit_str,
                "currency": _infer_currency_from_unit(unit_str),
                "context_ref": getattr(fact, "contextID", None),
                "period_end": (period_end or instant).date().isoformat() if (period_end or instant) else None,
                "entity": str(getattr(context, "entityIdentifier", (None, ""))[1]) if context else "",
                "dimensions": json.dumps(dims, ensure_ascii=False),
                "raw_label": str(getattr(fact, "qname", "")).split(":")[-1],
                "source_doc_id": source_doc_id,
            }
        )

    validation_rows: list[dict[str, Any]] = []
    if validate:
        ValidateXbrl.ValidateXbrl(model_xbrl).validate(model_xbrl)
        for entry in getattr(model_xbrl, "errors", []):
            validation_rows.append(
                {
                    "severity": "error",
                    "code": "ARELLE_VALIDATION",
                    "message": str(entry),
                    "location": str(file_path),
                    "source_doc_id": source_doc_id,
                }
            )
    model_xbrl.close()
    return facts_out, validation_rows


def parse_reports(docs_df: pd.DataFrame, arelle_mode: str = "off") -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    all_facts: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    arelle_validation_rows: list[dict[str, Any]] = []

    for row in docs_df.itertuples(index=False):
        if row.status not in {"downloaded", "cached"}:
            continue
        file_path = Path(row.file_path)
        source_type = str(row.source_type).lower()

        parsed: list[dict[str, Any]] = []
        if source_type == "esef":
            try:
                if arelle_mode in {"parse", "parse_validate"}:
                    parsed, val_rows = parse_esef_with_arelle(file_path, source_doc_id=row.doc_id, validate=arelle_mode == "parse_validate")
                    arelle_validation_rows.extend(val_rows)
                else:
                    parsed = _parse_esef_file(file_path)
            except Exception as exc:
                issues.append({"severity": "error", "issue_type": "parse_failed", "doc_id": row.doc_id, "details": str(exc)})
        elif source_type == "pdf":
            issues.append(
                {
                    "severity": "warning",
                    "issue_type": "pdf_review_required",
                    "doc_id": row.doc_id,
                    "details": "PDF fallback used; manual review required unless confidence is explicitly raised.",
                }
            )
        else:
            issues.append({"severity": "warning", "issue_type": "source_unsupported", "doc_id": row.doc_id, "details": source_type})

        for fact in parsed:
            all_facts.append(
                {
                    "source_doc_id": row.doc_id,
                    "company_id": row.company_id,
                    "ticker": row.ticker,
                    "period_end": f"{int(row.year)}-12-31",
                    "concept": fact.get("concept"),
                    "value": fact.get("value"),
                    "decimals": fact.get("decimals"),
                    "unit": fact.get("unit"),
                    "currency": fact.get("currency") or _infer_currency_from_unit(fact.get("unit")),
                    "context_ref": fact.get("context_ref"),
                    "entity": fact.get("entity") or row.company_id,
                    "dimensions": fact.get("dimensions") or "",
                    "taxonomy_ref": "",
                    "raw_label": fact.get("raw_label"),
                    "source_type": source_type,
                }
            )

    if not all_facts:
        return pd.DataFrame(columns=RAW_FACT_COLUMNS), issues, pd.DataFrame(columns=ARELLE_VALIDATION_COLUMNS)
    return pd.DataFrame(all_facts, columns=RAW_FACT_COLUMNS), issues, pd.DataFrame(arelle_validation_rows, columns=ARELLE_VALIDATION_COLUMNS)


def _load_mapping(path: Path) -> pd.DataFrame:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mappings = data.get("mappings", [])
    rows = []
    for m in mappings:
        rows.append(
            {
                "concept": m.get("concept"),
                "field_id": m.get("field_id"),
                "confidence": m.get("confidence", "medium"),
            }
        )
    return pd.DataFrame(rows)


def load_merged_mapping(default_mapping: Path, overrides_dir: Path, ticker: str | None = None) -> pd.DataFrame:
    base = _load_mapping(default_mapping)
    if ticker is None:
        return base
    override_file = overrides_dir / f"{str(ticker).lower()}.yaml"
    if not override_file.exists():
        return base
    ov = _load_mapping(override_file)
    if ov.empty:
        return base
    combined = pd.concat([base, ov], ignore_index=True)
    combined = combined.drop_duplicates(subset=["concept"], keep="last")
    return combined


def map_to_canonical(facts: pd.DataFrame, cfg: AgentConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if facts.empty:
        return pd.DataFrame(columns=LONG_COLUMNS), []

    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for ticker, part in facts.groupby("ticker", dropna=False):
        mapping = load_merged_mapping(cfg.mapping_default, cfg.mapping_overrides_dir, ticker=ticker)
        merged = part.merge(mapping, how="left", on="concept")
        missing = merged[merged["field_id"].isna()]
        for rec in missing.itertuples(index=False):
            issues.append(
                {
                    "severity": "warning",
                    "issue_type": "mapping_gap",
                    "doc_id": rec.source_doc_id,
                    "company_id": rec.company_id,
                    "field_id": None,
                    "details": rec.concept,
                }
            )
        mapped = merged[merged["field_id"].notna()].copy()
        if mapped.empty:
            continue
        grouped = (
            mapped.groupby(["company_id", "period_end", "field_id", "currency", "source_doc_id", "concept", "raw_label", "confidence"], dropna=False)["value"]
            .sum(min_count=1)
            .reset_index()
        )
        for rec in grouped.itertuples(index=False):
            rows.append(
                {
                    "company_id": rec.company_id,
                    "period_end": rec.period_end,
                    "statement": "UNKNOWN",
                    "field_id": rec.field_id,
                    "value": rec.value,
                    "unit": "",
                    "currency": rec.currency,
                    "source_doc_id": rec.source_doc_id,
                    "confidence": rec.confidence,
                    "raw_tag": rec.concept,
                    "raw_label": rec.raw_label,
                    "period_type": "annual",
                    "original_value": rec.value,
                    "original_unit": "",
                    "normalized_value": rec.value,
                    "normalized_unit": "currency" if rec.currency else "",
                }
            )

    out = pd.DataFrame(rows, columns=LONG_COLUMNS)
    if out.empty:
        return out, issues

    schema = yaml.safe_load(cfg.canonical_schema.read_text(encoding="utf-8"))
    field_meta = pd.DataFrame(schema.get("fields", []))[["field_id", "statement", "required_core"]]
    out = out.merge(field_meta, on="field_id", how="left", suffixes=("", "_schema"))
    out["statement"] = out["statement_schema"].fillna(out["statement"])
    out = out.drop(columns=["statement_schema"])
    return out[LONG_COLUMNS], issues


def resolve_conflicting_filings(long_df: pd.DataFrame, docs_df: pd.DataFrame, cfg: AgentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if long_df.empty:
        empty = pd.DataFrame(columns=["company_id", "period_end", "doc_id_winner", "doc_ids_all", "reasons", "stats"])
        return empty, long_df

    schema = yaml.safe_load(cfg.canonical_schema.read_text(encoding="utf-8")) or {}
    required_core = {f.get("field_id") for f in schema.get("fields", []) if f.get("required_core")}

    docs = docs_df.copy()
    if "filing_datetime" not in docs.columns:
        docs["filing_datetime"] = pd.NaT
    docs["filing_datetime"] = pd.to_datetime(docs["filing_datetime"], errors="coerce")
    if "year" in docs.columns:
        fallback = pd.to_datetime(docs["year"].astype("Int64").astype(str) + "-12-31", errors="coerce")
        docs["filing_datetime"] = docs["filing_datetime"].fillna(fallback)
    docs["source_priority"] = docs["source_type"].map(lambda x: SOURCE_PRIORITY.get(str(x).lower(), 0))

    base = long_df.merge(docs[["doc_id", "filing_datetime", "source_priority"]], left_on="source_doc_id", right_on="doc_id", how="left")
    base["required_core_present"] = base["field_id"].isin(required_core).astype(int)

    stats = base.groupby(["company_id", "period_end", "period_type", "source_doc_id"], dropna=False).agg(
        source_priority=("source_priority", "max"),
        filing_datetime=("filing_datetime", "max"),
        required_core_present=("required_core_present", "sum"),
    ).reset_index()

    winners = stats.sort_values(["source_priority", "filing_datetime", "required_core_present"], ascending=[False, False, False]).groupby(["company_id", "period_end", "period_type"], as_index=False).head(1)
    winners = winners.rename(columns={"source_doc_id": "doc_id_winner"})

    doc_lists = stats.groupby(["company_id", "period_end", "period_type"], dropna=False)["source_doc_id"].apply(list).rename("doc_ids_all").reset_index()
    resolution = winners.merge(doc_lists, on=["company_id", "period_end", "period_type"], how="left")
    resolution["reasons"] = "priority>filing_datetime>required_core_present"
    resolution["stats"] = resolution[["source_priority", "required_core_present"]].astype(str).agg("|".join, axis=1)

    winner_long = long_df.merge(
        resolution[["company_id", "period_end", "period_type", "doc_id_winner"]],
        on=["company_id", "period_end", "period_type"],
        how="left",
    )
    winner_long = winner_long[winner_long["source_doc_id"] == winner_long["doc_id_winner"]].drop(columns=["doc_id_winner"])

    return resolution[["company_id", "period_end", "doc_id_winner", "doc_ids_all", "reasons", "stats"]], winner_long


def build_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["company_id", "period_end"])
    pivot = (
        long_df.pivot_table(index=["company_id", "period_end"], columns="field_id", values="value", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return pivot


def validate_financials(long_df: pd.DataFrame, wide_df: pd.DataFrame, cfg: AgentConfig) -> pd.DataFrame:
    issues: list[dict[str, Any]] = []
    if wide_df.empty:
        return pd.DataFrame(columns=["severity", "issue_type", "company_id", "period_end", "field_id", "details"])

    def _row_issue(row: pd.Series, issue_type: str, details: str, severity: str = "warning", field_id: str | None = None) -> None:
        issues.append(
            {
                "severity": severity,
                "issue_type": issue_type,
                "company_id": row.get("company_id"),
                "period_end": row.get("period_end"),
                "field_id": field_id,
                "details": details,
            }
        )

    req_fields = [
        f["field_id"]
        for f in (yaml.safe_load(cfg.canonical_schema.read_text(encoding="utf-8")) or {}).get("fields", [])
        if f.get("required_core")
    ]

    for _, row in wide_df.iterrows():
        assets = row.get("total_assets")
        eq = row.get("total_equity")
        liab = row.get("total_liabilities")
        if pd.notna(assets) and pd.notna(eq) and pd.notna(liab):
            rhs = eq + liab
            if abs(assets - rhs) > cfg.tolerance_balance * max(abs(assets), 1.0):
                _row_issue(row, "balance_sheet_not_balanced", f"assets={assets}, equity+liabilities={rhs}", "error")

        cfo = row.get("cash_flow_from_operations")
        cfi = row.get("cash_flow_from_investing")
        cff = row.get("cash_flow_from_financing")
        d_cash = row.get("net_change_in_cash")
        fx = row.get("fx_effect_on_cash") if pd.notna(row.get("fx_effect_on_cash")) else 0.0
        if all(pd.notna(x) for x in [cfo, cfi, cff, d_cash]):
            expected = cfo + cfi + cff + fx
            if abs(d_cash - expected) > cfg.tolerance_cashflow * max(abs(d_cash), 1.0):
                _row_issue(row, "cashflow_bridge_mismatch", f"delta_cash={d_cash}, expected={expected}", "warning")

        for field in ["revenue_total", "total_assets", "short_term_debt", "long_term_debt"]:
            if field in row and pd.notna(row[field]) and row[field] < 0:
                _row_issue(row, "sign_sanity_failed", f"{field} is negative: {row[field]}", "warning", field)

        for field in req_fields:
            if field not in row or pd.isna(row[field]):
                _row_issue(row, "required_core_missing", f"Missing required field {field}", "warning", field)

    dup = long_df.groupby(["company_id", "period_end", "field_id"], dropna=False).size().reset_index(name="n")
    for rec in dup[dup["n"] > 1].itertuples(index=False):
        issues.append(
            {
                "severity": "warning",
                "issue_type": "duplicate_period_field",
                "company_id": rec.company_id,
                "period_end": rec.period_end,
                "field_id": rec.field_id,
                "details": f"{rec.n} entries detected; latest source_doc_id should be preferred downstream.",
            }
        )

    currency_series = long_df.get("currency", pd.Series(["" for _ in range(len(long_df))]))
    monetary_fields = set(long_df[currency_series.astype(str) != ""]["field_id"].dropna().unique())
    group_cols = ["company_id", "period_end"] + (["source_doc_id"] if "source_doc_id" in long_df.columns else [])
    for key, part in long_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        company_id = key[0] if len(key) > 0 else None
        period_end = key[1] if len(key) > 1 else None
        source_doc_id = key[2] if len(key) > 2 else "unknown_doc"
        inferred = part.get("currency", pd.Series(["" for _ in range(len(part))])).fillna("").astype(str).tolist()
        inferred += [_infer_currency_from_unit(u) for u in part.get("unit", pd.Series(["" for _ in range(len(part))])).tolist()]
        currencies = sorted({c for c in inferred if c})
        if len(currencies) > 1:
            issues.append(
                {
                    "severity": "warning",
                    "issue_type": "mixed_currencies",
                    "company_id": company_id,
                    "period_end": period_end,
                    "field_id": None,
                    "details": f"{source_doc_id}: {currencies}",
                }
            )

    for rec in long_df.itertuples(index=False):
        if rec.field_id in monetary_fields and not str(getattr(rec, "currency", "") or ""):
            issues.append(
                {
                    "severity": "warning",
                    "issue_type": "missing_currency",
                    "company_id": rec.company_id,
                    "period_end": rec.period_end,
                    "field_id": rec.field_id,
                    "details": rec.source_doc_id,
                }
            )

        if "share" in str(getattr(rec, "unit", "")).lower() and str(getattr(rec, "normalized_unit", "")) == "currency":
            issues.append(
                {
                    "severity": "warning",
                    "issue_type": "inconsistent_units",
                    "company_id": rec.company_id,
                    "period_end": rec.period_end,
                    "field_id": rec.field_id,
                    "details": rec.source_doc_id,
                }
            )

    return pd.DataFrame(issues)


def export_outputs(
    asof: str,
    processed_dir: Path,
    exports_dir: Path,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    filing_resolution_df: pd.DataFrame | None = None,
    arelle_validation_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    long_path = processed_dir / "financials_long.parquet"
    wide_path = processed_dir / "financials_wide.parquet"
    issues_path = processed_dir / "issues.parquet"
    resolution_path = processed_dir / "filing_resolution.parquet"
    arelle_validation_path = processed_dir / "arelle_validation.parquet"

    filing_resolution_df = filing_resolution_df if filing_resolution_df is not None else pd.DataFrame(columns=["company_id", "period_end", "doc_id_winner", "doc_ids_all", "reasons", "stats"])
    arelle_validation_df = arelle_validation_df if arelle_validation_df is not None else pd.DataFrame(columns=ARELLE_VALIDATION_COLUMNS)

    long_df.to_parquet(long_path, index=False)
    wide_df.to_parquet(wide_path, index=False)
    issues_df.to_parquet(issues_path, index=False)
    filing_resolution_df.to_parquet(resolution_path, index=False)
    arelle_validation_df.to_parquet(arelle_validation_path, index=False)

    xlsx_path = exports_dir / "financials.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="financials_long", index=False)
        wide_df.to_excel(writer, sheet_name="financials_wide", index=False)
        issues_df.to_excel(writer, sheet_name="issues", index=False)

    return {
        "financials_long": str(long_path),
        "financials_wide": str(wide_path),
        "issues": str(issues_path),
        "filing_resolution": str(resolution_path),
        "arelle_validation": str(arelle_validation_path),
        "xlsx": str(xlsx_path),
    }


def write_manifest(
    asof: str,
    run_root: Path,
    docs_df: pd.DataFrame,
    facts_df: pd.DataFrame,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    outputs: dict[str, str],
    started_ts: float,
) -> Path:
    manifest = {
        "asof": asof,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "duration_sec": round(time.time() - started_ts, 3),
        "counts": {
            "documents": int(len(docs_df)),
            "raw_facts": int(len(facts_df)),
            "financials_long": int(len(long_df)),
            "financials_wide": int(len(wide_df)),
            "issues": int(len(issues_df)),
        },
        "inputs": docs_df.to_dict(orient="records"),
        "artifacts": outputs,
        "versions": {
            "schema": "v1",
            "mapping": "esef_ifrs_default.yaml",
            "python": f"{pd.__version__}",
        },
    }
    path = run_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def run_pipeline(args: argparse.Namespace) -> None:
    started = time.time()
    asof = args.asof or date.today().isoformat()
    cfg = load_agent_config(Path(args.config))

    run_root = Path("data")
    raw_dir = run_root / "raw" / asof
    processed_dir = run_root / "processed" / asof
    exports_dir = Path("exports") / asof

    selected_steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    sources = read_report_sources(cfg.report_sources_csv, max_companies=args.max_companies)
    if args.source != "all":
        sources = sources[sources["source_type"].str.lower() == args.source.lower()].copy()

    docs_df = pd.DataFrame()
    facts_df = pd.DataFrame(columns=RAW_FACT_COLUMNS)
    long_df = pd.DataFrame(columns=LONG_COLUMNS)
    wide_df = pd.DataFrame()
    filing_resolution_df = pd.DataFrame(columns=["company_id", "period_end", "doc_id_winner", "doc_ids_all", "reasons", "stats"])
    arelle_validation_df = pd.DataFrame(columns=ARELLE_VALIDATION_COLUMNS)
    issues_df = pd.DataFrame(columns=["severity", "issue_type", "company_id", "period_end", "field_id", "details"])

    issue_rows: list[dict[str, Any]] = []

    if "download" in selected_steps:
        docs_df, i = download_reports(sources, raw_dir, args.dry_run, args.force, cfg.timeout_sec, cfg.user_agent)
        issue_rows.extend(i)
    else:
        index_path = raw_dir / "download_index.parquet"
        docs_df = pd.read_parquet(index_path) if index_path.exists() else pd.DataFrame()

    if "parse" in selected_steps and not docs_df.empty:
        facts_df, i, arelle_validation_df = parse_reports(docs_df, arelle_mode=args.arelle_mode)
        issue_rows.extend(i)
        raw_facts_path = processed_dir / "raw_facts.parquet"
        raw_facts_path.parent.mkdir(parents=True, exist_ok=True)
        facts_df.to_parquet(raw_facts_path, index=False)

    if "map" in selected_steps:
        long_df, i = map_to_canonical(facts_df, cfg)
        issue_rows.extend(i)
        filing_resolution_df, winner_long_df = resolve_conflicting_filings(long_df, docs_df, cfg)
        wide_df = build_wide(winner_long_df)

    if "validate" in selected_steps:
        val_issues = validate_financials(long_df, wide_df, cfg)
        if not val_issues.empty:
            issue_rows.extend(val_issues.to_dict(orient="records"))

    issues_df = pd.DataFrame(issue_rows)

    outputs: dict[str, str] = {}
    if "export" in selected_steps:
        outputs = export_outputs(asof, processed_dir, exports_dir, long_df, wide_df, issues_df, filing_resolution_df=filing_resolution_df, arelle_validation_df=arelle_validation_df)

    manifest_path = write_manifest(asof, processed_dir, docs_df, facts_df, long_df, wide_df, issues_df, outputs, started)
    LOGGER.info("Run complete. Manifest: %s", manifest_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic financials extraction agent (ESEF-first).")
    parser.add_argument("--asof", required=True, help="As-of date (YYYY-MM-DD)")
    parser.add_argument("--config", default="config/agent.yaml", help="Path to agent configuration YAML")
    parser.add_argument(
        "--steps",
        default="download,parse,map,validate,export",
        help="Comma-separated steps: download,parse,map,validate,export",
    )
    parser.add_argument("--source", default="all", choices=["esef", "pdf", "all"], help="Source filter")
    parser.add_argument("--max-companies", type=int, default=None, help="Limit number of companies for development runs")
    parser.add_argument("--dry-run", action="store_true", help="Do not download files")
    parser.add_argument("--force", action="store_true", help="Force re-download and overwrite existing raw files")
    parser.add_argument("--arelle-mode", default="off", choices=["off", "parse", "parse_validate"], help="Arelle strict mode: off, parse, parse_validate")
    return parser
