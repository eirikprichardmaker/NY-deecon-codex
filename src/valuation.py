from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd

MARKET_BY_COUNTRY = {
    "NO": "NO",
    "NORWAY": "NO",
    "SE": "SE",
    "SWEDEN": "SE",
    "DK": "DK",
    "DENMARK": "DK",
    "FI": "FI",
    "FINLAND": "FI",
}

MARKET_BY_SUFFIX = {
    "OL": "NO",
    "ST": "SE",
    "CO": "DK",
    "HE": "FI",
}


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """
    Supports nested dotted keys in dict-like configs.
    Example: _cfg_get(cfg, "valuation.dcf.years", 5)
    """
    if not isinstance(cfg, dict):
        return default
    cur = cfg
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _norm_yahoo(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _is_price_like_col(col: str) -> bool:
    c = str(col).strip().lower()
    patterns = [
        r"(?:^|_)adj_close(?:$|_)",
        r"(?:^|_)close(?:$|_)",
        r"(?:^|_)price(?:$|_)",
        r"(?:^|_)ma200(?:$|_)",
        r"(?:^|_)mad(?:$|_)",
        r"(?:^|_)index_(?:$|_)",
    ]
    return any(re.search(p, c) for p in patterns)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


@dataclass
class DCFParams:
    years: int = 5
    growth: float = 0.02          # annual growth on FCF
    terminal_growth: float = 0.02 # capped <= 2%
    fcf_mult: float = 1.00        # margin proxy in sensitivity
    wacc_default: float = 0.09
    wacc_min: float = 0.03
    wacc_max: float = 0.25


def _dcf_value(
    fcf0_m: float,
    wacc: float,
    years: int,
    growth: float,
    terminal_growth: float,
    fcf_mult: float,
) -> float:
    """
    Simple DCF on FCF in *millions*.
    Returns EV in millions.
    """
    if not np.isfinite(fcf0_m):
        return np.nan
    if not np.isfinite(wacc):
        return np.nan
    if wacc <= 0 or wacc <= terminal_growth:
        return np.nan

    g = float(growth)
    tg = float(min(terminal_growth, 0.02))
    n = int(years)
    f0 = float(fcf0_m) * float(fcf_mult)

    # Forecast and discount
    pv_sum = 0.0
    for t in range(1, n + 1):
        ft = f0 * ((1.0 + g) ** t)
        pv_sum += ft / ((1.0 + wacc) ** t)

    f_n = f0 * ((1.0 + g) ** n)
    tv = (f_n * (1.0 + tg)) / (wacc - tg)
    pv_tv = tv / ((1.0 + wacc) ** n)

    return pv_sum + pv_tv


def _resolve_paths(ctx) -> Dict[str, Path]:
    """
    Try project resolve_paths; otherwise fallback to local defaults.
    """
    try:
        from src.common.config import resolve_paths  # type: ignore
        paths = resolve_paths(ctx.cfg, ctx.project_root)
        # normalize to Path
        for k, v in list(paths.items()):
            if not isinstance(v, Path):
                paths[k] = Path(v)
        return paths
    except Exception:
        root = Path(getattr(ctx, "project_root", "."))  # best effort
        return {
            "data_dir": root / "data",
            "raw_dir": root / "data" / "raw",
            "processed_dir": root / "data" / "processed",
            "runs_dir": root / "runs",
        }


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def canon(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")

    out.columns = [canon(c) for c in out.columns]
    return out


def _quarter_end_from_year_period(year_raw: Any, period_raw: Any) -> Optional[str]:
    try:
        y = int(float(year_raw))
        p = int(float(period_raw))
    except Exception:
        return None
    if y < 1900 or y > 2100:
        return None
    mmdd = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31", 5: "12-31"}.get(p)
    if not mmdd:
        return None
    return f"{y:04d}-{mmdd}"


def _suffix_from_yahoo(yahoo_ticker: str) -> str:
    s = str(yahoo_ticker).strip().upper()
    if "." not in s:
        return ""
    return s.rsplit(".", 1)[-1]


def _latest_raw_snapshot_dir_with_data(raw_dir: Path, asof: str, subdir: str) -> Optional[Path]:
    if not raw_dir.exists():
        return None
    asof_dt = pd.to_datetime(asof, errors="coerce")
    if pd.isna(asof_dt):
        return None

    rows: list[tuple[pd.Timestamp, Path]] = []
    for d in raw_dir.iterdir():
        if not d.is_dir():
            continue
        ts = pd.to_datetime(d.name, format="%Y-%m-%d", errors="coerce")
        if pd.isna(ts) or ts.normalize() > asof_dt.normalize():
            continue
        if (d / subdir).exists():
            rows.append((ts.normalize(), d))
    if not rows:
        return None
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[0][1]


def _load_insid_mapping(project_root: Path, allowed_yahoo: Optional[pd.Series] = None) -> pd.DataFrame:
    candidates = [
        project_root / "config" / "tickers_with_insid_clean.csv",
        project_root / "config" / "tickers_with_insid.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        cols = {str(c).strip().lower(): c for c in df.columns}
        y_col = cols.get("yahoo_ticker")
        i_col = cols.get("ins_id") or cols.get("insid")
        c_col = cols.get("country")
        if not y_col or not i_col:
            continue

        out = df[[y_col, i_col] + ([c_col] if c_col else [])].copy().rename(
            columns={y_col: "yahoo_ticker", i_col: "ins_id", (c_col or ""): "country"}
        )
        if "country" not in out.columns:
            out["country"] = ""
        out["yahoo_ticker"] = out["yahoo_ticker"].astype(str).str.upper().str.strip()
        out["ins_id"] = pd.to_numeric(out["ins_id"], errors="coerce")
        out = out[out["ins_id"].notna() & out["yahoo_ticker"].ne("")].copy()
        out["ins_id"] = out["ins_id"].astype(int)
        out["country"] = out["country"].astype(str).str.upper().str.strip()
        out["market"] = out["country"].map(lambda x: MARKET_BY_COUNTRY.get(x, ""))
        needs_market = out["market"].eq("")
        if needs_market.any():
            out.loc[needs_market, "market"] = out.loc[needs_market, "yahoo_ticker"].map(
                lambda x: MARKET_BY_SUFFIX.get(_suffix_from_yahoo(x), "")
            )
        out = out[out["market"].ne("")].copy()
        out = out[["yahoo_ticker", "ins_id", "market"]]
        if allowed_yahoo is not None:
            allow = allowed_yahoo.astype(str).str.upper().str.strip()
            allow_set = {x for x in allow.tolist() if x}
            if allow_set:
                out = out[out["yahoo_ticker"].isin(allow_set)].copy()
        if not out.empty:
            stats = out.groupby("yahoo_ticker", dropna=False).agg(
                ins_id_nunique=("ins_id", "nunique"),
                market_nunique=("market", "nunique"),
            )
            bad = stats[(stats["ins_id_nunique"] > 1) | (stats["market_nunique"] > 1)]
            if not bad.empty:
                # Keep valuation deterministic: ambiguous mapping is excluded from
                # quarterly bridge (falls back to missing quarterly data handling).
                bad_keys = set(bad.index.tolist())
                out = out[~out["yahoo_ticker"].isin(bad_keys)].copy()
            out = out.sort_values(["yahoo_ticker", "ins_id", "market"], kind="mergesort").drop_duplicates(
                subset=["yahoo_ticker"],
                keep="first",
            )
        return out
    return pd.DataFrame(columns=["yahoo_ticker", "ins_id", "market"])


def _pick_latest_report_row(df: pd.DataFrame, asof_dt: pd.Timestamp) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    d = _canon_cols(df)

    date_col = "report_date" if "report_date" in d.columns else ("report_end_date" if "report_end_date" in d.columns else None)
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d[d[date_col].notna() & (d[date_col] <= asof_dt)]
    if d.empty:
        return None

    if "year" in d.columns:
        d["year"] = pd.to_numeric(d["year"], errors="coerce")
    if "period" in d.columns:
        d["period"] = pd.to_numeric(d["period"], errors="coerce")

    sort_cols = [c for c in ["year", "period"] if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols)
    return d.iloc[-1]


def _load_quarterly_r12_fcf_snapshot(
    *,
    project_root: Path,
    raw_dir: Path,
    asof: str,
    yahoo_tickers: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame({"yahoo_ticker": yahoo_tickers.astype(str).str.upper().str.strip()})
    out["quarterly_fcf_millions"] = np.nan
    out["quarterly_period_end"] = ""
    out["quarterly_snapshot"] = ""
    out["quarterly_source"] = ""

    if out.empty:
        return out

    snapshot = _latest_raw_snapshot_dir_with_data(raw_dir, asof, "reports_r12")
    if snapshot is None:
        return out

    ins_map = _load_insid_mapping(project_root, allowed_yahoo=out["yahoo_ticker"])
    if ins_map.empty:
        return out

    work = out.merge(ins_map, on="yahoo_ticker", how="left", validate="one_to_one")
    asof_dt = pd.to_datetime(asof, errors="coerce")
    if pd.isna(asof_dt):
        return out

    for i, row in work.iterrows():
        ins_id = row.get("ins_id")
        market = str(row.get("market", "")).upper()
        if not np.isfinite(pd.to_numeric(pd.Series([ins_id]), errors="coerce").iloc[0]) or not market:
            continue
        p = snapshot / "reports_r12" / f"market={market}" / f"ins_id={int(ins_id)}.parquet"
        if not p.exists():
            continue
        try:
            rep = pd.read_parquet(p)
        except Exception:
            continue
        latest = _pick_latest_report_row(rep, asof_dt=asof_dt)
        if latest is None:
            continue
        latest_d = _canon_cols(pd.DataFrame([latest])).iloc[0]
        fcf = pd.to_numeric(
            pd.Series([latest_d.get("free_cash_flow", latest_d.get("free_cash_flow_for_the_year"))]),
            errors="coerce",
        ).iloc[0]
        if not np.isfinite(fcf):
            continue
        per_end = latest_d.get("report_end_date", latest_d.get("report_date", ""))
        per_end_dt = pd.to_datetime(per_end, errors="coerce")
        per_end_s = ""
        if pd.notna(per_end_dt):
            per_end_s = per_end_dt.date().isoformat()
        else:
            per_end_s = _quarter_end_from_year_period(latest_d.get("year"), latest_d.get("period")) or ""

        out.at[i, "quarterly_fcf_millions"] = float(fcf)
        out.at[i, "quarterly_period_end"] = per_end_s
        out.at[i, "quarterly_snapshot"] = snapshot.name
        out.at[i, "quarterly_source"] = "reports_r12.free_cash_flow"

    out["quarterly_period_end"] = out["quarterly_period_end"].astype(str)
    period_dt = pd.to_datetime(out["quarterly_period_end"], errors="coerce")
    out["quarterly_age_days"] = (asof_dt.normalize() - period_dt).dt.days
    out["quarterly_data_ok"] = pd.to_numeric(out["quarterly_fcf_millions"], errors="coerce").notna()
    return out


def _merge_fundamentals_summary(
    df: "pd.DataFrame",
    raw_dir: Path,
    asof: str,
    use_normalized_fcf: bool,
    log,
) -> None:
    """
    Merger inn normalisert FCF og EBITDA fra fundamentals_summary.parquet (hvis finnes).
    Endrer df in-place ved å legge til kolonner:
      - fcf_m_median_3y, fcf_m_median_5y, fcf_m_latest
      - ebitda_m_latest, ebit_m_latest, roic_latest
    """
    # Finn nyeste summary <= asof
    summary_path: Optional[Path] = None
    raw_base = raw_dir.parent if raw_dir.name == asof else raw_dir
    for candidate_dir in sorted(raw_base.iterdir(), reverse=True):
        if not candidate_dir.is_dir():
            continue
        if candidate_dir.name > asof:
            continue
        p = candidate_dir / "fundamentals_summary.parquet"
        if p.exists():
            summary_path = p
            break
    # Also check raw_dir directly
    if summary_path is None and (raw_dir / "fundamentals_summary.parquet").exists():
        summary_path = raw_dir / "fundamentals_summary.parquet"

    if summary_path is None:
        if use_normalized_fcf:
            log.warning("valuation: use_normalized_fcf=True men fundamentals_summary.parquet ikke funnet. "
                        "Kjør ingest_fundamentals_from_freeze for å generere den.")
        return

    try:
        summary = pd.read_parquet(summary_path)
    except Exception as e:
        log.warning(f"valuation: kunne ikke lese {summary_path}: {e}")
        return

    id_col = "yahoo_ticker" if "yahoo_ticker" in summary.columns else "ticker"
    if id_col not in summary.columns:
        return

    summary[id_col] = summary[id_col].astype(str).str.upper().str.strip()

    merge_cols = [c for c in ["fcf_m_latest", "fcf_m_median_3y", "fcf_m_median_5y",
                               "ebitda_m_latest", "ebit_m_latest", "roic_latest",
                               "fcf_m_years_available", "fcf_m_cv_3y",
                               "netdebt_m_latest", "netdebt_ebitda_latest"]
                  if c in summary.columns]
    if not merge_cols:
        return

    # Don't overwrite existing columns
    new_cols = [c for c in merge_cols if c not in df.columns]
    if not new_cols:
        return

    sub = summary[[id_col] + new_cols].drop_duplicates(subset=[id_col])
    before = len(df)
    df_key = "yahoo_ticker" if "yahoo_ticker" in df.columns else id_col
    merged = df.merge(sub.rename(columns={id_col: df_key}), on=df_key, how="left", validate="many_to_one")
    if len(merged) != before:
        log.warning(f"valuation: merge med fundamentals_summary endret rowcount {before} → {len(merged)}")
        return
    for c in new_cols:
        df[c] = merged[c].values
    log.info(f"valuation: merged fundamentals_summary fra {summary_path.parent.name} "
             f"({', '.join(new_cols)})")


def run(ctx, log) -> int:
    """
    Builds:
      - runs/<run_id>/valuation.csv
      - runs/<run_id>/valuation_sensitivity.csv
      - data/processed/master_valued.parquet

    Key guarantee:
      - Uses yahoo_ticker as unique instrument id for valuation output and merges.
      - Enforces one-to-one merge, stable rowcount.
    """
    paths = _resolve_paths(ctx)
    project_root = Path(getattr(ctx, "project_root", "."))
    raw_dir = paths.get("raw_dir", project_root / "data" / "raw")
    processed_dir = paths.get("processed_dir", Path("data/processed"))
    runs_dir = paths.get("runs_dir", Path("runs"))

    # Run dir (prefer ctx.run_dir if present)
    run_dir = Path(getattr(ctx, "run_dir", "")) if getattr(ctx, "run_dir", None) else (runs_dir / getattr(ctx, "run_id", "run"))
    _ensure_dir(run_dir)
    _ensure_dir(processed_dir)

    asof = getattr(ctx, "asof", None)
    asof_s = str(asof) if asof is not None else datetime.utcnow().date().isoformat()

    # Load base (master_cost is the safest pre-valuation frame)
    master_cost_path = processed_dir / "master_cost.parquet"
    if not master_cost_path.exists():
        raise FileNotFoundError(f"valuation: missing {master_cost_path} (run cost_of_capital first)")

    df = pd.read_parquet(master_cost_path)

    # Required key
    if "yahoo_ticker" not in df.columns:
        raise ValueError("valuation: master_cost is missing yahoo_ticker (cannot merge safely).")

    df = df.copy()
    df["yahoo_ticker"] = _norm_yahoo(df["yahoo_ticker"])

    # DoD-guard: base must be unique on yahoo_ticker
    base_dup = int(df["yahoo_ticker"].duplicated().sum())
    if base_dup:
        raise ValueError(f"valuation: master_cost has duplicate yahoo_ticker keys: {base_dup}")

    # Merge normalized FCF and EBITDA from fundamentals_summary.parquet (if available)
    use_normalized_fcf = bool(_cfg_get(ctx.cfg, "valuation.use_normalized_fcf", False))
    _merge_fundamentals_summary(df, raw_dir, asof_s, use_normalized_fcf, log)

    # Pick columns — normalized FCF (3y median) takes priority when enabled
    fcf_candidates = (
        ["fcf_m_median_3y", "fcf_m_median_5y", "fcf_millions", "fcf_-_millions", "fcf", "free_cash_flow_millions", "free_cash_flow"]
        if use_normalized_fcf
        else ["fcf_millions", "fcf_-_millions", "fcf", "free_cash_flow_millions", "free_cash_flow"]
    )
    fcf_col = _pick_first_col(df, fcf_candidates)
    ocf_col = _pick_first_col(df, ["ocf_millions", "ocf_-_millions", "ocf"])
    capex_col = _pick_first_col(df, ["capex_millions", "capex_-_millions", "capex"])
    net_debt_col = _pick_first_col(df, ["net_debt_millions", "net_debt_-_current", "net_debt_current", "net_debt", "netdebt_m_latest"])
    wacc_col = _pick_first_col(df, ["wacc_used", "wacc"])
    coe_col = _pick_first_col(df, ["coe_used", "coe"])
    ticker_col = _pick_first_col(df, ["ticker"])
    company_col = _pick_first_col(df, ["company"])

    valuation_inputs = [x for x in [fcf_col, ocf_col, capex_col, net_debt_col, wacc_col, coe_col] if x]
    forbidden_hits = [c for c in valuation_inputs if _is_price_like_col(c)]
    valuation_input_audit = {
        "asof": asof_s,
        "valuation_model": "DCF",
        "fundamental_inputs_selected": valuation_inputs,
        "forbidden_price_like_hits": forbidden_hits,
        "fundamental_only_guard_passed": len(forbidden_hits) == 0,
        "note": "Intrinsic value must be based on fundamentals + risk inputs, never price/technical fields.",
    }
    if forbidden_hits:
        raise ValueError(
            "valuation: fundamental-only guard failed. Price/technical-like columns selected as valuation inputs: "
            + ", ".join(forbidden_hits)
        )

    # Build base FCF (millions) from master_* fields.
    reason = pd.Series(index=df.index, dtype="object")

    if fcf_col:
        base_fcf_m = _to_num(df[fcf_col])
    else:
        base_fcf_m = pd.Series(np.nan, index=df.index)

    if base_fcf_m.notna().mean() == 0.0 and ocf_col and capex_col:
        ocf_m = _to_num(df[ocf_col])
        capex_m = _to_num(df[capex_col])
        base_fcf_m = ocf_m - capex_m

    prefer_quarterly = bool(_cfg_get(ctx.cfg, "valuation.prefer_quarterly_reports", False))
    require_quarterly = bool(_cfg_get(ctx.cfg, "valuation.require_quarterly_reports", False))
    max_quarterly_age_days = int(_cfg_get(ctx.cfg, "valuation.max_quarterly_report_age_days", 220))

    q_snap = pd.DataFrame(
        {
            "yahoo_ticker": df["yahoo_ticker"],
            "quarterly_fcf_millions": np.nan,
            "quarterly_period_end": "",
            "quarterly_snapshot": "",
            "quarterly_source": "",
            "quarterly_age_days": np.nan,
            "quarterly_data_ok": False,
        }
    )
    if prefer_quarterly or require_quarterly:
        q_snap = _load_quarterly_r12_fcf_snapshot(
            project_root=project_root,
            raw_dir=raw_dir,
            asof=asof_s,
            yahoo_tickers=df["yahoo_ticker"],
        )
        q_snap["yahoo_ticker"] = _norm_yahoo(q_snap["yahoo_ticker"])
        q_dup = int(q_snap["yahoo_ticker"].duplicated().sum())
        if q_dup:
            raise ValueError(f"valuation: quarterly snapshot duplicate yahoo_ticker rows: {q_dup}")
        df = df.merge(q_snap, on="yahoo_ticker", how="left", validate="one_to_one")
    else:
        for c in q_snap.columns:
            if c != "yahoo_ticker":
                df[c] = q_snap[c].values

    q_fcf_m = _to_num(df.get("quarterly_fcf_millions", pd.Series(np.nan, index=df.index)))
    q_age_days = _to_num(df.get("quarterly_age_days", pd.Series(np.nan, index=df.index)))
    q_has_data = q_fcf_m.notna()
    q_stale = q_has_data & q_age_days.notna() & (q_age_days > max_quarterly_age_days)
    q_ok = q_has_data & ~q_stale

    if use_normalized_fcf:
        # Normalized mode: use 3y median FCF — do NOT override with quarterly R12 (peak-earnings risk)
        fcf_m = base_fcf_m.copy()
        fcf_source = pd.Series("normalized_3y", index=df.index, dtype="object")
        # Fall back to quarterly R12 only where normalized FCF is missing
        q_fallback = q_ok & fcf_m.isna()
        fcf_m = fcf_m.where(~q_fallback, q_fcf_m)
        fcf_source = fcf_source.where(~q_fallback, "reports_r12_fallback")
    elif prefer_quarterly:
        fcf_m = base_fcf_m.where(~q_ok, q_fcf_m)
        fcf_source = pd.Series(np.where(q_ok, "reports_r12", "master"), index=df.index, dtype="object")
    else:
        fcf_m = base_fcf_m.copy()
        fcf_source = pd.Series("master", index=df.index, dtype="object")

    reason = np.where(fcf_m.notna(), None, "missing_fcf")
    reason = pd.Series(reason, index=df.index, dtype="object")
    if require_quarterly:
        reason = reason.where(
            q_ok,
            other=np.where(q_stale, "stale_quarterly_fcf", "missing_quarterly_fcf"),
        )
    valuation_input_audit["quarterly_reports"] = {
        "prefer_quarterly_reports": prefer_quarterly,
        "require_quarterly_reports": require_quarterly,
        "max_quarterly_report_age_days": max_quarterly_age_days,
        "quarterly_fcf_coverage": float(q_has_data.mean()) if len(q_has_data) else 0.0,
        "quarterly_fcf_fresh_coverage": float(q_ok.mean()) if len(q_ok) else 0.0,
        "quarterly_fcf_stale_count": int(q_stale.sum()),
        "fcf_source_reports_r12_count": int((fcf_source == "reports_r12").sum()),
    }
    # FCF/EBITDA sanity check — flag if FCF > 3× EBITDA (likely currency mismatch or stale data)
    ebitda_col = _pick_first_col(df, ["ebitda_m_latest", "ebitda_m", "ebitda_millions", "ebitda"])
    if ebitda_col:
        ebitda_m = _to_num(df[ebitda_col])
        fcf_ebitda_ratio = fcf_m / ebitda_m.replace(0, np.nan)
        implausible_fcf = fcf_ebitda_ratio.notna() & (fcf_ebitda_ratio.abs() > 3.0)
        df["fcf_ebitda_ratio"] = fcf_ebitda_ratio
        df["fcf_implausible"] = implausible_fcf
        n_implausible = int(implausible_fcf.sum())
        if n_implausible > 0:
            log.warning(
                f"valuation: {n_implausible} ticker(e) har FCF > 3× EBITDA (sannsynlig data-/valutafeil). "
                f"Sjekk valuation_input_audit.json for detaljer."
            )
        valuation_input_audit["fcf_implausible_count"] = n_implausible
        if n_implausible > 0:
            ticker_col_tmp = _pick_first_col(df, ["ticker", "yahoo_ticker"])
            if ticker_col_tmp:
                valuation_input_audit["fcf_implausible_tickers"] = df.loc[
                    implausible_fcf, ticker_col_tmp
                ].tolist()
    else:
        df["fcf_ebitda_ratio"] = np.nan
        df["fcf_implausible"] = False

    _atomic_write_text(
        run_dir / "valuation_input_audit.json",
        json.dumps(valuation_input_audit, ensure_ascii=False, indent=2),
    )

    # WACC/COE
    params = DCFParams(
        years=int(_cfg_get(ctx.cfg, "valuation.dcf.years", 5)),
        growth=float(_cfg_get(ctx.cfg, "valuation.dcf.growth", 0.02)),
        terminal_growth=float(min(_cfg_get(ctx.cfg, "valuation.dcf.terminal_growth", 0.02), 0.02)),
        fcf_mult=float(_cfg_get(ctx.cfg, "valuation.dcf.fcf_mult", 1.0)),
        wacc_default=float(_cfg_get(ctx.cfg, "valuation.wacc_default", 0.09)),
        wacc_min=float(_cfg_get(ctx.cfg, "valuation.wacc_min", 0.03)),
        wacc_max=float(_cfg_get(ctx.cfg, "valuation.wacc_max", 0.25)),
    )

    if wacc_col:
        wacc = _to_num(df[wacc_col]).fillna(params.wacc_default)
    else:
        wacc = pd.Series(params.wacc_default, index=df.index)

    if coe_col:
        coe = _to_num(df[coe_col]).fillna(params.wacc_default)
    else:
        coe = pd.Series(params.wacc_default, index=df.index)

    # Validate WACC bounds
    bad_wacc = (wacc <= params.wacc_min) | (wacc >= params.wacc_max) | (~np.isfinite(wacc))
    if bad_wacc.any():
        reason = reason.where(~bad_wacc, other="invalid_wacc")

    # Net debt (millions)
    if net_debt_col:
        net_debt_m = _to_num(df[net_debt_col]).fillna(0.0)
    else:
        net_debt_m = pd.Series(0.0, index=df.index)

    # DCF EV (millions)
    intrinsic_ev_m = []
    model = []
    for i in range(len(df)):
        if reason.iat[i] is not None:
            intrinsic_ev_m.append(np.nan)
            model.append("DCF")
            continue
        ev_m = _dcf_value(
            fcf0_m=float(fcf_m.iat[i]),
            wacc=float(wacc.iat[i]),
            years=params.years,
            growth=params.growth,
            terminal_growth=params.terminal_growth,
            fcf_mult=params.fcf_mult,
        )
        if not np.isfinite(ev_m):
            intrinsic_ev_m.append(np.nan)
            reason.iat[i] = "dcf_failed"
        else:
            intrinsic_ev_m.append(ev_m)
        model.append("DCF")

    intrinsic_ev_m = pd.Series(intrinsic_ev_m, index=df.index, dtype="float64")
    intrinsic_equity_m = intrinsic_ev_m - net_debt_m

    # Convert to currency units (not millions) for downstream decision logic
    intrinsic_ev = intrinsic_ev_m * 1e6
    intrinsic_equity = intrinsic_equity_m * 1e6
    net_debt_used = net_debt_m * 1e6

    # Build valuation_df (A + C)
    valuation_df = pd.DataFrame({
        "yahoo_ticker": df["yahoo_ticker"],
        "ticker": df[ticker_col] if ticker_col else df["yahoo_ticker"],
        "model": model,
        "fcf_used_millions": fcf_m,
        "fcf_source": fcf_source,
        "quarterly_fcf_millions": q_fcf_m,
        "quarterly_period_end": df.get("quarterly_period_end", pd.Series("", index=df.index)),
        "quarterly_age_days": q_age_days,
        "quarterly_data_ok": q_ok.astype(bool),
        "intrinsic_ev": intrinsic_ev,
        "intrinsic_equity": intrinsic_equity,
        "net_debt_used": net_debt_used,
        "wacc_used": wacc,
        "coe_used": coe,
        "reason": reason.fillna(""),
        "fcf_implausible": df["fcf_implausible"].values if "fcf_implausible" in df.columns else False,
        "fcf_ebitda_ratio": df["fcf_ebitda_ratio"].values if "fcf_ebitda_ratio" in df.columns else np.nan,
    })

    # B1) valuation_df must be unique on yahoo_ticker
    dup = int(valuation_df["yahoo_ticker"].duplicated().sum())
    if dup:
        raise ValueError(f"valuation: duplicate yahoo_ticker in valuation_df: {dup}")

    # Write valuation.csv (C)
    out_csv = run_dir / "valuation.csv"
    valuation_df.sort_values(["yahoo_ticker"]).to_csv(out_csv, index=False)
    log.info(f"valuation: wrote {out_csv}")

    # Sensitivity grid: growth +/-1pp, wacc +/-1pp, fcf_mult +/-2%
    g_shocks = [-0.01, 0.0, 0.01]
    w_shocks = [-0.01, 0.0, 0.01]
    f_shocks = [0.98, 1.00, 1.02]

    rows = []
    base_g = params.growth
    base_tg = params.terminal_growth
    base_n = params.years

    for dg in g_shocks:
        for dw in w_shocks:
            for fm in f_shocks:
                scen = f"g{dg:+.2%}_w{dw:+.2%}_f{(fm-1.0):+.2%}"
                # vectorized compute via apply (fast enough at ~1741*27)
                ev_m_list = []
                eq_m_list = []
                rsn_list = []
                wacc_s = (wacc + dw).astype(float)
                g_s = base_g + dg
                tg_s = min(base_tg, 0.02)

                for i in range(len(df)):
                    # carry base reason for missing_fcf/invalid_wacc etc
                    if reason.iat[i] in ("missing_fcf", "missing_quarterly_fcf", "stale_quarterly_fcf", "invalid_wacc"):
                        ev_m_list.append(np.nan)
                        eq_m_list.append(np.nan)
                        rsn_list.append(reason.iat[i])
                        continue
                    evm = _dcf_value(
                        fcf0_m=float(fcf_m.iat[i]),
                        wacc=float(wacc_s.iat[i]),
                        years=base_n,
                        growth=float(g_s),
                        terminal_growth=float(tg_s),
                        fcf_mult=float(fm),
                    )
                    if not np.isfinite(evm):
                        ev_m_list.append(np.nan)
                        eq_m_list.append(np.nan)
                        rsn_list.append("dcf_failed")
                    else:
                        ev_m_list.append(evm)
                        eq_m_list.append(evm - float(net_debt_m.iat[i]))
                        rsn_list.append("")
                rows.append(pd.DataFrame({
                    "scenario": scen,
                    "yahoo_ticker": df["yahoo_ticker"].values,
                    "ticker": (df[ticker_col].values if ticker_col else df["yahoo_ticker"].values),
                    "model": "DCF",
                    "growth_used": g_s,
                    "wacc_used": wacc_s.values,
                    "fcf_mult": fm,
                    "intrinsic_ev": (np.array(ev_m_list) * 1e6),
                    "intrinsic_equity": (np.array(eq_m_list) * 1e6),
                    "reason": rsn_list,
                }))

    sens_df = pd.concat(rows, ignore_index=True)
    sens_out = run_dir / "valuation_sensitivity.csv"
    sens_df.to_csv(sens_out, index=False)
    log.info(f"valuation: wrote {sens_out}")

    # B2) merge back 1:1 on yahoo_ticker (not ticker!)
    master_cost = df
    before = len(master_cost)

    master_valued = master_cost.merge(
        valuation_df.drop(columns=["ticker"]),
        on="yahoo_ticker",
        how="left",
        validate="one_to_one",
    )

    # B3) DoD-guard: rowcount must not change
    if len(master_valued) != before:
        raise ValueError(f"valuation: rowcount changed after merge ({before} -> {len(master_valued)}). Check join key!")

    # Final DoD: no dup yahoo_ticker
    final_dup = int(master_valued["yahoo_ticker"].duplicated().sum())
    if final_dup:
        raise ValueError(f"valuation: master_valued has duplicate yahoo_ticker after merge: {final_dup}")

    out_parq = processed_dir / "master_valued.parquet"
    master_valued.to_parquet(out_parq, index=False)
    log.info(f"valuation: wrote {out_parq}")

    return 0
