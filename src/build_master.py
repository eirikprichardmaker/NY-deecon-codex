from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.common.config import resolve_paths
from src.common.io import read_parquet


MARKET_BY_SUFFIX = {
    "OL": "NO",
    "ST": "SE",
    "CO": "DK",
    "HE": "FI",
}

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

DIVIDEND_FEATURE_COLS = [
    "dividend_history_years",
    "dividend_paid_years",
    "dividend_growth_5y",
    "dividend_cagr_5y",
    "profit_growth_5y",
    "payout_ratio",
    "share_count_growth_5y",
    "fcf_margin",
    "ocf_margin",
    "profit_margin",
    "roa",
    "dividend_yield",
]


def _quality_write(run_dir, lines: list[str]) -> None:
    path = run_dir / "quality.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _norm_ticker(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper()
    s = s.lstrip("^")
    if "." in s:
        s = s.split(".", 1)[0]
    return s.strip()


def _suffix_from_symbol(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper()
    if "." not in s:
        return ""
    return s.rsplit(".", 1)[-1].strip()


def _ensure_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Ensure SMA-based MA columns exist and recompute derived technical flags."""
    prices = prices.copy()
    need = ["ma21", "ma200"]
    missing_any = any(c not in prices.columns for c in need)
    all_nan = any((c in prices.columns and pd.to_numeric(prices[c], errors="coerce").notna().mean() == 0.0) for c in need)

    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    if missing_any or all_nan:
        prices = prices.sort_values(["ticker_norm", "date"])
        g = prices.groupby("ticker_norm", group_keys=False)
        prices["ma21"] = g["adj_close"].transform(lambda s: s.rolling(21, min_periods=21).mean())
        prices["ma200"] = g["adj_close"].transform(lambda s: s.rolling(200, min_periods=200).mean())
    else:
        prices["ma21"] = pd.to_numeric(prices["ma21"], errors="coerce")
        prices["ma200"] = pd.to_numeric(prices["ma200"], errors="coerce")

    prices["mad"] = (prices["ma21"] - prices["ma200"]) / prices["ma200"]
    prices["above_ma200"] = prices["adj_close"] > prices["ma200"]
    return prices


def _load_ticker_mapping(ctx) -> pd.DataFrame:
    candidates = [
        ctx.project_root / "config" / "tickers.csv",
        ctx.project_root / "config" / "tickers_with_insid_clean.csv",
        ctx.project_root / "config" / "tickers_with_insid.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            cols = {str(c).strip().lower(): c for c in df.columns}
            t_col = cols.get("ticker")
            y_col = cols.get("yahoo_ticker") or cols.get("yahoo")
            if t_col and y_col:
                out = df[[t_col, y_col]].rename(columns={t_col: "ticker", y_col: "yahoo_ticker"}).copy()
                out["ticker_norm"] = out["ticker"].map(_norm_ticker)
                out["yahoo_ticker"] = out["yahoo_ticker"].astype(str).str.strip()
                out = out[(out["ticker_norm"] != "") & (out["yahoo_ticker"].replace("", np.nan).notna())]
                return out[["ticker_norm", "yahoo_ticker"]].drop_duplicates("ticker_norm", keep="first")
    return pd.DataFrame(columns=["ticker_norm", "yahoo_ticker"])


def _load_insid_mapping(ctx) -> pd.DataFrame:
    candidates = [
        ctx.project_root / "config" / "tickers_with_insid_clean.csv",
        ctx.project_root / "config" / "tickers_with_insid.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        cols = {str(c).strip().lower(): c for c in df.columns}
        y_col = cols.get("yahoo_ticker")
        i_col = cols.get("ins_id") or cols.get("insid")
        if not y_col or not i_col:
            continue

        t_col = cols.get("ticker") or cols.get("ticker_x") or cols.get("ticker_y")
        c_col = cols.get("country")
        keep = [y_col, i_col]
        if t_col:
            keep.append(t_col)
        if c_col:
            keep.append(c_col)

        out = df[keep].copy().rename(columns={y_col: "yahoo_ticker", i_col: "ins_id"})
        if t_col:
            out = out.rename(columns={t_col: "ticker"})
        else:
            out["ticker"] = ""
        if c_col:
            out = out.rename(columns={c_col: "country"})
        else:
            out["country"] = ""

        out["yahoo_ticker"] = out["yahoo_ticker"].astype(str).str.strip()
        out["ticker_norm"] = out["ticker"].map(_norm_ticker)
        out["ins_id"] = pd.to_numeric(out["ins_id"], errors="coerce")
        out = out[out["ins_id"].notna()].copy()
        out["ins_id"] = out["ins_id"].astype(int)
        out["market"] = out["country"].map(lambda x: MARKET_BY_COUNTRY.get(str(x).strip().upper(), ""))
        out["yahoo_key"] = out["yahoo_ticker"].str.upper()
        out = out[(out["yahoo_ticker"] != "") | (out["ticker_norm"] != "")]
        return out[["yahoo_ticker", "yahoo_key", "ticker_norm", "ins_id", "market"]].drop_duplicates(
            subset=["yahoo_key", "ticker_norm"], keep="first"
        )

    return pd.DataFrame(columns=["yahoo_ticker", "yahoo_key", "ticker_norm", "ins_id", "market"])


def _normalize_prices_with_mapping(prices: pd.DataFrame, mapping: pd.DataFrame, log) -> pd.DataFrame:
    prices = prices.copy()
    if "ticker" in prices.columns:
        prices["ticker_norm"] = prices["ticker"].map(_norm_ticker)
    else:
        prices["ticker_norm"] = ""

    if "yahoo_ticker" in prices.columns:
        prices["yahoo_ticker"] = prices["yahoo_ticker"].astype(str).str.strip()
    else:
        prices["yahoo_ticker"] = ""

    missing_yahoo = prices["yahoo_ticker"].eq("") | prices["yahoo_ticker"].isna() | prices["yahoo_ticker"].eq("nan")
    if missing_yahoo.any() and "ticker_norm" in prices.columns:
        inferable = missing_yahoo & prices["ticker_norm"].ne("")
        prices.loc[inferable, "yahoo_ticker"] = prices.loc[inferable, "ticker_norm"]
        log.info(f"MASTER: inferred yahoo_ticker from ticker for {int(inferable.sum())} rows")

    if not mapping.empty:
        prices = prices.merge(mapping, on="ticker_norm", how="left", suffixes=("", "_map"))
        needs_map = prices["yahoo_ticker"].eq("") | prices["yahoo_ticker"].isna() | prices["yahoo_ticker"].eq("nan")
        prices.loc[needs_map, "yahoo_ticker"] = prices.loc[needs_map, "yahoo_ticker_map"]
        prices = prices.drop(columns=[c for c in ["yahoo_ticker_map"] if c in prices.columns])
        log.info(f"MASTER: mapped yahoo_ticker from config for {int(needs_map.sum())} candidate rows")

    prices["missing_price_reason"] = np.where(
        prices["yahoo_ticker"].isna() | prices["yahoo_ticker"].eq("") | prices["yahoo_ticker"].eq("nan"),
        "missing_yahoo_ticker",
        "",
    )
    return prices


def _coalesce_master_yahoo_ticker(master: pd.DataFrame) -> pd.DataFrame:
    out = master.copy()
    candidates = [c for c in ["yahoo_ticker", "yahoo_ticker_x", "yahoo_ticker_y"] if c in out.columns]
    if not candidates:
        out["yahoo_ticker"] = ""
        return out

    merged = pd.Series("", index=out.index, dtype=object)
    for c in candidates:
        s = out[c].astype(str).str.strip()
        valid = s.ne("") & s.ne("nan")
        merged = merged.where(merged.str.len() > 0, s.where(valid, ""))

    out["yahoo_ticker"] = merged
    drop_cols = [c for c in ["yahoo_ticker_x", "yahoo_ticker_y"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def _latest_raw_snapshot_dir(raw_dir: Path, asof: str) -> Path | None:
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
        if pd.isna(ts):
            continue
        rows.append((ts.normalize(), d))

    if not rows:
        return None

    eligible = [r for r in rows if r[0] <= asof_dt.normalize()]
    if not eligible:
        return None

    eligible.sort(key=lambda x: x[0], reverse=True)
    return eligible[0][1]


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def canon(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        return s.strip("_")

    out.columns = [canon(c) for c in out.columns]
    return out


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pick_latest_row(df: pd.DataFrame, asof_dt: pd.Timestamp) -> pd.Series | None:
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
        d["year"] = _to_num(d["year"])
    if "period" in d.columns:
        d["period"] = _to_num(d["period"])

    sort_cols = [c for c in ["year", "period"] if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols)
    return d.iloc[-1]


def _annual_frame(df: pd.DataFrame, asof_dt: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = _canon_cols(df)
    if "year" not in d.columns:
        return pd.DataFrame()

    date_col = "report_date" if "report_date" in d.columns else ("report_end_date" if "report_end_date" in d.columns else None)
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d[d[date_col].notna() & (d[date_col] <= asof_dt)]

    d["year"] = _to_num(d["year"])
    d = d[d["year"].notna()].copy()
    if d.empty:
        return d

    if "period" in d.columns:
        d["period"] = _to_num(d["period"])
    else:
        d["period"] = 0

    d = d.sort_values(["year", "period"]).drop_duplicates(subset=["year"], keep="last")
    return d


def _cagr_from_points(latest_year: float, latest_value: float, older_year: float, older_value: float) -> float:
    if not np.isfinite(latest_year) or not np.isfinite(older_year):
        return float("nan")
    if not np.isfinite(latest_value) or not np.isfinite(older_value):
        return float("nan")
    if latest_value <= 0 or older_value <= 0:
        return float("nan")
    years = float(latest_year - older_year)
    if years <= 0:
        return float("nan")
    return float((latest_value / older_value) ** (1.0 / years) - 1.0)


def _calc_growth_5y(annual: pd.DataFrame, value_col: str) -> float:
    if annual.empty or value_col not in annual.columns:
        return float("nan")

    d = annual[["year", value_col]].copy()
    d[value_col] = _to_num(d[value_col])
    d = d[d[value_col].notna() & (d[value_col] > 0)].sort_values("year")
    if len(d) < 2:
        return float("nan")

    latest = d.iloc[-1]
    target_year = float(latest["year"]) - 5.0
    older = d[d["year"] <= target_year]
    if older.empty:
        return float("nan")

    old = older.iloc[-1]
    return _cagr_from_points(float(latest["year"]), float(latest[value_col]), float(old["year"]), float(old[value_col]))


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den):
        return float("nan")
    if den == 0:
        return float("nan")
    return float(num / den)


def _derive_dividend_features(reports_y: pd.DataFrame, reports_r12: pd.DataFrame, asof_dt: pd.Timestamp) -> dict[str, float]:
    out: dict[str, float] = {k: float("nan") for k in DIVIDEND_FEATURE_COLS}

    annual = _annual_frame(reports_y, asof_dt)
    if not annual.empty:
        div = _to_num(annual.get("dividend", pd.Series(np.nan, index=annual.index)))
        hist_mask = div > 0
        out["dividend_history_years"] = float(int(hist_mask.sum()))
        out["dividend_paid_years"] = out["dividend_history_years"]
        out["dividend_growth_5y"] = _calc_growth_5y(annual, "dividend")
        out["dividend_cagr_5y"] = out["dividend_growth_5y"]
        out["profit_growth_5y"] = _calc_growth_5y(annual, "profit_to_equity_holders")
        out["share_count_growth_5y"] = _calc_growth_5y(annual, "number_of_shares")

        latest_a = annual.iloc[-1]
        dps = float(_to_num(pd.Series([latest_a.get("dividend")])).iloc[0])
        eps = float(_to_num(pd.Series([latest_a.get("earnings_per_share")])).iloc[0])
        px = float(_to_num(pd.Series([latest_a.get("stock_price_average")])).iloc[0])

        if np.isfinite(dps) and dps >= 0 and np.isfinite(eps) and eps > 0:
            out["payout_ratio"] = _safe_ratio(dps, eps)
        if np.isfinite(dps) and dps >= 0 and np.isfinite(px) and px > 0:
            out["dividend_yield"] = _safe_ratio(dps, px)

    latest_r12 = _pick_latest_row(reports_r12, asof_dt)
    if latest_r12 is not None:
        rev = float(_to_num(pd.Series([latest_r12.get("revenues", latest_r12.get("net_sales"))])).iloc[0])
        profit = float(_to_num(pd.Series([latest_r12.get("profit_to_equity_holders")])).iloc[0])
        ocf = float(_to_num(pd.Series([latest_r12.get("cash_flow_from_operating_activities")])).iloc[0])
        fcf = float(_to_num(pd.Series([latest_r12.get("free_cash_flow")])).iloc[0])
        assets = float(_to_num(pd.Series([latest_r12.get("total_assets")])).iloc[0])

        out["fcf_margin"] = _safe_ratio(fcf, rev)
        out["ocf_margin"] = _safe_ratio(ocf, rev)
        out["profit_margin"] = _safe_ratio(profit, rev)
        out["roa"] = _safe_ratio(profit, assets)

        if not np.isfinite(out["payout_ratio"]):
            dps = float(_to_num(pd.Series([latest_r12.get("dividend")])).iloc[0])
            eps = float(_to_num(pd.Series([latest_r12.get("earnings_per_share")])).iloc[0])
            if np.isfinite(dps) and dps >= 0 and np.isfinite(eps) and eps > 0:
                out["payout_ratio"] = _safe_ratio(dps, eps)

        if not np.isfinite(out["dividend_yield"]):
            dps = float(_to_num(pd.Series([latest_r12.get("dividend")])).iloc[0])
            px = float(_to_num(pd.Series([latest_r12.get("stock_price_average")])).iloc[0])
            if np.isfinite(dps) and dps >= 0 and np.isfinite(px) and px > 0:
                out["dividend_yield"] = _safe_ratio(dps, px)

    return out


def _build_dividend_feature_snapshot(
    master: pd.DataFrame,
    raw_snapshot_dir: Path | None,
    asof_dt: pd.Timestamp,
    ins_map: pd.DataFrame,
) -> pd.DataFrame:
    if raw_snapshot_dir is None or not raw_snapshot_dir.exists() or ins_map.empty:
        return pd.DataFrame(columns=["yahoo_ticker"] + DIVIDEND_FEATURE_COLS)

    if "yahoo_ticker" not in master.columns:
        return pd.DataFrame(columns=["yahoo_ticker"] + DIVIDEND_FEATURE_COLS)

    work = master[[c for c in ["ticker", "yahoo_ticker"] if c in master.columns]].copy()
    if "ticker" not in work.columns:
        work["ticker"] = ""
    work["ticker_norm"] = work["ticker"].map(_norm_ticker)
    work["yahoo_ticker"] = work["yahoo_ticker"].astype(str).str.strip()
    work["yahoo_key"] = work["yahoo_ticker"].str.upper()

    m = work.merge(ins_map, on="yahoo_key", how="left", suffixes=("", "_map"))
    missing_ins = m["ins_id"].isna() & m["ticker_norm"].ne("")
    if missing_ins.any():
        by_ticker = ins_map.dropna(subset=["ticker_norm"]).drop_duplicates(subset=["ticker_norm"], keep="first")
        by_ticker = by_ticker[["ticker_norm", "ins_id", "market"]].rename(
            columns={"ins_id": "ins_id_t", "market": "market_t"}
        )
        m = m.merge(by_ticker, on="ticker_norm", how="left")
        m.loc[missing_ins, "ins_id"] = m.loc[missing_ins, "ins_id_t"]
        m.loc[missing_ins, "market"] = m.loc[missing_ins, "market_t"]
        m = m.drop(columns=[c for c in ["ins_id_t", "market_t"] if c in m.columns])

    market_from_suffix = m["yahoo_ticker"].map(lambda x: MARKET_BY_SUFFIX.get(_suffix_from_symbol(x), ""))
    m["market"] = m["market"].astype(str)
    m.loc[m["market"].eq("") | m["market"].eq("nan"), "market"] = market_from_suffix

    m["ins_id"] = pd.to_numeric(m["ins_id"], errors="coerce")
    m = m[m["ins_id"].notna() & m["market"].astype(str).ne("")].copy()
    if m.empty:
        return pd.DataFrame(columns=["yahoo_ticker"] + DIVIDEND_FEATURE_COLS)

    m["ins_id"] = m["ins_id"].astype(int)
    m = m.drop_duplicates(subset=["yahoo_ticker", "ins_id", "market"], keep="first")

    rows: list[dict[str, float | str]] = []
    for _, r in m.iterrows():
        market = str(r["market"]).upper()
        ins_id = int(r["ins_id"])
        p_y = raw_snapshot_dir / "reports_y" / f"market={market}" / f"ins_id={ins_id}.parquet"
        p_r12 = raw_snapshot_dir / "reports_r12" / f"market={market}" / f"ins_id={ins_id}.parquet"
        if (not p_y.exists()) and (not p_r12.exists()):
            continue

        rep_y = pd.read_parquet(p_y) if p_y.exists() else pd.DataFrame()
        rep_r12 = pd.read_parquet(p_r12) if p_r12.exists() else pd.DataFrame()
        feat = _derive_dividend_features(rep_y, rep_r12, asof_dt=asof_dt)

        row: dict[str, float | str] = {"yahoo_ticker": str(r["yahoo_ticker"])}
        row.update(feat)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["yahoo_ticker"] + DIVIDEND_FEATURE_COLS)

    out = pd.DataFrame(rows).drop_duplicates(subset=["yahoo_ticker"], keep="first")
    return out


def run(ctx, log) -> int:
    paths = resolve_paths(ctx.cfg, ctx.project_root)
    raw_dir = paths["raw_dir"]
    processed_dir = paths["processed_dir"]

    # 1) Base master: prefer raw/master_input if present (som hos deg)
    master_input = raw_dir / "master_input.parquet"
    if master_input.exists():
        master = read_parquet(master_input)
        log.info("MASTER: base from master_input")
    else:
        # fallback: fundamentals snapshot
        fpath = processed_dir / "fundamentals.parquet"
        master = read_parquet(fpath)
        log.info("MASTER: base from fundamentals.parquet")

    if "ticker" not in master.columns:
        raise ValueError("MASTER: missing 'ticker' in base dataset")

    master = master.copy()
    master["ticker_norm"] = master["ticker"].map(_norm_ticker)

    # 2) Prices snapshot (siste dato <= asof) ALWAYS merged in
    ppath = processed_dir / "prices.parquet"
    if not ppath.exists():
        raise ValueError(f"MASTER: missing {ppath} (run transform_prices first)")

    prices = read_parquet(ppath).copy()
    if "ticker" not in prices.columns or "date" not in prices.columns:
        raise ValueError("MASTER: prices.parquet must have ticker,date")

    mapping = _load_ticker_mapping(ctx)
    prices = _normalize_prices_with_mapping(prices, mapping, log)

    prices["date"] = pd.to_datetime(prices["date"])
    prices["ticker_norm"] = prices["ticker"].map(_norm_ticker)

    # unngå å blande inn indekser i master (tickere som starter med ^)
    # (etter norm er ^OSEAX -> OSEAX, så filtrer før norm hvis felt finnes)
    if prices["ticker"].astype(str).str.startswith("^").any():
        prices = prices[~prices["ticker"].astype(str).str.startswith("^")].copy()

    asof_dt = pd.to_datetime(ctx.asof)
    prices = prices[prices["date"] <= asof_dt].copy()

    # compute MA/MAD if needed
    prices = _ensure_price_features(prices)

    snap = (
        prices.sort_values(["ticker_norm", "date"])
        .groupby("ticker_norm", as_index=False)
        .tail(1)
        .copy()
    )

    keep_cols = [c for c in ["ticker_norm", "yahoo_ticker", "date", "adj_close", "volume", "ma21", "ma200", "mad", "above_ma200", "missing_price_reason"] if c in snap.columns]
    snap = snap[keep_cols]

    # 3) Drop any old price cols in master (prevents "all NaN" from master_input)
    drop_old = [c for c in ["date", "adj_close", "volume", "ma21", "ma200", "mad", "above_ma200", "missing_price", "missing_technical_data", "reason_technical_fail"] if c in master.columns]
    if drop_old:
        master = master.drop(columns=drop_old)

    master = master.merge(snap, on="ticker_norm", how="left")
    master = _coalesce_master_yahoo_ticker(master)

    # 4) Enrich dividend-quality features from raw reports snapshot (as-of)
    raw_snapshot_dir = _latest_raw_snapshot_dir(raw_dir, ctx.asof)
    ins_map = _load_insid_mapping(ctx)
    div_snapshot = _build_dividend_feature_snapshot(master=master, raw_snapshot_dir=raw_snapshot_dir, asof_dt=asof_dt, ins_map=ins_map)
    if not div_snapshot.empty:
        for col in DIVIDEND_FEATURE_COLS:
            if col not in div_snapshot.columns:
                div_snapshot[col] = np.nan

        merged = master.merge(
            div_snapshot[["yahoo_ticker"] + DIVIDEND_FEATURE_COLS],
            on="yahoo_ticker",
            how="left",
            suffixes=("", "_derived"),
        )
        for c in DIVIDEND_FEATURE_COLS:
            dc = f"{c}_derived"
            if dc not in merged.columns:
                continue
            if c in merged.columns:
                merged[c] = merged[c].where(merged[c].notna(), merged[dc])
            else:
                merged[c] = merged[dc]
        merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_derived")])
        master = merged

    price_s = pd.to_numeric(master.get("adj_close"), errors="coerce")
    ma200_s = pd.to_numeric(master.get("ma200"), errors="coerce")
    mad_s = pd.to_numeric(master.get("mad"), errors="coerce")

    master["above_ma200"] = pd.Series(
        np.where(price_s.notna() & ma200_s.notna(), price_s > ma200_s, pd.NA),
        index=master.index,
        dtype="boolean",
    )
    master["ma200_ok"] = price_s.notna() & ma200_s.notna() & (price_s > ma200_s)
    master["index_ma200_ok"] = True
    master["missing_price"] = price_s.isna()
    master["missing_technical_data"] = (~master["missing_price"]) & (ma200_s.isna() | mad_s.isna())
    master["reason_technical_fail"] = ""
    master.loc[master["missing_price"], "reason_technical_fail"] = "missing_price"
    master.loc[master["missing_technical_data"], "reason_technical_fail"] = "missing_technical_data"
    master.loc[~master["missing_price"] & ~master["missing_technical_data"] & ~master["ma200_ok"], "reason_technical_fail"] = "below_ma200"
    if "mad" in master.columns:
        master.loc[~master["missing_price"] & ~master["missing_technical_data"] & master["reason_technical_fail"].eq("") & mad_s.lt(0), "reason_technical_fail"] = "negative_mad"
    master = master.drop(columns=["ticker_norm"])

    # Coverage log
    cov = float(pd.to_numeric(master["adj_close"], errors="coerce").notna().mean()) if "adj_close" in master.columns else 0.0
    log.info(f"MASTER: price coverage adj_close={cov:.3f}")

    dividend_coverage: dict[str, float] = {}
    for col in DIVIDEND_FEATURE_COLS:
        if col in master.columns:
            dividend_coverage[col] = float(pd.to_numeric(master[col], errors="coerce").notna().mean())

    if raw_snapshot_dir is not None:
        log.info(f"MASTER: dividend feature source snapshot={raw_snapshot_dir.name}")
    if dividend_coverage:
        top_cov = ", ".join([f"{k}={v:.1%}" for k, v in dividend_coverage.items()])
        log.info(f"MASTER: dividend feature coverage {top_cov}")

    # Data quality summary for explainability
    missing_count = int(master["missing_price"].sum()) if "missing_price" in master.columns else 0
    no_yahoo = int((master.get("yahoo_ticker").astype(str).str.strip() == "").sum()) if "yahoo_ticker" in master.columns else len(master)
    quality_lines = [
        f"# Data quality ({ctx.asof})",
        "",
        f"- Universe rows: {len(master)}",
        f"- Price coverage (adj_close): {cov:.1%}",
        f"- Rows missing price: {missing_count}",
        f"- Rows without yahoo_ticker in master: {no_yahoo}",
        f"- Dividend snapshot used (<= asof): {raw_snapshot_dir.name if raw_snapshot_dir is not None else 'none'}",
        "",
        "## Technical fail reasons (top)",
    ]
    reason_counts = master.get("reason_technical_fail", pd.Series(dtype=str)).value_counts(dropna=False)
    if len(reason_counts) == 0:
        quality_lines.append("- none")
    else:
        for reason, count in reason_counts.head(10).items():
            r = reason if isinstance(reason, str) and reason else "ok"
            quality_lines.append(f"- {r}: {int(count)}")

    quality_lines.append("")
    quality_lines.append("## Dividend Feature Coverage")
    if dividend_coverage:
        for col, share in dividend_coverage.items():
            quality_lines.append(f"- {col}: {share:.1%}")
    else:
        quality_lines.append("- none")

    _quality_write(ctx.run_dir, quality_lines)

    out = processed_dir / "master.parquet"
    master.to_parquet(out, index=False)
    log.info(f"MASTER: wrote {out} (from master_input + prices snapshot)")
    return 0
