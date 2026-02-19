from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from src.common.config import load_config, project_root_from_file, resolve_paths


@dataclass(frozen=True)
class WFTParams:
    mos_threshold: float
    mad_min: float
    weakness_rule_variant: str


def _norm_ticker(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper().lstrip("^")
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


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_decimal_rate(s: pd.Series) -> pd.Series:
    x = _to_num(s)
    if int(x.notna().sum()) == 0:
        return x
    med = x.abs().median(skipna=True)
    if np.isfinite(med) and med > 2.0:
        x = x / 100.0
    return x


def _zscore(s: pd.Series) -> pd.Series:
    x = _to_num(s)
    if int(x.notna().sum()) < 2:
        return pd.Series(0.0, index=s.index)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (x - mu) / sd


def _index_map_by_suffix() -> dict[str, str]:
    return {
        "OL": "^OSEAX",
        "ST": "^OMXS",
        "CO": "^OMXC25",
        "HE": "^HEX",
    }


def _is_bank_proxy(df: pd.DataFrame) -> pd.Series:
    company = df.get("company", pd.Series("", index=df.index)).astype(str).str.lower()
    ticker = df.get("ticker", pd.Series("", index=df.index)).astype(str).str.upper()

    keyword_hit = company.str.contains(r"\bbank\b|sparebank|banc|finans|finance|kredit", regex=True)
    ticker_hit = ticker.isin({"DNB", "NDA", "SEB", "SWED", "SHB", "DANSKE", "JYSK", "SYDB"})
    return keyword_hit | ticker_hit


def _max_drawdown(ret: pd.Series) -> float:
    r = _to_num(ret).fillna(0.0)
    nav = (1.0 + r).cumprod()
    if nav.empty:
        return 0.0
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def _annualized_sharpe(ret: pd.Series) -> float:
    r = _to_num(ret).fillna(0.0)
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return 0.0
    return float(r.mean() / sd * math.sqrt(12.0))


def _cagr_from_monthly(ret: pd.Series) -> float:
    r = _to_num(ret).fillna(0.0)
    n = len(r)
    if n == 0:
        return 0.0
    nav = float((1.0 + r).prod())
    if nav <= 0:
        return -1.0
    return float(nav ** (12.0 / n) - 1.0)


def _stability(turnover: float) -> float:
    return float(1.0 - turnover)


def _param_to_json(p: WFTParams) -> str:
    return json.dumps(
        {
            "mos_threshold": float(p.mos_threshold),
            "mad_min": float(p.mad_min),
            "weakness_rule_variant": str(p.weakness_rule_variant),
        },
        sort_keys=True,
    )


def _iter_param_grid(
    mos_grid: Iterable[float],
    mad_grid: Iterable[float],
    weakness_variants: Iterable[str],
) -> list[WFTParams]:
    out: list[WFTParams] = []
    for mos in mos_grid:
        for mad in mad_grid:
            for wv in weakness_variants:
                out.append(WFTParams(float(mos), float(mad), str(wv)))
    return out


def _build_folds(
    monthly_dates: pd.Series,
    train_years: int = 12,
    test_years: int = 1,
    step_years: int = 1,
    require_full_test_year: bool = True,
) -> list[dict]:
    months = pd.to_datetime(monthly_dates, errors="coerce").dropna().sort_values().unique()
    if len(months) == 0:
        return []

    month_df = pd.DataFrame({"month": pd.to_datetime(months)})
    month_df["year"] = month_df["month"].dt.year

    by_year = month_df.groupby("year")["month"].apply(list).to_dict()
    years = sorted(by_year.keys())
    first_test_year = years[0] + train_years

    folds: list[dict] = []
    y = first_test_year
    while y <= years[-1]:
        test_year_range = list(range(y, y + test_years))
        if any(yy not in by_year for yy in test_year_range):
            y += step_years
            continue

        if require_full_test_year and any(len(by_year[yy]) < 12 for yy in test_year_range):
            y += step_years
            continue

        train_year_range = list(range(y - train_years, y))
        if any(yy not in by_year for yy in train_year_range):
            y += step_years
            continue

        train_months = sorted([m for yy in train_year_range for m in by_year[yy]])
        test_months = sorted([m for yy in test_year_range for m in by_year[yy]])
        if not train_months or len(test_months) < 2:
            y += step_years
            continue

        folds.append(
            {
                "train_start": pd.Timestamp(train_months[0]),
                "train_end": pd.Timestamp(train_months[-1]),
                "test_start": pd.Timestamp(test_months[0]),
                "test_end": pd.Timestamp(test_months[-1]),
                "test_year": int(y),
            }
        )
        y += step_years

    return folds


def _load_static_universe(master_path: Path) -> pd.DataFrame:
    m = pd.read_parquet(master_path).copy()
    m.columns = [str(c).strip().lower().replace(" ", "_") for c in m.columns]

    if "ticker" not in m.columns:
        raise RuntimeError("master_valued mangler ticker")

    if "yahoo_ticker" not in m.columns:
        m["yahoo_ticker"] = m["ticker"]

    m["k"] = m["yahoo_ticker"].map(_norm_ticker)
    m = m[m["k"].ne("")].copy()

    mcap = _to_num(m.get("market_cap_current", m.get("market_cap", pd.Series(np.nan, index=m.index))))
    med = float(mcap.median(skipna=True)) if mcap.notna().any() else float("nan")
    if np.isfinite(med) and med < 1e6:
        mcap = mcap * 1_000_000.0

    net_debt = _to_num(m.get("net_debt_current", m.get("net_debt", m.get("net_debt_used", pd.Series(0.0, index=m.index)))))
    intrinsic = _to_num(m.get("intrinsic_equity", pd.Series(np.nan, index=m.index)))
    mos = intrinsic / mcap - 1.0

    roic = _to_decimal_rate(_to_num(m.get("roic", m.get("roic_current", pd.Series(np.nan, index=m.index)))))
    wacc = _to_decimal_rate(_to_num(m.get("wacc_used", m.get("wacc", pd.Series(np.nan, index=m.index)))))
    roe = _to_decimal_rate(_to_num(m.get("roe", m.get("roe_current", pd.Series(np.nan, index=m.index)))))
    coe = _to_decimal_rate(_to_num(m.get("coe_used", m.get("coe", pd.Series(np.nan, index=m.index)))))

    nd_ebitda = _to_num(m.get("nd_ebitda", m.get("n_debt_ebitda_current", pd.Series(np.nan, index=m.index))))
    beta = _to_num(m.get("beta", pd.Series(np.nan, index=m.index)))
    ev_ebit = _to_num(m.get("ev_ebit", m.get("ev_ebit_current", pd.Series(np.nan, index=m.index))))
    fcf_yield = _to_num(m.get("fcf_yield", pd.Series(np.nan, index=m.index)))

    high_risk_flag = (beta.fillna(0) >= 1.5) | (nd_ebitda.fillna(0) >= 3.5)

    weak_roic = roic.isna() | (roic <= 0.0)
    weak_fcf = fcf_yield.isna() | (fcf_yield <= 0.0)
    weak_nd = nd_ebitda.isna() | (nd_ebitda > 3.5)
    weak_ev = ev_ebit.isna() | (ev_ebit <= 0) | (ev_ebit > 20.0)
    quality_weak_count = weak_roic.astype(int) + weak_fcf.astype(int) + weak_nd.astype(int) + weak_ev.astype(int)

    spread_non_bank = roic - wacc
    decay = 0.01
    nonbank_value_creation_ok = (
        roic.notna() & wacc.notna() &
        ((spread_non_bank - decay) > 0) &
        ((spread_non_bank - 2 * decay) > 0) &
        ((spread_non_bank - 3 * decay) > 0)
    )

    bank_proxy = _is_bank_proxy(m)
    spread_bank = roe - coe
    bank_value_creation_ok = (
        bank_proxy &
        m.get("model", pd.Series("", index=m.index)).astype(str).str.upper().eq("RIM") &
        roe.notna() & coe.notna() &
        ((spread_bank - decay) > 0) &
        ((spread_bank - 2 * decay) > 0) &
        ((spread_bank - 3 * decay) > 0)
    )

    value_creation_ok_base = np.where(bank_proxy, bank_value_creation_ok, nonbank_value_creation_ok)

    suffix = m["yahoo_ticker"].map(_suffix_from_symbol)
    idx_map = _index_map_by_suffix()
    relevant_index_symbol = suffix.map(idx_map).fillna("")
    relevant_index_key = relevant_index_symbol.map(_norm_ticker)

    out = pd.DataFrame(
        {
            "k": m["k"],
            "ticker": m["ticker"],
            "company": m.get("company", pd.Series("", index=m.index)),
            "market_cap": mcap,
            "mos": mos,
            "high_risk_flag": high_risk_flag.astype(bool),
            "quality_weak_count": quality_weak_count,
            "value_creation_ok_base": pd.Series(value_creation_ok_base, index=m.index).astype(bool),
            "roic": roic,
            "fcf_yield": fcf_yield,
            "relevant_index_symbol": relevant_index_symbol,
            "relevant_index_key": relevant_index_key,
            "is_bank_proxy": bank_proxy.astype(bool),
        }
    )

    out = out.sort_values(["k", "market_cap"], ascending=[True, False]).drop_duplicates(subset=["k"], keep="first")
    return out.reset_index(drop=True)


def _load_monthly_panel(prices_path: Path, static_universe: pd.DataFrame) -> pd.DataFrame:
    px = pd.read_parquet(prices_path).copy()
    px.columns = [str(c).strip().lower().replace(" ", "_") for c in px.columns]

    for c in ["ticker", "date", "adj_close"]:
        if c not in px.columns:
            raise RuntimeError(f"prices.parquet mangler {c}")

    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["adj_close"] = _to_num(px["adj_close"])
    px = px.dropna(subset=["ticker", "date", "adj_close"]).copy()

    if "ma21" not in px.columns or "ma200" not in px.columns:
        px = px.sort_values(["ticker", "date"])
        g = px.groupby("ticker", group_keys=False)
        px["ma21"] = g["adj_close"].transform(lambda s: s.rolling(21, min_periods=21).mean())
        px["ma200"] = g["adj_close"].transform(lambda s: s.rolling(200, min_periods=200).mean())

    if "mad" not in px.columns:
        px["mad"] = (px["ma21"] - px["ma200"]) / px["ma200"]

    if "above_ma200" not in px.columns:
        px["above_ma200"] = px["adj_close"] > px["ma200"]

    px["month"] = px["date"].dt.to_period("M").dt.to_timestamp("M")

    snap = (
        px.sort_values(["month", "ticker", "date"]) 
        .groupby(["month", "ticker"], as_index=False)
        .tail(1)
        .copy()
    )
    snap["k"] = snap["ticker"].map(_norm_ticker)

    idx = snap[snap["ticker"].astype(str).str.startswith("^")].copy()
    idx = idx.rename(
        columns={
            "k": "relevant_index_key",
            "adj_close": "index_price",
            "ma21": "index_ma21",
            "ma200": "index_ma200",
            "mad": "index_mad",
            "above_ma200": "index_above_ma200",
        }
    )[["month", "relevant_index_key", "index_price", "index_ma21", "index_ma200", "index_mad", "index_above_ma200"]]

    stk = snap[~snap["ticker"].astype(str).str.startswith("^")].copy()
    stk = stk[["month", "ticker", "k", "adj_close", "ma21", "ma200", "mad", "above_ma200"]]

    static = static_universe.copy()
    if "ticker" in static.columns:
        static = static.rename(columns={"ticker": "master_ticker"})

    panel = stk.merge(static, on="k", how="inner")
    panel = panel.merge(idx, on=["month", "relevant_index_key"], how="left")

    panel["index_data_ok"] = panel["index_price"].notna() & panel["index_ma200"].notna() & panel["index_mad"].notna()

    return panel.sort_values(["month", "k"]).reset_index(drop=True)


def _apply_filters(month_df: pd.DataFrame, params: WFTParams) -> pd.DataFrame:
    d = month_df.copy()

    base_mos = float(params.mos_threshold)
    mos_req = np.where(d["high_risk_flag"].fillna(False), np.maximum(0.40, base_mos), base_mos)

    weak_fail_min = 2 if params.weakness_rule_variant == "baseline" else 1

    d["fundamental_ok"] = (
        d["mos"].notna() &
        (d["mos"] >= mos_req) &
        d["value_creation_ok_base"].fillna(False).astype(bool) &
        (pd.to_numeric(d["quality_weak_count"], errors="coerce") < weak_fail_min)
    )

    d["technical_ok"] = (
        d["above_ma200"].astype("boolean").fillna(False).astype(bool) &
        pd.to_numeric(d["mad"], errors="coerce").ge(float(params.mad_min)).fillna(False) &
        d["index_data_ok"].astype(bool) &
        d["index_above_ma200"].astype("boolean").fillna(False).astype(bool) &
        pd.to_numeric(d["index_mad"], errors="coerce").ge(float(params.mad_min)).fillna(False)
    )

    d["eligible"] = d["fundamental_ok"] & d["technical_ok"]

    roic_z = _zscore(d["roic"])
    fcf_z = _zscore(d["fcf_yield"])
    if roic_z.notna().any() and fcf_z.notna().any():
        d["quality_score"] = 0.6 * roic_z + 0.4 * fcf_z
    elif roic_z.notna().any():
        d["quality_score"] = roic_z
    elif fcf_z.notna().any():
        d["quality_score"] = fcf_z
    else:
        d["quality_score"] = 0.0

    return d


def _pick_ticker(month_df: pd.DataFrame, params: WFTParams) -> str:
    d = _apply_filters(month_df, params)
    elig = d[d["eligible"]].copy()
    if elig.empty:
        return "CASH"

    elig = elig.sort_values(
        by=["quality_score", "mos", "market_cap", "ticker"],
        ascending=[False, False, False, True],
        na_position="last",
        kind="mergesort",
    )
    return str(elig.iloc[0]["k"])


def _simulate_window(window_df: pd.DataFrame, params: WFTParams) -> pd.DataFrame:
    months = sorted(window_df["month"].dropna().unique().tolist())
    if len(months) < 2:
        return pd.DataFrame(columns=["month", "position", "ret"])

    rows: list[dict] = []
    for i in range(len(months) - 1):
        m = months[i]
        n = months[i + 1]
        cur = window_df[window_df["month"] == m]
        nxt = window_df[window_df["month"] == n]

        pos = _pick_ticker(cur, params)
        if pos == "CASH":
            ret = 0.0
        else:
            c = cur[cur["k"] == pos]
            x = nxt[nxt["k"] == pos]
            if c.empty or x.empty:
                ret = 0.0
                pos = "CASH"
            else:
                p0 = float(c.iloc[0]["adj_close"])
                p1 = float(x.iloc[0]["adj_close"])
                ret = float((p1 / p0) - 1.0) if np.isfinite(p0) and p0 > 0 and np.isfinite(p1) else 0.0

        rows.append({"month": pd.Timestamp(m), "position": pos, "ret": ret})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    prev = out["position"].shift(1)
    out["position_change"] = (out["position"] != prev).astype(int)
    out.loc[out.index[0], "position_change"] = 0
    out["is_cash"] = out["position"].eq("CASH")
    return out


def _window_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "return": 0.0,
            "max_dd": 0.0,
            "turnover": 0.0,
            "pct_cash": 1.0,
            "stability": 1.0,
            "sharpe_ann": 0.0,
            "cagr": 0.0,
        }

    ret = _to_num(trades["ret"]).fillna(0.0)
    changes = _to_num(trades["position_change"]).fillna(0.0)
    is_cash = trades["is_cash"].astype(bool)

    total_return = float((1.0 + ret).prod() - 1.0)
    turnover = float(changes.iloc[1:].mean()) if len(changes) > 1 else 0.0
    pct_cash = float(is_cash.mean()) if len(is_cash) else 1.0

    return {
        "return": total_return,
        "max_dd": _max_drawdown(ret),
        "turnover": turnover,
        "pct_cash": pct_cash,
        "stability": _stability(turnover),
        "sharpe_ann": _annualized_sharpe(ret),
        "cagr": _cagr_from_monthly(ret),
    }


def _complexity_distance(p: WFTParams, baseline: WFTParams) -> float:
    d = abs(float(p.mos_threshold) - float(baseline.mos_threshold))
    d += abs(float(p.mad_min) - float(baseline.mad_min))
    d += 0.1 if p.weakness_rule_variant != baseline.weakness_rule_variant else 0.0
    return float(d)


def _objective(metrics: dict) -> float:
    return float(metrics["sharpe_ann"] - 0.25 * metrics["pct_cash"] - 0.10 * metrics["turnover"])


def tune_params(
    train_df: pd.DataFrame,
    grid: list[WFTParams],
    baseline: WFTParams,
    seed: int = 42,
    overfit_guard_eps: float = 0.02,
) -> tuple[WFTParams, pd.DataFrame]:
    if not grid:
        raise ValueError("grid empty")

    rows = []
    for p in grid:
        trades = _simulate_window(train_df, p)
        m = _window_metrics(trades)
        rows.append(
            {
                "params": _param_to_json(p),
                "mos_threshold": p.mos_threshold,
                "mad_min": p.mad_min,
                "weakness_rule_variant": p.weakness_rule_variant,
                "train_return": m["return"],
                "train_sharpe": m["sharpe_ann"],
                "train_turnover": m["turnover"],
                "train_pct_cash": m["pct_cash"],
                "train_objective": _objective(m),
                "complexity": _complexity_distance(p, baseline),
            }
        )

    diag = pd.DataFrame(rows)
    if diag.empty:
        return baseline, diag

    rng = np.random.default_rng(int(seed))
    diag = diag.copy()
    diag["seed_tiebreak"] = rng.random(len(diag))

    best = float(diag["train_objective"].max())
    pool = diag[diag["train_objective"] >= (best - overfit_guard_eps)].copy()

    pool = pool.sort_values(
        by=["complexity", "train_objective", "seed_tiebreak", "mos_threshold", "mad_min", "weakness_rule_variant"],
        ascending=[True, False, True, True, True, True],
        kind="mergesort",
    )

    chosen_row = pool.iloc[0]
    chosen = WFTParams(
        mos_threshold=float(chosen_row["mos_threshold"]),
        mad_min=float(chosen_row["mad_min"]),
        weakness_rule_variant=str(chosen_row["weakness_rule_variant"]),
    )
    return chosen, diag


def run_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    grid: list[WFTParams],
    baseline: WFTParams,
    seed: int,
) -> tuple[WFTParams, dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chosen, diag = tune_params(train_df, grid, baseline=baseline, seed=seed)

    tuned_trades = _simulate_window(test_df, chosen)
    base_trades = _simulate_window(test_df, baseline)

    tuned_metrics = _window_metrics(tuned_trades)
    base_metrics = _window_metrics(base_trades)
    return chosen, tuned_metrics, base_metrics, tuned_trades, base_trades, diag


def _candidate_or_cash(pct_cash: float) -> str:
    return "CASH" if float(pct_cash) >= 0.999 else "CANDIDATE"


def _aggregate_from_monthly(monthly_perf: pd.DataFrame) -> dict:
    if monthly_perf.empty:
        return {"cagr": 0.0, "max_dd": 0.0, "turnover": 0.0, "pct_cash": 1.0, "stability": 1.0}

    ret = _to_num(monthly_perf["ret"]).fillna(0.0)
    turn = _to_num(monthly_perf["position_change"]).fillna(0.0)
    is_cash = monthly_perf["position"].astype(str).eq("CASH")

    turnover = float(turn.iloc[1:].mean()) if len(turn) > 1 else 0.0
    pct_cash = float(is_cash.mean()) if len(is_cash) else 1.0

    return {
        "cagr": _cagr_from_monthly(ret),
        "max_dd": _max_drawdown(ret),
        "turnover": turnover,
        "pct_cash": pct_cash,
        "stability": _stability(turnover),
    }


def _choose_global_tuned_params(fold_df: pd.DataFrame, baseline: WFTParams) -> tuple[WFTParams, str]:
    tuned_rows = fold_df[fold_df["mode"] == "tuned"].copy()
    if tuned_rows.empty:
        return baseline, "No tuned folds available; fallback to baseline."

    grp = tuned_rows.groupby("chosen_params_json", as_index=False).agg(
        n=("test_year", "count"),
        avg_test_return=("return", "mean"),
    )
    grp = grp.sort_values(by=["n", "avg_test_return", "chosen_params_json"], ascending=[False, False, True], kind="mergesort")
    chosen_json = str(grp.iloc[0]["chosen_params_json"])
    obj = json.loads(chosen_json)
    chosen = WFTParams(
        mos_threshold=float(obj["mos_threshold"]),
        mad_min=float(obj["mad_min"]),
        weakness_rule_variant=str(obj["weakness_rule_variant"]),
    )

    reason = (
        f"Chosen by frequency across folds (n={int(grp.iloc[0]['n'])}) and avg test return "
        f"({float(grp.iloc[0]['avg_test_return']):.4f}) with overfit guard against baseline proximity."
    )
    return chosen, reason


def _write_summary_md(path: Path, baseline: dict, tuned: dict, n_folds: int) -> None:
    lines = [
        "# WFT Summary",
        "",
        f"Folds: {n_folds}",
        "",
        "| Metric | Baseline | Tuned |",
        "| --- | ---: | ---: |",
        f"| CAGR | {baseline['cagr']:.2%} | {tuned['cagr']:.2%} |",
        f"| MaxDD | {baseline['max_dd']:.2%} | {tuned['max_dd']:.2%} |",
        f"| Turnover | {baseline['turnover']:.2%} | {tuned['turnover']:.2%} |",
        f"| pct_cash | {baseline['pct_cash']:.2%} | {tuned['pct_cash']:.2%} |",
        f"| Stability | {baseline['stability']:.2%} | {tuned['stability']:.2%} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_wft(
    project_root: Path,
    config_path: Path,
    run_dir: Path,
    train_years: int,
    test_years: int,
    step_years: int,
    seed: int,
    mos_grid: list[float],
    mad_grid: list[float],
    weakness_variants: list[str],
    start_year: int | None = None,
    end_year: int | None = None,
    asof: str | None = None,
    master_path: Path | None = None,
    prices_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    cfg = load_config(config_path)
    paths = resolve_paths(cfg, project_root)

    if master_path is None:
        master_path = paths["processed_dir"] / "master_valued.parquet"
    if prices_path is None:
        prices_path = paths["processed_dir"] / "prices.parquet"

    static = _load_static_universe(master_path)
    panel = _load_monthly_panel(prices_path, static)
    if asof:
        cutoff_month = pd.to_datetime(asof).to_period("M").to_timestamp("M")
        panel = panel[pd.to_datetime(panel["month"], errors="coerce") <= cutoff_month].copy()

    folds = _build_folds(panel["month"], train_years=train_years, test_years=test_years, step_years=step_years)
    if not folds:
        raise RuntimeError("No valid WFT folds found for given windows and data range.")
    if start_year is not None or end_year is not None:
        lo = int(start_year) if start_year is not None else min(int(f["test_year"]) for f in folds)
        hi = int(end_year) if end_year is not None else max(int(f["test_year"]) for f in folds)
        folds = [f for f in folds if int(f["test_year"]) >= lo and int(f["test_year"]) <= hi]
    if not folds:
        raise RuntimeError("No WFT folds found inside requested --start/--end range.")

    grid = _iter_param_grid(mos_grid, mad_grid, weakness_variants)
    baseline = WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")

    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    perf_rows: list[dict] = []

    for i, fold in enumerate(folds):
        train_df = panel[(panel["month"] >= fold["train_start"]) & (panel["month"] <= fold["train_end"])].copy()
        test_df = panel[(panel["month"] >= fold["test_start"]) & (panel["month"] <= fold["test_end"])].copy()

        chosen, tuned_metrics, base_metrics, tuned_trades, base_trades, diag = run_fold(
            train_df=train_df,
            test_df=test_df,
            grid=grid,
            baseline=baseline,
            seed=seed + i,
        )

        for mode, p, metrics, trades in [
            ("baseline", baseline, base_metrics, base_trades),
            ("tuned", chosen, tuned_metrics, tuned_trades),
        ]:
            rows.append(
                {
                    "mode": mode,
                    "train_start": fold["train_start"].date().isoformat(),
                    "train_end": fold["train_end"].date().isoformat(),
                    "test_year": int(fold["test_year"]),
                    "chosen_params": _param_to_json(p),
                    "chosen_params_json": _param_to_json(p),
                    "candidate_or_cash": _candidate_or_cash(metrics["pct_cash"]),
                    "return": metrics["return"],
                    "max_dd": metrics["max_dd"],
                    "turnover": metrics["turnover"],
                    "pct_cash": metrics["pct_cash"],
                }
            )

            if not trades.empty:
                t = trades.copy()
                t["mode"] = mode
                t["test_year"] = int(fold["test_year"])
                perf_rows.append(t)

        diag_out = run_dir / f"wft_tuning_diag_{int(fold['test_year'])}.csv"
        diag.to_csv(diag_out, index=False)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(["test_year", "mode"], kind="mergesort")

    perf_df = pd.concat(perf_rows, ignore_index=True) if perf_rows else pd.DataFrame(columns=["month", "position", "ret", "position_change", "is_cash", "mode", "test_year"])

    base_agg = _aggregate_from_monthly(perf_df[perf_df["mode"] == "baseline"]) if not perf_df.empty else _aggregate_from_monthly(pd.DataFrame())
    tuned_agg = _aggregate_from_monthly(perf_df[perf_df["mode"] == "tuned"]) if not perf_df.empty else _aggregate_from_monthly(pd.DataFrame())

    wft_results_path = run_dir / "wft_results.csv"
    results_df[[
        "mode", "train_start", "train_end", "test_year", "chosen_params", "candidate_or_cash", "return", "max_dd", "turnover", "pct_cash"
    ]].to_csv(wft_results_path, index=False)

    wft_summary_path = run_dir / "wft_summary.md"
    _write_summary_md(wft_summary_path, baseline=base_agg, tuned=tuned_agg, n_folds=len(folds))

    chosen_global, rationale = _choose_global_tuned_params(results_df, baseline=baseline)
    tuned_cfg = {
        "wft": {
            "seed": int(seed),
            "train_years": int(train_years),
            "test_years": int(test_years),
            "step_years": int(step_years),
            "selected_params": {
                "mos_threshold": float(chosen_global.mos_threshold),
                "mad_min": float(chosen_global.mad_min),
                "weakness_rule_variant": str(chosen_global.weakness_rule_variant),
            },
            "rationale": rationale,
        }
    }
    tuned_cfg_path = run_dir / "tuned_config.yaml"
    tuned_cfg_path.write_text(yaml.safe_dump(tuned_cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")

    return wft_results_path, wft_summary_path, tuned_cfg_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--start", required=True, type=int, help="First test year (YYYY)")
    p.add_argument("--end", required=True, type=int, help="Last test year (YYYY)")
    p.add_argument("--rebalance", default="monthly", choices=["monthly"])
    p.add_argument("--test-window-years", type=int, default=1)
    p.add_argument("--train-window-years", type=int, default=12)
    p.add_argument("--step-years", type=int, default=1)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--master-path", default=None)
    p.add_argument("--prices-path", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.start) > int(args.end):
        raise ValueError("--start must be <= --end")
    if int(args.test_window_years) != 1:
        raise ValueError("--test-window-years must be 1 for annual test folds")
    if str(args.rebalance).lower() != "monthly":
        raise ValueError("--rebalance must be 'monthly'")

    root = project_root_from_file()
    cfg_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(cfg_path)
    paths = resolve_paths(cfg, root)

    run_dir = Path(args.run_dir) if args.run_dir else (paths["runs_dir"] / f"wft_{int(args.start)}_{int(args.end)}_{time.strftime('%H%M%S')}")
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()

    mos_grid = [0.30, 0.35, 0.40, 0.45]
    mad_grid = [-0.05, -0.02, 0.00, 0.02]
    weakness_variants = ["baseline", "stricter"]

    master_path = Path(args.master_path).resolve() if args.master_path else None
    prices_path = Path(args.prices_path).resolve() if args.prices_path else None

    wft_results_path, wft_summary_path, tuned_cfg_path = run_wft(
        project_root=root,
        config_path=cfg_path,
        run_dir=run_dir,
        train_years=int(args.train_window_years),
        test_years=int(args.test_window_years),
        step_years=int(args.step_years),
        seed=int(args.seed),
        mos_grid=mos_grid,
        mad_grid=mad_grid,
        weakness_variants=weakness_variants,
        start_year=int(args.start),
        end_year=int(args.end),
        master_path=master_path,
        prices_path=prices_path,
    )

    print(f"OK: {wft_results_path}")
    print(f"OK: {wft_summary_path}")
    print(f"OK: {tuned_cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
