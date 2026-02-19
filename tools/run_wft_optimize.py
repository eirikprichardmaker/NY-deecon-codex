from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import wft
from src.common.config import load_config, resolve_paths


COUNTRY_FROM_SUFFIX = {
    "OL": "NO",
    "ST": "SE",
    "CO": "DK",
    "HE": "FI",
}

INDEX_KEY_TO_COUNTRY = {
    "OSEAX": "NO",
    "OSEBX": "NO",
    "OBX": "NO",
    "OMXS": "SE",
    "OMXS30": "SE",
    "OMXC25": "DK",
    "HEX": "FI",
    "OMXH25": "FI",
}

COUNTRY_BENCHMARK_PRIORITY = {
    "NO": ["^OSEBX", "^OBX", "^OSEAX"],
    "SE": ["^OMXS30", "^OMXS"],
    "DK": ["^OMXC25"],
    "FI": ["^OMXH25", "^HEX"],
}


@dataclass(frozen=True)
class CostModel:
    commission_rate_bps: float
    commission_min: float
    commission_max: float
    order_notional: float
    fx_bps_non_nok: float


def _parse_float_grid(text: str) -> list[float]:
    out: list[float] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise ValueError("Grid cannot be empty.")
    return out


def _parse_str_grid(text: str) -> list[str]:
    out: list[str] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        out.append(token)
    if not out:
        raise ValueError("Grid cannot be empty.")
    return out


def _detect_supported_weakness_variants(project_root: Path) -> set[str]:
    src_path = project_root / "src" / "wft.py"
    txt = src_path.read_text(encoding="utf-8")
    out: set[str] = set()
    for variant in ("baseline", "strict", "stricter"):
        if f'"{variant}"' in txt or f"'{variant}'" in txt:
            out.add(variant)
    return out


def _resolve_weakness_grid(raw: str | None, project_root: Path) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    supported = _detect_supported_weakness_variants(project_root)

    if raw:
        requested = _parse_str_grid(raw)
        resolved = [v for v in requested if v in supported]
        dropped = [v for v in requested if v not in supported]
        if dropped:
            notes.append(f"Dropped unsupported weakness variants: {', '.join(dropped)}")
        if not resolved:
            resolved = ["baseline"] if "baseline" in supported else ["baseline"]
            notes.append("No requested weakness variant was supported; fallback to baseline.")
        return resolved, notes

    resolved: list[str] = []
    for cand in ("baseline", "stricter", "strict"):
        if cand in supported:
            resolved.append(cand)
    if not resolved:
        resolved = ["baseline"]
    return resolved, notes


def _resolve_path(base_root: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else (base_root / p).resolve()


def _norm_index_symbol(x: str) -> str:
    s = str(x).strip().upper()
    if not s:
        return ""
    s = s.lstrip("^")
    return f"^{s}"


def _country_from_ticker_symbol(x) -> str:
    suffix = wft._suffix_from_symbol(x)
    return COUNTRY_FROM_SUFFIX.get(suffix, "")


def _country_from_index_key(x) -> str:
    key = wft._norm_ticker(x)
    return INDEX_KEY_TO_COUNTRY.get(key, "")


def _attach_country_code(panel: pd.DataFrame) -> pd.DataFrame:
    d = panel.copy()
    stock_country = d["ticker"].map(_country_from_ticker_symbol) if "ticker" in d.columns else pd.Series("", index=d.index)
    index_country = d.get("relevant_index_key", pd.Series("", index=d.index)).map(_country_from_index_key)
    d["country_code"] = stock_country.where(stock_country.ne(""), index_country).fillna("")
    return d


def _filter_universe(panel: pd.DataFrame, universe: str) -> pd.DataFrame:
    scope = str(universe).strip().upper()
    if scope == "NO":
        return panel[panel["country_code"].eq("NO")].copy()
    if scope == "NORDIC":
        return panel[panel["country_code"].isin(["NO", "SE", "DK", "FI"])].copy()
    raise ValueError(f"Unsupported --universe: {universe}")


def _build_position_country_lookup(panel: pd.DataFrame) -> dict[str, str]:
    if panel.empty:
        return {}
    c = (
        panel[["k", "country_code"]]
        .dropna(subset=["k"])
        .astype({"k": str, "country_code": str})
        .copy()
    )
    c = c[c["country_code"].ne("")]
    if c.empty:
        return {}
    c = c.drop_duplicates(subset=["k"], keep="first")
    return dict(zip(c["k"], c["country_code"]))


def _build_index_monthly_returns(prices_path: Path) -> dict[str, pd.Series]:
    px = pd.read_parquet(prices_path).copy()
    px.columns = [str(c).strip().lower().replace(" ", "_") for c in px.columns]

    for c in ("ticker", "date", "adj_close"):
        if c not in px.columns:
            raise RuntimeError(f"prices.parquet mangler {c}")

    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px["adj_close"] = pd.to_numeric(px["adj_close"], errors="coerce")
    px = px.dropna(subset=["ticker", "date", "adj_close"]).copy()
    px["month"] = px["date"].dt.to_period("M").dt.to_timestamp("M")

    snap = (
        px.sort_values(["month", "ticker", "date"])
        .groupby(["month", "ticker"], as_index=False)
        .tail(1)
        .copy()
    )

    idx = snap[snap["ticker"].astype(str).str.startswith("^")].copy()
    if idx.empty:
        return {}
    idx["index_symbol"] = idx["ticker"].map(_norm_index_symbol)

    out: dict[str, pd.Series] = {}
    for sym, g in idx.groupby("index_symbol", as_index=False):
        d = g.sort_values("month")[["month", "adj_close"]].drop_duplicates(subset=["month"], keep="last").copy()
        d["ret"] = (d["adj_close"].shift(-1) / d["adj_close"]) - 1.0
        out[str(sym)] = pd.Series(d["ret"].values, index=pd.to_datetime(d["month"]), dtype=float)
    return out


def _select_country_benchmarks(index_returns: dict[str, pd.Series]) -> tuple[dict[str, str], list[str]]:
    available = set(index_returns.keys())
    selected: dict[str, str] = {}
    notes: list[str] = []
    for country, candidates in COUNTRY_BENCHMARK_PRIORITY.items():
        picked = ""
        for c in candidates:
            cc = _norm_index_symbol(c)
            if cc in available:
                picked = cc
                break
        selected[country] = picked
        if not picked:
            notes.append(f"No benchmark series available for {country}; benchmark_return defaults to 0 for those months.")
    return selected, notes


def _to_period_return(ret: pd.Series) -> float:
    r = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    if r.empty:
        return 0.0
    return float((1.0 + r).prod() - 1.0)


def _annual_cagr(period_returns: pd.Series, years_per_period: float) -> float:
    r = pd.to_numeric(period_returns, errors="coerce").dropna()
    n = len(r)
    if n == 0:
        return 0.0
    years = float(n) * float(years_per_period)
    if years <= 0:
        return 0.0
    nav = float((1.0 + r).prod())
    if nav <= 0:
        return -1.0
    return float(nav ** (1.0 / years) - 1.0)


def _per_trade_cost_fraction(country: str, cfg: CostModel) -> float:
    rate = float(cfg.commission_rate_bps) / 10000.0
    notional = float(cfg.order_notional)
    if notional <= 0:
        raise ValueError("--order-notional must be > 0")

    commission_amount = rate * notional
    commission_amount = min(max(commission_amount, float(cfg.commission_min)), float(cfg.commission_max))
    commission_frac = commission_amount / notional

    fx_frac = 0.0
    if str(country).upper() != "NO":
        fx_frac = float(cfg.fx_bps_non_nok) / 10000.0

    return float(commission_frac + fx_frac)


def _compute_net_returns(
    trades: pd.DataFrame,
    position_country: dict[str, str],
    cost_cfg: CostModel,
) -> tuple[pd.Series, int]:
    if trades.empty:
        return pd.Series(dtype=float), 0

    out: list[float] = []
    trade_count = 0
    prev_pos = "CASH"

    for _, row in trades.iterrows():
        cur_pos = str(row.get("position", "CASH"))
        gross_ret = float(pd.to_numeric(pd.Series([row.get("ret", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        cost_frac = 0.0

        if prev_pos == "CASH" and cur_pos != "CASH":
            cc = position_country.get(cur_pos, "NO")
            cost_frac += _per_trade_cost_fraction(cc, cost_cfg)
            trade_count += 1
        elif prev_pos != "CASH" and cur_pos == "CASH":
            cc = position_country.get(prev_pos, "NO")
            cost_frac += _per_trade_cost_fraction(cc, cost_cfg)
            trade_count += 1
        elif prev_pos != "CASH" and cur_pos != "CASH" and cur_pos != prev_pos:
            cc_sell = position_country.get(prev_pos, "NO")
            cc_buy = position_country.get(cur_pos, "NO")
            cost_frac += _per_trade_cost_fraction(cc_sell, cost_cfg)
            cost_frac += _per_trade_cost_fraction(cc_buy, cost_cfg)
            trade_count += 2

        net_ret = (1.0 + gross_ret) * (1.0 - cost_frac) - 1.0
        out.append(float(net_ret))
        prev_pos = cur_pos

    net_series = pd.Series(out, index=trades.index, dtype=float)
    return net_series, trade_count


def _compute_benchmark_returns(
    trades: pd.DataFrame,
    position_country: dict[str, str],
    country_benchmarks: dict[str, str],
    index_returns: dict[str, pd.Series],
) -> tuple[pd.Series, str, int]:
    if trades.empty:
        return pd.Series(dtype=float), "", 0

    out: list[float] = []
    used: set[str] = set()
    missing = 0

    for _, row in trades.iterrows():
        pos = str(row.get("position", "CASH"))
        if pos == "CASH":
            out.append(0.0)
            continue

        month = pd.to_datetime(row.get("month"), errors="coerce")
        if pd.isna(month):
            out.append(0.0)
            missing += 1
            continue

        country = position_country.get(pos, "")
        bench_sym = country_benchmarks.get(country, "")
        if not bench_sym:
            out.append(0.0)
            missing += 1
            continue

        bench_series = index_returns.get(bench_sym)
        if bench_series is None or month not in bench_series.index:
            out.append(0.0)
            missing += 1
            continue

        val = bench_series.loc[month]
        if pd.isna(val):
            out.append(0.0)
            missing += 1
            continue

        out.append(float(val))
        used.add(bench_sym)

    symbols_used = ",".join(sorted(used))
    return pd.Series(out, index=trades.index, dtype=float), symbols_used, int(missing)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    d = df.copy()
    for c in d.columns:
        if c == "rank":
            d[c] = d[c].map(lambda x: str(int(x)) if pd.notna(x) else "")
        elif pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
        else:
            d[c] = d[c].astype(str)
    cols = list(d.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for i in range(len(d)):
        lines.append("| " + " | ".join(d.iloc[i].tolist()) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested walk-forward optimizer for single-stock WFT strategy.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--start", required=True, type=int)
    p.add_argument("--end", required=True, type=int)
    p.add_argument("--train-window-years", required=True, type=int)
    p.add_argument("--test-window-years", required=True, type=int)
    p.add_argument("--rebalance", required=True, default="monthly")
    p.add_argument("--n-trials", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)

    p.add_argument("--master-path", default=None)
    p.add_argument("--prices-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--mos-grid", default="0.30,0.35,0.40,0.45")
    p.add_argument("--mad-grid", default="-0.05,-0.02,0.00,0.02")
    p.add_argument("--weakness-variants", default=None)
    p.add_argument("--search-mode", choices=["random", "grid"], default="random")
    p.add_argument("--universe", choices=["NO", "NORDIC"], default="NO")
    p.add_argument("--step-years", type=int, default=1)

    p.add_argument("--commission-rate-bps", type=float, default=4.0)
    p.add_argument("--commission-min", type=float, default=39.0)
    p.add_argument("--commission-max", type=float, default=149.0)
    p.add_argument("--order-notional", type=float, default=100000.0)
    p.add_argument("--fx-bps-non-nok", type=float, default=0.0)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.start) > int(args.end):
        raise ValueError("--start must be <= --end")
    if int(args.train_window_years) <= 0:
        raise ValueError("--train-window-years must be > 0")
    if int(args.test_window_years) <= 0:
        raise ValueError("--test-window-years must be > 0")
    if int(args.n_trials) <= 0:
        raise ValueError("--n-trials must be > 0")
    if str(args.rebalance).lower() != "monthly":
        raise ValueError("--rebalance must be monthly")

    cfg_path = _resolve_path(PROJECT_ROOT, args.config)
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = load_config(cfg_path)
    paths = resolve_paths(cfg, PROJECT_ROOT)

    output_dir = _resolve_path(PROJECT_ROOT, args.output_dir)
    run_dir = output_dir if output_dir else (paths["runs_dir"] / f"wft_opt_{time.strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    master_path = _resolve_path(PROJECT_ROOT, args.master_path)
    prices_path = _resolve_path(PROJECT_ROOT, args.prices_path)
    if master_path is None:
        master_path = paths["processed_dir"] / "master_valued.parquet"
    if prices_path is None:
        prices_path = paths["processed_dir"] / "prices.parquet"
    if not master_path.exists():
        raise FileNotFoundError(f"Missing master file: {master_path}")
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing prices file: {prices_path}")

    mos_grid = _parse_float_grid(args.mos_grid)
    mad_grid = _parse_float_grid(args.mad_grid)
    weakness_grid, weakness_notes = _resolve_weakness_grid(args.weakness_variants, PROJECT_ROOT)

    full_grid = wft._iter_param_grid(mos_grid, mad_grid, weakness_grid)
    if not full_grid:
        raise RuntimeError("No parameters available for optimization.")

    n_trials = min(int(args.n_trials), len(full_grid))
    if str(args.search_mode).lower() == "grid":
        selected_params = full_grid[:n_trials]
    else:
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(full_grid), size=n_trials, replace=False)
        selected_params = [full_grid[int(i)] for i in idx]

    trial_defs = [{"trial_id": f"trial_{i:04d}", "params": p} for i, p in enumerate(selected_params, start=1)]

    static = wft._load_static_universe(master_path)
    panel = wft._load_monthly_panel(prices_path, static)
    panel = _attach_country_code(panel)
    panel = _filter_universe(panel, args.universe)
    if panel.empty:
        raise RuntimeError(f"No rows left in panel after universe filter: {args.universe}")

    folds = wft._build_folds(
        panel["month"],
        train_years=int(args.train_window_years),
        test_years=int(args.test_window_years),
        step_years=int(args.step_years),
    )
    folds = [f for f in folds if int(f["test_year"]) >= int(args.start) and int(f["test_year"]) <= int(args.end)]
    if not folds:
        raise RuntimeError("No folds in requested --start/--end range.")

    index_returns = _build_index_monthly_returns(prices_path)
    country_benchmarks, benchmark_notes = _select_country_benchmarks(index_returns)
    position_country = _build_position_country_lookup(panel)

    cost_cfg = CostModel(
        commission_rate_bps=float(args.commission_rate_bps),
        commission_min=float(args.commission_min),
        commission_max=float(args.commission_max),
        order_notional=float(args.order_notional),
        fx_bps_non_nok=float(args.fx_bps_non_nok),
    )

    rows: list[dict] = []
    for fold_idx, fold in enumerate(folds, start=1):
        train_df = panel[(panel["month"] >= fold["train_start"]) & (panel["month"] <= fold["train_end"])].copy()
        test_df = panel[(panel["month"] >= fold["test_start"]) & (panel["month"] <= fold["test_end"])].copy()

        fold_rows: list[dict] = []
        for t in trial_defs:
            trial_id = str(t["trial_id"])
            params = t["params"]

            train_trades = wft._simulate_window(train_df, params)
            train_metrics = wft._window_metrics(train_trades)
            train_objective = wft._objective(train_metrics)

            test_trades = wft._simulate_window(test_df, params)
            test_metrics = wft._window_metrics(test_trades)

            test_net_ret, trade_count = _compute_net_returns(test_trades, position_country, cost_cfg)
            benchmark_ret, symbols_used, benchmark_missing_months = _compute_benchmark_returns(
                test_trades,
                position_country=position_country,
                country_benchmarks=country_benchmarks,
                index_returns=index_returns,
            )

            test_return_gross = float(test_metrics["return"])
            test_return_net = _to_period_return(test_net_ret)
            benchmark_return = _to_period_return(benchmark_ret)
            excess_return_net = float(test_return_net - benchmark_return)
            test_max_dd = wft._max_drawdown(test_net_ret)

            fold_rows.append(
                {
                    "trial_id": trial_id,
                    "fold_id": int(fold_idx),
                    "test_year": int(fold["test_year"]),
                    "is_selected_train": False,
                    "mos_threshold": float(params.mos_threshold),
                    "mad_min": float(params.mad_min),
                    "weakness_rule_variant": str(params.weakness_rule_variant),
                    "train_objective": float(train_objective),
                    "train_return": float(train_metrics["return"]),
                    "train_sharpe": float(train_metrics["sharpe_ann"]),
                    "train_turnover": float(train_metrics["turnover"]),
                    "train_pct_cash": float(train_metrics["pct_cash"]),
                    "test_return_gross": float(test_return_gross),
                    "test_return_net": float(test_return_net),
                    "benchmark_return": float(benchmark_return),
                    "excess_return_net": float(excess_return_net),
                    "test_max_dd": float(test_max_dd),
                    "test_turnover": float(test_metrics["turnover"]),
                    "test_pct_cash": float(test_metrics["pct_cash"]),
                    "candidate_or_cash": "CASH" if float(test_metrics["pct_cash"]) >= 0.999 else "CANDIDATE",
                    "benchmark_symbols_used": symbols_used,
                    "benchmark_missing_months": int(benchmark_missing_months),
                    "trade_count": int(trade_count),
                }
            )

        fold_df = pd.DataFrame(fold_rows)
        fold_df = fold_df.sort_values(
            by=[
                "train_objective",
                "train_sharpe",
                "train_return",
                "train_turnover",
                "train_pct_cash",
                "mos_threshold",
                "mad_min",
                "weakness_rule_variant",
            ],
            ascending=[False, False, False, True, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        if not fold_df.empty:
            winner = str(fold_df.iloc[0]["trial_id"])
            fold_df["is_selected_train"] = fold_df["trial_id"].eq(winner)

        rows.extend(fold_df.to_dict(orient="records"))

    trials_df = pd.DataFrame(rows)
    trials_df = trials_df.sort_values(by=["fold_id", "trial_id"], kind="mergesort").reset_index(drop=True)
    trials_csv = run_dir / "trials.csv"
    trials_df.to_csv(trials_csv, index=False)

    agg = (
        trials_df.groupby(["trial_id", "mos_threshold", "mad_min", "weakness_rule_variant"], as_index=False)
        .agg(
            folds=("fold_id", "count"),
            selected_folds=("is_selected_train", "sum"),
            mean_train_objective=("train_objective", "mean"),
            mean_test_return_gross=("test_return_gross", "mean"),
            mean_test_return_net=("test_return_net", "mean"),
            median_excess_return_net=("excess_return_net", "median"),
            worst_test_max_dd=("test_max_dd", "min"),
            mean_test_turnover=("test_turnover", "mean"),
            mean_test_pct_cash=("test_pct_cash", "mean"),
            mean_benchmark_return=("benchmark_return", "mean"),
            total_trade_count=("trade_count", "sum"),
            total_benchmark_missing_months=("benchmark_missing_months", "sum"),
        )
        .copy()
    )

    cagr_rows: list[dict] = []
    for trial_id, g in trials_df.groupby("trial_id", as_index=False):
        cagr_rows.append(
            {
                "trial_id": str(trial_id),
                "oos_cagr_gross": _annual_cagr(g["test_return_gross"], years_per_period=float(args.test_window_years)),
                "oos_cagr_net": _annual_cagr(g["test_return_net"], years_per_period=float(args.test_window_years)),
                "benchmark_cagr": _annual_cagr(g["benchmark_return"], years_per_period=float(args.test_window_years)),
            }
        )

    agg = agg.merge(pd.DataFrame(cagr_rows), on="trial_id", how="left")
    agg = agg.sort_values(
        by=[
            "selected_folds",
            "median_excess_return_net",
            "worst_test_max_dd",
            "mean_test_turnover",
            "mos_threshold",
            "mad_min",
            "weakness_rule_variant",
        ],
        ascending=[False, False, False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)

    best = agg.iloc[0]
    best_params = {
        "mos_threshold": float(best["mos_threshold"]),
        "mad_min": float(best["mad_min"]),
        "weakness_rule_variant": str(best["weakness_rule_variant"]),
    }

    rationale = (
        f"Selected by nested train winners across folds ({int(best['selected_folds'])}/{int(best['folds'])}) "
        f"with tie-break on median excess net return ({float(best['median_excess_return_net']):.4f}), "
        f"drawdown ({float(best['worst_test_max_dd']):.4f}), and turnover ({float(best['mean_test_turnover']):.4f})."
    )

    best_config = {
        "optimizer": {
            "method": "nested_walk_forward",
            "search_mode": str(args.search_mode),
            "seed": int(args.seed),
            "n_trials": int(n_trials),
            "start": int(args.start),
            "end": int(args.end),
            "train_window_years": int(args.train_window_years),
            "test_window_years": int(args.test_window_years),
            "step_years": int(args.step_years),
            "rebalance": str(args.rebalance),
            "universe": str(args.universe),
        },
        "selected_params": best_params,
        "selection_metrics": {
            "trial_id": str(best["trial_id"]),
            "selected_folds": int(best["selected_folds"]),
            "folds": int(best["folds"]),
            "median_excess_return_net": float(best["median_excess_return_net"]),
            "worst_test_max_dd": float(best["worst_test_max_dd"]),
            "mean_test_turnover": float(best["mean_test_turnover"]),
            "mean_test_return_net": float(best["mean_test_return_net"]),
            "oos_cagr_net": float(best["oos_cagr_net"]),
            "benchmark_cagr": float(best["benchmark_cagr"]),
        },
        "benchmark_policy": {
            "priority_map": COUNTRY_BENCHMARK_PRIORITY,
            "selected_symbols": country_benchmarks,
            "notes": benchmark_notes,
        },
        "cost_model": {
            "commission_rate_bps": float(cost_cfg.commission_rate_bps),
            "commission_min": float(cost_cfg.commission_min),
            "commission_max": float(cost_cfg.commission_max),
            "order_notional": float(cost_cfg.order_notional),
            "fx_bps_non_nok": float(cost_cfg.fx_bps_non_nok),
        },
        "rationale": rationale,
    }
    best_yaml = run_dir / "best_config.yaml"
    best_yaml.write_text(yaml.safe_dump(best_config, sort_keys=False, allow_unicode=False), encoding="utf-8")

    top10 = agg.head(10).copy()
    top10 = top10[
        [
            "rank",
            "trial_id",
            "mos_threshold",
            "mad_min",
            "weakness_rule_variant",
            "selected_folds",
            "median_excess_return_net",
            "worst_test_max_dd",
            "mean_test_turnover",
            "mean_test_return_net",
            "oos_cagr_net",
            "mean_benchmark_return",
        ]
    ]

    selected_rows = trials_df[trials_df["is_selected_train"]].copy()
    selected_missing = int(selected_rows["benchmark_missing_months"].sum()) if not selected_rows.empty else 0
    selected_trade_count = int(selected_rows["trade_count"].sum()) if not selected_rows.empty else 0
    sfreq = agg["selected_folds"].value_counts().sort_index()

    summary_lines = [
        "# WFT Optimize Summary",
        "",
        f"- Output dir: `{run_dir}`",
        f"- Config: `{cfg_path}`",
        f"- Master path: `{master_path}`",
        f"- Prices path: `{prices_path}`",
        f"- Period: {int(args.start)}-{int(args.end)}",
        f"- Train/Test windows: {int(args.train_window_years)}y / {int(args.test_window_years)}y",
        f"- Rebalance: {args.rebalance}",
        f"- Universe: {args.universe}",
        f"- Search mode: {args.search_mode}",
        f"- Trials requested/used: {int(args.n_trials)} / {int(n_trials)}",
        f"- Folds: {len(folds)}",
        "",
        "## Benchmark Selection",
    ]
    for c in ("NO", "SE", "DK", "FI"):
        sym = country_benchmarks.get(c, "")
        summary_lines.append(f"- {c}: `{sym if sym else 'N/A'}`")
    summary_lines.append("")

    if weakness_notes or benchmark_notes:
        summary_lines.append("## Notes")
        for note in weakness_notes + benchmark_notes:
            summary_lines.append(f"- {note}")
        summary_lines.append("")

    summary_lines.extend(
        [
            "## Ranking Logic",
            "1. Highest selected_folds (nested inner-loop winners).",
            "2. Highest median_excess_return_net.",
            "3. Best worst_test_max_dd (least negative).",
            "4. Lowest mean_test_turnover.",
            "",
            "## Top 10 Trials",
            _md_table(top10),
            "",
            "## Selected Best Config",
            f"- Trial: `{best['trial_id']}`",
            f"- Params: `{json.dumps(best_params, sort_keys=True)}`",
            f"- Selected folds: {int(best['selected_folds'])}/{int(best['folds'])}",
            (
                f"- Key metrics: median_excess_net={float(best['median_excess_return_net']):.2%}, "
                f"worst_maxdd={float(best['worst_test_max_dd']):.2%}, "
                f"turnover={float(best['mean_test_turnover']):.2%}, "
                f"oos_cagr_net={float(best['oos_cagr_net']):.2%}"
            ),
            f"- Rationale: {rationale}",
            "",
            "## Robustness Stats",
            f"- selected train folds total rows: {len(selected_rows)}",
            f"- selected rows benchmark missing months: {selected_missing}",
            f"- selected rows total trade count: {selected_trade_count}",
            "- selected_folds distribution:",
        ]
    )
    for k, v in sfreq.items():
        summary_lines.append(f"  - {int(k)} folds selected: {int(v)} trial(s)")

    summary_md = run_dir / "optimize_summary.md"
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"OK: {trials_csv}")
    print(f"OK: {best_yaml}")
    print(f"OK: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
