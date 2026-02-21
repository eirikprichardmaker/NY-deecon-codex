from __future__ import annotations

import argparse
import itertools
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

WEIGHT_KEYS = ["quality", "value", "lowrisk", "balance"]
DEFAULT_WEIGHTS_PRIOR = {
    "quality": 0.4,
    "value": 0.3,
    "lowrisk": 0.2,
    "balance": 0.1,
}
DEFAULT_WEIGHTS_BOUNDS = (0.0, 0.7)
DEFAULT_WEIGHTS_REG_LAMBDA = 0.5
WFT_RESULTS_COLS = [
    "mode",
    "train_start",
    "train_end",
    "test_year",
    "chosen_params",
    "candidate_or_cash",
    "return",
    "max_dd",
    "turnover",
    "pct_cash",
]


@dataclass(frozen=True)
class CostModel:
    commission_rate_bps: float
    commission_min: float
    commission_max: float
    order_notional: float
    fx_bps_non_nok: float


@dataclass(frozen=True)
class StrategyAParams:
    mos_threshold: float
    mad_min: float
    mad_penalty_k: float
    min_hold_months: int
    score_gap: float
    weight_quality: float
    weight_value: float
    weight_lowrisk: float
    weight_balance: float
    weakness_rule_variant: str = "baseline"

    def weights_tuple(self) -> tuple[float, float, float, float]:
        return (
            float(self.weight_quality),
            float(self.weight_value),
            float(self.weight_lowrisk),
            float(self.weight_balance),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "mos_threshold": float(self.mos_threshold),
                "mad_min": float(self.mad_min),
                "mad_penalty_k": float(self.mad_penalty_k),
                "min_hold_months": int(self.min_hold_months),
                "score_gap": float(self.score_gap),
                "weights": {
                    "quality": float(self.weight_quality),
                    "value": float(self.weight_value),
                    "lowrisk": float(self.weight_lowrisk),
                    "balance": float(self.weight_balance),
                },
                "weakness_rule_variant": str(self.weakness_rule_variant),
            },
            sort_keys=True,
        )


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_z(s: pd.Series) -> pd.Series:
    x = _to_num(s)
    if int(x.notna().sum()) < 2:
        return pd.Series(0.0, index=s.index, dtype=float)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index, dtype=float)
    return ((x - mu) / sd).fillna(0.0)


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


def _parse_int_grid(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("Grid cannot be empty.")
    return out


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
    c = panel[["k", "country_code"]].dropna(subset=["k"]).copy()
    c["k"] = c["k"].astype(str)
    c["country_code"] = c["country_code"].astype(str)
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
    px["adj_close"] = _to_num(px["adj_close"])
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
    r = _to_num(ret).fillna(0.0)
    if r.empty:
        return 0.0
    return float((1.0 + r).prod() - 1.0)


def _annual_cagr(period_returns: pd.Series, years_per_period: float) -> float:
    r = _to_num(period_returns).dropna()
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
        gross_ret = float(_to_num(pd.Series([row.get("ret", 0.0)])).fillna(0.0).iloc[0])
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

        out.append(float((1.0 + gross_ret) * (1.0 - cost_frac) - 1.0))
        prev_pos = cur_pos

    return pd.Series(out, index=trades.index, dtype=float), int(trade_count)


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

    return pd.Series(out, index=trades.index, dtype=float), ",".join(sorted(used)), int(missing)


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


def _normalize_prior_weights(d: dict) -> dict[str, float]:
    out = dict(DEFAULT_WEIGHTS_PRIOR)
    if isinstance(d, dict):
        for k in WEIGHT_KEYS:
            if k in d:
                out[k] = float(d[k])
    for k in WEIGHT_KEYS:
        out[k] = max(0.0, float(out[k]))
    total = sum(out.values())
    if total <= 0:
        out = dict(DEFAULT_WEIGHTS_PRIOR)
        total = sum(out.values())
    return {k: float(out[k] / total) for k in WEIGHT_KEYS}


def _parse_weight_bounds(v) -> tuple[float, float]:
    lo, hi = DEFAULT_WEIGHTS_BOUNDS
    if isinstance(v, (list, tuple)) and len(v) == 2:
        lo = float(v[0])
        hi = float(v[1])
    lo = max(0.0, lo)
    hi = min(1.0, hi)
    if lo > hi:
        raise ValueError("Invalid weights_bounds: lower must be <= upper")
    if (4.0 * lo) > 1.0 or (4.0 * hi) < 1.0:
        raise ValueError("Invalid weights_bounds: simplex sum=1 not feasible for 4 weights")
    return float(lo), float(hi)


def _weights_reg_penalty(weights: tuple[float, float, float, float], prior: dict[str, float], reg_lambda: float) -> float:
    p = np.array([float(prior[k]) for k in WEIGHT_KEYS], dtype=float)
    w = np.array(weights, dtype=float)
    return float(reg_lambda * np.sum((w - p) ** 2))


def _weight_grid(prior: dict[str, float], bounds: tuple[float, float], step: float) -> list[tuple[float, float, float, float]]:
    lo, hi = bounds
    if step <= 0:
        raise ValueError("--weights-grid-step must be > 0")
    vals = np.arange(lo, hi + (step / 2.0), step)
    vals = [float(round(v, 6)) for v in vals]

    out: set[tuple[float, float, float, float]] = set()
    for q in vals:
        for v in vals:
            for lr in vals:
                b = 1.0 - q - v - lr
                b_r = round(b, 6)
                if b_r < (lo - 1e-9) or b_r > (hi + 1e-9):
                    continue
                w = (float(round(q, 6)), float(round(v, 6)), float(round(lr, 6)), float(b_r))
                if any(x < (lo - 1e-9) or x > (hi + 1e-9) for x in w):
                    continue
                if abs(sum(w) - 1.0) > 1e-6:
                    continue
                out.add(tuple(float(round(x, 6)) for x in w))

    prior_tuple = tuple(float(round(prior[k], 6)) for k in WEIGHT_KEYS)
    out_list = list(out)
    if prior_tuple not in out_list:
        out_list.append(prior_tuple)

    p = np.array(prior_tuple, dtype=float)
    out_list = sorted(
        out_list,
        key=lambda w: (float(np.sum((np.array(w, dtype=float) - p) ** 2)), w),
    )
    return out_list


def _sample_weight_random(
    rng: np.random.Generator,
    prior: dict[str, float],
    bounds: tuple[float, float],
) -> tuple[float, float, float, float]:
    lo, hi = bounds
    alpha = np.array([max(float(prior[k]) * 25.0, 0.25) for k in WEIGHT_KEYS], dtype=float)

    for _ in range(5000):
        w = rng.dirichlet(alpha)
        if np.all(w >= (lo - 1e-10)) and np.all(w <= (hi + 1e-10)):
            w = w / w.sum()
            return tuple(float(round(x, 6)) for x in w.tolist())

    fallback = np.array([float(prior[k]) for k in WEIGHT_KEYS], dtype=float)
    fallback = np.clip(fallback, lo, hi)
    fallback = fallback / fallback.sum()
    return tuple(float(round(x, 6)) for x in fallback.tolist())


def _build_trials(
    args: argparse.Namespace,
    prior: dict[str, float],
    bounds: tuple[float, float],
) -> tuple[list[StrategyAParams], list[str]]:
    notes: list[str] = []
    mos_grid = _parse_float_grid(args.mos_grid)
    mad_grid = _parse_float_grid(args.mad_grid)
    k_grid = _parse_float_grid(args.mad_penalty_k_grid)
    min_hold_grid = _parse_int_grid(args.min_hold_grid)
    score_gap_grid = _parse_float_grid(args.score_gap_grid)

    base_combos = list(itertools.product(mos_grid, mad_grid, k_grid, min_hold_grid, score_gap_grid))
    if not base_combos:
        raise RuntimeError("Empty parameter grid.")

    trials: list[StrategyAParams] = []
    n_trials = int(args.n_trials)
    if n_trials <= 0:
        raise ValueError("--n-trials must be > 0")

    if str(args.search_mode).lower() == "grid":
        weight_candidates = _weight_grid(prior=prior, bounds=bounds, step=float(args.weights_grid_step))
        if not weight_candidates:
            raise RuntimeError("Could not create weight grid.")
        for w in weight_candidates:
            for mos, mad, k, mh, sg in base_combos:
                trials.append(
                    StrategyAParams(
                        mos_threshold=float(mos),
                        mad_min=float(mad),
                        mad_penalty_k=float(k),
                        min_hold_months=int(mh),
                        score_gap=float(sg),
                        weight_quality=float(w[0]),
                        weight_value=float(w[1]),
                        weight_lowrisk=float(w[2]),
                        weight_balance=float(w[3]),
                        weakness_rule_variant=str(args.weakness_rule_variant),
                    )
                )
                if len(trials) >= n_trials:
                    break
            if len(trials) >= n_trials:
                break
    else:
        rng = np.random.default_rng(int(args.seed))
        seen: set[tuple] = set()
        attempts = 0
        while len(trials) < n_trials and attempts < (n_trials * 500):
            attempts += 1
            mos, mad, k, mh, sg = base_combos[int(rng.integers(0, len(base_combos)))]
            w = _sample_weight_random(rng=rng, prior=prior, bounds=bounds)
            key = (
                float(round(mos, 6)),
                float(round(mad, 6)),
                float(round(k, 6)),
                int(mh),
                float(round(sg, 6)),
                tuple(float(round(x, 6)) for x in w),
                str(args.weakness_rule_variant),
            )
            if key in seen:
                continue
            seen.add(key)
            trials.append(
                StrategyAParams(
                    mos_threshold=float(mos),
                    mad_min=float(mad),
                    mad_penalty_k=float(k),
                    min_hold_months=int(mh),
                    score_gap=float(sg),
                    weight_quality=float(w[0]),
                    weight_value=float(w[1]),
                    weight_lowrisk=float(w[2]),
                    weight_balance=float(w[3]),
                    weakness_rule_variant=str(args.weakness_rule_variant),
                )
            )
        if len(trials) < n_trials:
            notes.append(f"Requested {n_trials} trials, generated {len(trials)} unique trials.")

    if not trials:
        raise RuntimeError("No trials generated.")
    return trials, notes


def _augment_static_universe(static: pd.DataFrame, master_path: Path) -> pd.DataFrame:
    m = pd.read_parquet(master_path).copy()
    m.columns = [str(c).strip().lower().replace(" ", "_") for c in m.columns]
    if "ticker" not in m.columns:
        raise RuntimeError("master_valued mangler ticker")
    if "yahoo_ticker" not in m.columns:
        m["yahoo_ticker"] = m["ticker"]

    m["k"] = m["yahoo_ticker"].map(wft._norm_ticker)
    m = m[m["k"].ne("")].copy()

    beta = _to_num(m.get("beta", pd.Series(np.nan, index=m.index)))
    nd_ebitda = _to_num(m.get("nd_ebitda", m.get("n_debt_ebitda_current", pd.Series(np.nan, index=m.index))))
    extra = pd.DataFrame(
        {
            "k": m["k"],
            "beta_extra": beta,
            "nd_ebitda_extra": nd_ebitda,
        }
    )
    extra = extra.sort_values(["k"]).drop_duplicates(subset=["k"], keep="first")

    out = static.merge(extra, on="k", how="left")
    out["beta"] = _to_num(out.get("beta", out.get("beta_extra", pd.Series(np.nan, index=out.index))))
    out["nd_ebitda"] = _to_num(out.get("nd_ebitda", out.get("nd_ebitda_extra", pd.Series(np.nan, index=out.index))))
    out["beta"] = out["beta"].fillna(1.0)
    out["nd_ebitda"] = out["nd_ebitda"].fillna(2.0)

    out["market_cap"] = _to_num(out.get("market_cap", pd.Series(np.nan, index=out.index)))
    if out["market_cap"].notna().any():
        med = float(out["market_cap"].median(skipna=True))
        out["market_cap"] = out["market_cap"].fillna(med if np.isfinite(med) else 1.0)
    else:
        out["market_cap"] = 1.0

    return out


def _compute_group_scores(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()

    roic_z = _safe_z(out.get("roic", pd.Series(np.nan, index=out.index)))
    fcf_z = _safe_z(out.get("fcf_yield", pd.Series(np.nan, index=out.index)))
    out["score_quality"] = (0.6 * roic_z + 0.4 * fcf_z).fillna(0.0)

    out["score_value"] = _safe_z(out.get("mos", pd.Series(np.nan, index=out.index))).fillna(0.0)

    beta = _to_num(out.get("beta", pd.Series(np.nan, index=out.index)))
    nd = _to_num(out.get("nd_ebitda", pd.Series(np.nan, index=out.index)))
    out["score_lowrisk"] = (0.5 * _safe_z(-beta) + 0.5 * _safe_z(-nd)).fillna(0.0)

    mcap = _to_num(out.get("market_cap", pd.Series(np.nan, index=out.index))).clip(lower=0.0)
    out["score_balance"] = _safe_z(np.log1p(mcap)).fillna(0.0)
    return out


def _score_month(month_df: pd.DataFrame, params: StrategyAParams) -> pd.DataFrame:
    d = month_df.copy()

    base_mos = float(params.mos_threshold)
    mos_req = np.where(d["high_risk_flag"].fillna(False), np.maximum(0.40, base_mos), base_mos)
    weak_fail_min = 2 if str(params.weakness_rule_variant) == "baseline" else 1

    d["fundamental_ok"] = (
        d["mos"].notna()
        & (d["mos"] >= mos_req)
        & d["value_creation_ok_base"].fillna(False).astype(bool)
        & (_to_num(d["quality_weak_count"]).lt(float(weak_fail_min)).fillna(False))
    )

    stock_price = _to_num(d.get("adj_close", pd.Series(np.nan, index=d.index)))
    stock_ma200 = _to_num(d.get("ma200", pd.Series(np.nan, index=d.index)))
    stock_mad = _to_num(d.get("mad", pd.Series(np.nan, index=d.index)))
    index_price = _to_num(d.get("index_price", pd.Series(np.nan, index=d.index)))
    index_ma200 = _to_num(d.get("index_ma200", pd.Series(np.nan, index=d.index)))
    index_mad = _to_num(d.get("index_mad", pd.Series(np.nan, index=d.index)))

    d["stock_data_ok"] = stock_price.notna() & stock_ma200.notna()
    d["index_data_ok"] = index_price.notna() & index_ma200.notna()
    d["stock_above_ma200"] = stock_price > stock_ma200
    d["index_above_ma200"] = index_price > index_ma200
    d["stock_mad_ok"] = stock_mad.notna() & stock_mad.ge(float(params.mad_min))
    d["tech_signal_count"] = (
        (d["stock_data_ok"].astype(bool) & d["stock_above_ma200"].astype(bool)).astype(int) +
        (d["index_data_ok"].astype(bool) & d["index_above_ma200"].astype(bool)).astype(int) +
        d["stock_mad_ok"].astype(int)
    )

    d["technical_ok"] = (
        d["stock_data_ok"].astype(bool)
        & d["index_data_ok"].astype(bool)
        & d["index_above_ma200"].astype(bool)
        & d["tech_signal_count"].ge(2)
    )
    d["eligible"] = d["fundamental_ok"] & d["technical_ok"]

    d = _compute_group_scores(d)
    d["score_raw"] = (
        float(params.weight_quality) * d["score_quality"]
        + float(params.weight_value) * d["score_value"]
        + float(params.weight_lowrisk) * d["score_lowrisk"]
        + float(params.weight_balance) * d["score_balance"]
    )

    stock_mad_penalty = stock_mad.fillna(float(params.mad_min) - 0.10)
    index_mad_penalty = index_mad.fillna(float(params.mad_min) - 0.10)
    deficit_stock = np.maximum(0.0, float(params.mad_min) - stock_mad_penalty)
    deficit_index = np.maximum(0.0, float(params.mad_min) - index_mad_penalty)
    d["mad_penalty"] = float(params.mad_penalty_k) * (deficit_stock + deficit_index)
    d["selection_score"] = d["score_raw"] - d["mad_penalty"]

    return d


def _top_candidate(scored: pd.DataFrame) -> tuple[str, float]:
    elig = scored[scored["eligible"]].copy()
    if elig.empty:
        return "CASH", 0.0
    elig = elig.sort_values(
        by=["selection_score", "mos", "market_cap", "ticker"],
        ascending=[False, False, False, True],
        na_position="last",
        kind="mergesort",
    )
    r = elig.iloc[0]
    return str(r["k"]), float(r["selection_score"])


def _position_score(scored: pd.DataFrame, pos: str) -> float:
    if pos == "CASH":
        return 0.0
    row = scored[scored["k"] == pos]
    if row.empty:
        return -1e9
    x = float(_to_num(pd.Series([row.iloc[0]["selection_score"]])).fillna(-1e9).iloc[0])
    return x if np.isfinite(x) else -1e9


def _position_is_eligible(scored: pd.DataFrame, pos: str) -> bool:
    if pos == "CASH":
        return True
    row = scored[scored["k"] == pos]
    if row.empty:
        return False
    return bool(row.iloc[0]["eligible"])


def _simulate_window_strategy_a(window_df: pd.DataFrame, params: StrategyAParams) -> pd.DataFrame:
    months = sorted(window_df["month"].dropna().unique().tolist())
    if len(months) < 2:
        return pd.DataFrame(columns=["month", "position", "ret"])

    rows: list[dict] = []
    current_pos = "CASH"
    hold_months = 0

    for i in range(len(months) - 1):
        m = months[i]
        n = months[i + 1]
        cur = window_df[window_df["month"] == m]
        nxt = window_df[window_df["month"] == n]
        scored = _score_month(cur, params)

        challenger_pos, challenger_score = _top_candidate(scored)
        current_score = 0.0 if current_pos == "CASH" else _position_score(scored, current_pos)
        current_eligible = True if current_pos == "CASH" else _position_is_eligible(scored, current_pos)

        target_pos = current_pos
        if current_pos != "CASH" and (not current_eligible):
            target_pos = "CASH"
        elif current_pos == "CASH":
            if challenger_pos != "CASH" and challenger_score >= (0.0 + float(params.score_gap)):
                target_pos = challenger_pos
            else:
                target_pos = "CASH"
        else:
            if challenger_pos == current_pos:
                target_pos = current_pos
            elif int(hold_months) < int(params.min_hold_months):
                target_pos = current_pos
            elif challenger_score >= (current_score + float(params.score_gap)):
                target_pos = challenger_pos
            else:
                target_pos = current_pos

        final_pos = target_pos
        ret = 0.0
        if final_pos != "CASH":
            c = cur[cur["k"] == final_pos]
            x = nxt[nxt["k"] == final_pos]
            if c.empty or x.empty:
                ret = 0.0
                final_pos = "CASH"
            else:
                p0 = float(c.iloc[0]["adj_close"])
                p1 = float(x.iloc[0]["adj_close"])
                ret = float((p1 / p0) - 1.0) if np.isfinite(p0) and p0 > 0 and np.isfinite(p1) else 0.0

        rows.append(
            {
                "month": pd.Timestamp(m),
                "position": final_pos,
                "ret": float(ret),
                "hold_months_before": int(hold_months),
                "challenger": challenger_pos,
                "challenger_score": float(challenger_score),
                "current_score": float(current_score),
            }
        )

        if final_pos == "CASH":
            hold_months = hold_months + 1 if current_pos == "CASH" else 1
        elif final_pos == current_pos:
            hold_months = hold_months + 1
        else:
            hold_months = 1
        current_pos = final_pos

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    prev = out["position"].shift(1)
    out["position_change"] = (out["position"] != prev).astype(int)
    out.loc[out.index[0], "position_change"] = 0
    out["is_cash"] = out["position"].eq("CASH")
    return out


def _train_objective(metrics: dict, reg_penalty: float) -> float:
    return float(
        (0.35 * float(metrics["sharpe_ann"]))
        + (0.35 * float(metrics["return"]))
        + (0.15 * float(metrics["max_dd"]))
        - (0.10 * float(metrics["turnover"]))
        - (0.05 * float(metrics["pct_cash"]))
        - float(reg_penalty)
    )


def _robust_utility(
    median_excess_return_net: float,
    worst_fold_excess_return_net: float,
    median_test_max_dd: float,
    mean_test_turnover: float,
    mean_test_pct_cash: float,
) -> float:
    return float(
        float(median_excess_return_net)
        + 0.50 * float(worst_fold_excess_return_net)
        + 0.25 * float(median_test_max_dd)
        - 0.10 * float(mean_test_turnover)
        - 0.05 * float(mean_test_pct_cash)
    )


def _load_strategy_a_cfg(cfg: dict, args: argparse.Namespace) -> tuple[dict[str, float], tuple[float, float], float]:
    sec = cfg.get("wft_opt_strategy_a", {}) if isinstance(cfg, dict) else {}
    if not isinstance(sec, dict):
        sec = {}

    prior = _normalize_prior_weights(sec.get("weights_prior", DEFAULT_WEIGHTS_PRIOR))
    bounds = _parse_weight_bounds(sec.get("weights_bounds", list(DEFAULT_WEIGHTS_BOUNDS)))
    reg_lambda = float(sec.get("weights_reg_lambda", DEFAULT_WEIGHTS_REG_LAMBDA))
    if args.weights_reg_lambda is not None:
        reg_lambda = float(args.weights_reg_lambda)
    if reg_lambda < 0:
        raise ValueError("weights_reg_lambda must be >= 0")
    return prior, bounds, reg_lambda


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested walk-forward optimizer (Strategy A) for single-stock WFT strategy.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--start", required=True, type=int)
    p.add_argument("--end", required=True, type=int)
    p.add_argument("--train-window-years", type=int, default=8)
    p.add_argument("--test-window-years", required=True, type=int)
    p.add_argument("--rebalance", required=True, default="monthly")
    p.add_argument("--n-trials", required=True, type=int)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--holdout-start", type=int, default=2023)
    p.add_argument("--holdout-end", type=int, default=2025)

    p.add_argument("--master-path", default=None)
    p.add_argument("--prices-path", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--search-mode", choices=["random", "grid"], default="random")
    p.add_argument("--universe", choices=["NO", "NORDIC"], default="NO")
    p.add_argument("--step-years", type=int, default=1)

    p.add_argument("--mos-grid", default="0.30,0.35,0.40")
    p.add_argument("--mad-grid", default="-0.02,-0.05,-0.08")
    p.add_argument("--mad-penalty-k-grid", default="0.0,0.5,1.0")
    p.add_argument("--min-hold-grid", default="3,6")
    p.add_argument("--score-gap-grid", default="0.25,0.5,1.0")
    p.add_argument("--weights-grid-step", type=float, default=0.1)
    p.add_argument("--weights-reg-lambda", type=float, default=None)
    p.add_argument("--weakness-rule-variant", default="baseline")

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
    if int(args.holdout_start) > int(args.holdout_end):
        raise ValueError("--holdout-start must be <= --holdout-end")

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

    weights_prior, weights_bounds, weights_reg_lambda = _load_strategy_a_cfg(cfg, args)
    trials, build_notes = _build_trials(args=args, prior=weights_prior, bounds=weights_bounds)
    n_trials = len(trials)

    static = wft._load_static_universe(master_path)
    static = _augment_static_universe(static, master_path=master_path)
    panel = wft._load_monthly_panel(prices_path, static)
    panel = _attach_country_code(panel)
    panel = _filter_universe(panel, args.universe)
    if panel.empty:
        raise RuntimeError(f"No rows left in panel after universe filter: {args.universe}")

    folds_all = wft._build_folds(
        panel["month"],
        train_years=int(args.train_window_years),
        test_years=int(args.test_window_years),
        step_years=int(args.step_years),
    )
    folds_all = [f for f in folds_all if int(f["test_year"]) >= int(args.start) and int(f["test_year"]) <= int(args.end)]
    if not folds_all:
        raise RuntimeError("No folds in requested --start/--end range.")

    holdout_folds = [
        f
        for f in folds_all
        if int(args.holdout_start) <= int(f["test_year"]) <= int(args.holdout_end)
    ]
    selection_folds = [
        f
        for f in folds_all
        if not (int(args.holdout_start) <= int(f["test_year"]) <= int(args.holdout_end))
    ]
    if not selection_folds:
        raise RuntimeError("No non-holdout folds available for selection/tuning. Adjust holdout range.")

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

    trial_rows: list[dict] = []
    for fold_idx, fold in enumerate(selection_folds, start=1):
        train_df = panel[(panel["month"] >= fold["train_start"]) & (panel["month"] <= fold["train_end"])].copy()
        test_df = panel[(panel["month"] >= fold["test_start"]) & (panel["month"] <= fold["test_end"])].copy()

        fold_rows: list[dict] = []
        for i, trial in enumerate(trials, start=1):
            trial_id = f"trial_{i:04d}"
            weights = trial.weights_tuple()
            reg_penalty = _weights_reg_penalty(weights=weights, prior=weights_prior, reg_lambda=weights_reg_lambda)

            train_trades = _simulate_window_strategy_a(train_df, trial)
            train_metrics = wft._window_metrics(train_trades)
            train_objective = _train_objective(train_metrics, reg_penalty=reg_penalty)

            test_trades = _simulate_window_strategy_a(test_df, trial)
            test_metrics = wft._window_metrics(test_trades)
            test_net_ret, trade_count = _compute_net_returns(test_trades, position_country, cost_cfg)
            benchmark_ret, bench_symbols, bench_missing = _compute_benchmark_returns(
                test_trades,
                position_country=position_country,
                country_benchmarks=country_benchmarks,
                index_returns=index_returns,
            )

            test_return_gross = float(test_metrics["return"])
            test_return_net = _to_period_return(test_net_ret)
            benchmark_return = _to_period_return(benchmark_ret)
            excess_return_net = float(test_return_net - benchmark_return)
            test_max_dd = float(wft._max_drawdown(test_net_ret))

            fold_rows.append(
                {
                    "trial_id": trial_id,
                    "fold_id": int(fold_idx),
                    "test_year": int(fold["test_year"]),
                    "is_selected_train": False,
                    "mos_threshold": float(trial.mos_threshold),
                    "mad_min": float(trial.mad_min),
                    "mad_penalty_k": float(trial.mad_penalty_k),
                    "min_hold_months": int(trial.min_hold_months),
                    "score_gap": float(trial.score_gap),
                    "weakness_rule_variant": str(trial.weakness_rule_variant),
                    "weight_quality": float(trial.weight_quality),
                    "weight_value": float(trial.weight_value),
                    "weight_lowrisk": float(trial.weight_lowrisk),
                    "weight_balance": float(trial.weight_balance),
                    "weights_reg_lambda": float(weights_reg_lambda),
                    "weights_reg_penalty": float(reg_penalty),
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
                    "benchmark_symbols_used": str(bench_symbols),
                    "benchmark_missing_months": int(bench_missing),
                    "trade_count": int(trade_count),
                    "chosen_params_json": trial.to_json(),
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
                "weights_reg_penalty",
                "trial_id",
            ],
            ascending=[False, False, False, True, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        if not fold_df.empty:
            winner = str(fold_df.iloc[0]["trial_id"])
            fold_df["is_selected_train"] = fold_df["trial_id"].eq(winner)

        trial_rows.extend(fold_df.to_dict(orient="records"))

    trials_df = pd.DataFrame(trial_rows)
    trials_df = trials_df.sort_values(by=["fold_id", "trial_id"], kind="mergesort").reset_index(drop=True)
    trials_csv = run_dir / "trials.csv"
    trials_df.to_csv(trials_csv, index=False)

    agg = (
        trials_df.groupby(
            [
                "trial_id",
                "mos_threshold",
                "mad_min",
                "mad_penalty_k",
                "min_hold_months",
                "score_gap",
                "weakness_rule_variant",
                "weight_quality",
                "weight_value",
                "weight_lowrisk",
                "weight_balance",
                "weights_reg_lambda",
                "weights_reg_penalty",
                "chosen_params_json",
            ],
            as_index=False,
        )
        .agg(
            folds=("fold_id", "count"),
            selected_folds=("is_selected_train", "sum"),
            mean_train_objective=("train_objective", "mean"),
            median_excess_return_net=("excess_return_net", "median"),
            worst_fold_excess_return_net=("excess_return_net", "min"),
            median_test_max_dd=("test_max_dd", "median"),
            mean_test_turnover=("test_turnover", "mean"),
            mean_test_pct_cash=("test_pct_cash", "mean"),
            mean_test_return_net=("test_return_net", "mean"),
            mean_test_return_gross=("test_return_gross", "mean"),
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

    agg["robust_utility"] = agg.apply(
        lambda r: _robust_utility(
            median_excess_return_net=float(r["median_excess_return_net"]),
            worst_fold_excess_return_net=float(r["worst_fold_excess_return_net"]),
            median_test_max_dd=float(r["median_test_max_dd"]),
            mean_test_turnover=float(r["mean_test_turnover"]),
            mean_test_pct_cash=float(r["mean_test_pct_cash"]),
        ),
        axis=1,
    )

    agg = agg.sort_values(
        by=[
            "robust_utility",
            "worst_fold_excess_return_net",
            "median_test_max_dd",
            "mean_test_turnover",
            "mean_test_pct_cash",
            "trial_id",
        ],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)

    best = agg.iloc[0]
    best_params_json = str(best["chosen_params_json"])
    best_params_obj = json.loads(best_params_json)
    rationale = (
        "Selected by robust utility on non-holdout outer folds "
        f"({float(best['robust_utility']):.6f}) with penalties on turnover/pct_cash."
    )

    holdout_rows: list[dict] = []
    holdout_net_monthly: list[float] = []
    holdout_bench_monthly: list[float] = []
    holdout_trades_all: list[pd.DataFrame] = []
    holdout_switches = 0

    best_trial = StrategyAParams(
        mos_threshold=float(best["mos_threshold"]),
        mad_min=float(best["mad_min"]),
        mad_penalty_k=float(best["mad_penalty_k"]),
        min_hold_months=int(best["min_hold_months"]),
        score_gap=float(best["score_gap"]),
        weight_quality=float(best["weight_quality"]),
        weight_value=float(best["weight_value"]),
        weight_lowrisk=float(best["weight_lowrisk"]),
        weight_balance=float(best["weight_balance"]),
        weakness_rule_variant=str(best["weakness_rule_variant"]),
    )

    for fold in holdout_folds:
        train_df = panel[(panel["month"] >= fold["train_start"]) & (panel["month"] <= fold["train_end"])].copy()
        test_df = panel[(panel["month"] >= fold["test_start"]) & (panel["month"] <= fold["test_end"])].copy()
        _ = train_df  # explicit: not used for tuning in holdout

        test_trades = _simulate_window_strategy_a(test_df, best_trial)
        test_metrics = wft._window_metrics(test_trades)
        net_ret, _trade_count = _compute_net_returns(test_trades, position_country, cost_cfg)
        bench_ret, _bench_symbols, _bench_missing = _compute_benchmark_returns(
            test_trades,
            position_country=position_country,
            country_benchmarks=country_benchmarks,
            index_returns=index_returns,
        )

        holdout_net_monthly.extend(_to_num(net_ret).fillna(0.0).tolist())
        holdout_bench_monthly.extend(_to_num(bench_ret).fillna(0.0).tolist())
        if not test_trades.empty:
            t = test_trades.copy()
            t["test_year"] = int(fold["test_year"])
            holdout_trades_all.append(t)
            holdout_switches += int(_to_num(t["position_change"]).fillna(0.0).sum())

        holdout_rows.append(
            {
                "mode": "holdout_best",
                "train_start": fold["train_start"].date().isoformat(),
                "train_end": fold["train_end"].date().isoformat(),
                "test_year": int(fold["test_year"]),
                "chosen_params": best_params_json,
                "candidate_or_cash": "CASH" if float(test_metrics["pct_cash"]) >= 0.999 else "CANDIDATE",
                "return": float(test_metrics["return"]),
                "max_dd": float(test_metrics["max_dd"]),
                "turnover": float(test_metrics["turnover"]),
                "pct_cash": float(test_metrics["pct_cash"]),
            }
        )

    holdout_df = pd.DataFrame(holdout_rows, columns=WFT_RESULTS_COLS)
    holdout_csv = run_dir / "holdout_results.csv"
    holdout_df.to_csv(holdout_csv, index=False)

    holdout_net_series = pd.Series(holdout_net_monthly, dtype=float)
    holdout_bench_series = pd.Series(holdout_bench_monthly, dtype=float)
    holdout_cum_net = _to_period_return(holdout_net_series)
    holdout_cum_bench = _to_period_return(holdout_bench_series)
    holdout_excess_net = float(holdout_cum_net - holdout_cum_bench)
    holdout_max_dd = float(wft._max_drawdown(holdout_net_series))
    if holdout_trades_all:
        htr = pd.concat(holdout_trades_all, ignore_index=True)
        holdout_turnover = float(_to_num(htr["position_change"]).fillna(0.0).iloc[1:].mean()) if len(htr) > 1 else 0.0
        holdout_pct_cash = float(htr["position"].astype(str).eq("CASH").mean()) if len(htr) > 0 else 1.0
    else:
        holdout_turnover = 0.0
        holdout_pct_cash = 1.0

    best_config = {
        "optimizer": {
            "method": "nested_walk_forward_strategy_a",
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
            "holdout_start": int(args.holdout_start),
            "holdout_end": int(args.holdout_end),
            "selection_folds": int(len(selection_folds)),
            "holdout_folds": int(len(holdout_folds)),
        },
        "strategy_a": {
            "weights_prior": {k: float(weights_prior[k]) for k in WEIGHT_KEYS},
            "weights_bounds": [float(weights_bounds[0]), float(weights_bounds[1])],
            "weights_reg_lambda": float(weights_reg_lambda),
        },
        "selected_params": best_params_obj,
        "selection_metrics": {
            "trial_id": str(best["trial_id"]),
            "robust_utility": float(best["robust_utility"]),
            "median_excess_return_net": float(best["median_excess_return_net"]),
            "worst_fold_excess_return_net": float(best["worst_fold_excess_return_net"]),
            "median_test_max_dd": float(best["median_test_max_dd"]),
            "mean_test_turnover": float(best["mean_test_turnover"]),
            "mean_test_pct_cash": float(best["mean_test_pct_cash"]),
            "selected_folds": int(best["selected_folds"]),
            "folds": int(best["folds"]),
            "oos_cagr_net": float(best["oos_cagr_net"]),
            "benchmark_cagr": float(best["benchmark_cagr"]),
        },
        "holdout_metrics": {
            "cumulative_return_net": float(holdout_cum_net),
            "excess_return_net": float(holdout_excess_net),
            "max_dd": float(holdout_max_dd),
            "turnover": float(holdout_turnover),
            "pct_cash": float(holdout_pct_cash),
            "switches": int(holdout_switches),
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
            "mad_penalty_k",
            "min_hold_months",
            "score_gap",
            "weight_quality",
            "weight_value",
            "weight_lowrisk",
            "weight_balance",
            "robust_utility",
            "median_excess_return_net",
            "worst_fold_excess_return_net",
            "median_test_max_dd",
            "mean_test_turnover",
            "mean_test_pct_cash",
            "oos_cagr_net",
        ]
    ]

    summary_lines = [
        "# WFT Optimize Summary",
        "",
        f"- Output dir: `{run_dir}`",
        f"- Config: `{cfg_path}`",
        f"- Master path: `{master_path}`",
        f"- Prices path: `{prices_path}`",
        f"- Period: {int(args.start)}-{int(args.end)}",
        f"- Holdout: {int(args.holdout_start)}-{int(args.holdout_end)} (excluded from selection/tuning)",
        f"- Train/Test windows: {int(args.train_window_years)}y / {int(args.test_window_years)}y",
        f"- Rebalance: {args.rebalance}",
        f"- Universe: {args.universe}",
        f"- Search mode: {args.search_mode}",
        f"- Trials requested/used: {int(args.n_trials)} / {int(n_trials)}",
        f"- Selection folds: {len(selection_folds)}",
        f"- Holdout folds: {len(holdout_folds)}",
        "",
        "## Benchmark Selection",
    ]
    for c in ("NO", "SE", "DK", "FI"):
        sym = country_benchmarks.get(c, "")
        summary_lines.append(f"- {c}: `{sym if sym else 'N/A'}`")
    summary_lines.append("")

    all_notes = build_notes + benchmark_notes
    if all_notes:
        summary_lines.append("## Notes")
        for note in all_notes:
            summary_lines.append(f"- {note}")
        summary_lines.append("")

    summary_lines.extend(
        [
            "## Ranking Logic",
            "Robust utility = median_excess_return_net + 0.50*worst_fold_excess_return_net + 0.25*median_test_max_dd - 0.10*turnover - 0.05*pct_cash.",
            "",
            "## Top 10 Trials",
            _md_table(top10),
            "",
            "## Selected Best Config",
            f"- Trial: `{best['trial_id']}`",
            f"- Params: `{best_params_json}`",
            (
                f"- Key metrics: robust_utility={float(best['robust_utility']):.6f}, "
                f"median_excess_net={float(best['median_excess_return_net']):.2%}, "
                f"worst_fold_excess_net={float(best['worst_fold_excess_return_net']):.2%}, "
                f"median_maxdd={float(best['median_test_max_dd']):.2%}, "
                f"turnover={float(best['mean_test_turnover']):.2%}, "
                f"pct_cash={float(best['mean_test_pct_cash']):.2%}"
            ),
            f"- Rationale: {rationale}",
            "",
            "## Holdout Evaluation",
            f"- Holdout years: {int(args.holdout_start)}-{int(args.holdout_end)}",
            f"- cumulative_return_net: {holdout_cum_net:.2%}",
            f"- excess_return_net: {holdout_excess_net:.2%}",
            f"- max_dd: {holdout_max_dd:.2%}",
            f"- turnover: {holdout_turnover:.2%}",
            f"- pct_cash: {holdout_pct_cash:.2%}",
            f"- switches: {int(holdout_switches)}",
            f"- holdout_results.csv: `{holdout_csv}`",
        ]
    )

    summary_md = run_dir / "optimize_summary.md"
    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"OK: {trials_csv}")
    print(f"OK: {best_yaml}")
    print(f"OK: {summary_md}")
    print(f"OK: {holdout_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
