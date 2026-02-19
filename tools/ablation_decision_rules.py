from __future__ import annotations

import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import _quality_gate, _value_creation_gate
from src.common.utils import safe_div, zscore


def _canon_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        "".join(ch if ch.isalnum() else "_" for ch in str(c).strip().lower()).strip("_")
        for c in out.columns
    ]
    return out


def _norm_ticker(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper().lstrip("^")
    if "." in s:
        s = s.split(".", 1)[0]
    return s.strip()


def _as_num_series(df: pd.DataFrame, aliases: list[str]) -> pd.Series:
    for c in aliases:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _compute_static(master_valued: Path) -> pd.DataFrame:
    df = _canon_cols(pd.read_parquet(master_valued))

    if "ticker" not in df.columns:
        raise RuntimeError("master_valued mangler ticker")

    df["k"] = df["ticker"].map(_norm_ticker)
    df = df[df["k"] != ""].copy()

    mcap = _as_num_series(df, ["market_cap_current", "market_cap"])
    med = float(mcap.median(skipna=True)) if mcap.notna().any() else float("nan")
    if np.isfinite(med) and med < 1e6:
        mcap = mcap * 1_000_000.0

    net_debt = _as_num_series(df, ["net_debt_current", "net_debt", "net_debt_used"])
    ev = mcap + net_debt.fillna(0.0)

    ie = _as_num_series(df, ["intrinsic_equity"])
    iev = _as_num_series(df, ["intrinsic_ev"])

    if ie.notna().any():
        intrinsic = ie
        mos = pd.Series(safe_div(intrinsic, mcap), index=df.index) - 1.0
        mos_basis = "equity_vs_mcap"
    elif iev.notna().any():
        intrinsic = iev
        mos = pd.Series(safe_div(intrinsic, ev), index=df.index) - 1.0
        mos_basis = "ev_vs_ev"
    else:
        raise RuntimeError("master_valued mangler intrinsic_equity/intrinsic_ev")

    beta = _as_num_series(df, ["beta"])
    nd_ebitda = _as_num_series(df, ["nd_ebitda", "n_debt_ebitda_current"])

    high_risk = (
        (beta.fillna(0.0) >= 1.5) |
        (nd_ebitda.fillna(0.0) >= 3.5)
    )

    roic = _as_num_series(df, ["roic", "roic_current"])
    fcf_yield = _as_num_series(df, ["fcf_yield"])
    comps: list[pd.Series] = []
    wts: list[float] = []
    if roic.notna().any():
        comps.append(zscore(roic))
        wts.append(0.60)
    if fcf_yield.notna().any():
        comps.append(zscore(fcf_yield))
        wts.append(0.40)
    if comps:
        w = np.array(wts, dtype=float)
        w = w / w.sum()
        quality_score = sum(w[i] * comps[i] for i in range(len(comps)))
    else:
        quality_score = pd.Series(0.0, index=df.index)

    out = pd.DataFrame(
        {
            "k": df["k"],
            "ticker": df["ticker"],
            "company": df.get("company", pd.Series("", index=df.index)),
            "market_cap": mcap,
            "mos": mos,
            "mos_req": np.where(high_risk, 0.40, 0.30),
            "mos_basis": mos_basis,
            "quality_score": quality_score,
            "beta": beta,
            "nd_ebitda": nd_ebitda,
            "roic": roic,
            "fcf_yield": fcf_yield,
            "ev_ebit": _as_num_series(df, ["ev_ebit", "ev_ebit_current"]),
            "wacc_used": _as_num_series(df, ["wacc_used", "wacc"]),
        }
    ).drop_duplicates(subset=["k"], keep="last")

    gates_input = out.copy()
    gates_input["roic_current"] = gates_input["roic"]

    vg = _value_creation_gate(gates_input, {"value_creation_spread_decay_per_year": 0.01})
    qg = _quality_gate(gates_input, {"quality_weak_fail_min": 2})
    for c in vg.columns:
        out[c] = vg[c].values
    for c in qg.columns:
        out[c] = qg[c].values
    return out


def _monthly_snapshots(prices_panel: Path) -> pd.DataFrame:
    px = pd.read_parquet(prices_panel)
    need = ["date", "ticker", "adj_close", "above_ma200", "mad"]
    miss = [c for c in need if c not in px.columns]
    if miss:
        raise RuntimeError(f"prices_panel mangler kolonner: {miss}")

    px = px.copy()
    px["date"] = pd.to_datetime(px["date"], errors="coerce")
    px = px.dropna(subset=["date", "ticker", "adj_close"]) 
    px["k"] = px["ticker"].map(_norm_ticker)
    px = px[px["k"] != ""].copy()

    px["period"] = px["date"].dt.to_period("M")
    snap = (
        px.sort_values(["period", "k", "date"])
        .groupby(["period", "k"], as_index=False)
        .tail(1)
        .copy()
    )
    return snap[["period", "date", "k", "adj_close", "above_ma200", "mad"]]


def _pick_for_period(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    above = frame["above_ma200"].astype("boolean").fillna(False)
    mad_s = pd.to_numeric(frame["mad"], errors="coerce")
    mad_ok = mad_s.notna() & (mad_s >= -0.05)
    tech_ok = above.astype(bool) & mad_ok.astype(bool)

    base_fund = frame["mos"].notna() & frame["mos_req"].notna() & (frame["mos"] >= frame["mos_req"])

    if mode == "baseline":
        fund_ok = base_fund
    elif mode == "change1_quality":
        fund_ok = base_fund & frame["quality_gate_ok"].fillna(False)
    elif mode == "change2_value":
        fund_ok = base_fund & frame["value_creation_ok"].fillna(False)
    elif mode == "v2":
        fund_ok = base_fund & frame["value_creation_ok"].fillna(False) & frame["quality_gate_ok"].fillna(False)
    else:
        raise ValueError(mode)

    eligible = frame[fund_ok & tech_ok].copy()
    if eligible.empty:
        return pd.DataFrame(
            [
                {
                    "action": "CASH",
                    "k": "CASH",
                    "ticker": "CASH",
                    "company": "CASH",
                    "quality_score": np.nan,
                    "mos": np.nan,
                }
            ]
        )

    eligible = eligible.sort_values(by=["quality_score", "mos", "market_cap"], ascending=[False, False, False], na_position="last")
    top = eligible.iloc[[0]].copy()
    top["action"] = "BUY"
    return top[["action", "k", "ticker", "company", "quality_score", "mos"]]


def _max_drawdown(nav: pd.Series) -> float:
    cummax = nav.cummax()
    dd = (nav / cummax) - 1.0
    return float(dd.min()) if not dd.empty else 0.0


def _deflated_sharpe_prob(returns: pd.Series, n_trials: int = 2) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 3:
        return float("nan")

    sr = float(r.mean() / r.std(ddof=1) * math.sqrt(12.0)) if float(r.std(ddof=1)) > 0 else float("nan")
    if not np.isfinite(sr):
        return float("nan")

    t = len(r)
    skew = float(r.skew()) if np.isfinite(r.skew()) else 0.0
    kurt = float(r.kurt()) + 3.0 if np.isfinite(r.kurt()) else 3.0
    sigma_sr = math.sqrt(max(1e-12, (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * (sr ** 2)) / max(1, t - 1)))

    nd = NormalDist()
    euler = 0.5772156649
    z1 = nd.inv_cdf(1.0 - 1.0 / max(2, n_trials))
    z2 = nd.inv_cdf(1.0 - 1.0 / (max(2, n_trials) * math.e))
    sr_star = sigma_sr * ((1.0 - euler) * z1 + euler * z2)
    z = (sr - sr_star) / sigma_sr
    return float(nd.cdf(z))


def _summarize(period_df: pd.DataFrame) -> dict:
    ret = pd.to_numeric(period_df["ret"], errors="coerce").fillna(0.0)
    nav = (1.0 + ret).cumprod()

    invested = period_df[period_df["k"] != "CASH"].copy()
    hit_rate = float((pd.to_numeric(invested["ret"], errors="coerce") > 0).mean()) if len(invested) else float("nan")

    prev_k = period_df["k"].shift(1)
    turnover = float((period_df["k"] != prev_k).iloc[1:].mean()) if len(period_df) > 1 else 0.0
    share_cash = float((period_df["k"] == "CASH").mean())
    stability = 1.0 - turnover

    sr = float(ret.mean() / ret.std(ddof=1) * math.sqrt(12.0)) if float(ret.std(ddof=1)) > 0 else float("nan")
    return {
        "periods": int(len(period_df)),
        "hit_rate": hit_rate,
        "max_drawdown": _max_drawdown(nav),
        "sharpe_ann": sr,
        "deflated_sharpe_prob": _deflated_sharpe_prob(ret, n_trials=2),
        "turnover": turnover,
        "share_cash": share_cash,
        "selection_stability": stability,
        "terminal_nav": float(nav.iloc[-1]) if len(nav) else 1.0,
    }


def run_ablation(master_valued: Path, prices_panel: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    static = _compute_static(master_valued)
    snap = _monthly_snapshots(prices_panel)

    periods = sorted(snap["period"].unique().tolist())
    rows: list[dict] = []

    period_by_mode: dict[str, pd.DataFrame] = {}
    for mode in ["baseline", "change1_quality", "change2_value", "v2"]:
        picks = []
        for i in range(len(periods) - 1):
            p = periods[i]
            p_next = periods[i + 1]
            cur = snap[snap["period"] == p].copy()
            nxt = snap[snap["period"] == p_next].copy()

            frame = static.merge(cur, on="k", how="left")
            choice = _pick_for_period(frame, mode).iloc[0].to_dict()
            choice["mode"] = mode
            choice["period"] = str(p)
            choice["date"] = str(cur["date"].max().date()) if len(cur) else ""

            if choice["k"] == "CASH":
                choice["ret"] = 0.0
            else:
                entry = float(cur.loc[cur["k"] == choice["k"], "adj_close"].iloc[0]) if (cur["k"] == choice["k"]).any() else float("nan")
                if (nxt["k"] == choice["k"]).any() and np.isfinite(entry) and entry > 0:
                    exit_p = float(nxt.loc[nxt["k"] == choice["k"], "adj_close"].iloc[0])
                    choice["ret"] = float((exit_p / entry) - 1.0)
                else:
                    choice["ret"] = 0.0
                    choice["action"] = "CASH_NO_NEXT_PRICE"
                    choice["k"] = "CASH"
                    choice["ticker"] = "CASH"
                    choice["company"] = "CASH"
            picks.append(choice)

        period_df = pd.DataFrame(picks)
        summary = _summarize(period_df)
        summary["mode"] = mode
        rows.append(summary)

        period_by_mode[mode] = period_df

    metrics = pd.DataFrame(rows)
    periods_joined = period_by_mode["baseline"].rename(
        columns={c: f"{c}_baseline" for c in period_by_mode["baseline"].columns if c != "period"}
    )
    for mode in ["change1_quality", "change2_value", "v2"]:
        right = period_by_mode[mode].rename(
            columns={c: f"{c}_{mode}" for c in period_by_mode[mode].columns if c != "period"}
        )
        periods_joined = periods_joined.merge(right, on=["period"], how="outer")
    periods_joined = periods_joined.sort_values("period")
    return metrics, periods_joined


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    master_valued = repo / "data" / "processed" / "master_valued.parquet"
    prices_panel = repo / "data" / "raw" / "prices" / "prices_panel.parquet"

    metrics, periods = run_ablation(master_valued, prices_panel)

    out_dir = repo / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "ablation_baseline_vs_v2_metrics.csv"
    periods_csv = out_dir / "ablation_baseline_vs_v2_periods.csv"
    metrics.to_csv(metrics_csv, index=False)
    periods.to_csv(periods_csv, index=False)

    docs_dir = repo / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "ablation_baseline_vs_v2.md"

    m = metrics.set_index("mode")

    def _fmt(v: float, pct: bool = False) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "nan"
        return f"{v:.2%}" if pct else f"{v:.4f}"

    md = []
    md.append("# Ablation: baseline vs v2 decision rules")
    md.append("")
    md.append("Samme datasett/freeze:")
    md.append(f"- master_valued: `{master_valued}`")
    md.append(f"- prices_panel: `{prices_panel}`")
    md.append("")
    md.append("Endringer:")
    md.append("- #1 Kvalitetsgate: >=2 svekkede kvalitetsindikatorer => ikke kvalifisert")
    md.append("- #2 Verdiskapingsgate: ROIC > WACC i 3-ar konservativ bane")
    md.append("")
    md.append("## Metrics")
    md.append("| Metric | Baseline | #1 only | #2 only | #1+#2 |")
    md.append("| --- | ---: | ---: | ---: | ---: |")
    mode_label = {
        "baseline": "baseline",
        "change1_quality": "change1_quality",
        "change2_value": "change2_value",
        "v2": "v2",
    }
    for k, pct in [
        ("periods", False),
        ("hit_rate", True),
        ("max_drawdown", True),
        ("sharpe_ann", False),
        ("deflated_sharpe_prob", True),
        ("turnover", True),
        ("share_cash", True),
        ("selection_stability", True),
        ("terminal_nav", False),
    ]:
        b = float(m.loc[mode_label["baseline"], k]) if k in m.columns else float("nan")
        c1 = float(m.loc[mode_label["change1_quality"], k]) if k in m.columns else float("nan")
        c2 = float(m.loc[mode_label["change2_value"], k]) if k in m.columns else float("nan")
        v = float(m.loc[mode_label["v2"], k]) if k in m.columns else float("nan")
        md.append(f"| {k} | {_fmt(b, pct=pct)} | {_fmt(c1, pct=pct)} | {_fmt(c2, pct=pct)} | {_fmt(v, pct=pct)} |")

    md.append("")
    md.append("Artefakter:")
    md.append(f"- `{metrics_csv}`")
    md.append(f"- `{periods_csv}`")

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"OK: {metrics_csv}")
    print(f"OK: {periods_csv}")
    print(f"OK: {md_path}")


if __name__ == "__main__":
    main()
