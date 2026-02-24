from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src import wft
from src.common.utils import get_first_env

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    load_dotenv = None  # type: ignore[assignment]


DEFAULT_PROXY_KPI_IDS = {
    "ev_ebit": 10,
    "nd_ebitda": 42,
    "roe_pct": 33,
    "ebitda_m": 54,
    "ocf_per_share": 68,
    "ebitda_per_share": 71,
    "net_debt_per_share": 73,
    "fcf_per_share_alt": 23,
}

MARKET_TO_COUNTRY_ID = {
    "SE": 1,
    "NO": 2,
    "FI": 3,
    "DK": 4,
    "US": 5,
    "CA": 6,
    "UK": 7,
    "DE": 8,
    "FR": 9,
}


def _load_env_defaults() -> None:
    if load_dotenv is not None:
        try:
            load_dotenv(override=False)
        except Exception:
            pass


def _auth_key() -> str:
    _load_env_defaults()
    key = get_first_env(["BORSDATA_AUTHKEY", "BORSDATA_API_KEY", "BORSDATA_KEY"])
    if not key:
        raise RuntimeError("Missing Borsdata auth key. Set BORSDATA_AUTHKEY (or BORSDATA_API_KEY).")
    return key


def _fetch_true_market_instrument_ids(markets: list[str], timeout: int = 60) -> dict[str, set[int]]:
    req_markets = sorted({str(m).strip().upper() for m in markets if str(m).strip()})
    out: dict[str, set[int]] = {m: set() for m in req_markets}
    if not req_markets:
        return out

    key = _auth_key()
    r = requests.get("https://apiservice.borsdata.se/v1/instruments", params={"authKey": key}, timeout=timeout)
    r.raise_for_status()
    payload = r.json() or {}
    rows = payload.get("instruments") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise RuntimeError("Unexpected /instruments payload while building true market filter.")

    for item in rows:
        if not isinstance(item, dict):
            continue
        ins_raw = item.get("insId")
        country_raw = item.get("countryId")
        instrument_raw = item.get("instrument")
        try:
            ins_id = int(ins_raw)
            country_id = int(country_raw)
        except Exception:
            continue
        # instrument==0 => ordinary stock/equity in Borsdata payload
        try:
            instrument_type = int(instrument_raw)
        except Exception:
            instrument_type = -1

        if instrument_type != 0:
            continue

        for market, cid in MARKET_TO_COUNTRY_ID.items():
            if market in out and country_id == cid:
                out[market].add(ins_id)
    return out


def _latest_kpi_value(values_obj) -> float:
    if isinstance(values_obj, np.ndarray):
        values = values_obj.tolist()
    elif isinstance(values_obj, (list, tuple)):
        values = list(values_obj)
    else:
        return float("nan")

    if not values:
        return float("nan")

    rows = []
    for item in values:
        if not isinstance(item, dict):
            continue
        year = pd.to_numeric(item.get("y"), errors="coerce")
        period = pd.to_numeric(item.get("p"), errors="coerce")
        val = pd.to_numeric(item.get("v"), errors="coerce")
        if pd.isna(year) or pd.isna(val):
            continue
        rows.append((int(year), float(period) if not pd.isna(period) else np.nan, float(val)))

    if not rows:
        return float("nan")

    annual = [r for r in rows if np.isfinite(r[1]) and int(r[1]) == 5]
    chosen = annual if annual else rows
    chosen.sort(key=lambda r: (r[0], r[1] if np.isfinite(r[1]) else -1.0), reverse=True)
    return float(chosen[0][2])


def _dcf_value_m(cashflow0_m: float, wacc: float, years: int = 5, growth: float = 0.02, terminal_growth: float = 0.02) -> float:
    if not np.isfinite(cashflow0_m) or not np.isfinite(wacc):
        return float("nan")
    if wacc <= 0 or wacc <= terminal_growth:
        return float("nan")

    pv_sum = 0.0
    for t in range(1, years + 1):
        ft = cashflow0_m * ((1.0 + growth) ** t)
        pv_sum += ft / ((1.0 + wacc) ** t)

    fn = cashflow0_m * ((1.0 + growth) ** years)
    tv = (fn * (1.0 + terminal_growth)) / (wacc - terminal_growth)
    return float(pv_sum + (tv / ((1.0 + wacc) ** years)))


def _load_market_prices(raw_asof_dir: Path, market: str, allowed_ids: set[int] | None = None) -> pd.DataFrame:
    prices_dir = raw_asof_dir / "prices" / f"market={market}"
    files = sorted(prices_dir.glob("ins_id=*.parquet"))

    chunks = []
    for path in files:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        cols = {str(c).lower(): c for c in df.columns}
        date_col = cols.get("d", "date")
        close_col = cols.get("c", "adj_close")
        vol_col = cols.get("v", "volume")
        ins_col = cols.get("ins_id", "ins_id")
        if date_col not in df.columns or close_col not in df.columns:
            continue
        if ins_col not in df.columns:
            continue

        out = pd.DataFrame(
            {
                "ins_id": pd.to_numeric(df[ins_col], errors="coerce"),
                "date": pd.to_datetime(df[date_col], errors="coerce"),
                "adj_close": pd.to_numeric(df[close_col], errors="coerce"),
                "volume": pd.to_numeric(df[vol_col], errors="coerce") if vol_col in df.columns else np.nan,
            }
        )
        out = out.dropna(subset=["ins_id", "date", "adj_close"])
        if allowed_ids is not None:
            out = out[pd.to_numeric(out["ins_id"], errors="coerce").isin(allowed_ids)].copy()
        chunks.append(out)

    if not chunks:
        return pd.DataFrame(columns=["ins_id", "date", "adj_close", "volume"])
    out = pd.concat(chunks, ignore_index=True)
    out["ins_id"] = out["ins_id"].astype(int)
    return out


def _load_market_kpi_snapshot(
    raw_asof_dir: Path,
    market: str,
    kpi_ids: Iterable[int],
    allowed_ids: set[int] | None = None,
) -> pd.DataFrame:
    wanted = set(int(x) for x in kpi_ids)
    kpi_dir = raw_asof_dir / "kpis" / f"market={market}"
    files = sorted(kpi_dir.glob("ins_id=*.parquet"))

    rows = []
    for path in files:
        df = pd.read_parquet(path)
        if df.empty or "KpiId" not in df.columns or "values" not in df.columns:
            continue
        kid = pd.to_numeric(df["KpiId"], errors="coerce")
        for target in wanted:
            mask = kid.eq(target)
            if not bool(mask.any()):
                continue
            v = _latest_kpi_value(df.loc[mask, "values"].iloc[0])
            ins_id = int(pd.to_numeric(df.loc[mask, "ins_id"].iloc[0], errors="coerce"))
            if allowed_ids is not None and ins_id not in allowed_ids:
                continue
            rows.append((ins_id, target, v))

    if not rows:
        return pd.DataFrame(columns=["ins_id"])

    out = pd.DataFrame(rows, columns=["ins_id", "kpi_id", "value"]).drop_duplicates(subset=["ins_id", "kpi_id"], keep="last")
    piv = out.pivot(index="ins_id", columns="kpi_id", values="value").reset_index()
    piv.columns = ["ins_id"] + [str(int(c)) for c in piv.columns[1:]]
    return piv


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _build_static_proxy(market: str, kpi_snap: pd.DataFrame, latest_price: pd.DataFrame, wacc: float) -> pd.DataFrame:
    d = kpi_snap.merge(latest_price, on="ins_id", how="left").copy()

    ev_ebit = _series(d, str(DEFAULT_PROXY_KPI_IDS["ev_ebit"]))
    nd_ebitda = _series(d, str(DEFAULT_PROXY_KPI_IDS["nd_ebitda"]))
    roe_pct = _series(d, str(DEFAULT_PROXY_KPI_IDS["roe_pct"]))
    ebitda_m = _series(d, str(DEFAULT_PROXY_KPI_IDS["ebitda_m"]))
    ocf_ps = _series(d, str(DEFAULT_PROXY_KPI_IDS["ocf_per_share"]))
    ebitda_ps = _series(d, str(DEFAULT_PROXY_KPI_IDS["ebitda_per_share"]))
    nd_ps = _series(d, str(DEFAULT_PROXY_KPI_IDS["net_debt_per_share"]))
    fcf_alt_ps = _series(d, str(DEFAULT_PROXY_KPI_IDS["fcf_per_share_alt"]))
    px_last = pd.to_numeric(d["latest_price"], errors="coerce")

    shares_est = (ebitda_m * 1_000_000.0) / ebitda_ps
    shares_est = shares_est.where(np.isfinite(shares_est) & (shares_est > 0))

    market_cap = px_last * shares_est
    net_debt = nd_ps * shares_est
    cashflow_total = (ocf_ps * shares_est).where((ocf_ps * shares_est).notna(), fcf_alt_ps * shares_est)

    roic_proxy = roe_pct.copy()
    if roic_proxy.notna().any():
        med = float(roic_proxy.abs().median(skipna=True))
        if np.isfinite(med) and med > 2.0:
            roic_proxy = roic_proxy / 100.0

    intrinsic_ev_m = pd.Series(
        [_dcf_value_m(float(v) / 1_000_000.0, float(wacc)) if np.isfinite(v) else np.nan for v in cashflow_total.values],
        index=d.index,
        dtype=float,
    )
    intrinsic_equity = (intrinsic_ev_m * 1_000_000.0) - net_debt
    mos = (intrinsic_equity / market_cap) - 1.0
    fcf_yield = cashflow_total / market_cap

    weak_roic = roic_proxy.isna() | (roic_proxy <= 0.0)
    weak_fcf = fcf_yield.isna() | (fcf_yield <= 0.0)
    weak_nd = nd_ebitda.isna() | (nd_ebitda > 3.5)
    weak_ev = ev_ebit.isna() | (ev_ebit <= 0.0) | (ev_ebit > 20.0)
    quality_weak_count = weak_roic.astype(int) + weak_fcf.astype(int) + weak_nd.astype(int) + weak_ev.astype(int)
    value_creation_ok = roic_proxy.notna() & ((roic_proxy - float(wacc)) > 0.03)

    out = pd.DataFrame(
        {
            "ticker": d["ins_id"].map(lambda x: f"I{int(x)}.{market}"),
            "yahoo_ticker": d["ins_id"].map(lambda x: f"I{int(x)}.{market}"),
            "company": d["ins_id"].map(lambda x: f"INS_{int(x)}"),
            "k": d["ins_id"].map(lambda x: f"I{int(x)}"),
            "market_cap": market_cap,
            "mos": mos,
            "high_risk_flag": (nd_ebitda >= 3.5).fillna(False),
            "quality_weak_count": quality_weak_count,
            "value_creation_ok_base": value_creation_ok.fillna(False),
            "roic": roic_proxy,
            "fcf_yield": fcf_yield,
            "ev_ebit": ev_ebit,
            "nd_ebitda": nd_ebitda,
            "relevant_index_symbol": f"^IDX_{market}",
            "relevant_index_key": f"IDX_{market}",
            "is_bank_proxy": False,
            "intrinsic_equity": intrinsic_equity,
            "proxy_shares_est": shares_est,
            "proxy_cashflow_total": cashflow_total,
        }
    )
    return out.drop_duplicates(subset=["k"], keep="first")


def _build_prices_with_proxy_index(market_prices: pd.DataFrame, market: str) -> pd.DataFrame:
    stock = market_prices.copy()
    stock["ticker"] = stock["ins_id"].map(lambda x: f"I{int(x)}.{market}")

    idx = stock.groupby("date", as_index=False).agg(adj_close=("adj_close", "mean"), volume=("volume", "sum"))
    idx["ticker"] = f"^IDX_{market}"

    out = pd.concat(
        [stock[["date", "ticker", "adj_close", "volume"]], idx[["date", "ticker", "adj_close", "volume"]]],
        ignore_index=True,
    )
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def _summarize_market(
    market: str,
    trades: pd.DataFrame,
    initial_capital: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    static_df: pd.DataFrame,
    panel_df: pd.DataFrame,
) -> dict:
    ret = pd.to_numeric(trades.get("ret", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    final_capital = float(initial_capital * float((1.0 + ret).prod())) if len(ret) else float(initial_capital)
    years = float(len(ret) / 12.0) if len(ret) else 0.0
    cagr = float((final_capital / initial_capital) ** (1.0 / years) - 1.0) if years > 0 and final_capital > 0 else 0.0

    return {
        "market": market,
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "initial_capital_nok": float(initial_capital),
        "final_capital_nok": final_capital,
        "total_return": float(final_capital / initial_capital - 1.0),
        "cagr": cagr,
        "max_dd": float(wft._max_drawdown(ret)) if len(ret) else 0.0,
        "pct_cash": float(trades["position"].astype(str).eq("CASH").mean()) if len(trades) else 1.0,
        "months": int(len(trades)),
        "rows_static": int(len(static_df)),
        "rows_panel": int(len(panel_df)),
        "non_null_market_cap": int(pd.to_numeric(static_df["market_cap"], errors="coerce").notna().sum()),
        "non_null_mos": int(pd.to_numeric(static_df["mos"], errors="coerce").notna().sum()),
        "non_null_roic": int(pd.to_numeric(static_df["roic"], errors="coerce").notna().sum()),
        "proxy_note": "Stress test proxy for DE/FR. Uses KPI ids 10,42,33,54,68,71,73 (fallback 23).",
    }


def _default_start(asof: pd.Timestamp) -> pd.Timestamp:
    return asof - pd.DateOffset(years=10)


def run_stress_test(
    project_root: Path,
    asof: str,
    markets: list[str],
    start_date: str | None,
    end_date: str | None,
    initial_capital: float,
    wacc: float,
    run_dir: Path | None = None,
) -> Path:
    asof_ts = pd.to_datetime(asof, errors="raise")
    start_ts = pd.to_datetime(start_date, errors="raise") if start_date else _default_start(asof_ts)
    end_ts = pd.to_datetime(end_date, errors="raise") if end_date else asof_ts

    if run_dir is None:
        run_dir = project_root / "runs" / f"stress_markets_{time.strftime('%Y%m%d_%H%M%S')}"
    if not run_dir.is_absolute():
        run_dir = (project_root / run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_asof_dir = project_root / "data" / "raw" / asof
    if not raw_asof_dir.exists():
        raise FileNotFoundError(f"Missing raw asof dir: {raw_asof_dir}")

    true_ids_by_market = _fetch_true_market_instrument_ids(markets)

    summaries = []
    for market in markets:
        market = str(market).strip().upper()
        if not market:
            continue

        allowed_ids = true_ids_by_market.get(market, set())
        if not allowed_ids:
            summaries.append({"market": market, "error": f"No true instrument ids resolved for market={market}."})
            continue

        prices = _load_market_prices(raw_asof_dir, market, allowed_ids=allowed_ids)
        if prices.empty:
            summaries.append({"market": market, "error": f"No prices loaded for true market ids under prices/market={market}."})
            continue

        kpi_snapshot = _load_market_kpi_snapshot(raw_asof_dir, market, DEFAULT_PROXY_KPI_IDS.values(), allowed_ids=allowed_ids)
        if kpi_snapshot.empty:
            summaries.append({"market": market, "error": f"No KPI snapshot loaded for true market ids under kpis/market={market}."})
            continue

        latest_price = (
            prices.sort_values(["ins_id", "date"])
            .groupby("ins_id", as_index=False)
            .tail(1)[["ins_id", "adj_close"]]
            .rename(columns={"adj_close": "latest_price"})
        )
        static_df = _build_static_proxy(market=market, kpi_snap=kpi_snapshot, latest_price=latest_price, wacc=wacc)
        price_df = _build_prices_with_proxy_index(prices, market=market)

        static_path = run_dir / f"stress_static_{market}.parquet"
        prices_path = run_dir / f"stress_prices_{market}.parquet"
        static_df.to_parquet(static_path, index=False)
        price_df.to_parquet(prices_path, index=False)

        panel = wft._load_monthly_panel(prices_path, static_df)
        panel = panel[
            (pd.to_datetime(panel["month"], errors="coerce") >= start_ts.to_period("M").to_timestamp("M"))
            & (pd.to_datetime(panel["month"], errors="coerce") <= end_ts.to_period("M").to_timestamp("M"))
        ].copy()

        params = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
        trades = wft._simulate_window(panel, params=params)
        trades.to_csv(run_dir / f"stress_trades_{market}.csv", index=False)

        summaries.append(
            {
                **_summarize_market(
                    market=market,
                    trades=trades,
                    initial_capital=float(initial_capital),
                    start_date=start_ts,
                    end_date=end_ts,
                    static_df=static_df,
                    panel_df=panel,
                ),
                "true_market_ids_total": int(len(allowed_ids)),
                "true_market_ids_used_prices": int(pd.to_numeric(prices["ins_id"], errors="coerce").dropna().astype(int).nunique()),
                "true_market_ids_used_kpis": int(pd.to_numeric(kpi_snapshot["ins_id"], errors="coerce").dropna().astype(int).nunique()),
            }
        )

    summary_df = pd.DataFrame(summaries)
    summary_csv = run_dir / "stress_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    lines = [
        "# Market Stress Test Summary",
        "",
        f"- asof: {asof}",
        f"- window: {start_ts.date()} -> {end_ts.date()}",
        f"- initial_capital_nok: {float(initial_capital):.2f}",
        "",
    ]
    if summary_df.empty:
        lines.append("No markets were processed.")
    else:
        lines.append("| market | final_capital_nok | cagr | max_dd | pct_cash | non_null_mos | note |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for _, row in summary_df.iterrows():
            if "error" in row and pd.notna(row["error"]):
                lines.append(f"| {row.get('market','')} | - | - | - | - | - | {row['error']} |")
            else:
                lines.append(
                    f"| {row['market']} | {float(row['final_capital_nok']):.2f} | {float(row['cagr']):.2%} | "
                    f"{float(row['max_dd']):.2%} | {float(row['pct_cash']):.2%} | {int(row['non_null_mos'])} | "
                    f"{row['proxy_note']} |"
                )
    (run_dir / "stress_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--asof", required=True, help="YYYY-MM-DD")
    p.add_argument("--markets", default="DE,FR")
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD. Default: asof - 10 years")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD. Default: asof")
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--wacc", type=float, default=0.09)
    p.add_argument("--run-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    markets = [x.strip().upper() for x in str(args.markets).split(",") if x.strip()]
    out_dir = run_stress_test(
        project_root=root,
        asof=str(args.asof),
        markets=markets,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=float(args.initial_capital),
        wacc=float(args.wacc),
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )
    print(f"OK: {out_dir}")
    print(f"OK: {out_dir / 'stress_summary.csv'}")
    print(f"OK: {out_dir / 'stress_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
