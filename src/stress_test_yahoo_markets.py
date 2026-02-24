from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from src import wft

WIKI_DE_URL = "https://en.wikipedia.org/wiki/DAX"
WIKI_FR_URL = "https://en.wikipedia.org/wiki/CAC_40"
YAHOO_INDEX = {"DE": "^GDAXI", "FR": "^FCHI"}


def _to_num(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _dcf_value_from_fcf(fcf: float, wacc: float = 0.09, years: int = 5, growth: float = 0.02, tg: float = 0.02) -> float:
    if not np.isfinite(fcf) or not np.isfinite(wacc) or wacc <= tg or wacc <= 0:
        return float("nan")
    pv = 0.0
    for t in range(1, years + 1):
        ft = fcf * ((1.0 + growth) ** t)
        pv += ft / ((1.0 + wacc) ** t)
    fn = fcf * ((1.0 + growth) ** years)
    tv = (fn * (1.0 + tg)) / (wacc - tg)
    return float(pv + tv / ((1.0 + wacc) ** years))


def _clean_symbol_token(txt: str) -> str:
    t = str(txt).strip().upper()
    t = re.sub(r"\[[^\]]+\]", "", t).strip()
    t = t.replace("\u2212", "-")
    t = t.replace("\xa0", " ")
    t = t.split(" ", 1)[0].strip()
    return re.sub(r"[^A-Z0-9\.\-]", "", t)


def _extract_wiki_tickers(html: str, default_suffix: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    out: list[str] = []
    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
        if not headers:
            continue
        col_idx = None
        for i, h in enumerate(headers):
            if "ticker" in h or h == "symbol":
                col_idx = i
                break
        if col_idx is None:
            continue

        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if col_idx >= len(tds):
                continue
            raw = tds[col_idx].get_text(" ", strip=True)
            token = _clean_symbol_token(raw)
            if not token:
                continue
            if "." not in token:
                token = f"{token}.{default_suffix}"
            out.append(token)

    dedup = []
    seen = set()
    for sym in out:
        if sym not in seen:
            seen.add(sym)
            dedup.append(sym)
    return dedup


def _fetch_wiki_universe() -> dict[str, list[str]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    r_de = requests.get(WIKI_DE_URL, headers=headers, timeout=30)
    r_de.raise_for_status()
    r_fr = requests.get(WIKI_FR_URL, headers=headers, timeout=30)
    r_fr.raise_for_status()

    de = _extract_wiki_tickers(r_de.text, default_suffix="DE")
    fr = _extract_wiki_tickers(r_fr.text, default_suffix="PA")
    if not de:
        raise RuntimeError("Failed to extract DE ticker universe from Wikipedia.")
    if not fr:
        raise RuntimeError("Failed to extract FR ticker universe from Wikipedia.")
    return {"DE": de, "FR": fr}


def _download_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])

    data = yf.download(
        tickers=tickers,
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])

    out_rows = []
    if isinstance(data.columns, pd.MultiIndex):
        level0 = list(dict.fromkeys(data.columns.get_level_values(0)))
        use_close = "Close" if "Close" in level0 else ("Adj Close" if "Adj Close" in level0 else None)
        if use_close is None:
            return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])
        close_df = data[use_close]
        vol_df = data["Volume"] if "Volume" in level0 else None
        for t in close_df.columns:
            c = pd.to_numeric(close_df[t], errors="coerce")
            v = pd.to_numeric(vol_df[t], errors="coerce") if vol_df is not None and t in vol_df.columns else np.nan
            s = pd.DataFrame({"date": close_df.index, "ticker": t, "adj_close": c, "volume": v}).dropna(subset=["adj_close"])
            if not s.empty:
                out_rows.append(s)
    else:
        use_close = "Close" if "Close" in data.columns else ("Adj Close" if "Adj Close" in data.columns else None)
        if use_close is None:
            return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])
        ticker = tickers[0]
        s = pd.DataFrame(
            {
                "date": data.index,
                "ticker": ticker,
                "adj_close": pd.to_numeric(data[use_close], errors="coerce"),
                "volume": pd.to_numeric(data["Volume"], errors="coerce") if "Volume" in data.columns else np.nan,
            }
        ).dropna(subset=["adj_close"])
        if not s.empty:
            out_rows.append(s)

    if not out_rows:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])
    out = pd.concat(out_rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def _statement_value(df: pd.DataFrame | None, candidates: list[str]) -> float:
    if df is None or df.empty:
        return float("nan")
    cols = [str(c).lower() for c in df.columns]
    if not cols:
        return float("nan")
    col = df.columns[0]
    for c in candidates:
        if c in df.index:
            return _to_num(df.loc[c, col])
    lookup = {str(i).lower(): i for i in df.index}
    for c in candidates:
        if c.lower() in lookup:
            return _to_num(df.loc[lookup[c.lower()], col])
    return float("nan")


def _fetch_fundamentals_for_ticker(ticker: str, sleep_sec: float = 0.15) -> dict:
    tk = yf.Ticker(ticker)
    info = {}
    fi = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    try:
        fi = dict(tk.fast_info)
    except Exception:
        fi = {}

    cashflow = None
    balance = None
    income = None
    try:
        cashflow = tk.cashflow
    except Exception:
        cashflow = None
    try:
        balance = tk.balance_sheet
    except Exception:
        balance = None
    try:
        income = tk.financials
    except Exception:
        income = None

    market_cap = _to_num(info.get("marketCap"))
    if not np.isfinite(market_cap):
        market_cap = _to_num(fi.get("marketCap"))

    ev = _to_num(info.get("enterpriseValue"))
    total_debt = _to_num(info.get("totalDebt"))
    total_cash = _to_num(info.get("totalCash"))
    if not np.isfinite(total_debt):
        total_debt = _statement_value(balance, ["Total Debt", "Total debt"])
    if not np.isfinite(total_cash):
        total_cash = _statement_value(balance, ["Cash And Cash Equivalents", "Cash and cash equivalents", "Cash"])

    fcf = _to_num(info.get("freeCashflow"))
    if not np.isfinite(fcf):
        fcf = _statement_value(cashflow, ["Free Cash Flow", "Free cash flow"])
    if not np.isfinite(fcf):
        ocf = _statement_value(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = _statement_value(cashflow, ["Capital Expenditure", "Capital expenditure"])
        if np.isfinite(ocf) and np.isfinite(capex):
            fcf = ocf + capex

    ebitda = _to_num(info.get("ebitda"))
    if not np.isfinite(ebitda):
        ebitda = _statement_value(income, ["EBITDA", "Ebitda"])

    roe = _to_num(info.get("returnOnEquity"))
    nd = total_debt - total_cash if np.isfinite(total_debt) and np.isfinite(total_cash) else float("nan")
    nd_ebitda = nd / ebitda if np.isfinite(nd) and np.isfinite(ebitda) and ebitda != 0 else float("nan")
    ev_ebit = ev / ebitda if np.isfinite(ev) and np.isfinite(ebitda) and ebitda != 0 else _to_num(info.get("enterpriseToEbitda"))
    fcf_yield = fcf / market_cap if np.isfinite(fcf) and np.isfinite(market_cap) and market_cap > 0 else float("nan")

    intrinsic_ev = _dcf_value_from_fcf(fcf)
    intrinsic_equity = intrinsic_ev - nd if np.isfinite(intrinsic_ev) and np.isfinite(nd) else float("nan")
    mos = intrinsic_equity / market_cap - 1.0 if np.isfinite(intrinsic_equity) and np.isfinite(market_cap) and market_cap > 0 else float("nan")

    roic_proxy = roe
    if np.isfinite(roic_proxy) and abs(roic_proxy) > 2.0:
        roic_proxy = roic_proxy / 100.0

    weak_roic = (not np.isfinite(roic_proxy)) or (roic_proxy <= 0.0)
    weak_fcf = (not np.isfinite(fcf_yield)) or (fcf_yield <= 0.0)
    weak_nd = (not np.isfinite(nd_ebitda)) or (nd_ebitda > 3.5)
    weak_ev = (not np.isfinite(ev_ebit)) or (ev_ebit <= 0.0) or (ev_ebit > 20.0)
    quality_weak_count = int(weak_roic) + int(weak_fcf) + int(weak_nd) + int(weak_ev)
    value_creation_ok = bool(np.isfinite(roic_proxy) and (roic_proxy - 0.09 > 0.03))

    time.sleep(max(0.0, float(sleep_sec)))
    return {
        "ticker": ticker,
        "market_cap": market_cap,
        "net_debt": nd,
        "fcf": fcf,
        "ebitda": ebitda,
        "roic": roic_proxy,
        "ev_ebit": ev_ebit,
        "nd_ebitda": nd_ebitda,
        "fcf_yield": fcf_yield,
        "intrinsic_equity": intrinsic_equity,
        "mos": mos,
        "quality_weak_count": quality_weak_count,
        "value_creation_ok_base": value_creation_ok,
        "high_risk_flag": bool(np.isfinite(nd_ebitda) and nd_ebitda >= 3.5),
    }


def _build_static(df_fund: pd.DataFrame, market: str) -> pd.DataFrame:
    out = df_fund.copy()
    out["yahoo_ticker"] = out["ticker"].astype(str)
    out["k"] = out["ticker"].map(wft._norm_ticker)
    out["company"] = out["ticker"].astype(str)
    out["relevant_index_symbol"] = YAHOO_INDEX[market]
    out["relevant_index_key"] = wft._norm_ticker(YAHOO_INDEX[market])
    out["is_bank_proxy"] = False
    keep = [
        "ticker",
        "yahoo_ticker",
        "company",
        "k",
        "market_cap",
        "mos",
        "high_risk_flag",
        "quality_weak_count",
        "value_creation_ok_base",
        "roic",
        "fcf_yield",
        "ev_ebit",
        "nd_ebitda",
        "relevant_index_symbol",
        "relevant_index_key",
        "is_bank_proxy",
        "intrinsic_equity",
    ]
    return out[keep].drop_duplicates(subset=["k"], keep="first")


def _run_market(
    market: str,
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float,
    run_dir: Path,
) -> dict:
    px = _download_prices(tickers, start=start, end=end)
    idx = _download_prices([YAHOO_INDEX[market]], start=start, end=end)
    prices = pd.concat([px, idx], ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices_path = run_dir / f"yahoo_prices_{market}.parquet"
    prices.to_parquet(prices_path, index=False)

    fund_rows = []
    for t in tickers:
        try:
            fund_rows.append(_fetch_fundamentals_for_ticker(t))
        except Exception:
            fund_rows.append({"ticker": t})
    fdf = pd.DataFrame(fund_rows)
    static = _build_static(fdf, market=market)
    static_path = run_dir / f"yahoo_static_{market}.parquet"
    static.to_parquet(static_path, index=False)

    panel = wft._load_monthly_panel(prices_path, static)
    panel = panel[(panel["month"] >= start.to_period("M").to_timestamp("M")) & (panel["month"] <= end.to_period("M").to_timestamp("M"))].copy()

    params = wft.WFTParams(mos_threshold=0.30, mad_min=-0.05, weakness_rule_variant="baseline")
    trades = wft._simulate_window(panel, params)
    trades.to_csv(run_dir / f"yahoo_trades_{market}.csv", index=False)

    ret = pd.to_numeric(trades.get("ret", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    final_capital = float(initial_capital * float((1.0 + ret).prod())) if len(ret) else float(initial_capital)
    years = float(len(ret) / 12.0) if len(ret) else 0.0
    cagr = float((final_capital / initial_capital) ** (1.0 / years) - 1.0) if years > 0 and final_capital > 0 else 0.0

    return {
        "market": market,
        "n_tickers": int(len(tickers)),
        "price_rows": int(len(prices)),
        "non_null_market_cap": int(pd.to_numeric(static["market_cap"], errors="coerce").notna().sum()),
        "non_null_fcf": int(pd.to_numeric(fdf.get("fcf", pd.Series(dtype=float)), errors="coerce").notna().sum()),
        "non_null_roic": int(pd.to_numeric(static["roic"], errors="coerce").notna().sum()),
        "non_null_mos": int(pd.to_numeric(static["mos"], errors="coerce").notna().sum()),
        "initial_capital_nok": float(initial_capital),
        "final_capital_nok": final_capital,
        "total_return": float(final_capital / initial_capital - 1.0),
        "cagr": cagr,
        "max_dd": float(wft._max_drawdown(ret)) if len(ret) else 0.0,
        "pct_cash": float(trades["position"].astype(str).eq("CASH").mean()) if len(trades) else 1.0,
        "months": int(len(trades)),
    }


def run_yahoo_stress(asof: str, years: int, initial_capital: float, run_dir: Path | None = None) -> Path:
    asof_ts = pd.to_datetime(asof, errors="raise")
    start = asof_ts - pd.DateOffset(years=int(years))
    end = asof_ts
    root = Path(__file__).resolve().parent.parent
    if run_dir is None:
        run_dir = root / "runs" / f"stress_yahoo_{time.strftime('%Y%m%d_%H%M%S')}"
    if not run_dir.is_absolute():
        run_dir = (root / run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    universe = _fetch_wiki_universe()
    pd.DataFrame({"ticker": universe["DE"]}).to_csv(run_dir / "universe_DE.csv", index=False)
    pd.DataFrame({"ticker": universe["FR"]}).to_csv(run_dir / "universe_FR.csv", index=False)

    rows = []
    for market in ("DE", "FR"):
        rows.append(
            _run_market(
                market=market,
                tickers=universe[market],
                start=start,
                end=end,
                initial_capital=float(initial_capital),
                run_dir=run_dir,
            )
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "yahoo_stress_summary.csv", index=False)

    md = [
        "# Yahoo Stress Summary",
        "",
        f"- asof: {asof}",
        f"- window: {start.date()} -> {end.date()}",
        "",
        "| market | n_tickers | non_null_mos | final_capital_nok | cagr | pct_cash |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in summary.iterrows():
        md.append(
            f"| {r['market']} | {int(r['n_tickers'])} | {int(r['non_null_mos'])} | "
            f"{float(r['final_capital_nok']):.2f} | {float(r['cagr']):.2%} | {float(r['pct_cash']):.2%} |"
        )
    (run_dir / "yahoo_stress_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return run_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--asof", required=True, help="YYYY-MM-DD")
    p.add_argument("--years", type=int, default=10)
    p.add_argument("--initial-capital", type=float, default=100000.0)
    p.add_argument("--run-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = run_yahoo_stress(
        asof=str(args.asof),
        years=int(args.years),
        initial_capital=float(args.initial_capital),
        run_dir=Path(args.run_dir) if args.run_dir else None,
    )
    print(f"OK: {out}")
    print(f"OK: {out / 'yahoo_stress_summary.csv'}")
    print(f"OK: {out / 'yahoo_stress_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
