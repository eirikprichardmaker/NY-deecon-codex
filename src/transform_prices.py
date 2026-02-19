from __future__ import annotations

import json
import io
import urllib.error
import urllib.parse
import urllib.request
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import resolve_paths
from src.common.io import read_csv, read_parquet, write_parquet

INDEX_TICKERS = ["^OSEAX", "^OMXS", "^OMXC25", "^HEX"]


def _load_any(path: Path) -> pd.DataFrame:
    return read_parquet(path) if path.suffix.lower() == ".parquet" else read_csv(path)


def _std_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _norm_ticker(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().upper()
    if "." in s:
        s = s.split(".", 1)[0]
    return s.lstrip("^")


def _ensure_price_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = _std_cols(df)
    aliases = {
        "adj_close": ["adj_close", "close", "price", "last"],
        "date": ["date", "price_date", "datetime", "timestamp", "time"],
        "ticker": ["ticker", "symbol", "yahoo_ticker"],
        "volume": ["volume"],
    }
    for target, cands in aliases.items():
        if target in out.columns:
            continue
        for c in cands:
            if c in out.columns:
                out = out.rename(columns={c: target})
                break

    if "ticker" not in out.columns or "date" not in out.columns or "adj_close" not in out.columns:
        raise ValueError("prices input must include ticker/date/adj_close (or aliases close/price/last)")

    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = np.nan
    else:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    out = out.dropna(subset=["ticker", "date", "adj_close"]).copy()
    return out


def _add_price_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["ticker", "date"])
    g = out.groupby("ticker", group_keys=False)
    out["ma21"] = g["adj_close"].transform(lambda s: s.rolling(21, min_periods=21).mean())
    out["ma200"] = g["adj_close"].transform(lambda s: s.rolling(200, min_periods=200).mean())
    out["mad"] = (out["ma21"] - out["ma200"]) / out["ma200"]
    out["above_ma200"] = out["adj_close"] > out["ma200"]
    return out


def _fetch_yahoo_chart(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    period1 = int(start.tz_localize(timezone.utc).timestamp())
    period2 = int((end + pd.Timedelta(days=1)).tz_localize(timezone.utc).timestamp())
    params = {
        "period1": str(period1),
        "period2": str(period2),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    qs = urllib.parse.urlencode(params)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{qs}"

    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = (((payload or {}).get("chart") or {}).get("result") or [None])[0] or {}
    ts = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    close = quote.get("close")
    vol = quote.get("volume")
    prices = adj if isinstance(adj, list) and len(adj) == len(ts) else close

    if not ts or not isinstance(prices, list):
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
            "ticker": symbol,
            "adj_close": pd.to_numeric(prices, errors="coerce"),
            "volume": pd.to_numeric(vol if isinstance(vol, list) else [np.nan] * len(ts), errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "adj_close"]).copy()
    return out


def _fetch_stooq_index(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # Stooq index symbols are typically lowercase and keep '^' prefix.
    sym = str(symbol).strip().lower()
    if not sym:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])
    stooq_symbol = sym if sym.startswith("^") else f"^{sym}"
    if not stooq_symbol:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    with urllib.request.urlopen(url, timeout=30) as resp:
        txt = resp.read().decode("utf-8")

    if "No data" in txt or "Brak danych" in txt:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])

    raw = pd.read_csv(io.StringIO(txt))
    raw = _std_cols(raw)
    if "date" not in raw.columns or "close" not in raw.columns:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["date"], errors="coerce"),
            "ticker": symbol,
            "adj_close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw["volume"], errors="coerce") if "volume" in raw.columns else np.nan,
        }
    )
    out = out.dropna(subset=["date", "adj_close"]).copy()
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    return out


def _fetch_missing_indices(index_tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, log) -> pd.DataFrame:
    rows = []
    for sym in index_tickers:
        try:
            hist = _fetch_yahoo_chart(sym, start, end)
            if not hist.empty:
                rows.append(hist)
                log.info(f"TRANSFORM prices: fetched {sym} rows={len(hist)}")
            else:
                log.info(f"TRANSFORM prices: fetched {sym} returned empty")
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
            log.info(f"TRANSFORM prices: yahoo failed for {sym}")
            try:
                hist2 = _fetch_stooq_index(sym, start, end)
                if not hist2.empty:
                    rows.append(hist2)
                    log.info(f"TRANSFORM prices: fetched {sym} from stooq rows={len(hist2)}")
                else:
                    log.info(f"TRANSFORM prices: stooq returned empty for {sym}")
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
                log.info(f"TRANSFORM prices: stooq failed for {sym}")
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "volume"])
    return pd.concat(rows, ignore_index=True)


def _append_missing_indices(df: pd.DataFrame, asof: str, log, fetcher=_fetch_missing_indices) -> pd.DataFrame:
    out = _ensure_price_schema(df)
    if out.empty:
        return _add_price_indicators(out)

    present = {_norm_ticker(t) for t in out["ticker"].dropna().astype(str).tolist()}
    missing = [s for s in INDEX_TICKERS if _norm_ticker(s) not in present]
    if not missing:
        return _add_price_indicators(out)

    start = pd.to_datetime(out["date"].min()).normalize()
    end = pd.to_datetime(asof).normalize()
    if end < start:
        end = pd.to_datetime(out["date"].max()).normalize()

    fetched = fetcher(missing, start, end, log)
    if fetched is None or fetched.empty:
        log.info("TRANSFORM prices: no index rows appended")
        return _add_price_indicators(out)

    fetched = _ensure_price_schema(fetched)
    out = pd.concat([out, fetched], ignore_index=True)
    out = out.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")
    return _add_price_indicators(out)


def run(ctx, log) -> int:
    paths = resolve_paths(ctx.cfg, ctx.project_root)
    raw_asof_dir = paths["raw_dir"] / ctx.asof
    processed_dir = paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    cand = list(raw_asof_dir.glob("prices.*"))
    if not cand:
        log.info("TRANSFORM prices: skipped (no prices.* in raw).")
        return 0

    df = _std_cols(_load_any(cand[0]))
    df = _append_missing_indices(df, asof=ctx.asof, log=log)

    out = processed_dir / "prices.parquet"
    write_parquet(out, df)
    log.info(f"TRANSFORM prices: wrote {out}")
    return 0
