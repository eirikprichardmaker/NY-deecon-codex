from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import resolve_paths
from src.common.io import read_parquet


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
    _quality_write(ctx.run_dir, quality_lines)

    out = processed_dir / "master.parquet"
    master.to_parquet(out, index=False)
    log.info(f"MASTER: wrote {out} (from master_input + prices snapshot)")
    return 0
