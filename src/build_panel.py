from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _norm_ticker(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper().lstrip("^")
    if "." in s:
        s = s.split(".", 1)[0]
    return s


def _load_mapping(repo_root: Path) -> pd.DataFrame:
    for name in ["config/tickers.csv", "config/tickers_with_insid_clean.csv", "config/tickers_with_insid.csv"]:
        p = repo_root / name
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        low = {str(c).strip().lower(): c for c in df.columns}
        tcol = low.get("ticker")
        ycol = low.get("yahoo_ticker") or low.get("yahoo")
        if tcol and ycol:
            out = df[[tcol, ycol]].rename(columns={tcol: "ticker", ycol: "yahoo_ticker"}).copy()
            out["ticker_norm"] = out["ticker"].map(_norm_ticker)
            out["yahoo_ticker"] = out["yahoo_ticker"].astype(str).str.strip()
            out = out[out["ticker_norm"].ne("") & out["yahoo_ticker"].ne("")]
            return out[["ticker_norm", "yahoo_ticker"]].drop_duplicates("ticker_norm", keep="first")
    return pd.DataFrame(columns=["ticker_norm", "yahoo_ticker"])


def _ensure_yahoo_ticker(prices: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    pz = prices.copy().reset_index()
    pz.columns = [str(c).strip() for c in pz.columns]
    cols = {c.lower(): c for c in pz.columns}

    # infer source ticker column
    t_col = None
    for cand in ["yahoo_ticker", "ticker", "symbol"]:
        if cand in cols:
            t_col = cols[cand]
            break

    if "yahoo_ticker" not in pz.columns:
        pz["yahoo_ticker"] = ""
    if t_col:
        pz["ticker_norm"] = pz[t_col].map(_norm_ticker)
        missing = pz["yahoo_ticker"].astype(str).str.strip().eq("")
        pz.loc[missing, "yahoo_ticker"] = pz.loc[missing, "ticker_norm"]
    else:
        pz["ticker_norm"] = ""

    inferred = int((pz["yahoo_ticker"].astype(str).str.strip() != "").sum())
    mapped = 0
    if not mapping.empty:
        pz = pz.merge(mapping, on="ticker_norm", how="left", suffixes=("", "_map"))
        need_map = pz["yahoo_ticker"].astype(str).str.strip().eq("")
        mapped = int(need_map.sum())
        pz.loc[need_map, "yahoo_ticker"] = pz.loc[need_map, "yahoo_ticker_map"]
        pz = pz.drop(columns=[c for c in ["yahoo_ticker_map"] if c in pz.columns])

    pz["yahoo_ticker"] = pz["yahoo_ticker"].astype(str).str.strip()
    pz["missing_price"] = pz["yahoo_ticker"].eq("")
    pz["reason_technical_fail"] = np.where(pz["missing_price"], "missing_yahoo_ticker", "")
    return pz, {"rows": len(pz), "with_yahoo": inferred, "mapped_candidates": mapped, "missing": int(pz["missing_price"].sum())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--asof", required=True, help="YYYY-MM-DD (for sporing i panel)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--fundamentals", default=None, help="default: data/golden/fundamentals_history.parquet")
    p.add_argument("--prices", default="data/golden/prices_panel.parquet", help="Yahoo panel parquet")
    p.add_argument("--quality-out", default=None, help="Optional path for quality.md")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    fundamentals_path = Path(args.fundamentals) if args.fundamentals else data_dir / "golden" / "fundamentals_history.parquet"
    prices_path = Path(args.prices)

    f = pd.read_parquet(fundamentals_path)
    pz = pd.read_parquet(prices_path)

    if "date" not in pz.columns:
        raise ValueError("prices mangler kolonne 'date'")
    pz["date"] = pd.to_datetime(pz["date"])

    mapping = _load_mapping(Path(__file__).resolve().parents[1])
    pz, quality = _ensure_yahoo_ticker(pz, mapping)

    # harmoniser pris-kolonne
    if "price" not in pz.columns:
        if "adj_close" in pz.columns:
            pz = pz.rename(columns={"adj_close": "price"})
        elif "close" in pz.columns:
            pz = pz.rename(columns={"close": "price"})
        else:
            raise ValueError("prices må ha 'price' eller ('adj_close'/'close')")

    if "period_end" not in f.columns:
        raise ValueError("fundamentals mangler 'period_end'")
    f["period_end"] = pd.to_datetime(f["period_end"])

    if "yahoo_ticker" not in f.columns:
        raise ValueError("fundamentals mangler 'yahoo_ticker' (legg i tickers.csv i ingest)")

    f = f.sort_values(["yahoo_ticker", "period_end"])
    pz = pz.sort_values(["yahoo_ticker", "date"])

    merged = pd.merge_asof(
        f,
        pz,
        by="yahoo_ticker",
        left_on="period_end",
        right_on="date",
        direction="backward",
        allow_exact_matches=True,
    )

    merged["asof"] = args.asof
    merged["missing_price"] = pd.to_numeric(merged.get("price"), errors="coerce").isna()
    merged["reason_technical_fail"] = np.where(merged["missing_price"], "missing_price", merged.get("reason_technical_fail", ""))

    required = ["yahoo_ticker", "asof", "country", "sector", "price"]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise RuntimeError(f"DoD feilet: panel mangler kolonner: {missing}")

    out_path = data_dir / "processed" / "panel.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    quality_path = Path(args.quality_out) if args.quality_out else data_dir / "processed" / "quality.md"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(
        "\n".join(
            [
                f"# Data quality ({args.asof})",
                f"- Price rows: {quality['rows']}",
                f"- Rows with yahoo_ticker: {quality['with_yahoo']}",
                f"- Rows mapped from config: {quality['mapped_candidates']}",
                f"- Rows still missing yahoo_ticker: {quality['missing']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"OK: skrev {out_path}")
    print(f"OK: skrev {quality_path}")


if __name__ == "__main__":
    main()
