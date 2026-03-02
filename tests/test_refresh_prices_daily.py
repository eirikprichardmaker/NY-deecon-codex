from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import refresh_prices_daily


def test_refresh_prices_writes_processed_and_raw(tmp_path: Path) -> None:
    src_prices = tmp_path / "prices_latest.csv"
    src_prices.write_text(
        "\n".join(
            [
                "ticker,date,adj_close,volume",
                "AAA.ST,2026-02-26,100,1000",
                "AAA.ST,2026-02-27,101,1100",
                "^OSEAX,2026-02-27,2000,0",
                "^OMXS,2026-02-27,3000,0",
                "^OMXC25,2026-02-27,1800,0",
                "^HEX,2026-02-27,12000,0",
            ]
        ),
        encoding="utf-8",
    )

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  raw_dir: {str((tmp_path / 'raw')).replace('\\', '/')}",
                f"  processed_dir: {str((tmp_path / 'processed')).replace('\\', '/')}",
                "sources:",
                "  prices:",
                f"    prices_latest_input: {str(src_prices).replace('\\', '/')}",
            ]
        ),
        encoding="utf-8",
    )

    rc = refresh_prices_daily.refresh_prices(
        asof="2026-02-27",
        config_path=str(cfg),
        require_fresh_days=0,
    )
    assert rc == 0

    out_processed = tmp_path / "processed" / "prices.parquet"
    out_raw = tmp_path / "raw" / "2026-02-27" / "prices.parquet"
    assert out_processed.exists()
    assert out_raw.exists()

    d1 = pd.read_parquet(out_processed)
    d2 = pd.read_parquet(out_raw)
    assert len(d1) == len(d2)
    assert pd.to_datetime(d1["date"], errors="coerce").max().date().isoformat() == "2026-02-27"


def test_refresh_prices_keeps_existing_when_new_source_is_too_narrow(tmp_path: Path) -> None:
    src_prices = tmp_path / "prices_latest.csv"
    src_prices.write_text(
        "\n".join(
            [
                "ticker,date,adj_close,volume",
                "AAA.ST,2026-02-27,101,1000",
                "^OSEAX,2026-02-27,2000,0",
                "^OMXS,2026-02-27,3000,0",
                "^OMXC25,2026-02-27,1800,0",
                "^HEX,2026-02-27,12000,0",
            ]
        ),
        encoding="utf-8",
    )

    universe = tmp_path / "tickers.csv"
    universe.write_text(
        "\n".join(
            [
                "ticker,yahoo_ticker,country",
                "AAA,AAA.ST,SE",
                "BBB,BBB.ST,SE",
            ]
        ),
        encoding="utf-8",
    )

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame(
        {
            "ticker": ["AAA.ST", "BBB.ST", "^OSEAX", "^OMXS", "^OMXC25", "^HEX"],
            "date": ["2026-02-27"] * 6,
            "adj_close": [101.0, 202.0, 2000.0, 3000.0, 1800.0, 12000.0],
            "volume": [1000.0, 2000.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    existing.to_parquet(processed_dir / "prices.parquet", index=False)

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  raw_dir: {str((tmp_path / 'raw')).replace('\\', '/')}",
                f"  processed_dir: {str(processed_dir).replace('\\', '/')}",
                "sources:",
                "  prices:",
                f"    prices_latest_input: {str(src_prices).replace('\\', '/')}",
                "refresh_prices_daily:",
                f"  universe_tickers_input: {str(universe).replace('\\', '/')}",
                "  min_universe_coverage: 0.60",
                "  min_relative_coverage_vs_existing: 0.80",
                "  max_absolute_coverage_drop_vs_existing: 0.15",
            ]
        ),
        encoding="utf-8",
    )

    rc = refresh_prices_daily.refresh_prices(
        asof="2026-02-27",
        config_path=str(cfg),
        require_fresh_days=0,
    )
    assert rc == 0

    out = pd.read_parquet(processed_dir / "prices.parquet")
    stocks = set(out.loc[~out["ticker"].astype(str).str.startswith("^"), "ticker"].astype(str).tolist())
    assert stocks == {"AAA.ST", "BBB.ST"}
