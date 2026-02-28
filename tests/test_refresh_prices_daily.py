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

