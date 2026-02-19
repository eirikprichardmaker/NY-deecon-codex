# tests/test_inputs.py
from pathlib import Path
import yaml

def test_inputs_exist_either_raw_or_sources():
    repo = Path(__file__).resolve().parents[1]

    raw_fund = [
        repo / "data/raw/borsdata/fundamentals_mapped.csv",
        repo / "data/raw/borsdata/fundamentals_clean.csv",
        repo / "data/raw/borsdata/fundamentals_export_latest.csv",
    ]
    raw_prices = [
        repo / "data/raw/prices/prices_panel.parquet",
        repo / "data/raw/prices/prices_latest.csv",
    ]

    sources_path = repo / "configs/sources.yaml"
    sources = {}
    if sources_path.exists():
        sources = yaml.safe_load(sources_path.read_text(encoding="utf-8")) or {}

    b = sources.get("borsdata") or {}
    p = sources.get("prices") or {}

    upstream_fund = [Path(x) for x in [
        b.get("fundamentals_input"),
        b.get("fundamentals_clean_input"),
        b.get("fundamentals_latest_input"),
    ] if x]
    upstream_prices = [Path(x) for x in [
        p.get("prices_panel_input"),
        p.get("prices_latest_input"),
    ] if x]

    fund_ok = any(x.exists() for x in raw_fund) or any(x.exists() for x in upstream_fund)
    price_ok = any(x.exists() for x in raw_prices) or any(x.exists() for x in upstream_prices)

    assert fund_ok, "No fundamentals found in data/raw/borsdata/ and no valid upstream sources in configs/sources.yaml"
    assert price_ok, "No prices found in data/raw/prices/ and no valid upstream sources in configs/sources.yaml"
