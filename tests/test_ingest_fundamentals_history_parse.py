from __future__ import annotations

from src.ingest_fundamentals_history import _parse_kpi_history_payload


def test_parse_kpi_history_payload_handles_kpis_list_wrapper() -> None:
    payload = {
        "kpiId": 37,
        "reportTime": "r12",
        "priceValue": "mean",
        "kpisList": [
            {
                "instrument": 478,
                "values": [
                    {"y": 2025, "p": 3, "v": 3431.9209},
                    {"y": 2025, "p": 4, "v": 523.8990},
                ],
            }
        ],
    }

    rows = _parse_kpi_history_payload(payload)

    assert len(rows) == 2
    assert rows[0]["ins_id"] == 478
    assert rows[0]["date"] == "2025-09-30"
    assert rows[0]["value"] == 3431.9209
    assert rows[1]["date"] == "2025-12-31"


def test_parse_kpi_history_payload_handles_point_list() -> None:
    payload = [
        {"date": "2024-12-31", "value": 12.3},
        {"date": "2025-12-31", "value": 15.7},
    ]

    rows = _parse_kpi_history_payload(payload)

    assert len(rows) == 2
    assert rows[0]["ins_id"] is None
    assert rows[1]["date"] == "2025-12-31"
    assert rows[1]["value"] == 15.7
