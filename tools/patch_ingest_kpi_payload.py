from __future__ import annotations

import re
import time
from pathlib import Path

TARGET = Path("src/ingest_fundamentals_history.py")


NEW_NORMALIZE_BLOCK = r'''
def _normalize_kpi_history_payload(payload: Any, reporttype: str) -> Any:
    """
    KPI history kan komme som:
      A) dict: {kpiId, reportTime, priceValue, kpisList:[...]}  <-- NY (det du har nå)
      B) dict: {instrument, reportsYear/reportsQuarter/reportsR12}  <-- annen shape (sett på /reports)
      C) list: gammel shape
    Returner noe parseren kan iterere på, uten å miste instrument/insId når det trengs.
    """
    if isinstance(payload, dict):
        # NY: kpisList
        if isinstance(payload.get("kpisList"), list):
            return payload["kpisList"]

        # bucket-shape (ikke vanlig her, men støttes)
        if any(k in payload for k in ("reportsYear", "reportsQuarter", "reportsR12")):
            return payload

        # generiske wrappere
        for k in ("data", "Data", "items", "Items", "instruments", "Instruments"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]

    return payload


def _coerce_date_like(d: Any) -> Any:
    """Gjør år/kvartal-varianter om til ISO-dato for bedre pd.to_datetime()."""
    if d is None:
        return None

    if isinstance(d, (int, float)):
        yi = int(d)
        if 1900 <= yi <= 2100:
            return f"{yi}-12-31"

    if isinstance(d, str):
        s = d.strip()
        # "2024"
        if len(s) == 4 and s.isdigit():
            return f"{s}-12-31"
        # "2024Q4"
        m = re.match(r"^(\d{4})\s*Q([1-4])$", s, re.I)
        if m:
            y = int(m.group(1))
            q = int(m.group(2))
            month = 3 * q
            day = 31 if month in (3, 12) else 30
            return f"{y}-{month:02d}-{day:02d}"

    return d
'''.lstrip("\n")


NEW_PARSE_BLOCK = r'''
def _parse_kpi_history_payload(payload: Any, reporttype: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    payload = _normalize_kpi_history_payload(payload, reporttype)

    def _ins_id(obj: Dict[str, Any]) -> Any:
        # direkte id-felt
        iid = (
            obj.get("insId")
            or obj.get("ins_id")
            or obj.get("InsId")
            or obj.get("instrumentId")
            or obj.get("instrument_id")
            or obj.get("id")
        )
        if iid is not None:
            return iid
        # noen ganger: instrument er dict
        inst = obj.get("instrument")
        if isinstance(inst, dict):
            return inst.get("insId") or inst.get("ins_id") or inst.get("InsId") or inst.get("id")
        return None

    def _points(obj: Dict[str, Any]) -> List[Any]:
        for k in ("values", "history", "data", "items", "points", "kpiHistory", "kpi_history"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
        return []

    def _date_val(pt: Dict[str, Any]):
        d = (
            pt.get("d")
            or pt.get("date")
            or pt.get("reportEndDate")
            or pt.get("reportDate")
            or pt.get("periodEnd")
            or pt.get("period_end")
            or pt.get("t")
            or pt.get("time")
            or pt.get("reportTime")
        )
        v = pt.get("v") if "v" in pt else pt.get("value", pt.get("val"))
        return _coerce_date_like(d), v

    # CASE 1: list (inkl. kpisList => list med objekter)
    if isinstance(payload, list):
        for obj in payload:
            if not isinstance(obj, dict):
                continue

            iid = _ins_id(obj)
            pts = _points(obj)

            # noen ganger er obj selv et punkt
            if not pts and any(k in obj for k in ("d", "date", "value", "v", "reportEndDate", "reportDate", "reportTime")):
                d, v = _date_val(obj)
                rows.append({"ins_id": iid, "date": d, "value": v})
                continue

            for pt in pts:
                if isinstance(pt, dict):
                    d, v = _date_val(pt)
                    rows.append({"ins_id": iid, "date": d, "value": v})
        return rows

    # CASE 2: dict med buckets (fallback)
    if isinstance(payload, dict) and any(k in payload for k in ("reportsYear", "reportsQuarter", "reportsR12")):
        inst = payload.get("instrument") if isinstance(payload.get("instrument"), dict) else {}
        iid = (inst.get("insId") or inst.get("id")) if isinstance(inst, dict) else None
        key_map = {"year": "reportsYear", "quarter": "reportsQuarter", "r12": "reportsR12"}
        bucket = payload.get(key_map.get(reporttype, ""), [])
        if isinstance(bucket, list):
            for pt in bucket:
                if isinstance(pt, dict):
                    d, v = _date_val(pt)
                    rows.append({"ins_id": iid, "date": d, "value": v})
        return rows

    return rows
'''.lstrip("\n")


def _replace_def_block(text: str, func_name: str, new_block: str) -> str:
    # match fra "def func_name" til rett før neste top-level "def " eller EOF
    pattern = rf"^def {re.escape(func_name)}\b[^\n]*\n(?:.*\n)*?(?=^def |\Z)"
    m = re.search(pattern, text, flags=re.M | re.S)
    if not m:
        raise SystemExit(f"Fant ikke funksjon i fila: {func_name}")
    return text[:m.start()] + new_block.rstrip() + "\n\n" + text[m.end():]


def _ensure_import_re(text: str) -> str:
    # finnes allerede?
    if re.search(r"^import re\b", text, flags=re.M):
        return text
    # legg inn før pandas-import hvis mulig
    if re.search(r"^import pandas as pd\b", text, flags=re.M):
        return re.sub(r"^import pandas as pd\b", "import re\nimport pandas as pd", text, count=1, flags=re.M)
    # ellers legg inn etter future-import
    return re.sub(r"^from __future__ import annotations\s*\n", "from __future__ import annotations\n\nimport re\n", text, count=1, flags=re.M)


def main():
    if not TARGET.exists():
        raise SystemExit(f"Fant ikke: {TARGET.resolve()}")

    text = TARGET.read_text(encoding="utf-8")

    # backup
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(f".py.bak_{ts}")
    backup.write_text(text, encoding="utf-8")

    # patch
    text = _ensure_import_re(text)
    text = _replace_def_block(text, "_normalize_kpi_history_payload", NEW_NORMALIZE_BLOCK)
    text = _replace_def_block(text, "_parse_kpi_history_payload", NEW_PARSE_BLOCK)

    TARGET.write_text(text, encoding="utf-8")

    print(f"[OK] Patched {TARGET}")
    print(f"[OK] Backup  {backup}")
    print("[NEXT] Kjør: python .\\src\\ingest_fundamentals_history.py --asof 2026-02-16 --mode both --include-r12")


if __name__ == "__main__":
    main()
