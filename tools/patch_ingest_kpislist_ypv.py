from __future__ import annotations
import re, time
from pathlib import Path

TARGET = Path("src/ingest_fundamentals_history.py")

NEW_PARSE_BLOCK = r"""
def _parse_kpi_history_payload(payload: Any, reporttype: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    payload = _normalize_kpi_history_payload(payload, reporttype)

    def _ins_id(obj: Dict[str, Any]) -> Any:
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

    def _date_from_yp(y: Any, p: Any) -> Any:
        # values-punkter fra Borsdata har ofte {y,p,v}
        if y is None:
            return None
        try:
            yy = int(y)
        except Exception:
            return None

        # Year: bruk alltid 31.12
        if reporttype == "year":
            return f"{yy}-12-31"

        # Quarter/R12: bruk p hvis mulig
        if p is None:
            return f"{yy}-12-31"
        try:
            pp = int(p)
        except Exception:
            return f"{yy}-12-31"

        # Hvis p ser ut som kvartal (1-4): bruk kvartalsslutt
        if 1 <= pp <= 4:
            month = 3 * pp
            return (pd.Timestamp(yy, month, 1) + pd.offsets.MonthEnd(0)).date().isoformat()

        # Hvis p ser ut som måned (1-12): bruk måneds-slutt
        if 1 <= pp <= 12:
            month = pp
            return (pd.Timestamp(yy, month, 1) + pd.offsets.MonthEnd(0)).date().isoformat()

        # fallback
        return f"{yy}-12-31"

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
        if d is None and ("y" in pt):
            d = _date_from_yp(pt.get("y"), pt.get("p"))
        v = pt.get("v") if "v" in pt else pt.get("value", pt.get("val"))
        return d, v

    # CASE 1: list (inkl. kpisList => list med objekter)
    if isinstance(payload, list):
        for obj in payload:
            if not isinstance(obj, dict):
                continue
            iid = _ins_id(obj)
            pts = _points(obj)

            # noen ganger er obj selv et punkt
            if not pts and any(k in obj for k in ("d","date","value","v","y","p","reportEndDate","reportDate","reportTime")):
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
""".lstrip("\n")

def _replace_def_block(text: str, func_name: str, new_block: str) -> str:
    pattern = rf"^def {re.escape(func_name)}\b[^\n]*\n(?:.*\n)*?(?=^def |\Z)"
    m = re.search(pattern, text, flags=re.M | re.S)
    if not m:
        raise SystemExit(f"Fant ikke funksjon i fila: {func_name}")
    return text[:m.start()] + new_block.rstrip() + "\n\n" + text[m.end():]

def main():
    if not TARGET.exists():
        raise SystemExit(f"Fant ikke: {TARGET.resolve()}")

    text = TARGET.read_text(encoding="utf-8")

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_suffix(f".py.bak_{ts}")
    backup.write_text(text, encoding="utf-8")

    text = _replace_def_block(text, "_parse_kpi_history_payload", NEW_PARSE_BLOCK)

    TARGET.write_text(text, encoding="utf-8")
    print(f"[OK] Patched {TARGET}")
    print(f"[OK] Backup  {backup}")

if __name__ == "__main__":
    main()
