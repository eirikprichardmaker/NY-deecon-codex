import re
from pathlib import Path
from datetime import datetime

TARGET = Path(r"tools/borsdata_freeze.py")

NEW_BLOCK = """    # ------------------ Reports per instrument (20y via reporttype endpoints) ------------------
    if cfg.do_reports_pi:
        out_raw = raw_dir / "reports_per_instrument"
        out_norm = norm_dir / "reports_per_instrument.jsonl.gz"
        ensure_dir(out_raw)

        reporttypes = ["year", "r12", "quarter"]
        for ins_id in ins_ids:
            for rt in reporttypes:
                key = f"reports_pi|rt={rt}|ins_id={ins_id}|original={cfg.reports_original}|maxcount={cfg.reports_pi_maxcount}"
                if state.is_done(key):
                    continue

                endpoint = f"{API_BASE}/instruments/{ins_id}/reports/{rt}"
                params = {
                    "authKey": auth,
                    "original": cfg.reports_original,
                    "maxcount": cfg.reports_pi_maxcount,
                }
                payload, meta = borsdata_get(session, endpoint, params, min_sleep=cfg.pace_s)

                raw_path = out_raw / f"ins_{ins_id}__{rt}.json.gz"
                gz_json_write(raw_path, {"meta": meta, "params": params, "payload": payload})

                nrows = gz_jsonl_append(
                    out_norm,
                    [{
                        "dataset": "reports_per_instrument",
                        "rt": rt,
                        "ins_id": ins_id,
                        "fetched_at_utc": utc_now().isoformat(timespec="seconds"),
                        "payload": payload,
                    }],
                )

                state.mark_done(
                    key,
                    {"raw_file": str(raw_path), "normalized_rows": nrows, "done_at_utc": utc_now().isoformat(timespec="seconds")},
                )
                state.save(state_path)

"""

def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    b = path.with_suffix(path.suffix + f".bak_reports_pi_v2_{ts}")
    b.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return b

def find_section_bounds(lines):
    """
    Finn start og slutt for eksisterende reports_pi-blokk.
    Strategi:
      - Finn linje som inneholder '/instruments/{ins_id}/reports' (gammel logikk)
      - Gå opp til nærmeste '    if cfg.do_reports_pi:' eller seksjonskommentar
      - Gå ned til neste seksjonskommentar '    # ------------------' som IKKE er del av blokken
    """
    old_endpoint_idx = None
    for i, ln in enumerate(lines):
        if "/instruments/{ins_id}/reports" in ln and "/reports/" not in ln:
            old_endpoint_idx = i
            break
        if 'endpoint = f"{API_BASE}/instruments/{ins_id}/reports"' in ln:
            old_endpoint_idx = i
            break

    if old_endpoint_idx is None:
        # fallback: bare finn if cfg.do_reports_pi:
        for i, ln in enumerate(lines):
            if ln.startswith("    if cfg.do_reports_pi:"):
                old_endpoint_idx = i
                break

    if old_endpoint_idx is None:
        return None, None

    # start: scan upwards
    start = old_endpoint_idx
    for j in range(old_endpoint_idx, -1, -1):
        if lines[j].startswith("    #") and "Reports per instrument" in lines[j]:
            start = j
            break
        if lines[j].startswith("    if cfg.do_reports_pi:"):
            start = j - 1 if (j - 1 >= 0 and lines[j - 1].startswith("    #")) else j
            break

    # end: scan down to next section marker at same indent
    end = None
    for j in range(old_endpoint_idx + 1, len(lines)):
        if lines[j].startswith("    # ------------------") and ("Stockprices" in lines[j] or "KPI" in lines[j] or "Holdings" in lines[j] or "Reports" in lines[j]):
            end = j
            break
    if end is None:
        # fallback: end of function
        end = len(lines)

    return start, end

def main():
    if not TARGET.exists():
        raise SystemExit(f"Finner ikke {TARGET}. Kjør fra repo-root.")

    txt = TARGET.read_text(encoding="utf-8")
    lines = txt.splitlines(True)

    start, end = find_section_bounds(lines)
    if start is None:
        raise SystemExit(
            "Fant ikke eksisterende reports_pi-blokk.\n"
            "Kjør: Select-String .\\tools\\borsdata_freeze.py -Pattern \"cfg.do_reports_pi|/instruments/{ins_id}/reports\" -Context 2,2\n"
            "og lim inn output, så justerer jeg patcheren."
        )

    b = backup(TARGET)
    print(f"Backup laget: {b}")

    # Bytt segmentet
    new_lines = lines[:start]
    # sørg for at blokken starter på ny linje
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"
    new_lines.append("\n" + NEW_BLOCK)
    new_lines.extend(lines[end:])

    TARGET.write_text("".join(new_lines), encoding="utf-8")
    print("OK: reports_pi er nå oppdatert til /reports/{year,r12,quarter}.")

if __name__ == "__main__":
    main()
