import re
from pathlib import Path
from datetime import datetime

TARGET = Path(r"tools/borsdata_freeze.py")

NEW_BLOCK = r"""    # ------------------ Reports per instrument (20y via reporttype endpoints) ------------------
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
    b = path.with_suffix(path.suffix + f".bak_reports_pi_{ts}")
    b.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return b

def main():
    if not TARGET.exists():
        raise SystemExit(f"Finner ikke {TARGET}. Kjør fra repo-root.")

    txt = TARGET.read_text(encoding="utf-8")

    # Vi bytter ut hele reports_pi-blokken inne i run_freeze:
    # fra linjen som starter med "    # ------------------ Reports per instrument"
    # og frem til like før marker-linjen "# ------------------ Stockprices history".
    pattern = r"(?s)(\n[ ]{4}# -{18,}[ ]*Reports per instrument.*?\n)(?=[ ]{4}# -{18,}[ ]*Stockprices history)"
    m = re.search(pattern, txt)
    if not m:
        raise SystemExit(
            "Fant ikke eksisterende 'Reports per instrument' blokk før 'Stockprices history'.\n"
            "Sjekk at filen fortsatt har disse kommentar-markørene."
        )

    b = backup(TARGET)
    print(f"Backup laget: {b}")

    start = m.start(1) + 1  # behold første newline
    end = m.end(1)

    # Fjern gammel blokk fra start til rett før Stockprices marker ved å matche på nytt med en bredere regex
    # (tar med hele blokken inkl. if cfg.do_reports_pi: ...).
    pattern2 = r"(?s)\n[ ]{4}# -{18,}[ ]*Reports per instrument.*?\n(?=[ ]{4}# -{18,}[ ]*Stockprices history)"
    txt2, n = re.subn(pattern2, "\n" + NEW_BLOCK, txt, count=1)
    if n != 1:
        raise SystemExit("Kunne ikke erstatte reports_pi-blokken (uventet match-count).")

    TARGET.write_text(txt2, encoding="utf-8")
    print("OK: Oppdatert reports_pi til /reports/{year|r12|quarter}.")
    print("Neste: python .\\tools\\borsdata_freeze.py -h  (og så kjør --only reports_pi)")

if __name__ == "__main__":
    main()
