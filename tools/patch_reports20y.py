import re
from pathlib import Path
from datetime import datetime

TARGET = Path(r"tools/borsdata_freeze.py")

SNIP_FIELDS = "    do_reports_pi: bool\n    reports_pi_maxcount: int\n"

SNIP_ARGS = """ap.add_argument(
        "--reports-per-instrument",
        action="store_true",
        help="Download reports per instrument using /v1/instruments/{id}/reports with &maxcount (needed for true 20y).",
    )
    ap.add_argument(
        "--reports-pi-maxcount",
        type=int,
        default=20,
        help="maxcount for per-instrument reports (20y year reports / 40 interim).",
    )
"""

SNIP_BUILD_FLAG = '    do_reports_pi = bool(args.reports_per_instrument) and (not args.only or "reports_pi" in args.only)\n\n'

SNIP_CFG_ARGS = "        do_reports_pi=do_reports_pi,\n        reports_pi_maxcount=args.reports_pi_maxcount,\n"

SNIP_RUN_BLOCK = """    # ------------------ Reports per instrument (20y via &maxcount) ------------------
    if cfg.do_reports_pi:
        out_raw = raw_dir / "reports_per_instrument"
        out_norm = norm_dir / "reports_per_instrument.jsonl.gz"
        ensure_dir(out_raw)

        for ins_id in ins_ids:
            key = f"reports_pi|ins_id={ins_id}|original={cfg.reports_original}|maxcount={cfg.reports_pi_maxcount}"
            if state.is_done(key):
                continue

            endpoint = f"{API_BASE}/instruments/{ins_id}/reports"
            params = {
                "authKey": auth,
                "original": cfg.reports_original,
                "maxcount": cfg.reports_pi_maxcount,
            }
            payload, meta = borsdata_get(session, endpoint, params, min_sleep=cfg.pace_s)

            raw_path = out_raw / f"ins_{ins_id}.json.gz"
            gz_json_write(raw_path, {"meta": meta, "params": params, "payload": payload})

            nrows = gz_jsonl_append(
                out_norm,
                [{
                    "dataset": "reports_per_instrument",
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

def backup_file(p: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    b = p.with_suffix(p.suffix + f".bak_{ts}")
    b.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    return b

def already_patched(txt: str) -> bool:
    return "--reports-per-instrument" in txt or "reports_per_instrument" in txt

def insert_after_line_containing(txt: str, needle: str, insert: str) -> str:
    idx = txt.find(needle)
    if idx < 0:
        raise ValueError(f"Fant ikke nøkkeltekst: {needle}")
    line_end = txt.find("\n", idx)
    if line_end < 0:
        return txt + "\n" + insert
    return txt[: line_end + 1] + insert + txt[line_end + 1 :]

def replace_only_help(txt: str) -> str:
    txt2 = txt
    txt2 = txt2.replace(
        "among reports,prices,prices_last,kpi,holdings",
        "among reports,reports_pi,prices,prices_last,kpi,holdings",
    )

    def _repl(m):
        s = m.group(1)
        if "reports_pi" in s:
            return m.group(0)
        if "reports," in s:
            s2 = s.replace("reports,", "reports,reports_pi,", 1)
        else:
            s2 = "reports,reports_pi," + s
        return f'help="Run only specific steps: comma-separated among {s2}"'

    txt2 = re.sub(r'help="Run only specific steps:\s*comma-separated among ([^"]+)"', _repl, txt2)
    return txt2

def main():
    if not TARGET.exists():
        raise SystemExit(f"Finner ikke {TARGET}. Kjør fra repo-root (G:\\Min disk\\Børsdata).")

    txt = TARGET.read_text(encoding="utf-8")
    if already_patched(txt):
        print("OK: Filen ser allerede patcha ut (fant reports-per-instrument/reports_per_instrument). Ingen endring.")
        return

    backup = backup_file(TARGET)
    print(f"Backup laget: {backup}")

    changed = 0

    # 1) FreezeConfig fields: add after reports_maxcount: int
    if "reports_pi_maxcount" not in txt:
        m = re.search(r"(^\s*reports_maxcount:\s*int\s*$\n)", txt, flags=re.M)
        if not m:
            raise SystemExit("Fant ikke linjen 'reports_maxcount: int' i FreezeConfig. Patch avbrutt.")
        insert_pos = m.end(1)
        txt = txt[:insert_pos] + SNIP_FIELDS + txt[insert_pos:]
        changed += 1

    # 2) parse_args: add CLI args after --reports-maxcount
    if "--reports-per-instrument" not in txt:
        txt = insert_after_line_containing(txt, 'ap.add_argument("--reports-maxcount"', SNIP_ARGS)
        changed += 1

    # 3) build_config: define do_reports_pi before return FreezeConfig(
    if "do_reports_pi = bool(args.reports_per_instrument)" not in txt:
        idx = txt.find("return FreezeConfig(")
        if idx < 0:
            raise SystemExit("Fant ikke 'return FreezeConfig(' i build_config. Patch avbrutt.")
        ls = txt.rfind("\n", 0, idx) + 1
        txt = txt[:ls] + SNIP_BUILD_FLAG + txt[ls:]
        changed += 1

    # 4) FreezeConfig(...) call: add cfg args after reports_maxcount=
    if "do_reports_pi=do_reports_pi" not in txt:
        txt = txt.replace(
            "        reports_maxcount=args.reports_maxcount,\n",
            "        reports_maxcount=args.reports_maxcount,\n" + SNIP_CFG_ARGS,
            1,
        )
        if "do_reports_pi=do_reports_pi" not in txt:
            raise SystemExit("Klarte ikke å injisere do_reports_pi/reports_pi_maxcount i FreezeConfig(...). Patch avbrutt.")
        changed += 1

    # 5) write_manifest: add steps entry and config entry (best-effort)
    if '"reports_per_instrument"' not in txt:
        txt = txt.replace(
            '            "reports": cfg.do_reports,\n',
            '            "reports": cfg.do_reports,\n            "reports_per_instrument": cfg.do_reports_pi,\n',
            1,
        )
        txt = txt.replace(
            '        "reports": {"original": cfg.reports_original, "maxcount": cfg.reports_maxcount},\n',
            '        "reports": {"original": cfg.reports_original, "maxcount": cfg.reports_maxcount},\n'
            '        "reports_per_instrument": {"original": cfg.reports_original, "maxcount": cfg.reports_pi_maxcount},\n',
            1,
        )
        changed += 1

    # 6) run_freeze: insert block before Stockprices history section
    if "Reports per instrument (20y via &maxcount)" not in txt:
        marker = "# ------------------ Stockprices history"
        if marker not in txt:
            raise SystemExit(f"Fant ikke marker '{marker}'. Patch avbrutt.")
        txt = txt.replace(marker, SNIP_RUN_BLOCK + marker, 1)
        changed += 1

    # 7) update --only help to include reports_pi
    txt2 = replace_only_help(txt)
    if txt2 != txt:
        txt = txt2
        changed += 1

    TARGET.write_text(txt, encoding="utf-8")
    print(f"OK: Patching ferdig. Antall endringsblokker: {changed}")
    print("Neste: Kjør `python .\\tools\\borsdata_freeze.py -h` og se at nye flag finnes.")

if __name__ == "__main__":
    main()
