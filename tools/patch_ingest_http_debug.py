# tools/patch_ingest_http_debug.py
from __future__ import annotations

import re
import shutil
from pathlib import Path
from datetime import datetime

TARGET = Path("src/ingest_fundamentals_history.py")
MARKER_START = "# --- DEBUG_HTTP PATCH START ---"
MARKER_END = "# --- DEBUG_HTTP PATCH END ---"

DEBUG_BLOCK = f"""{MARKER_START}
from collections import Counter
import os

DEBUG_HTTP = os.getenv("DEBUG_HTTP", "0") == "1"

def _init_http_debug():
    return Counter(), [], []

def _safe_url(u: str) -> str:
    # Fjern authKey/apikey fra URL i logger
    for k in ("authKey=", "apikey=", "apiKey="):
        if k in u:
            return u.split(k)[0]
    return u

def _record_http(status_counts, error_samples, ok_samples, r):
    \"\"\"Registrer statuskode og (begrensede) samples. Returnerer True hvis 200, ellers False.\"\"\"
    sc = getattr(r, "status_code", None)
    status_counts[sc] += 1

    # Non-200: ta en liten tekst-sample (maks 5)
    if sc != 200:
        if DEBUG_HTTP and len(error_samples) < 5:
            try:
                txt = getattr(r, "text", "")
            except Exception:
                txt = ""
            error_samples.append({{
                "status": sc,
                "url": _safe_url(str(getattr(r, "url", ""))),
                "text_head": (txt[:300].replace("\\n", " ") if isinstance(txt, str) else "")
            }})
        return False

    # 200: sample JSON-type/keys (maks 3)
    if DEBUG_HTTP and len(ok_samples) < 3:
        try:
            j = r.json()
            if isinstance(j, dict):
                ok_samples.append({{"type": "dict", "keys": list(j.keys())[:25]}})
            elif isinstance(j, list):
                ok_samples.append({{"type": "list", "len": len(j)}})
            else:
                ok_samples.append({{"type": str(type(j))}})
        except Exception as e:
            ok_samples.append({{"type": "json_error", "err": str(e)[:200]}})
    return True

def _attach_http_debug_to_meta(meta: dict, status_counts, error_samples, ok_samples):
    meta["http_status_counts"] = dict(status_counts)
    meta["http_error_samples"] = error_samples
    meta["http_ok_samples"] = ok_samples
    return meta
{MARKER_END}
"""

def _insert_after_imports(text: str, block: str) -> str:
    lines = text.splitlines(True)
    # finn "import-seksjonen" i toppen
    last_import_idx = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            last_import_idx = i
            continue
        # tillat tomme linjer og docstring etter imports? vi stopper ved første "ordentlige" kode
        if last_import_idx >= 0 and s != "":
            break
    if last_import_idx == -1:
        return block + "\n" + text
    insert_at = last_import_idx + 1
    lines.insert(insert_at, "\n" + block + "\n")
    return "".join(lines)

def _ensure_debug_block(text: str) -> str:
    if MARKER_START in text:
        return text
    return _insert_after_imports(text, DEBUG_BLOCK)

def _ensure_main_init(text: str) -> str:
    # Legg status_counts/... init i main() dersom main finnes
    m = re.search(r"(?m)^(?P<indent>[ \t]*)def\s+main\s*\(.*\)\s*:\s*$", text)
    if not m:
        return text
    indent = m.group("indent") + " " * 4
    # Finn linjen etter def main(...):
    lines = text.splitlines(True)
    # Finn main-signatur-linjen indeks
    main_line_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^[ \t]*def\s+main\s*\(.*\)\s*:\s*$", line):
            main_line_idx = i
            break
    if main_line_idx is None:
        return text
    # Ikke dobbel-injiser
    for j in range(main_line_idx + 1, min(main_line_idx + 15, len(lines))):
        if "_init_http_debug()" in lines[j]:
            return text
    inject = (
        f"{indent}# init http debug counters (autopatch)\n"
        f"{indent}status_counts, error_samples, ok_samples = _init_http_debug()\n\n"
    )
    lines.insert(main_line_idx + 1, inject)
    return "".join(lines)

def _instrument_requests_get(text: str) -> tuple[str, int]:
    # Instrumenter "resp = requests.get(...)" og "resp = httpx.get(...)".
    # Vi legger inn _record_http(...) rett etter linjen.
    lines = text.splitlines(True)
    out = []
    injected = 0

    pat = re.compile(r"^([ \t]*)(\w+)\s*=\s*(requests|httpx)\.get\s*\(")
    for i, line in enumerate(lines):
        out.append(line)
        m = pat.match(line)
        if not m:
            continue
        indent, varname = m.group(1), m.group(2)

        # Hvis neste ikke-tomme linje allerede er _record_http(...), skip
        k = i + 1
        while k < len(lines) and lines[k].strip() == "":
            k += 1
        if k < len(lines) and "_record_http(" in lines[k]:
            continue

        out.append(f"{indent}# http debug (autopatch)\n")
        out.append(f"{indent}_record_http(status_counts, error_samples, ok_samples, {varname})\n")
        injected += 1

    return "".join(out), injected

def _attach_meta_and_failfast(text: str) -> str:
    # Før json.dump(meta, ...) -> legg inn _attach_http_debug_to_meta(meta,...)
    # Etter json.dump -> fail-fast hvis rows==0
    lines = text.splitlines(True)
    out = []
    dump_pat = re.compile(r"^([ \t]*)json\.dump\(\s*(\w+)\s*,")
    already_attach = "http_status_counts" in text or "_attach_http_debug_to_meta" in text

    for i, line in enumerate(lines):
        m = dump_pat.match(line)
        if m and not already_attach:
            indent, meta_var = m.group(1), m.group(2)
            out.append(f"{indent}# attach http debug to meta (autopatch)\n")
            out.append(f"{indent}_attach_http_debug_to_meta({meta_var}, status_counts, error_samples, ok_samples)\n")
            out.append(line)
            # fail-fast etter dump (neste linjer)
            out.append(f"{indent}# fail-fast if empty (autopatch)\n")
            out.append(f"{indent}if {meta_var}.get('rows', 0) == 0:\n")
            out.append(f"{indent}    raise RuntimeError(f\"Ingest ga 0 rader. http_status_counts={{{meta_var}.get('http_status_counts')}}. Se ingest_meta.json for samples.\")\n")
            already_attach = True
            continue

        out.append(line)

    return "".join(out)

def main():
    if not TARGET.exists():
        raise SystemExit(f"Fant ikke {TARGET} (kjør fra repo-root).")

    text = TARGET.read_text(encoding="utf-8", errors="ignore")
    orig = text

    # Backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = TARGET.with_suffix(TARGET.suffix + f".bak_{ts}")
    shutil.copyfile(TARGET, bak)

    # Apply patches
    text = _ensure_debug_block(text)
    text = _ensure_main_init(text)
    text, n_injected = _instrument_requests_get(text)
    text = _attach_meta_and_failfast(text)

    if text == orig:
        print("Ingen endringer gjort (patch allerede til stede eller mønster ikke funnet).")
        print(f"Backup ligger her: {bak}")
        return

    TARGET.write_text(text, encoding="utf-8")
    print(f"Patchet: {TARGET}")
    print(f"Backup : {bak}")
    print(f"Injiserte _record_http etter requests/httpx.get: {n_injected} steder")

if __name__ == "__main__":
    main()
