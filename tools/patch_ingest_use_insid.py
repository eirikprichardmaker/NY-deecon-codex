from __future__ import annotations
from pathlib import Path
import re
import sys
import py_compile

FILE = Path("src/ingest_fundamentals_history.py")

PATCH_BLOCK = [
    '# Preferer ins_id hvis tickers.csv har det (da slipper vi match-problemer)',
    'if "ins_id" in tickers_df.columns:',
    '    merged = tickers_df.copy()',
    '    merged["ins_id"] = pd.to_numeric(merged["ins_id"], errors="coerce")',
    '    merged = merged.dropna(subset=["ins_id"]).copy()',
    '    merged["ins_id"] = merged["ins_id"].astype(int)',
    'else:',
    '    instruments_map = _fetch_instruments_map(authkey)',
    '    merged = tickers_df.merge(instruments_map, on="ticker", how="left")',
]

def leading_spaces(s: str) -> int:
    return len(s) - len(s.lstrip(" "))

def indent_block(block: list[str], base: int) -> list[str]:
    out = []
    for line in block:
        if line.strip() == "":
            out.append("")
        else:
            out.append((" " * base) + line)
    return out

def find_after(lines: list[str], pattern: str, start: int = 0) -> int | None:
    rx = re.compile(pattern)
    for i in range(start, len(lines)):
        if rx.search(lines[i]):
            return i
    return None

def replace_region(lines: list[str], start: int, end: int, repl: list[str]) -> list[str]:
    return lines[:start] + repl + lines[end:]

def main() -> int:
    if not FILE.exists():
        print(f"ERROR: Fant ikke {FILE}")
        return 1

    text = FILE.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Finn hvor tickers_df leses (for å patche riktig sted i main)
    tickers_i = find_after(lines, r"\btickers_df\s*=\s*_read_tickers_csv\(", 0)
    if tickers_i is None:
        print("ERROR: Fant ikke 'tickers_df = _read_tickers_csv(...)' i fila.")
        return 1

    # Strategi A: Finn eksisterende if-blokk som starter med if "ins_id" ...
    if_i = find_after(lines, r'^\s*if\s+"ins_id"\s+in\s+tickers_df\.columns\s*:', tickers_i)
    if if_i is not None:
        base_indent = leading_spaces(lines[if_i])
        j = if_i + 1
        # gå til første linje som er på samme indent-nivå (base_indent) og ikke er else/elif/except/finally,
        # etter at vi har passert if/else-blokken
        while j < len(lines):
            s = lines[j]
            if s.strip() == "":
                j += 1
                continue
            ind = leading_spaces(s)
            if ind <= base_indent and not s.lstrip().startswith(("else:", "elif ", "except", "finally")):
                break
            j += 1

        repl = indent_block(PATCH_BLOCK, base_indent)
        new_lines = replace_region(lines, if_i, j, repl)
        FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        py_compile.compile(str(FILE), doraise=True)
        print("OK: Patchet eksisterende if/else-blokk (ins_id) og py_compile OK.")
        return 0

    # Strategi B: Finn klassisk to-linjers mapping med instruments_map + merge
    inst_i = find_after(lines, r"^\s*instruments_map\s*=\s*_fetch_instruments_map\(\s*authkey\s*\)\s*$", tickers_i)
    if inst_i is None:
        print("ERROR: Fant verken ins_id-ifblokk eller instruments_map-linje å erstatte.")
        return 1

    base_indent = leading_spaces(lines[inst_i])

    # forvent merge-linje rett etterpå (tolerer litt whitespace)
    merge_i = inst_i + 1
    if merge_i >= len(lines) or "merged" not in lines[merge_i]:
        print("ERROR: Fant instruments_map, men ikke forventet merge-linje rett etter.")
        return 1

    repl = indent_block(PATCH_BLOCK, base_indent)
    new_lines = replace_region(lines, inst_i, merge_i + 1, repl)
    FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    py_compile.compile(str(FILE), doraise=True)
    print("OK: Erstattet instruments_map+merge med ins_id-blokk og py_compile OK.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
