from pathlib import Path
from datetime import datetime

P = Path(r"tools/borsdata_freeze.py")

def backup(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    b = path.with_suffix(path.suffix + f".bak_indentfix_{ts}")
    b.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return b

def main():
    if not P.exists():
        raise SystemExit(f"Finner ikke {P}. Kjør fra repo-root.")

    txt = P.read_text(encoding="utf-8").splitlines(True)  # keep newlines
    b = backup(P)
    print(f"Backup laget: {b}")

    out = []
    in_parse_args = False

    for line in txt:
        # Enter/exit def parse_args() block
        if line.startswith("def parse_args"):
            in_parse_args = True
            out.append(line)
            continue
        if in_parse_args and line.startswith("def ") and not line.startswith("def parse_args"):
            in_parse_args = False

        if in_parse_args:
            stripped = line.lstrip()
            # Fix: any ap.add_argument at column 0 inside parse_args should be indented 4 spaces
            if stripped.startswith("ap.add_argument(") and not line.startswith("    "):
                line = "    " + stripped

        out.append(line)

    P.write_text("".join(out), encoding="utf-8")
    print("OK: Innrykk i parse_args() er rettet. Test med: python .\\tools\\borsdata_freeze.py -h")

if __name__ == "__main__":
    main()
