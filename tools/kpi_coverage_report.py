from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import yaml

KEY_RE = re.compile(r"kpi\|kpi=(\d+)\|period=([^|]+)\|vt=([^|]+)\|batch=(\d+)\|n=(\d+)")


def parse_ids(raw: str) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for part in str(raw).split(","):
        s = part.strip()
        if not s:
            continue
        v = int(s)
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def load_set_from_yaml(path: Path, set_name: str) -> list[int]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    sets = (obj or {}).get("sets") or {}
    if set_name not in sets:
        raise ValueError(f"Set not found: {set_name}. Available: {sorted(sets.keys())}")
    ids = ((sets[set_name] or {}).get("ids")) or []
    return parse_ids(",".join(str(x) for x in ids))


def parse_state(path: Path) -> dict[tuple[str, str, int], int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    done = obj.get("done") or {}
    out: dict[tuple[str, str, int], int] = {}
    for key in done.keys():
        m = KEY_RE.search(str(key))
        if not m:
            continue
        kpi = int(m.group(1))
        period = str(m.group(2))
        value_type = str(m.group(3))
        t = (period, value_type, kpi)
        out[t] = out.get(t, 0) + 1
    return out


def expected_batches(ids_csv: Path, batch_size: int) -> int:
    lines = ids_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = max(0, len(lines) - 1)  # header
    return int(math.ceil(n / float(max(1, batch_size))))


def pct(done: int, expected: int) -> float:
    if expected <= 0:
        return 0.0
    return 100.0 * float(done) / float(expected)


def render_table(rows: Iterable[tuple[int, str, str, int, int]]) -> str:
    lines = []
    lines.append("kpi_id,period,value_type,batches_done,batches_expected,coverage_pct")
    for kpi_id, period, value_type, done, exp in rows:
        lines.append(f"{kpi_id},{period},{value_type},{done},{exp},{pct(done, exp):.2f}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Report KPI history coverage by reading freeze state.json")
    p.add_argument("--state-path", required=True)
    p.add_argument("--ids", default="", help="Comma-separated KPI IDs")
    p.add_argument("--set-file", default="config/kpi_candidate_sets.yaml")
    p.add_argument("--set-name", default="", help="Set name from --set-file")
    p.add_argument("--periods", default="year,r12")
    p.add_argument("--value-type", default="mean")
    p.add_argument("--ids-csv", default="", help="Optional ids csv used for freeze (for expected batch count)")
    p.add_argument("--batch-size", type=int, default=50)
    args = p.parse_args()

    state_path = Path(args.state_path)
    if not state_path.exists():
        raise SystemExit(f"Missing state file: {state_path}")

    ids: list[int] = []
    if str(args.set_name).strip():
        ids = load_set_from_yaml(Path(args.set_file), str(args.set_name).strip())
    elif str(args.ids).strip():
        ids = parse_ids(args.ids)
    else:
        raise SystemExit("Provide either --ids or --set-name.")

    periods = [x.strip() for x in str(args.periods).split(",") if x.strip()]
    value_type = str(args.value_type).strip()

    counts = parse_state(state_path)
    expected = 0
    if str(args.ids_csv).strip():
        expected = expected_batches(Path(args.ids_csv), int(args.batch_size))

    rows: list[tuple[int, str, str, int, int]] = []
    for kpi_id in ids:
        for period in periods:
            done = counts.get((period, value_type, int(kpi_id)), 0)
            exp = expected if expected > 0 else done
            rows.append((int(kpi_id), period, value_type, int(done), int(exp)))

    print(render_table(rows))


if __name__ == "__main__":
    main()
