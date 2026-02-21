from __future__ import annotations

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from src.common.utils import get_first_env

BASE_URL = "https://apiservice.borsdata.se/v1"
DEFAULT_DATASETS = ("prices", "reports_y", "reports_q", "reports_r12", "kpis", "dividends", "splits")
DEFAULT_MARKETS = ("NO", "SE", "DK", "FI")
MANIFEST_COLUMNS = [
    "dataset",
    "market",
    "ins_id",
    "file_path",
    "status",
    "rows",
    "fetched_at",
    "attempts",
    "last_error",
    "source",
]


class ApiError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class FreezeConfig:
    asof: str
    data_dir: Path
    markets: Tuple[str, ...]
    include_delisted: bool
    skip_existing: bool
    refresh_stale_days: int
    refetch_invalid_cache: bool
    force: bool
    datasets: Tuple[str, ...]
    timeout: int
    max_attempts: int


@dataclass(frozen=True)
class FreezeTask:
    dataset: str
    market: str
    ins_id: int
    file_path: Path
    file_path_rel: str


@dataclass(frozen=True)
class ParsedParquetPath:
    dataset: str
    market: str
    ins_id: Optional[int]
    extra_dims: Tuple[Tuple[str, str], ...]
    layout: str


class GlobalRateLimiter:
    """
    Keeps request rate under a safe threshold (default 90 calls / 10s).
    """

    def __init__(self, max_calls: int = 90, window_seconds: float = 10.0):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: deque[float] = deque()

    def wait_for_slot(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] >= self.window_seconds:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            sleep_for = self.window_seconds - (now - self._calls[0]) + 0.01
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            while self._calls and now - self._calls[0] >= self.window_seconds:
                self._calls.popleft()
        self._calls.append(time.monotonic())


class ApiClient:
    def __init__(self, auth_key: str, timeout: int, max_attempts: int, rate_limiter: GlobalRateLimiter):
        self.auth_key = auth_key
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.rate_limiter = rate_limiter
        self.session = requests.Session()

    def get_json(self, path: str, params: Dict[str, Any]) -> Tuple[Any, int]:
        merged_params = dict(params)
        merged_params["authKey"] = self.auth_key
        url = f"{BASE_URL}{path}"
        delay = 1.0
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            self.rate_limiter.wait_for_slot()
            try:
                resp = self.session.get(url, params=merged_params, timeout=self.timeout)
                if resp.status_code == 429:
                    sleep_s = _retry_after_seconds(resp.headers.get("Retry-After"), default=delay)
                    time.sleep(sleep_s + random.uniform(0.0, 0.4))
                    delay = min(delay * 2.0, 60.0)
                    last_error = ApiError("HTTP 429 Too Many Requests", status_code=429)
                    continue
                if 500 <= resp.status_code < 600:
                    time.sleep(delay + random.uniform(0.0, 0.4))
                    delay = min(delay * 2.0, 60.0)
                    last_error = ApiError(f"HTTP {resp.status_code}", status_code=resp.status_code)
                    continue
                if resp.status_code >= 400:
                    raise ApiError(f"HTTP {resp.status_code}: {resp.text[:200]}", status_code=resp.status_code)
                return resp.json(), attempt
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(delay + random.uniform(0.0, 0.4))
                delay = min(delay * 2.0, 60.0)
            except ValueError as exc:
                raise ApiError(f"Invalid JSON from {path}: {exc}") from exc
        raise ApiError(f"GET failed after {self.max_attempts} attempts: {path}: {last_error}")


def _retry_after_seconds(retry_after: Optional[str], default: float) -> float:
    if retry_after:
        value = retry_after.strip()
        if value.isdigit():
            return max(float(value), 0.1)
        try:
            dt = parsedate_to_datetime(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = (dt - datetime.now(timezone.utc)).total_seconds()
            return max(delta, 0.1)
        except Exception:
            pass
    return max(default, 0.1)


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    txt = str(v).strip().lower()
    if txt in {"1", "true", "t", "yes", "y"}:
        return True
    if txt in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def _parse_csv_list(raw: str) -> Tuple[str, ...]:
    items = [x.strip() for x in str(raw).split(",") if x.strip()]
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)


def _is_readable_non_empty_parquet(path: Path) -> Tuple[bool, int, str]:
    try:
        if not path.exists():
            return False, 0, "file does not exist"
        df = pd.read_parquet(path)
        rows = int(len(df))
        if rows <= 0:
            return False, 0, "empty parquet"
        return True, rows, ""
    except Exception as exc:
        return False, 0, str(exc)


def _is_stale(path: Path, refresh_stale_days: int) -> bool:
    if refresh_stale_days <= 0:
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) > timedelta(days=refresh_stale_days)


def _target_path_for(
    raw_dir: Path,
    dataset: str,
    market: str,
    ins_id: int,
    extra_dims: Optional[Dict[str, Any]] = None,
) -> Path:
    path = raw_dir / str(dataset)
    if extra_dims:
        for key, value in sorted(extra_dims.items()):
            path = path / f"{key}={value}"
    path = path / f"market={str(market).upper()}"
    return path / f"ins_id={int(ins_id)}.parquet"


def _to_rel_path(raw_dir: Path, path: Path) -> str:
    return str(path.relative_to(raw_dir)).replace("\\", "/")


def _parse_flat_contract(path: Path) -> Optional[ParsedParquetPath]:
    if len(path.parts) != 1:
        return None
    parts = path.stem.split("__")
    if len(parts) < 3 or not parts[-1].isdigit():
        return None
    return ParsedParquetPath(
        dataset=parts[0],
        market=parts[1].upper(),
        ins_id=int(parts[-1]),
        extra_dims=(),
        layout="flat",
    )


def _parse_partitioned_contract(path: Path) -> Optional[ParsedParquetPath]:
    parts = list(path.parts)
    if len(parts) < 3:
        return None
    ins_stem = Path(parts[-1]).stem
    if not ins_stem.startswith("ins_id="):
        return None
    ins_txt = ins_stem.split("=", 1)[1]
    if not ins_txt.isdigit():
        return None

    dataset = parts[0]
    market = ""
    extra_dims: List[Tuple[str, str]] = []
    for seg in parts[1:-1]:
        if "=" not in seg:
            continue
        key, value = seg.split("=", 1)
        if key == "market":
            market = value.upper()
        else:
            extra_dims.append((key, value))
    if not market:
        return None
    return ParsedParquetPath(
        dataset=dataset,
        market=market,
        ins_id=int(ins_txt),
        extra_dims=tuple(sorted(extra_dims)),
        layout="partitioned",
    )


def _parse_parquet_contract(path: Path) -> ParsedParquetPath:
    parsed = _parse_partitioned_contract(path)
    if parsed is None:
        parsed = _parse_flat_contract(path)
    if parsed is not None:
        return parsed

    dataset = path.parts[0] if len(path.parts) > 1 else path.stem
    return ParsedParquetPath(dataset=dataset, market="", ins_id=None, extra_dims=(), layout="other")


def _layout_priority(layout: str) -> int:
    if layout == "partitioned":
        return 2
    if layout == "flat":
        return 1
    return 0


def _normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame(columns=MANIFEST_COLUMNS)
    if df is None or df.empty:
        return base.copy()
    out = df.copy()
    for col in MANIFEST_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[MANIFEST_COLUMNS].copy()
    out["dataset"] = out["dataset"].fillna("").astype(str)
    out["market"] = out["market"].fillna("").astype(str).str.upper()
    out["file_path"] = out["file_path"].fillna("").astype(str)
    out["status"] = out["status"].fillna("missing").astype(str)
    out["source"] = out["source"].fillna("api").astype(str)
    out["rows"] = pd.to_numeric(out["rows"], errors="coerce").fillna(0).astype(int)
    out["attempts"] = pd.to_numeric(out["attempts"], errors="coerce").fillna(0).astype(int)
    out["last_error"] = out["last_error"].fillna("").astype(str)
    out["fetched_at"] = out["fetched_at"].fillna("").astype(str)
    out["ins_id"] = pd.to_numeric(out["ins_id"], errors="coerce").astype("Int64")
    return out


def _manifest_path(raw_dir: Path) -> Path:
    return raw_dir / "manifest.parquet"


def _bootstrap_manifest_if_missing(raw_dir: Path) -> pd.DataFrame:
    manifest_path = _manifest_path(raw_dir)
    if manifest_path.exists():
        return _normalize_manifest(pd.read_parquet(manifest_path))

    rows_by_key: Dict[Tuple[str, str, int, Tuple[Tuple[str, str], ...]], Tuple[int, str, Dict[str, Any]]] = {}
    passthrough_rows: List[Dict[str, Any]] = []
    for parquet_path in sorted(raw_dir.rglob("*.parquet")):
        if parquet_path.name.lower() == "manifest.parquet":
            continue
        rel_path = parquet_path.relative_to(raw_dir)
        parsed = _parse_parquet_contract(rel_path)
        ok, nrows, err = _is_readable_non_empty_parquet(parquet_path)
        fetched_at = datetime.fromtimestamp(parquet_path.stat().st_mtime, tz=timezone.utc).isoformat()
        row = {
            "dataset": parsed.dataset,
            "market": parsed.market,
            "ins_id": parsed.ins_id,
            "file_path": str(rel_path).replace("\\", "/"),
            "status": "cached" if ok else "missing",
            "rows": nrows,
            "fetched_at": fetched_at,
            "attempts": 0,
            "last_error": "" if ok else err,
            "source": "cached" if ok else "cache_invalid",
        }
        if parsed.ins_id is not None and parsed.market:
            key = (parsed.dataset, parsed.market, int(parsed.ins_id), parsed.extra_dims)
            rank = _layout_priority(parsed.layout)
            current = rows_by_key.get(key)
            if current is None or rank > current[0] or (rank == current[0] and row["file_path"] < current[1]):
                rows_by_key[key] = (rank, row["file_path"], row)
        else:
            passthrough_rows.append(row)
    rows = [x[2] for x in rows_by_key.values()] + passthrough_rows
    rows = sorted(rows, key=lambda r: (str(r["dataset"]), str(r["market"]), str(r["ins_id"]), str(r["file_path"])))
    manifest = _normalize_manifest(pd.DataFrame(rows, columns=MANIFEST_COLUMNS))
    manifest.to_parquet(manifest_path, index=False)
    return manifest


def _upsert_manifest_row(manifest: pd.DataFrame, row: Dict[str, Any]) -> pd.DataFrame:
    key_cols = ["dataset", "market", "ins_id", "file_path"]
    row_df = _normalize_manifest(pd.DataFrame([row], columns=MANIFEST_COLUMNS))
    if manifest.empty:
        return row_df
    manifest = _normalize_manifest(manifest)
    k = tuple(row_df.iloc[0][c] for c in key_cols)
    keep = []
    for _, r in manifest.iterrows():
        rk = tuple(r[c] for c in key_cols)
        keep.append(rk != k)
    kept = manifest.loc[keep].copy()
    return pd.concat([kept, row_df], ignore_index=True)


def _find_first_list(payload: Any) -> Optional[List[Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list):
                return v
        for v in payload.values():
            if isinstance(v, dict):
                nested = _find_first_list(v)
                if nested is not None:
                    return nested
    return None


def _payload_to_frame(dataset: str, market: str, ins_id: int, payload: Any, fetched_at: str) -> pd.DataFrame:
    data_list = _find_first_list(payload)
    if data_list and all(isinstance(x, dict) for x in data_list):
        df = pd.json_normalize(data_list, sep="_")
    elif isinstance(payload, dict):
        df = pd.json_normalize(payload, sep="_")
    else:
        df = pd.DataFrame([{"value": payload}])

    if df.empty:
        df = pd.DataFrame([{"_empty_payload": True}])

    df.insert(0, "dataset", dataset)
    df.insert(1, "market", market)
    df.insert(2, "ins_id", int(ins_id))
    df.insert(3, "fetched_at", fetched_at)
    return df


def _endpoint_candidates(dataset: str, asof: str, ins_id: int) -> Sequence[Tuple[str, Dict[str, Any]]]:
    if dataset == "prices":
        asof_date = datetime.strptime(asof, "%Y-%m-%d").date()
        from_date = (asof_date - timedelta(days=365 * 20)).isoformat()
        return (
            (f"/instruments/{ins_id}/stockprices", {"from": from_date, "to": asof}),
            (f"/instruments/{ins_id}/stockprices/last", {}),
        )
    if dataset == "reports_y":
        return ((f"/instruments/{ins_id}/reports/year", {"maxCount": 20}),)
    if dataset == "reports_q":
        return ((f"/instruments/{ins_id}/reports/quarter", {"maxCount": 80}),)
    if dataset == "reports_r12":
        return ((f"/instruments/{ins_id}/reports/r12", {"maxCount": 40}),)
    if dataset == "kpis":
        return (
            (f"/instruments/{ins_id}/kpis", {}),
            (f"/instruments/{ins_id}/kpi", {}),
        )
    if dataset == "dividends":
        return (
            (f"/instruments/{ins_id}/dividends", {}),
            (f"/instruments/{ins_id}/dividend", {}),
        )
    if dataset == "splits":
        return (
            (f"/instruments/{ins_id}/splits", {}),
            (f"/instruments/{ins_id}/split", {}),
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def _fetch_dataset_payload(client: ApiClient, dataset: str, asof: str, ins_id: int) -> Tuple[Any, int]:
    last_error: Optional[Exception] = None
    for path, params in _endpoint_candidates(dataset, asof, ins_id):
        try:
            payload, attempts = client.get_json(path, params)
            return payload, attempts
        except ApiError as exc:
            last_error = exc
            if exc.status_code == 404:
                continue
            raise
    raise ApiError(f"No endpoint succeeded for dataset={dataset}, ins_id={ins_id}: {last_error}", status_code=404)


def _market_from_instrument(item: Dict[str, Any]) -> Optional[str]:
    for key in ("market", "marketCode", "countryCode", "country", "exchange"):
        value = item.get(key)
        if value is None:
            continue
        txt = str(value).upper()
        if txt in {"NO", "SE", "DK", "FI"}:
            return txt
        if "NORWAY" in txt or "OSLO" in txt:
            return "NO"
        if "SWEDEN" in txt or "STOCKHOLM" in txt:
            return "SE"
        if "DENMARK" in txt or "COPENHAGEN" in txt:
            return "DK"
        if "FINLAND" in txt or "HELSINKI" in txt:
            return "FI"
    country_id = item.get("countryId")
    if country_id is not None:
        try:
            cid = int(country_id)
            return {1: "SE", 2: "NO", 3: "FI", 4: "DK"}.get(cid)
        except Exception:
            return None
    return None


def _is_delisted(item: Dict[str, Any]) -> bool:
    if "isDelisted" in item:
        try:
            return bool(item["isDelisted"])
        except Exception:
            return False
    if "isActive" in item:
        try:
            return not bool(item["isActive"])
        except Exception:
            return False
    if "isListed" in item:
        try:
            return not bool(item["isListed"])
        except Exception:
            return False
    return False


def _extract_instruments(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("instruments", "Instruments", "data", "Data", "items", "Items"):
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def _fetch_market_instruments(client: ApiClient, market: str, include_delisted: bool) -> List[int]:
    payload, _ = client.get_json(
        "/instruments",
        {"market": market, "includeDelisted": str(include_delisted).lower()},
    )
    rows = _extract_instruments(payload)
    out: List[int] = []
    seen = set()
    for item in rows:
        iid = item.get("insId") or item.get("ins_id") or item.get("id")
        if iid is None:
            continue
        try:
            ins_id = int(iid)
        except Exception:
            continue
        row_market = _market_from_instrument(item) or market
        if row_market != market:
            continue
        if not include_delisted and _is_delisted(item):
            continue
        if ins_id not in seen:
            seen.add(ins_id)
            out.append(ins_id)
    return out


def _load_all_markets_fallback(client: ApiClient, markets: Sequence[str], include_delisted: bool) -> Dict[str, List[int]]:
    payload, _ = client.get_json("/instruments", {"includeDelisted": str(include_delisted).lower()})
    rows = _extract_instruments(payload)
    out = {m: [] for m in markets}
    seen = {m: set() for m in markets}
    for item in rows:
        iid = item.get("insId") or item.get("ins_id") or item.get("id")
        if iid is None:
            continue
        try:
            ins_id = int(iid)
        except Exception:
            continue
        market = _market_from_instrument(item)
        if market not in out:
            continue
        if not include_delisted and _is_delisted(item):
            continue
        if ins_id not in seen[market]:
            seen[market].add(ins_id)
            out[market].append(ins_id)
    return out


def _preferred_task_paths_from_manifest(raw_dir: Path, manifest: pd.DataFrame) -> Dict[Tuple[str, str, int], Path]:
    if manifest is None or manifest.empty:
        return {}
    path_map: Dict[Tuple[str, str, int], Tuple[int, bool, str, Path]] = {}
    for _, row in manifest.iterrows():
        dataset = str(row.get("dataset", ""))
        market = str(row.get("market", "")).upper()
        ins_raw = row.get("ins_id")
        rel_str = str(row.get("file_path", "")).strip()
        if not dataset or not market or not rel_str or pd.isna(ins_raw):
            continue
        try:
            ins_id = int(ins_raw)
        except Exception:
            continue

        parsed = _parse_parquet_contract(Path(rel_str))
        extra_dims_empty = len(parsed.extra_dims) == 0
        rank = (
            _layout_priority(parsed.layout),
            1 if extra_dims_empty else 0,
            rel_str,
            raw_dir / Path(rel_str),
        )
        key = (dataset, market, ins_id)
        current = path_map.get(key)
        if current is None or rank[0] > current[0] or (rank[0] == current[0] and rank[1] > current[1]) or (
            rank[0] == current[0] and rank[1] == current[1] and rank[2] < current[2]
        ):
            path_map[key] = rank
    return {k: v[3] for k, v in path_map.items()}


def _build_tasks(
    raw_dir: Path,
    datasets: Sequence[str],
    by_market: Dict[str, List[int]],
    manifest: pd.DataFrame,
) -> List[FreezeTask]:
    preferred_paths = _preferred_task_paths_from_manifest(raw_dir, manifest)
    tasks: List[FreezeTask] = []
    for market in sorted(by_market.keys()):
        for ins_id in by_market[market]:
            for dataset in datasets:
                key = (dataset, market, int(ins_id))
                file_path = preferred_paths.get(key, _target_path_for(raw_dir, dataset, market, int(ins_id)))
                tasks.append(
                    FreezeTask(
                        dataset=dataset,
                        market=market,
                        ins_id=int(ins_id),
                        file_path=file_path,
                        file_path_rel=_to_rel_path(raw_dir, file_path),
                    )
                )
    return tasks


def _task_row_cached(task: FreezeTask, rows: int, fetched_at: str) -> Dict[str, Any]:
    return {
        "dataset": task.dataset,
        "market": task.market,
        "ins_id": task.ins_id,
        "file_path": task.file_path_rel,
        "status": "cached",
        "rows": int(rows),
        "fetched_at": fetched_at,
        "attempts": 0,
        "last_error": "",
        "source": "cached",
    }


def _run_task(task: FreezeTask, cfg: FreezeConfig, client: ApiClient, now_utc: str) -> Dict[str, Any]:
    path = task.file_path
    path.parent.mkdir(parents=True, exist_ok=True)

    ok_cached, rows_cached, cache_err = _is_readable_non_empty_parquet(path)
    fresh_cached = ok_cached and (not _is_stale(path, cfg.refresh_stale_days))
    use_cached = (not cfg.force) and cfg.skip_existing and fresh_cached
    if use_cached:
        fetched_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        return _task_row_cached(task, rows_cached, fetched_at)

    has_invalid_cache = path.exists() and not ok_cached
    if has_invalid_cache and cfg.skip_existing and (not cfg.force) and (not cfg.refetch_invalid_cache):
        fetched_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        return {
            "dataset": task.dataset,
            "market": task.market,
            "ins_id": task.ins_id,
            "file_path": task.file_path_rel,
            "status": "missing",
            "rows": 0,
            "fetched_at": fetched_at,
            "attempts": 0,
            "last_error": cache_err,
            "source": "cache_invalid",
        }

    try:
        payload, attempts = _fetch_dataset_payload(client, task.dataset, cfg.asof, task.ins_id)
        df = _payload_to_frame(task.dataset, task.market, task.ins_id, payload, fetched_at=now_utc)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        return {
            "dataset": task.dataset,
            "market": task.market,
            "ins_id": task.ins_id,
            "file_path": task.file_path_rel,
            "status": "ok",
            "rows": int(len(df)),
            "fetched_at": now_utc,
            "attempts": int(attempts),
            "last_error": "",
            "source": "api",
        }
    except Exception as exc:
        status = "missing" if isinstance(exc, ApiError) and exc.status_code == 404 else "failed"
        return {
            "dataset": task.dataset,
            "market": task.market,
            "ins_id": task.ins_id,
            "file_path": task.file_path_rel,
            "status": status,
            "rows": 0,
            "fetched_at": now_utc,
            "attempts": cfg.max_attempts,
            "last_error": str(exc),
            "source": "api",
        }


def _build_coverage(results: pd.DataFrame, by_market: Dict[str, List[int]]) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "market",
                "expected",
                "cached",
                "ok",
                "failed",
                "missing",
                "completed",
                "coverage_ratio",
                "rows_total",
            ]
        )

    grouped = (
        results.groupby(["dataset", "market", "status"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    rows = []
    for (dataset, market), chunk in grouped.groupby(["dataset", "market"], dropna=False):
        expected = int(len(by_market.get(market, [])))
        counts = {str(r["status"]): int(r["n"]) for _, r in chunk.iterrows()}
        cached = counts.get("cached", 0)
        ok = counts.get("ok", 0)
        failed = counts.get("failed", 0)
        missing = counts.get("missing", 0)
        completed = cached + ok
        ratio = (completed / expected) if expected > 0 else 0.0
        rows_total = int(results[(results["dataset"] == dataset) & (results["market"] == market)]["rows"].sum())
        rows.append(
            {
                "dataset": dataset,
                "market": market,
                "expected": expected,
                "cached": cached,
                "ok": ok,
                "failed": failed,
                "missing": missing,
                "completed": completed,
                "coverage_ratio": float(round(ratio, 6)),
                "rows_total": rows_total,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["dataset", "market"]).reset_index(drop=True)


def _write_coverage_md(
    path: Path,
    asof: str,
    coverage: pd.DataFrame,
    markets: Sequence[str],
    include_delisted: bool,
) -> None:
    lines = [f"# Freeze Coverage ({asof})", ""]
    lines.append("## Expected calculation")
    lines.append(
        f"Instrument scope bygges per marked fra `--markets={','.join(markets)}` via `/instruments` med "
        f"`includeDelisted={str(include_delisted).lower()}`."
    )
    lines.append(
        "Expected per dataset/market bruker strategi (i): expected = antall instrumenter i scope for markedet, "
        "uansett om dataset faktisk finnes i API for hvert instrument."
    )
    lines.append("")
    lines.append("## Missing semantics")
    lines.append(
        "Status `missing` betyr at dataset ikke ble hentet ferdig for instrumentet: API utilgjengelig (typisk 404), "
        "eller cache finnes men er ugyldig (tom/ulesbar parquet)."
    )
    lines.append(
        "Status `failed` betyr at et annet henteproblem oppstod (f.eks. nett/server-feil etter retries). "
        "Status `cached` og `ok` regnes som completed."
    )
    lines.append("Coverage-formel: `coverage_ratio = completed / expected`, der `completed = cached + ok`.")
    lines.append("")
    if coverage.empty:
        lines.append("Ingen tasks ble kjort.")
    else:
        lines.append("| dataset | market | expected | cached | ok | failed | missing | completed | coverage_ratio | rows_total |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in coverage.iterrows():
            lines.append(
                f"| {row['dataset']} | {row['market']} | {int(row['expected'])} | {int(row['cached'])} | "
                f"{int(row['ok'])} | {int(row['failed'])} | {int(row['missing'])} | "
                f"{int(row['completed'])} | {float(row['coverage_ratio']):.3f} | {int(row['rows_total'])} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_freeze(cfg: FreezeConfig, client: Optional[ApiClient] = None) -> Dict[str, Any]:
    raw_dir = cfg.data_dir / "raw" / cfg.asof
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = _bootstrap_manifest_if_missing(raw_dir)
    manifest_path = _manifest_path(raw_dir)

    auth_key = get_first_env(["BORSDATA_AUTHKEY", "BORSDATA_API_KEY", "BORSDATA_KEY"])
    if not auth_key:
        raise RuntimeError("Mangler Borsdata API-key. Sett env var BORSDATA_AUTHKEY (evt BORSDATA_API_KEY).")

    if client is None:
        client = ApiClient(
            auth_key=auth_key,
            timeout=cfg.timeout,
            max_attempts=cfg.max_attempts,
            rate_limiter=GlobalRateLimiter(max_calls=90, window_seconds=10.0),
        )

    by_market: Dict[str, List[int]] = {}
    failed_markets = []
    for market in cfg.markets:
        try:
            by_market[market] = _fetch_market_instruments(client, market, cfg.include_delisted)
        except Exception:
            failed_markets.append(market)
            by_market[market] = []
    if failed_markets:
        fallback = _load_all_markets_fallback(client, cfg.markets, cfg.include_delisted)
        for market in failed_markets:
            by_market[market] = fallback.get(market, [])

    tasks = _build_tasks(raw_dir, cfg.datasets, by_market, manifest)
    now_utc = datetime.now(timezone.utc).isoformat()
    result_rows: List[Dict[str, Any]] = []
    for task in tasks:
        row = _run_task(task, cfg, client, now_utc=now_utc)
        result_rows.append(row)
        manifest = _upsert_manifest_row(manifest, row)

    manifest = _normalize_manifest(manifest).sort_values(["dataset", "market", "ins_id", "file_path"]).reset_index(drop=True)
    manifest.to_parquet(manifest_path, index=False)

    results_df = _normalize_manifest(pd.DataFrame(result_rows, columns=MANIFEST_COLUMNS))
    coverage = _build_coverage(results_df, by_market)
    run_dir = Path("runs") / cfg.asof
    run_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_parquet(run_dir / "coverage.parquet", index=False)
    _write_coverage_md(
        run_dir / "freeze_coverage.md",
        cfg.asof,
        coverage,
        markets=cfg.markets,
        include_delisted=cfg.include_delisted,
    )

    summary = {
        "asof": cfg.asof,
        "tasks": int(len(tasks)),
        "cached": int((results_df["status"] == "cached").sum()) if not results_df.empty else 0,
        "ok": int((results_df["status"] == "ok").sum()) if not results_df.empty else 0,
        "failed": int((results_df["status"] == "failed").sum()) if not results_df.empty else 0,
        "missing": int((results_df["status"] == "missing").sum()) if not results_df.empty else 0,
        "manifest_path": str(manifest_path),
        "coverage_parquet": str((run_dir / "coverage.parquet").as_posix()),
        "coverage_md": str((run_dir / "freeze_coverage.md").as_posix()),
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incremental MAX FREEZE with manifest and cache-aware fetch.")
    p.add_argument("--asof", required=True, help="YYYY-MM-DD")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--markets", default="NO,SE,DK,FI")
    p.add_argument("--include-delisted", default="true")
    p.add_argument(
        "--skip-existing",
        default="true",
        help="Skip valid cache entries. Invalid cache is treated as incomplete and refetched only with --force or --refetch-invalid-cache.",
    )
    p.add_argument("--refresh-stale-days", type=int, default=0)
    p.add_argument(
        "--refetch-invalid-cache",
        default="false",
        help="When true, refetch files that exist but are unreadable/empty parquet even if --skip-existing is true.",
    )
    p.add_argument("--force", default="false")
    p.add_argument("--datasets", default="prices,reports_y,reports_q,reports_r12,kpis,dividends,splits")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--max-attempts", type=int, default=8)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> FreezeConfig:
    datasets = _parse_csv_list(args.datasets)
    unsupported = [d for d in datasets if d not in set(DEFAULT_DATASETS)]
    if unsupported:
        raise ValueError(f"Unsupported datasets: {unsupported}. Allowed: {', '.join(DEFAULT_DATASETS)}")
    markets = tuple(x.upper() for x in _parse_csv_list(args.markets))
    unsupported_mk = [m for m in markets if m not in set(DEFAULT_MARKETS)]
    if unsupported_mk:
        raise ValueError(f"Unsupported markets: {unsupported_mk}. Allowed: {', '.join(DEFAULT_MARKETS)}")
    return FreezeConfig(
        asof=args.asof,
        data_dir=Path(args.data_dir),
        markets=markets,
        include_delisted=_parse_bool(args.include_delisted),
        skip_existing=_parse_bool(args.skip_existing),
        refresh_stale_days=max(0, int(args.refresh_stale_days)),
        refetch_invalid_cache=_parse_bool(args.refetch_invalid_cache),
        force=_parse_bool(args.force),
        datasets=datasets,
        timeout=max(5, int(args.timeout)),
        max_attempts=max(1, int(args.max_attempts)),
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    summary = run_freeze(cfg)
    print(f"OK: manifest -> {summary['manifest_path']}")
    print(f"OK: coverage -> {summary['coverage_parquet']}")
    print(f"OK: report   -> {summary['coverage_md']}")
    print(
        "OK: tasks={tasks} cached={cached} ok={ok} failed={failed} missing={missing}".format(
            tasks=summary["tasks"],
            cached=summary["cached"],
            ok=summary["ok"],
            failed=summary["failed"],
            missing=summary["missing"],
        )
    )


if __name__ == "__main__":
    main()
