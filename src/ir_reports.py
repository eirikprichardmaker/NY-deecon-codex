from __future__ import annotations

import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.robotparser import RobotFileParser

import numpy as np
import pandas as pd

from src.common.config import resolve_paths

INDEX_COLS = [
    "ticker",
    "url",
    "report_date",
    "period",
    "source",
    "status",
    "error_code",
    "raw_path",
    "content_type",
    "is_cached",
    "bytes",
    "sha256",
]

FACT_COLS = [
    "ticker",
    "url",
    "report_date",
    "period",
    "source",
    "status",
    "error_code",
    "fact_revenue",
    "fact_ebit",
    "fact_eps",
    "text_len",
]


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _safe_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).strip()


def _safe_file_name(url: str, content_type: str | None = None) -> str:
    p = urllib.parse.urlparse(url)
    name = Path(p.path).name or "report"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    ext = Path(name).suffix.lower()
    if not ext:
        if content_type and "pdf" in content_type.lower():
            ext = ".pdf"
        else:
            ext = ".html"
        name = name + ext
    return name


def _parse_report_date(value: str, fallback: str) -> str:
    v = _safe_text(value)
    if not v:
        return fallback
    dt = pd.to_datetime(v, errors="coerce")
    if pd.isna(dt):
        return fallback
    return dt.date().isoformat()


def _read_mapping_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "url", "source", "period", "report_date"])
    df = pd.read_csv(path)
    for c in ["ticker", "url", "source", "period", "report_date"]:
        if c not in df.columns:
            df[c] = ""
    return df[["ticker", "url", "source", "period", "report_date"]].copy()


def _empty_index() -> pd.DataFrame:
    return pd.DataFrame(columns=INDEX_COLS)


def _empty_facts() -> pd.DataFrame:
    return pd.DataFrame(columns=FACT_COLS)


def _load_existing(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].copy()


def _write_parquet(df: pd.DataFrame, path: Path, sort_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].where(out[c].notna(), "")
    if sort_cols:
        out = out.sort_values(sort_cols, na_position="last", kind="mergesort")
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower().strip()


def _can_fetch(url: str, user_agent: str, cache: dict[str, RobotFileParser], timeout: int) -> tuple[bool, str]:
    p = urllib.parse.urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    robots_url = f"{base}/robots.txt"

    rp = cache.get(base)
    if rp is None:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            with urllib.request.urlopen(robots_url, timeout=timeout) as resp:
                txt = resp.read().decode("utf-8", errors="ignore")
            rp.parse(txt.splitlines())
        except Exception:
            return False, "ROBOTS_UNAVAILABLE"
        cache[base] = rp

    if not rp.can_fetch(user_agent, url):
        return False, "ROBOTS_DISALLOW"
    return True, ""


def _rate_limit(host: str, last_request_ts: dict[str, float], delay_sec: float) -> None:
    if delay_sec <= 0:
        return
    now = time.monotonic()
    prev = last_request_ts.get(host)
    if prev is not None:
        wait = delay_sec - (now - prev)
        if wait > 0:
            time.sleep(wait)
    last_request_ts[host] = time.monotonic()


def _fetch_url_bytes(url: str, user_agent: str, timeout: int) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        b = resp.read()
        content_type = resp.headers.get("Content-Type", "")
    return b, content_type


def _extract_html_text(b: bytes) -> str:
    html = b.decode("utf-8", errors="ignore")
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    txt = re.sub(r"(?is)<[^>]+>", " ", html)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _extract_pdf_text(b: bytes) -> str:
    try:
        from pypdf import PdfReader  # optional dependency
    except Exception:
        return ""

    try:
        reader = PdfReader(io.BytesIO(b))
        chunks: list[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            chunks.append(t)
        txt = "\n".join(chunks)
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt
    except Exception:
        return ""


def _extract_first_number_after(text: str, keywords: list[str]) -> float:
    if not text:
        return np.nan
    pat = r"(?i)(?:" + "|".join(re.escape(k) for k in keywords) + r")[^0-9\-]{0,20}([\-]?[0-9][0-9\s.,]{0,20})"
    m = re.search(pat, text)
    if not m:
        return np.nan
    raw = m.group(1).strip().replace(" ", "")
    # normalize decimal separator
    if raw.count(",") == 1 and raw.count(".") == 0:
        raw = raw.replace(",", ".")
    else:
        raw = raw.replace(",", "")
    try:
        return float(raw)
    except Exception:
        return np.nan


def _extract_facts(text: str) -> dict:
    return {
        "fact_revenue": _extract_first_number_after(text, ["revenue", "sales", "omsetning", "omsättning"]),
        "fact_ebit": _extract_first_number_after(text, ["ebit", "operating profit", "driftsresultat"]),
        "fact_eps": _extract_first_number_after(text, ["eps", "earnings per share", "resultat per aksje", "resultat per aktie"]),
        "text_len": int(len(text or "")),
    }


def _index_row_base(ticker: str, url: str, report_date: str, period: str, source: str) -> dict:
    return {
        "ticker": ticker,
        "url": url,
        "report_date": report_date,
        "period": period,
        "source": source,
        "status": "",
        "error_code": "",
        "raw_path": "",
        "content_type": "",
        "is_cached": False,
        "bytes": 0,
        "sha256": "",
    }


def _facts_row_base(ticker: str, url: str, report_date: str, period: str, source: str) -> dict:
    return {
        "ticker": ticker,
        "url": url,
        "report_date": report_date,
        "period": period,
        "source": source,
        "status": "",
        "error_code": "",
        "fact_revenue": np.nan,
        "fact_ebit": np.nan,
        "fact_eps": np.nan,
        "text_len": 0,
    }


def run(ctx, log) -> int:
    paths = resolve_paths(ctx.cfg, ctx.project_root)
    raw_ir_root = paths["raw_dir"] / "ir"
    raw_ir_root.mkdir(parents=True, exist_ok=True)

    golden_ir_path = ctx.project_root / "data" / "golden" / "ir_facts.parquet"
    index_path = raw_ir_root / "index.parquet"

    cfg = (ctx.cfg.get("ir_reports") or {})
    mapping_csv = _safe_text(cfg.get("mapping_csv", r"config\ir_sources.csv"))
    mapping_path = Path(mapping_csv)
    if not mapping_path.is_absolute():
        mapping_path = (ctx.project_root / mapping_path).resolve()

    user_agent = _safe_text(cfg.get("user_agent", "Deecon-IRBot/1.0 (+local research)"))
    timeout = int(cfg.get("timeout_sec", 20))
    rate_limit_sec = float(cfg.get("rate_limit_sec", 1.0))

    existing_index = _load_existing(index_path, INDEX_COLS)
    existing_facts = _load_existing(golden_ir_path, FACT_COLS)

    mapping = _read_mapping_csv(mapping_path)
    if mapping.empty:
        log.info(f"ir_reports: mapping empty or missing -> {mapping_path}")
        _write_parquet(existing_index, index_path, ["ticker", "report_date", "url", "source", "status", "error_code"])
        _write_parquet(existing_facts, golden_ir_path, ["ticker", "report_date", "url", "source", "status", "error_code"])
        return 0

    robots_cache: dict[str, RobotFileParser] = {}
    last_request_ts: dict[str, float] = {}

    known_keys = {
        (
            _safe_text(r["ticker"]).upper(),
            _safe_text(r["url"]),
            _safe_text(r["report_date"]),
            _safe_text(r["period"]),
            _safe_text(r["source"]),
        )
        for _, r in existing_index.iterrows()
    }

    new_index_rows: list[dict] = []
    new_fact_rows: list[dict] = []

    for _, row in mapping.iterrows():
        ticker = _safe_text(row.get("ticker")).upper()
        url = _safe_text(row.get("url"))
        source = _safe_text(row.get("source"))
        period = _safe_text(row.get("period")) or "unknown"
        report_date = _parse_report_date(_safe_text(row.get("report_date")), fallback=ctx.asof)

        if not source and url:
            source = _host(url)

        idx_row = _index_row_base(ticker, url, report_date, period, source)
        fact_row = _facts_row_base(ticker, url, report_date, period, source)

        key = (ticker, url, report_date, period, source)
        if key in known_keys:
            idx_row["status"] = "SKIPPED"
            idx_row["error_code"] = "DUPLICATE"
            fact_row["status"] = "SKIPPED"
            fact_row["error_code"] = "DUPLICATE"
            new_index_rows.append(idx_row)
            new_fact_rows.append(fact_row)
            continue
        known_keys.add(key)

        if not ticker or not url:
            idx_row["status"] = "ERROR"
            idx_row["error_code"] = "NO_URL"
            fact_row["status"] = "ERROR"
            fact_row["error_code"] = "NO_URL"
            new_index_rows.append(idx_row)
            new_fact_rows.append(fact_row)
            continue

        allowed, robots_err = _can_fetch(url, user_agent=user_agent, cache=robots_cache, timeout=timeout)
        if not allowed:
            idx_row["status"] = "ERROR"
            idx_row["error_code"] = robots_err
            fact_row["status"] = "ERROR"
            fact_row["error_code"] = robots_err
            new_index_rows.append(idx_row)
            new_fact_rows.append(fact_row)
            continue

        raw_dir = raw_ir_root / ticker / report_date
        raw_dir.mkdir(parents=True, exist_ok=True)

        content_type = ""
        raw_name = _safe_file_name(url)
        raw_path = raw_dir / raw_name

        try:
            if raw_path.exists() and raw_path.stat().st_size > 0:
                b = raw_path.read_bytes()
                is_cached = True
            else:
                _rate_limit(_host(url), last_request_ts, rate_limit_sec)
                b, content_type = _fetch_url_bytes(url, user_agent=user_agent, timeout=timeout)
                raw_name = _safe_file_name(url, content_type)
                raw_path = raw_dir / raw_name
                raw_path.write_bytes(b)
                is_cached = False

            is_pdf = raw_path.suffix.lower() == ".pdf" or ("pdf" in (content_type or "").lower())
            if is_pdf:
                text = _extract_pdf_text(b)
                if not text:
                    idx_row["status"] = "ERROR"
                    idx_row["error_code"] = "UNPARSABLE"
                    fact_row["status"] = "ERROR"
                    fact_row["error_code"] = "UNPARSABLE"
                else:
                    facts = _extract_facts(text)
                    fact_row.update(facts)
                    idx_row["status"] = "OK"
                    fact_row["status"] = "OK"
            else:
                text = _extract_html_text(b)
                facts = _extract_facts(text)
                fact_row.update(facts)
                idx_row["status"] = "OK"
                fact_row["status"] = "OK"

            idx_row["raw_path"] = str(raw_path)
            idx_row["content_type"] = content_type
            idx_row["is_cached"] = bool(is_cached)
            idx_row["bytes"] = int(len(b))
            idx_row["sha256"] = _sha256_bytes(b)
            if not fact_row["text_len"]:
                fact_row["text_len"] = int(len(text or ""))

        except urllib.error.HTTPError as e:
            idx_row["status"] = "ERROR"
            idx_row["error_code"] = f"HTTP_{int(e.code)}"
            fact_row["status"] = "ERROR"
            fact_row["error_code"] = f"HTTP_{int(e.code)}"
        except (urllib.error.URLError, TimeoutError):
            idx_row["status"] = "ERROR"
            idx_row["error_code"] = "NETWORK"
            fact_row["status"] = "ERROR"
            fact_row["error_code"] = "NETWORK"
        except Exception:
            idx_row["status"] = "ERROR"
            idx_row["error_code"] = "INTERNAL"
            fact_row["status"] = "ERROR"
            fact_row["error_code"] = "INTERNAL"

        new_index_rows.append(idx_row)
        new_fact_rows.append(fact_row)

    new_index_df = pd.DataFrame(new_index_rows, columns=INDEX_COLS)
    new_facts_df = pd.DataFrame(new_fact_rows, columns=FACT_COLS)

    if existing_index.empty:
        index_df = new_index_df
    elif new_index_df.empty:
        index_df = existing_index
    else:
        index_df = pd.concat([existing_index, new_index_df], ignore_index=True)

    if existing_facts.empty:
        facts_df = new_facts_df
    elif new_facts_df.empty:
        facts_df = existing_facts
    else:
        facts_df = pd.concat([existing_facts, new_facts_df], ignore_index=True)

    _write_parquet(index_df[INDEX_COLS], index_path, ["ticker", "report_date", "url", "source", "status", "error_code"])
    _write_parquet(facts_df[FACT_COLS], golden_ir_path, ["ticker", "report_date", "url", "source", "status", "error_code"])

    log.info(f"ir_reports: wrote {index_path}")
    log.info(f"ir_reports: wrote {golden_ir_path}")
    return 0
