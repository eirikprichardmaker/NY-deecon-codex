"""
Evidence builder: lager saniterte tekstpakker fra nedlastede kvartals-/årsrapporter.
Kobler report_watch-nedlastinger til Agent A (Business Quality Evaluator).
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.agents.sanitizer import sanitize_text

logger = logging.getLogger(__name__)

# Maks tegn per enkelt rapport-utdrag
_MAX_CHARS_PER_SOURCE = 8000
# Maks totalt per ticker (for å holde seg innenfor token-budsjettet)
_MAX_TOTAL_CHARS = 20000

# Typisk plassering for report_watch-nedlastinger
_DEFAULT_DOWNLOADS_DIR = Path("data/raw/ir_auto")


def build_evidence_pack(
    ticker: str,
    asof: str,
    downloads_dir: Path | None = None,
) -> list[dict]:
    """
    Bygg sanitert tekstpakke fra nedlastede rapporter for en gitt ticker.

    Søker i:
      {downloads_dir}/{ticker}/          (alle datoer, sortert nyest først)

    Returnerer liste med:
      {source_id, source_type, content (sanitert)}

    Tom liste returneres (ikke feil) dersom ingen rapporter finnes.
    """
    base = downloads_dir or _DEFAULT_DOWNLOADS_DIR
    ticker_dir = base / ticker

    if not ticker_dir.exists():
        logger.debug(f"evidence_builder [{ticker}]: ingen rapportmappe ({ticker_dir})")
        return []

    # Finn alle tekstfiler og PDF-er, sorter nyest dato-mappe først
    candidates: list[tuple[str, Path]] = []
    for date_dir in sorted(ticker_dir.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        # Hopp over mapper etter asof
        if date_dir.name > asof:
            continue
        for f in date_dir.iterdir():
            if f.suffix.lower() in (".txt", ".pdf", ".html", ".htm"):
                source_type = _classify_source(f)
                candidates.append((source_type, f))

    if not candidates:
        logger.debug(f"evidence_builder [{ticker}]: ingen rapportfiler funnet")
        return []

    evidence = []
    total_chars = 0

    for source_type, fpath in candidates:
        if total_chars >= _MAX_TOTAL_CHARS:
            logger.debug(f"evidence_builder [{ticker}]: makstegn nådd, stopper")
            break

        raw = _read_file(fpath)
        if not raw:
            continue

        clean, injection_suspected = sanitize_text(raw, max_chars=_MAX_CHARS_PER_SOURCE)
        if not clean:
            continue

        if injection_suspected:
            logger.warning(
                f"evidence_builder [{ticker}]: mulig prompt injection i {fpath.name} — inkludert med flagg"
            )

        source_id = f"{ticker}_{fpath.parent.name}_{fpath.stem}"
        evidence.append({
            "source_id": source_id,
            "source_type": source_type,
            "content": clean,
            "injection_suspected": injection_suspected,
            "file_path": str(fpath),
        })
        total_chars += len(clean)

    logger.info(
        f"evidence_builder [{ticker}]: {len(evidence)} kilde(r), {total_chars} tegn totalt"
    )
    return evidence


def _classify_source(path: Path) -> str:
    """Klassifiser rapporttype fra filnavn."""
    name = path.stem.lower()
    if any(k in name for k in ("annual", "årsrapport", "year")):
        return "annual_report_pdf"
    if any(k in name for k in ("quarter", "kvartal", "q1", "q2", "q3", "q4", "interim")):
        return "quarterly_report"
    if path.suffix.lower() in (".html", ".htm"):
        return "ir_webpage"
    return "report_pdf"


def _read_file(path: Path) -> str:
    """Les tekstinnhold fra fil. PDF leses som binær og konverteres til tekst via heuristikk."""
    try:
        if path.suffix.lower() == ".pdf":
            return _extract_pdf_text(path)
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.debug(f"evidence_builder: kunne ikke lese {path}: {e}")
        return ""


def _extract_pdf_text(path: Path) -> str:
    """
    Enkel PDF-tekstekstraksjon.
    Prøver pypdf/pdfminer ved behov. Fallback: tom streng.
    """
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages[:20]]
        return "\n".join(pages)
    except ImportError:
        pass
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        return extract_text(str(path))
    except ImportError:
        pass
    logger.debug(f"evidence_builder: ingen PDF-leser tilgjengelig for {path.name}")
    return ""
