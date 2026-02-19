from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.utils import get_first_env, normalize_secret


def test_normalize_secret_strips_wrapping_quotes():
    assert normalize_secret('"abc"') == "abc"
    assert normalize_secret("'abc'") == "abc"
    assert normalize_secret("  'abc'  ") == "abc"


def test_get_first_env_uses_first_non_empty(monkeypatch):
    monkeypatch.setenv("BORSDATA_AUTHKEY", '"key1"')
    monkeypatch.setenv("BORSDATA_API_KEY", "key2")
    assert get_first_env(["BORSDATA_AUTHKEY", "BORSDATA_API_KEY"]) == "key1"


def test_get_first_env_fallback(monkeypatch):
    monkeypatch.delenv("BORSDATA_AUTHKEY", raising=False)
    monkeypatch.setenv("BORSDATA_API_KEY", "'key2'")
    assert get_first_env(["BORSDATA_AUTHKEY", "BORSDATA_API_KEY", "BORSDATA_KEY"]) == "key2"
