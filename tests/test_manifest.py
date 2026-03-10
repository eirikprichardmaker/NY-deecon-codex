"""
Tests for src/manifest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.manifest import compute_config_hash, get_git_commit, write_manifest


# ---------------------------------------------------------------------------
# compute_config_hash
# ---------------------------------------------------------------------------

def test_config_hash_is_16_chars(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("key: value\n")
    h = compute_config_hash(cfg)
    assert len(h) == 16
    assert h.isalnum()


def test_config_hash_is_deterministic(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("key: value\n")
    assert compute_config_hash(cfg) == compute_config_hash(cfg)


def test_config_hash_changes_with_content(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("key: value1\n")
    h1 = compute_config_hash(cfg)
    cfg.write_text("key: value2\n")
    h2 = compute_config_hash(cfg)
    assert h1 != h2


# ---------------------------------------------------------------------------
# get_git_commit
# ---------------------------------------------------------------------------

def test_get_git_commit_returns_string():
    result = get_git_commit()
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_git_commit_is_12_chars_or_unknown():
    result = get_git_commit()
    assert result == "unknown" or len(result) == 12


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

def test_write_manifest_creates_file(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("paths:\n  runs_dir: runs\n")
    path = write_manifest(
        run_dir=tmp_path,
        asof="2026-03-10",
        config_path=cfg,
        seed=42,
        ticker_count=150,
        data_sources=["borsdata", "yahoo"],
    )
    assert path.exists()
    assert path.name == "manifest.json"


def test_write_manifest_valid_json(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("x: 1\n")
    write_manifest(
        run_dir=tmp_path,
        asof="2026-03-10",
        config_path=cfg,
        seed=0,
        ticker_count=50,
        data_sources=["borsdata"],
    )
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert isinstance(data, dict)


def test_write_manifest_required_fields(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("x: 1\n")
    write_manifest(
        run_dir=tmp_path,
        asof="2026-03-10",
        config_path=cfg,
        seed=7,
        ticker_count=100,
        data_sources=["borsdata", "yahoo"],
    )
    data = json.loads((tmp_path / "manifest.json").read_text())
    for field in ["asof_date", "config_hash", "git_commit", "seed",
                  "timestamp_utc", "ticker_count", "data_sources",
                  "agent_config", "python_version"]:
        assert field in data, f"Manglende felt: {field}"


def test_write_manifest_correct_values(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("x: 1\n")
    write_manifest(
        run_dir=tmp_path,
        asof="2026-03-10",
        config_path=cfg,
        seed=42,
        ticker_count=77,
        data_sources=["borsdata", "esef"],
        agent_config={"enabled": False},
    )
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["asof_date"] == "2026-03-10"
    assert data["seed"] == 42
    assert data["ticker_count"] == 77
    assert data["data_sources"] == ["borsdata", "esef"]
    assert data["agent_config"] == {"enabled": False}
    assert data["python_version"] == sys.version


def test_write_manifest_agent_config_defaults_to_empty(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("x: 1\n")
    write_manifest(
        run_dir=tmp_path,
        asof="2026-03-10",
        config_path=cfg,
        seed=0,
        ticker_count=0,
        data_sources=[],
    )
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["agent_config"] == {}


def test_write_manifest_missing_config_file(tmp_path):
    """Manglende config-fil skal ikke kaste exception — returnerer 'missing' som hash."""
    missing_cfg = tmp_path / "nonexistent.yaml"
    path = write_manifest(
        run_dir=tmp_path,
        asof="2026-03-10",
        config_path=missing_cfg,
        seed=0,
        ticker_count=0,
        data_sources=[],
    )
    data = json.loads(path.read_text())
    assert data["config_hash"] == "missing"


def test_write_manifest_timestamp_is_utc_iso(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("x: 1\n")
    write_manifest(
        run_dir=tmp_path,
        asof="2026-03-10",
        config_path=cfg,
        seed=0,
        ticker_count=0,
        data_sources=[],
    )
    data = json.loads((tmp_path / "manifest.json").read_text())
    ts = data["timestamp_utc"]
    assert "T" in ts
    assert ts.endswith("+00:00") or ts.endswith("Z") or "+00:00" in ts


def test_write_manifest_returns_path(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("x: 1\n")
    result = write_manifest(
        run_dir=tmp_path,
        asof="2026-03-10",
        config_path=cfg,
        seed=0,
        ticker_count=0,
        data_sources=[],
    )
    assert isinstance(result, Path)
    assert result == tmp_path / "manifest.json"
