from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_wft_cross_train_gate_smoke(tmp_path):
    train8 = tmp_path / "train8_summary.csv"
    train10 = tmp_path / "train10_summary.csv"
    out_dir = tmp_path / "out"

    pd.DataFrame(
        [
            {"seed": 11, "go_no_go": "GO", "go_no_go_reasons": "", "risk_adjusted_rank": 1},
            {"seed": 22, "go_no_go": "GO", "go_no_go_reasons": "", "risk_adjusted_rank": 2},
        ]
    ).to_csv(train8, index=False)

    pd.DataFrame(
        [
            {"seed": 11, "go_no_go": "NO_GO", "go_no_go_reasons": "info_ratio", "risk_adjusted_rank": 1},
            {"seed": 22, "go_no_go": "GO", "go_no_go_reasons": "", "risk_adjusted_rank": 2},
        ]
    ).to_csv(train10, index=False)

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "wft_cross_train_gate.py"),
        "--train8-summary",
        str(train8),
        "--train10-summary",
        str(train10),
        "--output-dir",
        str(out_dir),
    ]

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    out_csv = out_dir / "cross_train_gate.csv"
    out_md = out_dir / "cross_train_gate.md"
    assert out_csv.exists()
    assert out_md.exists()

    df = pd.read_csv(out_csv)
    expected = {
        "seed",
        "cross_train_rank",
        "go_no_go_train_8",
        "go_no_go_train_10",
        "cross_train_gate_pass",
        "model_outcome",
        "model_outcome_reasons",
    }
    assert expected.issubset(set(df.columns))

    row11 = df[df["seed"] == 11].iloc[0]
    row22 = df[df["seed"] == 22].iloc[0]
    assert str(row11["model_outcome"]) == "CASH"
    assert "train_10:info_ratio" in str(row11["model_outcome_reasons"])
    assert str(row22["model_outcome"]) == "ONE_STOCK"

    md = out_md.read_text(encoding="utf-8")
    assert "model_outcome: `CASH`" in md


def test_wft_cross_train_gate_legacy_summary_fallback(tmp_path):
    train8 = tmp_path / "train8_legacy.csv"
    train10 = tmp_path / "train10_legacy.csv"
    out_dir = tmp_path / "out_legacy"

    # Legacy summary shape without go_no_go/go_no_go_reasons.
    pd.DataFrame(
        [
            {
                "seed": 11,
                "risk_adjusted_rank": 1,
                "excess_cagr": 0.10,
                "info_ratio": 0.50,
                "max_dd": -0.20,
                "benchmark_maxdd": -0.18,
            }
        ]
    ).to_csv(train8, index=False)
    pd.DataFrame(
        [
            {
                "seed": 11,
                "risk_adjusted_rank": 1,
                "excess_cagr": 0.08,
                "info_ratio": 0.40,
                "max_dd": -0.19,
                "benchmark_maxdd": -0.17,
            }
        ]
    ).to_csv(train10, index=False)

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "wft_cross_train_gate.py"),
        "--train8-summary",
        str(train8),
        "--train10-summary",
        str(train10),
        "--output-dir",
        str(out_dir),
    ]

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    df = pd.read_csv(out_dir / "cross_train_gate.csv")
    row = df[df["seed"] == 11].iloc[0]
    assert str(row["model_outcome"]) == "CASH"
    assert "cash_rule_data_missing" in str(row["model_outcome_reasons"])
