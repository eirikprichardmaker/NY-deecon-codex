from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_BASE_COLS = {"seed"}


def _resolve_path(base_root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base_root / p).resolve()


def _load_summary(
    path: Path,
    *,
    label: str,
    go_min_excess_cagr: float,
    go_min_info_ratio: float,
    go_maxdd_gap_pp: float,
    go_max_cash_share: float,
    go_high_cash_risk_reason_min_share: float,
    go_high_cash_max_datamangler_share: float,
) -> pd.DataFrame:
    def _num_series(col: str) -> pd.Series:
        if col in d.columns:
            return pd.to_numeric(d[col], errors="coerce")
        return pd.Series(np.nan, index=d.index, dtype=float)

    if not path.exists():
        raise FileNotFoundError(f"{label} summary not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_BASE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{label} summary missing required columns: {sorted(missing)}")

    d = df.copy()
    d["seed"] = pd.to_numeric(d["seed"], errors="coerce")
    d = d.dropna(subset=["seed"]).copy()
    d["seed"] = d["seed"].astype(int)
    if {"go_no_go", "go_no_go_reasons"}.issubset(set(d.columns)):
        d["go_no_go"] = d["go_no_go"].astype(str).str.upper().str.strip()
        d["go_no_go_reasons"] = d["go_no_go_reasons"].fillna("").astype(str)
    else:
        excess = _num_series("excess_cagr")
        info = _num_series("info_ratio")
        max_dd = _num_series("max_dd")
        bench_max_dd = _num_series("benchmark_maxdd")
        cash_share = _num_series("cash_share")
        cash_risk = _num_series("cash_risk_reason_share")
        cash_data = _num_series("cash_datamangler_share")

        maxdd_gap = max_dd.abs() - bench_max_dd.abs()

        rule_excess = excess.gt(float(go_min_excess_cagr)).fillna(False)
        rule_info = info.gt(float(go_min_info_ratio)).fillna(False)
        rule_maxdd = maxdd_gap.le(float(go_maxdd_gap_pp) / 100.0).fillna(False)

        cash_rule = pd.Series(False, index=d.index, dtype=bool)
        has_cash = cash_share.notna()
        cash_rule.loc[has_cash & cash_share.le(float(go_max_cash_share))] = True
        cash_rule.loc[
            has_cash
            & cash_share.gt(float(go_max_cash_share))
            & cash_risk.ge(float(go_high_cash_risk_reason_min_share)).fillna(False)
            & cash_data.le(float(go_high_cash_max_datamangler_share)).fillna(False)
        ] = True

        d["go_no_go"] = np.where(rule_excess & rule_info & rule_maxdd & cash_rule, "GO", "NO_GO")

        reasons: list[str] = []
        for idx in d.index:
            fails: list[str] = []
            if not bool(rule_excess.loc[idx]):
                fails.append("excess_cagr")
            if not bool(rule_info.loc[idx]):
                fails.append("info_ratio")
            if not bool(rule_maxdd.loc[idx]):
                fails.append("maxdd_vs_benchmark")
            if not bool(cash_rule.loc[idx]):
                if not bool(has_cash.loc[idx]):
                    fails.append("cash_rule_data_missing")
                else:
                    fails.append("cash_share_reasons")
            reasons.append("|".join(fails))
        d["go_no_go_reasons"] = reasons

    if "risk_adjusted_rank" in d.columns:
        d["risk_adjusted_rank"] = pd.to_numeric(d["risk_adjusted_rank"], errors="coerce")
    else:
        d["risk_adjusted_rank"] = np.nan
    d["risk_adjusted_rank"] = d["risk_adjusted_rank"].fillna(1_000_000.0)

    d = d.sort_values(["risk_adjusted_rank", "seed"], kind="mergesort").drop_duplicates(subset=["seed"], keep="first")
    return d[["seed", "go_no_go", "go_no_go_reasons", "risk_adjusted_rank"]].reset_index(drop=True)


def _combine_failure_reasons(row: pd.Series) -> str:
    failures: list[str] = []
    if str(row["go_no_go_train_8"]) != "GO":
        reason8 = str(row["go_no_go_reasons_train_8"]).strip()
        failures.append(f"train_8:{reason8 if reason8 else 'failed'}")
    if str(row["go_no_go_train_10"]) != "GO":
        reason10 = str(row["go_no_go_reasons_train_10"]).strip()
        failures.append(f"train_10:{reason10 if reason10 else 'failed'}")
    return "|".join(failures)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(empty)_"
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_numeric_dtype(d[col]):
            d[col] = d[col].map(lambda v: f"{float(v):.6f}" if pd.notna(v) else "")
        else:
            d[col] = d[col].astype(str)
    header = "| " + " | ".join(d.columns.tolist()) + " |"
    sep = "| " + " | ".join(["---"] * len(d.columns)) + " |"
    lines = [header, sep]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(row.tolist()) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cross-train gate for 1-stock model: if any GO/NO-GO check fails in "
            "train=8 or train=10, model outcome is CASH."
        )
    )
    p.add_argument("--train8-summary", required=True, help="Path to sweep_summary.csv for train_8")
    p.add_argument("--train10-summary", required=True, help="Path to sweep_summary.csv for train_10")
    p.add_argument("--go-min-excess-cagr", type=float, default=0.0)
    p.add_argument("--go-min-info-ratio", type=float, default=0.20)
    p.add_argument("--go-maxdd-gap-pp", type=float, default=7.5)
    p.add_argument("--go-max-cash-share", type=float, default=0.50)
    p.add_argument("--go-high-cash-risk-reason-min-share", type=float, default=0.60)
    p.add_argument("--go-high-cash-max-datamangler-share", type=float, default=0.25)
    p.add_argument("--output-dir", default=None, help="Default: parent folder of train8 summary")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    train8_path = _resolve_path(root, args.train8_summary)
    train10_path = _resolve_path(root, args.train10_summary)
    output_dir = _resolve_path(root, args.output_dir) if args.output_dir else train8_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    t8 = _load_summary(
        train8_path,
        label="train_8",
        go_min_excess_cagr=float(args.go_min_excess_cagr),
        go_min_info_ratio=float(args.go_min_info_ratio),
        go_maxdd_gap_pp=float(args.go_maxdd_gap_pp),
        go_max_cash_share=float(args.go_max_cash_share),
        go_high_cash_risk_reason_min_share=float(args.go_high_cash_risk_reason_min_share),
        go_high_cash_max_datamangler_share=float(args.go_high_cash_max_datamangler_share),
    )
    t10 = _load_summary(
        train10_path,
        label="train_10",
        go_min_excess_cagr=float(args.go_min_excess_cagr),
        go_min_info_ratio=float(args.go_min_info_ratio),
        go_maxdd_gap_pp=float(args.go_maxdd_gap_pp),
        go_max_cash_share=float(args.go_max_cash_share),
        go_high_cash_risk_reason_min_share=float(args.go_high_cash_risk_reason_min_share),
        go_high_cash_max_datamangler_share=float(args.go_high_cash_max_datamangler_share),
    )

    merged = t8.merge(t10, on="seed", how="inner", suffixes=("_train_8", "_train_10"))
    if merged.empty:
        raise RuntimeError("No overlapping seeds between train_8 and train_10 summaries.")

    merged["cross_train_gate_pass"] = (
        merged["go_no_go_train_8"].astype(str).eq("GO") & merged["go_no_go_train_10"].astype(str).eq("GO")
    )
    merged["model_outcome"] = np.where(merged["cross_train_gate_pass"], "ONE_STOCK", "CASH")
    merged["model_outcome_reasons"] = merged.apply(_combine_failure_reasons, axis=1)
    merged = merged.sort_values(
        by=["risk_adjusted_rank_train_8", "risk_adjusted_rank_train_10", "seed"],
        kind="mergesort",
    ).reset_index(drop=True)
    merged["cross_train_rank"] = merged.index + 1

    out_csv = output_dir / "cross_train_gate.csv"
    merged[
        [
            "seed",
            "cross_train_rank",
            "go_no_go_train_8",
            "go_no_go_train_10",
            "cross_train_gate_pass",
            "model_outcome",
            "model_outcome_reasons",
        ]
    ].to_csv(out_csv, index=False)

    best = merged.iloc[0]
    out_md = output_dir / "cross_train_gate.md"
    md_lines = [
        "# Cross-Train Gate Decision",
        "",
        "- Rule: If one check fails across train=8 and train=10, hold CASH as model outcome.",
        f"- train_8 summary: `{train8_path}`",
        f"- train_10 summary: `{train10_path}`",
        "",
        "## Selected Seed",
        f"- seed: `{int(best['seed'])}`",
        f"- train_8 go_no_go: `{best['go_no_go_train_8']}`",
        f"- train_10 go_no_go: `{best['go_no_go_train_10']}`",
        f"- model_outcome: `{best['model_outcome']}`",
        f"- reasons: `{best['model_outcome_reasons']}`",
        "",
        "## Per Seed",
        _markdown_table(
            merged[
                [
                    "seed",
                    "cross_train_rank",
                    "go_no_go_train_8",
                    "go_no_go_train_10",
                    "model_outcome",
                    "model_outcome_reasons",
                ]
            ]
        ),
    ]
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"OK: {out_csv}")
    print(f"OK: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
