from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return float(s) if pd.notna(s) else float("nan")


@dataclass
class ModelArtifact:
    run_id: str
    label: str
    feature_cols: list[str]
    alpha: float
    fit_intercept: bool
    scaler_mean: list[float]
    scaler_scale: list[float]
    coef: list[float]
    intercept: float


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--input", default="data/processed/panel_labeled.parquet")
    p.add_argument("--label", default="fwd_12m_return")
    p.add_argument("--alpha", type=float, default=10.0, help="Ridge alpha (L2)")
    p.add_argument("--date-col", default="date")
    p.add_argument("--id-cols", default="yahoo_ticker,country,sector,asof", help="kolonner som ikke er features")
    args = p.parse_args()

    df = pd.read_parquet(Path(args.input))
    df[args.date_col] = pd.to_datetime(df[args.date_col])

    label = args.label
    if label not in df.columns:
        raise ValueError(f"Mangler label-kolonne {label}")

    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]

    # feature-kolonner: numeriske kolonner som ikke er ID/date/label
    blacklist = set(id_cols + [args.date_col, label])
    numeric = [c for c in df.columns if c not in blacklist and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise RuntimeError("Fant ingen numeriske feature-kolonner. Sjekk panel.")

    work = df.dropna(subset=[label]).copy()
    work = work.dropna(subset=numeric, how="any")  # start strengt

    work["year"] = work[args.date_col].dt.year
    years = sorted(work["year"].unique().tolist())
    if len(years) < 3:
        raise RuntimeError("For få år til walk-forward. Trenger minst 3.")

    run_id = _run_id()
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    preds_all = []

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=args.alpha, fit_intercept=True, random_state=0)),
        ]
    )

    # walk-forward: test år-for-år, tren på alle tidligere år
    for y in years[1:]:
        train = work[work["year"] < y]
        test = work[work["year"] == y]
        if train.empty or test.empty:
            continue

        Xtr, ytr = train[numeric].values, train[label].values
        Xte, yte = test[numeric].values, test[label].values

        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)

        rmse = mean_squared_error(yte, yhat, squared=False)
        ic = _spearman_ic(yte, yhat)

        rows.append({"test_year": y, "train_rows": len(train), "test_rows": len(test), "rmse": rmse, "ic_spearman": ic})
        tmp = test[["yahoo_ticker", args.date_col]].copy()
        tmp["y_true"] = yte
        tmp["y_pred"] = yhat
        tmp["test_year"] = y
        preds_all.append(tmp)

    diag = pd.DataFrame(rows)
    diag_path = run_dir / "regression_diagnostics.csv"
    diag.to_csv(diag_path, index=False)

    # fit final modell på alt (for bruk i decision/backtest)
    X = work[numeric].values
    y = work[label].values
    pipe.fit(X, y)

    scaler: StandardScaler = pipe.named_steps["scaler"]
    ridge: Ridge = pipe.named_steps["ridge"]

    artifact = ModelArtifact(
        run_id=run_id,
        label=label,
        feature_cols=numeric,
        alpha=float(args.alpha),
        fit_intercept=True,
        scaler_mean=[float(x) for x in scaler.mean_],
        scaler_scale=[float(x) for x in scaler.scale_],
        coef=[float(x) for x in ridge.coef_],
        intercept=float(ridge.intercept_),
    )

    weights_path = run_dir / "regression_weights.json"
    weights_path.write_text(json.dumps(asdict(artifact), indent=2), encoding="utf-8")

    # summary
    summary_path = run_dir / "regression_summary.md"
    avg_rmse = float(diag["rmse"].mean()) if not diag.empty else float("nan")
    avg_ic = float(diag["ic_spearman"].mean()) if not diag.empty else float("nan")
    summary = f"""# Regression summary

run_id: {run_id}
label: {label}
alpha (ridge): {args.alpha}

Folds (walk-forward):
- n_folds: {len(diag)}
- avg RMSE: {avg_rmse:.6f}
- avg Spearman IC: {avg_ic:.6f}

Artifacts:
- regression_weights.json (inkl scaler + coef)
- regression_diagnostics.csv (RMSE/IC pr fold)
"""
    summary_path.write_text(summary, encoding="utf-8")

    print(f"OK: {weights_path}")
    print(f"OK: {summary_path}")
    print(f"OK: {diag_path}")


if __name__ == "__main__":
    main()
