from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """
    Supports nested dotted keys in dict-like configs.
    Example: _cfg_get(cfg, "valuation.dcf.years", 5)
    """
    if not isinstance(cfg, dict):
        return default
    cur = cfg
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _norm_yahoo(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


@dataclass
class DCFParams:
    years: int = 5
    growth: float = 0.02          # annual growth on FCF
    terminal_growth: float = 0.02 # capped <= 2%
    fcf_mult: float = 1.00        # margin proxy in sensitivity
    wacc_default: float = 0.09
    wacc_min: float = 0.03
    wacc_max: float = 0.25


def _dcf_value(
    fcf0_m: float,
    wacc: float,
    years: int,
    growth: float,
    terminal_growth: float,
    fcf_mult: float,
) -> float:
    """
    Simple DCF on FCF in *millions*.
    Returns EV in millions.
    """
    if not np.isfinite(fcf0_m):
        return np.nan
    if not np.isfinite(wacc):
        return np.nan
    if wacc <= 0 or wacc <= terminal_growth:
        return np.nan

    g = float(growth)
    tg = float(min(terminal_growth, 0.02))
    n = int(years)
    f0 = float(fcf0_m) * float(fcf_mult)

    # Forecast and discount
    pv_sum = 0.0
    for t in range(1, n + 1):
        ft = f0 * ((1.0 + g) ** t)
        pv_sum += ft / ((1.0 + wacc) ** t)

    f_n = f0 * ((1.0 + g) ** n)
    tv = (f_n * (1.0 + tg)) / (wacc - tg)
    pv_tv = tv / ((1.0 + wacc) ** n)

    return pv_sum + pv_tv


def _resolve_paths(ctx) -> Dict[str, Path]:
    """
    Try project resolve_paths; otherwise fallback to local defaults.
    """
    try:
        from src.common.config import resolve_paths  # type: ignore
        paths = resolve_paths(ctx.cfg, ctx.project_root)
        # normalize to Path
        for k, v in list(paths.items()):
            if not isinstance(v, Path):
                paths[k] = Path(v)
        return paths
    except Exception:
        root = Path(getattr(ctx, "project_root", "."))  # best effort
        return {
            "processed_dir": root / "data" / "processed",
            "runs_dir": root / "runs",
        }


def run(ctx, log) -> int:
    """
    Builds:
      - runs/<run_id>/valuation.csv
      - runs/<run_id>/valuation_sensitivity.csv
      - data/processed/master_valued.parquet

    Key guarantee:
      - Uses yahoo_ticker as unique instrument id for valuation output and merges.
      - Enforces one-to-one merge, stable rowcount.
    """
    paths = _resolve_paths(ctx)
    processed_dir = paths.get("processed_dir", Path("data/processed"))
    runs_dir = paths.get("runs_dir", Path("runs"))

    # Run dir (prefer ctx.run_dir if present)
    run_dir = Path(getattr(ctx, "run_dir", "")) if getattr(ctx, "run_dir", None) else (runs_dir / getattr(ctx, "run_id", "run"))
    _ensure_dir(run_dir)
    _ensure_dir(processed_dir)

    asof = getattr(ctx, "asof", None)

    # Load base (master_cost is the safest pre-valuation frame)
    master_cost_path = processed_dir / "master_cost.parquet"
    if not master_cost_path.exists():
        raise FileNotFoundError(f"valuation: missing {master_cost_path} (run cost_of_capital first)")

    df = pd.read_parquet(master_cost_path)

    # Required key
    if "yahoo_ticker" not in df.columns:
        raise ValueError("valuation: master_cost is missing yahoo_ticker (cannot merge safely).")

    df = df.copy()
    df["yahoo_ticker"] = _norm_yahoo(df["yahoo_ticker"])

    # DoD-guard: base must be unique on yahoo_ticker
    base_dup = int(df["yahoo_ticker"].duplicated().sum())
    if base_dup:
        raise ValueError(f"valuation: master_cost has duplicate yahoo_ticker keys: {base_dup}")

    # Pick columns
    fcf_col = _pick_first_col(df, ["fcf_millions", "fcf_-_millions", "fcf", "free_cash_flow_millions", "free_cash_flow"])
    ocf_col = _pick_first_col(df, ["ocf_millions", "ocf_-_millions", "ocf"])
    capex_col = _pick_first_col(df, ["capex_millions", "capex_-_millions", "capex"])
    net_debt_col = _pick_first_col(df, ["net_debt_millions", "net_debt_-_current", "net_debt_current", "net_debt"])
    wacc_col = _pick_first_col(df, ["wacc_used", "wacc"])
    coe_col = _pick_first_col(df, ["coe_used", "coe"])
    ticker_col = _pick_first_col(df, ["ticker"])
    company_col = _pick_first_col(df, ["company"])

    # Build FCF (millions)
    reason = pd.Series(index=df.index, dtype="object")

    if fcf_col:
        fcf_m = _to_num(df[fcf_col])
    else:
        fcf_m = pd.Series(np.nan, index=df.index)

    if fcf_m.notna().mean() == 0.0 and ocf_col and capex_col:
        ocf_m = _to_num(df[ocf_col])
        capex_m = _to_num(df[capex_col])
        fcf_m = ocf_m - capex_m

    # Mark missing fcf
    reason = np.where(fcf_m.notna(), None, "missing_fcf")
    reason = pd.Series(reason, index=df.index, dtype="object")

    # WACC/COE
    params = DCFParams(
        years=int(_cfg_get(ctx.cfg, "valuation.dcf.years", 5)),
        growth=float(_cfg_get(ctx.cfg, "valuation.dcf.growth", 0.02)),
        terminal_growth=float(min(_cfg_get(ctx.cfg, "valuation.dcf.terminal_growth", 0.02), 0.02)),
        fcf_mult=float(_cfg_get(ctx.cfg, "valuation.dcf.fcf_mult", 1.0)),
        wacc_default=float(_cfg_get(ctx.cfg, "valuation.wacc_default", 0.09)),
        wacc_min=float(_cfg_get(ctx.cfg, "valuation.wacc_min", 0.03)),
        wacc_max=float(_cfg_get(ctx.cfg, "valuation.wacc_max", 0.25)),
    )

    if wacc_col:
        wacc = _to_num(df[wacc_col]).fillna(params.wacc_default)
    else:
        wacc = pd.Series(params.wacc_default, index=df.index)

    if coe_col:
        coe = _to_num(df[coe_col]).fillna(params.wacc_default)
    else:
        coe = pd.Series(params.wacc_default, index=df.index)

    # Validate WACC bounds
    bad_wacc = (wacc <= params.wacc_min) | (wacc >= params.wacc_max) | (~np.isfinite(wacc))
    if bad_wacc.any():
        reason = reason.where(~bad_wacc, other="invalid_wacc")

    # Net debt (millions)
    if net_debt_col:
        net_debt_m = _to_num(df[net_debt_col]).fillna(0.0)
    else:
        net_debt_m = pd.Series(0.0, index=df.index)

    # DCF EV (millions)
    intrinsic_ev_m = []
    model = []
    for i in range(len(df)):
        if reason.iat[i] is not None:
            intrinsic_ev_m.append(np.nan)
            model.append("DCF")
            continue
        ev_m = _dcf_value(
            fcf0_m=float(fcf_m.iat[i]),
            wacc=float(wacc.iat[i]),
            years=params.years,
            growth=params.growth,
            terminal_growth=params.terminal_growth,
            fcf_mult=params.fcf_mult,
        )
        if not np.isfinite(ev_m):
            intrinsic_ev_m.append(np.nan)
            reason.iat[i] = "dcf_failed"
        else:
            intrinsic_ev_m.append(ev_m)
        model.append("DCF")

    intrinsic_ev_m = pd.Series(intrinsic_ev_m, index=df.index, dtype="float64")
    intrinsic_equity_m = intrinsic_ev_m - net_debt_m

    # Convert to currency units (not millions) for downstream decision logic
    intrinsic_ev = intrinsic_ev_m * 1e6
    intrinsic_equity = intrinsic_equity_m * 1e6
    net_debt_used = net_debt_m * 1e6

    # Build valuation_df (A + C)
    valuation_df = pd.DataFrame({
        "yahoo_ticker": df["yahoo_ticker"],
        "ticker": df[ticker_col] if ticker_col else df["yahoo_ticker"],
        "model": model,
        "intrinsic_ev": intrinsic_ev,
        "intrinsic_equity": intrinsic_equity,
        "net_debt_used": net_debt_used,
        "wacc_used": wacc,
        "coe_used": coe,
        "reason": reason.fillna(""),
    })

    # B1) valuation_df must be unique on yahoo_ticker
    dup = int(valuation_df["yahoo_ticker"].duplicated().sum())
    if dup:
        raise ValueError(f"valuation: duplicate yahoo_ticker in valuation_df: {dup}")

    # Write valuation.csv (C)
    out_csv = run_dir / "valuation.csv"
    valuation_df.sort_values(["yahoo_ticker"]).to_csv(out_csv, index=False)
    log.info(f"valuation: wrote {out_csv}")

    # Sensitivity grid: growth +/-1pp, wacc +/-1pp, fcf_mult +/-2%
    g_shocks = [-0.01, 0.0, 0.01]
    w_shocks = [-0.01, 0.0, 0.01]
    f_shocks = [0.98, 1.00, 1.02]

    rows = []
    base_g = params.growth
    base_tg = params.terminal_growth
    base_n = params.years

    for dg in g_shocks:
        for dw in w_shocks:
            for fm in f_shocks:
                scen = f"g{dg:+.2%}_w{dw:+.2%}_f{(fm-1.0):+.2%}"
                # vectorized compute via apply (fast enough at ~1741*27)
                ev_m_list = []
                eq_m_list = []
                rsn_list = []
                wacc_s = (wacc + dw).astype(float)
                g_s = base_g + dg
                tg_s = min(base_tg, 0.02)

                for i in range(len(df)):
                    # carry base reason for missing_fcf/invalid_wacc etc
                    if reason.iat[i] in ("missing_fcf", "invalid_wacc"):
                        ev_m_list.append(np.nan)
                        eq_m_list.append(np.nan)
                        rsn_list.append(reason.iat[i])
                        continue
                    evm = _dcf_value(
                        fcf0_m=float(fcf_m.iat[i]),
                        wacc=float(wacc_s.iat[i]),
                        years=base_n,
                        growth=float(g_s),
                        terminal_growth=float(tg_s),
                        fcf_mult=float(fm),
                    )
                    if not np.isfinite(evm):
                        ev_m_list.append(np.nan)
                        eq_m_list.append(np.nan)
                        rsn_list.append("dcf_failed")
                    else:
                        ev_m_list.append(evm)
                        eq_m_list.append(evm - float(net_debt_m.iat[i]))
                        rsn_list.append("")
                rows.append(pd.DataFrame({
                    "scenario": scen,
                    "yahoo_ticker": df["yahoo_ticker"].values,
                    "ticker": (df[ticker_col].values if ticker_col else df["yahoo_ticker"].values),
                    "model": "DCF",
                    "growth_used": g_s,
                    "wacc_used": wacc_s.values,
                    "fcf_mult": fm,
                    "intrinsic_ev": (np.array(ev_m_list) * 1e6),
                    "intrinsic_equity": (np.array(eq_m_list) * 1e6),
                    "reason": rsn_list,
                }))

    sens_df = pd.concat(rows, ignore_index=True)
    sens_out = run_dir / "valuation_sensitivity.csv"
    sens_df.to_csv(sens_out, index=False)
    log.info(f"valuation: wrote {sens_out}")

    # B2) merge back 1:1 on yahoo_ticker (not ticker!)
    master_cost = df
    before = len(master_cost)

    master_valued = master_cost.merge(
        valuation_df.drop(columns=["ticker"]),
        on="yahoo_ticker",
        how="left",
        validate="one_to_one",
    )

    # B3) DoD-guard: rowcount must not change
    if len(master_valued) != before:
        raise ValueError(f"valuation: rowcount changed after merge ({before} -> {len(master_valued)}). Check join key!")

    # Final DoD: no dup yahoo_ticker
    final_dup = int(master_valued["yahoo_ticker"].duplicated().sum())
    if final_dup:
        raise ValueError(f"valuation: master_valued has duplicate yahoo_ticker after merge: {final_dup}")

    out_parq = processed_dir / "master_valued.parquet"
    master_valued.to_parquet(out_parq, index=False)
    log.info(f"valuation: wrote {out_parq}")

    return 0
