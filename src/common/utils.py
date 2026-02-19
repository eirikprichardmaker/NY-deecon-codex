# src/common/utils.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd

def safe_div(a, b):
    a = np.array(a, dtype="float64")
    b = np.array(b, dtype="float64")
    out = np.full_like(a, np.nan, dtype="float64")
    mask = (b != 0) & np.isfinite(b)
    out[mask] = a[mask] / b[mask]
    return out

def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def normalize_secret(value: str | None) -> str:
    if value is None:
        return ""
    v = str(value).strip()
    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
        v = v[1:-1].strip()
    return v


def get_first_env(keys: list[str]) -> str:
    for key in keys:
        val = normalize_secret(os.getenv(key))
        if val:
            return val
    return ""
