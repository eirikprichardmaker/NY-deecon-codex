# src/common/schema.py
from __future__ import annotations
from typing import Iterable
import pandas as pd
from .errors import SchemaError

def require_columns(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"{where}: missing columns: {missing}")

def require_unique(df: pd.DataFrame, keys: list[str], where: str) -> None:
    if df.duplicated(keys).any():
        raise SchemaError(f"{where}: duplicate keys for {keys}")
