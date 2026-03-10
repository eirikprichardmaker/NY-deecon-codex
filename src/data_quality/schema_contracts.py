"""
Pandera schema contracts for Deecon pipeline.
These are the "hard contracts" — if data breaks these, the pipeline STOPS.
"""
import pandera.pandas as pa
from pandera.pandas import DataFrameModel
from pandera.typing.pandas import Series
from typing import Optional


class MasterSchema(DataFrameModel):
    """Schema for master_valued.parquet — the main pipeline dataframe."""

    ticker: Series[str] = pa.Field(nullable=False, str_matches=r"^[A-Z0-9\-\.]+$")
    yahoo_ticker: Series[str] = pa.Field(nullable=False, unique=True)
    market_cap: Optional[Series[float]] = pa.Field(nullable=True, ge=0)
    adj_close: Optional[Series[float]] = pa.Field(nullable=True, gt=0)
    shares_outstanding: Optional[Series[float]] = pa.Field(nullable=True, gt=0)

    class Config:
        coerce = True
        strict = False
        ordered = False


class ValuationOutputSchema(DataFrameModel):
    """Schema for valuation.csv output."""

    ticker: Series[str] = pa.Field(nullable=False)
    intrinsic_value: Optional[Series[float]] = pa.Field(nullable=True)
    wacc_used: Optional[Series[float]] = pa.Field(nullable=True, ge=0.0, le=0.30)
    terminal_growth: Optional[Series[float]] = pa.Field(nullable=True, le=0.02)
    mos: Optional[Series[float]] = pa.Field(nullable=True, ge=-1.0, le=1.0)

    class Config:
        coerce = True
        strict = False
        ordered = False


class DecisionOutputSchema(DataFrameModel):
    """Schema for decision.csv — the final output contract."""

    ticker: Series[str] = pa.Field(nullable=False)
    decision: Series[str] = pa.Field(isin=["KANDIDAT", "CASH", "MANUAL_REVIEW"])
    reason: Series[str] = pa.Field(nullable=False)

    class Config:
        coerce = True
        strict = False
        ordered = False
