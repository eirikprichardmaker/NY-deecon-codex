from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class FundamentalsRequest:
    asof: str


class FundamentalsSource(Protocol):
    """Adapter contract for fundamentals provider (Børsdata now, Yahoo later)."""

    def fetch_history(self, req: FundamentalsRequest) -> pd.DataFrame:
        """Return normalized fundamentals history as DataFrame.

        Required columns (minimum):
        - ticker
        - yahoo_ticker
        - period_end
        """


class BorsdataFundamentalsSource:
    """Current provider adapter skeleton.

    TODO:
      - delegate to ingest_fundamentals_history.py/freeze scripts
      - enforce schema contract before returning
    """

    def fetch_history(self, req: FundamentalsRequest) -> pd.DataFrame:
        raise NotImplementedError("TODO: wire Børsdata ingest pipeline into adapter")


class MockFundamentalsSource:
    """Simple in-memory source for tests and dry-runs."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame.copy()

    def fetch_history(self, req: FundamentalsRequest) -> pd.DataFrame:
        out = self._frame.copy()
        if "asof" not in out.columns:
            out["asof"] = req.asof
        return out
