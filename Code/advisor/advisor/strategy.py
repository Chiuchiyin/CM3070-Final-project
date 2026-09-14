"""Simple allocation strategies with explicit constraints."""

from __future__ import annotations

import pandas as pd


class EqualWeightStrategy:
    name = "equal_weight"

    def allocate(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        tickers = sorted(forecasts["ticker"].unique())
        if not tickers:
            raise ValueError("Cannot allocate an empty ticker universe")
        weight = 1.0 / len(tickers)
        result = pd.DataFrame({"ticker": tickers, "weight": weight})
        result["strategy_name"] = self.name
        result["cash_weight"] = 0.0
        return result
