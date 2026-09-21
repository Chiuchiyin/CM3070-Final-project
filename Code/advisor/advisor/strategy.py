"""Simple allocation strategies with explicit constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_forecasts(forecasts: pd.DataFrame) -> list[str]:
    required = {"ticker", "predicted_return"}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Forecasts are missing columns: {', '.join(sorted(missing))}")
    if forecasts.empty:
        raise ValueError("Cannot allocate an empty ticker universe")
    if forecasts["ticker"].duplicated().any():
        raise ValueError("Forecasts must contain one row per ticker")
    predicted_returns = pd.to_numeric(forecasts["predicted_return"], errors="coerce")
    if not np.isfinite(predicted_returns.to_numpy(dtype=float)).all():
        raise ValueError("Forecast predicted returns must be finite numbers")
    return sorted(forecasts["ticker"].astype(str).tolist())


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


class ForecastRankedStrategy:
    """Equally weight the highest-ranked forecast returns."""

    name = "forecast_ranked"

    def __init__(
        self, top_k: int = 3, minimum_predicted_return: float | None = None
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.minimum_predicted_return = minimum_predicted_return

    def allocate(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        tickers = _validate_forecasts(forecasts)
        ranked = forecasts[["ticker", "predicted_return"]].copy()
        ranked["ticker"] = ranked["ticker"].astype(str)
        ranked["predicted_return"] = pd.to_numeric(
            ranked["predicted_return"], errors="raise"
        ).astype(float)
        if self.minimum_predicted_return is not None:
            ranked = ranked[
                ranked["predicted_return"] >= self.minimum_predicted_return
            ]
        ranked = ranked.sort_values(
            ["predicted_return", "ticker"], ascending=[False, True]
        ).head(self.top_k)

        selected = set(ranked["ticker"])
        selected_weight = 1.0 / len(selected) if selected else 0.0
        result = pd.DataFrame(
            {
                "ticker": tickers,
                "weight": [
                    selected_weight if ticker in selected else 0.0
                    for ticker in tickers
                ],
            }
        )
        result["strategy_name"] = self.name
        result["cash_weight"] = 0.0 if selected else 1.0
        return result
