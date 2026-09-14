"""Grounded narrative generation with a deterministic fallback."""

from __future__ import annotations

import pandas as pd


def template_explanation(forecasts: pd.DataFrame, allocations: pd.DataFrame) -> str:
    symbols = ", ".join(forecasts["ticker"].tolist())
    allocation = "; ".join(f"{row.ticker}: {row.weight:.1%}" for row in allocations.itertuples())
    date = pd.Timestamp(forecasts["as_of_date"].max()).date()
    model_names = sorted(forecasts["model_name"].unique())
    model = "last-close baseline" if model_names == ["last_close"] else ", ".join(model_names)
    forecast_summary = "; ".join(
        f"{row.ticker}: {row.predicted_close:.2f} ({row.predicted_return:+.2%})"
        for row in forecasts.itertuples()
    )
    return (f"As of {date}, the {model} analysed {symbols}. "
            f"Its next-close estimates are {forecast_summary}. "
            f"The illustrative equal-weight allocation is {allocation}. "
            "This is a historical-data demonstration, not personalized financial advice.")
