"""Grounded narrative generation with a deterministic fallback.

The template only formats values supplied by the advisor service.  It never
derives prices, metrics, or allocations, which keeps the explanation layer
safe to replace with an optional language model later.
"""

from __future__ import annotations

import pandas as pd


def template_explanation(
    forecasts: pd.DataFrame,
    allocations: pd.DataFrame,
    *,
    risk_profile: str = "moderate",
    warnings: list[str] | None = None,
    backtest_metrics: dict[str, dict[str, float]] | None = None,
) -> str:
    symbols = ", ".join(forecasts["ticker"].tolist())
    allocation = "; ".join(f"{row.ticker}: {row.weight:.1%}" for row in allocations.itertuples())
    date = pd.Timestamp(forecasts["as_of_date"].max()).date()
    model_names = sorted(forecasts["model_name"].unique())
    model = "last-close baseline" if model_names == ["last_close"] else ", ".join(model_names)
    forecast_summary = "; ".join(
        f"{row.ticker}: {row.predicted_close:.2f} ({row.predicted_return:+.2%})"
        for row in forecasts.itertuples()
    )
    cash = float(allocations["cash_weight"].iloc[0]) if "cash_weight" in allocations else 0.0
    metric_text = ""
    if backtest_metrics and "equal_weight" in backtest_metrics:
        cumulative = backtest_metrics["equal_weight"].get("cumulative_return")
        if cumulative is not None:
            metric_text = f" The equal-weight historical simulation returned {cumulative:+.2%}."
    warning_text = f" Warnings: {'; '.join(warnings)}." if warnings else ""
    return (
        f"As of {date}, the {model} analysed {symbols}. "
        f"Its next-close estimates are {forecast_summary}. "
        f"For the {risk_profile} profile, the illustrative allocation is {allocation} "
        f"with {cash:.1%} held as cash.{metric_text}{warning_text} "
        "This is an educational historical-data demonstration, not personalized financial advice."
    )
