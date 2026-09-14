"""Orchestration boundary for the unified advisor."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .explanation import template_explanation
from .forecasting import LastCloseForecaster
from .strategy import EqualWeightStrategy


@dataclass
class AnalysisResult:
    forecasts: pd.DataFrame
    allocations: pd.DataFrame
    explanation: str


class AdvisorService:
    def __init__(self, data_provider, forecaster=None, strategy=None, explainer=template_explanation):
        self.data_provider = data_provider
        self.forecaster = forecaster or LastCloseForecaster()
        self.strategy = strategy or EqualWeightStrategy()
        self.explainer = explainer

    def analyse(self, tickers, start=None, end=None) -> AnalysisResult:
        market_data = self.data_provider.load(tickers=tickers, start=start, end=end)
        forecasts = self.forecaster.predict(market_data)
        allocations = self.strategy.allocate(forecasts)
        if abs(float(allocations["weight"].sum() + allocations["cash_weight"].iloc[0]) - 1.0) > 1e-9:
            raise ValueError("Allocation weights must sum to one")
        return AnalysisResult(forecasts, allocations, self.explainer(forecasts, allocations))
