"""Orchestration boundary for the unified advisor."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .explanation import template_explanation
from .forecasting import LastCloseForecaster
from .strategy import EqualWeightStrategy
from .backtesting import BacktestConfig, EqualWeightPolicy, run_backtest


@dataclass
class AnalysisResult:
    forecasts: pd.DataFrame
    allocations: pd.DataFrame
    explanation: str
    market_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    as_of_date: pd.Timestamp | None = None
    risk_profile: str = "moderate"
    warnings: list[str] = field(default_factory=list)
    model_version: str = "unknown"
    dataset_version: str = "runtime"
    allocation_changes: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


class AdvisorService:
    RISK_PROFILES = {
        "conservative": {"cash_minimum": 0.20},
        "moderate": {"cash_minimum": 0.10},
        "growth": {"cash_minimum": 0.00},
    }

    def __init__(self, data_provider, forecaster=None, strategy=None, explainer=template_explanation,
                 dataset_version: str = "runtime", backtest_config: BacktestConfig | None = None):
        self.data_provider = data_provider
        self.forecaster = forecaster or LastCloseForecaster()
        self.strategy = strategy or EqualWeightStrategy()
        self.explainer = explainer
        self.dataset_version = dataset_version
        self.backtest_config = backtest_config or BacktestConfig()

    def analyse(self, tickers, start=None, end=None, as_of_date=None, risk_profile=None) -> AnalysisResult:
        requested_profile = risk_profile
        risk_profile = risk_profile or "moderate"
        if risk_profile not in self.RISK_PROFILES:
            raise ValueError(f"Unknown risk profile: {risk_profile}")
        market_data = self.data_provider.load(tickers=tickers, start=start, end=end)
        if as_of_date is not None:
            requested = pd.Timestamp(as_of_date)
            market_data = market_data[market_data["date"] <= requested].copy()
            if market_data.empty:
                raise ValueError("No observations exist on or before as_of_date")
        forecasts = self.forecaster.predict(market_data)
        allocations = self.strategy.allocate(forecasts)
        # Preserve the original dependency-light API when no profile was
        # supplied: callers received a fully invested equal-weight basket.
        cash_minimum = self.RISK_PROFILES[risk_profile]["cash_minimum"] if requested_profile else 0.0
        asset_scale = 1.0 - cash_minimum
        allocations = allocations.copy()
        allocations["weight"] = allocations["weight"].astype(float) * asset_scale
        allocations["cash_weight"] = cash_minimum
        allocations["as_of_date"] = pd.Timestamp(forecasts["as_of_date"].max())
        if abs(float(allocations["weight"].sum() + allocations["cash_weight"].iloc[0]) - 1.0) > 1e-9:
            raise ValueError("Allocation weights must sum to one")
        changes = allocations.set_index("ticker")["weight"].rename("target_weight").to_frame()
        changes["current_weight"] = 0.0
        changes["change"] = changes["target_weight"]
        changes = changes.reset_index()
        warnings = []
        freshness_date = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp(market_data["date"].max())
        if freshness_date < pd.Timestamp.now().normalize() - pd.Timedelta(days=5):
            warnings.append("The selected data may be stale.")
        backtest_metrics = {}
        if market_data["date"].nunique() >= 2:
            try:
                backtest_metrics["equal_weight"] = run_backtest(
                    market_data, EqualWeightPolicy(), self.backtest_config
                ).metrics
            except ValueError as exc:
                warnings.append(f"Backtest unavailable: {exc}")
        try:
            explanation = self.explainer(
                forecasts, allocations, risk_profile=risk_profile,
                warnings=warnings, backtest_metrics=backtest_metrics,
            )
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            explanation = self.explainer(forecasts, allocations)
        model_version = str(forecasts["model_version"].iloc[0])
        return AnalysisResult(
            forecasts=forecasts, allocations=allocations, explanation=explanation,
            market_data=market_data, as_of_date=pd.Timestamp(forecasts["as_of_date"].max()),
            risk_profile=risk_profile, warnings=warnings, model_version=model_version,
            dataset_version=self.dataset_version, allocation_changes=changes,
            backtest_metrics=backtest_metrics,
        )
