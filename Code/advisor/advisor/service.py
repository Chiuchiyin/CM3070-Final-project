"""Orchestration boundary for the unified advisor."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .explanation import template_explanation
from .forecasting import LastCloseForecaster, MovingAverageForecaster
from .strategy import EqualWeightStrategy
from .strategy import ForecastRankedStrategy
from .backtesting import CASH, BacktestConfig, EqualWeightPolicy, ForecastRankedPolicy, run_backtest


@dataclass
class AnalysisResult:
    forecasts: pd.DataFrame
    allocations: pd.DataFrame
    explanation: str
    explanation_source: str = "template"
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
        allocations = self._allocate(market_data, forecasts)
        # Preserve any cash selected by the strategy while enforcing the
        # selected profile's minimum cash floor.
        strategy_cash = float(allocations["cash_weight"].iloc[0]) if "cash_weight" in allocations else 0.0
        cash_minimum = self.RISK_PROFILES[risk_profile]["cash_minimum"] if requested_profile else 0.0
        cash_weight = max(strategy_cash, cash_minimum)
        allocations = allocations.copy()
        asset_weights = allocations["weight"].astype(float)
        asset_sum = float(asset_weights.sum())
        if asset_sum > 0:
            allocations["weight"] = asset_weights * ((1.0 - cash_weight) / asset_sum)
        else:
            allocations["weight"] = 0.0
            cash_weight = 1.0
        allocations["cash_weight"] = cash_weight
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
                # Keep the equal-weight result as a benchmark while exposing
                # the forecast-ranked production strategy's historical result.
                ranked_policy = ForecastRankedPolicy(
                    MovingAverageForecaster(window=20),
                    top_k=3,
                    minimum_predicted_return=0.0,
                )
                backtest_metrics[ranked_policy.name] = run_backtest(
                    market_data, ranked_policy, self.backtest_config
                ).metrics
                if hasattr(self.strategy, "target_weights"):
                    backtest_metrics[getattr(self.strategy, "name", "production_policy")] = run_backtest(
                        market_data, self.strategy, self.backtest_config
                    ).metrics
            except ValueError as exc:
                warnings.append(f"Backtest unavailable: {exc}")
        explanation_source = "template"
        try:
            explanation_kwargs = dict(
                risk_profile=risk_profile, warnings=warnings,
                backtest_metrics=backtest_metrics,
            )
            if hasattr(self.explainer, "explain_with_status"):
                explanation, explanation_source = self.explainer.explain_with_status(
                    forecasts, allocations, model_version=str(forecasts["model_version"].iloc[0]),
                    dataset_version=self.dataset_version, **explanation_kwargs,
                )
            else:
                explanation = self.explainer(forecasts, allocations, **explanation_kwargs)
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            explanation = self.explainer(forecasts, allocations)
        model_version = str(forecasts["model_version"].iloc[0])
        return AnalysisResult(
            forecasts=forecasts, allocations=allocations, explanation=explanation,
            explanation_source=explanation_source,
            market_data=market_data, as_of_date=pd.Timestamp(forecasts["as_of_date"].max()),
            risk_profile=risk_profile, warnings=warnings, model_version=model_version,
            dataset_version=self.dataset_version, allocation_changes=changes,
            backtest_metrics=backtest_metrics,
        )

    def _allocate(self, market_data: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Support both forecast allocators and saved-policy adapters."""
        if hasattr(self.strategy, "allocate"):
            return self.strategy.allocate(forecasts)

        if not hasattr(self.strategy, "target_weights"):
            raise TypeError("Strategy must provide allocate() or target_weights()")

        tickers = sorted(market_data["ticker"].astype(str).str.upper().unique())
        current_weights = pd.Series(0.0, index=tickers + [CASH], dtype=float)
        current_weights[CASH] = 1.0
        target = self.strategy.target_weights(market_data.copy(), current_weights)
        if target is None:
            target = pd.Series({CASH: 1.0}, dtype=float)
        target = target.astype(float).reindex(tickers + [CASH], fill_value=0.0)
        result = pd.DataFrame({"ticker": tickers, "weight": target.loc[tickers].to_numpy()})
        result["strategy_name"] = getattr(self.strategy, "name", "saved_policy")
        result["cash_weight"] = float(target.get(CASH, 0.0))
        return result
