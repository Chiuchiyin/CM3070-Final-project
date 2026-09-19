"""Chronological, long-only portfolio backtesting with explicit trading costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .data import validate_market_data

CASH = "CASH"


class PortfolioPolicy(Protocol):
    """Contract shared by baseline policies and future saved model adapters."""

    name: str

    def target_weights(
        self, history: pd.DataFrame, current_weights: pd.Series
    ) -> pd.Series | None:
        """Return asset weights for the next period, or None to keep holdings."""


@dataclass(frozen=True)
class BacktestConfig:
    initial_value: float = 10_000.0
    transaction_cost_bps: float = 0.0
    slippage_bps: float = 0.0
    periods_per_year: int = 252
    annual_risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_value <= 0:
            raise ValueError("initial_value must be positive")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Trading costs and slippage must be non-negative")
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")


@dataclass
class BacktestResult:
    strategy_name: str
    returns: pd.DataFrame
    weights: pd.DataFrame
    metrics: dict[str, float]


class EqualWeightPolicy:
    """Rebalance to equal asset weights every N decision dates."""

    name = "equal_weight"

    def __init__(self, rebalance_every: int = 1):
        if rebalance_every <= 0:
            raise ValueError("rebalance_every must be positive")
        self.rebalance_every = rebalance_every

    def target_weights(
        self, history: pd.DataFrame, current_weights: pd.Series
    ) -> pd.Series | None:
        step = history["date"].nunique() - 1
        if step % self.rebalance_every:
            return None
        tickers = sorted(history["ticker"].unique())
        return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)


class BuyAndHoldPolicy:
    """Buy an equal-weight basket once and allow its weights to drift."""

    name = "buy_and_hold"

    def target_weights(
        self, history: pd.DataFrame, current_weights: pd.Series
    ) -> pd.Series | None:
        if float(current_weights.drop(labels=CASH).sum()) > 1e-12:
            return None
        tickers = sorted(history["ticker"].unique())
        return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)


def validate_weights(weights: pd.Series, tickers: list[str]) -> pd.Series:
    """Return canonical asset-plus-cash weights after enforcing constraints."""
    if not isinstance(weights, pd.Series):
        raise ValueError("Policy weights must be a pandas Series")
    if weights.index.has_duplicates:
        raise ValueError("Policy weights contain duplicate tickers")
    unknown = set(weights.index) - set(tickers) - {CASH}
    if unknown:
        raise ValueError(f"Policy returned unknown tickers: {', '.join(sorted(unknown))}")
    canonical = weights.astype(float).reindex(tickers + [CASH], fill_value=0.0)
    if not np.isfinite(canonical.to_numpy()).all():
        raise ValueError("Policy weights must be finite")
    if (canonical < -1e-12).any():
        raise ValueError("Policy weights must be long-only")
    canonical[canonical.abs() < 1e-12] = 0.0
    asset_sum = float(canonical.drop(labels=CASH).sum())
    if CASH not in weights.index:
        canonical[CASH] = 1.0 - asset_sum
    if canonical[CASH] < -1e-12 or abs(float(canonical.sum()) - 1.0) > 1e-9:
        raise ValueError("Asset and cash weights must sum to one")
    canonical[CASH] = max(0.0, float(canonical[CASH]))
    return canonical


def calculate_metrics(
    period_returns: pd.Series,
    portfolio_values: pd.Series,
    turnover: pd.Series,
    config: BacktestConfig,
) -> dict[str, float]:
    """Calculate commonly reported portfolio performance and risk metrics."""
    count = len(period_returns)
    cumulative_return = float(portfolio_values.iloc[-1] / config.initial_value - 1.0)
    if 1.0 + cumulative_return <= 0:
        annual_return = -1.0
    else:
        annual_return = float((1.0 + cumulative_return) ** (config.periods_per_year / count) - 1.0)

    volatility = float(period_returns.std(ddof=1) * np.sqrt(config.periods_per_year)) if count > 1 else 0.0
    risk_free_period = (1.0 + config.annual_risk_free_rate) ** (1.0 / config.periods_per_year) - 1.0
    excess = period_returns - risk_free_period
    period_std = float(excess.std(ddof=1)) if count > 1 else 0.0
    sharpe = float(excess.mean() / period_std * np.sqrt(config.periods_per_year)) if period_std > 0 else np.nan
    downside_deviation = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)))
    sortino = (
        float(excess.mean() / downside_deviation * np.sqrt(config.periods_per_year))
        if downside_deviation > 0
        else np.nan
    )

    value_path = pd.concat([pd.Series([config.initial_value]), portfolio_values], ignore_index=True)
    drawdown = value_path / value_path.cummax() - 1.0
    return {
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": float(drawdown.min()),
        "total_turnover": float(turnover.sum()),
        "average_period_turnover": float(turnover.mean()),
        "final_value": float(portfolio_values.iloc[-1]),
    }


def run_backtest(
    market_data: pd.DataFrame,
    policy: PortfolioPolicy,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a close-to-close simulation without exposing future rows to the policy."""
    config = config or BacktestConfig()
    data = validate_market_data(market_data)
    tickers = sorted(data["ticker"].unique())
    close = data.pivot(index="date", columns="ticker", values="close").sort_index()
    close = close.reindex(columns=tickers)
    if len(close) < 2:
        raise ValueError("Backtesting requires at least two dates")
    if close.isna().any().any():
        raise ValueError("Backtesting requires a complete ticker/date price panel")

    current_weights = pd.Series(0.0, index=tickers + [CASH], dtype=float)
    current_weights[CASH] = 1.0
    portfolio_value = config.initial_value
    total_cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000.0
    return_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    dates = close.index.to_list()
    for index, decision_date in enumerate(dates[:-1]):
        history = data[data["date"] <= decision_date].copy()
        proposed = policy.target_weights(history, current_weights.copy())
        if proposed is None:
            target = current_weights.copy()
            turnover = 0.0
        else:
            target = validate_weights(proposed, tickers)
            turnover = float(0.5 * np.abs(target - current_weights).sum())

        for ticker, weight in target.items():
            weight_rows.append(
                {"date": decision_date, "ticker": ticker, "weight": float(weight)}
            )

        next_date = dates[index + 1]
        asset_returns = close.loc[next_date] / close.loc[decision_date] - 1.0
        gross_return = float((target.drop(labels=CASH) * asset_returns).sum())
        cost = portfolio_value * turnover * total_cost_rate
        next_value = (portfolio_value - cost) * (1.0 + gross_return)
        net_return = next_value / portfolio_value - 1.0
        return_rows.append(
            {
                "period_start": decision_date,
                "date": next_date,
                "gross_return": gross_return,
                "net_return": net_return,
                "portfolio_value": next_value,
                "turnover": turnover,
                "trading_cost": cost,
            }
        )

        growth = 1.0 + gross_return
        if growth <= 0:
            raise ValueError("Portfolio value was exhausted during the backtest")
        current_weights = target.copy()
        current_weights.loc[tickers] = target.loc[tickers] * (1.0 + asset_returns) / growth
        current_weights[CASH] = target[CASH] / growth
        portfolio_value = next_value

    returns = pd.DataFrame(return_rows)
    weights = pd.DataFrame(weight_rows)
    returns.insert(0, "strategy_name", policy.name)
    weights.insert(0, "strategy_name", policy.name)
    metrics = calculate_metrics(
        returns["net_return"], returns["portfolio_value"], returns["turnover"], config
    )
    return BacktestResult(policy.name, returns, weights, metrics)
