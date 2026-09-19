"""Adapters for running saved FinRL/Stable-Baselines3 allocation policies.

The adapter deliberately does not recreate a FinRL environment. A saved policy is
valid only with the exact observation schema used during training, so callers
must provide that schema as an observation builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .backtesting import CASH


class ObservationBuilder(Protocol):
    """Build one policy observation from information available at a decision date."""

    def build(
        self, history: pd.DataFrame, tickers: list[str], current_weights: pd.Series
    ) -> np.ndarray:
        """Return the observation in the exact shape expected by the saved policy."""


@dataclass(frozen=True)
class RollingReturnObservationBuilder:
    """A simple reproducible observation schema for newly trained policies.

    The returned matrix has one row per sorted ticker and `lookback` columns of
    close-to-close returns. It is not compatible with an existing FinRL model
    unless that model was trained with this exact schema.
    """

    lookback: int = 20

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")

    @property
    def minimum_history_dates(self) -> int:
        """Number of complete dates required before this schema is available."""
        return self.lookback + 1

    def build(
        self, history: pd.DataFrame, tickers: list[str], current_weights: pd.Series
    ) -> np.ndarray:
        close = (
            history.pivot(index="date", columns="ticker", values="close")
            .sort_index()
            .reindex(columns=tickers)
        )
        if close.isna().any().any():
            raise ValueError("Observation history must contain every ticker on every date")
        if len(close) < self.lookback + 1:
            raise ValueError(
                f"Rolling-return observation requires {self.lookback + 1} dates, got {len(close)}"
            )
        returns = close.pct_change().iloc[-self.lookback :]
        return returns.to_numpy(dtype=float).T


def stable_softmax(actions: np.ndarray) -> np.ndarray:
    """Convert finite model actions into numerically stable non-negative weights."""
    values = np.asarray(actions, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Policy action contains non-finite values")
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


class FinRLPolicyAdapter:
    """Expose a saved-policy action model through the shared portfolio-policy contract.

    Legacy FinRL portfolio environments apply a softmax to their one-action-per-
    asset output. This adapter preserves that action convention by default.
    """

    name = "finrl_policy"

    def __init__(
        self,
        model,
        observation_builder: ObservationBuilder,
        tickers: list[str],
        *,
        include_cash_action: bool = False,
        deterministic: bool = True,
        name: str | None = None,
    ) -> None:
        if not tickers:
            raise ValueError("A saved policy adapter requires at least one ticker")
        canonical = [ticker.upper().strip() for ticker in tickers]
        if len(set(canonical)) != len(canonical) or any(not ticker for ticker in canonical):
            raise ValueError("Policy tickers must be unique, non-empty symbols")
        self.model = model
        self.observation_builder = observation_builder
        self.tickers = sorted(canonical)
        self.include_cash_action = include_cash_action
        self.deterministic = deterministic
        self.name = name or self.name

    @classmethod
    def from_stable_baselines3(
        cls,
        model_path: Path | str,
        algorithm: str,
        observation_builder: ObservationBuilder,
        tickers: list[str],
        **kwargs,
    ) -> "FinRLPolicyAdapter":
        """Load a saved SB3 model lazily, keeping it an optional dependency."""
        try:
            import stable_baselines3
        except ImportError as exc:
            raise RuntimeError(
                "stable-baselines3 is required to load a saved FinRL policy; "
                "install the optional FinRL/SB3 dependencies first"
            ) from exc
        try:
            model_class = getattr(stable_baselines3, algorithm.upper())
        except AttributeError as exc:
            raise ValueError(f"Unsupported Stable-Baselines3 algorithm: {algorithm}") from exc
        model = model_class.load(str(model_path))
        return cls(model, observation_builder, tickers, name=f"finrl_{algorithm.lower()}", **kwargs)

    def target_weights(
        self, history: pd.DataFrame, current_weights: pd.Series
    ) -> pd.Series | None:
        observed = sorted(history["ticker"].astype(str).str.upper().unique())
        if observed != self.tickers:
            raise ValueError(
                "Saved policy ticker universe differs from the available history: "
                f"expected {self.tickers}, got {observed}"
            )
        required_dates = getattr(self.observation_builder, "minimum_history_dates", 1)
        if history["date"].nunique() < required_dates:
            return None
        observation = self.observation_builder.build(history.copy(), self.tickers, current_weights.copy())
        prediction = self.model.predict(observation, deterministic=self.deterministic)
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        expected_size = len(self.tickers) + int(self.include_cash_action)
        if np.asarray(action).size != expected_size:
            raise ValueError(
                f"Policy emitted {np.asarray(action).size} actions; expected {expected_size}"
            )
        allocation = stable_softmax(np.asarray(action))
        if self.include_cash_action:
            return pd.Series(
                allocation, index=self.tickers + [CASH], dtype=float
            )
        return pd.Series(allocation, index=self.tickers, dtype=float)
