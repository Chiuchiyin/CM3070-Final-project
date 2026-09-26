"""Adapters for running saved FinRL/Stable-Baselines3 allocation policies.

The adapter deliberately does not recreate a FinRL environment. A saved policy is
valid only with the exact observation schema used during training, so callers
must provide that schema as an observation builder.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
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


@dataclass(frozen=True)
class RollingReturnAndWeightsObservationBuilder:
    """Observation schema used by the reproducible A2C trainer."""

    lookback: int = 20

    def __post_init__(self):
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")

    @property
    def minimum_history_dates(self) -> int:
        return self.lookback + 1

    def build(self, history: pd.DataFrame, tickers: list[str], current_weights: pd.Series) -> np.ndarray:
        close = (history.pivot(index="date", columns="ticker", values="close")
                 .sort_index().reindex(columns=tickers))
        if close.isna().any().any():
            raise ValueError("Observation history must contain every ticker on every date")
        if len(close) < self.minimum_history_dates:
            raise ValueError(f"Observation requires {self.minimum_history_dates} dates, got {len(close)}")
        returns = close.pct_change().iloc[-self.lookback:]
        weights = current_weights.reindex(tickers + [CASH], fill_value=0.0).to_numpy(dtype=float)
        return np.concatenate((returns.to_numpy(dtype=float).T.reshape(-1), weights))


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



def load_approved_finrl_policy(
    model_path: Path | str,
    metadata_path: Path | str,
    *,
    tickers: list[str],
    algorithm: str = "A2C",
    observation_builder: ObservationBuilder | None = None,
    expected_schema: str = "rolling_returns_and_weights_v1",
    model_loader=None,
) -> tuple[FinRLPolicyAdapter, dict]:
    """Load an approved policy for inference after validating its metadata.

    This function only loads an existing artifact. It never trains or mutates
    the model. Metadata checks prevent a policy trained for another universe or
    observation schema from entering the application path.
    """
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Approved policy artifact not found: {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Policy metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid policy metadata JSON: {metadata_path}") from exc
    canonical_tickers = sorted(str(t).upper().strip() for t in tickers)
    stored_tickers = sorted(str(t).upper().strip() for t in metadata.get("tickers", []))
    if stored_tickers != canonical_tickers:
        raise ValueError(
            f"Policy metadata tickers {stored_tickers} do not match requested {canonical_tickers}"
        )
    if str(metadata.get("algorithm", "")).upper() != algorithm.upper():
        raise ValueError("Policy algorithm does not match the requested algorithm")
    if metadata.get("observation_schema") != expected_schema:
        raise ValueError("Policy observation schema is not approved for this application")
    lookback = int(metadata.get("lookback", 0))
    if lookback <= 0:
        raise ValueError("Policy metadata must contain a positive lookback")
    if observation_builder is None:
        observation_builder = RollingReturnAndWeightsObservationBuilder(lookback=lookback)
    if getattr(observation_builder, "lookback", lookback) != lookback:
        raise ValueError("Observation builder lookback does not match policy metadata")
    if model_loader is None:
        adapter = FinRLPolicyAdapter.from_stable_baselines3(
            model_path, algorithm, observation_builder, canonical_tickers,
            include_cash_action=bool(metadata.get("include_cash", True)),
        )
    else:
        model = model_loader(model_path, algorithm)
        adapter = FinRLPolicyAdapter(
            model, observation_builder, canonical_tickers,
            include_cash_action=bool(metadata.get("include_cash", True)),
            name=f"finrl_{algorithm.lower()}",
        )
    return adapter, metadata
