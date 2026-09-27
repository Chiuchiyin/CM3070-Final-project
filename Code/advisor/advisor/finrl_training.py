"""Selected A2C portfolio training contract and reproducible trainer."""

from pathlib import Path
from dataclasses import dataclass, replace
from datetime import date
import hashlib
import json

import yaml
import pandas as pd

from .data import normalize_columns

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional training dependency
    gym = None
    spaces = None


@dataclass(frozen=True)
class StrategyConfig:
    framework: str
    algorithm: str
    source_notebook: str
    observation_schema: str
    lookback: int
    include_cash: bool
    reward: str
    reward_scaling: float
    total_timesteps: int
    train_end: str
    validation_end: str
    test_end: str
    model_path: Path
    metadata_path: Path


def override_strategy_config(
    strategy_config: StrategyConfig,
    *,
    train_end: str | None = None,
    validation_end: str | None = None,
    test_end: str | None = None,
    lookback: int | None = None,
    total_timesteps: int | None = None,
) -> StrategyConfig:
    """Return a reproducible strategy variant for an available data period.

    The checked-in MVP dates describe the intended 2025 evaluation. Legacy
    data can still be used for a separately labelled training run by
    overriding the chronological boundaries at the CLI. The normal
    ``train_a2c`` validation remains in force, so invalid or empty splits are
    rejected rather than silently producing an artifact.
    """
    updates = {
        name: value
        for name, value in {
            "train_end": train_end,
            "validation_end": validation_end,
            "test_end": test_end,
        }.items()
        if value is not None
    }
    if lookback is not None:
        if int(lookback) <= 0:
            raise ValueError("lookback must be positive")
        updates["lookback"] = int(lookback)
    if total_timesteps is not None:
        if int(total_timesteps) <= 0:
            raise ValueError("total_timesteps must be positive")
        updates["total_timesteps"] = int(total_timesteps)
    return replace(strategy_config, **updates)


def load_strategy_config(path: str | Path) -> StrategyConfig:
    path = Path(path).resolve()
    section = yaml.safe_load(path.read_text(encoding="utf-8"))["strategy"]
    return StrategyConfig(
        framework=section["framework"], algorithm=section["algorithm"],
        source_notebook=section["source_notebook"],
        observation_schema=section["observation_schema"],
        lookback=int(section["lookback"]), include_cash=bool(section["include_cash"]),
        reward=section["reward"], reward_scaling=float(section["reward_scaling"]),
        total_timesteps=int(section["total_timesteps"]),
        train_end=section["train_end"], validation_end=section["validation_end"],
        test_end=section["test_end"],
        model_path=(path.parent / section["model_path"]).resolve(),
        metadata_path=(path.parent / section["metadata_path"]).resolve(),
    )


def chronological_partitions(market_data, train_end, validation_end, test_end):
    # Partitioning is a boundary helper.  Full OHLCV validation happens when
    # data enters the provider; keeping this helper focused also lets it be
    # used with compact FinRL training frames whose synthetic high/low values
    # are not guaranteed to envelope every close.
    data = normalize_columns(market_data).sort_values(["date", "ticker"]).reset_index(drop=True)
    ends = [date.fromisoformat(value) for value in (train_end, validation_end, test_end)]
    timestamps = [pd.Timestamp(value) for value in ends]
    if not ends[0] < ends[1] or not ends[1] < ends[2]:
        raise ValueError("Strategy dates must satisfy train < validation < test")
    partitions = {
        "train": data[data["date"] <= timestamps[0]],
        "validation": data[(data["date"] > timestamps[0]) & (data["date"] <= timestamps[1])],
        "test": data[(data["date"] > timestamps[1]) & (data["date"] <= timestamps[2])],
    }
    if any(frame.empty for frame in partitions.values()):
        raise ValueError("Train, validation, and test partitions must be non-empty")
    return {name: frame.reset_index(drop=True) for name, frame in partitions.items()}




def _stable_softmax(values):
    import numpy as np
    logits = np.asarray(values, dtype=float).reshape(-1)
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


class PortfolioAllocationEnv(gym.Env if gym is not None else object):
    """Gymnasium environment for target-weight portfolio allocation.

    Observations contain rolling per-asset close returns followed by current
    asset weights and cash. Actions are one logit per asset plus cash and are
    converted to target weights by stable softmax.
    """

    metadata = {"render_modes": []}

    def __init__(self, market_data, tickers, lookback=20, transaction_cost_bps=10.0,
                 slippage_bps=5.0, reward_scaling=100.0):
        if gym is None or spaces is None:
            raise RuntimeError("gymnasium is required for A2C training")
        import numpy as np
        self.np = np
        self.tickers = sorted(str(t).upper() for t in tickers)
        self.lookback = int(lookback)
        if self.lookback <= 0 or not self.tickers:
            raise ValueError("lookback and tickers must be positive")
        data = normalize_columns(market_data)
        close = data.pivot(index="date", columns="ticker", values="close").sort_index()
        close = close.reindex(columns=self.tickers)
        if close.isna().any().any():
            raise ValueError("Training data must contain a complete ticker/date panel")
        if len(close) <= self.lookback + 1:
            raise ValueError("Training data does not contain enough dates for the lookback")
        self.close_prices = close
        self.returns = close.pct_change().fillna(0.0)
        self.transaction_cost_rate = (float(transaction_cost_bps) + float(slippage_bps)) / 10000.0
        self.reward_scaling = float(reward_scaling)
        observation_size = len(self.tickers) * self.lookback + len(self.tickers) + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(observation_size,), dtype=np.float32)
        # Stable-Baselines3 requires finite continuous action bounds. These
        # logits are mapped to simplex weights by softmax in step().
        self.action_space = spaces.Box(-10.0, 10.0, shape=(len(self.tickers) + 1,), dtype=np.float32)
        self.reset()

    def _observation(self):
        window = self.returns.iloc[self.index - self.lookback + 1:self.index + 1]
        values = window.to_numpy(dtype=float).T.reshape(-1)
        return self.np.concatenate((values, self.weights)).astype(self.np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = self.lookback
        self.weights = self.np.zeros(len(self.tickers) + 1, dtype=float)
        self.weights[-1] = 1.0
        self.portfolio_value = 1.0
        return self._observation(), {}

    def step(self, action):
        target = _stable_softmax(action)
        turnover = 0.5 * float(self.np.abs(target - self.weights).sum())
        cost = turnover * self.transaction_cost_rate
        asset_returns = self.returns.iloc[self.index + 1].to_numpy(dtype=float)
        gross_return = float((target[:-1] * asset_returns).sum())
        growth = (1.0 - cost) * (1.0 + gross_return)
        if growth <= 0 or not self.np.isfinite(growth):
            raise ValueError("Portfolio value became non-positive during training")
        reward = float(self.np.log(growth) * self.reward_scaling)
        self.portfolio_value *= growth
        post_cost_growth = 1.0 + gross_return
        next_weights = target.copy()
        next_weights[:-1] = target[:-1] * (1.0 + asset_returns) / post_cost_growth
        next_weights[-1] = target[-1] / post_cost_growth
        self.weights = next_weights / next_weights.sum()
        self.index += 1
        terminated = self.index >= len(self.close_prices.index) - 1
        return self._observation(), reward, terminated, False, {
            "turnover": turnover,
            "trading_cost": cost,
            "portfolio_value": self.portfolio_value,
        }


def dataset_sha256(market_data) -> str:
    """Return a stable digest for canonical training rows."""
    canonical = normalize_columns(market_data).sort_values(["date", "ticker"]).to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def train_a2c(market_data, strategy_config, *, seed=42, tickers=None,
              output_path=None, metadata_path=None, transaction_cost_bps=10.0,
              slippage_bps=5.0):
    """Train and save one A2C artifact using only the chronological train split."""
    data = normalize_columns(market_data)
    universe = sorted(tickers or data["ticker"].astype(str).str.upper().unique().tolist())
    configured_test_end = strategy_config.test_end
    if not isinstance(configured_test_end, date):
        configured_test_end = date.fromisoformat(str(configured_test_end))
    if data["date"].max().date() < configured_test_end:
        raise ValueError(
            f"Dataset ends {data['date'].max().date()}, before configured test_end "
            f"{strategy_config.test_end}; use a dataset covering all configured splits"
        )
    partitions = chronological_partitions(
        data, strategy_config.train_end, strategy_config.validation_end, strategy_config.test_end
    )
    try:
        from stable_baselines3 import A2C
    except ImportError as exc:
        raise RuntimeError("stable-baselines3 is required for A2C training") from exc
    train_data = partitions["train"]
    env = PortfolioAllocationEnv(
        train_data, universe, lookback=strategy_config.lookback,
        transaction_cost_bps=transaction_cost_bps, slippage_bps=slippage_bps,
        reward_scaling=strategy_config.reward_scaling,
    )
    model = A2C("MlpPolicy", env, seed=seed, verbose=0,
                n_steps=min(5, max(1, len(train_data))))
    model.learn(total_timesteps=strategy_config.total_timesteps)
    target = Path(output_path or strategy_config.model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(target.with_suffix("")))
    metadata = {
        "schema_version": 1,
        "algorithm": "A2C",
        "policy": "MlpPolicy",
        "tickers": universe,
        "observation_schema": strategy_config.observation_schema,
        "lookback": strategy_config.lookback,
        "include_cash": strategy_config.include_cash,
        "reward": strategy_config.reward,
        "reward_scaling": strategy_config.reward_scaling,
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "seed": seed,
        "total_timesteps": strategy_config.total_timesteps,
        "split_config": {
            "train_end": str(strategy_config.train_end),
            "validation_end": str(strategy_config.validation_end),
            "test_end": str(strategy_config.test_end),
        },
        "data_sha256": dataset_sha256(data),
        "partitions": {
            name: {"start": str(frame["date"].min().date()),
                   "end": str(frame["date"].max().date()), "rows": len(frame)}
            for name, frame in partitions.items()
        },
    }
    metadata_target = Path(metadata_path or strategy_config.metadata_path)
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    metadata_target.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    if hasattr(env, "close"):
        env.close()
    return target.with_suffix(".zip"), metadata_target, metadata
