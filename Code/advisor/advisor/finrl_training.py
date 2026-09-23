"""Selected FinRL A2C training contract."""

from pathlib import Path
from dataclasses import dataclass
from datetime import date

import yaml
import pandas as pd

from .data import normalize_columns


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

