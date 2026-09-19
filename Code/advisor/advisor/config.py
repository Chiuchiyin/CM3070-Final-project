"""Versioned configuration loading for the unified MVP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


class ConfigValidationError(ValueError):
    """Raised when the MVP configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class MVPConfig:
    """Runtime settings that affect data, models, and backtest reproducibility."""

    source_path: Path
    schema_version: int
    universe_name: str
    tickers: tuple[str, ...]
    start: str
    end: str
    cache_dir: Path
    prepared_data_path: Path
    metadata_path: Path
    forecast_backend: str
    reservoir_size: int
    washout: int
    seed: int
    initial_value: float
    transaction_cost_bps: float
    slippage_bps: float
    rebalance_every: int
    default_risk_profile: str
    allowed_risk_profiles: tuple[str, ...]

    def metadata(self) -> dict[str, object]:
        """Return the reproducibility-relevant config values for a dataset artifact."""
        return {
            "schema_version": self.schema_version,
            "universe_name": self.universe_name,
            "tickers": list(self.tickers),
            "start": self.start,
            "end": self.end,
            "forecast": {
                "backend": self.forecast_backend,
                "reservoir_size": self.reservoir_size,
                "washout": self.washout,
                "seed": self.seed,
            },
            "backtest": {
                "initial_value": self.initial_value,
                "transaction_cost_bps": self.transaction_cost_bps,
                "slippage_bps": self.slippage_bps,
                "rebalance_every": self.rebalance_every,
            },
        }


def default_mvp_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "mvp.yaml"


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{name} must be a mapping")
    return value


def _value(section: dict, key: str, section_name: str):
    if key not in section:
        raise ConfigValidationError(f"Missing {section_name}.{key}")
    return section[key]


def _resolved_path(value: object, config_path: Path, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"data.{key} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def load_mvp_config(path: str | Path | None = None) -> MVPConfig:
    """Load and validate the versioned MVP YAML configuration."""
    config_path = Path(path) if path is not None else default_mvp_config_path()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigValidationError(f"Configuration file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Invalid YAML in {config_path}: {exc}") from exc
    root = _mapping(raw, "configuration")
    if root.get("schema_version") != 1:
        raise ConfigValidationError("Only schema_version 1 is supported")

    universe = _mapping(_value(root, "universe", "configuration"), "universe")
    raw_tickers = _value(universe, "tickers", "universe")
    if not isinstance(raw_tickers, list) or not raw_tickers:
        raise ConfigValidationError("universe.tickers must be a non-empty list")
    tickers = tuple(str(ticker).upper().strip() for ticker in raw_tickers)
    if any(not ticker for ticker in tickers) or len(set(tickers)) != len(tickers):
        raise ConfigValidationError("universe.tickers must contain unique non-empty symbols")

    data = _mapping(_value(root, "data", "configuration"), "data")
    start = str(_value(data, "start", "data"))
    end = str(_value(data, "end", "data"))
    try:
        if date.fromisoformat(start) > date.fromisoformat(end):
            raise ConfigValidationError("data.start must be on or before data.end")
    except ValueError as exc:
        raise ConfigValidationError("data.start and data.end must use YYYY-MM-DD") from exc

    forecast = _mapping(_value(root, "forecast", "configuration"), "forecast")
    backend = str(_value(forecast, "backend", "forecast")).lower()
    if backend not in {"numpy", "reservoirpy"}:
        raise ConfigValidationError("forecast.backend must be numpy or reservoirpy")
    reservoir_size = int(_value(forecast, "reservoir_size", "forecast"))
    washout = int(_value(forecast, "washout", "forecast"))
    if reservoir_size < 2 or washout < 0:
        raise ConfigValidationError("forecast.reservoir_size and forecast.washout are invalid")

    backtest = _mapping(_value(root, "backtest", "configuration"), "backtest")
    initial_value = float(_value(backtest, "initial_value", "backtest"))
    transaction_cost_bps = float(_value(backtest, "transaction_cost_bps", "backtest"))
    slippage_bps = float(_value(backtest, "slippage_bps", "backtest"))
    rebalance_every = int(_value(backtest, "rebalance_every", "backtest"))
    if initial_value <= 0 or transaction_cost_bps < 0 or slippage_bps < 0 or rebalance_every <= 0:
        raise ConfigValidationError("backtest values must be positive and costs non-negative")

    application = _mapping(_value(root, "application", "configuration"), "application")
    allowed_profiles = tuple(str(profile).lower().strip() for profile in _value(
        application, "allowed_risk_profiles", "application"
    ))
    default_profile = str(_value(application, "default_risk_profile", "application")).lower().strip()
    if not allowed_profiles or default_profile not in allowed_profiles:
        raise ConfigValidationError("application.default_risk_profile must be allowed")

    return MVPConfig(
        source_path=config_path.resolve(),
        schema_version=1,
        universe_name=str(_value(universe, "name", "universe")),
        tickers=tickers,
        start=start,
        end=end,
        cache_dir=_resolved_path(_value(data, "cache_dir", "data"), config_path, "cache_dir"),
        prepared_data_path=_resolved_path(
            _value(data, "prepared_data_path", "data"), config_path, "prepared_data_path"
        ),
        metadata_path=_resolved_path(_value(data, "metadata_path", "data"), config_path, "metadata_path"),
        forecast_backend=backend,
        reservoir_size=reservoir_size,
        washout=washout,
        seed=int(_value(forecast, "seed", "forecast")),
        initial_value=initial_value,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        rebalance_every=rebalance_every,
        default_risk_profile=default_profile,
        allowed_risk_profiles=allowed_profiles,
    )
