"""Forecasting components used by the unified advisor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


class LastCloseModel:
    """Stateful form of the last-close baseline used by evaluation code."""

    name = "last_close"

    def fit(self, values) -> "LastCloseModel":
        series = np.asarray(values, dtype=float).reshape(-1)
        if not len(series):
            raise ValueError("At least one observation is required")
        self.last_value = float(series[-1])
        return self

    def predict_next(self) -> float:
        return self.last_value


class LastCloseForecaster:
    """A leakage-free baseline: tomorrow's close equals the latest observed close."""

    name = "last_close"

    def predict(self, market_data: pd.DataFrame) -> pd.DataFrame:
        latest = (market_data.sort_values("date").groupby("ticker", as_index=False).tail(1))
        result = latest[["ticker", "date", "close"]].rename(columns={"date": "as_of_date", "close": "predicted_close"})
        result["horizon"] = 1
        result["predicted_return"] = 0.0
        result["model_name"] = self.name
        result["model_version"] = "baseline_v1"
        result["training_observations"] = market_data.groupby("ticker")["close"].count().reindex(result["ticker"]).to_numpy()
        return result.reset_index(drop=True)


@dataclass(frozen=True)
class ESNConfig:
    reservoir_size: int = 50
    spectral_radius: float = 0.9
    leak_rate: float = 0.5
    input_scaling: float = 0.5
    connectivity: float = 0.15
    ridge: float = 1e-6
    washout: int = 20
    seed: int = 42

    def validate(self) -> None:
        if self.reservoir_size < 2:
            raise ValueError("reservoir_size must be at least 2")
        if not 0 < self.spectral_radius:
            raise ValueError("spectral_radius must be positive")
        if not 0 < self.leak_rate <= 1:
            raise ValueError("leak_rate must be in (0, 1]")
        if not 0 < self.connectivity <= 1:
            raise ValueError("connectivity must be in (0, 1]")
        if self.ridge < 0 or self.washout < 0:
            raise ValueError("ridge and washout must be non-negative")


class EchoStateNetwork:
    """Deterministic one-step ESN with a ridge-regression readout.

    Scaling is fitted exclusively from values supplied to ``fit``. During
    walk-forward evaluation those values are always earlier than the target.
    """

    name = "numpy_esn"
    version = "numpy_esn_v1"

    def __init__(self, config: ESNConfig | None = None):
        self.config = config or ESNConfig()
        self.config.validate()
        self._fitted = False

    def _initialize_reservoir(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        size = self.config.reservoir_size
        recurrent = rng.uniform(-0.5, 0.5, (size, size))
        recurrent[rng.random((size, size)) > self.config.connectivity] = 0.0
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        if radius == 0:
            raise ValueError("Reservoir has zero spectral radius; increase connectivity")
        self.recurrent_weights = recurrent * (self.config.spectral_radius / radius)
        self.input_weights = rng.uniform(-0.5, 0.5, (size, 2)) * self.config.input_scaling

    def _step(self, value: float, state: np.ndarray) -> np.ndarray:
        candidate = np.tanh(self.input_weights @ np.array([1.0, value]) + self.recurrent_weights @ state)
        return (1.0 - self.config.leak_rate) * state + self.config.leak_rate * candidate

    def fit(self, values) -> "EchoStateNetwork":
        series = np.asarray(values, dtype=float).reshape(-1)
        if not np.isfinite(series).all():
            raise ValueError("Training values must all be finite")
        minimum = max(self.config.washout + 3, 4)
        if len(series) < minimum:
            raise ValueError(f"At least {minimum} observations are required")
        self.scale_min = float(series.min())
        self.scale_max = float(series.max())
        scale = self.scale_max - self.scale_min
        self.scale_range = scale if scale > 0 else 1.0
        normalized = (series - self.scale_min) / self.scale_range
        self._initialize_reservoir()
        state = np.zeros(self.config.reservoir_size)
        features, targets = [], []
        for index in range(len(normalized) - 1):
            value = float(normalized[index])
            state = self._step(value, state)
            if index >= self.config.washout:
                features.append(np.concatenate(([1.0, value], state)))
                targets.append(float(normalized[index + 1]))
        design = np.asarray(features)
        target = np.asarray(targets)
        penalty = self.config.ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        self.output_weights = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        self.state = state
        self.last_scaled_value = float(normalized[-1])
        self.training_observations = len(series)
        self._fitted = True
        return self

    def predict_next(self) -> float:
        if not self._fitted:
            raise RuntimeError("Fit the ESN before requesting a prediction")
        prediction_state = self._step(self.last_scaled_value, self.state)
        features = np.concatenate(([1.0, self.last_scaled_value], prediction_state))
        normalized_prediction = float(features @ self.output_weights)
        return normalized_prediction * self.scale_range + self.scale_min

    def metadata(self) -> dict:
        if not self._fitted:
            raise RuntimeError("Fit the ESN before requesting metadata")
        return {
            "model_name": self.name,
            "model_version": self.version,
            "training_observations": self.training_observations,
            "training_min": self.scale_min,
            "training_max": self.scale_max,
            "config": asdict(self.config),
        }

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("Fit the ESN before saving it")
        np.savez_compressed(
            Path(path),
            recurrent_weights=self.recurrent_weights,
            input_weights=self.input_weights,
            output_weights=self.output_weights,
            state=self.state,
            last_scaled_value=self.last_scaled_value,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            scale_range=self.scale_range,
            training_observations=self.training_observations,
            config=json.dumps(asdict(self.config)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EchoStateNetwork":
        artifact = np.load(Path(path), allow_pickle=False)
        model = cls(ESNConfig(**json.loads(str(artifact["config"]))))
        for name in ("recurrent_weights", "input_weights", "output_weights", "state"):
            setattr(model, name, artifact[name])
        for name in ("last_scaled_value", "scale_min", "scale_max", "scale_range"):
            setattr(model, name, float(artifact[name]))
        model.training_observations = int(artifact["training_observations"])
        model._fitted = True
        return model


class ESNForecaster:
    """Adapter that exposes one ESN per ticker through the service contract."""

    name = EchoStateNetwork.name

    def __init__(self, config: ESNConfig | None = None):
        self.config = config or ESNConfig()
        self.models: dict[str, EchoStateNetwork] = {}

    def predict(self, market_data: pd.DataFrame) -> pd.DataFrame:
        records = []
        for ticker, group in market_data.sort_values("date").groupby("ticker"):
            closes = group["close"].to_numpy(dtype=float)
            model = EchoStateNetwork(self.config).fit(closes)
            predicted_close = model.predict_next()
            latest_close = float(closes[-1])
            self.models[ticker] = model
            records.append({
                "ticker": ticker,
                "as_of_date": group["date"].iloc[-1],
                "predicted_close": predicted_close,
                "horizon": 1,
                "predicted_return": predicted_close / latest_close - 1.0,
                "model_name": model.name,
                "model_version": model.version,
                "training_observations": len(closes),
            })
        return pd.DataFrame.from_records(records)
