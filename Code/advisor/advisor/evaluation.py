"""Chronological evaluation for one-step forecasting models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .forecasting import EchoStateNetwork, ESNConfig, LastCloseModel, ReservoirPyESN


def regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    actual = predictions["actual"].to_numpy(dtype=float)
    predicted = predictions["predicted"].to_numpy(dtype=float)
    previous = predictions["previous"].to_numpy(dtype=float)
    error = actual - predicted
    nonzero = actual != 0
    denominator = np.sum((actual - actual.mean()) ** 2)
    return {
        "observations": float(len(actual)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan"),
        "r2": float(1 - np.sum(error ** 2) / denominator) if denominator else float("nan"),
        "directional_accuracy": float(np.mean(np.sign(predicted - previous) == np.sign(actual - previous))),
    }


def walk_forward_predictions(
    market_data: pd.DataFrame,
    model_factory: Callable[[], object],
    model_name: str,
    min_train_size: int,
    max_steps: int | None = None,
) -> pd.DataFrame:
    """Refit using past observations only before every one-step prediction."""
    records = []
    for ticker, group in market_data.sort_values("date").groupby("ticker"):
        group = group.reset_index(drop=True)
        first_target = max(min_train_size, 1)
        if max_steps is not None:
            first_target = max(first_target, len(group) - max_steps)
        for target_index in range(first_target, len(group)):
            training = group.loc[: target_index - 1, "close"].to_numpy(dtype=float)
            model = model_factory().fit(training)
            records.append({
                "ticker": ticker,
                "date": group.loc[target_index, "date"],
                "model_name": model_name,
                "training_observations": len(training),
                "previous": float(training[-1]),
                "actual": float(group.loc[target_index, "close"]),
                "predicted": float(model.predict_next()),
            })
    if not records:
        raise ValueError("No evaluation rows; reduce min_train_size or provide more data")
    return pd.DataFrame.from_records(records)


def compare_esn_with_baseline(
    market_data: pd.DataFrame,
    config: ESNConfig | None = None,
    min_train_size: int = 100,
    max_steps: int | None = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or ESNConfig()
    baseline = walk_forward_predictions(
        market_data, LastCloseModel, LastCloseModel.name, min_train_size, max_steps
    )
    esn = walk_forward_predictions(
        market_data, lambda: EchoStateNetwork(config), EchoStateNetwork.name,
        max(min_train_size, config.washout + 3), max_steps,
    )
    predictions = pd.concat([baseline, esn], ignore_index=True)
    metric_rows = []
    for (model_name, ticker), group in predictions.groupby(["model_name", "ticker"]):
        metric_rows.append({"model_name": model_name, "ticker": ticker, **regression_metrics(group)})
    per_ticker = pd.DataFrame(metric_rows)
    aggregate_rows = []
    metric_columns = [column for column in per_ticker.columns if column not in ("model_name", "ticker", "observations")]
    for model_name, group in per_ticker.groupby("model_name"):
        aggregate_rows.append({
            "model_name": model_name,
            "ticker": "ALL",
            "observations": float(group["observations"].sum()),
            **{column: float(group[column].mean()) for column in metric_columns},
        })
    metrics = pd.concat([per_ticker, pd.DataFrame(aggregate_rows)], ignore_index=True)
    return predictions, metrics.sort_values(["ticker", "model_name"]).reset_index(drop=True)


def compare_reservoirpy_with_baseline(
    market_data: pd.DataFrame,
    config: ESNConfig | None = None,
    min_train_size: int = 100,
    max_steps: int | None = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the optional ReservoirPy ESN against the last-close baseline."""
    config = config or ESNConfig()
    baseline = walk_forward_predictions(
        market_data, LastCloseModel, LastCloseModel.name, min_train_size, max_steps
    )
    reservoirpy = walk_forward_predictions(
        market_data,
        lambda: ReservoirPyESN(config),
        ReservoirPyESN.name,
        max(min_train_size, config.washout + 3),
        max_steps,
    )
    predictions = pd.concat([baseline, reservoirpy], ignore_index=True)
    metric_rows = []
    for (model_name, ticker), group in predictions.groupby(["model_name", "ticker"]):
        metric_rows.append({"model_name": model_name, "ticker": ticker, **regression_metrics(group)})
    per_ticker = pd.DataFrame(metric_rows)
    aggregate_rows = []
    metric_columns = [column for column in per_ticker.columns if column not in ("model_name", "ticker", "observations")]
    for model_name, group in per_ticker.groupby("model_name"):
        aggregate_rows.append({
            "model_name": model_name,
            "ticker": "ALL",
            "observations": float(group["observations"].sum()),
            **{column: float(group[column].mean()) for column in metric_columns},
        })
    metrics = pd.concat([per_ticker, pd.DataFrame(aggregate_rows)], ignore_index=True)
    return predictions, metrics.sort_values(["ticker", "model_name"]).reset_index(drop=True)


def evaluate_seeds(
    market_data: pd.DataFrame,
    config: ESNConfig,
    seeds: list[int],
    *,
    backend: str = "numpy",
    min_train_size: int = 100,
    max_steps: int | None = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Walk forward over identical targets for each seed and report dispersion."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty list of unique integers")
    compare = {
        "numpy": compare_esn_with_baseline,
        "reservoirpy": compare_reservoirpy_with_baseline,
    }.get(backend)
    if compare is None:
        raise ValueError("backend must be numpy or reservoirpy")
    prediction_parts = []
    metric_parts = []
    expected_targets = None
    for seed in seeds:
        predictions, metrics = compare(
            market_data, replace(config, seed=int(seed)), min_train_size, max_steps
        )
        targets = predictions[["ticker", "date", "model_name"]].sort_values(
            ["ticker", "date", "model_name"]
        ).reset_index(drop=True)
        if expected_targets is None:
            expected_targets = targets
        elif not targets.equals(expected_targets):
            raise ValueError("Seeds were evaluated on different target dates")
        prediction_parts.append(predictions.assign(seed=int(seed)))
        metric_parts.append(metrics.assign(seed=int(seed)))
    all_predictions = pd.concat(prediction_parts, ignore_index=True)
    all_metrics = pd.concat(metric_parts, ignore_index=True)
    metric_names = ["mae", "rmse", "mape", "r2", "directional_accuracy"]
    summary = all_metrics.groupby(["model_name", "ticker"], as_index=False)[metric_names].agg(
        ["mean", "std"]
    )
    summary.columns = [
        "_".join(part for part in column if part) if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    summary = summary.rename(columns={"model_name_": "model_name", "ticker_": "ticker"})
    summary = summary.fillna({f"{name}_std": 0.0 for name in metric_names})
    return all_predictions, all_metrics, summary


def write_forecast_evaluation(
    output_dir: str | Path,
    market_data: pd.DataFrame,
    config: ESNConfig,
    seeds: list[int],
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    backend: str,
    min_train_size: int,
    max_steps: int | None,
) -> Path:
    """Write deterministic evaluation tables and provenance metadata."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(directory / "forecast_predictions.csv", index=False)
    metrics.to_csv(directory / "forecast_metrics.csv", index=False)
    summary.to_csv(directory / "forecast_seed_summary.csv", index=False)
    canonical = market_data.sort_values(["date", "ticker"]).to_csv(
        index=False, lineterminator="\n"
    )
    metadata = {
        "artifact_schema_version": 1,
        "backend": backend,
        "model_name": "numpy_esn" if backend == "numpy" else "reservoirpy_esn",
        "model_version": "numpy_esn_v1" if backend == "numpy" else "reservoirpy_esn_v1",
        "configuration": asdict(config),
        "seeds": [int(seed) for seed in seeds],
        "min_train_size": int(min_train_size),
        "max_steps": max_steps,
        "data_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "data_start": str(pd.Timestamp(market_data["date"].min()).date()),
        "data_end": str(pd.Timestamp(market_data["date"].max()).date()),
        "evaluation_start": str(pd.Timestamp(predictions["date"].min()).date()),
        "evaluation_end": str(pd.Timestamp(predictions["date"].max()).date()),
        "tickers": sorted(market_data["ticker"].unique().tolist()),
        "metrics": summary.to_dict(orient="records"),
    }
    path = directory / "forecast_evaluation.metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
