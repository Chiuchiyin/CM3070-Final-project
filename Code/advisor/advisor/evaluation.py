"""Chronological evaluation for one-step forecasting models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .forecasting import EchoStateNetwork, ESNConfig, LastCloseModel


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
