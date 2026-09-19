"""Evaluate a selected ESN backend against the last-close baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from advisor.data import CsvMarketDataProvider
from advisor.evaluation import compare_esn_with_baseline, compare_reservoirpy_with_baseline
from advisor.forecasting import ESNConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT"])
    parser.add_argument("--min-train-size", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--reservoir-size", type=int, default=50)
    parser.add_argument("--washout", type=int, default=20)
    parser.add_argument("--backend", choices=["numpy", "reservoirpy"], default="numpy")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = CsvMarketDataProvider(args.dataset).load(args.tickers)
    config = ESNConfig(reservoir_size=args.reservoir_size, washout=args.washout)
    compare = compare_reservoirpy_with_baseline if args.backend == "reservoirpy" else compare_esn_with_baseline
    predictions, metrics = compare(data, config, args.min_train_size, args.max_steps)
    print(metrics.to_string(index=False))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output_dir / "forecast_predictions.csv", index=False)
        metrics.to_csv(args.output_dir / "forecast_metrics.csv", index=False)


if __name__ == "__main__":
    main()
