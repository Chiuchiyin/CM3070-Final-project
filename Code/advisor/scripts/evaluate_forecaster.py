"""Evaluate a selected ESN backend against the last-close baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from advisor.data import CsvMarketDataProvider
from advisor.evaluation import evaluate_seeds, write_forecast_evaluation
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = CsvMarketDataProvider(args.dataset).load(args.tickers)
    config = ESNConfig(reservoir_size=args.reservoir_size, washout=args.washout)
    predictions, metrics, summary = evaluate_seeds(
        data, config, args.seeds, backend=args.backend,
        min_train_size=args.min_train_size, max_steps=args.max_steps,
    )
    print(summary.to_string(index=False))
    if args.output_dir:
        metadata = write_forecast_evaluation(
            args.output_dir, data, config, args.seeds, predictions, metrics, summary,
            backend=args.backend, min_train_size=args.min_train_size,
            max_steps=args.max_steps,
        )
        print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
