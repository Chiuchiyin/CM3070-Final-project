"""Run baseline portfolio backtests and write reproducible CSV artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor.backtesting import (  # noqa: E402
    BacktestConfig,
    BuyAndHoldPolicy,
    EqualWeightPolicy,
    ForecastRankedPolicy,
    run_baseline_comparison,
)
from advisor.data import CsvMarketDataProvider  # noqa: E402
from advisor.forecasting import MovingAverageForecaster  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Canonical or FinRL-compatible market-data CSV")
    parser.add_argument("--tickers", nargs="+", help="Optional ticker subset")
    parser.add_argument("--start", help="Inclusive start date")
    parser.add_argument("--end", help="Inclusive end date")
    parser.add_argument("--initial-value", type=float, default=10_000.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--rebalance-every", type=int, default=1)
    parser.add_argument("--forecast-window", type=int, default=20)
    parser.add_argument("--forecast-top-k", type=int, default=3)
    parser.add_argument(
        "--forecast-min-return",
        type=float,
        help="Optional minimum predicted return; hold cash if no asset qualifies",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/backtesting"))
    parser.add_argument(
        "--index-csv", type=Path,
        help="Optional canonical one-ticker index CSV evaluated over overlapping dates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    market_data = CsvMarketDataProvider(args.csv).load(args.tickers, args.start, args.end)
    config = BacktestConfig(
        initial_value=args.initial_value,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )
    index_data = CsvMarketDataProvider(args.index_csv).load() if args.index_csv else None
    results = list(run_baseline_comparison(
        market_data, config, forecast_window=args.forecast_window,
        forecast_top_k=args.forecast_top_k,
        rebalance_every=args.rebalance_every,
        forecast_minimum_return=args.forecast_min_return,
        index_data=index_data,
    ).values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([result.returns for result in results], ignore_index=True).to_csv(
        args.output_dir / "backtest_returns.csv", index=False
    )
    pd.concat([result.weights for result in results], ignore_index=True).to_csv(
        args.output_dir / "backtest_weights.csv", index=False
    )
    pd.DataFrame(
        [{"strategy_name": result.strategy_name, **result.metrics} for result in results]
    ).to_csv(args.output_dir / "backtest_metrics.csv", index=False)

    if index_data is None:
        (args.output_dir / "backtest_metadata.txt").write_text(
            "market_index: unavailable (no --index-csv supplied)\n", encoding="utf-8"
        )

    for result in results:
        metrics = result.metrics
        print(
            f"{result.strategy_name}: cumulative return {metrics['cumulative_return']:.2%}, "
            f"Sharpe {metrics['sharpe_ratio']:.3f}, max drawdown "
            f"{metrics['maximum_drawdown']:.2%}"
        )
    print(f"Wrote backtest artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
