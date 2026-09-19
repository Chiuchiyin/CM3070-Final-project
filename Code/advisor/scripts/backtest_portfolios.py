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
    run_backtest,
)
from advisor.data import CsvMarketDataProvider  # noqa: E402


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
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/backtesting"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    market_data = CsvMarketDataProvider(args.csv).load(args.tickers, args.start, args.end)
    config = BacktestConfig(
        initial_value=args.initial_value,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
    )
    policies = [EqualWeightPolicy(args.rebalance_every), BuyAndHoldPolicy()]
    results = [run_backtest(market_data, policy, config) for policy in policies]

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
