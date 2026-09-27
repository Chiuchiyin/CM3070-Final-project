"""Evaluate an approved FinRL policy against baselines on one held-out interval."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor.backtesting import BacktestConfig, run_backtest, run_baseline_comparison
from advisor.data import CsvMarketDataProvider
from advisor.finrl_adapter import load_approved_finrl_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    market_data = CsvMarketDataProvider(args.data).load(args.tickers, args.start, args.end)
    adapter, metadata = load_approved_finrl_policy(
        args.model, args.metadata, tickers=args.tickers
    )
    config = BacktestConfig(transaction_cost_bps=10.0, slippage_bps=5.0)
    results = run_baseline_comparison(market_data, config)
    results[adapter.name] = run_backtest(market_data, adapter, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(
        [{"strategy_name": result.strategy_name, **result.metrics} for result in results.values()]
    )
    metrics.to_csv(args.output_dir / "finrl_legacy_comparison_metrics.csv", index=False)
    pd.concat([result.returns for result in results.values()], ignore_index=True).to_csv(
        args.output_dir / "finrl_legacy_comparison_returns.csv", index=False
    )
    pd.concat([result.weights for result in results.values()], ignore_index=True).to_csv(
        args.output_dir / "finrl_legacy_comparison_weights.csv", index=False
    )
    (args.output_dir / "finrl_legacy_comparison_metadata.txt").write_text(
        "policy_metadata:\n" + repr(metadata) + "\n"
        f"interval: {args.start} to {args.end}\n"
        "costs: transaction_cost_bps=10.0, slippage_bps=5.0\n",
        encoding="utf-8",
    )
    for result in results.values():
        print(
            f"{result.strategy_name}: cumulative return "
            f"{result.metrics['cumulative_return']:.2%}, "
            f"Sharpe {result.metrics['sharpe_ratio']:.3f}, "
            f"max drawdown {result.metrics['maximum_drawdown']:.2%}"
        )
    print(f"Wrote comparison artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
