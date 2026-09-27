"""Train the configured FinRL A2C policy from a prepared CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from advisor.config import default_mvp_config_path, load_mvp_config
from advisor.data import CsvMarketDataProvider
from advisor.finrl_training import load_strategy_config, override_strategy_config, train_a2c

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=default_mvp_config_path())
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--train-end', help='Override the training split end date (YYYY-MM-DD).')
    parser.add_argument('--validation-end', help='Override the validation split end date (YYYY-MM-DD).')
    parser.add_argument('--test-end', help='Override the test split end date (YYYY-MM-DD).')
    parser.add_argument('--lookback', type=int, help='Override the rolling observation lookback.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--metadata-output', type=Path)
    args=parser.parse_args()
    config=load_mvp_config(args.config)
    strategy=load_strategy_config(args.config)
    strategy = override_strategy_config(
        strategy,
        train_end=args.train_end,
        validation_end=args.validation_end,
        test_end=args.test_end,
        lookback=args.lookback,
        total_timesteps=args.timesteps,
    )
    # A legacy split may begin before the MVP data.start. In override mode,
    # use all available history through the selected test boundary.
    override_dates = any((args.train_end, args.validation_end, args.test_end))
    data_start = None if override_dates else config.start
    data_end = strategy.test_end if override_dates else config.end
    data=CsvMarketDataProvider(args.data).load(config.tickers, data_start, data_end)
    artifact, metadata, _ = train_a2c(
        data, strategy, seed=args.seed, tickers=list(config.tickers),
        output_path=args.output, metadata_path=args.metadata_output,
        transaction_cost_bps=config.transaction_cost_bps,
        slippage_bps=config.slippage_bps,
    )
    print(f'Policy: {artifact}')
    print(f'Metadata: {metadata}')
if __name__ == '__main__':
    main()
