"""Prepare a validated, versioned market-data artifact for the MVP universe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advisor.config import default_mvp_config_path, load_mvp_config  # noqa: E402
from advisor.data import CsvMarketDataProvider, YahooMarketDataProvider  # noqa: E402
from advisor.preparation import write_prepared_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_mvp_config_path())
    parser.add_argument("--source-csv", type=Path, help="Use a frozen CSV instead of Yahoo Finance")
    parser.add_argument("--output", type=Path, help="Override data.prepared_data_path")
    parser.add_argument("--metadata-output", type=Path, help="Override data.metadata_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_mvp_config(args.config)
    if args.source_csv:
        source = "csv"
        provider = CsvMarketDataProvider(args.source_csv)
    else:
        source = "yahoo_finance"
        provider = YahooMarketDataProvider(config.cache_dir)
    market_data = provider.load(config.tickers, config.start, config.end)
    output, metadata_output, metadata = write_prepared_dataset(
        market_data, config, source, args.output, args.metadata_output
    )
    print(
        f"Prepared {metadata['rows']} rows for {', '.join(metadata['tickers'])} "
        f"from {metadata['date_start']} to {metadata['date_end']}"
    )
    print(f"Data: {output}")
    print(f"Metadata: {metadata_output}")


if __name__ == "__main__":
    main()
