import unittest
from pathlib import Path

import pandas as pd

from advisor.data import (
    CsvMarketDataProvider,
    DataValidationError,
    _normalize_yahoo_download,
    validate_market_data,
)
from advisor.forecasting import ESNConfig, ESNForecaster
from advisor.service import AdvisorService


FIXTURE = Path(__file__).parent / "fixtures" / "market_data.csv"


class VerticalSliceTests(unittest.TestCase):
    def test_fixture_loads_and_filters(self):
        frame = CsvMarketDataProvider(FIXTURE).load(["AAPL", "MSFT"])
        self.assertEqual(set(frame["ticker"]), {"AAPL", "MSFT"})
        self.assertTrue(frame["date"].is_monotonic_increasing)

    def test_invalid_ohlc_is_rejected(self):
        frame = pd.DataFrame({"date": ["2025-01-01"], "ticker": ["ABC"], "open": [2],
                              "high": [1], "low": [1], "close": [1], "volume": [1]})
        with self.assertRaises(DataValidationError):
            validate_market_data(frame)

    def test_yahoo_multiindex_layouts_are_normalized(self):
        dates = pd.date_range("2025-01-01", periods=2)
        fields = ["Open", "High", "Low", "Close", "Volume"]
        values = [[100, 101, 99, 100.5, 1000], [101, 102, 100, 101.5, 1100]]
        rows = {}
        for ticker, offset in (("AAPL", 0), ("MSFT", 10)):
            for field_index, field in enumerate(fields):
                rows[(ticker, field)] = [row[field_index] + offset for row in values]
        ticker_first = pd.DataFrame(rows, index=dates)
        ticker_first.columns = pd.MultiIndex.from_tuples(ticker_first.columns)
        normalized = validate_market_data(_normalize_yahoo_download(ticker_first, ["AAPL", "MSFT"]))
        self.assertEqual(set(normalized["ticker"]), {"AAPL", "MSFT"})

        field_first = ticker_first.copy()
        field_first.columns = pd.MultiIndex.from_tuples([(field, ticker) for ticker, field in ticker_first.columns])
        normalized = validate_market_data(_normalize_yahoo_download(field_first, ["AAPL", "MSFT"]))
        self.assertEqual(len(normalized), 4)

    def test_service_returns_grounded_equal_weights(self):
        result = AdvisorService(CsvMarketDataProvider(FIXTURE)).analyse(["AAPL", "MSFT"])
        self.assertAlmostEqual(result.allocations["weight"].sum(), 1.0)
        self.assertIn("last-close baseline", result.explanation)
        self.assertEqual(set(result.forecasts["ticker"]), {"AAPL", "MSFT"})

    def test_service_accepts_esn_without_changing_its_contract(self):
        config = ESNConfig(reservoir_size=8, connectivity=0.5, washout=0)
        result = AdvisorService(
            CsvMarketDataProvider(FIXTURE), forecaster=ESNForecaster(config)
        ).analyse(["AAPL", "MSFT"])
        self.assertEqual(set(result.forecasts["model_name"]), {"numpy_esn"})
        self.assertIn("numpy_esn", result.explanation)
        self.assertAlmostEqual(result.allocations["weight"].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
