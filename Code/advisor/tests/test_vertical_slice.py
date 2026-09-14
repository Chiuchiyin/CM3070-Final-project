import unittest
from pathlib import Path

import pandas as pd

from advisor.data import CsvMarketDataProvider, DataValidationError, validate_market_data
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
