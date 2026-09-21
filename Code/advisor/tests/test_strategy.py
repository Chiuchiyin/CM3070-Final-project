import unittest

import pandas as pd

from advisor.strategy import ForecastRankedStrategy


class ForecastRankedStrategyTests(unittest.TestCase):
    def test_selects_top_forecasts_with_deterministic_tie_breaking(self):
        forecasts = pd.DataFrame(
            {
                "ticker": ["MSFT", "JPM", "AAPL"],
                "predicted_return": [0.02, 0.01, 0.02],
            }
        )
        allocation = ForecastRankedStrategy(top_k=2).allocate(forecasts)
        weights = allocation.set_index("ticker")["weight"]

        self.assertAlmostEqual(weights["AAPL"], 0.5)
        self.assertAlmostEqual(weights["MSFT"], 0.5)
        self.assertAlmostEqual(weights["JPM"], 0.0)
        self.assertAlmostEqual(allocation["cash_weight"].iloc[0], 0.0)

    def test_holds_cash_when_no_forecast_clears_threshold(self):
        forecasts = pd.DataFrame(
            {"ticker": ["AAPL", "MSFT"], "predicted_return": [-0.01, -0.02]}
        )
        allocation = ForecastRankedStrategy(
            top_k=1, minimum_predicted_return=0.0
        ).allocate(forecasts)

        self.assertAlmostEqual(allocation["weight"].sum(), 0.0)
        self.assertAlmostEqual(allocation["cash_weight"].iloc[0], 1.0)

    def test_rejects_non_finite_forecast_returns(self):
        forecasts = pd.DataFrame(
            {"ticker": ["AAPL", "MSFT"], "predicted_return": [0.01, float("inf")]}
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            ForecastRankedStrategy(top_k=1).allocate(forecasts)


if __name__ == "__main__":
    unittest.main()
