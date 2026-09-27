import unittest
from pathlib import Path

import pandas as pd

from advisor.backtesting import (
    CASH,
    BacktestConfig,
    BuyAndHoldPolicy,
    EqualWeightPolicy,
    ForecastRankedPolicy,
    run_baseline_comparison,
    run_backtest,
)
from advisor.data import CsvMarketDataProvider
from advisor.forecasting import MovingAverageForecaster


FIXTURE = Path(__file__).parent / "fixtures" / "market_data.csv"


class BacktestingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market_data = CsvMarketDataProvider(FIXTURE).load(["AAPL", "MSFT"])

    def test_equal_weight_backtest_is_chronological_and_fully_invested(self):
        result = run_backtest(self.market_data, EqualWeightPolicy())

        self.assertEqual(len(result.returns), 5)
        self.assertTrue(result.returns["date"].is_monotonic_increasing)
        totals = result.weights.groupby("date")["weight"].sum()
        self.assertTrue((totals - 1.0).abs().lt(1e-9).all())
        first = result.weights[result.weights["date"] == result.weights["date"].min()]
        self.assertAlmostEqual(first.loc[first["ticker"] == "AAPL", "weight"].iloc[0], 0.5)
        self.assertAlmostEqual(first.loc[first["ticker"] == CASH, "weight"].iloc[0], 0.0)
        self.assertEqual(result.returns.iloc[0]["period_start"], pd.Timestamp("2021-11-26"))
        self.assertEqual(result.returns.iloc[0]["date"], pd.Timestamp("2021-11-29"))

    def test_buy_and_hold_trades_only_at_inception(self):
        result = run_backtest(self.market_data, BuyAndHoldPolicy())

        self.assertAlmostEqual(result.returns.iloc[0]["turnover"], 1.0)
        self.assertTrue((result.returns.iloc[1:]["turnover"] == 0.0).all())
        later = result.weights[result.weights["date"] == result.weights["date"].unique()[1]]
        asset_weights = later[later["ticker"] != CASH].set_index("ticker")["weight"]
        self.assertNotAlmostEqual(asset_weights["AAPL"], asset_weights["MSFT"])

    def test_costs_and_slippage_reduce_final_value(self):
        free = run_backtest(self.market_data, EqualWeightPolicy())
        costly = run_backtest(
            self.market_data,
            EqualWeightPolicy(),
            BacktestConfig(transaction_cost_bps=10, slippage_bps=5),
        )

        self.assertLess(costly.metrics["final_value"], free.metrics["final_value"])
        self.assertGreater(costly.returns["trading_cost"].sum(), 0.0)

    def test_policy_never_receives_future_rows(self):
        class RecordingPolicy(EqualWeightPolicy):
            def __init__(self):
                super().__init__()
                self.seen = []

            def target_weights(self, history, current_weights):
                self.seen.append(history["date"].max())
                return super().target_weights(history, current_weights)

        policy = RecordingPolicy()
        result = run_backtest(self.market_data, policy)
        self.assertEqual(policy.seen, result.returns["period_start"].tolist())

    def test_short_or_unbalanced_policy_is_rejected(self):
        class InvalidPolicy:
            name = "invalid"

            def target_weights(self, history, current_weights):
                return pd.Series({"AAPL": 1.1, "MSFT": -0.1})

        with self.assertRaisesRegex(ValueError, "long-only"):
            run_backtest(self.market_data, InvalidPolicy())

    def test_metrics_include_required_risk_and_turnover_fields(self):
        metrics = run_backtest(self.market_data, BuyAndHoldPolicy()).metrics
        expected = {
            "cumulative_return",
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "maximum_drawdown",
            "total_turnover",
            "average_period_turnover",
            "final_value",
        }
        self.assertEqual(set(metrics), expected)
        self.assertLessEqual(metrics["maximum_drawdown"], 0.0)

    def test_forecast_ranked_policy_holds_cash_during_warmup_then_selects_top_asset(self):
        policy = ForecastRankedPolicy(
            MovingAverageForecaster(window=2), top_k=1
        )
        result = run_backtest(self.market_data, policy)
        weights = result.weights.pivot(index="date", columns="ticker", values="weight")

        self.assertAlmostEqual(weights.iloc[0][CASH], 1.0)
        self.assertAlmostEqual(weights.iloc[1][CASH], 1.0)
        self.assertAlmostEqual(weights.iloc[2]["AAPL"], 1.0)
        self.assertAlmostEqual(weights.iloc[2]["MSFT"], 0.0)
        self.assertAlmostEqual(weights.iloc[2][CASH], 0.0)

    def test_forecast_ranked_policy_passes_only_current_history_to_forecaster(self):
        class RecordingForecaster:
            name = "recording"
            minimum_observations = 1

            def __init__(self):
                self.seen = []

            def predict(self, history):
                self.seen.append(history["date"].max())
                tickers = sorted(history["ticker"].unique())
                return pd.DataFrame(
                    {
                        "ticker": tickers,
                        "predicted_return": range(len(tickers)),
                    }
                )

        forecaster = RecordingForecaster()
        result = run_backtest(
            self.market_data, ForecastRankedPolicy(forecaster, top_k=1)
        )
        self.assertEqual(forecaster.seen, result.returns["period_start"].tolist())

    def test_baseline_comparison_includes_index_only_when_supplied(self):
        without_index = run_baseline_comparison(self.market_data)
        self.assertEqual(set(without_index), {"equal_weight", "buy_and_hold", "forecast_ranked_moving_average_return"})
        index = self.market_data[self.market_data["ticker"] == "AAPL"].copy()
        index["ticker"] = "INDEX"
        with_index = run_baseline_comparison(self.market_data, index_data=index)
        self.assertIn("market_index", with_index)
        self.assertEqual(
            with_index["market_index"].returns["date"].tolist(),
            without_index["equal_weight"].returns["date"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
