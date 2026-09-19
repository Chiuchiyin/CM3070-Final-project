import unittest
from pathlib import Path

import pandas as pd

from advisor.backtesting import (
    CASH,
    BacktestConfig,
    BuyAndHoldPolicy,
    EqualWeightPolicy,
    run_backtest,
)
from advisor.data import CsvMarketDataProvider


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


if __name__ == "__main__":
    unittest.main()
