import unittest
from pathlib import Path

import numpy as np

from advisor.backtesting import CASH, run_backtest
from advisor.data import CsvMarketDataProvider
from advisor.finrl_adapter import FinRLPolicyAdapter, RollingReturnObservationBuilder


FIXTURE = Path(__file__).parent / "fixtures" / "market_data.csv"


class RecordingModel:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=float)
        self.observations = []
        self.deterministic = []

    def predict(self, observation, deterministic=True):
        self.observations.append(np.asarray(observation))
        self.deterministic.append(deterministic)
        return self.action, None


class FinRLAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = CsvMarketDataProvider(FIXTURE).load(["AAPL", "MSFT"])
        cls.current_weights = np.array([0.5, 0.5])

    def test_adapter_preserves_sorted_universe_and_softmax_action_convention(self):
        model = RecordingModel([0.0, np.log(3.0)])
        adapter = FinRLPolicyAdapter(
            model,
            RollingReturnObservationBuilder(lookback=2),
            ["MSFT", "AAPL"],
        )
        weights = adapter.target_weights(self.history, self.current_weights)

        self.assertEqual(weights.index.tolist(), ["AAPL", "MSFT"])
        self.assertAlmostEqual(weights["AAPL"], 0.25)
        self.assertAlmostEqual(weights["MSFT"], 0.75)
        self.assertEqual(model.observations[0].shape, (2, 2))
        self.assertEqual(model.deterministic, [True])

    def test_adapter_can_allocate_explicit_cash(self):
        model = RecordingModel([0.0, 0.0, np.log(2.0)])
        adapter = FinRLPolicyAdapter(
            model,
            RollingReturnObservationBuilder(lookback=1),
            ["AAPL", "MSFT"],
            include_cash_action=True,
        )
        weights = adapter.target_weights(self.history, self.current_weights)

        self.assertEqual(weights.index.tolist(), ["AAPL", "MSFT", CASH])
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertAlmostEqual(weights[CASH], 0.5)

    def test_adapter_rejects_wrong_action_shape(self):
        adapter = FinRLPolicyAdapter(
            RecordingModel([0.0]), RollingReturnObservationBuilder(lookback=1), ["AAPL", "MSFT"]
        )
        with self.assertRaisesRegex(ValueError, "expected 2"):
            adapter.target_weights(self.history, self.current_weights)

    def test_observation_builder_requires_sufficient_history(self):
        builder = RollingReturnObservationBuilder(lookback=10)
        with self.assertRaisesRegex(ValueError, "requires 11 dates"):
            builder.build(self.history, ["AAPL", "MSFT"], self.current_weights)

    def test_adapter_waits_for_observation_warmup_when_backtested(self):
        model = RecordingModel([0.0, 0.0])
        adapter = FinRLPolicyAdapter(
            model, RollingReturnObservationBuilder(lookback=2), ["AAPL", "MSFT"]
        )
        result = run_backtest(self.history, adapter)

        self.assertTrue((result.returns.iloc[:2]["turnover"] == 0.0).all())
        self.assertEqual(len(model.observations), len(result.returns) - 2)
        self.assertTrue((result.weights.groupby("date")["weight"].sum() - 1.0).abs().lt(1e-9).all())


if __name__ == "__main__":
    unittest.main()
