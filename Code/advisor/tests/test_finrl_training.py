import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from advisor.finrl_training import (
    PortfolioAllocationEnv,
    chronological_partitions,
    dataset_sha256,
    load_strategy_config,
    train_a2c,
)


CONFIG = Path(__file__).parents[1] / "configs" / "mvp.yaml"


class FinRLTrainingContractTests(unittest.TestCase):
    def test_mvp_strategy_is_finrl_a2c_with_cash(self):
        strategy = load_strategy_config(CONFIG)
        self.assertEqual(strategy.algorithm, "a2c")
        self.assertTrue(strategy.include_cash)
        self.assertEqual(strategy.reward, "net_log_return")

    def test_partitions_are_chronological_and_non_overlapping(self):
        dates = pd.date_range("2025-01-01", periods=6, freq="D")
        frame = pd.DataFrame({
            "date": dates, "ticker": "TEST", "open": 100.0, "high": 101.0,
            "low": 99.0, "close": np.arange(100.0, 106.0), "volume": 1000,
        })
        partitions = chronological_partitions(frame, "2025-01-02", "2025-01-04", "2025-01-06")
        self.assertEqual([len(partitions[name]) for name in ("train", "validation", "test")], [2, 2, 2])
        self.assertLess(partitions["train"]["date"].max(), partitions["validation"]["date"].min())
        self.assertLess(partitions["validation"]["date"].max(), partitions["test"]["date"].min())

    def test_dataset_digest_is_stable(self):
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        frame = pd.DataFrame({
            "date": dates, "ticker": "TEST", "open": 100.0, "high": 101.0,
            "low": 99.0, "close": np.arange(100.0, 103.0), "volume": 1000,
        })
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        self.assertEqual(dataset_sha256(frame), dataset_sha256(shuffled))

    def test_training_rejects_data_before_configured_test_boundary(self):
        dates = pd.date_range("2025-01-01", periods=6, freq="D")
        frame = pd.DataFrame({
            "date": dates, "ticker": "TEST", "open": 100.0, "high": 101.0,
            "low": 99.0, "close": np.arange(100.0, 106.0), "volume": 1000,
        })
        strategy = load_strategy_config(CONFIG)
        with self.assertRaisesRegex(ValueError, "before configured test_end"):
            train_a2c(frame, strategy, tickers=["TEST"])

    @unittest.skipUnless(__import__("importlib").util.find_spec("gymnasium"), "gymnasium is optional")
    def test_portfolio_environment_exposes_cash_action_and_finite_step(self):
        dates = pd.date_range("2025-01-01", periods=12, freq="D")
        frame = pd.DataFrame({
            "date": dates, "ticker": "TEST", "open": 100.0, "high": 101.0,
            "low": 99.0, "close": np.arange(100.0, 112.0), "volume": 1000,
        })
        env = PortfolioAllocationEnv(frame, ["TEST"], lookback=3)
        observation, _ = env.reset(seed=7)
        self.assertEqual(observation.shape, (3 + 1 + 1,))
        next_observation, reward, terminated, truncated, info = env.step([0.0, 0.0])
        self.assertEqual(next_observation.shape, observation.shape)
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(truncated)
        self.assertIn("turnover", info)
        env.close()


if __name__ == "__main__":
    unittest.main()
