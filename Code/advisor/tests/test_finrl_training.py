import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from advisor.finrl_training import chronological_partitions, load_strategy_config


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


if __name__ == "__main__":
    unittest.main()
