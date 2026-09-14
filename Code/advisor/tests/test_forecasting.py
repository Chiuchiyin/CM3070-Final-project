import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from advisor.evaluation import compare_esn_with_baseline
from advisor.forecasting import EchoStateNetwork, ESNConfig


class ESNTests(unittest.TestCase):
    def setUp(self):
        self.config = ESNConfig(reservoir_size=12, connectivity=0.5, washout=3, seed=7)
        self.series = np.linspace(10.0, 30.0, 40) + np.sin(np.arange(40) / 3)

    def test_training_is_deterministic_and_scaling_uses_training_data(self):
        first = EchoStateNetwork(self.config).fit(self.series[:30])
        second = EchoStateNetwork(self.config).fit(self.series[:30])
        self.assertAlmostEqual(first.predict_next(), second.predict_next())
        self.assertEqual(first.scale_min, float(self.series[:30].min()))
        self.assertEqual(first.scale_max, float(self.series[:30].max()))

    def test_saved_model_round_trip(self):
        model = EchoStateNetwork(self.config).fit(self.series)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            loaded = EchoStateNetwork.load(path)
            self.assertAlmostEqual(model.predict_next(), loaded.predict_next())
            self.assertEqual(loaded.metadata()["training_observations"], len(self.series))

    def test_walk_forward_comparison_has_matching_targets(self):
        dates = pd.date_range("2024-01-01", periods=len(self.series), freq="B")
        frame = pd.DataFrame({"date": dates, "ticker": "TEST", "close": self.series})
        predictions, metrics = compare_esn_with_baseline(
            frame, self.config, min_train_size=25, max_steps=5
        )
        self.assertEqual(len(predictions), 10)
        self.assertEqual(set(predictions["model_name"]), {"last_close", "numpy_esn"})
        self.assertEqual(len(metrics), 4)
        baseline = predictions[predictions["model_name"] == "last_close"]
        np.testing.assert_allclose(baseline["predicted"], baseline["previous"])


if __name__ == "__main__":
    unittest.main()
