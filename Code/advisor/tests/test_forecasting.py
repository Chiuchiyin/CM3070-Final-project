import tempfile
import unittest
from pathlib import Path
import json

import numpy as np
import pandas as pd

from advisor.evaluation import compare_esn_with_baseline, evaluate_seeds, write_forecast_evaluation
from advisor.forecasting import (
    EchoStateNetwork,
    ESNConfig,
    MovingAverageForecaster,
    ReservoirPyESN,
    ReservoirPyForecaster,
)


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

    def test_reservoirpy_forecaster_exposes_the_same_service_identity(self):
        self.assertEqual(ReservoirPyForecaster.name, "reservoirpy_esn")
        self.assertEqual(ReservoirPyESN.name, "reservoirpy_esn")

    def test_moving_average_forecaster_uses_only_trailing_returns(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        frame = pd.DataFrame(
            {"date": dates, "ticker": "TEST", "close": [100.0, 110.0, 99.0, 108.9]}
        )
        forecast = MovingAverageForecaster(window=2).predict(frame).iloc[0]

        self.assertAlmostEqual(forecast["predicted_return"], 0.0)
        self.assertAlmostEqual(forecast["predicted_close"], 108.9)
        self.assertEqual(forecast["as_of_date"], dates[-1])

    def test_multi_seed_evaluation_reports_dispersion_on_matching_targets(self):
        dates = pd.date_range("2024-01-01", periods=len(self.series), freq="B")
        frame = pd.DataFrame({"date": dates, "ticker": "TEST", "close": self.series})
        predictions, metrics, summary = evaluate_seeds(
            frame, self.config, [7, 8], min_train_size=25, max_steps=5
        )
        self.assertEqual(set(predictions["seed"]), {7, 8})
        self.assertEqual(len(metrics), 8)
        self.assertIn("mae_mean", summary.columns)
        self.assertIn("mae_std", summary.columns)

    def test_forecast_evaluation_writer_records_data_and_seed_metadata(self):
        dates = pd.date_range("2024-01-01", periods=len(self.series), freq="B")
        frame = pd.DataFrame({"date": dates, "ticker": "TEST", "close": self.series})
        predictions, metrics, summary = evaluate_seeds(
            frame, self.config, [7], min_train_size=25, max_steps=5
        )
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = write_forecast_evaluation(
                directory, frame, self.config, [7], predictions, metrics, summary,
                backend="numpy", min_train_size=25, max_steps=5,
            )
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["seeds"], [7])
            self.assertEqual(metadata["tickers"], ["TEST"])
            self.assertTrue((Path(directory) / "forecast_seed_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
