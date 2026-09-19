import json
import tempfile
import unittest
from pathlib import Path

from advisor.config import ConfigValidationError, default_mvp_config_path, load_mvp_config
from advisor.data import CsvMarketDataProvider
from advisor.preparation import market_data_digest, write_prepared_dataset


FIXTURE = Path(__file__).parent / "fixtures" / "market_data.csv"


class ConfigAndPreparationTests(unittest.TestCase):
    def test_default_configuration_defines_the_five_ticker_mvp(self):
        config = load_mvp_config(default_mvp_config_path())

        self.assertEqual(config.tickers, ("AAPL", "MSFT", "JPM", "JNJ", "PG"))
        self.assertEqual(config.forecast_backend, "reservoirpy")
        self.assertEqual(config.default_risk_profile, "moderate")

    def test_duplicate_tickers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(
                default_mvp_config_path().read_text(encoding="utf-8").replace(
                    "[AAPL, MSFT, JPM, JNJ, PG]", "[AAPL, AAPL]"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigValidationError, "unique"):
                load_mvp_config(path)

    def test_prepared_artifact_has_stable_digest_and_metadata(self):
        market_data = CsvMarketDataProvider(FIXTURE).load(["AAPL", "MSFT"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "mvp.yaml"
            path.write_text(
                default_mvp_config_path().read_text(encoding="utf-8")
                .replace("[AAPL, MSFT, JPM, JNJ, PG]", "[AAPL, MSFT]")
                .replace("../data/cache", "cache")
                .replace("../data/processed/mvp_market_data.csv", "prepared.csv")
                .replace("../data/processed/mvp_market_data.metadata.json", "prepared.json"),
                encoding="utf-8",
            )
            config = load_mvp_config(path)
            output, metadata_path, metadata = write_prepared_dataset(market_data, config, "fixture")

            self.assertTrue(output.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(metadata["data_sha256"], market_data_digest(market_data))
            self.assertEqual(metadata["tickers"], ["AAPL", "MSFT"])
            self.assertEqual(json.loads(metadata_path.read_text(encoding="utf-8"))["rows"], 12)


if __name__ == "__main__":
    unittest.main()
