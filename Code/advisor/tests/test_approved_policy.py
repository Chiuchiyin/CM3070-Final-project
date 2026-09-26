import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from advisor.finrl_adapter import (
    RollingReturnAndWeightsObservationBuilder,
    load_approved_finrl_policy,
)


class FakeModel:
    def predict(self, observation, deterministic=True):
        return np.zeros(3), None


class ApprovedPolicyLoaderTests(unittest.TestCase):
    def _files(self, metadata):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        model = root / 'policy.zip'
        metadata_path = root / 'policy.metadata.json'
        model.write_bytes(b'placeholder')
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        self.addCleanup(directory.cleanup)
        return model, metadata_path

    def _metadata(self):
        return {
            'algorithm': 'A2C',
            'tickers': ['AAPL', 'MSFT'],
            'observation_schema': 'rolling_returns_and_weights_v1',
            'lookback': 2,
            'include_cash': True,
        }

    def test_loader_validates_metadata_and_never_trains(self):
        model_path, metadata_path = self._files(self._metadata())
        calls = []
        adapter, metadata = load_approved_finrl_policy(
            model_path, metadata_path,
            tickers=['MSFT', 'AAPL'],
            model_loader=lambda path, algorithm: (calls.append((path, algorithm)) or FakeModel()),
        )
        self.assertEqual(metadata['algorithm'], 'A2C')
        self.assertEqual(adapter.tickers, ['AAPL', 'MSFT'])
        self.assertEqual(adapter.observation_builder.lookback, 2)
        self.assertEqual(len(calls), 1)

    def test_loader_rejects_unapproved_universe(self):
        model_path, metadata_path = self._files(self._metadata())
        with self.assertRaisesRegex(ValueError, 'tickers'):
            load_approved_finrl_policy(
                model_path, metadata_path, tickers=['AAPL', 'JPM'],
                model_loader=lambda *_: FakeModel(),
            )

    def test_loader_rejects_schema_or_lookback_mismatch(self):
        metadata = self._metadata()
        metadata['observation_schema'] = 'legacy_not_approved'
        model_path, metadata_path = self._files(metadata)
        with self.assertRaisesRegex(ValueError, 'schema'):
            load_approved_finrl_policy(
                model_path, metadata_path, tickers=['AAPL', 'MSFT'],
                model_loader=lambda *_: FakeModel(),
            )

        metadata = self._metadata()
        model_path, metadata_path = self._files(metadata)
        with self.assertRaisesRegex(ValueError, 'lookback'):
            load_approved_finrl_policy(
                model_path, metadata_path, tickers=['AAPL', 'MSFT'],
                observation_builder=RollingReturnAndWeightsObservationBuilder(lookback=3),
                model_loader=lambda *_: FakeModel(),
            )


if __name__ == '__main__':
    unittest.main()
