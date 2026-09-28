from pathlib import Path
import unittest
from advisor.data import CsvMarketDataProvider
from advisor.explanation import (QwenExplainer, build_explanation_context,
                                 numeric_claims_are_grounded, template_explanation)
from advisor.service import AdvisorService

FIXTURE = Path(__file__).parent / 'fixtures' / 'market_data.csv'

class ExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = AdvisorService(CsvMarketDataProvider(FIXTURE)).analyse(['AAPL', 'MSFT'], risk_profile='moderate')

    def test_context_contains_only_service_facts(self):
        context = build_explanation_context(
            self.result.forecasts, self.result.allocations,
            risk_profile=self.result.risk_profile,
            warnings=self.result.warnings,
            backtest_metrics=self.result.backtest_metrics,
            model_version=self.result.model_version,
            dataset_version=self.result.dataset_version,
        )
        self.assertEqual(context.risk_profile, 'moderate')
        self.assertEqual(context.as_of_date, '2021-12-03')
        self.assertEqual(len(context.forecasts), 2)
        self.assertIn('FACTS_JSON=', context.prompt())

    def test_numeric_claim_validation_accepts_context_values_and_rejects_new_values(self):
        context = build_explanation_context(self.result.forecasts, self.result.allocations)
        self.assertTrue(numeric_claims_are_grounded('The cash allocation is 10.0%.', context))
        self.assertFalse(numeric_claims_are_grounded('The allocation gained 87.3%.', context))
        self.assertTrue(numeric_claims_are_grounded('As of December 3, 2021, cash is 0.0.', context))
        self.assertFalse(numeric_claims_are_grounded('As of December 3, 2025, cash is 0.0.', context))
        self.assertFalse(numeric_claims_are_grounded('The portfolio is 100% cash.', context))

    def test_qwen_adapter_falls_back_when_claims_are_unsupported(self):
        explainer = QwenExplainer(generator=lambda prompt: 'The portfolio gained 87.3%.')
        generated = explainer(
            self.result.forecasts, self.result.allocations,
            risk_profile=self.result.risk_profile,
            warnings=self.result.warnings,
            backtest_metrics=self.result.backtest_metrics,
        )
        self.assertEqual(generated, template_explanation(
            self.result.forecasts, self.result.allocations,
            risk_profile=self.result.risk_profile,
            warnings=self.result.warnings,
            backtest_metrics=self.result.backtest_metrics,
        ))

    def test_qwen_adapter_accepts_grounded_text(self):
        explainer = QwenExplainer(generator=lambda prompt: 'As of 2021-12-03, cash is 10.0%.')
        generated = explainer(
            self.result.forecasts, self.result.allocations,
            risk_profile=self.result.risk_profile,
            warnings=self.result.warnings,
            backtest_metrics=self.result.backtest_metrics,
        )
        self.assertIn('cash is 10.0%', generated)

    def test_service_reports_qwen_source_and_grounded_fallback(self):
        provider = CsvMarketDataProvider(FIXTURE)
        generated = AdvisorService(
            provider,
            explainer=QwenExplainer(generator=lambda prompt: 'As of 2021-12-03, cash is 10.0%.'),
        ).analyse(['AAPL', 'MSFT'], risk_profile='moderate')
        self.assertEqual(generated.explanation_source, 'qwen')
        self.assertIn('cash is 10.0%', generated.explanation)

        fallback = AdvisorService(
            provider,
            explainer=QwenExplainer(generator=lambda prompt: 'The portfolio gained 87.3%.'),
        ).analyse(['AAPL', 'MSFT'], risk_profile='moderate')
        self.assertTrue(fallback.explanation_source.startswith('template:'))
        self.assertIn('educational historical-data demonstration', fallback.explanation)

if __name__ == '__main__':
    unittest.main()
