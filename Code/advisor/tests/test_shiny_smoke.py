from pathlib import Path
import py_compile
import unittest

APP = Path(__file__).parents[2] / 'stock-app' / 'app-express.py'

class ShinySmokeTests(unittest.TestCase):
    def test_unified_app_compiles_and_contains_required_views(self):
        py_compile.compile(str(APP), doraise=True)
        source = APP.read_text(encoding='utf-8')
        for marker in ('AdvisorService', 'analysis_state', 'Summary', 'Portfolio', 'Forecast', 'Evaluation', 'Analysis unavailable', 'disclosures', 'Model version'):
            self.assertIn(marker, source)

    def test_app_uses_the_advisor_service_boundary(self):
        source = APP.read_text(encoding='utf-8')
        self.assertIn('service.analyse', source)
        self.assertIn('qwen_service', source)
        self.assertIn('SmolagentsQwenGenerator', source)
        self.assertIn('explanation_status', source)
        self.assertNotIn('yf.Ticker', source)

if __name__ == '__main__':
    unittest.main()
