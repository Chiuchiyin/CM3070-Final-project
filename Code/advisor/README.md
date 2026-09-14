# Unified advisor vertical slice

This folder is the isolated implementation path for the project unification. Existing notebooks and `Code/stock-app/` are intentionally unchanged.

The current slice is dependency-light and works offline:

- canonical market-data validation;
- CSV fixture/cache provider;
- last-close forecast baseline;
- equal-weight long-only allocation;
- grounded deterministic explanation;
- an orchestration service and offline tests.
- deterministic NumPy Echo State Network forecasting;
- leakage-free walk-forward evaluation against the last-close baseline;
- versioned ESN artifact save/load support.

Run from this directory (the package is intentionally kept directly under `advisor/`):

```powershell
python -m unittest discover -s tests -v
```

Try the service interactively:

```powershell
python -c "from pathlib import Path; from advisor.data import CsvMarketDataProvider; from advisor.service import AdvisorService; r=AdvisorService(CsvMarketDataProvider(Path('tests/fixtures/market_data.csv'))).analyse(['AAPL','MSFT']); print(r.explanation); print(r.allocations)"
```

The next replacement should preserve the `AdvisorService` boundary: add portfolio backtesting and a saved FinRL strategy, then an optional Qwen explainer, and finally connect the result to Shiny.

Evaluate ESN forecasting on the existing historical data:

```powershell
python scripts/evaluate_forecaster.py ../data/2025-06-16_dow30.csv --tickers AAPL MSFT --max-steps 100
```

Write the predictions and metrics to an artifact directory by adding `--output-dir artifacts/evaluation`.

The short evaluation command is a pipeline smoke test, not evidence that the ESN outperforms the baseline. Final model claims require a documented walk-forward interval, multiple seeds, and an untouched test period.
