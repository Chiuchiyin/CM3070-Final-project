# Unified advisor vertical slice

This folder is the isolated implementation path for the project unification. Existing notebooks and `Code/stock-app/` are intentionally unchanged.

The current slice is dependency-light and works offline:

- canonical market-data validation;
- CSV fixture/cache provider;
- last-close forecast baseline;
- equal-weight long-only allocation;
- grounded deterministic explanation;
- an orchestration service and offline tests.
- deterministic NumPy ESN fallback and optional ReservoirPy ESN forecasting;
- leakage-free walk-forward evaluation against the last-close baseline;
- versioned ESN artifact save/load support.
- chronological portfolio backtesting with no future data exposed to policies;
- equal-weight rebalancing and buy-and-hold baselines;
- transaction costs, slippage, portfolio risk metrics, and turnover reporting.

Run from this directory (the package is intentionally kept directly under `advisor/`):

```powershell
python -m unittest discover -s tests -v
```

Prepare the versioned five-stock MVP dataset from Yahoo Finance (or its local
cache):

```powershell
python scripts/prepare_data.py
```

For an offline reproducible input, supply a canonical CSV and a matching
configuration with its ticker universe:

```powershell
python scripts/prepare_data.py --config configs/mvp.yaml --source-csv path/to/market_data.csv
```

The configuration at `configs/mvp.yaml` is the shared source for the MVP
universe, date interval, ESN settings, transaction-cost assumptions, and
prepared-data artifact paths.

Try the service interactively:

```powershell
python -c "from pathlib import Path; from advisor.data import CsvMarketDataProvider; from advisor.service import AdvisorService; r=AdvisorService(CsvMarketDataProvider(Path('tests/fixtures/market_data.csv'))).analyse(['AAPL','MSFT']); print(r.explanation); print(r.allocations)"
```

The next replacement should preserve the `AdvisorService` boundary: add portfolio backtesting and a saved FinRL strategy, then an optional Qwen explainer, and finally connect the result to Shiny.

Evaluate ESN forecasting on the existing historical data:

```powershell
python scripts/evaluate_forecaster.py ../data/2025-06-16_dow30.csv --tickers AAPL MSFT --max-steps 100
```

For short experiments, lower the ESN washout explicitly, for example
`--washout 0`.

Install the full modelling stack, including ReservoirPy and FinRL:

```powershell
pip install -r requirements-ml.txt
```

Use the ReservoirPy backend once installed:

```powershell
python scripts/evaluate_forecaster.py ../data/2025-06-16_dow30.csv --tickers AAPL MSFT --backend reservoirpy
```

Write the predictions and metrics to an artifact directory by adding `--output-dir artifacts/evaluation`.

Run both portfolio baselines and write returns, weights, and metrics:

```powershell
python scripts/backtest_portfolios.py ../data/2025-06-16_dow30.csv --tickers AAPL MSFT
```

The default simulation charges 10 basis points of transaction costs and 5 basis
points of slippage. Use `--transaction-cost-bps`, `--slippage-bps`, and
`--rebalance-every` to make these assumptions explicit for an experiment.

## Saved FinRL policies

`advisor.finrl_adapter.FinRLPolicyAdapter` implements the same allocation-policy
contract as the baseline strategies, so it can be evaluated with
`run_backtest` without coupling FinRL to the UI or backtesting engine. A saved
policy must receive exactly the ticker ordering and observation schema it used
during training. Supply that schema through an `ObservationBuilder`; the
included `RollingReturnObservationBuilder` is for newly trained compatible
models, not for silently loading a legacy notebook model.

When backtesting a builder that declares a warm-up requirement, the adapter
holds cash until the required number of dates is available. This makes that
warm-up visible in results instead of filling unavailable observations with
future data or fabricated values.

Legacy FinRL portfolio notebooks use one raw action per asset and apply softmax
inside their environment. The adapter preserves that convention. For a policy
loaded from Stable-Baselines3, use `FinRLPolicyAdapter.from_stable_baselines3`;
the package is intentionally optional and only imported when that method runs.
`requirements-ml.txt` installs FinRL, Stable-Baselines3, and ReservoirPy as the
full modelling stack; the base installation keeps the offline examples and
tests lightweight.

The short evaluation command is a pipeline smoke test, not evidence that the ESN outperforms the baseline. Final model claims require a documented walk-forward interval, multiple seeds, and an untouched test period.
