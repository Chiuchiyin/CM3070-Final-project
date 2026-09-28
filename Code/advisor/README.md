# Unified advisor vertical slice

This folder is the implementation path for the project unification. The
notebooks remain research and provenance material; the production workflow is
the shared `AdvisorService` consumed by the Python Shiny app in
`Code/stock-app/app-express.py`.

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
- risk-profile cash floors, freshness warnings, allocation changes, and
  baseline metrics in the service result;
- Summary, Portfolio, Forecast, and Evaluation views in the unified Shiny app.

Run from this directory (the package is intentionally kept directly under `advisor/`):

```powershell
python -m unittest discover -s tests -v
```

Launch the cached Shiny application from the repository root:

```powershell
shiny run --reload Code/stock-app/app-express.py
```

The app automatically uses `tests/fixtures/market_data.csv` when it is
available, so the complete baseline workflow works without network access.
Set `ADVISOR_DATA_PATH` to use another canonical CSV, or remove the fixture
and configure the Yahoo cache provider for live data.

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

The next replacement should preserve the `AdvisorService` boundary: add a
saved FinRL strategy and optional Qwen/Smolagents explainer, then generate the
final untouched-period evidence. The baseline service and Shiny workflow are
already connected.

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

The project Conda environment is `CM3070-FP`. Use it for the optional-model
commands and verification:

```powershell
conda activate CM3070-FP
$env:PYTHONPATH="Code/advisor"
python Code/advisor/scripts/train_finrl.py --data path/to/mvp_market_data.csv
```

When only older data is available, train a separately labelled legacy policy
with explicit chronological boundaries:

```powershell
python Code/advisor/scripts/train_finrl.py `
  --data Code/data/2025-06-16_dow30.csv `
  --train-end 2018-12-31 --validation-end 2020-06-30 --test-end 2021-11-30 `
  --timesteps 20000 `
  --output Code/advisor/artifacts/models/finrl_a2c_legacy.zip `
  --metadata-output Code/advisor/artifacts/models/finrl_a2c_legacy.metadata.json
```

Evaluate that approved artifact against the baselines on the same held-out
interval:

```powershell
python Code/advisor/scripts/evaluate_finrl_legacy.py `
  --data Code/data/2025-06-16_dow30.csv `
  --model Code/advisor/artifacts/models/finrl_a2c_legacy.zip `
  --metadata Code/advisor/artifacts/models/finrl_a2c_legacy.metadata.json `
  --tickers AAPL MSFT JPM JNJ PG --start 2020-07-01 --end 2021-11-30 `
  --output-dir Code/advisor/artifacts/evaluation/finrl_legacy
```

Refresh the Yahoo cache and rebuild the prepared artifact with retries:

```powershell
python Code/advisor/scripts/prepare_data.py --refresh --retries 5 --retry-delay 5
```

The updater validates all five tickers and handles either Yahoo MultiIndex
column layout. If Yahoo is rate limited, it exits without replacing the
prepared artifact; retry later or pass `--source-csv` with a canonical local
download.

The explanation boundary is implemented in `advisor.explanation`. The Shiny
app offers an instant deterministic summary or an on-demand local
`Qwen/Qwen3-1.7B` explanation through Smolagents. Smolagents receives one
read-only `get_advisor_facts` tool containing validated service results; it has
no file, network, price, metric, or allocation access. `QwenExplainer` checks
numeric claims against the supplied context and falls back to the deterministic
template on timeout, unavailable models, empty output, or unsupported claims.
Set `ADVISOR_QWEN_MODEL` to select another locally cached Hugging Face Qwen
snapshot. Model loading happens only when the Qwen explanation mode is used.

The Shiny app wraps each analysis request in a recoverable state. Empty ticker
selection, unavailable data, model failures, and partial backtest results are
shown in the relevant view instead of terminating the application.

Notebook and app provenance is recorded in [`Code/PROVENANCE.md`](../PROVENANCE.md).
The Summary view also displays educational-use, data-date, model-version, and
dataset-version disclosures for every successful analysis.

The currently tested full-stack environment is Python 3.10.16 in Conda
environment `CM3070-FP`. The versioned universe correction is recorded in
`configs/universe.yaml`; `SWH` must be normalized to Yahoo symbol `SHW` before
dataset preparation.

The training command refuses to create an artifact unless the selected train,
validation, and test dates are covered. The checked-in MVP dates still target
2025, but explicit legacy overrides allow reproducible training on the tracked
CSV, which ends on 2021-11-30.

Approved policies are loaded for inference with
`advisor.finrl_adapter.load_approved_finrl_policy`. The loader validates the
artifact metadata before use and does not expose training through the app.

Use the ReservoirPy backend once installed:

```powershell
python scripts/evaluate_forecaster.py ../data/2025-06-16_dow30.csv --tickers AAPL MSFT --backend reservoirpy
```

Write the predictions and metrics to an artifact directory by adding `--output-dir artifacts/evaluation`.

Run multiple deterministic seeds and save a provenance bundle containing
predictions, per-seed metrics, mean/standard-deviation summaries, and metadata:

```powershell
python scripts/evaluate_forecaster.py ../data/2025-06-16_dow30.csv --tickers AAPL MSFT --min-train-size 25 --max-steps 100 --seeds 42 43 44 --output-dir artifacts/evaluation
```

Run both portfolio baselines and write returns, weights, and metrics:

```powershell
python scripts/backtest_portfolios.py ../data/2025-06-16_dow30.csv --tickers AAPL MSFT
```

The default simulation charges 10 basis points of transaction costs and 5 basis
points of slippage. Use `--transaction-cost-bps`, `--slippage-bps`, and
`--rebalance-every` to make these assumptions explicit for an experiment.

Add a canonical one-ticker market-index file for a fair index comparison:

```powershell
python scripts/backtest_portfolios.py path/to/mvp_market_data.csv --tickers AAPL MSFT JPM JNJ PG --index-csv path/to/index.csv
```

The comparison command uses identical overlapping dates and costs for every
baseline. Without `--index-csv`, it records that the index benchmark is
unavailable rather than substituting an invented series.

## Offline MVP scope

The reproducible deliverable is the offline MVP: fixture-backed data
validation, NumPy and ReservoirPy forecast evaluation, baseline portfolio
comparison, a legacy FinRL artifact and held-out comparison, grounded
explanation, and the unified Shiny app. Live Yahoo data and the 2025
untouched-period evaluation remain deferred until a source dataset covering
the configured dates is available.

The NumPy and ReservoirPy forecast commands have been smoke-tested in
`CM3070-FP` against the frozen fixture. Their generated evaluation bundles are
written under `artifacts/evaluation/` and include seed dispersion and data
provenance; they are not substitutes for the final untouched-period run.

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
