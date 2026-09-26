# Unification Plan: Agentic Financial Advisor

> **Implementation update — 2026-09-23:** The baseline vertical slice is now
> unified. `AdvisorService` supplies forecasts, risk-constrained allocations,
> warnings, allocation changes, and baseline metrics to one Python Shiny app.
> The deterministic explanation is grounded in those structured results. The
> remaining work is optional-stack evidence: approved FinRL artifacts,
> Qwen/Smolagents integration, and final untouched-period evaluation.

> **Implementation update — 2026-09-26:** The selected A2C training boundary
> is implemented with Gymnasium/Stable-Baselines3, versioned metadata, a
> matching inference observation builder, and a reproducible CLI. Training is
> intentionally blocked until the input dataset covers the configured test
> boundary; the available CSV ends in 2021 while the MVP test period ends in
> 2025.

> **Data updater update — 2026-09-26:** Yahoo cache refresh now supports both
> yfinance MultiIndex layouts, retries transient failures, validates ticker
> completeness, and exposes refresh controls through `prepare_data.py`. A live
> refresh was rate limited in the current environment, so no artifact was
> replaced.

> **Explanation update — 2026-09-26:** The language layer now has a structured
> facts contract, a narrow Smolagents tool, optional bounded Qwen generation,
> numeric-claim validation, and deterministic fallback. Quantitative outputs
> remain owned by the advisor service.

> **Shiny resilience update — 2026-09-26:** The unified app now handles empty
> selections, unavailable data, model failures, stale/partial results, and
> unavailable backtests through recoverable status states. Offline smoke tests
> verify the app compiles and remains connected to `AdvisorService`.

> **Provenance update — 2026-09-27:** Retained notebooks and source apps are
> classified in `Code/PROVENANCE.md`, and the Shiny Summary view now displays
> educational-use, data-date, model-version, dataset-version, and risk-profile
> disclosures.

> **Verification update — 2026-09-27:** Phase 7 offline coverage now includes
> scaling, observation construction, allocation constraints, risk metrics,
> explanation fallback, service integration, and Shiny smoke checks. Repository
> ignore rules and the versioned `SWH` to `SHW` correction are in place.

> **Artifact loading update — 2026-09-26:** Approved FinRL policies now have
> an inference-only loader that validates metadata before use. The UI cannot
> trigger training, and mismatched universes or observation schemas are
> rejected.

## 1. Goal

Turn the current collection of research notebooks and the Shiny stock explorer into one reproducible application that:

1. Loads and validates historical market data.
2. Produces next-period forecasts with a reservoir computing model.
3. Converts forecasts and market features into a portfolio allocation.
4. Backtests the allocation against simple baselines.
5. Explains the result in plain language through a Shiny interface.

The final system is an educational decision-support prototype, not an automated broker or a source of regulated financial advice.

## 2. Recommended MVP

The first integrated version should deliberately be smaller than the full proposal.

- Universe: five liquid stocks, selected from the current Dow universe.
- Frequency: daily bars.
- Forecast horizon: next trading day initially.
- Actions: target long-only portfolio weights plus cash.
- Data source: Yahoo Finance with a local CSV cache.
- Forecast model: Echo State Network (ESN), compared with a last-price baseline.
- Strategy model: a saved FinRL policy, compared with equal-weight and buy-and-hold baselines.
- Interface: the existing Python Shiny application.
- Language layer: Qwen through Smolagents, with a deterministic template fallback.
- Execution: recommendations and simulated allocations only; no live orders.

Once this path works end to end, expanding from five stocks to the Dow 30 is primarily a performance and evaluation task rather than another integration project.

## 3. System Boundaries

```text
User input
    |
    v
Shiny application
    |
    v
Advisor service -----------------------------------+
    |                                               |
    +--> Market data service --> validated dataset  |
    |             |                                 |
    |             +--> local dated cache            |
    |                                               |
    +--> ESN forecaster --> forecast records        |
    |                                               |
    +--> Strategy service --> target allocations    |
    |                                               |
    +--> Backtest service --> metrics and charts    |
    |                                               |
    +--> Explanation service <----------------------+
             |
             +--> Qwen/Smolagents when available
             +--> deterministic template fallback
```

The language model must not calculate prices, metrics, or portfolio weights. It receives validated structured results and explains them. This makes its output traceable and keeps the application functional when the model is unavailable.

Training and inference should also be separated:

- Offline jobs download data, fit models, evaluate them, and save versioned artifacts.
- The web application loads approved artifacts and performs inference only.
- A development-only option may retrain models, but it should not be part of the normal user request path.

## 4. Target Repository Structure

```text
Code/
|-- README.md
|-- UNIFICATION_PLAN.md
|-- pyproject.toml
|-- configs/
|   `-- mvp.yaml
|-- src/
|   `-- advisor/
|       |-- __init__.py
|       |-- config.py
|       |-- schemas.py
|       |-- data/
|       |   |-- provider.py
|       |   |-- yahoo.py
|       |   `-- validation.py
|       |-- forecasting/
|       |   |-- baseline.py
|       |   `-- esn.py
|       |-- strategy/
|       |   |-- baseline.py
|       |   `-- finrl_policy.py
|       |-- evaluation/
|       |   |-- backtest.py
|       |   `-- metrics.py
|       |-- explanation/
|       |   |-- context.py
|       |   |-- qwen.py
|       |   `-- template.py
|       `-- service.py
|-- app/
|   |-- app.py
|   `-- styles.css
|-- scripts/
|   |-- prepare_data.py
|   |-- train_forecaster.py
|   |-- train_strategy.py
|   `-- evaluate.py
|-- tests/
|   |-- unit/
|   |-- integration/
|   `-- fixtures/
|-- notebooks/
|   |-- exploration/
|   `-- upstream_examples/
|-- data/
|   |-- raw/
|   `-- processed/
|-- artifacts/
|   |-- models/
|   `-- evaluations/
`-- results/
```

Large data, downloaded models, caches, TensorBoard events, and generated charts should normally be ignored by Git. Small frozen test fixtures and final evaluation summaries should remain tracked.

## 5. Shared Data Contracts

Define the contracts before moving notebook code. Dataclasses or validated Pydantic models are suitable.

### Market data

Required columns:

```text
date, ticker, open, high, low, close, volume
```

Rules:

- Dates must be ordered and unique per ticker.
- Prices must be positive and OHLC relationships must be valid.
- Missing trading days must not be manufactured as zero-price observations.
- The ticker universe must be explicit and versioned.
- Correct the likely `SWH`/`SHW` ticker issue before generating a dataset.

### Forecast record

```text
ticker
as_of_date
horizon
predicted_close
predicted_return
model_name
model_version
training_end_date
validation_metrics
```

### Allocation record

```text
as_of_date
weights_by_ticker
cash_weight
strategy_name
strategy_version
risk_constraints
```

### Explanation context

The language model receives only structured facts such as forecasts, weights, backtest metrics, model dates, risk warnings, and baseline comparisons. The response should retain those values and clearly distinguish historical performance from future expectations.

## 6. Implementation Phases

### Phase 0: Preserve and classify the current work

- Record the current notebook outputs and result CSVs as a baseline.
- Move copied FinRL tutorials into `notebooks/upstream_examples`.
- Move original experiments into `notebooks/exploration` and give them descriptive names.
- Remove saved error outputs and incomplete cells from notebooks used in the final demonstration.
- Document which code is original, adapted, or directly sourced from FinRL tutorials.
- Add ignore rules for notebook checkpoints, model downloads, event logs, caches, and temporary Office files.

Exit criterion: every retained notebook has a stated purpose and provenance.

### Phase 1: Create the reproducible foundation

- Use Python 3.11 as the supported version unless dependency testing establishes another version.
- Create one dependency definition covering Shiny, FinRL, ReservoirPy, Smolagents, Qwen dependencies, testing, and formatting.
- Add `configs/mvp.yaml` for dates, tickers, model parameters, transaction costs, seeds, and artifact paths.
- Implement the Yahoo data provider and local cache.
- Add validation for empty responses, duplicates, missing values, invalid tickers, and incomplete date ranges.
- Store a small fixed dataset under `tests/fixtures` so tests do not require internet access.

Exit criterion: one command builds the same validated dataset from cache, and tests run offline.

### Phase 2: Establish honest forecasting evaluation

- Extract ESN code from `FinRLRCModel.ipynb` into `forecasting/esn.py`.
- Fit scalers only on the training window. Never use test-set minima, maxima, or future values during preprocessing.
- Implement expanding-window or rolling-window validation.
- Compare ESN predictions with at least:
  - Last observed price.
  - Moving average.
  - Optional ARIMA baseline if time permits.
- Report MAE, RMSE, MAPE with zero handling, directional accuracy, and R-squared.
- Run multiple random seeds and report mean and dispersion.
- Save the fitted scaler, model, configuration, data interval, and metrics together as one versioned artifact.

Exit criterion: `train_forecaster.py` creates an artifact and `evaluate.py` reproduces its saved metrics without data leakage.

### Phase 3: Unify portfolio strategy and backtesting

- First implement equal-weight and forecast-ranked strategies. These provide a working end-to-end path before FinRL is introduced.
- Extract one chosen FinRL environment and algorithm into `strategy/finrl_policy.py`; avoid maintaining multiple tutorial variants.
- Define the observation space, action space, reward, cash handling, rebalance frequency, and portfolio constraints in code and documentation.
- Include transaction costs and slippage in every reported strategy backtest.
- Use chronological train, validation, and test periods. The final test interval must remain untouched until model selection is complete.
- Compare against equal-weight, buy-and-hold, and an appropriate market index over identical dates.
- Report annual return, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, turnover, and cumulative return.

Exit criterion: a saved strategy artifact produces valid weights and a repeatable baseline comparison on an unseen period.

### Phase 4: Build one advisor service

Implement a single orchestration entry point similar to:

```python
result = advisor.analyse(
    tickers=["AAPL", "MSFT", "JPM", "JNJ", "PG"],
    as_of_date="2026-06-30",
    risk_profile="moderate",
)
```

The returned object should contain:

- Data freshness and warnings.
- Forecasts and uncertainty information.
- Target allocation and current-to-target changes.
- Baseline and model backtest metrics.
- Machine-generated or template-generated explanation.
- Model and dataset version identifiers.

No UI-specific code should be required to run this service. This boundary allows command-line tests and makes the Shiny application replaceable.

Exit criterion: one integration test runs the complete service from a fixed dataset without network access or an LLM.

### Phase 5: Integrate the explanation layer

- Expose narrow Smolagents tools that return structured advisor results; do not expose unrestricted calculation or arbitrary file access.
- Prompt Qwen to summarize supplied facts, disclose uncertainty, and avoid unsupported claims.
- Validate that numeric values mentioned in the generated response occur in the supplied context.
- Add a timeout and catch model loading or generation failures.
- Use `explanation/template.py` as a guaranteed fallback.
- Label all output as educational and show the data and model dates.

Exit criterion: the same analysis produces a useful response both with Qwen enabled and with Qwen disabled.

### Phase 6: Replace the stock demo with the unified Shiny app

Keep one implementation style; use `app-express.py` as the starting point and retire the duplicate core version once feature parity is reached.

Suggested interface:

- Sidebar: portfolio/ticker selection, as-of date, risk profile, and analysis button.
- Summary: data date, forecast direction, expected return, recommended cash weight, and risk status.
- Portfolio view: current and target allocations with rebalance differences.
- Forecast view: historical prices, predictions, and baseline comparison.
- Evaluation view: cumulative returns, drawdown, and metric table.
- Explanation view: grounded narrative plus explicit assumptions and limitations.

Fix the current application issues during migration:

- Use the final row for "Latest data" rather than `[:1]`.
- Handle empty datasets and ranges with fewer than two observations.
- Call the value "Latest close" unless a real-time quote endpoint is used.
- Show loading, model-unavailable, stale-data, and partial-result states.

Exit criterion: a user can perform the complete MVP workflow without opening a notebook.

### Phase 7: Verification and final evidence

- Unit-test data validation, lag creation, scaling, metrics, allocation constraints, and template explanations.
- Integration-test the offline end-to-end advisor service.
- Smoke-test the Shiny application with a cached fixture and with live Yahoo data.
- Confirm weights are finite, non-negative, and sum to one including cash.
- Confirm train/test boundaries and artifact metadata programmatically.
- Run a final untouched-period evaluation and generate tables directly from saved results.
- Update the report figures and claims from those generated results rather than manual transcription.

Exit criterion: a new checkout can follow the README and reproduce the final evaluation and application demo.

## 7. Suggested Milestones

| Milestone | Deliverable | Estimated effort |
| --- | --- | ---: |
| 1 | Repository structure, environment, configuration, and offline data tests | 3-4 days |
| 2 | ESN module with leakage-free walk-forward evaluation | 4-6 days |
| 3 | Baseline strategy and end-to-end advisor service | 3-4 days |
| 4 | FinRL policy with saved artifacts and fair backtest | 5-8 days |
| 5 | Qwen explanation layer and deterministic fallback | 2-4 days |
| 6 | Unified Shiny interface | 4-6 days |
| 7 | Testing, final evaluation, documentation, and demo preparation | 4-6 days |

These are implementation estimates, not calendar commitments. FinRL tuning and local Qwen performance are the largest sources of uncertainty.

## 8. Definition of Done

The MVP is complete when all of the following are true:

- A documented command starts the application from a clean environment.
- A second documented command reproduces training and evaluation.
- The app runs from cached data without network access.
- Forecast preprocessing uses training data only.
- The final evaluation uses a previously untouched chronological test period.
- Strategy results include costs and at least two simple baselines.
- Every displayed recommendation identifies its data date and model version.
- The language model cannot change quantitative outputs.
- Failure of Yahoo Finance or Qwen results in a clear recoverable state.
- Core behavior is covered by unit and integration tests.
- The final report clearly distinguishes original work from adapted tutorials.

## 9. Work Explicitly Deferred Until After the MVP

- Bloomberg and CCXT integrations.
- Live brokerage connectivity or order execution.
- Intraday or streaming data.
- Automatic continuous retraining.
- Full Dow 30 scale.
- User accounts and persistent real portfolios.
- Multimodal news or social-media signals.
- Multiple competing RL algorithms in the production path.
- Claims of personalized financial advice.

Deferring these items protects the project's strongest contribution: a transparent, reproducible connection between forecasting, portfolio decisions, evaluation, and explanation.

## 10. Immediate First Sprint

The first sprint should produce a thin vertical slice rather than improve the notebooks independently:

1. Create the package structure and dependency file.
2. Implement cached Yahoo data loading and validation for five tickers.
3. Implement last-price forecasting and equal-weight allocation baselines.
4. Implement the advisor service and deterministic explanation.
5. Connect that service to the Shiny app.
6. Add one offline integration test.

At the end of this sprint, the entire architecture will work with simple models. ESN, FinRL, and Qwen can then replace individual baseline components one at a time without changing the user workflow.
