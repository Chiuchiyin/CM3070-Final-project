# Unified Advisor Implementation Checklist

This is the execution checklist for [the unification plan](../UNIFICATION_PLAN.md).
It keeps new work inside `Code/advisor/` until the MVP is demonstrably complete.
An item is complete only when its stated verification evidence exists.

Last updated: 2026-09-26

## Status reconciliation (2026-09-26, approved artifact loading)

- Added inference-only FinRL artifact loading with metadata validation for
  ticker universe, algorithm, observation schema, and lookback.
- The loader never trains or mutates a policy and can be injected with a fake
  model loader for offline tests.
- Added three artifact-loading contract tests; the `CM3070-FP` suite now
  passes 43 tests.

## Status reconciliation (2026-09-26, Shiny resilience)

- Added a recoverable analysis state to the unified app for empty selections,
  unavailable data, model errors, and partial results.
- Added explicit unavailable/empty messages for summary, portfolio, forecast,
  and evaluation views.
- Added offline Shiny smoke tests for compilation, required views, and the
  `AdvisorService` boundary.
- Verification now passes 40 tests in `CM3070-FP`.

## Status reconciliation (2026-09-26, explanation layer)

- Added `ExplanationContext` as the structured boundary between quantitative
  service results and language generation.
- Added a narrow Smolagents facts tool with no file access or calculations.
- Added bounded `QwenExplainer` support with lazy loading, timeout handling,
  numeric-claim validation, and deterministic fallback.
- Added four offline explanation tests; the `CM3070-FP` suite now passes 38
  tests.

## Status reconciliation (2026-09-26, data updater)

- Fixed Yahoo downloads for both ticker-first and field-first MultiIndex
  layouts, with explicit requested-ticker completeness validation.
- Added `--refresh`, retry count, and retry delay controls to `prepare_data.py`.
- Added retry-aware error messages so rate limits do not silently leave an old
  or incomplete dataset in place.
- Added regression coverage; the `CM3070-FP` suite now passes 34 tests.
- A live refresh was attempted, but Yahoo returned `YFRateLimitError` for all
  five tickers. No invalid prepared artifact was written. Retry later or use a
  downloaded source CSV with the same canonical columns.

## Status reconciliation (2026-09-23)

- Updated the checklist after the unified service and Shiny migration.
- The cached baseline workflow is implemented and verified by 30 offline tests.
- The remaining work is now concentrated in provenance, optional-model
  artifacts, resilient UI error states, and final evaluation evidence.

## Status reconciliation (2026-09-26)

- Added the reproducible A2C training environment, artifact metadata writer,
  matching observation builder, and `scripts/train_finrl.py`.
- Added contract tests for stable dataset digests, cash actions, finite
  environment steps, and chronological data-boundary rejection.
- Verification now passes 33 offline tests in the `CM3070-FP` Conda
  environment.
- The configured artifact run is blocked by the available source dataset:
  `Code/data/2025-06-16_dow30.csv` ends on 2021-11-30, before the configured
  2025-06-16 test boundary. The trainer rejects this safely and does not create
  an invalid artifact.

## Change log (2026-09-23)

- Expanded `AdvisorService.analyse` with `as_of_date`, risk profiles, data
  freshness warnings, model/dataset versions, allocation changes, and
  baseline backtest metrics.
- Added cash floors for conservative, moderate, and growth profiles while
  preserving the original fully invested API when no profile is supplied.
- Grounded the deterministic explanation in supplied warnings, risk profile,
  cash allocation, and historical metrics. The explanation layer still cannot
  change quantitative outputs.
- Replaced the stock-only Express demo with a single AdvisorService-driven
  Shiny application exposing Summary, Portfolio, Forecast, and Evaluation
  views. Cached fixture mode is selected automatically for offline use.
- Kept Python Shiny, Plotly, yfinance, faicons, shinywidgets, and the websocket
  pin in the unified `requirements.txt`; no duplicate UI dependency profile is
  needed.
- Fixed `chronological_partitions` to remain a boundary helper for compact
  FinRL frames while providers continue to enforce full OHLCV validation.
- Verification: 30 offline unit tests pass and all updated Python modules
  compile successfully.

## Current Status Snapshot (2026-09-23)

- 30 offline tests pass.
- Data validation, Yahoo/cache loading, MVP configuration, and reproducible preparation are implemented.
- NumPy ESN, optional ReservoirPy ESN, moving-average, and last-close forecasters are available.
- Equal-weight, buy-and-hold, and forecast-ranked strategies are available; chronological backtesting includes costs and risk metrics.
- The FinRL A2C portfolio path is selected and specified; the trainer and
  artifact contract are implemented, but a trained artifact remains pending
  until a dataset covers all configured splits.
- The Python Shiny app now uses `AdvisorService` and exposes summary, portfolio, forecast, and evaluation views; the cached fixture path is offline-capable.
- Qwen/Smolagents, approved FinRL artifacts, and final untouched-period evidence remain pending.
- Preserve the current uncommitted strategy and test changes while continuing.

## Product Scope

The end product is one educational decision-support application, not a bundle
of notebooks. It must let a user select a small stock universe, inspect market
data and forecasts, receive a constrained simulated allocation, compare it to
fair baselines, and read a grounded explanation in the unified Shiny UI.

## Source Feature Map

| Existing source | Capability to preserve or replace | Unified destination | Status |
| --- | --- | --- | --- |
| `FinRLRCModel.ipynb` | Reservoir-computing forecast experiment | `advisor.forecasting` and evaluation CLI | Partial: NumPy and ReservoirPy backends exist; artifact/evaluation evidence remains |
| `FinRLRC_indicators_testbed.ipynb` | Technical-indicator experiments | Versioned feature/observation builders | Pending |
| `Code/FinRL/*.ipynb` | Portfolio environments, DRL training, and performance comparisons | One selected FinRL training pipeline plus `FinRLPolicyAdapter` | Partial: saved-policy adapter exists; reproducible environment, training, and artifacts remain |
| `portfolio_demo.ipynb` | Portfolio allocation presentation | Shiny portfolio and evaluation views | Implemented in baseline UI; final evidence pending |
| `LLM.ipynb`, `LLM_demo.ipynb` | Qwen-assisted financial narrative | Grounded Qwen/Smolagents explanation adapter | Pending |
| `LSTM vs. RC/simulation.ipynb` | Reservoir-model comparison evidence | Optional documented benchmark, not the production path | Pending review |
| `stock-app/app-express.py`, `stocks.py`, `styles.css` | Ticker selection, historical charts, controls, and Shiny styling | One unified Shiny app using `AdvisorService` | Implemented in `app-express.py`; smoke coverage pending |
| `stock-app/app-core.py` | Duplicate Shiny implementation | Retire only after Express feature parity | Pending |
| `Code/Untitled.ipynb` | Unclassified experiment | Provenance review; do not use in production until classified | Pending |

## Phase 0: Provenance And Repository Hygiene

- [ ] Classify each retained notebook as exploration, upstream example, or final evidence.
- [ ] Document original versus adapted FinRL and Qwen code.
- [ ] Move or link legacy notebooks without deleting user work.
- [ ] Add ignore rules for checkpoints, downloaded models, TensorBoard logs, caches, and generated charts.
- [ ] Record the known `SWH` to `SHW` ticker correction in a versioned universe file.

Exit evidence: every retained notebook has a stated purpose and provenance.

## Phase 1: Reproducible Foundation

- [x] Keep the unified implementation isolated in `Code/advisor/`.
- [x] Validate canonical OHLCV data and support frozen CSV data.
- [x] Implement cached Yahoo Finance data loading.
- [x] Add explicit cache refresh, retry, and Yahoo column-layout handling.
- [x] Maintain small offline market-data fixtures.
- [ ] Establish the supported Python version after testing the full optional stack.
- [x] Add a versioned MVP configuration for five tickers, dates, seeds, costs, paths, and risk profiles.
- [x] Consolidate runtime and optional-model dependencies into documented install profiles (`requirements.txt` and `requirements-ml.txt`).
- [x] Merge the existing app dependencies (`shiny`, `shinywidgets`, `plotly`, `faicons`, `yfinance`, and the websocket pin) into the unified install profile.
- [ ] Keep Python Shiny as the single UI library and retire duplicate app implementations only after feature parity.
- [x] Add a `prepare_data` command that writes a validated dataset and deterministic metadata.

Exit evidence: a fresh environment creates the same validated cached dataset and runs offline tests.

## Phase 2: Honest Forecasting

- [x] Implement a last-close baseline.
- [x] Implement deterministic NumPy ESN forecasting with training-only scaling.
- [x] Add optional ReservoirPy ESN support and a backend-selecting CLI.
- [x] Implement leakage-free chronological walk-forward evaluation.
- [x] Report MAE, RMSE, MAPE, R-squared, and directional accuracy.
- [x] Add moving-average baseline; decide separately whether ARIMA is justified.
- [ ] Evaluate multiple seeds and report mean and dispersion.
- [ ] Save complete NumPy and ReservoirPy forecast artifacts with configuration, scaler, data interval, and metrics.
- [ ] Reproduce saved-artifact metrics on a fixed untouched period.

Exit evidence: forecasting artifacts reproduce their evaluation without using future data.

## Phase 3: Portfolio Strategy And Backtesting

- [x] Backtest chronological equal-weight and buy-and-hold baselines.
- [x] Model transaction costs, slippage, turnover, drawdown, and risk metrics.
- [x] Enforce long-only asset-plus-cash weight constraints.
- [x] Define a shared policy contract and saved FinRL/SB3 policy adapter.
- [x] Add a forecast-ranked baseline strategy with deterministic ranking, cash threshold, and warm-up behavior.
- [x] Select the A2C portfolio-allocation path from the FinRL notebooks; retire other tutorial variants from the production path.
- [x] Define and version its observation space, action space, reward, cash handling, and rebalance schedule.
- [x] Add a reproducible FinRL training command that saves an approved policy artifact.
- [ ] Backtest FinRL, equal-weight, buy-and-hold, and a market-index benchmark over identical unseen dates.
- [x] Validate optional package availability in the `CM3070-FP` environment.
- [ ] Validate live ReservoirPy forecasting and FinRL inference on project data.

Exit evidence: a saved FinRL artifact produces valid allocations and a repeatable fair comparison on an untouched period.

## Phase 4: Advisor Service Contract

- [x] Provide a dependency-light `AdvisorService` for forecasts, allocations, and template explanations.
- [x] Extend `AnalysisResult` with data freshness, warnings, model/dataset versions, allocation changes, backtest results, and risk-profile constraints.
- [x] Accept `as_of_date` and `risk_profile` in the service contract.
- [x] Load approved model artifacts for inference only; prevent UI-triggered retraining.
- [x] Add an offline end-to-end service integration test using the frozen market-data fixture.

Exit evidence: one service call returns every fact required by the UI without network or LLM access.

## Phase 5: Qwen Explanation Layer

- [x] Add structured explanation context from validated service results.
- [x] Add narrow Smolagents tools with no arbitrary file or calculation access.
- [x] Add optional Qwen loading, timeout, failure handling, and deterministic template fallback.
- [x] Validate generated numeric claims against supplied context.
- [ ] Display educational-use, uncertainty, data-date, and model-date disclosures.

Exit evidence: the same analysis works with Qwen enabled or disabled and cannot alter quantitative results.

## Phase 6: Unified Shiny Application

- [x] Use `app-express.py` as the single migration starting point.
- [x] Replace direct stock-demo data logic with an `AdvisorService` call.
- [x] Add sidebar inputs for ticker universe, as-of date, risk profile, and analysis action.
- [x] Add summary view for data freshness, forecast direction, expected return, cash allocation, and risk state.
- [x] Add portfolio view for current versus target allocation and rebalance changes.
- [x] Add forecast view for price history, prediction, and baseline comparison.
- [x] Add evaluation view for cumulative return, drawdown, and metric comparison.
- [x] Add explanation view with assumptions, limitations, and unavailable-model state.
- [x] Handle empty data, short ranges, stale cache, loading, partial result, and model-unavailable states.
- [x] Use the final row for latest data and label it `Latest close` unless a real-time feed exists.
- [ ] Retire the duplicate core app only after feature parity and smoke-test coverage.

Exit evidence: the complete MVP workflow runs in Shiny without opening a notebook.

## Phase 7: Final Evidence And Delivery

- [ ] Add tests for feature construction, scaling, allocation constraints, metrics, and explanation fallback.
- [x] Add offline service and Shiny smoke tests.
- [ ] Smoke-test cached and live Yahoo data paths.
- [ ] Verify all artifact metadata and chronological boundaries programmatically.
- [ ] Run the final untouched-period evaluation and save generated tables/charts.
- [ ] Update report claims and figures only from saved generated results.
- [ ] Document clean setup, training/evaluation, app launch, known limitations, and provenance.

Exit evidence: a new checkout can reproduce the final evaluation and application demo from the README.

## Current Next Item

1. Obtain or prepare a canonical five-ticker dataset covering the configured
   2021-01-01 to 2025-06-16 interval.
2. Run `scripts/train_finrl.py` in `CM3070-FP` and save the approved A2C
   artifact plus metadata.
3. Validate the policy on the configured validation period before opening the
   untouched test period.
4. Add explicit Shiny error-state handling and smoke tests for cached and live
   data paths.

The unified baseline workflow is available offline through the Shiny
application and `AdvisorService`.
