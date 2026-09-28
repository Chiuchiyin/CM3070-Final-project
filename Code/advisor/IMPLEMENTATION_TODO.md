# Unified Advisor Implementation Checklist

This is the execution checklist for [the unification plan](../UNIFICATION_PLAN.md).
It keeps new work inside `Code/advisor/` until the MVP is demonstrably complete.
An item is complete only when its stated verification evidence exists.

Last updated: 2026-09-28 (requirements and delivery backlog reconciled)

## Sidebar advisor chatbot (2026-09-28)

- [x] Add an Ask advisor control to the Shiny sidebar that uses the current
  analysis result rather than a separate data path.
- [x] Answer common questions about allocations, forecast estimates,
  historical backtest performance, risk, and data date from service-owned facts.
- [x] Allow Qwen + Smolagents to answer questions through the existing read-only
  facts tool, with numeric grounding validation and deterministic fallback.
- [x] Keep the chatbot educational and explicit about historical versus
  estimated outcomes; show the answer source in the sidebar.
- [x] Add chatbot behavior tests; 53 advisor tests pass in `CM3070-FP`.
- [ ] Add a visible loading indicator and multi-turn conversation history if
  user testing shows a need for them.

## Assessment brief audit (2026-09-28)

The section 4.2 Financial Advisor Bot requirements are covered as follows:

- [x] Define the problem as active portfolio advice for the stock market.
- [x] Ingest and validate historical OHLCV data through Yahoo/cache and CSV
  providers, with an offline fallback and an explicit historical-data label.
- [x] Implement forecasting and decision methods: last-close and ESN forecast
  paths, equal-weight and forecast-ranked strategies, plus a trained legacy
  FinRL A2C policy and fair held-out comparison.
- [x] Provide a non-technical web interaction through the Python Shiny app,
  including ticker selection, risk profile, date selection, allocations,
  forecasts, charts, and recoverable error states.
- [x] Provide grounded explanations through the deterministic template and
  optional local Qwen/Smolagents generation with fallback validation.
- [x] Provide software-engineering and data-science evidence: 51 passing
  offline tests in `CM3070-FP`, compile/smoke checks, provenance, and saved
  forecast/backtest/FinRL artifacts.
- [ ] Expose the saved FinRL policy and the complete multi-strategy comparison
  directly in the production app. The artifact and held-out evaluation exist,
  but the live app currently presents the dependency-light LastClose and
  equal-weight path.
- [ ] Complete the final report and figures from saved artifacts. The 2025
  untouched-period claim remains deferred because the available CSV ends on
  2021-11-30.
- [x] Finish the responsive Summary and Evaluation layouts so all content fits
  common laptop widths without horizontal crowding.

### Summary and Evaluation responsive pass (2026-09-28)

- [x] Split the KPI strip into two compact rows so cards wrap cleanly on
  laptop screens.
- [x] Put the Summary explanation and warnings into responsive cards, wrap
  generated prose, and keep disclosures in a separate readable panel.
- [x] Format Evaluation metrics with short human-readable labels, safe values
  for undefined Sharpe/Sortino results, and a compact metric guide.
- [x] Add responsive CSS for narrow screens and verify app compilation plus
  all 51 offline tests in `CM3070-FP`.
- [ ] Revisit the Evaluation panel when the complete saved FinRL comparison is
  exposed through the production app.

## Qwen and Smolagents app integration (2026-09-28)

- [x] Add an on-demand Qwen + Smolagents explanation choice in the unified
  Shiny app, alongside the instant deterministic summary.
- [x] Use the read-only `get_advisor_facts` tool and preformatted service facts
  for Qwen generation; keep model loading out of app startup.
- [x] Show whether an explanation came from Qwen or deterministic fallback.
- [x] Verify cached `Qwen/Qwen3-1.7B` inference through `AdvisorService` in
  `CM3070-FP`; the real response passed the grounded-claim validator.
- [x] Preserve fallback for unavailable models, timeouts, empty responses,
  and unsupported numeric claims.

## Launch verification update (2026-09-28)

- [x] Install the declared `shinywidgets` dependency in `CM3070-FP`.
- [x] Resolve Shiny runtime incompatibility in the production entry point.
- [x] Verify `app-express.py:app` starts and returns HTTP 200 on port 8000.
- [x] Restore original stock-explorer features: candlestick/SMA chart, date
  range, latest OHLCV table, price change KPIs, and expanded Dow 30 choices.
- [x] Default the offline demo to the active CSV's last historical entry and
  label the latest close as historical rather than real-time data.

## Consolidated status update (2026-09-27)

- [x] Train the legacy FinRL A2C artifact in `CM3070-FP` using train through
  2018-12-31, validation through 2020-06-30, and test through 2021-11-30.
- [x] Validate the saved artifact through the approved loader and adapter.
- [x] Run a fair held-out comparison on 2020-07-01 to 2021-11-30. Saved
  metrics are under `artifacts/evaluation/finrl_legacy/`.
- [ ] Complete the report and README reproducibility walkthrough.
- [ ] Obtain 2025 data before making any 2025 untouched-period claim.

## Scope decision (2026-09-27, data-access constraint)

- The project will deliver a reproducible offline MVP using the tracked fixture
  and available historical CSV evidence.
- Live Yahoo refresh, a 2025 five-ticker dataset, and the configured 2025
  untouched-period FinRL comparison are deferred when external data access is
  unavailable or rate limited.
- FinRL A2C training proceeds against the tracked older CSV with explicit
  legacy split dates and a separately labelled artifact.
- Deferred items remain documented as future work; they are not represented as
  completed performance claims.

## Status reconciliation (2026-09-27, fair portfolio comparison)

- Added `run_baseline_comparison` for equal-weight, buy-and-hold, and
  forecast-ranked strategies on one identical interval and cost configuration.
- Added optional one-ticker market-index input with overlap validation; the
  index is never fabricated when unavailable.
- Updated `backtest_portfolios.py` to write a comparison table and explicitly
  label index-unavailable runs.
- Added comparison tests; verification now passes 47 tests in `CM3070-FP`.

## Status reconciliation (2026-09-27, optional forecast backends)

- Ran the multi-seed NumPy ESN workflow in `CM3070-FP` on the frozen fixture
  and wrote a provenance bundle under `artifacts/evaluation/fixture_numpy`.
- Ran the ReservoirPy ESN workflow in `CM3070-FP` on the same fixture and wrote
  a matching bundle under `artifacts/evaluation/fixture_reservoirpy`.
- Both backends completed walk-forward evaluation without future rows in the
  training window. The artifacts are ignored generated evidence, not final
  untouched-period claims.

## Status reconciliation (2026-09-27, forecasting evidence)

- Added multi-seed walk-forward evaluation for NumPy ESN and ReservoirPy ESN.
- Added seed mean/standard-deviation summaries for MAE, RMSE, MAPE, R-squared,
  and directional accuracy.
- Added versioned forecast-evaluation artifacts containing predictions, metrics,
  seed dispersion, configuration, data digest, and evaluation interval.
- Extended `evaluate_forecaster.py` with `--seeds`; two new tests cover seed
  matching and artifact metadata.

## Status reconciliation (2026-09-27, ticker correction and verification)

- Wired the versioned `SWH` to `SHW` correction into canonical data
  normalization and Yahoo requests.
- Added a regression test proving notebook-style `SWH` input becomes canonical
  `SHW` output.
- Verification now passes 44 tests in `CM3070-FP`.

## Status reconciliation (2026-09-27, verification and repository hygiene)

- Confirmed existing tests cover training-only scaling, allocation constraints,
  risk metrics, explanation fallback, service integration, and Shiny smoke
  behavior.
- Added repository ignore rules for notebook checkpoints, model/evaluation
  artifacts, caches, logs, TensorBoard events, and generated charts.
- Added `configs/universe.yaml` with the versioned `SWH` to `SHW` correction.
- Confirmed the full optional stack in `CM3070-FP` uses Python 3.10.16; this is
  the tested project environment until a Python 3.11 compatibility run is
  completed.

## Status reconciliation (2026-09-27, provenance and disclosures)

- Added `Code/PROVENANCE.md` classifying retained notebooks, the original Shiny
  app, upstream examples, adapted production contracts, and generated
  checkpoints.
- Added explicit Summary disclosures for educational use, data date, model
  version, dataset version, and risk profile.
- Added smoke coverage for the disclosure view; the next verified suite will
  include this UI contract.

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
- The FinRL A2C portfolio path is selected and specified; the legacy trainer,
  approved artifact, and held-out comparison are complete. The configured 2025
  evaluation remains deferred until the dataset covers that period.
- The Python Shiny app now uses `AdvisorService` and exposes summary, portfolio, forecast, and evaluation views; the cached fixture path is offline-capable.
- Qwen/Smolagents integration and final untouched-period evidence remain
  optional/deferred; the legacy FinRL artifact is complete and approved.
- Preserve the current uncommitted strategy and test changes while continuing.

## Product Scope

The end product is one educational decision-support application, not a bundle
of notebooks. It must let a user select a small stock universe, inspect market
data and forecasts, receive a constrained simulated allocation, compare it to
fair baselines, and read a grounded explanation in the unified Shiny UI.

## Source Feature Map

| Existing source | Capability to preserve or replace | Unified destination | Status |
| --- | --- | --- | --- |
| `FinRLRCModel.ipynb` | Reservoir-computing forecast experiment | `advisor.forecasting` and evaluation CLI | Implemented for NumPy and ReservoirPy fixture evidence; 2025 evidence deferred |
| `FinRLRC_indicators_testbed.ipynb` | Technical-indicator experiments | Versioned feature/observation builders | Pending |
| `Code/FinRL/*.ipynb` | Portfolio environments, DRL training, and performance comparisons | One selected FinRL training pipeline plus `FinRLPolicyAdapter` | Implemented for the legacy A2C artifact and held-out comparison; 2025 evidence deferred |
| `portfolio_demo.ipynb` | Portfolio allocation presentation | Shiny portfolio and evaluation views | Implemented in baseline UI; final evidence pending |
| `LLM.ipynb`, `LLM_demo.ipynb` | Qwen-assisted financial narrative | Grounded Qwen/Smolagents explanation adapter | Implemented in the unified Shiny app with on-demand local model and fallback |
| `LSTM vs. RC/simulation.ipynb` | Reservoir-model comparison evidence | Optional documented benchmark, not the production path | Pending review |
| `stock-app/app-express.py`, `stocks.py`, `styles.css` | Ticker selection, historical charts, controls, and Shiny styling | One unified Shiny app using `AdvisorService` | Implemented in `app-express.py` with smoke coverage |
| `stock-app/app-core.py` | Duplicate Shiny implementation | Retained as legacy provenance only | No longer a production entry point |
| `Code/Untitled.ipynb` | Unclassified experiment | Provenance review; do not use in production until classified | Pending |

## Phase 0: Provenance And Repository Hygiene

- [x] Classify each retained notebook as exploration, upstream example, or final evidence.
- [x] Document original versus adapted FinRL and Qwen code.
- [x] Move or link legacy notebooks without deleting user work. (`Code/PROVENANCE.md` links the retained material without deleting it.)
- [x] Add ignore rules for checkpoints, downloaded models, TensorBoard logs, caches, and generated charts.
- [x] Record the known `SWH` to `SHW` ticker correction in a versioned universe file.

Exit evidence: every retained notebook has a stated purpose and provenance.

## Phase 1: Reproducible Foundation

- [x] Keep the unified implementation isolated in `Code/advisor/`.
- [x] Validate canonical OHLCV data and support frozen CSV data.
- [x] Implement cached Yahoo Finance data loading.
- [x] Add explicit cache refresh, retry, and Yahoo column-layout handling.
- [x] Maintain small offline market-data fixtures.
- [x] Establish the tested Python version: Python 3.10.16 in `CM3070-FP`.
- [x] Add a versioned MVP configuration for five tickers, dates, seeds, costs, paths, and risk profiles.
- [x] Consolidate runtime and optional-model dependencies into documented install profiles (`requirements.txt` and `requirements-ml.txt`).
- [x] Merge the existing app dependencies (`shiny`, `shinywidgets`, `plotly`, `faicons`, `yfinance`, and the websocket pin) into the unified install profile.
- [x] Keep Python Shiny as the single UI library and designate `app-express.py`
  as the only production entry point; retain `app-core.py` for provenance.
- [x] Add a `prepare_data` command that writes a validated dataset and deterministic metadata.

Exit evidence: a fresh environment creates the same validated cached dataset and runs offline tests.

## Phase 2: Honest Forecasting

- [x] Implement a last-close baseline.
- [x] Implement deterministic NumPy ESN forecasting with training-only scaling.
- [x] Add optional ReservoirPy ESN support and a backend-selecting CLI.
- [x] Implement leakage-free chronological walk-forward evaluation.
- [x] Report MAE, RMSE, MAPE, R-squared, and directional accuracy.
- [x] Add moving-average baseline; decide separately whether ARIMA is justified.
- [x] Evaluate multiple seeds and report mean and dispersion.
- [x] Save complete NumPy and ReservoirPy forecast artifacts with configuration, data interval, seeds, and metrics.
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
- [x] Add fair comparison path for equal-weight, buy-and-hold, forecast-ranked,
  and optional market-index baselines over identical dates.
- [x] Backtest the legacy FinRL artifact with the comparison path on its
  held-out 2020-07-01 to 2021-11-30 test interval.
- [x] Validate optional package availability in the `CM3070-FP` environment.
- [x] Validate ReservoirPy forecasting on the offline project fixture.
- [x] Validate legacy FinRL inference on project data after the full artifact
  run.
- [ ] Keep the 2025 untouched-period FinRL comparison deferred until data
  through 2025-06-16 is available.

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
- [x] Display educational-use, uncertainty, data-date, and model-date disclosures.

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
- [x] Complete Express feature-parity review and keep `app-core.py` as a
  non-production provenance copy.

Exit evidence: the complete MVP workflow runs in Shiny without opening a notebook.

## Phase 7: Final Evidence And Delivery

- [x] Add tests for feature construction, scaling, allocation constraints, metrics, and explanation fallback.
- [x] Add offline service and Shiny smoke tests.
- [x] Smoke-test the cached/offline data path.
- [x] Verify available artifact metadata and chronological boundaries programmatically.
- [ ] Smoke-test the live Yahoo path (deferred: external rate limit).
- [ ] Run the final untouched-period evaluation and save generated tables/charts
  (deferred: no 2025 dataset; the legacy artifact is complete).
- [ ] Update report claims and figures only from saved generated results.
- [x] Document clean setup, training/evaluation, app launch, known limitations,
  and provenance.

Exit evidence: a new checkout can reproduce the final evaluation and application demo from the README.

## Consolidated remaining work and improvement backlog

This is the current execution order. Items marked deferred depend on access to
newer external data and must not block the offline MVP demonstration.

### Required next for final delivery

- [x] Finish and verify the 20,000-timestep legacy FinRL artifact.
- [x] Load the artifact through `load_approved_finrl_policy` and run inference
  on the five-ticker historical panel.
- [x] Run fair legacy test-period comparison across FinRL and all baselines.
- [x] Record exact results, artifact paths, data digest, and limitations.
- [x] Add the legacy setup/training/evaluation commands to the README.
- [x] Reconcile the section 4.2 Financial Advisor Bot requirements with
  implementation evidence and document the remaining gaps.
- [x] Make the Summary and Evaluation views responsive and verify the live app
  after the layout change.
- [ ] Add a final report walkthrough covering problem definition, data source,
  model design, portfolio constraints, explanation layer, and limitations.
- [ ] Generate the final report tables and figures only from saved artifacts,
  including the legacy FinRL comparison and forecast evaluation bundles.
- [ ] Add a reproducibility checklist to the README: environment activation,
  tests, app launch, data preparation, training, and evaluation commands.
- [ ] Decide whether the production app should expose a selectable FinRL
  strategy. If yes, load the approved artifact for inference only and add a
  UI comparison against equal-weight and forecast-ranked strategies.

### Deferred

- [ ] Refresh Yahoo data through 2025-06-16 when rate limits permit.
- [ ] Run the untouched 2025 forecast and FinRL comparison.
- [ ] Update report figures only from regenerated saved artifacts.

### Possible improvements

- [x] Force SB3 training to CPU; expose checkpoint/resume options remains open.
- [ ] Add training progress/evaluation checkpoints and seed dispersion plots.
- [x] Complete the Shiny feature-parity review; retain `app-core.py` only as
  historical provenance.
- [ ] Add CI for compile, offline tests, and a short legacy training smoke run.
- [ ] Review the pending technical-indicator notebook and classify whether any
  indicator should enter the shared observation builder.
- [ ] Review `LSTM vs. RC/simulation.ipynb` and `Code/Untitled.ipynb`; retain
  them as documented experiments unless they provide reproducible evidence.
- [ ] Add a user-facing loading/progress state while the local Qwen model is
  loaded and cache successful explanations for repeated analyses.
- [ ] Add a dedicated evaluation chart showing cumulative portfolio value and
  drawdown for the saved comparison artifacts.

## Current Next Items

1. Finalize the offline MVP report and demo using saved fixture-based evidence.
2. Add the reproducibility walkthrough and final artifact-backed figures.
3. Decide and document whether FinRL becomes a selectable production-app
   strategy; keep UI inference separate from training.
4. Keep the documented Yahoo/FinRL workflow available for a future data
   refresh.
5. Do not claim 2025 untouched-period performance until data through
   2025-06-16 is available.

The unified baseline workflow is available offline through the Shiny
application and `AdvisorService`.
