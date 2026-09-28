# Unified Advisor Working Log

This log tracks implementation work kept isolated under `Code/advisor/`.

## 2026-09-14

### Completed before this session

- Established canonical market-data loading and validation.
- Added last-close and deterministic NumPy ESN forecasters.
- Added leakage-free chronological forecast evaluation and model persistence.
- Added an equal-weight strategy, deterministic explanation, and `AdvisorService` boundary.
- Added offline unit tests and a forecasting evaluation CLI.

### Completed milestone: portfolio backtesting

Status: complete

Verified with 13 passing offline unit tests and a fixture-based CLI smoke test.

Planned work:

- Add a chronological portfolio backtest engine.
- Support equal-weight rebalancing and buy-and-hold baselines.
- Include configurable transaction costs and slippage.
- Calculate return, risk, drawdown, Sharpe, Sortino, and turnover metrics.
- Validate long-only asset weights plus cash summing to one.
- Add fixture-based tests and a CLI that writes returns, weights, and metrics.
- Update project documentation and run the complete test suite.

### Completed milestone: saved FinRL policy adapter

Status: complete

Verified with 18 passing offline unit tests.

Delivered:

- Added `FinRLPolicyAdapter`, which implements the shared portfolio-policy contract.
- Added lazy Stable-Baselines3 model loading with a clear optional-dependency error.
- Required an explicit observation builder so saved policy schemas cannot be guessed.
- Preserved the legacy FinRL portfolio action convention using stable softmax.
- Added long-only asset-plus-cash action support and strict ticker/action validation.
- Added warm-up handling that remains in cash until features are available.
- Added fake-model tests and documented the saved-model contract.

### Next milestone: optional Qwen explanation adapter

Status: pending

Planned work:

- Add a structured prompt/response boundary grounded in forecasts, allocations, and backtest metrics.
- Keep deterministic explanations as the default fallback.
- Load model dependencies only when explicitly configured.
- Test output validation and fallback behavior without requiring a local LLM.

### Stack decision: ReservoirPy and FinRL

Status: complete

- Added the `reservoirpy_esn` backend and service-compatible forecaster.
- Added a `--backend reservoirpy` evaluation option and a configurable ESN
  washout setting.
- Added `requirements-ml.txt`, which declares ReservoirPy, Stable-Baselines3,
  and FinRL as the full modelling stack.
- Kept the deterministic NumPy ESN as the dependency-free offline fallback.
- Verified the base stack with 19 passing tests and a forecasting CLI smoke test.

Environment note: ReservoirPy, FinRL, and Stable-Baselines3 are not installed
in the current Python environment, so live execution of those optional
backends remains to be validated after `pip install -r requirements-ml.txt`.

### Plan-tracking checklist

Status: active

`IMPLEMENTATION_TODO.md` is the operational checklist for following
`Code/UNIFICATION_PLAN.md`. It maps retained notebook and Shiny functionality
to a planned unified implementation item, tracks completion evidence, and
identifies the next safe milestone.

### Current milestone: MVP configuration and data preparation

Status: complete

Work started: 2026-09-14. This milestone begins only after the present working
log update. It will add a versioned five-ticker configuration and a command
that validates and writes a reproducible prepared-data artifact with metadata.

Delivered:

- Added `configs/mvp.yaml` with the five-ticker universe, dates, model settings,
  costs, risk profiles, and artifact paths.
- Added validated YAML loading in `advisor.config`.
- Added deterministic prepared CSV and SHA-256 metadata generation in
  `advisor.preparation`.
- Added `scripts/prepare_data.py` for Yahoo/cache or explicit CSV sources.
- Added offline tests for configuration validation and metadata stability.
- Verified with 22 passing tests and a five-ticker project-data smoke run.
- Confirmed that a source missing configured tickers is rejected rather than
  silently producing an incomplete MVP dataset.

### Current milestone: forecast-ranked strategy

Status: complete

Work started: 2026-09-15. This milestone begins only after the present working
log update and will add the next baseline required by `UNIFICATION_PLAN.md`
before selecting and training one FinRL strategy.

Delivered:

- Added a moving-average return forecaster for a transparent non-ESN baseline.
- Added deterministic forecast-ranked allocation with top-k selection, threshold-to-cash behavior, and tie-breaking.
- Added chronological `ForecastRankedPolicy` backtesting with warm-up cash handling and no future observations.
- Added CLI options and tests for the new baseline.
- Verified the complete offline suite with 28 passing tests.

### Current milestone: select and specify one FinRL training path

Status: in progress

Implementation resumed: 2026-09-21. Selected direction: the A2C portfolio
allocation path from `FinRL_PortfolioAllocation_NeurIPS_2020.ipynb`, adapted
to include explicit cash, transaction costs, slippage, chronological splits,
and versioned artifacts required by `UNIFICATION_PLAN.md`.

Delivered:

- Added the versioned A2C strategy section to `configs/mvp.yaml`.
- Added `finrl_training.py` with the strategy contract loader and chronological split validation.
- Added `FINRL_POLICY_SPEC.md` documenting the selected notebook path and production observation/action/reward/cash contract.
- Added contract tests and merged the existing Python Shiny dependencies into the unified runtime requirements.

The live FinRL trainer and saved policy artifact remain pending because the optional ReservoirPy/FinRL stack is not installed in this environment.

### Current milestone: train and validate the FinRL artifact

Status: pending

### Deferred items noted: 2026-09-23

- FinRL A2C training and live ReservoirPy/FinRL validation are skipped for now:
  the optional modelling stack is unavailable in the current environment and
  the interrupted training-module edits must not be guessed or force-completed.
- Git commit/push is deferred until the next verified implementation batch;
  no unverified FinRL artifact will be committed.
- A temporary Windows sandbox-helper lock blocked normal shell access during
  this continuation. Elevated read-only access was used to resume safely.

### Current feasible milestone: unified Shiny application scaffold

Status: in progress

The next work will focus on the same Python Shiny library used by
`Code/stock-app`, preserving ticker selection, date ranges, latest-close and
change metrics, candlestick/SMA charts, while adding the advisor-service
integration boundary. FinRL training remains explicitly deferred above.

The next step is to choose one portfolio environment and algorithm from the
FinRL notebooks, document its observation, action, reward, cash, and rebalance
contract, and add a reproducible training and artifact boundary. The production
UI will use Python Shiny, matching the existing stock app, through
`AdvisorService`; it will not call notebooks directly.

### Guardrails

- Keep all new implementation inside `Code/advisor/`.
- Avoid look-ahead: weights chosen for a date earn only subsequent returns.
- Keep strategy allocation independent from FinRL and UI integrations.
- Treat backtest output as research evidence, not guaranteed future performance.

## 2026-09-23

### Completed milestone: unified baseline service and Shiny application

Status: complete for the dependency-light MVP path

Delivered:

- Expanded `AdvisorService.analyse` with `as_of_date` and supported risk
  profiles (`conservative`, `moderate`, and `growth`).
- Added risk-profile cash floors while preserving the original fully invested
  behavior when no profile is supplied.
- Extended `AnalysisResult` with market data, freshness warnings, model and
  dataset versions, allocation changes, and baseline backtest metrics.
- Added an equal-weight historical backtest to service results when the input
  contains at least two dates.
- Updated the deterministic explanation to report supplied forecasts,
  allocations, cash, warnings, and historical metrics only.
- Replaced the stock-only `app-express.py` flow with one AdvisorService-driven
  Python Shiny application containing Summary, Portfolio, Forecast, and
  Evaluation views.
- Added automatic frozen-fixture mode so the baseline application can run
  without network access. `ADVISOR_DATA_PATH` can override the fixture.
- Updated `UNIFICATION_PLAN.md`, `IMPLEMENTATION_TODO.md`, and the advisor
  README with the new architecture, commands, status, and change history.
- Fixed `chronological_partitions` so FinRL boundary tests can use compact
  synthetic frames while provider-level OHLCV validation remains strict.

Verification:

- 30 offline unit tests pass with `PYTHONPATH=Code/advisor`.
- Updated service, explanation, training helper, and Shiny modules compile
  successfully.
- Manual fixture run confirmed conservative cash allocation, baseline metrics,
  and grounded explanation output.

### Remaining implementation work

Status: active

- Install and validate the optional ReservoirPy/FinRL stack.
- Add the reproducible FinRL A2C training command and save an approved policy
  artifact.
- Evaluate FinRL, equal-weight, buy-and-hold, and index baselines on identical
  untouched dates.
- Add Qwen/Smolagents as an optional explanation adapter with numeric-claim
  validation and deterministic fallback.
- Add explicit Shiny handling for empty data, short ranges, model failures,
  loading, and partial results, then add UI smoke tests.
- Complete notebook provenance classification and final generated evaluation
  evidence.

The baseline vertical slice is now connected end to end; remaining work is
optional-model validation, resilience coverage, provenance, and final evidence.

## 2026-09-26

### Completed milestone: reproducible A2C training boundary

Status: implementation complete; artifact validation blocked by dataset coverage

Delivered:

- Added `PortfolioAllocationEnv` using Gymnasium and Stable-Baselines3 A2C.
- Implemented rolling close-return observations with current asset and cash
  weights, explicit cash logits, stable-softmax target weights, transaction
  costs, slippage, and net log-return rewards.
- Added `dataset_sha256` and versioned artifact metadata containing the policy
  contract, seed, ticker universe, data digest, and train/validation/test
  boundaries.
- Added `RollingReturnAndWeightsObservationBuilder` to the saved-policy adapter
  so inference matches the training observation schema.
- Added `scripts/train_finrl.py` for reproducible offline training.
- Added four contract tests covering dataset digest stability, chronological
  boundary rejection, environment shapes, cash actions, and finite rewards.

Verification:

- `CM3070-FP` contains the optional stack, including ReservoirPy, FinRL,
  Stable-Baselines3, Smolagents, Transformers, Shiny, and yfinance.
- 33 offline tests pass in `CM3070-FP`.
- The trainer safely rejects `Code/data/2025-06-16_dow30.csv` because it ends
  on 2021-11-30 while the configured test boundary is 2025-06-16. No invalid
  policy artifact was produced.

Next action: prepare a dataset covering every configured chronological split,
then train and validate the approved artifact before opening the untouched test
period.

### Completed milestone: dataset updater repair

Status: implementation complete; live refresh rate limited

Delivered:

- Made Yahoo normalization independent of whether yfinance returns ticker-first
  or field-first MultiIndex columns.
- Added strict completeness checks for all requested tickers.
- Added `--refresh`, `--retries`, and `--retry-delay` to `prepare_data.py`.
- Added retry-aware `DataValidationError` handling so failed downloads cannot
  overwrite a valid cache or produce incomplete prepared data.
- Added regression tests for both Yahoo column layouts.

Verification:

- 34 offline tests pass in `CM3070-FP`.
- A configured refresh was attempted, but Yahoo Finance rate limited all five
  requested tickers. The updater failed safely without writing an artifact.
- The available tracked CSV remains insufficient for the 2025 test boundary;
  a refreshed source covering 2021-01-01 through 2025-06-16 is still required.

### Completed milestone: grounded explanation adapter

Status: implementation complete; model download/configuration remains optional

Delivered:

- Added `ExplanationContext`, a JSON-safe contract containing only validated
  forecasts, allocations, risk profile, warnings, versions, and backtest facts.
- Added the narrow `make_smolagents_facts_tool` surface. It returns facts only;
  it cannot read files or perform calculations.
- Added `QwenExplainer` with lazy model loading, bounded generation timeout,
  numeric-claim validation, and deterministic template fallback.
- Added tests for context construction, grounded numeric claims, unsupported
  claim fallback, and accepted grounded text.

Verification:

- 38 offline tests pass in `CM3070-FP`.
- The default service behavior remains deterministic and does not require a
  downloaded Qwen model.

Next action: configure a local Qwen generator in `CM3070-FP`, smoke-test it
against the same context, and keep the deterministic fallback as the default
when model loading or generation fails.

### Completed milestone: resilient unified Shiny states

Status: complete for offline smoke coverage

Delivered:

- Added a single recoverable `analysis_state` boundary around service calls.
- Empty ticker selections and service/data/model failures now produce status
  text rather than crashing reactive render functions.
- Summary, Portfolio, Forecast, and Evaluation views now show explicit
  unavailable or partial-result messages when analysis has not succeeded.
- Added offline smoke tests for app compilation, required views, and the
  AdvisorService integration boundary.

Verification:

- 40 tests pass in `CM3070-FP`.
- The Shiny script compiles successfully.
- Live Yahoo and untouched-period evaluation remain pending because Yahoo is
  currently rate limited and the tracked dataset ends in 2021.

### Completed milestone: inference-only approved policy loading

Status: complete for artifact contract; real artifact pending dataset refresh

Delivered:

- Added `load_approved_finrl_policy` to validate policy metadata before loading.
- Checks include ticker universe, A2C algorithm, approved observation schema,
  cash-action setting, and lookback compatibility.
- Loading is inference-only: no training code is reachable from this boundary.
- Added offline fake-loader tests for valid metadata and rejection of universe,
  schema, and lookback mismatches.

Verification:

- 43 tests pass in `CM3070-FP`.
- The real artifact remains pending until the refreshed dataset covers the
  configured 2025 test period.

### Environment update: CM3070-FP

The project verification environment is `CM3070-FP` at
`C:\\Users\\silve\\anaconda3\\envs\\CM3070-FP`. The full offline advisor
suite was rerun with that interpreter and passed all 30 tests. Its package list
includes FinRL 0.3.8, ReservoirPy 0.3.13.post1, Stable-Baselines3 2.6.1a1,
Smolagents 1.16.1, Transformers 4.52.3, Shiny 1.4.0, and yfinance 0.2.61.
Future training, evaluation, and app smoke-test commands should use this
environment rather than the base interpreter.

### Completed milestone: applied universe correction

Status: complete

Delivered:

- Added `canonical_ticker` to the data layer and applied the versioned
  `SWH` to `SHW` correction before filtering, caching, and Yahoo requests.
- Added a regression test for notebook-style `SWH` input.

Verification:

- 44 tests pass in `CM3070-FP`.
- The universe correction is now executable behavior, not documentation only.

### Completed milestone: multi-seed forecasting evidence

Status: complete for offline evaluation artifacts

Delivered:

- Added `evaluate_seeds` for identical-target walk-forward comparisons across
  NumPy ESN or ReservoirPy seeds.
- Added mean and standard-deviation summaries for the required regression and
  directional metrics.
- Added `write_forecast_evaluation` for predictions, per-seed metrics, seed
  summaries, and JSON provenance metadata with data digest and date interval.
- Extended `evaluate_forecaster.py` with configurable `--seeds`.
- Added tests for target consistency, dispersion columns, and metadata output.

Verification:

- Forecast tests pass in `CM3070-FP`; the full suite is the next gate.
- This evidence remains independent of the blocked live Yahoo refresh.

### Optional-stack validation: forecast backends

Status: complete on offline fixture

Verification:

- NumPy ESN multi-seed evaluation completed in `CM3070-FP` with seeds 42 and
  43, writing predictions, per-seed metrics, dispersion summary, and metadata.
- ReservoirPy ESN evaluation completed in `CM3070-FP` on the same fixture with
  seed 42 and the same walk-forward boundaries.
- Both runs used the frozen AAPL/MSFT fixture and therefore do not establish
  final 2025 untouched-period performance.

Next action: obtain the full five-ticker dataset through 2025-06-16, then run
the FinRL A2C trainer and validate all strategy baselines on identical dates.

### Completed milestone: fair baseline comparison path

Status: implementation complete; final unseen-period run pending data and policy

Delivered:

- Added `run_baseline_comparison` for equal-weight, buy-and-hold, and
  forecast-ranked policies with shared dates and cost assumptions.
- Added optional market-index evaluation from a supplied one-ticker canonical
  series with overlap validation.
- Updated `backtest_portfolios.py` with `--index-csv` and explicit metadata when
  no index is supplied.
- Preserved the CLI rebalance interval across all applicable baseline policies.

Verification:

- Fixture comparison CLI completed successfully for AAPL/MSFT.
- 47 tests pass in `CM3070-FP`.
- FinRL comparison remains pending until the approved artifact and full-date
  dataset exist.

## Scope decision: offline MVP delivery

Status: accepted constraint

Because Yahoo access is rate limited and the tracked historical CSV ends in
2021, the deliverable will use the reproducible offline path as its evidence
base. It includes validated fixture data, NumPy and ReservoirPy forecast
evaluation, multi-seed artifacts, equal-weight/buy-and-hold/forecast-ranked
portfolio comparisons, grounded explanation, and the unified Shiny app.

The following are explicitly deferred and must not be claimed as completed:

- live Yahoo refresh and live-data smoke evidence;
- a five-ticker dataset through 2025-06-16;
- FinRL A2C training and an approved policy artifact;
- FinRL versus baseline comparison on the untouched 2025 period.

This keeps the final report reproducible and honest while preserving the code
paths needed to complete those items when data access becomes available.

### Completed milestone: verification and repository hygiene

Status: complete for offline evidence

Delivered:

- Audited the test suite against Phase 7 requirements. Existing tests cover
  training-only scaling, observation construction, allocation constraints,
  transaction-cost metrics, explanation fallback, service integration, and
  Shiny smoke behavior.
- Expanded root ignore rules for notebook checkpoints, model/evaluation
  artifacts, caches, logs, TensorBoard events, and generated charts.
- Added `Code/advisor/configs/universe.yaml` with the corrected `SWH` to `SHW`
  ticker mapping.
- Recorded Python 3.10.16 as the currently tested full-stack environment in
  `CM3070-FP`.

Verification:

- The complete offline suite passes in `CM3070-FP`.
- Final live-data refresh, trained policy evaluation, and untouched-period
  evidence remain blocked by Yahoo rate limiting and dataset coverage.

## 2026-09-27

### Sidebar advisor chatbot (2026-09-28)

Status: implemented and verified offline

- Added a sidebar question box and Ask advisor action tied to the current
  `AdvisorService` analysis result.
- Deterministic responses cover allocations, forecasts, historical backtest,
  risk, and data-date questions without inventing missing figures.
- Qwen/Smolagents mode now accepts a question through the existing read-only
  facts tool and applies the numeric grounding validator before displaying it;
  timeouts, unavailable models, and unsupported claims fall back to the
  deterministic response.
- The sidebar shows whether an answer came from Qwen or the grounded fallback.
- `app-express.py` and `explanation.py` compile; all 53 advisor tests pass in
  the `CM3070-FP` environment.

### Section 4.2 Financial Advisor Bot requirements audit (2026-09-28)

Status: core requirements satisfied; delivery/UI gaps remain

- The project now defines a stock-market active portfolio advisor and has a
  reproducible data path using Yahoo/cache providers plus historical CSV data.
- The implementation includes transparent baselines, ESN forecasting,
  FinRL A2C training/evaluation on the available legacy period, and grounded
  deterministic or local Qwen/Smolagents explanations.
- The Python Shiny application provides the required non-technical workflow;
  51 tests pass in `CM3070-FP`, and saved evaluation artifacts provide evidence
  for the advice-generation workflow.
- The production app still uses the dependency-light LastClose forecast and
  equal-weight allocation by default. FinRL is currently an approved,
  inference-capable offline artifact and held-out comparison, not yet a
  selectable app strategy.
- The app Evaluation view currently exposes the service backtest result rather
  than the complete saved FinRL-versus-baselines comparison. This is a
  remaining integration task, alongside the responsive Summary/Evaluation
  layout fix and final report figures.

### Responsive Summary and Evaluation layout (2026-09-28)

Status: implemented and verified

- Split the six KPI cards into two compact rows to prevent horizontal overflow
  on common laptop widths.
- Reworked Summary into responsive explanation, data-quality, and disclosure
  cards. Long deterministic or Qwen text now wraps within the available width.
- Reworked Evaluation into a responsive comparison card with compact labels,
  formatted percentages/currency, and a short metric guide for non-technical
  users.
- Added narrow-screen CSS and retained the existing historical-data wording.
- Verification: `app-express.py` compiles and all 51 offline tests pass in
  `CM3070-FP`.
- Corrected the Shiny page-sidebar child order after the first restart check;
  the refreshed app now serves HTTP 200 on port 8000.

### Requirements backlog reconciliation (2026-09-28)

Status: complete

- Reconciled `IMPLEMENTATION_TODO.md` with the section 4.2 Financial Advisor
  Bot brief and marked the responsive Summary/Evaluation work complete.
- Consolidated the remaining work into final report/reproducibility tasks,
  an explicit decision about exposing the approved FinRL policy in the live
  app, deferred 2025 data work, and optional improvements.
- The current next item is the offline MVP report and artifact-backed figures;
  no 2025 performance claim will be made until matching data is available.

### Qwen and Smolagents explanation in Shiny (2026-09-28)

Status: complete for the local cached model

- Added a Shiny explanation selector for an instant template or local Qwen.
- Connected the existing `QwenExplainer` boundary to `AdvisorService` and
  exposed the explanation source in the Summary view.
- Added a local `Qwen/Qwen3-1.7B` generator. It invokes the narrow Smolagents
  `get_advisor_facts` tool, formats only service-owned figures, and uses Qwen's
  no-thinking chat template for a concise paragraph.
- Model loading is lazy and local-cache-only. Missing models, timeouts, empty
  output, and unsupported claims use the deterministic explanation.
- A real `CM3070-FP` run returned `explanation_source=qwen` with grounded
  AAPL allocation, cash, forecast, and historical cumulative-return figures.
- Added a service-level source/fallback test and stricter date and percentage
  validation; 51 advisor tests pass.

### Original stock explorer features restored (2026-09-28)

Status: complete

- Added a Market Data tab to the unified app with a Plotly candlestick chart,
  20-day SMA, chart-ticker selection, date-range filtering, and the latest
  OHLCV table.
- Restored current price, absolute change, and percentage-change KPI values
  from the original Shiny explorer.
- Expanded offline stock choices to the available tracked Dow 30 CSV, while
  keeping the five-stock MVP selection as the default.
- Verified the live app HTML contains Market Data, Summary, Portfolio,
  Forecast, and Evaluation tabs; HTTP 200 returned on port 8000.
- Full advisor suite remains green at 50 tests.

### Last-entry demo behavior (2026-09-28)

Status: complete

- The Shiny app now defaults its analysis date and six-month chart window to
  the last date found in the active CSV, rather than the current calendar date.
- The sidebar discloses that demo prices are historical, and the KPI labels the
  displayed price as the latest close.
- Chart ticker choices follow the selected analysis stocks, so the chart and
  KPIs use an available selected ticker.
- Verified the restarted app serves HTTP 200 on port 8000.

### Shiny launch compatibility fix (2026-09-28)

Status: complete

- Installed the declared `shinywidgets` dependency in `CM3070-FP`.
- The original Express entry point failed under the installed Shiny runtime
  while tagifying nested dynamic outputs. Converted `app-express.py` to the
  equivalent Shiny Core `App`/`server` form while preserving the Summary,
  Portfolio, Forecast, and Evaluation views and the `AdvisorService` boundary.
- Verified the app starts successfully at `http://127.0.0.1:8000/` and returns
  HTTP 200.

### Consolidated delivery backlog review

Status: documentation reconciled on 2026-09-28

- Removed stale statements that described legacy FinRL training, artifact
  loading, and held-out comparison as pending.
- Marked `app-express.py` as the only production Shiny entry point; retained
  `app-core.py` as provenance without deleting the original user work.
- Consolidated the remaining work into three categories in
  `IMPLEMENTATION_TODO.md`: required report/delivery work, deferred 2025/live
  data work, and optional improvements.
- Added `Code/stock-app/README.md` instructions for the production entry point
  and legacy app status.

### Legacy FinRL training path and consolidated backlog

Status: in progress

- Added explicit `--train-end`, `--validation-end`, `--test-end`, `--lookback`,
  and `--timesteps` overrides to `scripts/train_finrl.py`.
- Added `override_strategy_config` so older data can be used without changing
  the checked-in 2025 MVP dates.
- Added split metadata to saved policy artifacts.
- Made the Gymnasium action space finite, as required by Stable-Baselines3.
- Pinned the MLP A2C trainer to CPU for repeatable non-CNN training without
  unnecessary GPU warnings.
- Short legacy smoke training succeeded with the tracked CSV using train
  through 2018-12-31, validation through 2020-06-30, and test through
  2021-11-30. The smoke artifact is
  `artifacts/models/finrl_a2c_legacy_smoke.zip`.
- Full test suite passes 50 tests in `CM3070-FP`.

Remaining work is consolidated in `IMPLEMENTATION_TODO.md`: validate the full
legacy artifact and fair comparison, document the offline report, and defer only
the 2025 untouched-period claim until data access is restored. Possible
improvements include CPU-pinned SB3 training, checkpoint/resume support, CI,
and retiring the duplicate Shiny core app after parity review.

### Legacy FinRL artifact completed

Status: complete for the available historical period

- Completed the 20,000-timestep A2C run in `CM3070-FP`.
- Artifact: `artifacts/models/finrl_a2c_legacy.zip`.
- Metadata: `artifacts/models/finrl_a2c_legacy.metadata.json`.
- Split: train through 2018-12-31, validation through 2020-06-30, held-out test
  from 2020-07-01 through 2021-11-30.
- Fair comparison artifacts: `artifacts/evaluation/finrl_legacy/`.
- Held-out cumulative returns: FinRL 71.60%, equal-weight 53.44%, buy-and-hold
  52.85%, forecast-ranked 46.36%.
- The comparison uses 10 bps transaction costs and 5 bps slippage for every
  policy. Results are legacy-period evidence only; the 2025 untouched-period
  claim remains deferred.

### Completed milestone: provenance and disclosure coverage

Status: complete for documentation and offline UI evidence

Delivered:

- Added `Code/PROVENANCE.md` with classifications for all retained notebooks,
  checkpoint files, the original Shiny apps, and adapted FinRL/Qwen sources.
- Documented original implementation versus upstream/adapted ideas and the
  evidence rules for final report claims.
- Added a Shiny Summary disclosures table showing educational-use status, data
  date, model version, dataset version, and risk profile.
- Extended the Shiny smoke test to require the disclosure view.

Verification:

- The provenance inventory is tracked and linked from the unification work.
- Offline service, explanation, policy, data, and Shiny tests remain the next
  verification gate after this change.
