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

### Environment update: CM3070-FP

The project verification environment is `CM3070-FP` at
`C:\\Users\\silve\\anaconda3\\envs\\CM3070-FP`. The full offline advisor
suite was rerun with that interpreter and passed all 30 tests. Its package list
includes FinRL 0.3.8, ReservoirPy 0.3.13.post1, Stable-Baselines3 2.6.1a1,
Smolagents 1.16.1, Transformers 4.52.3, Shiny 1.4.0, and yfinance 0.2.61.
Future training, evaluation, and app smoke-test commands should use this
environment rather than the base interpreter.
