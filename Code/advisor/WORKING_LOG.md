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
