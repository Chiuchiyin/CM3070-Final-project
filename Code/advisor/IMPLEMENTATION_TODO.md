# Unified Advisor Implementation Checklist

This is the execution checklist for [the unification plan](../UNIFICATION_PLAN.md).
It keeps new work inside `Code/advisor/` until the MVP is demonstrably complete.
An item is complete only when its stated verification evidence exists.

Last updated: 2026-09-15

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
| `portfolio_demo.ipynb` | Portfolio allocation presentation | Shiny portfolio and evaluation views | Pending |
| `LLM.ipynb`, `LLM_demo.ipynb` | Qwen-assisted financial narrative | Grounded Qwen/Smolagents explanation adapter | Pending |
| `LSTM vs. RC/simulation.ipynb` | Reservoir-model comparison evidence | Optional documented benchmark, not the production path | Pending review |
| `stock-app/app-express.py`, `stocks.py`, `styles.css` | Ticker selection, historical charts, controls, and Shiny styling | One unified Shiny app using `AdvisorService` | Pending |
| `stock-app/app-core.py` | Duplicate Shiny implementation | Retire only after Express feature parity | Pending |

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
- [x] Maintain small offline market-data fixtures.
- [ ] Establish the supported Python version after testing the full optional stack.
- [x] Add a versioned MVP configuration for five tickers, dates, seeds, costs, paths, and risk profiles.
- [x] Consolidate runtime and optional-model dependencies into documented install profiles (`requirements.txt` and `requirements-ml.txt`).
- [x] Add a `prepare_data` command that writes a validated dataset and deterministic metadata.

Exit evidence: a fresh environment creates the same validated cached dataset and runs offline tests.

## Phase 2: Honest Forecasting

- [x] Implement a last-close baseline.
- [x] Implement deterministic NumPy ESN forecasting with training-only scaling.
- [x] Add optional ReservoirPy ESN support and a backend-selecting CLI.
- [x] Implement leakage-free chronological walk-forward evaluation.
- [x] Report MAE, RMSE, MAPE, R-squared, and directional accuracy.
- [ ] Add moving-average baseline and decide whether ARIMA is justified.
- [ ] Evaluate multiple seeds and report mean and dispersion.
- [ ] Save complete NumPy and ReservoirPy forecast artifacts with configuration, scaler, data interval, and metrics.
- [ ] Reproduce saved-artifact metrics on a fixed untouched period.

Exit evidence: forecasting artifacts reproduce their evaluation without using future data.

## Phase 3: Portfolio Strategy And Backtesting

- [x] Backtest chronological equal-weight and buy-and-hold baselines.
- [x] Model transaction costs, slippage, turnover, drawdown, and risk metrics.
- [x] Enforce long-only asset-plus-cash weight constraints.
- [x] Define a shared policy contract and saved FinRL/SB3 policy adapter.
- [ ] Add a forecast-ranked baseline strategy.
- [ ] Select one FinRL portfolio environment and algorithm from the notebooks.
- [ ] Define and version its observation space, action space, reward, cash handling, and rebalance schedule.
- [ ] Add a reproducible FinRL training command that saves an approved policy artifact.
- [ ] Backtest FinRL, equal-weight, buy-and-hold, and a market-index benchmark over identical unseen dates.
- [ ] Validate live ReservoirPy and FinRL paths after installing `requirements-ml.txt`.

Exit evidence: a saved FinRL artifact produces valid allocations and a repeatable fair comparison on an untouched period.

## Phase 4: Advisor Service Contract

- [x] Provide a dependency-light `AdvisorService` for forecasts, allocations, and template explanations.
- [ ] Extend `AnalysisResult` with data freshness, warnings, model/dataset versions, allocation changes, backtest results, and risk-profile constraints.
- [ ] Accept `as_of_date` and `risk_profile` in the service contract.
- [ ] Load approved model artifacts for inference only; prevent UI-triggered retraining.
- [ ] Add an offline end-to-end service integration test using the five-ticker fixture.

Exit evidence: one service call returns every fact required by the UI without network or LLM access.

## Phase 5: Qwen Explanation Layer

- [ ] Add structured explanation context from validated service results.
- [ ] Add narrow Smolagents tools with no arbitrary file or calculation access.
- [ ] Add optional Qwen loading, timeout, failure handling, and deterministic template fallback.
- [ ] Validate generated numeric claims against supplied context.
- [ ] Display educational-use, uncertainty, data-date, and model-date disclosures.

Exit evidence: the same analysis works with Qwen enabled or disabled and cannot alter quantitative results.

## Phase 6: Unified Shiny Application

- [ ] Use `app-express.py` as the single migration starting point.
- [ ] Replace direct stock-demo data logic with an `AdvisorService` call.
- [ ] Add sidebar inputs for ticker universe, as-of date, risk profile, and analysis action.
- [ ] Add summary view for data freshness, forecast direction, expected return, cash allocation, and risk state.
- [ ] Add portfolio view for current versus target allocation and rebalance changes.
- [ ] Add forecast view for price history, prediction, and baseline comparison.
- [ ] Add evaluation view for cumulative return, drawdown, and metric comparison.
- [ ] Add explanation view with assumptions, limitations, and unavailable-model state.
- [ ] Handle empty data, short ranges, stale cache, loading, partial result, and model-unavailable states.
- [ ] Use the final row for latest data and label it `Latest close` unless a real-time feed exists.
- [ ] Retire the duplicate core app only after feature parity and smoke-test coverage.

Exit evidence: the complete MVP workflow runs in Shiny without opening a notebook.

## Phase 7: Final Evidence And Delivery

- [ ] Add tests for feature construction, scaling, allocation constraints, metrics, and explanation fallback.
- [ ] Add offline service and Shiny smoke tests.
- [ ] Smoke-test cached and live Yahoo data paths.
- [ ] Verify all artifact metadata and chronological boundaries programmatically.
- [ ] Run the final untouched-period evaluation and save generated tables/charts.
- [ ] Update report claims and figures only from saved generated results.
- [ ] Document clean setup, training/evaluation, app launch, known limitations, and provenance.

Exit evidence: a new checkout can reproduce the final evaluation and application demo from the README.

## Current Next Item

Add the forecast-ranked baseline strategy. It is the next honest allocation
baseline needed before selecting and training a FinRL portfolio policy.
