# Final Report Evidence: Financial Advisor Bot

## Problem and scope

The system is an educational stock-market advisor for active portfolio
management. It consumes validated historical OHLCV data, forecasts the next
close, creates a long-only asset-plus-cash allocation, and explains the result
through a Python Shiny interface. It does not provide real-time prices or
personalised financial advice.

## Implemented workflow

1. CSV/Yahoo cache data is normalised to canonical OHLCV columns and validated.
2. A 20-period moving-average return forecast ranks the selected stocks for the
   live default allocation.
3. The selected risk profile imposes a cash floor and weights sum to one.
4. Equal-weight, buy-and-hold, forecast-ranked, and saved FinRL policies share
   one chronological backtesting contract.
5. Qwen + Smolagents receives only the validated advisor facts and is checked
   for unsupported numeric claims and structured payload leakage.

## Saved evaluation evidence

The approved legacy FinRL A2C artifact uses five stocks, a 20-step rolling
return-and-weights observation, 20,000 training steps, and chronological
train/validation/test boundaries. The held-out comparison is 2020-07-01 to
2021-11-30 with 10 bps transaction costs and 5 bps slippage.

| Strategy | Cumulative return | Annual return | Volatility | Maximum drawdown | Final value |
| --- | ---: | ---: | ---: | ---: | ---: |
| FinRL A2C | 71.60% | 46.40% | 28.54% | -19.46% | 17,159.77 |
| Equal weight | 53.44% | 35.28% | 14.28% | -9.97% | 15,343.72 |
| Buy and hold | 52.85% | 34.92% | 14.54% | -10.37% | 15,284.85 |
| Forecast ranked | 46.36% | 30.85% | 15.64% | -13.63% | 14,635.82 |

Source: `Code/advisor/artifacts/evaluation/finrl_legacy/finrl_legacy_comparison_metrics.csv`.

The comparison shows a higher historical return for FinRL alongside higher
volatility and drawdown. It is legacy-period evidence, not a claim about future
performance.

## Forecast evidence

Forecast artifacts under `Code/advisor/artifacts/evaluation/` record per-model
MAE, RMSE, MAPE, R-squared, directional accuracy, seed configuration, and data
intervals. The evaluation is walk-forward and keeps future rows out of each
training window. The simple last-close baseline remains a required comparator.

## Testing and reproducibility

- 54 offline tests pass in Conda environment `CM3070-FP`.
- The Shiny entry point compiles and returns HTTP 200 on port 8000.
- FinRL metadata validates the ticker universe, algorithm, observation schema,
  lookback, data digest, and chronological split boundaries.
- Qwen model loading is lazy and local-cache-only; deterministic fallback is
  always available.

## Limitations and deferred evidence

- The tracked dataset ends on 2021-11-30, so the configured 2025 untouched
  period cannot be claimed.
- Yahoo refresh was rate limited during development; the offline path is the
  reproducible demonstration path.
- The live default uses forecast-ranked allocation. FinRL A2C is selectable
  only for the exact five-stock MVP universe and is labelled as a legacy
  artifact.
- Historical backtests do not guarantee future returns.
