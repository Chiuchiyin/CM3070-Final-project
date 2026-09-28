# Financial Advisor Bot: 3–5 Minute Presentation Plan

## Target length

Four minutes, with the Shiny app open for the demonstration.

## 0:00–0:35 — Problem and scope

“This project is an educational financial advisor for stock-market portfolio
management. It analyses historical OHLCV data, forecasts the next close,
produces a constrained allocation, and explains the result for a non-technical
user. The application is explicitly a historical demonstration, not a live
trading system or personalised financial advice.”

Show the app title, historical-data disclosure, and five-stock default universe.

## 0:35–1:20 — End-to-end workflow

1. Select stocks, risk profile, and allocation strategy.
2. Click **Analyse**.
3. Show the Summary tab: forecast direction, cash floor, warnings, model/date
   disclosures, and active strategy.
4. Briefly show the Market Data candlestick/SMA chart and Forecast tab.

Explain that the production default is forecast-ranked allocation using a
20-period moving-average return forecast. FinRL A2C is selectable for the
five-stock MVP universe.

## 1:20–2:05 — AI chatbot

In the right-hand panel, leave **Qwen + Smolagents** selected and ask:

> Why is cash held, and what does the forecast mean?

Point out that Qwen receives only validated advisor facts through the
read-only `get_advisor_facts` tool. The app rejects raw JSON/tool payloads,
checks numeric claims, and falls back to grounded prose if the model fails.

## 2:05–3:00 — Algorithms and engineering

- Forecasting: last-close baseline, moving-average baseline, NumPy ESN, and
  optional ReservoirPy ESN.
- Portfolio decisions: forecast-ranked strategy for the live default and a
  saved FinRL A2C policy with rolling returns, current weights, cash, costs,
  and chronological splits.
- Engineering: shared `AdvisorService`, Python Shiny UI, validated OHLCV
  provider, lazy optional models, and inference-only saved-policy loading.

## 3:00–3:40 — Evaluation evidence

Open Evaluation and explain that all strategies use the same held-out interval
and trading costs. The saved legacy comparison covers 2020-07-01 to 2021-11-30:

| Strategy | Cumulative return |
| --- | ---: |
| FinRL A2C | 71.60% |
| Equal weight | 53.44% |
| Buy and hold | 52.85% |
| Forecast ranked | 46.36% |

State the limitation clearly: these are legacy historical results and do not
establish future performance. The intended 2025 untouched-period evaluation
is deferred because the available dataset ends in 2021.

## 3:40–4:00 — Verification and close

“The project has 54 offline tests in `CM3070-FP`, compile checks, saved model
metadata, reproducible evaluation CSVs, and documented provenance. The next
data-dependent step is refreshing the dataset and rerunning the untouched
2025 evaluation.”

## Backup answers

- **Why not real-time prices?** External Yahoo access is rate limited; the app
  labels the last historical row and keeps the demo reproducible.
- **Why keep equal weight?** It is a transparent benchmark for measuring the
  learned and forecast-based strategies.
- **Can the user retrain FinRL?** No. Training is offline; the app loads an
  approved artifact for inference only.
- **What happens if Qwen is unavailable?** The deterministic grounded responder
  remains available and the source is shown in the chatbot status.
