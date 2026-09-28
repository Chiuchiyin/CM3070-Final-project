# Notebook and application provenance

This inventory records how retained research material maps to the unified
`Code/advisor/` implementation. Notebook cells are not executed by the Shiny
request path. Quantitative outputs shown by the app come from the validated
service and its saved artifacts.

| Source | Classification | Unified use | Provenance note |
| --- | --- | --- | --- |
| `FinRLRCModel.ipynb` | Exploration | `advisor.forecasting` ESN backends and evaluation | Research starting point; production code was reimplemented with explicit training-only scaling. |
| `FinRLRC_indicators_testbed.ipynb` | Exploration | Candidate feature ideas | Reviewed; indicators remain outside the production observation schema because the approved FinRL contract uses rolling returns and current weights. |
| `FinRL/FinRL_PortfolioAllocation_NeurIPS_2020.ipynb` | Upstream example adapted | Selected A2C policy contract | Source for the portfolio direction; production environment, cash action, costs, metadata, and chronological splits are implemented locally. |
| `FinRL/*.ipynb` other variants | Upstream examples | Reference only | Not in the production path because their action spaces do not match the MVP target-weight contract. |
| `portfolio_demo.ipynb` | Exploration/presentation | Shiny Portfolio and Evaluation views | Presentation concepts were reimplemented in the service-driven UI. |
| `LLM.ipynb`, `LLM_demo.ipynb` | Exploration | Optional Qwen/Smolagents explanation layer | `LLM_demo.ipynb` is the rough reference: Qwen, a `fetch_latest_data` tool, direct chat history, and CodeAgent. Production keeps the Qwen/Smolagents behavior but restricts the model to validated advisor facts and rejects raw tool/JSON payloads. |
| `LSTM vs. RC/simulation.ipynb` | Exploration | Optional benchmark evidence | Reviewed; retained as research only because it has no saved artifact integrated into the service contract. |
| `stock-app/app-express.py` | Original application | Unified Python Shiny app | Existing Shiny library and interaction style retained; direct Yahoo logic replaced by `AdvisorService`. |
| `stock-app/app-core.py` | Duplicate application | Retire after parity review | Kept for provenance until final smoke coverage confirms the Express app is the single entry point. |
| `Untitled.ipynb` | Unclassified experiment | None | Reviewed; no production dependency or final quantitative claim uses it. |
| `LSTM vs. RC/.ipynb_checkpoints`, `FinRL/.ipynb_checkpoints`, `.ipynb_checkpoints` | Generated notebook checkpoints | None | Ignored/generated artifacts; not evidence. |

## Original versus adapted code

- **Original implementation:** data validation/cache handling, forecast service
  contracts, NumPy ESN fallback, strategy allocation, backtesting, A2C
  environment, artifact metadata, explanation validation, and Shiny service
  integration under `Code/advisor/`.
- **Adapted direction:** A2C portfolio allocation and reservoir-computing ideas
  were selected from the FinRL and reservoir notebooks, then rewritten behind
  explicit contracts.
- **Optional language layer:** Qwen/Smolagents is an adapter around structured
  service facts. It cannot calculate or modify forecasts, metrics, or weights.

## Evidence rules

Final report claims must cite generated artifacts with data digest, dates, model
version, and evaluation interval. Notebook screenshots or manually transcribed
values are not final evidence.
