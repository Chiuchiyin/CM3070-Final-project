## Stock app

`app-express.py` is the single production entry point for the unified Python
Shiny application. It uses `AdvisorService` from `Code/advisor/` and exposes
the Summary, Portfolio, Forecast, and Evaluation views.

The live allocation uses the forecast-ranked strategy: a 20-period moving
average return forecast ranks the selected stocks, holds the top three when
their estimated return is non-negative, and applies the selected risk
profile's cash floor. Equal-weight remains visible as a historical benchmark
in Evaluation.

The sidebar also includes an advisor chatbot. Run an analysis, enter a question
about allocations, forecasts, historical performance, risk, or the data date,
then click **Ask advisor**. The response uses the current analysis facts.
Choosing **Qwen + Smolagents** enables local model answers with a grounded
fallback; the app shows which source produced the response.

Run it from the repository root with:

```powershell
shiny run --reload Code/stock-app/app-express.py
```

`app-core.py` is retained as a historical provenance copy of the original
stock explorer. It is not part of the production workflow and should not be
used for new features.

<a href='https://connect.posit.cloud/publish?framework=shiny&sourceRepositoryURL=https%3A%2F%2Fgithub.com%2Fposit-dev%2Fpy-shiny-templates&sourceRef=main&sourceRefType=branch&primaryFile=stock-app%2Fapp-express.py&pythonVersion=3.11'><img src='https://cdn.connect.posit.cloud/assets/deploy-to-connect-blue.svg' align="right" /></a>
