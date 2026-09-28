## Stock app

`app-express.py` is the single production entry point for the unified Python
Shiny application. It uses `AdvisorService` from `Code/advisor/` and exposes
the Summary, Portfolio, Forecast, and Evaluation views.

Run it from the repository root with:

```powershell
shiny run --reload Code/stock-app/app-express.py
```

`app-core.py` is retained as a historical provenance copy of the original
stock explorer. It is not part of the production workflow and should not be
used for new features.

<a href='https://connect.posit.cloud/publish?framework=shiny&sourceRepositoryURL=https%3A%2F%2Fgithub.com%2Fposit-dev%2Fpy-shiny-templates&sourceRef=main&sourceRefType=branch&primaryFile=stock-app%2Fapp-express.py&pythonVersion=3.11'><img src='https://cdn.connect.posit.cloud/assets/deploy-to-connect-blue.svg' align="right" /></a>
