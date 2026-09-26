"""Unified Python Shiny interface for the educational advisor."""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_plotly

ADVISOR_ROOT = Path(__file__).resolve().parents[1] / "advisor"
sys.path.insert(0, str(ADVISOR_ROOT))
from advisor.data import CsvMarketDataProvider, YahooMarketDataProvider
from advisor.service import AdvisorService

DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "PG"]
fixture = ADVISOR_ROOT / "tests" / "fixtures" / "market_data.csv"
data_path = os.getenv("ADVISOR_DATA_PATH")
if data_path or fixture.exists():
    fixture_path = Path(data_path) if data_path else fixture
    provider = CsvMarketDataProvider(fixture_path)
    if not data_path:
        DEFAULT_TICKERS = sorted(pd.read_csv(fixture_path)["ticker"].unique().tolist())
else:
    provider = YahooMarketDataProvider(ADVISOR_ROOT.parent / "data" / "cache")
service = AdvisorService(provider, dataset_version="mvp_runtime")

ui.page_opts(title="Agentic Financial Advisor", fillable=True)
with ui.sidebar():
    ui.h4("Analysis inputs")
    ui.input_selectize("tickers", "Stocks", choices=DEFAULT_TICKERS, selected=DEFAULT_TICKERS, multiple=True)
    ui.input_date("as_of", "As-of date", value=None)
    ui.input_select("risk_profile", "Risk profile", choices={"conservative": "Conservative", "moderate": "Moderate", "growth": "Growth"}, selected="moderate")
    ui.input_action_button("analyse", "Analyse", class_="btn-primary")
    ui.p("Educational decision-support prototype. Simulated allocations only.")

@reactive.calc
@reactive.event(input.analyse)
def analysis_state():
    try:
        if not input.tickers():
            return {"result": None, "error": "Select at least one ticker."}
        value = input.as_of()
        result = service.analyse(tickers=input.tickers(), as_of_date=value if value else None, risk_profile=input.risk_profile())
        return {"result": result, "error": None}
    except Exception as exc:
        return {"result": None, "error": str(exc)}

def current_result():
    state = analysis_state()
    if state["result"] is None:
        return None
    return state["result"]

def status_frame():
    state = analysis_state()
    if state["error"]:
        return pd.DataFrame({"Status": [f"Analysis unavailable: {state['error']}"]})
    result = state["result"]
    rows = result.warnings or ["Analysis ready. No data-quality warnings."]
    return pd.DataFrame({"Status": rows})

def empty_frame(message):
    return pd.DataFrame({"Status": [message]})

with ui.layout_column_wrap(fill=False):
    with ui.value_box():
        "Latest close"
        @render.text
        def latest_close():
            result = current_result()
            if result is None: return "Unavailable"
            return str(result.market_data.sort_values("date")["date"].iloc[-1].date())
    with ui.value_box():
        "Expected return"
        @render.text
        def expected_return():
            result = current_result()
            if result is None: return "Unavailable"
            return f"{result.forecasts['predicted_return'].mean():+.2%}"
    with ui.value_box():
        "Cash allocation"
        @render.text
        def cash_allocation():
            result = current_result()
            if result is None: return "Unavailable"
            return f"{result.allocations['cash_weight'].iloc[0]:.1%}"

with ui.navset_tab():
    with ui.nav_panel("Summary"):
        @render.text
        def explanation():
            result = current_result()
            if result is None: return analysis_state()["error"] or "Press Analyse to begin."
            return result.explanation
        @render.data_frame
        def warnings(): return status_frame()
        @render.data_frame
        def disclosures():
            result = current_result()
            if result is None:
                return empty_frame("Model and data disclosures will appear after analysis.")
            return pd.DataFrame({
                "Field": ["Use", "Data date", "Model version", "Dataset version", "Risk profile"],
                "Value": [
                    "Educational decision support; simulated allocations only",
                    str(result.as_of_date.date()) if result.as_of_date is not None else "Unavailable",
                    result.model_version,
                    result.dataset_version,
                    result.risk_profile,
                ],
            })
    with ui.nav_panel("Portfolio"):
        @render.data_frame
        def allocations():
            result = current_result()
            return result.allocations if result is not None else empty_frame("Run an analysis to view allocations.")
        @render.data_frame
        def allocation_changes():
            result = current_result()
            return result.allocation_changes if result is not None else empty_frame("Run an analysis to view allocation changes.")
    with ui.nav_panel("Forecast"):
        @render_plotly
        def forecast_chart():
            result = current_result()
            if result is None:
                return go.Figure().update_layout(title="Forecast unavailable until analysis succeeds")
            frame = result.market_data.sort_values("date"); fig = go.Figure()
            for ticker, group in frame.groupby("ticker"):
                fig.add_scatter(x=group["date"], y=group["close"], mode="lines", name=f"{ticker} close")
                forecast = result.forecasts[result.forecasts["ticker"] == ticker].iloc[0]
                fig.add_scatter(x=[forecast["as_of_date"]], y=[forecast["predicted_close"]], mode="markers", name=f"{ticker} forecast")
            fig.update_layout(hovermode="x unified", yaxis_title="Close", template="plotly_white")
            return fig
        @render.data_frame
        def forecast_table():
            result = current_result()
            return result.forecasts if result is not None else empty_frame("Run an analysis to view forecasts.")
    with ui.nav_panel("Evaluation"):
        @render.data_frame
        def evaluation_metrics():
            result = current_result()
            if result is None: return empty_frame("Run an analysis to view evaluation metrics.")
            rows = [{"strategy": n, **m} for n, m in result.backtest_metrics.items()]
            return pd.DataFrame(rows) if rows else empty_frame("Backtest unavailable for this range.")

ui.include_css(Path(__file__).parent / "styles.css")
