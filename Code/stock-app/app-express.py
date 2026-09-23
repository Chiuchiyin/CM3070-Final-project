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
def analysis():
    value = input.as_of()
    return service.analyse(tickers=input.tickers(), as_of_date=value if value else None, risk_profile=input.risk_profile())

with ui.layout_column_wrap(fill=False):
    with ui.value_box():
        "Latest close"
        @render.text
        def latest_close():
            result = analysis()
            return str(result.market_data.sort_values("date")["date"].iloc[-1].date())
    with ui.value_box():
        "Expected return"
        @render.text
        def expected_return():
            return f"{analysis().forecasts['predicted_return'].mean():+.2%}"
    with ui.value_box():
        "Cash allocation"
        @render.text
        def cash_allocation():
            return f"{analysis().allocations['cash_weight'].iloc[0]:.1%}"

with ui.navset_tab():
    with ui.nav_panel("Summary"):
        @render.text
        def explanation(): return analysis().explanation
        @render.data_frame
        def warnings():
            rows = analysis().warnings or ["No data-quality warnings."]
            return pd.DataFrame({"Status": rows})
    with ui.nav_panel("Portfolio"):
        @render.data_frame
        def allocations(): return analysis().allocations
        @render.data_frame
        def allocation_changes(): return analysis().allocation_changes
    with ui.nav_panel("Forecast"):
        @render_plotly
        def forecast_chart():
            result = analysis(); frame = result.market_data.sort_values("date"); fig = go.Figure()
            for ticker, group in frame.groupby("ticker"):
                fig.add_scatter(x=group["date"], y=group["close"], mode="lines", name=f"{ticker} close")
                forecast = result.forecasts[result.forecasts["ticker"] == ticker].iloc[0]
                fig.add_scatter(x=[forecast["as_of_date"]], y=[forecast["predicted_close"]], mode="markers", name=f"{ticker} forecast")
            fig.update_layout(hovermode="x unified", yaxis_title="Close", template="plotly_white")
            return fig
        @render.data_frame
        def forecast_table(): return analysis().forecasts
    with ui.nav_panel("Evaluation"):
        @render.data_frame
        def evaluation_metrics():
            result = analysis(); rows = [{"strategy": n, **m} for n, m in result.backtest_metrics.items()]
            return pd.DataFrame(rows) if rows else pd.DataFrame({"status": ["Backtest unavailable"]})

ui.include_css(Path(__file__).parent / "styles.css")
