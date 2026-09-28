"""Unified Python Shiny interface for the educational advisor."""
from __future__ import annotations
import os, sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget

ADVISOR_ROOT = Path(__file__).resolve().parents[1] / "advisor"
sys.path.insert(0, str(ADVISOR_ROOT))
from advisor.data import CsvMarketDataProvider, YahooMarketDataProvider, canonical_ticker
from advisor.forecasting import MovingAverageForecaster
from advisor.finrl_adapter import load_approved_finrl_policy
from advisor.explanation import (
    QwenExplainer,
    SmolagentsQwenGenerator,
    template_chat_response,
)
from advisor.service import AdvisorService
from advisor.strategy import ForecastRankedStrategy

MVP_TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "PG"]
ORIGINAL_STOCKS = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.", "META": "Meta Platforms", "V": "Visa Inc.",
    "JNJ": "Johnson & Johnson", "WMT": "Walmart Inc.", "JPM": "JPMorgan Chase & Co.",
    "PG": "The Procter & Gamble Company", "UNH": "UnitedHealth Group Incorporated",
    "DIS": "The Walt Disney Company", "HD": "The Home Depot, Inc.", "NVDA": "NVIDIA Corporation",
    "KO": "The Coca-Cola Company", "NKE": "NIKE, Inc.", "MRK": "Merck & Co., Inc.",
    "CVX": "Chevron Corporation", "CSCO": "Cisco Systems, Inc.", "CRM": "Salesforce",
    "AXP": "American Express", "BA": "Boeing", "CAT": "Caterpillar", "GS": "Goldman Sachs",
    "HON": "Honeywell", "IBM": "IBM", "MCD": "McDonald's", "MMM": "3M",
    "TRV": "Travelers", "VZ": "Verizon", "SHW": "Sherwin-Williams",
}
DEFAULT_TICKERS = MVP_TICKERS.copy()
fixture = ADVISOR_ROOT / "tests" / "fixtures" / "market_data.csv"
tracked_dow = ADVISOR_ROOT.parent / "data" / "2025-06-16_dow30.csv"
data_path = os.getenv("ADVISOR_DATA_PATH")
if data_path or tracked_dow.exists() or fixture.exists():
    fixture_path = Path(data_path) if data_path else (tracked_dow if tracked_dow.exists() else fixture)
    provider = CsvMarketDataProvider(fixture_path)
    raw_frame = pd.read_csv(fixture_path)
    raw_ticker_column = "ticker" if "ticker" in raw_frame.columns else "tic"
    available_tickers = sorted({canonical_ticker(t) for t in raw_frame[raw_ticker_column]})
    latest_available = pd.to_datetime(raw_frame["date"], errors="coerce").max()
    DEMO_END_DATE = latest_available.date()
    DEMO_START_DATE = (latest_available - pd.Timedelta(weeks=26)).date()
    STOCK_CHOICES = {ticker: f"{ticker} — {ORIGINAL_STOCKS.get(ticker, ticker)}" for ticker in available_tickers}
    DEFAULT_TICKERS = [ticker for ticker in MVP_TICKERS if ticker in available_tickers] or available_tickers[:1]
else:
    provider = YahooMarketDataProvider(ADVISOR_ROOT.parent / "data" / "cache")
    STOCK_CHOICES = {ticker: f"{ticker} — {name}" for ticker, name in ORIGINAL_STOCKS.items()}
    DEMO_START_DATE = None
    DEMO_END_DATE = None
production_forecaster = MovingAverageForecaster(window=20)
production_strategy = ForecastRankedStrategy(top_k=3, minimum_predicted_return=0.0)
service = AdvisorService(
    provider,
    forecaster=production_forecaster,
    strategy=production_strategy,
    dataset_version="mvp_runtime",
)
qwen_model_id = os.getenv("ADVISOR_QWEN_MODEL", "Qwen/Qwen3-1.7B")
qwen_service = AdvisorService(
    provider,
    forecaster=MovingAverageForecaster(window=20),
    strategy=ForecastRankedStrategy(top_k=3, minimum_predicted_return=0.0),
    dataset_version="mvp_runtime",
    explainer=QwenExplainer(
        model_loader=lambda: SmolagentsQwenGenerator(model_id=qwen_model_id),
        timeout_seconds=180.0,
    ),
)
FINRL_MODEL = ADVISOR_ROOT / "artifacts" / "models" / "finrl_a2c_legacy.zip"
FINRL_METADATA = ADVISOR_ROOT / "artifacts" / "models" / "finrl_a2c_legacy.metadata.json"
_finrl_policy = None
_finrl_error = None


def get_finrl_service():
    """Load the approved legacy A2C policy only when selected by the user."""
    global _finrl_policy, _finrl_error
    if _finrl_policy is None and _finrl_error is None:
        try:
            _finrl_policy, _ = load_approved_finrl_policy(
                FINRL_MODEL,
                FINRL_METADATA,
                tickers=MVP_TICKERS,
            )
        except Exception as exc:
            _finrl_error = f"FinRL policy unavailable: {exc}"
    if _finrl_policy is None:
        raise RuntimeError(_finrl_error or "FinRL policy unavailable")
    return AdvisorService(
        provider,
        forecaster=MovingAverageForecaster(window=20),
        strategy=_finrl_policy,
        dataset_version="mvp_runtime_finrl_legacy",
    )

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Analysis inputs"),
        ui.input_selectize("tickers", "Stocks", choices=STOCK_CHOICES, selected=DEFAULT_TICKERS, multiple=True),
        ui.input_select("chart_ticker", "Chart ticker", choices=STOCK_CHOICES, selected=DEFAULT_TICKERS[0]),
        ui.input_date_range("history_dates", "Chart dates", start=DEMO_START_DATE, end=DEMO_END_DATE),
        ui.input_date("as_of", "As-of date (blank uses last available)", value=DEMO_END_DATE),
        ui.input_select("risk_profile", "Risk profile", choices={"conservative": "Conservative", "moderate": "Moderate", "growth": "Growth"}, selected="moderate"),
        ui.input_select(
            "allocation_strategy",
            "Allocation strategy",
            choices={
                "forecast_ranked": "Forecast-ranked (default)",
                "finrl_a2c": "FinRL A2C (legacy artifact)",
            },
            selected="forecast_ranked",
        ),
        ui.input_select("explanation_mode", "Explanation", choices={"template": "Instant grounded summary", "qwen": "Qwen + Smolagents (local model)"}, selected="template"),
        ui.input_action_button("analyse", "Analyse", class_="btn-primary"),
        ui.p("Educational decision-support prototype. Simulated allocations only."),
        ui.p("Demo uses the last available historical entry as the latest close. Prices are not real time."),
    ),
    ui.include_css(Path(__file__).with_name("styles.css")),
    ui.layout_columns(
        ui.value_box("Latest close price", ui.output_text("current_price")),
        ui.value_box("Change", ui.output_text("price_change")),
        ui.value_box("Percent change", ui.output_text("price_change_percent")),
        fill=False,
        class_="kpi-grid",
    ),
    ui.layout_columns(
        ui.value_box("Latest close date", ui.output_text("latest_close")),
        ui.value_box("Expected return", ui.output_text("expected_return")),
        ui.value_box("Cash allocation", ui.output_text("cash_allocation")),
        fill=False,
        class_="kpi-grid",
    ),
    ui.layout_columns(
        ui.div(
            ui.navset_tab(
                ui.nav_panel("Market Data", output_widget("market_chart"), ui.output_data_frame("latest_market_data")),
                ui.nav_panel(
                    "Summary",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Advisor explanation"),
                            ui.output_text("explanation_status"),
                            ui.div(ui.output_text_verbatim("explanation"), class_="advisor-explanation"),
                        ),
                        ui.card(
                            ui.card_header("Data quality"),
                            ui.output_data_frame("warnings"),
                        ),
                        col_widths=(8, 4),
                        fill=False,
                        class_="summary-grid",
                    ),
                    ui.card(
                        ui.card_header("Disclosures"),
                        ui.output_data_frame("disclosures"),
                    ),
                ),
                ui.nav_panel("Portfolio", ui.output_data_frame("allocations"), ui.output_data_frame("allocation_changes")),
                ui.nav_panel("Forecast", output_widget("forecast_chart"), ui.output_data_frame("forecast_table")),
                ui.nav_panel(
                    "Evaluation",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Historical strategy comparison"),
                            ui.p(
                                "Metrics are calculated on the available historical window "
                                "with the same assumptions for each strategy. They are not "
                                "a guarantee of future performance."
                            ),
                            ui.output_data_frame("evaluation_metrics"),
                        ),
                        ui.card(
                            ui.card_header("Metric guide"),
                            ui.tags.ul(
                                ui.tags.li("Cumulative return: total simulated growth."),
                                ui.tags.li("Annual return and volatility: annualised performance and variability."),
                                ui.tags.li("Sharpe: return relative to variability."),
                                ui.tags.li("Max drawdown: largest peak-to-trough loss."),
                                ui.tags.li("Final value: simulated value from the fixed starting capital."),
                            ),
                        ),
                        col_widths=(9, 3),
                        fill=False,
                        class_="evaluation-grid",
                    ),
                ),
            ),
            class_="main-content",
        ),
        ui.card(
            ui.card_header("Advisor chatbot"),
            ui.p("Ask about the current allocation, forecast, historical performance, risk, or data date."),
            ui.input_text_area(
                "chat_question",
                "Question",
                placeholder="Why is cash held?",
                rows=3,
            ),
            ui.input_select(
                "chat_mode",
                "Chatbot model",
                choices={
                    "qwen": "Qwen + Smolagents (local model)",
                    "template": "Instant grounded responder",
                },
                selected="qwen",
            ),
            ui.input_action_button("chat_send", "Ask advisor", class_="btn-secondary"),
            ui.output_text("chat_status"),
            ui.div(ui.output_text_verbatim("chat_response"), class_="chat-response"),
            ui.p("Answers are grounded in the current analysis and are for educational use."),
            class_="chat-panel",
        ),
        col_widths=(9, 3),
        fill=False,
        class_="app-content-grid",
    ),
    title="Agentic Financial Advisor", fillable=True,
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc
    @reactive.event(input.analyse)
    def analysis_state():
        try:
            tickers = input.tickers()
            if not tickers:
                return {"result": None, "error": "Select at least one ticker."}
            value = input.as_of()
            if input.allocation_strategy() == "finrl_a2c":
                selected_service = get_finrl_service()
                if input.explanation_mode() == "qwen":
                    selected_service.explainer = qwen_service.explainer
            else:
                selected_service = qwen_service if input.explanation_mode() == "qwen" else service
            result = selected_service.analyse(tickers=tickers, as_of_date=value if value else None, risk_profile=input.risk_profile())
            return {"result": result, "error": None}
        except Exception as exc:
            return {"result": None, "error": str(exc)}

    def current_result():
        return analysis_state()["result"]

    def empty_frame(message):
        return pd.DataFrame({"Status": [message]})

    def selected_chart_frame(result):
        ticker = input.chart_ticker()
        frame = result.market_data[result.market_data["ticker"] == ticker].sort_values("date").copy()
        if frame.empty:
            ticker = sorted(result.market_data["ticker"].unique())[0]
            frame = result.market_data[result.market_data["ticker"] == ticker].sort_values("date").copy()
        dates = input.history_dates()
        if dates and len(dates) == 2:
            if dates[0]:
                frame = frame[frame["date"] >= pd.Timestamp(dates[0])]
            if dates[1]:
                frame = frame[frame["date"] <= pd.Timestamp(dates[1])]
        return frame, ticker

    def selected_ticker_history(result):
        ticker = input.chart_ticker()
        frame = result.market_data[result.market_data["ticker"] == ticker].sort_values("date")
        if frame.empty:
            ticker = sorted(result.market_data["ticker"].unique())[0]
            frame = result.market_data[result.market_data["ticker"] == ticker].sort_values("date")
        return frame, ticker

    @reactive.effect
    def update_chart_ticker_choices():
        tickers = input.tickers() or []
        if not tickers:
            return
        selected = input.chart_ticker()
        choices = {ticker: STOCK_CHOICES.get(ticker, ticker) for ticker in tickers}
        ui.update_select("chart_ticker", choices=choices, selected=selected if selected in tickers else tickers[0])

    @output
    @render.text
    def current_price():
        result = current_result()
        if result is None:
            return "Unavailable"
        frame, ticker = selected_ticker_history(result)
        return "Unavailable" if frame.empty else f"${frame['close'].iloc[-1]:.2f}"

    @output
    @render.text
    def price_change():
        result = current_result()
        if result is None:
            return "Unavailable"
        frame, _ = selected_ticker_history(result)
        close = frame["close"]
        return "Unavailable" if len(close) < 2 else f"${close.iloc[-1] - close.iloc[-2]:+.2f}"

    @output
    @render.text
    def price_change_percent():
        result = current_result()
        if result is None:
            return "Unavailable"
        frame, _ = selected_ticker_history(result)
        close = frame["close"]
        if len(close) < 2 or close.iloc[-2] == 0:
            return "Unavailable"
        return f"{(close.iloc[-1] / close.iloc[-2] - 1.0):+.2%}"

    @output
    @render.text
    def latest_close():
        result = current_result()
        return "Unavailable" if result is None else str(result.market_data.sort_values("date")["date"].iloc[-1].date())

    @output
    @render.text
    def expected_return():
        result = current_result()
        return "Unavailable" if result is None else f"{result.forecasts['predicted_return'].mean():+.2%}"

    @output
    @render.text
    def cash_allocation():
        result = current_result()
        return "Unavailable" if result is None else f"{result.allocations['cash_weight'].iloc[0]:.1%}"

    @output
    @render.text
    def explanation_status():
        result = current_result()
        if result is None:
            return "Explanation source: available after analysis"
        if result.explanation_source == "qwen":
            return f"Explanation source: Qwen + Smolagents ({qwen_model_id})"
        if result.explanation_source.startswith("template:"):
            return f"Explanation source: deterministic fallback ({result.explanation_source.removeprefix('template: ')})"
        return "Explanation source: deterministic template"

    @output
    @render.text
    def explanation():
        state = analysis_state()
        return state["error"] or "Press Analyse to begin." if state["result"] is None else state["result"].explanation

    @reactive.calc
    @reactive.event(input.chat_send)
    def chat_state():
        result = current_result()
        question = (input.chat_question() or "").strip()
        if result is None:
            return {"response": "Run Analyse first so I have a current advisor result to discuss.", "source": "unavailable"}
        if not question:
            return {"response": "Ask about the allocation, forecast, historical performance, risk, or data date.", "source": "template"}
        kwargs = {
            "risk_profile": result.risk_profile,
            "warnings": result.warnings,
            "backtest_metrics": result.backtest_metrics,
            "model_version": result.model_version,
            "dataset_version": result.dataset_version,
        }
        if input.chat_mode() == "qwen":
            response, source = qwen_service.explainer.answer_question(
                question, result.forecasts, result.allocations, **kwargs
            )
        else:
            response = template_chat_response(
                question,
                result.forecasts,
                result.allocations,
                risk_profile=result.risk_profile,
                warnings=result.warnings,
                backtest_metrics=result.backtest_metrics,
            )
            source = "template"
        return {"response": response, "source": source}

    @output
    @render.text
    def chat_status():
        state = chat_state()
        if state["source"] == "qwen":
            return f"Chat source: Qwen + Smolagents ({qwen_model_id})"
        if state["source"].startswith("template:"):
            return f"Chat source: grounded fallback ({state['source'].removeprefix('template: ')})"
        if state["source"] == "unavailable":
            return "Chat status: analyse data first"
        return "Chat source: grounded advisor facts"

    @output
    @render.text
    def chat_response():
        return chat_state()["response"]

    @output
    @render.data_frame
    def warnings():
        state = analysis_state()
        if state["error"]:
            return pd.DataFrame({"Status": [f"Analysis unavailable: {state['error']}"]})
        return pd.DataFrame({"Status": state["result"].warnings or ["Analysis ready. No data-quality warnings."]})

    @output
    @render.data_frame
    def disclosures():
        result = current_result()
        if result is None:
            return empty_frame("Model and data disclosures will appear after analysis.")
        strategy_name = result.allocations["strategy_name"].iloc[0] if "strategy_name" in result.allocations else "Unavailable"
        return pd.DataFrame({"Field": ["Use", "Data date", "Model version", "Dataset version", "Risk profile", "Allocation strategy"], "Value": ["Educational decision support; simulated allocations only", str(result.as_of_date.date()) if result.as_of_date is not None else "Unavailable", result.model_version, result.dataset_version, result.risk_profile, strategy_name]})

    @output
    @render.data_frame
    def allocations():
        result = current_result()
        return result.allocations if result is not None else empty_frame("Run an analysis to view allocations.")

    @output
    @render.data_frame
    def allocation_changes():
        result = current_result()
        return result.allocation_changes if result is not None else empty_frame("Run an analysis to view allocation changes.")

    @output
    @render_widget
    def market_chart():
        result = current_result()
        if result is None:
            return go.Figure().update_layout(title="Market data unavailable until analysis succeeds")
        frame, ticker = selected_chart_frame(result)
        if frame.empty:
            return go.Figure().update_layout(title="No market data matches the selected chart dates")
        frame["sma20"] = frame["close"].rolling(window=20, min_periods=1).mean()
        fig = go.Figure()
        fig.add_candlestick(
            x=frame["date"], open=frame["open"], high=frame["high"],
            low=frame["low"], close=frame["close"], name=ticker,
            increasing_line_color="#44bb70", decreasing_line_color="#040548",
        )
        fig.add_scatter(x=frame["date"], y=frame["sma20"], mode="lines", name="SMA (20)", line={"color": "orange", "dash": "dash"})
        fig.update_layout(
            title=f"{ticker} price history",
            hovermode="x unified", template="plotly_white",
            yaxis_title="Price", xaxis_rangeslider_visible=False,
            legend={"orientation": "h", "yanchor": "top", "y": 1.08, "xanchor": "right", "x": 1},
        )
        return fig

    @output
    @render.data_frame
    def latest_market_data():
        result = current_result()
        if result is None:
            return empty_frame("Run an analysis to view the latest OHLCV data.")
        frame, _ = selected_chart_frame(result)
        if frame.empty:
            return empty_frame("No market data matches the selected chart dates.")
        return frame.tail(1)[["date", "ticker", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    @output
    @render_widget
    def forecast_chart():
        result = current_result()
        if result is None:
            return go.Figure().update_layout(title="Forecast unavailable until analysis succeeds")
        frame = result.market_data.sort_values("date")
        fig = go.Figure()
        for ticker, group in frame.groupby("ticker"):
            fig.add_scatter(x=group["date"], y=group["close"], mode="lines", name=f"{ticker} close")
            forecast = result.forecasts[result.forecasts["ticker"] == ticker].iloc[0]
            fig.add_scatter(x=[forecast["as_of_date"]], y=[forecast["predicted_close"]], mode="markers", name=f"{ticker} forecast")
        fig.update_layout(hovermode="x unified", yaxis_title="Close", template="plotly_white")
        return fig

    @output
    @render.data_frame
    def forecast_table():
        result = current_result()
        return result.forecasts if result is not None else empty_frame("Run an analysis to view forecasts.")

    @output
    @render.data_frame
    def evaluation_metrics():
        result = current_result()
        if result is None:
            return empty_frame("Run an analysis to view evaluation metrics.")
        if not result.backtest_metrics:
            return empty_frame("Backtest unavailable for this range.")

        def percent(value):
            return "—" if pd.isna(value) else f"{float(value):+.1%}"

        def number(value):
            return "—" if pd.isna(value) else f"{float(value):.2f}"

        def currency(value):
            return "—" if pd.isna(value) else f"${float(value):,.0f}"

        labels = {
            "equal_weight": "Equal weight",
            "buy_and_hold": "Buy and hold",
            "forecast_ranked_moving_average": "Forecast ranked",
            "forecast_ranked_moving_average_return": "Forecast ranked",
            "market_index": "Market index",
        }
        rows = []
        for name, metrics in result.backtest_metrics.items():
            rows.append(
                {
                    "Strategy": labels.get(name, name.replace("_", " ").title()),
                    "Cumulative return": percent(metrics.get("cumulative_return")),
                    "Annual return": percent(metrics.get("annual_return")),
                    "Volatility": percent(metrics.get("annual_volatility")),
                    "Sharpe": number(metrics.get("sharpe_ratio")),
                    "Max drawdown": percent(metrics.get("maximum_drawdown")),
                    "Final value": currency(metrics.get("final_value")),
                }
            )
        return pd.DataFrame(rows)

app = App(app_ui, server)
