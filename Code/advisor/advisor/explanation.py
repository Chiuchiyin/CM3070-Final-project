"""Grounded explanation context, deterministic fallback, and optional Qwen."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
import json, re
from typing import Any, Callable
import pandas as pd

@dataclass(frozen=True)
class ExplanationContext:
    as_of_date: str
    model_version: str
    dataset_version: str
    risk_profile: str
    forecasts: list[dict[str, Any]]
    allocations: list[dict[str, Any]]
    cash_weight: float
    warnings: list[str]
    backtest_metrics: dict[str, dict[str, float]]
    def as_dict(self): return asdict(self)
    def prompt(self):
        facts=json.dumps(self.as_dict(), sort_keys=True, default=str)
        return ("Summarize this educational portfolio analysis using only the supplied JSON. "
                "Distinguish historical simulation from next-period estimates, disclose warnings, "
                "and do not give personalized financial advice. Return one concise paragraph.\n"
                f"FACTS_JSON={facts}")

def build_explanation_context(forecasts, allocations, *, risk_profile='moderate', warnings=None,
                              backtest_metrics=None, model_version=None, dataset_version='runtime'):
    f=forecasts.copy(); a=allocations.copy()
    for frame in (f,a):
        for col in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[col]): frame[col]=frame[col].dt.strftime('%Y-%m-%d')
    return ExplanationContext(
        as_of_date=pd.Timestamp(forecasts['as_of_date'].max()).strftime('%Y-%m-%d'),
        model_version=model_version or str(forecasts['model_version'].iloc[0]),
        dataset_version=dataset_version, risk_profile=risk_profile,
        forecasts=f.to_dict(orient='records'), allocations=a.to_dict(orient='records'),
        cash_weight=float(allocations['cash_weight'].iloc[0]) if 'cash_weight' in allocations else 0.0,
        warnings=list(warnings or []), backtest_metrics=backtest_metrics or {})

def _numeric_values(value):
    if isinstance(value, bool): return []
    if isinstance(value, (int,float)): return [float(value)]
    if isinstance(value, dict): return sum((_numeric_values(v) for v in value.values()), [])
    if isinstance(value, (list,tuple)): return sum((_numeric_values(v) for v in value), [])
    return []

def numeric_claims_are_grounded(text, context):
    allowed=_numeric_values(context.as_dict())
    scrubbed = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)
    for token, percent in re.findall(r'(?<![A-Za-z])(-?\d+(?:\.\d+)?)(\s*%)?', scrubbed):
        number=float(token)
        if len(token.lstrip('-')) == 4 and number.is_integer(): continue
        candidate=number/100.0 if percent else number
        if not any(abs(candidate-item) <= max(1e-6, abs(item)*1e-4) for item in allowed): return False
    return True

def advisor_facts_tool(context): return context.as_dict()

def make_smolagents_facts_tool(context):
    try: from smolagents import tool
    except ImportError as exc: raise RuntimeError('Smolagents is required for the facts tool') from exc
    @tool
    def get_advisor_facts() -> dict[str, Any]:
        """Return validated advisor facts only."""
        return advisor_facts_tool(context)
    return get_advisor_facts

def template_explanation(forecasts, allocations, *, risk_profile='moderate', warnings=None, backtest_metrics=None):
    symbols=', '.join(forecasts['ticker'].tolist())
    allocation='; '.join(f'{row.ticker}: {row.weight:.1%}' for row in allocations.itertuples())
    date=pd.Timestamp(forecasts['as_of_date'].max()).date()
    names=sorted(forecasts['model_name'].unique()); model='last-close baseline' if names == ['last_close'] else ', '.join(names)
    summary='; '.join(f'{row.ticker}: {row.predicted_close:.2f} ({row.predicted_return:+.2%})' for row in forecasts.itertuples())
    cash=float(allocations['cash_weight'].iloc[0]) if 'cash_weight' in allocations else 0.0
    metric=''
    if backtest_metrics and 'equal_weight' in backtest_metrics and backtest_metrics['equal_weight'].get('cumulative_return') is not None:
        metric=f" The equal-weight historical simulation returned {backtest_metrics['equal_weight']['cumulative_return']:+.2%}."
    warning=f" Warnings: {'; '.join(warnings)}." if warnings else ''
    return (f'As of {date}, the {model} analysed {symbols}. Its next-close estimates are {summary}. '
            f'For the {risk_profile} profile, the illustrative allocation is {allocation} with {cash:.1%} held as cash.'
            f'{metric}{warning} This is an educational historical-data demonstration, not personalized financial advice.')

class QwenExplainer:
    """Optional bounded generator with deterministic fallback on failure."""
    def __init__(self, generator=None, *, model_loader=None, timeout_seconds=20.0):
        self.generator=generator; self.model_loader=model_loader; self.timeout_seconds=timeout_seconds
    def _get_generator(self):
        if self.generator is not None: return self.generator
        if self.model_loader is None: raise RuntimeError('No Qwen model loader configured')
        self.generator=self.model_loader(); return self.generator
    def __call__(self, forecasts, allocations, **kwargs):
        context=build_explanation_context(forecasts, allocations, **kwargs)
        fallback=template_explanation(forecasts, allocations, **kwargs)
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                generated=str(executor.submit(self._get_generator(), context.prompt()).result(timeout=self.timeout_seconds)).strip()
            return generated if generated and numeric_claims_are_grounded(generated, context) else fallback
        except (Exception, FutureTimeout):
            return fallback
