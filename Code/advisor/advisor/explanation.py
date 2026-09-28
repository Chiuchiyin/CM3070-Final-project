"""Grounded explanation context, deterministic fallback, and optional Qwen."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from datetime import datetime
import json, re
from pathlib import Path
from threading import Lock
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
    percent_keys={'predicted_return', 'weight', 'cash_weight', 'cumulative_return',
                  'annual_return', 'annual_volatility', 'maximum_drawdown',
                  'average_period_turnover', 'net_return', 'gross_return'}
    percent_allowed=[]
    def collect_percent_values(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in percent_keys and isinstance(item, (int, float)) and not isinstance(item, bool):
                    percent_allowed.append(float(item))
                collect_percent_values(item)
        elif isinstance(value, list):
            for item in value: collect_percent_values(item)
    collect_percent_values(context.as_dict())
    allowed_dates={context.as_of_date}
    for record in context.forecasts + context.allocations:
        for value in record.values():
            if isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
                allowed_dates.add(value)
    scrubbed=text
    for match in re.finditer(r'\b\d{4}-\d{2}-\d{2}\b', text):
        if match.group() not in allowed_dates: return False
    scrubbed=re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', scrubbed)
    month_date=r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
    for match in re.finditer(month_date, scrubbed, flags=re.IGNORECASE):
        try:
            parsed=datetime.strptime(match.group().replace(',', ''), '%B %d %Y').date().isoformat()
        except ValueError:
            return False
        if parsed not in allowed_dates: return False
    scrubbed=re.sub(month_date, '', scrubbed, flags=re.IGNORECASE)
    for token, percent in re.findall(r'(?<![A-Za-z])(-?\d+(?:\.\d+)?)(\s*%)?', scrubbed):
        number=float(token)
        if re.fullmatch(r'\d{4}', token):
            if any(date_value.startswith(token) for date_value in allowed_dates): continue
            return False
        candidate=number/100.0 if percent else number
        decimals=len(token.split('.', 1)[1]) if '.' in token else 0
        rounding_tolerance=0.5 * 10**(-decimals) / (100.0 if percent else 1.0)
        candidates=percent_allowed if percent else allowed
        if not any(abs(candidate-item) <= rounding_tolerance + 1e-9 for item in candidates): return False
    return True


def _looks_like_structured_output(text):
    """Return True for JSON/tool payloads that must not be shown as prose."""
    candidate = str(text or '').strip()
    if candidate.startswith('```') and candidate.endswith('```'):
        candidate = candidate.strip('`').strip()
        if candidate.lower().startswith('json'):
            candidate = candidate[4:].strip()
    if not ((candidate.startswith('{') and candidate.endswith('}')) or
            (candidate.startswith('[') and candidate.endswith(']'))):
        return False
    try:
        parsed = json.loads(candidate)
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(parsed, (dict, list))

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


def template_chat_response(question, forecasts, allocations, *, risk_profile='moderate',
                           warnings=None, backtest_metrics=None):
    """Answer common advisor questions using only validated service results."""
    question = str(question or '').strip()
    if not question:
        return 'Ask about allocations, forecasts, historical performance, risk, or the data date.'

    lowered = question.lower()
    as_of = pd.Timestamp(forecasts['as_of_date'].max()).date()
    cash = float(allocations['cash_weight'].iloc[0]) if 'cash_weight' in allocations else 0.0
    allocation_text = ', '.join(
        f'{row.ticker} {row.weight:.1%}' for row in allocations.itertuples()
    )
    forecast_text = '; '.join(
        f'{row.ticker}: {row.predicted_close:.2f} ({row.predicted_return:+.2%})'
        for row in forecasts.itertuples()
    )
    historical = (backtest_metrics or {}).get('equal_weight', {})

    if any(term in lowered for term in ('allocation', 'weight', 'portfolio', 'hold')):
        return (
            f'As of {as_of}, the illustrative {risk_profile} allocation is {allocation_text}, '
            f'with {cash:.1%} held as cash. This is a model output for education, not personalized advice.'
        )
    if any(term in lowered for term in ('forecast', 'predict', 'price', 'return', 'rise', 'fall')):
        return (
            f'As of {as_of}, the next-close estimates are {forecast_text}. '
            'These are model estimates rather than guaranteed returns.'
        )
    if any(term in lowered for term in ('performance', 'backtest', 'evaluation', 'historical')):
        if historical and historical.get('cumulative_return') is not None:
            return (
                f'The equal-weight historical simulation returned '
                f"{float(historical['cumulative_return']):+.2%}. It is historical evidence, "
                'not a forecast of future performance.'
            )
        return 'No historical backtest is available for the selected data range.'
    if any(term in lowered for term in ('risk', 'cash', 'safe', 'conservative')):
        warning_text = f" Warnings: {'; '.join(warnings)}." if warnings else ''
        return (
            f'The selected risk profile is {risk_profile} with {cash:.1%} held as cash.'
            f'{warning_text} The allocation is simulated and educational.'
        )
    if any(term in lowered for term in ('data', 'date', 'latest', 'real-time', 'realtime')):
        warning_text = f" {'; '.join(warnings)}" if warnings else ''
        return (
            f'The analysis uses the latest available historical row through {as_of}; '
            f'it is not a real-time price feed.{warning_text}'
        )
    return (
        'I can explain the current allocation, forecast estimates, historical '
        'backtest, risk profile, or data date. Ask one of those questions and '
        'I will answer from the current analysis.'
    )

class QwenExplainer:
    """Optional bounded generator with deterministic fallback on failure."""
    def __init__(self, generator=None, *, model_loader=None, timeout_seconds=20.0):
        self.generator=generator; self.model_loader=model_loader; self.timeout_seconds=timeout_seconds
        self._generator_lock=Lock()
    def _get_generator(self):
        if self.generator is not None: return self.generator
        if self.model_loader is None: raise RuntimeError('No Qwen model loader configured')
        with self._generator_lock:
            if self.generator is None:
                self.generator=self.model_loader()
        return self.generator
    def __call__(self, forecasts, allocations, **kwargs):
        return self.explain_with_status(forecasts, allocations, **kwargs)[0]

    def explain_with_status(self, forecasts, allocations, **kwargs):
        """Return the explanation and its source without shared mutable status."""
        context=build_explanation_context(forecasts, allocations, **kwargs)
        fallback=template_explanation(
            forecasts, allocations,
            risk_profile=kwargs.get('risk_profile', 'moderate'),
            warnings=kwargs.get('warnings'),
            backtest_metrics=kwargs.get('backtest_metrics'),
        )
        try:
            executor=ThreadPoolExecutor(max_workers=1)
            try:
                def generate():
                    generator=self._get_generator()
                    if hasattr(generator, 'generate_from_context'):
                        return generator.generate_from_context(context)
                    return generator(context.prompt())
                future=executor.submit(generate)
                generated=str(future.result(timeout=self.timeout_seconds)).strip()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            if (generated and not _looks_like_structured_output(generated)
                    and numeric_claims_are_grounded(generated, context)):
                return generated, 'qwen'
            reason = 'Qwen returned structured data' if _looks_like_structured_output(generated) else 'unsupported or empty model output'
            return fallback, f'template: {reason}'
        except FutureTimeout:
            return fallback, 'template: Qwen timed out'
        except Exception as exc:
            return fallback, f'template: Qwen unavailable ({type(exc).__name__})'

    def answer_question(self, question, forecasts, allocations, **kwargs):
        """Answer a user question with grounded facts and a safe fallback."""
        context = build_explanation_context(forecasts, allocations, **kwargs)
        fallback = template_chat_response(
            question, forecasts, allocations,
            risk_profile=kwargs.get('risk_profile', 'moderate'),
            warnings=kwargs.get('warnings'),
            backtest_metrics=kwargs.get('backtest_metrics'),
        )
        try:
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                def generate():
                    generator = self._get_generator()
                    if hasattr(generator, 'generate_answer'):
                        return generator.generate_answer(context, question)
                    prompt = (
                        context.prompt().replace(
                            'Return one concise paragraph.',
                            'Answer the user question in one concise paragraph. '
                            f'USER_QUESTION={question!r}',
                        )
                    )
                    return generator(prompt)
                future = executor.submit(generate)
                generated = str(future.result(timeout=self.timeout_seconds)).strip()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            if (generated and not _looks_like_structured_output(generated)
                    and numeric_claims_are_grounded(generated, context)):
                return generated, 'qwen'
            reason = 'Qwen returned structured data' if _looks_like_structured_output(generated) else 'unsupported or empty model output'
            return fallback, f'template: {reason}'
        except FutureTimeout:
            return fallback, 'template: Qwen timed out'
        except Exception as exc:
            return fallback, f'template: Qwen unavailable ({type(exc).__name__})'


class SmolagentsQwenGenerator:
    """Local Qwen model supplied only by the read-only advisor facts tool."""

    def __init__(self, model_id='Qwen/Qwen3-1.7B', *, max_new_tokens=256):
        from huggingface_hub import snapshot_download
        from smolagents import TransformersModel

        # Loading by snapshot path prevents an app request from downloading a
        # large model unexpectedly. Use the cached notebook Qwen model.
        local_model=Path(snapshot_download(repo_id=model_id, local_files_only=True))
        self.model=TransformersModel(
            model_id=str(local_model),
            max_new_tokens=max_new_tokens, trust_remote_code=False,
        )
        self.max_new_tokens=max_new_tokens
        self._run_lock=Lock()

    def generate_from_context(self, context):
        facts_tool=make_smolagents_facts_tool(context)
        facts=facts_tool()
        display_facts={
            'as_of_date': facts['as_of_date'],
            'risk_profile': facts['risk_profile'],
            'model_version': facts['model_version'],
            'dataset_version': facts['dataset_version'],
            'forecasts': [
                {'ticker': row['ticker'], 'predicted_close': row['predicted_close'],
                 'predicted_return': f"{float(row['predicted_return']):.2%}"}
                for row in facts['forecasts']
            ],
            'allocations': [
                {'ticker': row['ticker'], 'weight': f"{float(row['weight']):.1%}"}
                for row in facts['allocations']
            ],
            'cash_weight': f"{facts['cash_weight']:.1%}",
            'warnings': facts['warnings'],
        }
        historical=facts['backtest_metrics'].get('equal_weight', {})
        if 'cumulative_return' in historical:
            display_facts['historical_equal_weight_cumulative_return']=f"{historical['cumulative_return']:.2%}"
        task=(
            'You are explaining validated advisor results. /no_think\n'
            'Use only DISPLAY_FACTS_JSON below. Copy displayed values exactly; '
            'do not calculate or invent numbers, dates, tickers, assets, or performance. '
            'Explain next-period estimates separately from the historical '
            'equal-weight simulation, mention warnings, and say this is educational '
            'rather than personalized financial advice. '
            'Return one concise natural-language paragraph only. Never return '
            'JSON, YAML, a dictionary, field names, or a tool payload.\n'
            f'DISPLAY_FACTS_JSON={json.dumps(display_facts, sort_keys=True, default=str)}'
        )
        return self._generate_text(task)

    def generate_answer(self, context, question):
        facts_tool = make_smolagents_facts_tool(context)
        facts = facts_tool()
        display_facts = {
            'as_of_date': facts['as_of_date'],
            'risk_profile': facts['risk_profile'],
            'forecasts': [
                {'ticker': row['ticker'], 'predicted_close': row['predicted_close'],
                 'predicted_return': f"{float(row['predicted_return']):.2%}"}
                for row in facts['forecasts']
            ],
            'allocations': [
                {'ticker': row['ticker'], 'weight': f"{float(row['weight']):.1%}"}
                for row in facts['allocations']
            ],
            'cash_weight': f"{facts['cash_weight']:.1%}",
            'warnings': facts['warnings'],
            'backtest_metrics': facts['backtest_metrics'],
        }
        task = (
            'You are a grounded financial-advisor assistant. /no_think\n'
            'Answer the USER_QUESTION using only DISPLAY_FACTS_JSON. Do not invent '
            'numbers, dates, tickers, assets, or performance. If the facts do not '
            'answer the question, say that clearly. Distinguish estimates from '
            'historical simulations, mention relevant warnings, and do not give '
            'personalized financial advice. Return one concise natural-language '
            'paragraph only. Never return JSON, YAML, a dictionary, field names, '
            'or a tool payload.\n'
            f'USER_QUESTION={question!r}\n'
            f'DISPLAY_FACTS_JSON={json.dumps(display_facts, sort_keys=True, default=str)}'
        )
        return self._generate_text(task)

    def _generate_text(self, task):
        # Qwen3 must be prompted with enable_thinking=False; the installed
        # Smolagents wrapper does not expose that chat-template argument.
        # Its locally loaded tokenizer/model are reused for generation.
        import torch
        with self._run_lock:
            encoded=self.model.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': task}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
                return_tensors='pt', return_dict=True,
            ).to(self.model.model.device)
            with torch.inference_mode():
                output=self.model.model.generate(
                    **encoded, max_new_tokens=self.max_new_tokens,
                    do_sample=False, pad_token_id=self.model.tokenizer.eos_token_id,
                )
            answer=self.model.tokenizer.decode(
                output[0, encoded['input_ids'].shape[1]:], skip_special_tokens=True,
            ).strip()
        if '</think>' in answer:
            answer=answer.split('</think>', 1)[1].strip()
        return answer
