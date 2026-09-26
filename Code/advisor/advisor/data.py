"""Market-data loading, normalization, validation, and local caching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import time

import pandas as pd

REQUIRED_COLUMNS = ("date", "ticker", "open", "high", "low", "close", "volume")


class DataValidationError(ValueError):
    """Raised when market data cannot safely enter the advisor pipeline."""


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the project's canonical market-data column names."""
    rename = {"tic": "ticker", "Date": "date", "Close": "close", "Open": "open",
              "High": "high", "Low": "low", "Volume": "volume"}
    result = frame.rename(columns=rename).copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in result.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.tz_localize(None)
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result[list(REQUIRED_COLUMNS)]


def validate_market_data(frame: pd.DataFrame, tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Validate and return chronologically sorted market data.

    Missing rows are retained as missing observations rather than converted to zero;
    callers can decide whether a requested range is usable.
    """
    result = normalize_columns(frame)
    if result.empty:
        raise DataValidationError("Market data is empty")
    if result["date"].isna().any():
        raise DataValidationError("Market data contains invalid dates")
    if result["ticker"].isna().any() or (result["ticker"].astype(str).str.strip() == "").any():
        raise DataValidationError("Market data contains an empty ticker")
    key = result["ticker"].astype(str) + "|" + result["date"].astype(str)
    if key.duplicated().any():
        raise DataValidationError("Market data contains duplicate ticker/date rows")
    price_columns = ("open", "high", "low", "close")
    if result[list(price_columns)].isna().any().any() or result["volume"].isna().any():
        raise DataValidationError("Market data contains missing numeric values")
    if (result[list(price_columns)] <= 0).any().any() or (result["volume"] < 0).any():
        raise DataValidationError("Prices must be positive and volume must be non-negative")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any():
        raise DataValidationError("High price is below another OHLC value")
    if (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        raise DataValidationError("Low price is above another OHLC value")
    result["ticker"] = result["ticker"].astype(str).str.upper().str.strip()
    if tickers is not None:
        requested = {ticker.upper() for ticker in tickers}
        result = result[result["ticker"].isin(requested)]
        if result.empty:
            raise DataValidationError("None of the requested tickers are present")
        missing_tickers = requested - set(result["ticker"])
        if missing_tickers:
            raise DataValidationError(f"Missing requested tickers: {', '.join(sorted(missing_tickers))}")
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


@dataclass
class CsvMarketDataProvider:
    """Read a frozen CSV fixture or cache without requiring a network connection."""

    path: Path

    def load(self, tickers: Iterable[str] | None = None,
             start: str | pd.Timestamp | None = None,
             end: str | pd.Timestamp | None = None) -> pd.DataFrame:
        frame = validate_market_data(pd.read_csv(self.path), tickers)
        if start is not None:
            frame = frame[frame["date"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["date"] <= pd.Timestamp(end)]
        if frame.empty:
            raise DataValidationError("No observations match the requested range")
        return frame.reset_index(drop=True)


def _normalize_yahoo_download(downloaded: pd.DataFrame, ticker_list: list[str]) -> pd.DataFrame:
    """Convert either Yahoo MultiIndex column layout into canonical rows."""
    if downloaded.empty:
        raise DataValidationError("Yahoo Finance returned no data")
    downloaded = downloaded.copy()
    if downloaded.index.name is None:
        downloaded.index.name = "date"
    if isinstance(downloaded.columns, pd.MultiIndex):
        rows = []
        for ticker in ticker_list:
            levels = [level for level in range(downloaded.columns.nlevels)
                      if ticker in downloaded.columns.get_level_values(level)]
            if not levels:
                continue
            part = downloaded.xs(ticker, axis=1, level=levels[0], drop_level=True).reset_index()
            if isinstance(part.columns, pd.MultiIndex):
                part.columns = [column[-1] if isinstance(column, tuple) else column
                                for column in part.columns]
            part["ticker"] = ticker
            rows.append(part)
        if not rows:
            raise DataValidationError("Yahoo Finance returned none of the requested tickers")
        return pd.concat(rows, ignore_index=True)
    result = downloaded.reset_index()
    result["ticker"] = ticker_list[0]
    return result


@dataclass
class YahooMarketDataProvider:
    """Download daily data and cache a validated canonical response locally."""

    cache_dir: Path
    retries: int = 3
    retry_delay_seconds: float = 2.0

    def load(
        self,
        tickers: Iterable[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is required for live downloads; use the CSV provider offline") from exc
        ticker_list = [str(ticker).upper().strip() for ticker in tickers]
        if not ticker_list or len(set(ticker_list)) != len(ticker_list):
            raise DataValidationError("Yahoo ticker list must be non-empty and unique")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"market_{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}_{'-'.join(ticker_list)}.csv"
        cache_path = self.cache_dir / cache_name
        if cache_path.exists() and not refresh:
            return CsvMarketDataProvider(cache_path).load(ticker_list, start, end)
        download_kwargs = {
            "start": str(pd.Timestamp(start).date()),
            "end": str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
            "auto_adjust": False,
            "group_by": "ticker",
            "progress": False,
            "threads": False,
        }
        downloaded = None
        last_error = None
        for attempt in range(max(1, self.retries)):
            try:
                downloaded = yf.download(ticker_list, **download_kwargs)
                if not downloaded.empty:
                    break
            except Exception as exc:  # yfinance exposes version-specific exception classes
                last_error = exc
            if attempt + 1 < max(1, self.retries):
                time.sleep(max(0.0, self.retry_delay_seconds) * (attempt + 1))
        if downloaded is None or downloaded.empty:
            detail = f": {last_error}" if last_error is not None else ""
            raise DataValidationError(
                "Yahoo Finance returned no data after retries. "
                "Try again later or use a local source CSV" + detail
            ) from last_error
        downloaded = _normalize_yahoo_download(downloaded, ticker_list)
        normalized = validate_market_data(downloaded, ticker_list)
        missing_tickers = set(ticker_list) - set(normalized["ticker"].unique())
        if missing_tickers:
            raise DataValidationError(
                f"Yahoo Finance returned incomplete data; missing: {', '.join(sorted(missing_tickers))}"
            )
        normalized.to_csv(cache_path, index=False)
        return normalized
