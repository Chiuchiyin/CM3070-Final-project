"""Market-data loading, normalization, validation, and local caching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


@dataclass
class YahooMarketDataProvider:
    """Download daily data and optionally cache the raw response locally."""

    cache_dir: Path

    def load(self, tickers: Iterable[str], start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is required for live downloads; use the CSV provider offline") from exc
        ticker_list = [ticker.upper() for ticker in tickers]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"market_{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}_{'-'.join(ticker_list)}.csv"
        cache_path = self.cache_dir / cache_name
        if cache_path.exists():
            return CsvMarketDataProvider(cache_path).load(ticker_list, start, end)
        downloaded = yf.download(ticker_list, start=str(pd.Timestamp(start).date()),
                                 end=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
                                 auto_adjust=False, group_by="column", progress=False)
        if downloaded.empty:
            raise DataValidationError("Yahoo Finance returned no data")
        if isinstance(downloaded.columns, pd.MultiIndex):
            rows = []
            for ticker in ticker_list:
                if ticker not in downloaded.columns.get_level_values(-1):
                    continue
                part = downloaded.xs(ticker, axis=1, level=-1).reset_index()
                part["ticker"] = ticker
                rows.append(part)
            if not rows:
                raise DataValidationError("Yahoo Finance returned none of the requested tickers")
            downloaded = pd.concat(rows, ignore_index=True)
        else:
            downloaded = downloaded.reset_index()
            downloaded["ticker"] = ticker_list[0]
        normalized = validate_market_data(downloaded, ticker_list)
        normalized.to_csv(cache_path, index=False)
        return normalized
