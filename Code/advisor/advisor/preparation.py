"""Reproducible preparation of validated market-data artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import MVPConfig
from .data import validate_market_data


def market_data_digest(market_data: pd.DataFrame) -> str:
    """Hash canonical CSV content so the prepared dataset is identifiable."""
    canonical = validate_market_data(market_data).copy()
    canonical["date"] = canonical["date"].dt.strftime("%Y-%m-%d")
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_metadata(market_data: pd.DataFrame, config: MVPConfig, source: str) -> dict[str, object]:
    """Build metadata that identifies data content and its preparation settings."""
    canonical = validate_market_data(market_data, config.tickers)
    return {
        "artifact_schema_version": 1,
        "source": source,
        "data_sha256": market_data_digest(canonical),
        "rows": int(len(canonical)),
        "tickers": sorted(canonical["ticker"].unique().tolist()),
        "date_start": canonical["date"].min().strftime("%Y-%m-%d"),
        "date_end": canonical["date"].max().strftime("%Y-%m-%d"),
        "config": config.metadata(),
    }


def write_prepared_dataset(
    market_data: pd.DataFrame,
    config: MVPConfig,
    source: str,
    output_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    """Validate and write a canonical CSV plus deterministic JSON metadata."""
    canonical = validate_market_data(market_data, config.tickers)
    output = Path(output_path) if output_path is not None else config.prepared_data_path
    metadata_output = Path(metadata_path) if metadata_path is not None else config.metadata_path
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(output, index=False, lineterminator="\n")
    metadata = dataset_metadata(canonical, config, source)
    metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, metadata_output, metadata
