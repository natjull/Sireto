"""Export helpers for Pipe V6 (task 9.1)."""

from __future__ import annotations

from pathlib import Path
import csv
import logging
from typing import Iterable

import pandas as pd

from .config import PipelineConfig
from .crm_loader import load_crm

LOGGER = logging.getLogger(__name__)


EXPORT_COLUMNS: list[str] = [
    # CRM columns first (7)
    "crm_id",
    "crm_name",
    "street_number",
    "street_name",
    "postcode",
    "city",
    "insee_code",
    # Pipeline result columns (10)
    "status",
    "chosen_siret",
    "confidence",
    "crm_category",
    "normalized_name",
    "normalized_address",
    "reason",
    "sources",
    "candidate_count_total",
    "candidate_count_used",
]


def _normalize_sources(values: Iterable[str] | float | int | None) -> str:
    """Return pipe-delimited, alphabetically sorted sources string.

    ``df_results`` stores sources as a Python list. For robustness we accept any
    falsy / non-iterable input and fall back to the empty string.
    """

    if isinstance(values, (list, tuple, set)):
        if not values:
            return ""
        return "|".join(sorted(str(item) for item in values if item))
    return ""


def export_results(
    df_results: pd.DataFrame,
    config: PipelineConfig,
    output_path: Path | None = None,
) -> None:
    """Merge CRM data with pipeline results and export to CSV or JSON.

    The function reloads the CRM from ``config.crm_path`` to keep ``run_pipeline``
    unchanged. All CRM rows are preserved (left join); missing pipeline results
    yield NaN in the result columns. The output format is inferred from the
    suffix of ``output_path`` (``.csv`` or ``.json``). Any other suffix raises a
    ``ValueError``.
    """

    log = LOGGER
    target = Path(output_path) if output_path is not None else config.output_path

    if target.suffix.lower() not in {".csv", ".json"}:
        raise ValueError("output_path must end with .csv or .json")

    # Reload CRM and merge on crm_id (all CRM rows kept).
    df_crm = load_crm(config.crm_path)

    df_export = df_crm.merge(df_results, on="crm_id", how="left", suffixes=("", "_res"))

    # Warn if some pipeline rows could not be merged (crm_id missing in CRM).
    missing_results = set(df_results.get("crm_id", [])) - set(df_crm["crm_id"])
    if missing_results:
        log.warning("Export: %d pipeline rows not found in CRM (crm_id mismatch)", len(missing_results))

    # Ensure expected columns exist even if merge produced NaN
    for col in EXPORT_COLUMNS:
        if col not in df_export.columns:
            df_export[col] = pd.NA

    # Normalize sources column and confidence precision.
    df_export["sources"] = df_export["sources"].apply(_normalize_sources)
    if "confidence" in df_export:
        df_export["confidence"] = pd.to_numeric(
            df_export["confidence"], errors="coerce"
        ).round(4)

    # Keep only ordered export columns.
    df_export = df_export[EXPORT_COLUMNS]

    if target.suffix.lower() == ".csv":
        df_export.to_csv(
            target,
            sep=";",
            encoding="utf-8",
            index=False,
            quoting=csv.QUOTE_MINIMAL,
        )
    else:  # .json
        df_export.to_json(
            target,
            orient="records",
            force_ascii=False,
            indent=2,
        )

    log.info("Exported results to %s", target)


__all__ = ["export_results", "EXPORT_COLUMNS"]
