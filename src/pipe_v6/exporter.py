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
    "chosen_name",
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


# ---------------------------------------------------------------------------
# Stats helpers (task 9.2)
# ---------------------------------------------------------------------------


def _percentages(counts: dict[str, int], total: int) -> dict[str, float]:
    if total == 0:
        return {key: 0.0 for key in counts}
    return {key: round(value / total, 4) for key, value in counts.items()}


def compute_stats(df_results: pd.DataFrame) -> dict:
    """Compute aggregate metrics over pipeline results.

    - Includes all rows (MATCH / REVIEW / NO_MATCH, erreurs comprises).
    - Percentages are floats in [0,1], rounded to 4 decimals.
    - Candidate averages use zeros for missing values.
    - Confidence stats are computed on MATCH rows only; 0.0 if none.
    """

    total_rows = int(len(df_results))

    # Status counts / percentages
    status_keys = ["MATCH", "REVIEW", "NO_MATCH"]
    status_counts = {key: 0 for key in status_keys}
    value_counts = df_results.get("status", pd.Series(dtype=str)).value_counts()
    for key in status_keys:
        status_counts[key] = int(value_counts.get(key, 0))
    status_percentages = _percentages(status_counts, total_rows)

    # CRM category counts / percentages
    category_keys = ["PUBLIC", "PRIVE", "EQUIPEMENT_URBAIN", "INCONNU"]
    category_counts = {key: 0 for key in category_keys}
    cat_counts = df_results.get("crm_category", pd.Series(dtype=str)).value_counts()
    for key in category_keys:
        category_counts[key] = int(cat_counts.get(key, 0))
    category_percentages = _percentages(category_counts, total_rows)

    # Candidate averages (use zeros when missing)
    if total_rows == 0:
        avg_total = 0.0
        avg_used = 0.0
    else:
        avg_total = float(
            df_results.get("candidate_count_total", pd.Series(dtype=float))
            .fillna(0)
            .mean()
            or 0.0
        )
        avg_used = float(
            df_results.get("candidate_count_used", pd.Series(dtype=float))
            .fillna(0)
            .mean()
            or 0.0
        )

    # Confidence stats on MATCH rows only
    match_mask = df_results.get("status", pd.Series(dtype=str)) == "MATCH"
    match_conf = pd.to_numeric(
        df_results.get("confidence", pd.Series(dtype=float)).where(match_mask),
        errors="coerce",
    ).dropna()
    if match_conf.empty:
        conf_stats = {"mean": 0.0, "min": 0.0, "max": 0.0}
    else:
        conf_stats = {
            "mean": float(round(match_conf.mean(), 4)),
            "min": float(round(match_conf.min(), 4)),
            "max": float(round(match_conf.max(), 4)),
        }

    return {
        "total_rows": total_rows,
        "status": {
            "counts": status_counts,
            "percentages": status_percentages,
        },
        "crm_category": {
            "counts": category_counts,
            "percentages": category_percentages,
        },
        "candidates": {
            "avg_total": round(avg_total, 4),
            "avg_used": round(avg_used, 4),
        },
        "confidence": conf_stats,
    }


def save_stats_json(stats: dict, path: Path) -> None:
    """Persist stats dict to JSON file."""

    import json

    target = Path(path)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)


__all__.extend(["compute_stats", "save_stats_json"])
