"""Human adjudication contract for the frozen V9 open-set benchmark."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .contracts import GroundTruthKind
from .v9_dataset import file_sha256, normalize_siret


ADJUDICATION_REQUIRED_COLUMNS = [
    "query_id",
    "label_kind",
    "ground_truth_siret",
    "validator",
    "validated_at",
    "evidence_refs",
    "sirene_snapshot_id",
    "reference_date",
]


def validate_adjudications(rows: pd.DataFrame) -> pd.DataFrame:
    missing = set(ADJUDICATION_REQUIRED_COLUMNS) - set(rows.columns)
    if missing:
        raise ValueError(f"Missing adjudication columns: {sorted(missing)}")
    output = rows.copy()
    allowed = {kind.value for kind in GroundTruthKind}
    invalid = sorted(set(output["label_kind"].astype(str)) - allowed)
    if invalid:
        raise ValueError(f"Unsupported label_kind values: {invalid}")
    if output["query_id"].astype(str).duplicated().any():
        raise ValueError("Open-set benchmark query_id values must be unique")

    output["ground_truth_siret"] = output["ground_truth_siret"].map(normalize_siret)
    match = output["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
    if output.loc[match, "ground_truth_siret"].isna().any():
        raise ValueError("MATCH_EXACT adjudications require a SIRET")
    if output.loc[~match, "ground_truth_siret"].notna().any():
        raise ValueError("Only MATCH_EXACT adjudications may carry a SIRET")
    if output["validator"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Every adjudication requires a human validator")
    if output["evidence_refs"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Every adjudication requires evidence references")
    if output["validated_at"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Every adjudication requires validated_at")

    temporal = output["label_kind"].isin(
        [GroundTruthKind.NO_MATCH.value, GroundTruthKind.MATCH_EXACT.value]
    )
    if output.loc[temporal, "sirene_snapshot_id"].fillna("").str.strip().eq("").any():
        raise ValueError("MATCH_EXACT and NO_MATCH require sirene_snapshot_id")
    if output.loc[temporal, "reference_date"].fillna("").str.strip().eq("").any():
        raise ValueError("MATCH_EXACT and NO_MATCH require reference_date")
    output["ground_truth_siren"] = output["ground_truth_siret"].map(
        lambda value: value[:9] if value else None
    )
    return output


def freeze_adjudications(
    rows: pd.DataFrame,
    *,
    output_dir: Path,
    source_path: Path,
) -> Path:
    validated = validate_adjudications(rows)
    csv_payload = validated.sort_values("query_id").to_csv(index=False)
    benchmark_id = hashlib.sha256(csv_payload.encode("utf-8")).hexdigest()[:16]
    target = output_dir / benchmark_id
    target.mkdir(parents=True, exist_ok=False)
    validated.to_parquet(target / "labels.parquet", index=False)
    manifest = {
        "schema_version": "v9-open-set-1",
        "benchmark_id": benchmark_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "row_count": len(validated),
        "label_counts": validated["label_kind"].value_counts().to_dict(),
        "human_validation_required": True,
        "llm_has_label_authority": False,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


__all__ = [
    "ADJUDICATION_REQUIRED_COLUMNS",
    "validate_adjudications",
    "freeze_adjudications",
]
