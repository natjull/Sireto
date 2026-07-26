"""Versioned, leak-safe contracts for the V4.1 local shadow run.

This module deliberately does not perform retrieval or model inference.  It
defines the boundary around those operations: which CRM rows may be scored,
the deterministic pre-prediction panel, and the immutable artifacts emitted
by a shadow run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import uuid

import pandas as pd

from .v9_dataset import file_sha256, read_table


SHADOW_SCHEMA_VERSION = "sireto-shadow-v4.1-1"
FEEDBACK_SCHEMA_VERSION = "sireto-shadow-feedback-v1"
DEFAULT_PANEL_QUOTAS = {
    "REPRESENTATIVE": 300,
    "CLOSED_WITH_ACTIVE_SIBLING": 50,
    "CLOSED_WITHOUT_ACTIVE_SIBLING": 50,
    "ABSENT_OR_INVALID": 50,
    "ACTIVE_MULTI_SITE": 50,
}


class InputSiretState(str, Enum):
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    NOT_FOUND = "NOT_FOUND"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class ShadowDecision(str, Enum):
    AUTO_MATCH = "AUTO_MATCH"
    REVIEW = "REVIEW"


class ShadowReviewReason(str, Enum):
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS_SIREN = "AMBIGUOUS_SIREN"
    AMBIGUOUS_SITE = "AMBIGUOUS_SITE"
    NO_CANDIDATE = "NO_CANDIDATE"
    NO_ACTIVE_CANDIDATE = "NO_ACTIVE_CANDIDATE"
    AMBIGUOUS_DIRECT = "AMBIGUOUS_DIRECT"
    CLOSED_CANDIDATE = "CLOSED_CANDIDATE"
    INPUT_CONFLICT = "INPUT_CONFLICT"
    RETRIEVAL_DISAGREEMENT = "RETRIEVAL_DISAGREEMENT"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"


class FeedbackOutcome(str, Enum):
    UNKNOWN = "UNKNOWN"
    CONFIRMED = "CONFIRMED"
    CORRECTED = "CORRECTED"


@dataclass(frozen=True)
class DenylistSpec:
    """A consumed evaluation population that must never enter shadow scoring."""

    name: str
    path: Path
    id_column: str = "crm_record_id"
    split_column: str | None = None
    split_value: str | None = None
    manifest_path: Path | None = None

    def load_ids(self) -> set[str]:
        frame = read_table(self.path)
        if self.split_column is not None:
            if self.split_column not in frame:
                raise ValueError(
                    f"{self.name}: missing split column {self.split_column!r}"
                )
            if self.split_value is None:
                raise ValueError(f"{self.name}: split_value is required")
            frame = frame[
                frame[self.split_column].fillna("").astype(str).eq(self.split_value)
            ]
        if self.id_column not in frame:
            raise ValueError(f"{self.name}: missing ID column {self.id_column!r}")
        return {
            value
            for value in frame[self.id_column].map(_normalize_service_id)
            if value is not None
        }

    def provenance(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "path": str(self.path),
            "sha256": file_sha256(self.path),
            "id_column": self.id_column,
            "split_column": self.split_column,
            "split_value": self.split_value,
        }
        if self.manifest_path is not None:
            payload["manifest_path"] = str(self.manifest_path)
            payload["manifest_sha256"] = file_sha256(self.manifest_path)
        return payload


@dataclass(frozen=True)
class V41ShadowDecision:
    """Stable V4.1 result contract with the legacy ``routing_status`` field."""

    service_id: str
    decision: ShadowDecision
    predicted_siret: str | None
    predicted_siren: str | None
    confidence: float
    confidence_kind: str
    review_reason: ShadowReviewReason | None
    model_bundle_id: str
    dataset_manifest_id: str
    shadow_run_id: str
    input_siret: str | None = None
    input_siret_state: InputSiretState = InputSiretState.UNKNOWN
    evidence_tier: str = "NONE"
    candidate_count: int = 0

    def __post_init__(self) -> None:
        normalized = _valid_siret(self.predicted_siret)
        if self.decision == ShadowDecision.AUTO_MATCH:
            if normalized is None or len(normalized) != 14:
                raise ValueError("AUTO_MATCH requires a normalized 14-digit SIRET")
            if self.review_reason is not None:
                raise ValueError("AUTO_MATCH cannot carry a review_reason")
        elif self.review_reason is None:
            raise ValueError("REVIEW requires a review_reason")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0 <= int(self.candidate_count) <= 100:
            raise ValueError("candidate_count must be between 0 and 100")
        if normalized is not None and self.predicted_siren not in {
            None,
            normalized[:9],
        }:
            raise ValueError("predicted_siren must match predicted_siret")
        if not self.service_id.strip():
            raise ValueError("service_id is required")
        if not self.shadow_run_id.strip():
            raise ValueError("shadow_run_id is required")

    @property
    def routing_status(self) -> str:
        return "AUTO" if self.decision == ShadowDecision.AUTO_MATCH else "REVIEW"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SHADOW_SCHEMA_VERSION
        payload["decision"] = self.decision.value
        payload["review_reason"] = (
            self.review_reason.value if self.review_reason is not None else None
        )
        payload["input_siret_state"] = self.input_siret_state.value
        payload["routing_status"] = self.routing_status
        return payload


def _normalize_service_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    output = str(value).strip()
    if not output or output.lower() == "nan":
        return None
    return output


def _valid_siret(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None
    if "e" in raw.lower():
        try:
            raw = str(int(float(raw)))
        except (ValueError, OverflowError):
            return None
    digits = "".join(character for character in raw if character.isdigit())
    return digits if len(digits) == 14 else None


def build_shadow_inventory(
    source: pd.DataFrame,
    *,
    denylists: Mapping[str, Iterable[str]],
    service_id_column: str = "SERVICE ID",
    input_siret_column: str = "SIRET",
) -> pd.DataFrame:
    """Inventory every source row and mark the only rows allowed to be scored."""

    if service_id_column not in source:
        raise ValueError(f"CRM source is missing {service_id_column!r}")
    output = source.copy()
    output.insert(0, "source_row_number", range(1, len(output) + 1))
    output.insert(
        1,
        "service_id",
        output[service_id_column].map(_normalize_service_id),
    )
    non_missing = output["service_id"].dropna()
    duplicates = sorted(non_missing[non_missing.duplicated(keep=False)].unique())
    if duplicates:
        raise ValueError(
            "Non-empty SERVICE ID values must be unique; examples: "
            + ", ".join(duplicates[:5])
        )

    normalized_denylists = {
        name: {
            normalized
            for value in values
            if (normalized := _normalize_service_id(value)) is not None
        }
        for name, values in denylists.items()
    }

    def reasons(service_id: str | None) -> list[str]:
        if service_id is None:
            return ["MISSING_SERVICE_ID"]
        return [
            f"CONSUMED_{name.upper()}"
            for name, ids in normalized_denylists.items()
            if service_id in ids
        ]

    exclusion_reasons = output["service_id"].map(reasons)
    output["eligible_for_shadow"] = exclusion_reasons.map(lambda values: not values)
    output["exclusion_reason"] = exclusion_reasons.map(
        lambda values: "|".join(values) if values else None
    )
    output["denylist_sources"] = exclusion_reasons.map(
        lambda values: json.dumps(
            [value.removeprefix("CONSUMED_") for value in values],
            separators=(",", ":"),
        )
    )
    source_siret = (
        output[input_siret_column]
        if input_siret_column in output
        else pd.Series([None] * len(output), index=output.index)
    )
    output["input_siret"] = source_siret.map(_valid_siret)
    output["input_siren"] = output["input_siret"].map(
        lambda value: value[:9] if value else None
    )
    output["input_siret_state"] = output["input_siret"].map(
        lambda value: (
            InputSiretState.UNKNOWN.value
            if value
            else InputSiretState.INVALID.value
        )
    )
    output["active_sibling_count"] = 0
    output["active_siret_count_for_siren"] = 0
    return output


def enrich_inventory_siret_state(
    inventory: pd.DataFrame,
    sirene: pd.DataFrame,
    *,
    siret_column: str = "siret",
    siren_column: str = "siren",
    state_column: str = "etatAdministratifEtablissement",
) -> pd.DataFrame:
    """Classify suspect input SIRETs against a supplied SIRENE snapshot."""

    required = {siret_column, siren_column, state_column}
    missing = required - set(sirene.columns)
    if missing:
        raise ValueError(f"SIRENE snapshot is missing columns: {sorted(missing)}")
    registry = sirene[[siret_column, siren_column, state_column]].copy()
    registry["_siret"] = registry[siret_column].map(_valid_siret)
    registry["_siren"] = registry[siren_column].fillna("").astype(str).str.zfill(9)
    registry["_active"] = (
        registry[state_column].fillna("").astype(str).str.upper().eq("A")
    )
    registry = registry.dropna(subset=["_siret"]).drop_duplicates("_siret")
    state_by_siret = registry.set_index("_siret")["_active"].to_dict()
    active_counts = (
        registry.loc[registry["_active"]].groupby("_siren")["_siret"].nunique().to_dict()
    )

    output = inventory.copy()

    def classify(value: str | None) -> str:
        if value is None:
            return InputSiretState.INVALID.value
        active = state_by_siret.get(value)
        if active is None:
            return InputSiretState.NOT_FOUND.value
        return (
            InputSiretState.ACTIVE.value
            if active
            else InputSiretState.CLOSED.value
        )

    output["input_siret_state"] = output["input_siret"].map(classify)
    output["active_siret_count_for_siren"] = output["input_siren"].map(
        lambda value: int(active_counts.get(value, 0)) if value else 0
    )
    output["active_sibling_count"] = [
        max(count - (1 if state == InputSiretState.ACTIVE.value else 0), 0)
        for count, state in zip(
            output["active_siret_count_for_siren"],
            output["input_siret_state"],
        )
    ]
    return output


def select_scoreable_rows(
    source: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    service_id_column: str = "SERVICE ID",
) -> pd.DataFrame:
    """Return only authorized source rows, failing closed on reconciliation."""

    required = {"source_row_number", "service_id", "eligible_for_shadow"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Inventory is missing columns: {sorted(missing)}")
    if len(source) != len(inventory):
        raise ValueError("Source/inventory row count mismatch")
    observed = source[service_id_column].map(_normalize_service_id).reset_index(drop=True)
    expected = inventory["service_id"].reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError("Source/inventory SERVICE ID order mismatch")
    allowed = inventory["eligible_for_shadow"].astype(bool).to_numpy()
    selected = source.loc[allowed].copy()
    selected.insert(0, "service_id", expected.loc[allowed].to_numpy())
    if selected["service_id"].isna().any():
        raise AssertionError("Missing SERVICE ID escaped inventory exclusion")
    return selected.reset_index(drop=True)


def _stable_panel_order(service_id: str, seed: int, stratum: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{stratum}\0{service_id}".encode("utf-8")
    ).hexdigest()


def build_pre_prediction_panel(
    inventory: pd.DataFrame,
    *,
    seed: int = 42,
    quotas: Mapping[str, int] = DEFAULT_PANEL_QUOTAS,
) -> pd.DataFrame:
    """Build the deterministic, disjoint 300+50*4 panel before prediction."""

    required = {
        "service_id",
        "eligible_for_shadow",
        "input_siret_state",
        "active_sibling_count",
        "active_siret_count_for_siren",
    }
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Inventory is missing panel columns: {sorted(missing)}")
    if set(quotas) != set(DEFAULT_PANEL_QUOTAS):
        raise ValueError(
            f"Panel quotas must define exactly {sorted(DEFAULT_PANEL_QUOTAS)}"
        )
    eligible = inventory.loc[inventory["eligible_for_shadow"].astype(bool)].copy()
    if eligible["service_id"].duplicated().any():
        raise ValueError("Panel requires unique service_id values")

    state = eligible["input_siret_state"].astype(str)
    masks = {
        "CLOSED_WITH_ACTIVE_SIBLING": state.eq(InputSiretState.CLOSED.value)
        & eligible["active_sibling_count"].astype(int).gt(0),
        "CLOSED_WITHOUT_ACTIVE_SIBLING": state.eq(InputSiretState.CLOSED.value)
        & eligible["active_sibling_count"].astype(int).eq(0),
        "ABSENT_OR_INVALID": state.isin(
            [InputSiretState.NOT_FOUND.value, InputSiretState.INVALID.value]
        ),
        "ACTIVE_MULTI_SITE": state.eq(InputSiretState.ACTIVE.value)
        & eligible["active_siret_count_for_siren"].astype(int).gt(1),
    }
    selected_ids: set[str] = set()
    selected: list[pd.DataFrame] = []
    for stratum in (
        "CLOSED_WITH_ACTIVE_SIBLING",
        "CLOSED_WITHOUT_ACTIVE_SIBLING",
        "ABSENT_OR_INVALID",
        "ACTIVE_MULTI_SITE",
    ):
        candidates = eligible.loc[masks[stratum]].copy()
        candidates = candidates.loc[~candidates["service_id"].isin(selected_ids)]
        candidates["_panel_order"] = candidates["service_id"].map(
            lambda value: _stable_panel_order(value, seed, stratum)
        )
        candidates = candidates.sort_values(
            ["_panel_order", "service_id"]
        ).head(int(quotas[stratum]))
        if len(candidates) != int(quotas[stratum]):
            raise ValueError(
                f"Insufficient rows for panel stratum {stratum}: "
                f"{len(candidates)} < {quotas[stratum]}"
            )
        candidates["panel_stratum"] = stratum
        selected_ids.update(candidates["service_id"])
        selected.append(candidates)

    representative = eligible.loc[~eligible["service_id"].isin(selected_ids)].copy()
    representative["_panel_order"] = representative["service_id"].map(
        lambda value: _stable_panel_order(value, seed, "REPRESENTATIVE")
    )
    representative = representative.sort_values(
        ["_panel_order", "service_id"]
    ).head(int(quotas["REPRESENTATIVE"]))
    if len(representative) != int(quotas["REPRESENTATIVE"]):
        raise ValueError(
            "Insufficient rows for panel stratum REPRESENTATIVE: "
            f"{len(representative)} < {quotas['REPRESENTATIVE']}"
        )
    representative["panel_stratum"] = "REPRESENTATIVE"
    selected.insert(0, representative)
    panel = pd.concat(selected, ignore_index=True)
    panel["panel_seed"] = seed
    panel["panel_order"] = panel.groupby("panel_stratum").cumcount() + 1
    panel = panel.drop(columns=["_panel_order"])
    if panel["service_id"].duplicated().any():
        raise AssertionError("Pre-prediction panel is not disjoint")
    return panel


def inventory_summary(inventory: pd.DataFrame) -> dict[str, Any]:
    eligible = inventory["eligible_for_shadow"].astype(bool)
    return {
        "source_row_count": int(len(inventory)),
        "eligible_row_count": int(eligible.sum()),
        "excluded_row_count": int((~eligible).sum()),
        "exclusion_reason_counts": inventory.loc[
            ~eligible, "exclusion_reason"
        ].value_counts(dropna=False).to_dict(),
        "eligible_input_siret_state_counts": inventory.loc[
            eligible, "input_siret_state"
        ].value_counts(dropna=False).to_dict(),
    }


def _decision_frame(
    decisions: pd.DataFrame | Sequence[V41ShadowDecision],
) -> pd.DataFrame:
    if isinstance(decisions, pd.DataFrame):
        output = decisions.copy()
        if "schema_version" not in output:
            output.insert(0, "schema_version", SHADOW_SCHEMA_VERSION)
        return output
    return pd.DataFrame([decision.to_dict() for decision in decisions])


def _validate_shadow_outputs(
    inventory: pd.DataFrame,
    decisions: pd.DataFrame,
    candidates_top10: pd.DataFrame,
    panel: pd.DataFrame,
) -> None:
    eligible_ids = set(
        inventory.loc[
            inventory["eligible_for_shadow"].astype(bool), "service_id"
        ].astype(str)
    )
    required_decision_columns = {
        "schema_version",
        "service_id",
        "decision",
        "predicted_siret",
        "predicted_siren",
        "confidence",
        "confidence_kind",
        "review_reason",
        "model_bundle_id",
        "dataset_manifest_id",
        "shadow_run_id",
        "candidate_count",
        "routing_status",
    }
    missing_decision_columns = required_decision_columns - set(decisions)
    if missing_decision_columns:
        raise ValueError(
            "Decisions are missing columns: "
            f"{sorted(missing_decision_columns)}"
        )
    if not decisions["schema_version"].astype(str).eq(
        SHADOW_SCHEMA_VERSION
    ).all():
        raise ValueError("Unsupported shadow decision schema_version")
    decision_ids = set(decisions["service_id"].astype(str))
    if decisions["service_id"].astype(str).duplicated().any():
        raise ValueError("Decisions must contain one row per service_id")
    if decision_ids != eligible_ids:
        missing = sorted(eligible_ids - decision_ids)[:5]
        forbidden = sorted(decision_ids - eligible_ids)[:5]
        raise ValueError(
            "Decisions must exactly match eligible inventory rows; "
            f"missing={missing}, forbidden={forbidden}"
        )
    if len(decisions) != len(eligible_ids):
        raise ValueError("Decision cardinality differs from eligible inventory")
    if "routing_status" not in decisions:
        raise ValueError("Decisions must retain legacy routing_status")
    expected_routing = decisions["decision"].astype(str).map(
        {"AUTO_MATCH": "AUTO", "REVIEW": "REVIEW"}
    )
    if expected_routing.isna().any() or not expected_routing.equals(
        decisions["routing_status"].astype(str)
    ):
        raise ValueError("routing_status is incompatible with decision")
    confidence = pd.to_numeric(decisions["confidence"], errors="coerce")
    if confidence.isna().any() or not confidence.between(0.0, 1.0).all():
        raise ValueError("Decision confidence must be between 0 and 1")
    candidate_count = pd.to_numeric(
        decisions["candidate_count"], errors="coerce"
    )
    if candidate_count.isna().any() or not candidate_count.between(0, 100).all():
        raise ValueError("Decision candidate_count must be between 0 and 100")
    auto = decisions["decision"].astype(str).eq("AUTO_MATCH")
    auto_sirets = decisions.loc[auto, "predicted_siret"].map(_valid_siret)
    if auto_sirets.isna().any():
        raise ValueError("AUTO_MATCH decisions require a 14-digit SIRET")
    if decisions.loc[auto, "review_reason"].notna().any():
        raise ValueError("AUTO_MATCH decisions cannot carry a review_reason")
    if (
        decisions.loc[~auto, "review_reason"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError("REVIEW decisions require a review_reason")
    predicted_pairs = decisions.loc[
        decisions["predicted_siret"].notna(),
        ["predicted_siret", "predicted_siren"],
    ]
    for raw_siret, raw_siren in predicted_pairs.itertuples(index=False, name=None):
        normalized_siret = _valid_siret(raw_siret)
        if normalized_siret is None:
            raise ValueError("predicted_siret must contain exactly 14 digits")
        if (
            raw_siren is not None
            and not pd.isna(raw_siren)
            and str(raw_siren) != normalized_siret[:9]
        ):
            raise ValueError("predicted_siren must match predicted_siret")

    if not candidates_top10.empty:
        required = {"service_id", "candidate_siret", "rank"}
        missing_columns = required - set(candidates_top10)
        if missing_columns:
            raise ValueError(
                f"Top-10 candidates are missing columns: {sorted(missing_columns)}"
            )
        candidate_ids = set(candidates_top10["service_id"].astype(str))
        if not candidate_ids <= eligible_ids:
            raise ValueError("Top-10 candidates contain excluded SERVICE IDs")
        ranks = pd.to_numeric(candidates_top10["rank"], errors="coerce")
        if ranks.isna().any() or not ranks.between(1, 10).all():
            raise ValueError("Top-10 candidate ranks must be between 1 and 10")
        if candidates_top10.groupby("service_id").size().gt(10).any():
            raise ValueError("More than ten evidence candidates for a SERVICE ID")
        if "etat_admin" in candidates_top10 and (
            candidates_top10["etat_admin"].fillna("").astype(str).str.upper().ne("A")
        ).any():
            raise ValueError("Closed candidates are forbidden from shadow output")

    panel_ids = set(panel["service_id"].astype(str))
    if not panel_ids <= eligible_ids:
        raise ValueError("Panel contains excluded SERVICE IDs")
    if panel["service_id"].astype(str).duplicated().any():
        raise ValueError("Panel SERVICE IDs must be unique")


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def write_shadow_run(
    *,
    output_root: Path,
    run_id: str,
    inventory: pd.DataFrame,
    decisions: pd.DataFrame | Sequence[V41ShadowDecision],
    candidates_top10: pd.DataFrame,
    evidence: Iterable[Mapping[str, Any]],
    panel: pd.DataFrame,
    input_artifacts: Mapping[str, Path],
    run_metadata: Mapping[str, Any],
) -> Path:
    """Atomically publish a complete local shadow export on one filesystem."""

    if not run_id.strip() or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty filesystem-safe name")
    inventory = inventory.copy()
    inventory["artifact_schema_version"] = SHADOW_SCHEMA_VERSION
    decision_frame = _decision_frame(decisions)
    if not decision_frame["shadow_run_id"].astype(str).eq(run_id).all():
        raise ValueError("Every decision shadow_run_id must equal run_id")
    candidates_top10 = candidates_top10.copy()
    if "schema_version" not in candidates_top10:
        candidates_top10.insert(
            0, "schema_version", "sireto-shadow-candidates-v4.1-1"
        )
    panel = panel.copy()
    panel["artifact_schema_version"] = "sireto-shadow-panel-v4.1-1"
    evidence_records = []
    eligible_ids = set(
        inventory.loc[
            inventory["eligible_for_shadow"].astype(bool), "service_id"
        ].astype(str)
    )
    for raw_record in evidence:
        record = dict(raw_record)
        record.setdefault("schema_version", "sireto-shadow-evidence-v4.1-1")
        service_id = _normalize_service_id(record.get("service_id"))
        if service_id is None:
            raise ValueError("Every evidence record requires service_id")
        if service_id not in eligible_ids:
            raise ValueError("Evidence contains an excluded SERVICE ID")
        evidence_records.append(record)
    _validate_shadow_outputs(
        inventory,
        decision_frame,
        candidates_top10,
        panel,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    if final_dir.exists():
        raise FileExistsError(f"Immutable shadow run already exists: {final_dir}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output_root)
    )
    try:
        inventory.to_parquet(staging / "inventory.parquet", index=False)
        decision_frame.to_parquet(staging / "decisions.parquet", index=False)
        decision_frame.to_csv(staging / "decisions.csv", index=False)
        candidates_top10.to_parquet(
            staging / "candidates_top10.parquet", index=False
        )
        panel.to_parquet(staging / "panel500.parquet", index=False)
        evidence_path = staging / "evidence.jsonl"
        _write_jsonl(evidence_path, evidence_records)

        decision_counts = (
            decision_frame["decision"].astype(str).value_counts().to_dict()
        )
        review_counts = (
            decision_frame.loc[
                decision_frame["decision"].astype(str).eq("REVIEW"),
                "review_reason",
            ]
            .fillna("UNSPECIFIED")
            .astype(str)
            .value_counts()
            .to_dict()
        )
        latency_summary = None
        if "latency_ms" in decision_frame:
            latency = pd.to_numeric(
                decision_frame["latency_ms"], errors="coerce"
            ).dropna()
            if not latency.empty:
                latency_summary = {
                    "count": int(len(latency)),
                    "p50_ms": float(latency.quantile(0.50)),
                    "p95_ms": float(latency.quantile(0.95)),
                    "max_ms": float(latency.max()),
                }
        summary = {
            **inventory_summary(inventory),
            "decision_count": int(len(decision_frame)),
            "decision_counts": decision_counts,
            "review_reason_counts": review_counts,
            "latency": latency_summary,
            "candidate_evidence_row_count": int(len(candidates_top10)),
            "evidence_record_count": int(len(evidence_records)),
            "panel_row_count": int(len(panel)),
            "precision_claim": None,
            "shadow_is_independent_evaluation": False,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        output_names = [
            "inventory.parquet",
            "decisions.parquet",
            "decisions.csv",
            "candidates_top10.parquet",
            "evidence.jsonl",
            "panel500.parquet",
            "summary.json",
        ]
        manifest_core = {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_metadata": dict(run_metadata),
            "inputs": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in sorted(input_artifacts.items())
            },
            "outputs": {
                name: file_sha256(staging / name) for name in output_names
            },
            "row_counts": {
                "inventory": int(len(inventory)),
                "decisions": int(len(decision_frame)),
                "candidates_top10": int(len(candidates_top10)),
                "panel": int(len(panel)),
            },
            "invariants": {
                "excluded_rows_scored": 0,
                "decision_population_equals_eligible_inventory": True,
                "candidate_cap": 100,
                "closed_final_candidates_allowed": False,
                "crm_writes_performed": False,
                "precision_claim_allowed": False,
            },
        }
        canonical = json.dumps(
            manifest_core,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
        manifest_core["manifest_id"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()[:16]
        (staging / "manifest.json").write_text(
            json.dumps(
                manifest_core,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_dir


def feedback_outcome(
    *,
    proposed_siret: str | None,
    later_crm_siret: str | None,
    explicit_crm_change: bool,
) -> FeedbackOutcome:
    """Map a later CRM state to feedback without treating silence as truth."""

    if not explicit_crm_change:
        return FeedbackOutcome.UNKNOWN
    proposed = _valid_siret(proposed_siret)
    later = _valid_siret(later_crm_siret)
    if proposed is not None and later == proposed:
        return FeedbackOutcome.CONFIRMED
    return FeedbackOutcome.CORRECTED


def append_feedback_event(
    journal_path: Path,
    *,
    shadow_run_id: str,
    service_id: str,
    proposed_siret: str | None,
    later_crm_siret: str | None,
    explicit_crm_change: bool,
    source_snapshot_id: str,
    observed_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one immutable feedback event under an advisory file lock."""

    if not service_id.strip():
        raise ValueError("service_id is required")
    event = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "observed_at": observed_at or datetime.now(timezone.utc).isoformat(),
        "shadow_run_id": shadow_run_id,
        "service_id": service_id,
        "proposed_siret": _valid_siret(proposed_siret),
        "later_crm_siret": _valid_siret(later_crm_siret),
        "explicit_crm_change": bool(explicit_crm_change),
        "outcome": feedback_outcome(
            proposed_siret=proposed_siret,
            later_crm_siret=later_crm_siret,
            explicit_crm_change=explicit_crm_change,
        ).value,
        "source_snapshot_id": source_snapshot_id,
        "metadata": dict(metadata or {}),
        "eligible_for_automatic_retraining": False,
    }
    journal_path = Path(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            default=_json_default,
        )
        + "\n"
    )
    with journal_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return event


__all__ = [
    "DEFAULT_PANEL_QUOTAS",
    "DenylistSpec",
    "FEEDBACK_SCHEMA_VERSION",
    "FeedbackOutcome",
    "InputSiretState",
    "SHADOW_SCHEMA_VERSION",
    "ShadowDecision",
    "ShadowReviewReason",
    "V41ShadowDecision",
    "append_feedback_event",
    "build_pre_prediction_panel",
    "build_shadow_inventory",
    "enrich_inventory_siret_state",
    "feedback_outcome",
    "inventory_summary",
    "select_scoreable_rows",
    "write_shadow_run",
]
