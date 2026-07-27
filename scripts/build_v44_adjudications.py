#!/usr/bin/env python3
"""Build immutable V4.4 adjudications from facts, proofs and explicit judgments.

The builder never infers a judgment from model scores or lexical similarities.
It validates an explicit judgment, determines whether the cited proofs satisfy
the frozen V4.4 independence policy, and derives all training targets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.4-adjudications-1"
POLICY_VERSION = "two-independent-identity-proofs-v1"
LABELS = {"TOP1_CORRECT", "TOP1_WRONG", "AMBIGUOUS", "UNRESOLVED"}
TRAINING_LABELS = LABELS - {"UNRESOLVED"}
SIRENE_INDEPENDENCE_GROUP = "SIRENE_REGISTRY"
SIRENE_SOURCE_FAMILIES = {
    "SIRENE",
    "SIRENE_SNAPSHOT",
    "SIRENE_DERIVED_RECHERCHE_ENTREPRISES_API",
    "RECHERCHE_ENTREPRISES_API",
    "API_RECHERCHE_ENTREPRISES",
    "ANNUAIRE_ENTREPRISES",
    "ANNUAIRE_ENTREPRISES_SIRENE_VIEW",
}
FORBIDDEN_PROOF_FAMILY_TOKENS = {
    "MODEL",
    "RANKER",
    "ACCEPTOR",
    "DECIDER",
    "CRM_ADDRESS",
}
DERIVED_INPUT_COLUMNS = {"evidence_validated", "training_eligible"}

FACT_REQUIRED = {
    "audit_case_id",
    "service_id",
    "frozen_top1_siret",
    "frozen_model_bundle_id",
    "frozen_retrieval_signature",
    "frozen_candidate_sirets_json",
    "frozen_candidate_pool_sha256",
}
PROOF_REQUIRED = {
    "proof_id",
    "audit_case_id",
    "producer",
    "source_family",
    "independence_group",
    "source_locator",
    "collected_at",
    "proof_kind",
    "supports_label",
    "identity_consistent",
    "contradiction_unresolved",
}
JUDGMENT_REQUIRED = {
    "audit_case_id",
    "adjudication_label",
    "validated_correct_siret",
    "evidence_ref_ids_json",
    "adjudication_reason",
    "adjudication_rule_version",
    "adjudicated_at",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _siret(value: Any, *, required: bool = False) -> str | None:
    digits = "".join(character for character in _text(value) if character.isdigit())
    if not digits and not required:
        return None
    if len(digits) != 14:
        raise ValueError(f"Invalid SIRET: {value!r}")
    return digits


def _utc_iso(value: Any, *, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    try:
        timestamp = pd.Timestamp(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not a valid timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return timestamp.tz_convert("UTC").isoformat()


def _boolean(value: Any, *, field: str) -> bool:
    if value is True or value is False or type(value).__name__ == "bool_":
        return bool(value)
    raise ValueError(f"{field} must be a boolean")


def _json_list(value: Any, *, field: str) -> list[str]:
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        try:
            raw = json.loads(_text(value) or "[]")
        except json.JSONDecodeError as error:
            raise ValueError(f"{field} must be a JSON list") from error
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a JSON list")
    values = [_text(item) for item in raw]
    if any(not item for item in values):
        raise ValueError(f"{field} cannot contain empty values")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} cannot contain duplicates")
    return values


def candidate_pool_sha256(sirets: Iterable[str]) -> str:
    """Hash an ordered frozen candidate pool in its canonical representation."""

    normalized = [_siret(value, required=True) for value in sirets]
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sirene_family(source_family: str) -> bool:
    family = source_family.upper()
    return (
        family in SIRENE_SOURCE_FAMILIES
        or "SIRENE" in family
        or "RECHERCHE_ENTREPRISES" in family
    )


def _validate_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    name: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    forbidden = DERIVED_INPUT_COLUMNS & set(frame.columns)
    if forbidden:
        raise ValueError(
            f"{name} cannot provide derived columns: {sorted(forbidden)}"
        )


def _canonical_facts(facts: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(facts, FACT_REQUIRED, name="facts")
    output = facts.copy()
    output["audit_case_id"] = output["audit_case_id"].map(_text)
    output["service_id"] = output["service_id"].map(_text)
    if output["audit_case_id"].eq("").any() or output["service_id"].eq("").any():
        raise ValueError("facts require non-empty case and service IDs")
    if output["audit_case_id"].duplicated().any():
        raise ValueError("facts.audit_case_id must be unique")
    if output["service_id"].duplicated().any():
        raise ValueError("facts.service_id must be unique")

    records: list[dict[str, Any]] = []
    for raw in output.to_dict("records"):
        top1 = _siret(raw["frozen_top1_siret"], required=True)
        pool = _json_list(
            raw["frozen_candidate_sirets_json"],
            field="frozen_candidate_sirets_json",
        )
        pool = [_siret(value, required=True) for value in pool]
        if not 1 <= len(pool) <= 100:
            raise ValueError("Frozen candidate pools must contain 1 to 100 SIRETs")
        if top1 not in pool:
            raise ValueError("frozen_top1_siret must belong to its candidate pool")
        expected_pool_hash = candidate_pool_sha256(pool)
        observed_pool_hash = _text(raw["frozen_candidate_pool_sha256"]).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", observed_pool_hash):
            raise ValueError("frozen_candidate_pool_sha256 must be a SHA-256")
        if observed_pool_hash != expected_pool_hash:
            raise ValueError("Frozen candidate pool hash mismatch")
        supplied_siren = _text(raw.get("frozen_top1_siren"))
        if supplied_siren and supplied_siren != top1[:9]:
            raise ValueError("frozen_top1_siren does not match frozen_top1_siret")
        model_bundle = _text(raw["frozen_model_bundle_id"])
        retrieval = _text(raw["frozen_retrieval_signature"])
        if not model_bundle or not retrieval:
            raise ValueError("Frozen model and retrieval identifiers are required")
        records.append(
            {
                **raw,
                "audit_case_id": _text(raw["audit_case_id"]),
                "service_id": _text(raw["service_id"]),
                "query_id": _text(raw.get("query_id")) or _text(raw["service_id"]),
                "frozen_top1_siret": top1,
                "frozen_top1_siren": top1[:9],
                "frozen_model_bundle_id": model_bundle,
                "frozen_retrieval_signature": retrieval,
                "frozen_candidate_sirets_json": json.dumps(
                    pool, separators=(",", ":")
                ),
                "frozen_candidate_pool_sha256": observed_pool_hash,
                "sampling_stratum": _text(raw.get("sampling_stratum")),
                "priority_reason": _text(raw.get("priority_reason")),
            }
        )
    return pd.DataFrame(records)


def _canonical_proofs(proofs: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(proofs, PROOF_REQUIRED, name="proofs")
    output = proofs.copy()
    if output.empty:
        output["is_sirene_correlated"] = pd.Series(dtype=bool)
        return output
    output["proof_id"] = output["proof_id"].map(_text)
    output["audit_case_id"] = output["audit_case_id"].map(_text)
    if output["proof_id"].eq("").any() or output["audit_case_id"].eq("").any():
        raise ValueError("proofs require non-empty proof and case IDs")
    if output["proof_id"].duplicated().any():
        raise ValueError("proofs.proof_id must be unique")

    records: list[dict[str, Any]] = []
    family_groups: dict[str, str] = {}
    producer_groups: dict[str, str] = {}
    locator_groups: dict[str, str] = {}
    for raw in output.to_dict("records"):
        producer = _text(raw["producer"]).upper()
        family = _text(raw["source_family"]).upper()
        group = _text(raw["independence_group"]).upper()
        locator = _text(raw["source_locator"])
        if not producer or not family or not group or not locator:
            raise ValueError(
                "Every proof requires producer, family, independence group and locator"
            )
        if any(token in family for token in FORBIDDEN_PROOF_FAMILY_TOKENS):
            raise ValueError(f"Model/address evidence is forbidden: {family}")
        sirene_derived = _is_sirene_family(family)
        if sirene_derived and group != SIRENE_INDEPENDENCE_GROUP:
            raise ValueError(
                f"{family} is SIRENE-correlated and must use "
                f"{SIRENE_INDEPENDENCE_GROUP}"
            )
        if family in family_groups and family_groups[family] != group:
            raise ValueError(
                f"One source family cannot claim multiple independence groups: {family}"
            )
        family_groups[family] = group
        if producer in producer_groups and producer_groups[producer] != group:
            raise ValueError(
                "One producer cannot claim multiple independence groups: "
                f"{producer}"
            )
        producer_groups[producer] = group
        hostname = (urlparse(locator).hostname or "").lower()
        if (
            hostname
            and not sirene_derived
            and hostname in locator_groups
            and locator_groups[hostname] != group
        ):
            raise ValueError(
                f"One source hostname cannot claim multiple independence groups: "
                f"{hostname}"
            )
        if hostname and not sirene_derived:
            locator_groups[hostname] = group
        supports_label = _text(raw["supports_label"]).upper()
        if supports_label not in LABELS:
            raise ValueError(f"Unsupported proof judgment: {supports_label}")
        proof_kind = _text(raw["proof_kind"]).upper()
        if not proof_kind.startswith("IDENTITY_"):
            raise ValueError(
                "Only explicit identity evidence can support an adjudication"
            )
        records.append(
            {
                **raw,
                "proof_id": _text(raw["proof_id"]),
                "audit_case_id": _text(raw["audit_case_id"]),
                "producer": producer,
                "source_family": family,
                "independence_group": group,
                "source_locator": locator,
                "collected_at": _utc_iso(
                    raw["collected_at"], field="proofs.collected_at"
                ),
                "proof_kind": proof_kind,
                "supports_label": supports_label,
                "identity_consistent": _boolean(
                    raw["identity_consistent"],
                    field="proofs.identity_consistent",
                ),
                "contradiction_unresolved": _boolean(
                    raw["contradiction_unresolved"],
                    field="proofs.contradiction_unresolved",
                ),
                "is_sirene_correlated": sirene_derived,
            }
        )
    return pd.DataFrame(records)


def _canonical_judgments(judgments: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(judgments, JUDGMENT_REQUIRED, name="judgments")
    output = judgments.copy()
    output["audit_case_id"] = output["audit_case_id"].map(_text)
    if output["audit_case_id"].eq("").any():
        raise ValueError("judgments require non-empty case IDs")
    if output["audit_case_id"].duplicated().any():
        raise ValueError("judgments.audit_case_id must be unique")
    records: list[dict[str, Any]] = []
    for raw in output.to_dict("records"):
        label = _text(raw["adjudication_label"]).upper()
        if label not in LABELS:
            raise ValueError(f"Unsupported adjudication label: {label}")
        reason = _text(raw["adjudication_reason"])
        rule = _text(raw["adjudication_rule_version"])
        if not reason or not rule:
            raise ValueError("Every judgment requires a reason and rule version")
        records.append(
            {
                **raw,
                "audit_case_id": _text(raw["audit_case_id"]),
                "adjudication_label": label,
                "validated_correct_siret": _siret(
                    raw.get("validated_correct_siret")
                ),
                "evidence_ref_ids": _json_list(
                    raw["evidence_ref_ids_json"],
                    field="evidence_ref_ids_json",
                ),
                "adjudication_reason": reason,
                "adjudication_rule_version": rule,
                "adjudicated_at": _utc_iso(
                    raw["adjudicated_at"], field="judgments.adjudicated_at"
                ),
            }
        )
    return pd.DataFrame(records)


def build_adjudications(
    facts: pd.DataFrame,
    proofs: pd.DataFrame,
    judgments: pd.DataFrame,
) -> pd.DataFrame:
    """Validate inputs and derive canonical adjudications and training targets."""

    facts = _canonical_facts(facts)
    proofs = _canonical_proofs(proofs)
    judgments = _canonical_judgments(judgments)
    fact_ids = set(facts["audit_case_id"])
    judgment_ids = set(judgments["audit_case_id"])
    if fact_ids != judgment_ids:
        raise ValueError("Facts and judgments must cover exactly the same cases")
    unknown_proof_cases = set(proofs["audit_case_id"]) - fact_ids
    if unknown_proof_cases:
        raise ValueError("Proofs reference cases absent from facts")
    proof_index = proofs.set_index("proof_id", drop=False)
    fact_index = facts.set_index("audit_case_id")
    records: list[dict[str, Any]] = []

    for judgment in judgments.sort_values("audit_case_id").to_dict("records"):
        case_id = judgment["audit_case_id"]
        fact = fact_index.loc[case_id]
        label = judgment["adjudication_label"]
        top1 = fact["frozen_top1_siret"]
        validated = judgment["validated_correct_siret"]
        if label == "TOP1_CORRECT" and validated != top1:
            raise ValueError(
                f"{case_id}: TOP1_CORRECT requires validated SIRET equal to top1"
            )
        if label == "TOP1_WRONG" and validated == top1:
            raise ValueError(
                f"{case_id}: TOP1_WRONG replacement cannot equal the frozen top1"
            )
        if label in {"AMBIGUOUS", "UNRESOLVED"} and validated is not None:
            raise ValueError(
                f"{case_id}: {label} cannot carry a validated exact SIRET"
            )

        reference_ids = judgment["evidence_ref_ids"]
        unknown_refs = sorted(set(reference_ids) - set(proof_index.index))
        if unknown_refs:
            raise ValueError(f"{case_id}: unknown proof references: {unknown_refs}")
        cited = (
            proof_index.loc[reference_ids].copy()
            if reference_ids
            else proofs.iloc[0:0].copy()
        )
        if not cited.empty and cited["audit_case_id"].ne(case_id).any():
            raise ValueError(f"{case_id}: cited proof belongs to another case")
        if not cited.empty and cited["supports_label"].ne(label).any():
            raise ValueError(f"{case_id}: cited proof supports another judgment")

        group_count = int(cited["independence_group"].nunique())
        family_count = int(cited["source_family"].nunique())
        contradiction = bool(
            cited["contradiction_unresolved"].any()
        ) if not cited.empty else False
        identity_consistent = bool(
            cited["identity_consistent"].all()
        ) if not cited.empty else False
        evidence_validated = bool(
            label in TRAINING_LABELS
            and len(cited) >= 2
            and group_count >= 2
            and identity_consistent
            and not contradiction
        )
        training_eligible = evidence_validated
        pool = _json_list(
            fact["frozen_candidate_sirets_json"],
            field="frozen_candidate_sirets_json",
        )
        ranker_target = (
            validated
            if label in {"TOP1_CORRECT", "TOP1_WRONG"} and validated
            else None
        )
        ranker_target_in_pool = bool(ranker_target and ranker_target in pool)
        ranker_eligible = bool(
            training_eligible and ranker_target_in_pool
        )
        acceptor_target = (
            1 if label == "TOP1_CORRECT" else (0 if label in TRAINING_LABELS else None)
        )
        collected = sorted(cited["collected_at"].astype(str)) if not cited.empty else []
        records.append(
            {
                "audit_case_id": case_id,
                "query_id": fact["query_id"],
                "service_id": fact["service_id"],
                "adjudication_label": label,
                "evidence_validated": evidence_validated,
                "training_eligible": training_eligible,
                "acceptor_target": acceptor_target,
                "acceptor_eligible": training_eligible,
                "ranker_target_siret": ranker_target,
                "ranker_target_in_frozen_pool": ranker_target_in_pool,
                "ranker_eligible": ranker_eligible,
                "frozen_top1_siret": top1,
                "frozen_top1_siren": fact["frozen_top1_siren"],
                "frozen_model_bundle_id": fact["frozen_model_bundle_id"],
                "frozen_retrieval_signature": fact["frozen_retrieval_signature"],
                "frozen_candidate_pool_sha256": fact[
                    "frozen_candidate_pool_sha256"
                ],
                "validated_correct_siret": validated,
                "validated_correct_siren": validated[:9] if validated else None,
                "evidence_ref_ids_json": json.dumps(
                    reference_ids, separators=(",", ":")
                ),
                "evidence_source_families_json": json.dumps(
                    sorted(set(cited["source_family"])),
                    separators=(",", ":"),
                ),
                "cited_proof_count": int(len(cited)),
                "independent_evidence_group_count": group_count,
                "independent_source_family_count": family_count,
                "sirene_correlated_proof_count": int(
                    cited["is_sirene_correlated"].sum()
                ) if not cited.empty else 0,
                "all_cited_identity_consistent": identity_consistent,
                "has_unresolved_contradiction": contradiction,
                "evidence_collected_at_min": collected[0] if collected else None,
                "evidence_collected_at_max": collected[-1] if collected else None,
                "adjudication_reason": judgment["adjudication_reason"],
                "adjudication_rule_version": judgment[
                    "adjudication_rule_version"
                ],
                "adjudicated_at": judgment["adjudicated_at"],
                "sampling_stratum": fact["sampling_stratum"],
                "priority_reason": fact["priority_reason"],
            }
        )
    output = pd.DataFrame(records)
    validate_canonical_adjudications(output)
    return output


def validate_canonical_adjudications(adjudications: pd.DataFrame) -> None:
    """Validate the self-contained invariants of a canonical output table."""

    required = {
        "audit_case_id",
        "adjudication_label",
        "evidence_validated",
        "training_eligible",
        "acceptor_target",
        "acceptor_eligible",
        "ranker_target_siret",
        "ranker_target_in_frozen_pool",
        "ranker_eligible",
        "frozen_top1_siret",
        "frozen_top1_siren",
        "validated_correct_siret",
        "validated_correct_siren",
        "cited_proof_count",
        "independent_evidence_group_count",
        "all_cited_identity_consistent",
        "has_unresolved_contradiction",
    }
    missing = required - set(adjudications.columns)
    if missing:
        raise ValueError(f"Canonical adjudications missing: {sorted(missing)}")
    if adjudications["audit_case_id"].astype(str).duplicated().any():
        raise ValueError("Canonical audit_case_id must be unique")
    if not set(adjudications["adjudication_label"]).issubset(LABELS):
        raise ValueError("Canonical adjudications contain an unsupported label")
    if not adjudications["training_eligible"].astype(bool).equals(
        adjudications["evidence_validated"].astype(bool)
    ):
        raise ValueError("training_eligible must be derived from evidence_validated")
    if not adjudications["acceptor_eligible"].astype(bool).equals(
        adjudications["training_eligible"].astype(bool)
    ):
        raise ValueError("acceptor_eligible must be derived from training_eligible")
    eligible = adjudications["training_eligible"].astype(bool)
    expected_evidence = (
        adjudications["adjudication_label"].isin(TRAINING_LABELS)
        & adjudications["cited_proof_count"].ge(2)
        & adjudications["independent_evidence_group_count"].ge(2)
        & adjudications["all_cited_identity_consistent"].astype(bool)
        & ~adjudications["has_unresolved_contradiction"].astype(bool)
    )
    if not eligible.equals(expected_evidence):
        raise ValueError("Canonical evidence eligibility fields are inconsistent")
    expected_acceptor = adjudications["adjudication_label"].map(
        {"TOP1_CORRECT": 1, "TOP1_WRONG": 0, "AMBIGUOUS": 0}
    )
    observed_acceptor = pd.to_numeric(
        adjudications["acceptor_target"], errors="coerce"
    )
    if not expected_acceptor.fillna(-1).equals(observed_acceptor.fillna(-1)):
        raise ValueError("acceptor_target is inconsistent with adjudication_label")
    correct = adjudications["adjudication_label"].eq("TOP1_CORRECT")
    if not adjudications.loc[correct, "validated_correct_siret"].equals(
        adjudications.loc[correct, "frozen_top1_siret"]
    ):
        raise ValueError("TOP1_CORRECT exact target invariant failed")
    wrong = adjudications["adjudication_label"].eq("TOP1_WRONG")
    wrong_with_target = wrong & adjudications["validated_correct_siret"].notna()
    if (
        adjudications.loc[wrong_with_target, "validated_correct_siret"]
        == adjudications.loc[wrong_with_target, "frozen_top1_siret"]
    ).any():
        raise ValueError("TOP1_WRONG exact target invariant failed")
    non_exact = adjudications["adjudication_label"].isin(
        {"AMBIGUOUS", "UNRESOLVED"}
    )
    if adjudications.loc[non_exact, "validated_correct_siret"].notna().any():
        raise ValueError("Non-exact adjudications cannot carry an exact target")
    expected_ranker_eligible = (
        adjudications["training_eligible"].astype(bool)
        & adjudications["ranker_target_siret"].notna()
        & adjudications["ranker_target_in_frozen_pool"].astype(bool)
    )
    if not adjudications["ranker_eligible"].astype(bool).equals(
        expected_ranker_eligible
    ):
        raise ValueError("ranker_eligible is inconsistent with its exact target")
    target_siren = adjudications["validated_correct_siret"].map(
        lambda value: str(value)[:9] if pd.notna(value) else None
    )
    if not target_siren.fillna("").equals(
        adjudications["validated_correct_siren"].fillna("").astype(str)
    ):
        raise ValueError("validated_correct_siren is inconsistent with its SIRET")


def build_artifact(
    *,
    facts_path: Path,
    proofs_path: Path,
    judgments_path: Path,
    output_root: Path,
) -> Path:
    """Build an atomic, content-addressed adjudication artifact."""

    inputs = {
        "facts": Path(facts_path),
        "proofs": Path(proofs_path),
        "judgments": Path(judgments_path),
    }
    input_hashes = {name: file_sha256(path) for name, path in inputs.items()}
    identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "input_hashes": input_hashes,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.4 adjudications exist: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent)
    )
    try:
        adjudications = build_adjudications(
            pd.read_parquet(inputs["facts"]),
            pd.read_parquet(inputs["proofs"]),
            pd.read_parquet(inputs["judgments"]),
        )
        adjudications_path = staging / "adjudications.parquet"
        adjudications.to_parquet(adjudications_path, index=False)
        label_counts = {
            str(key): int(value)
            for key, value in adjudications["adjudication_label"]
            .value_counts()
            .items()
        }
        summary = {
            "case_count": int(len(adjudications)),
            "label_counts": label_counts,
            "evidence_validated_count": int(
                adjudications["evidence_validated"].sum()
            ),
            "training_eligible_count": int(
                adjudications["training_eligible"].sum()
            ),
            "acceptor_eligible_count": int(
                adjudications["acceptor_eligible"].sum()
            ),
            "ranker_eligible_count": int(
                adjudications["ranker_eligible"].sum()
            ),
            "unresolved_count": label_counts.get("UNRESOLVED", 0),
        }
        summary_path = staging / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        outputs = {
            adjudications_path.name: file_sha256(adjudications_path),
            summary_path.name: file_sha256(summary_path),
        }
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                name: {"path": str(path), "sha256": input_hashes[name]}
                for name, path in inputs.items()
            },
            "outputs": outputs,
            "invariants": {
                "training_eligible_is_derived": True,
                "minimum_independent_evidence_groups": 2,
                "sirene_views_share_one_independence_group": True,
                "model_or_address_only_proof_forbidden": True,
                "positive_injection": False,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_artifact(artifact_dir: Path) -> None:
    """Verify hashes and recompute the canonical table from the frozen inputs."""

    artifact_dir = Path(artifact_dir)
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported adjudication artifact schema")
    for filename, expected in manifest.get("outputs", {}).items():
        if file_sha256(artifact_dir / filename) != expected:
            raise ValueError(f"Adjudication output hash mismatch: {filename}")
    input_frames: dict[str, pd.DataFrame] = {}
    for name in ("facts", "proofs", "judgments"):
        record = manifest.get("inputs", {}).get(name) or {}
        path = Path(record.get("path") or "")
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise ValueError(f"Adjudication input hash mismatch: {name}")
        input_frames[name] = pd.read_parquet(path)
    expected = build_adjudications(
        input_frames["facts"],
        input_frames["proofs"],
        input_frames["judgments"],
    ).reset_index(drop=True)
    observed = pd.read_parquet(
        artifact_dir / "adjudications.parquet"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected, check_dtype=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path)
    parser.add_argument("--proofs", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--validate-artifact",
        type=Path,
        help="Validate an existing artifact instead of building one",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_artifact is not None:
        validate_artifact(args.validate_artifact)
        print(args.validate_artifact)
        return
    missing = [
        name
        for name in ("facts", "proofs", "judgments", "output_root")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            "Building requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    print(
        build_artifact(
            facts_path=args.facts,
            proofs_path=args.proofs,
            judgments_path=args.judgments,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
