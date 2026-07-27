#!/usr/bin/env python3
"""Build the frozen V4.3 adjudication queue for unresolved hard cases."""

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
import unicodedata

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.3-hard-label-queue-1"
EXPECTED_UNRESOLVED_COUNT = 542

GENERIC_NAME_TOKENS = {
    "association",
    "centre",
    "cie",
    "compagnie",
    "de",
    "des",
    "du",
    "et",
    "ets",
    "france",
    "groupe",
    "la",
    "le",
    "les",
    "sa",
    "sarl",
    "sas",
    "service",
    "societe",
}

ENTITY_TYPES = {
    "PUBLIC_ADMIN": {
        "mairie",
        "commune",
        "municipal",
        "municipale",
        "prefecture",
        "departement",
        "region",
    },
    "EDUCATION": {
        "ecol",
        "ecole",
        "college",
        "lycee",
        "universite",
        "creche",
        "mate",
        "maternelle",
        "scolaire",
    },
    "CULTURE": {
        "mediatheque",
        "bibliotheque",
        "musee",
        "theatre",
        "culturel",
    },
    "HEALTH_CARE": {
        "hopital",
        "clinique",
        "ehpad",
        "pharmacie",
        "medical",
        "medico",
    },
    "RELIGIOUS": {
        "eglise",
        "paroisse",
        "diocese",
        "culte",
    },
    "BEAUTY": {
        "beaute",
        "coiffure",
        "esthetique",
        "miroir",
    },
    "PARENT_ASSOCIATION": {
        "apel",
        "parents",
        "eleves",
    },
}


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_words(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKD", _text(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.findall(r"[a-z0-9]+", text.lower())


def informative_tokens(value: Any) -> set[str]:
    return {
        token
        for token in normalize_words(value)
        if len(token) > 1 and token not in GENERIC_NAME_TOKENS
    }


def token_overlap(left: Any, right: Any) -> float:
    left_compact = {
        "".join(normalize_words(variant))
        for variant in _text(left).split("|")
        if normalize_words(variant)
    }
    right_compact = {
        "".join(normalize_words(variant))
        for variant in _text(right).split("|")
        if normalize_words(variant)
    }
    if left_compact & right_compact:
        return 1.0
    left_tokens = informative_tokens(left)
    right_tokens = informative_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(
        len(left_tokens),
        len(right_tokens),
    )


def address_overlap(left: Any, right: Any) -> float:
    left_tokens = set(normalize_words(left))
    right_tokens = set(normalize_words(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def entity_types(value: Any) -> set[str]:
    tokens = set(normalize_words(value))
    return {
        entity_type
        for entity_type, markers in ENTITY_TYPES.items()
        if tokens & markers
    }


def parse_candidate_names(value: Any) -> str:
    try:
        payload = json.loads(_text(value) or "[]")
    except json.JSONDecodeError:
        return _text(value)
    if not isinstance(payload, list):
        return _text(value)
    return " | ".join(
        _text(item.get("text"))
        for item in payload
        if isinstance(item, dict) and _text(item.get("text"))
    )


def risk_signals(
    *,
    crm_name: str,
    crm_address: str,
    predicted_name: str,
    predicted_address: str,
    input_siret_state: str,
    input_siret: str,
    predicted_siret: str,
) -> dict[str, Any]:
    name_score = token_overlap(crm_name, predicted_name)
    address_score = address_overlap(crm_address, predicted_address)
    crm_types = entity_types(crm_name)
    predicted_types = entity_types(predicted_name)
    type_conflict = bool(
        crm_types
        and predicted_types
        and crm_types.isdisjoint(predicted_types)
    )
    address_only = address_score >= 0.8 and name_score < 0.25
    active_input_disagreement = bool(
        input_siret_state == "ACTIVE"
        and input_siret
        and predicted_siret
        and input_siret != predicted_siret
    )
    signals: list[str] = []
    if type_conflict:
        signals.append("ENTITY_TYPE_CONFLICT")
    if address_only:
        signals.append("ADDRESS_ONLY_WEAK_NAME")
    if active_input_disagreement:
        signals.append("PREDICTION_DIFFERS_ACTIVE_INPUT")
    if name_score == 0.0 and predicted_name:
        signals.append("ZERO_INFORMATIVE_NAME_OVERLAP")
    return {
        "name_token_overlap": name_score,
        "crm_entity_types_json": json.dumps(sorted(crm_types)),
        "predicted_entity_types_json": json.dumps(sorted(predicted_types)),
        "address_token_coverage": address_score,
        "entity_type_conflict": type_conflict,
        "address_only_weak_name": address_only,
        "prediction_differs_active_input": active_input_disagreement,
        "risk_signals_json": json.dumps(signals),
    }


def priority(
    *,
    decision: str,
    sampling_stratum: str,
    known_contradiction: bool,
    entity_type_conflict: bool,
    address_only_weak_name: bool,
    prediction_differs_active_input: bool,
) -> tuple[int, str]:
    if known_contradiction:
        return 100, "P0_KNOWN_PROVISIONAL_CONTRADICTION"
    if decision == "AUTO_MATCH" and entity_type_conflict:
        return 90, "P0_AUTO_ENTITY_TYPE_CONFLICT"
    if decision == "AUTO_MATCH" and address_only_weak_name:
        return 80, "P0_AUTO_ADDRESS_ONLY"
    if decision == "AUTO_MATCH" and prediction_differs_active_input:
        return 70, "P1_AUTO_ACTIVE_INPUT_DISAGREEMENT"
    if decision == "AUTO_MATCH":
        return 60, "P1_OTHER_AUTO_UNRESOLVED"
    if sampling_stratum == "REVIEW_NEAR_THRESHOLD":
        return 50, "P2_REVIEW_NEAR_THRESHOLD"
    if sampling_stratum == "NO_ACTIVE_CANDIDATE":
        return 40, "P2_NO_ACTIVE_CANDIDATE"
    return 30, "P3_OTHER_REVIEW"


def _candidate_evidence_json(
    rows: pd.DataFrame,
    *,
    limit: int = 25,
) -> str:
    if rows.empty:
        return "[]"
    ordered = rows.sort_values(
        ["evidence_source", "candidate_siret"],
        na_position="last",
    ).head(limit)
    fields = [
        "candidate_siret",
        "candidate_siren",
        "candidate_state",
        "evidence_source",
        "evidence_class",
        "name_evidence_class",
        "address_evidence_class",
        "name_jaro_max",
        "name_token_overlap_max",
        "addr_jaro",
        "postcode_match",
        "city_match",
        "candidate_name",
        "candidate_address",
    ]
    records = [
        {
            field: (
                None
                if pd.isna(value)
                else value.item()
                if hasattr(value, "item")
                else value
            )
            for field in fields
            if field in row
            for value in [row[field]]
        }
        for row in ordered.to_dict("records")
    ]
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def build_queue(
    *,
    blind: pd.DataFrame,
    adjudications: pd.DataFrame,
    case_evidence: pd.DataFrame,
    candidate_evidence: pd.DataFrame,
    registry: pd.DataFrame,
    top10: pd.DataFrame,
    known_contradictions: pd.DataFrame,
) -> pd.DataFrame:
    unresolved = adjudications.loc[
        adjudications["label_kind"].astype(str).eq("UNRESOLVED")
    ].copy()
    if len(unresolved) != EXPECTED_UNRESOLVED_COUNT:
        raise ValueError(
            f"Frozen unresolved population changed: {len(unresolved)} "
            f"!= {EXPECTED_UNRESOLVED_COUNT}"
        )
    top1 = top10.loc[top10["rank"].astype(int).eq(1)].copy()
    top1["predicted_candidate_name"] = top1["candidate_names_json"].map(
        parse_candidate_names
    )
    top1 = top1.rename(
        columns={
            "candidate_siret": "top1_siret",
            "candidate_siren": "top1_siren",
            "candidate_address": "top1_address",
            "etat_admin": "top1_state",
        }
    )
    registry_columns = [
        "audit_case_id",
        "decision",
        "confidence",
        "predicted_siret",
        "predicted_siren",
        "review_reason",
        "sampling_stratum",
        "input_siret",
        "input_siret_state",
    ]
    top1_columns = [
        "service_id",
        "top1_siret",
        "top1_siren",
        "top1_state",
        "predicted_candidate_name",
        "top1_address",
        "postcode",
        "city",
        "insee",
        "ranker_score",
        "retrieval_source",
    ]
    queue = unresolved[
        [
            "audit_case_id",
            "service_id",
            "adjudication_status",
            "rule_code",
            "evidence_refs",
        ]
    ].merge(
        blind,
        on=["audit_case_id", "service_id"],
        validate="one_to_one",
    ).merge(
        case_evidence,
        on=["audit_case_id", "service_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_evidence"),
    ).merge(
        registry[registry_columns],
        on="audit_case_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_registry"),
    ).merge(
        top1[top1_columns],
        on="service_id",
        how="left",
        validate="one_to_one",
    )
    if len(queue) != EXPECTED_UNRESOLVED_COUNT:
        raise ValueError("Hard-label queue join changed the frozen population")

    evidence_by_case = {
        str(case_id): _candidate_evidence_json(group)
        for case_id, group in candidate_evidence.groupby(
            "audit_case_id",
            sort=False,
        )
    }
    known_by_case = {
        str(row.audit_case_id): row
        for row in known_contradictions.itertuples(index=False)
    }
    records: list[dict[str, Any]] = []
    for raw in queue.to_dict("records"):
        case_id = str(raw["audit_case_id"])
        predicted_siret = _text(
            raw.get("top1_siret") or raw.get("predicted_siret")
        )
        signals = risk_signals(
            crm_name=_text(raw.get("SITE")),
            crm_address=_text(raw.get("SITE_CLI_ADRESSE")),
            predicted_name=_text(raw.get("predicted_candidate_name")),
            predicted_address=_text(raw.get("top1_address")),
            input_siret_state=_text(
                raw.get("input_siret_state_registry")
                or raw.get("input_siret_state")
            ),
            input_siret=_text(
                raw.get("input_siret_registry") or raw.get("input_siret")
            ),
            predicted_siret=predicted_siret,
        )
        known = known_by_case.get(case_id)
        priority_score, priority_reason = priority(
            decision=_text(raw.get("decision")),
            sampling_stratum=_text(raw.get("sampling_stratum")),
            known_contradiction=known is not None,
            entity_type_conflict=bool(signals["entity_type_conflict"]),
            address_only_weak_name=bool(signals["address_only_weak_name"]),
            prediction_differs_active_input=bool(
                signals["prediction_differs_active_input"]
            ),
        )
        output = dict(raw)
        output.update(signals)
        output["priority_score"] = priority_score
        output["priority_reason"] = priority_reason
        output["sirene_evidence_json"] = evidence_by_case.get(case_id, "[]")
        output["provisional_label"] = (
            "WRONG_TOP1" if known is not None else "UNRESOLVED"
        )
        output["provisional_reason"] = (
            _text(getattr(known, "contradiction_code", ""))
            if known is not None
            else "NO_TRAINING_GRADE_PROOF"
        )
        output["provisional_evidence_summary"] = (
            _text(getattr(known, "evidence_summary", ""))
            if known is not None
            else ""
        )
        output["adjudication_status_v43"] = "PROVISIONAL"
        output["training_eligible"] = False
        output["human_label"] = None
        output["human_ground_truth_siret"] = None
        output["human_validator"] = None
        output["human_evidence_refs"] = None
        records.append(output)
    return pd.DataFrame(records).sort_values(
        ["priority_score", "confidence", "audit_case_id"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def freeze_queue(
    *,
    blind_cases_path: Path,
    adjudications_path: Path,
    case_evidence_path: Path,
    candidate_evidence_path: Path,
    sample_registry_path: Path,
    top10_path: Path,
    contradictions_path: Path,
    output_root: Path,
) -> Path:
    input_paths = {
        "blind_cases": Path(blind_cases_path),
        "adjudications": Path(adjudications_path),
        "case_evidence": Path(case_evidence_path),
        "candidate_evidence": Path(candidate_evidence_path),
        "sample_registry": Path(sample_registry_path),
        "top10": Path(top10_path),
        "known_contradictions": Path(contradictions_path),
    }
    input_hashes = {
        name: file_sha256(path) for name, path in input_paths.items()
    }
    queue = build_queue(
        blind=pd.read_parquet(blind_cases_path),
        adjudications=pd.read_parquet(adjudications_path),
        case_evidence=pd.read_parquet(case_evidence_path),
        candidate_evidence=pd.read_parquet(candidate_evidence_path),
        registry=pd.read_parquet(sample_registry_path),
        top10=pd.read_parquet(top10_path),
        known_contradictions=pd.read_csv(contradictions_path),
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "input_hashes": input_hashes,
        "policy": "hard-label-priority-v3",
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.3 queue exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent)
    )
    try:
        queue_path = staging / "hard_label_queue.parquet"
        auto_path = staging / "auto_priority.csv"
        template_path = staging / "human_adjudication_template.csv"
        batch250_path = staging / "human_adjudication_batch250.csv"
        queue.to_parquet(queue_path, index=False)
        auto_columns = [
            "audit_case_id",
            "service_id",
            "priority_score",
            "priority_reason",
            "SITE",
            "SITE_CLI_ADRESSE",
            "CODE_POSTAL",
            "COMMUNE",
            "input_siret",
            "input_siret_state",
            "top1_siret",
            "predicted_candidate_name",
            "top1_address",
            "confidence",
            "risk_signals_json",
            "provisional_label",
            "provisional_reason",
            "provisional_evidence_summary",
            "sirene_evidence_json",
        ]
        queue.loc[queue["decision"].eq("AUTO_MATCH"), auto_columns].to_csv(
            auto_path,
            index=False,
        )
        template_columns = [
            "audit_case_id",
            "service_id",
            "priority_reason",
            "SITE",
            "SITE_CLI_ADRESSE",
            "CODE_POSTAL",
            "COMMUNE",
            "input_siret",
            "input_siret_state",
            "top1_siret",
            "predicted_candidate_name",
            "top1_address",
            "risk_signals_json",
            "sirene_evidence_json",
            "human_label",
            "human_ground_truth_siret",
            "human_validator",
            "human_evidence_refs",
        ]
        queue[template_columns].to_csv(template_path, index=False)
        queue.head(250)[template_columns].to_csv(batch250_path, index=False)
        priority_counts = {
            str(key): int(value)
            for key, value in queue["priority_reason"]
            .value_counts()
            .sort_index()
            .items()
        }
        summary = {
            "case_count": int(len(queue)),
            "decision_counts": {
                str(key): int(value)
                for key, value in queue["decision"].value_counts().items()
            },
            "priority_counts": priority_counts,
            "provisional_label_counts": {
                str(key): int(value)
                for key, value in queue["provisional_label"]
                .value_counts()
                .items()
            },
            "training_eligible_count": int(queue["training_eligible"].sum()),
            "random_population_count": int(
                queue["sampling_stratum"].eq("RANDOM_POPULATION").sum()
            ),
            "auto_count": int(queue["decision"].eq("AUTO_MATCH").sum()),
            "review_count": int(queue["decision"].eq("REVIEW").sum()),
            "adjudication_status": "PROVISIONAL",
            "verdict": "PIVOT_VALIDATION",
        }
        summary_path = staging / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        outputs = [
            queue_path,
            auto_path,
            template_path,
            batch250_path,
            summary_path,
        ]
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                name: {"path": str(input_paths[name]), "sha256": sha256}
                for name, sha256 in input_hashes.items()
            },
            "outputs": {
                path.name: file_sha256(path) for path in outputs
            },
            "model_outputs_used_for_priority_only": True,
            "training_eligible_count": int(queue["training_eligible"].sum()),
            "verdict": "PIVOT_VALIDATION",
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-cases", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--case-evidence", type=Path, required=True)
    parser.add_argument("--candidate-evidence", type=Path, required=True)
    parser.add_argument("--sample-registry", type=Path, required=True)
    parser.add_argument("--top10", type=Path, required=True)
    parser.add_argument("--contradictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        freeze_queue(
            blind_cases_path=args.blind_cases,
            adjudications_path=args.adjudications,
            case_evidence_path=args.case_evidence,
            candidate_evidence_path=args.candidate_evidence,
            sample_registry_path=args.sample_registry,
            top10_path=args.top10,
            contradictions_path=args.contradictions,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
