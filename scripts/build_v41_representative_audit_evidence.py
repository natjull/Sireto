#!/usr/bin/env python3
"""Build blind SIRENE evidence and conservative provisional labels for V4.1."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_benchmark_v3_evidence import classify_direct_evidence  # noqa: E402
from scripts.build_benchmark_v4_current_snapshot import (  # noqa: E402
    _load_partition,
    _planned_partition_key,
    build_active_partition_index,
    find_direct_active_candidates,
)
from src.xgb_matcher.features import (  # noqa: E402
    make_features_from_preprocessed,
    preprocess_crm_row,
)
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.1-representative-audit-evidence-1"
POLICY_VERSION = "blind-sirene-provisional-v1"
PROVISIONAL_VALIDATOR = "DETERMINISTIC_V4_1_AUDIT_V1"


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _identifier(value: Any, width: int) -> str | None:
    digits = "".join(character for character in _text(value) if character.isdigit())
    return digits.zfill(width) if len(digits) == width else None


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _crm_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "query_id": str(row["audit_case_id"]),
        "split": "blind_audit",
        "crm_name": _text(row.get("SITE")),
        "crm_address": _text(row.get("SITE_CLI_ADRESSE")),
        "crm_city": _first_text(
            row.get("COMMUNE"),
            row.get("SITE_CLI_COMMUNE"),
        ),
        "postcode": _text(row.get("CODE_POSTAL")),
        "insee": _text(row.get("CODE_INSEE")),
    }


def _snapshot_candidate(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "siret": _identifier(raw.get("siret"), 14),
        "siren": _identifier(raw.get("siren"), 9),
        "denomination": _text(raw.get("denomination")),
        "enseigne1": _text(raw.get("enseigne1")),
        "enseigne2": _text(raw.get("enseigne2")),
        "enseigne3": _text(raw.get("enseigne3")),
        "denomination_ul": _text(raw.get("denomination_ul")),
        "denomination_usuelle_ul": _text(
            raw.get("denomination_usuelle_ul")
        ),
        "sigle_ul": _text(raw.get("sigle_ul")),
        "nom_ul": _text(raw.get("nom_ul")),
        "prenom_usuel_ul": _text(raw.get("prenom_usuel_ul")),
        "numeroVoie": _text(raw.get("numeroVoie")),
        "typeVoie": _text(raw.get("typeVoie")),
        "libelleVoie": _text(raw.get("libelleVoie")),
        "complementAdresse": _text(raw.get("complementAdresse")),
        "postcode": _text(raw.get("postcode")),
        "city": _text(raw.get("city")),
        "insee": _text(raw.get("insee")),
        "etat_admin": _text(raw.get("etat_admin")).upper(),
        "is_siege": bool(raw.get("is_siege") or False),
    }


def load_relevant_lineage(
    *,
    blind_cases: pd.DataFrame,
    establishment_snapshot: Path,
    legal_unit_snapshot: Path,
    temp_dir: Path,
) -> pd.DataFrame:
    """Scan the full snapshots once for the input SIREN lineages."""

    sirens = sorted(
        {
            siren
            for value in blind_cases["input_siren"]
            if (siren := _identifier(value, 9))
        }
    )
    connection = duckdb.connect()
    connection.execute("SET memory_limit='8GB'")
    connection.execute(f"SET temp_directory='{str(temp_dir).replace(chr(39), chr(39)*2)}'")
    connection.register("audit_sirens", pd.DataFrame({"siren": sirens}))
    query = """
        WITH relevant_establishments AS (
            SELECT
                e.siren AS siren,
                e.siret AS siret,
                e.etatAdministratifEtablissement AS etat_admin,
                e.denominationUsuelleEtablissement AS denomination,
                e.enseigne1Etablissement AS enseigne1,
                e.enseigne2Etablissement AS enseigne2,
                e.enseigne3Etablissement AS enseigne3,
                CAST(e.numeroVoieEtablissement AS VARCHAR) AS numeroVoie,
                e.typeVoieEtablissement AS typeVoie,
                e.libelleVoieEtablissement AS libelleVoie,
                e.complementAdresseEtablissement AS complementAdresse,
                CAST(e.codePostalEtablissement AS VARCHAR) AS postcode,
                e.libelleCommuneEtablissement AS city,
                CAST(e.codeCommuneEtablissement AS VARCHAR) AS insee,
                e.etablissementSiege AS is_siege
            FROM read_parquet(?) e
            JOIN audit_sirens s
              ON e.siren = s.siren
        ),
        relevant_legal_units AS (
            SELECT
                u.siren AS siren,
                u.denominationUniteLegale AS denomination_ul,
                u.denominationUsuelle1UniteLegale AS denomination_usuelle_ul,
                u.sigleUniteLegale AS sigle_ul,
                u.nomUniteLegale AS nom_ul,
                u.prenomUsuelUniteLegale AS prenom_usuel_ul
            FROM read_parquet(?) u
            JOIN audit_sirens s
              ON u.siren = s.siren
        )
        SELECT e.*, u.* EXCLUDE (siren)
        FROM relevant_establishments e
        LEFT JOIN relevant_legal_units u USING (siren)
        ORDER BY e.siren, e.siret
    """
    try:
        return connection.execute(
            query,
            [
                Path(establishment_snapshot).resolve().as_posix(),
                Path(legal_unit_snapshot).resolve().as_posix(),
            ],
        ).df()
    finally:
        connection.close()


def _candidate_feature_evidence(
    *,
    case: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    crm = preprocess_crm_row(_crm_row(case))
    features = make_features_from_preprocessed(
        crm,
        candidate,
        skip_semantic=True,
    )
    evidence_class, name_class, address_class = classify_direct_evidence(
        features,
        exact_address_hash=False,
        crm_number_present=bool(crm.get("crm_street_num")),
        candidate_number_present=bool(candidate.get("numeroVoie")),
    )
    return {
        "audit_case_id": str(case["audit_case_id"]),
        "service_id": str(case["service_id"]),
        "candidate_siret": _identifier(candidate.get("siret"), 14),
        "candidate_siren": _identifier(candidate.get("siren"), 9),
        "candidate_state": _text(candidate.get("etat_admin")).upper(),
        "evidence_source": source,
        "evidence_class": evidence_class,
        "name_evidence_class": name_class,
        "address_evidence_class": address_class,
        "name_jaro_max": float(features["name_jaro_max"]),
        "name_token_overlap_max": float(features["name_token_overlap_max"]),
        "name_norm_exact": float(features["name_norm_exact"]),
        "addr_jaro": float(features["addr_jaro"]),
        "street_name_jaro": float(features["street_name_jaro"]),
        "street_number_match": float(features["street_number_match"]),
        "postcode_match": float(features["postcode_match"]),
        "city_match": float(features["city_match"]),
        "candidate_name": " | ".join(
            value
            for value in (
                _text(candidate.get("denomination")),
                _text(candidate.get("enseigne1")),
                _text(candidate.get("denomination_ul")),
                _text(candidate.get("nom_ul")),
            )
            if value
        ),
        "candidate_address": " ".join(
            value
            for value in (
                _text(candidate.get("numeroVoie")),
                _text(candidate.get("typeVoie")),
                _text(candidate.get("libelleVoie")),
                _text(candidate.get("postcode")),
                _text(candidate.get("city")),
            )
            if value
        ),
    }


def provisional_adjudication(
    *,
    audit_case_id: str,
    direct_sirets: Iterable[str],
    lineage_strong_sirets: Iterable[str],
) -> dict[str, Any]:
    """Apply the conservative preregistered evidence hierarchy."""

    direct = sorted(set(direct_sirets))
    lineage = sorted(set(lineage_strong_sirets))
    if len(direct) == 1:
        label, siret, rule = "MATCH_EXACT", direct[0], "UNIQUE_DIRECT_GEO"
    elif len(direct) > 1:
        label, siret, rule = "AMBIGUOUS", None, "MULTIPLE_DIRECT_GEO"
    elif len(lineage) == 1:
        label, siret, rule = (
            "MATCH_EXACT",
            lineage[0],
            "UNIQUE_STRONG_ACTIVE_LINEAGE",
        )
    elif len(lineage) > 1:
        label, siret, rule = (
            "AMBIGUOUS",
            None,
            "MULTIPLE_STRONG_ACTIVE_LINEAGE",
        )
    else:
        label, siret, rule = "UNRESOLVED", None, "NO_CONCLUSIVE_LOCAL_EVIDENCE"
    return {
        "audit_case_id": str(audit_case_id),
        "label_kind": label,
        "ground_truth_siret": siret,
        "ground_truth_siren": siret[:9] if siret else None,
        "adjudication_status": "PROVISIONAL",
        "validator": PROVISIONAL_VALIDATOR,
        "rule_code": rule,
    }


def build_evidence(
    *,
    blind_cases: pd.DataFrame,
    partitions_dir: Path,
    establishment_snapshot: Path,
    legal_unit_snapshot: Path,
    temp_dir: Path,
    sirene_snapshot_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    store = PartitionedCandidateStore(partitions_dir)
    cases = blind_cases.copy()
    case_records = [_crm_row(row) for row in cases.to_dict("records")]
    cases["partition_key"] = [
        _planned_partition_key(row, store) for row in case_records
    ]

    direct_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for partition_key, group in cases.groupby("partition_key", sort=True):
        rows = _load_partition(str(partition_key), store)
        index = build_active_partition_index(rows)
        for case in group.to_dict("records"):
            evidence = find_direct_active_candidates(
                _crm_row(case),
                index,
                partition_key=str(partition_key),
            )
            direct_by_case[str(case["audit_case_id"])].extend(evidence)

    lineage = load_relevant_lineage(
        blind_cases=cases,
        establishment_snapshot=establishment_snapshot,
        legal_unit_snapshot=legal_unit_snapshot,
        temp_dir=temp_dir,
    )
    lineage_by_siren: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in lineage.to_dict("records"):
        candidate = _snapshot_candidate(raw)
        if candidate["siren"]:
            lineage_by_siren[candidate["siren"]].append(candidate)

    candidate_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    adjudications: list[dict[str, Any]] = []
    for case in cases.sort_values("audit_case_id").to_dict("records"):
        case_id = str(case["audit_case_id"])
        direct = direct_by_case.get(case_id, [])
        direct_sirets = sorted(
            {
                str(row["candidate_siret"])
                for row in direct
                if row.get("candidate_siret")
            }
        )
        evidence_by_siret: dict[str, dict[str, Any]] = {}
        for raw in direct:
            siret = str(raw["candidate_siret"])
            evidence_by_siret[siret] = {
                "audit_case_id": case_id,
                "service_id": str(case["service_id"]),
                **raw,
                "evidence_source": "DIRECT_GEO",
                "evidence_class": raw["direct_evidence_class"],
            }

        input_siret = _identifier(case.get("input_siret"), 14)
        input_siren = _identifier(case.get("input_siren"), 9)
        lineage_strong: set[str] = set()
        lineage_count = 0
        active_lineage_count = 0
        for candidate in lineage_by_siren.get(input_siren or "", []):
            lineage_count += 1
            if candidate["etat_admin"] != "A":
                continue
            active_lineage_count += 1
            source = (
                "INPUT_ACTIVE"
                if candidate["siret"] == input_siret
                else "ACTIVE_SIBLING"
            )
            evidence = _candidate_feature_evidence(
                case=case,
                candidate=candidate,
                source=source,
            )
            if evidence["evidence_class"] == "NAME_AND_ADDRESS":
                lineage_strong.add(str(candidate["siret"]))
            existing = evidence_by_siret.get(str(candidate["siret"]))
            if existing is None or source == "INPUT_ACTIVE":
                evidence_by_siret[str(candidate["siret"])] = evidence

        candidate_rows.extend(evidence_by_siret.values())
        adjudication = provisional_adjudication(
            audit_case_id=case_id,
            direct_sirets=direct_sirets,
            lineage_strong_sirets=lineage_strong,
        )
        adjudication["service_id"] = str(case["service_id"])
        adjudication["evidence_refs"] = json.dumps(
            sorted(evidence_by_siret),
            separators=(",", ":"),
        )
        adjudication["sirene_snapshot_id"] = sirene_snapshot_id
        adjudication["reference_date"] = datetime.now(timezone.utc).date().isoformat()
        adjudications.append(adjudication)
        case_rows.append(
            {
                "audit_case_id": case_id,
                "service_id": str(case["service_id"]),
                "partition_key": str(case["partition_key"]),
                "input_siret": input_siret,
                "input_siren": input_siren,
                "input_siret_state": str(case["input_siret_state"]),
                "direct_active_candidate_count": len(direct_sirets),
                "direct_active_sirets_json": json.dumps(
                    direct_sirets,
                    separators=(",", ":"),
                ),
                "lineage_establishment_count": lineage_count,
                "active_lineage_count": active_lineage_count,
                "strong_active_lineage_count": len(lineage_strong),
                "strong_active_lineage_sirets_json": json.dumps(
                    sorted(lineage_strong),
                    separators=(",", ":"),
                ),
            }
        )
    return (
        pd.DataFrame(case_rows),
        pd.DataFrame(candidate_rows),
        pd.DataFrame(adjudications),
    )


def freeze_evidence(
    *,
    sample_dir: Path,
    partitions_dir: Path,
    establishment_snapshot: Path,
    legal_unit_snapshot: Path,
    output_root: Path,
    temp_dir: Path,
) -> Path:
    sample_dir = Path(sample_dir)
    sample_manifest_path = sample_dir / "manifest.json"
    sample_manifest = json.loads(sample_manifest_path.read_text(encoding="utf-8"))
    if sample_manifest.get("schema_version") != (
        "sireto-v4.1-representative-audit-sample-1"
    ):
        raise ValueError("Unsupported representative sample manifest")
    blind_path = sample_dir / "blind_cases.parquet"
    if sample_manifest["outputs"].get("blind_cases.parquet") != file_sha256(
        blind_path
    ):
        raise ValueError("Blind-case hash mismatch")
    blind = pd.read_parquet(blind_path)
    forbidden = {
        "decision",
        "confidence",
        "predicted_siret",
        "review_reason",
        "sampling_stratum",
    }
    if forbidden & set(blind):
        raise ValueError("Blind evidence input leaks model outputs")

    establishment_snapshot_sha256 = file_sha256(establishment_snapshot)
    legal_unit_snapshot_sha256 = file_sha256(legal_unit_snapshot)
    cases, candidates, adjudications = build_evidence(
        blind_cases=blind,
        partitions_dir=partitions_dir,
        establishment_snapshot=establishment_snapshot,
        legal_unit_snapshot=legal_unit_snapshot,
        temp_dir=temp_dir,
        sirene_snapshot_id=establishment_snapshot_sha256[:16],
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "sample_manifest_sha256": file_sha256(sample_manifest_path),
        "blind_cases_sha256": file_sha256(blind_path),
        "partition_count_manifests": {
            name: file_sha256(Path(partitions_dir) / "manifest" / name)
            for name in ("insee_counts.parquet", "postcode_counts.parquet")
        },
        "establishment_snapshot_sha256": establishment_snapshot_sha256,
        "legal_unit_snapshot_sha256": legal_unit_snapshot_sha256,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable evidence build already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent)
    )
    try:
        paths = {
            "case_evidence.parquet": cases,
            "candidate_evidence.parquet": candidates,
            "provisional_adjudications.parquet": adjudications,
        }
        for name, frame in paths.items():
            frame.to_parquet(staging / name, index=False)
        summary = {
            "case_count": int(len(cases)),
            "candidate_evidence_count": int(len(candidates)),
            "label_counts": {
                str(key): int(value)
                for key, value in adjudications["label_kind"]
                .value_counts()
                .sort_index()
                .items()
            },
            "rule_counts": {
                str(key): int(value)
                for key, value in adjudications["rule_code"]
                .value_counts()
                .sort_index()
                .items()
            },
            "validation_status": "PROVISIONAL",
            "model_outputs_read": False,
            "precision_claim_allowed": False,
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        output_names = [*paths, "summary.json"]
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_outputs_read": False,
            "adjudication_status": "PROVISIONAL",
            "outputs": {
                name: file_sha256(staging / name) for name in output_names
            },
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
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--partitions-dir", type=Path, required=True)
    parser.add_argument("--establishment-snapshot", type=Path, required=True)
    parser.add_argument("--legal-unit-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        freeze_evidence(
            sample_dir=args.sample_dir,
            partitions_dir=args.partitions_dir,
            establishment_snapshot=args.establishment_snapshot,
            legal_unit_snapshot=args.legal_unit_snapshot,
            output_root=args.output_root,
            temp_dir=args.temp_dir,
        )
    )


if __name__ == "__main__":
    main()
