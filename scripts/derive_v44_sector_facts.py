#!/usr/bin/env python3
"""Derive label-free producer facts and an adjudication priority for V4.4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.4-sector-facts-1"
POLICY_VERSION = "producer-facts-and-review-priority-no-adjudication-v1"
EXPECTED_CASE_COUNT = 172


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _json(value: Any) -> Any:
    if isinstance(value, (Mapping, list)):
        return value
    try:
        return json.loads(_text(value) or "{}")
    except json.JSONDecodeError:
        return {}


def _siret(value: Any) -> str:
    digits = "".join(character for character in _text(value) if character.isdigit())
    return digits if len(digits) == 14 else ""


def _unique_text(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(_text(value) for value in values if _text(value)))


def _address_text(address: Mapping[str, Any]) -> str:
    explicit_lines = [
        address.get("ligneUne"),
        address.get("ligneDeux"),
        address.get("ligneTrois"),
        address.get("ligneQuatre"),
        address.get("ligneCinq"),
        address.get("ligneSix"),
    ]
    lines = _unique_text(explicit_lines)
    if lines:
        return " ".join(lines)
    return " ".join(
        _unique_text(
            [
                address.get("lieu"),
                address.get("adresse"),
                address.get("adresse_1"),
                address.get("adresse_2"),
                address.get("adresse_3"),
                address.get("codePostal") or address.get("code_postal"),
                address.get("ville")
                or address.get("commune")
                or address.get("nom_commune"),
                address.get("pays"),
            ]
        )
    )


def _finess_index(snapshot_path: Path, wanted: set[str]) -> dict[str, list[dict]]:
    """Index full EGE records, including their address and state."""

    with gzip.open(snapshot_path, "rt", encoding="utf-8") as stream:
        snapshot = json.load(stream)
    found: dict[str, list[dict]] = {}
    for pmej in snapshot.get("pmej") or []:
        if not isinstance(pmej, Mapping):
            continue
        for ege in pmej.get("ege") or []:
            if not isinstance(ege, Mapping):
                continue
            info = ege.get("informationsGeneralesEGE") or {}
            identifier = _text(info.get("numFinessEge"))
            if identifier not in wanted:
                continue
            found.setdefault(identifier, []).append(
                {
                    "info": dict(info),
                    "addresses": list(ege.get("adresse") or []),
                    "state": _text(ege.get("etatObjet")),
                    "date_last_update": _text(ege.get("dateDerniereMaj")),
                }
            )
    return found


def _extract_uai(payload: Any, identifier: str) -> dict[str, Any]:
    results = payload.get("results") or [] if isinstance(payload, Mapping) else []
    exact = [
        item
        for item in results
        if isinstance(item, Mapping)
        and _text(item.get("identifiant_de_l_etablissement")) == identifier
    ]
    return {
        "exact": exact,
        "sirets": [_siret(item.get("siren_siret")) for item in exact],
        "names": [item.get("nom_etablissement") for item in exact],
        "addresses": [
            _address_text(
                {
                    "adresse_1": item.get("adresse_1"),
                    "adresse_2": item.get("adresse_2"),
                    "adresse_3": item.get("adresse_3"),
                    "code_postal": item.get("code_postal"),
                    "nom_commune": item.get("nom_commune"),
                }
            )
            for item in exact
        ],
        "dates": [
            {
                "date_ouverture": _text(item.get("date_ouverture")),
                "date_maj_ligne": _text(item.get("date_maj_ligne")),
            }
            for item in exact
        ],
        "statuses": [item.get("etat") for item in exact],
    }


def _extract_bio(payload: Any, identifier: str) -> dict[str, Any]:
    if isinstance(payload, list):
        results = payload
    elif isinstance(payload, Mapping):
        results = payload.get("items") or []
    else:
        results = []
    exact = [
        item
        for item in results
        if isinstance(item, Mapping)
        and _text(item.get("numeroBio")) == identifier
    ]
    addresses = []
    dates = []
    statuses = []
    for item in exact:
        addresses.extend(
            _address_text(address)
            for address in item.get("adressesOperateurs") or []
            if isinstance(address, Mapping)
        )
        certificates = [
            certificate
            for certificate in item.get("certificats") or []
            if isinstance(certificate, Mapping)
        ]
        dates.append(
            {
                "date_maj": _text(item.get("dateMaj")),
                "date_premier_engagement": _text(
                    item.get("datePremierEngagement")
                ),
                "certificats": [
                    {
                        "date_engagement": _text(
                            certificate.get("dateEngagement")
                        ),
                        "date_arret": _text(certificate.get("dateArret")),
                        "date_suspension": _text(
                            certificate.get("dateSuspension")
                        ),
                    }
                    for certificate in certificates
                ],
            }
        )
        statuses.extend(
            certificate.get("etatCertification")
            for certificate in certificates
        )
    return {
        "exact": exact,
        "sirets": [_siret(item.get("siret")) for item in exact],
        "names": [
            name
            for item in exact
            for name in (
                item.get("raisonSociale"),
                item.get("denominationcourante"),
            )
        ],
        "addresses": addresses,
        "dates": dates,
        "statuses": statuses,
    }


def _extract_rge(payload: Any, identifier: str) -> dict[str, Any]:
    results = payload.get("results") or [] if isinstance(payload, Mapping) else []
    exact = [
        item
        for item in results
        if isinstance(item, Mapping)
        and _text(item.get("code_qualification")) == identifier
    ]
    return {
        "exact": exact,
        "sirets": [_siret(item.get("siret")) for item in exact],
        "names": [item.get("nom_entreprise") for item in exact],
        "addresses": [
            _address_text(
                {
                    "adresse": item.get("adresse"),
                    "code_postal": item.get("code_postal"),
                    "commune": item.get("commune"),
                }
            )
            for item in exact
        ],
        "dates": [
            {
                "date_debut": _text(item.get("lien_date_debut")),
                "date_fin": _text(item.get("lien_date_fin")),
            }
            for item in exact
        ],
        "statuses": [],
    }


def _extract_finess(
    identifier: str,
    finess_index: Mapping[str, list[dict]],
) -> dict[str, Any]:
    exact = list(finess_index.get(identifier) or [])
    return {
        "exact": exact,
        "sirets": [
            _siret(item.get("info", {}).get("siret")) for item in exact
        ],
        "names": [
            name
            for item in exact
            for name in (
                item.get("info", {}).get("nomEgeLong"),
                item.get("info", {}).get("nomEgeCourt"),
                item.get("info", {}).get("complementDenominationEg"),
            )
        ],
        "addresses": [
            _address_text(address)
            for item in exact
            for address in item.get("addresses") or []
            if isinstance(address, Mapping)
        ],
        "dates": [
            {
                "date_ouverture": _text(
                    item.get("info", {}).get("dateOuverture")
                ),
                "date_fermeture": _text(
                    item.get("info", {}).get("dateFermeture")
                ),
                "date_premiere_autorisation": _text(
                    item.get("info", {}).get("datePremiereAutorisation")
                ),
                "date_derniere_maj": _text(item.get("date_last_update")),
            }
            for item in exact
        ],
        "statuses": [item.get("state") for item in exact],
    }


def derive_producer_fact(
    observation: Mapping[str, Any],
    response: Mapping[str, Any],
    *,
    finess_index: Mapping[str, list[dict]] | None = None,
) -> dict[str, Any]:
    """Derive observable producer facts for one sector observation."""

    kind = _text(observation.get("identifier_kind"))
    identifier = _text(observation.get("identifier"))
    observed_siret = _siret(observation.get("observed_siret"))
    payload = _json(response.get("response_excerpt_json"))
    if kind == "UAI":
        extracted = _extract_uai(payload, identifier)
    elif kind == "BIO":
        extracted = _extract_bio(payload, identifier)
    elif kind == "RGE":
        extracted = _extract_rge(payload, identifier)
    elif kind == "FINESS":
        extracted = _extract_finess(identifier, finess_index or {})
    else:
        raise ValueError(f"Unsupported identifier kind: {kind}")

    sirets = _unique_text(_siret(value) for value in extracted["sirets"])
    identifier_returned = bool(extracted["exact"])
    explicit_siret_present = bool(sirets)
    observed_siret_returned = bool(
        observed_siret and observed_siret in set(sirets)
    )
    facts = []
    if identifier_returned:
        facts.append("PRODUCER_RETURNS_IDENTIFIER")
    else:
        facts.append("PRODUCER_DOES_NOT_RETURN_IDENTIFIER")
    if explicit_siret_present:
        facts.append("PRODUCER_RETURNS_EXPLICIT_SIRET")
    if observed_siret_returned:
        facts.append("PRODUCER_RETURNS_OBSERVED_SIRET")
    elif explicit_siret_present:
        facts.append("PRODUCER_SIRET_DIFFERS_FROM_OBSERVED")

    return {
        **dict(observation),
        "producer": _text(response.get("producer")),
        "producer_http_ok": int(response.get("http_status") or 0) == 200,
        "producer_identifier_returned": identifier_returned,
        "producer_result_count": int(response.get("result_count") or 0),
        "producer_explicit_siret_present": explicit_siret_present,
        "producer_sirets_json": json.dumps(sirets, ensure_ascii=False),
        "producer_siret_count": len(sirets),
        "producer_observed_siret_returned": observed_siret_returned,
        "producer_names_json": json.dumps(
            _unique_text(extracted["names"]), ensure_ascii=False
        ),
        "producer_addresses_json": json.dumps(
            _unique_text(extracted["addresses"]), ensure_ascii=False
        ),
        "producer_dates_json": json.dumps(
            extracted["dates"], ensure_ascii=False, sort_keys=True
        ),
        "producer_statuses_json": json.dumps(
            _unique_text(extracted["statuses"]), ensure_ascii=False
        ),
        "producer_data_date": _text(response.get("producer_data_date")),
        "producer_collected_at": _text(response.get("collected_at")),
        "producer_response_url": _text(response.get("response_url")),
        "producer_raw_response_path": _text(response.get("raw_response_path")),
        "producer_raw_response_sha256": _text(
            response.get("raw_response_sha256")
        ),
        "producer_fact_codes_json": json.dumps(facts, ensure_ascii=False),
        "correctness_conclusion": "NOT_DERIVED",
        "training_eligible": False,
    }


def _relation(observed_siret: str, top1: str, input_siret: str) -> str:
    is_top1 = bool(observed_siret and observed_siret == top1)
    is_input = bool(observed_siret and observed_siret == input_siret)
    if is_top1 and is_input:
        return "TOP1_AND_INPUT"
    if is_top1:
        return "TOP1_ONLY"
    if is_input:
        return "INPUT_ONLY"
    return "OTHER"


def _priority(case_facts: pd.DataFrame) -> tuple[int, str, list[str]]:
    if case_facts.empty:
        return 5, "P5_NO_SECTOR_OBSERVATION", ["NO_SECTOR_OBSERVATION"]
    reasons: list[str] = []
    explicit_conflict = (
        case_facts["producer_identifier_returned"]
        & case_facts["producer_explicit_siret_present"]
        & ~case_facts["producer_observed_siret_returned"]
    )
    if explicit_conflict.any():
        reasons.append("PRODUCER_SIRET_DIFFERS_FROM_OBSERVED")
    non_top1 = case_facts["observed_siret_relation"].isin(
        ["INPUT_ONLY", "OTHER"]
    )
    if non_top1.any():
        reasons.append("SECTOR_OBSERVATION_NOT_ATTACHED_TO_TOP1")
    unresolved = ~case_facts["producer_identifier_returned"]
    if unresolved.any():
        reasons.append("PRODUCER_DOES_NOT_RETURN_IDENTIFIER")
    if explicit_conflict.any():
        return 1, "P1_PRODUCER_SIRET_CONFLICT", reasons
    if non_top1.any():
        return 2, "P2_NON_TOP1_SECTOR_OBSERVATION", reasons
    if unresolved.any():
        return 3, "P3_PRODUCER_IDENTIFIER_UNRESOLVED", reasons
    reasons.append("SECTOR_OBSERVATION_ATTACHED_TO_TOP1")
    return 4, "P4_TOP1_SECTOR_OBSERVATION", reasons


def build_adjudication_priority(
    evidence_facts: pd.DataFrame,
    producer_facts: pd.DataFrame,
) -> pd.DataFrame:
    """Join all 172 cases and append a transparent review priority."""

    if evidence_facts["audit_case_id"].duplicated().any():
        raise ValueError("evidence_facts audit_case_id must be unique")
    rows = []
    for _, case in evidence_facts.sort_values("audit_case_id").iterrows():
        audit_case_id = _text(case["audit_case_id"])
        subset = producer_facts.loc[
            producer_facts["audit_case_id"].astype(str).eq(audit_case_id)
        ].copy()
        top1 = _siret(case.get("top1_siret"))
        input_siret = _siret(case.get("input_siret"))
        if not subset.empty:
            subset["observed_siret_relation"] = subset["observed_siret"].map(
                lambda value: _relation(_siret(value), top1, input_siret)
            )
        priority_rank, priority_code, priority_reasons = _priority(subset)
        relation_counts = {
            relation: int(
                subset["observed_siret_relation"].eq(relation).sum()
            )
            if not subset.empty
            else 0
            for relation in (
                "TOP1_AND_INPUT",
                "TOP1_ONLY",
                "INPUT_ONLY",
                "OTHER",
            )
        }
        kinds = sorted(
            set(subset["identifier_kind"].astype(str))
            if not subset.empty
            else set()
        )
        producer_names = sorted(
            set(subset["producer"].astype(str))
            if not subset.empty
            else set()
        )
        record = case.to_dict()
        record.update(
            {
                "sector_observation_count": int(len(subset)),
                "sector_identifier_kinds_json": json.dumps(kinds),
                "sector_producers_json": json.dumps(
                    producer_names, ensure_ascii=False
                ),
                "sector_relation_counts_json": json.dumps(
                    relation_counts, sort_keys=True
                ),
                "sector_top1_and_input_count": relation_counts[
                    "TOP1_AND_INPUT"
                ],
                "sector_top1_only_count": relation_counts["TOP1_ONLY"],
                "sector_input_only_count": relation_counts["INPUT_ONLY"],
                "sector_other_count": relation_counts["OTHER"],
                "sector_identifier_returned_count": int(
                    subset["producer_identifier_returned"].sum()
                )
                if not subset.empty
                else 0,
                "sector_observed_siret_returned_count": int(
                    subset["producer_observed_siret_returned"].sum()
                )
                if not subset.empty
                else 0,
                "sector_explicit_siret_conflict_count": int(
                    (
                        subset["producer_identifier_returned"]
                        & subset["producer_explicit_siret_present"]
                        & ~subset["producer_observed_siret_returned"]
                    ).sum()
                )
                if not subset.empty
                else 0,
                "adjudication_priority_rank": priority_rank,
                "adjudication_priority_code": priority_code,
                "adjudication_reason_codes_json": json.dumps(
                    priority_reasons, ensure_ascii=False
                ),
                "sector_evidence_correctness_conclusion": "NOT_DERIVED",
                "sector_evidence_training_eligible": False,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def derive(
    *,
    sector_artifact: Path,
    evidence_facts_path: Path,
    output_root: Path,
) -> Path:
    """Build an immutable producer-facts and review-priority artifact."""

    sector_artifact = Path(sector_artifact)
    observations_path = sector_artifact / "sector_identifier_observations.parquet"
    responses_path = sector_artifact / "producer_responses.parquet"
    sector_manifest_path = sector_artifact / "manifest.json"
    inputs = {
        "sector_manifest": file_sha256(sector_manifest_path),
        "observations": file_sha256(observations_path),
        "responses": file_sha256(responses_path),
        "evidence_facts": file_sha256(evidence_facts_path),
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "inputs": inputs,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.4 sector facts exist: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent)
    )
    try:
        observations = pd.read_parquet(observations_path)
        responses = pd.read_parquet(responses_path)
        evidence_facts = pd.read_parquet(evidence_facts_path)
        if len(evidence_facts) != EXPECTED_CASE_COUNT:
            raise ValueError(
                f"Frozen evidence fact population changed: "
                f"{len(evidence_facts)} != {EXPECTED_CASE_COUNT}"
            )
        merged = observations.merge(
            responses,
            left_on="producer_request_key",
            right_on="request_key",
            how="left",
            validate="many_to_one",
            suffixes=("", "_response"),
        )
        if merged["request_key"].isna().any():
            raise ValueError("At least one observation lacks a producer response")
        finess_ids = set(
            observations.loc[
                observations["identifier_kind"].eq("FINESS"), "identifier"
            ].astype(str)
        )
        finess_index: dict[str, list[dict]] = {}
        if finess_ids:
            raw_paths = responses.loc[
                responses["identifier_kind"].eq("FINESS"),
                "raw_response_path",
            ].drop_duplicates()
            if len(raw_paths) != 1:
                raise ValueError("Expected one shared FINESS snapshot")
            finess_index = _finess_index(
                sector_artifact / str(raw_paths.iloc[0]), finess_ids
            )
        producer_rows = []
        for _, row in merged.iterrows():
            observation = {
                column: row[column] for column in observations.columns
            }
            response = {
                column: row[column]
                for column in responses.columns
                if column in row.index
            }
            producer_rows.append(
                derive_producer_fact(
                    observation,
                    response,
                    finess_index=finess_index,
                )
            )
        producer_facts = pd.DataFrame(producer_rows).sort_values(
            ["audit_case_id", "identifier_kind", "identifier"]
        )
        producer_facts_path = staging / "producer_observation_facts.parquet"
        producer_facts.to_parquet(producer_facts_path, index=False)

        priority = build_adjudication_priority(evidence_facts, producer_facts)
        priority_path = staging / "adjudication_priority.parquet"
        priority.to_parquet(priority_path, index=False)
        summary = {
            "case_count": int(len(priority)),
            "case_with_sector_observation_count": int(
                priority["sector_observation_count"].gt(0).sum()
            ),
            "producer_observation_fact_count": int(len(producer_facts)),
            "producer_identifier_returned_count": int(
                producer_facts["producer_identifier_returned"].sum()
            ),
            "producer_observed_siret_returned_count": int(
                producer_facts["producer_observed_siret_returned"].sum()
            ),
            "producer_explicit_siret_conflict_count": int(
                (
                    producer_facts["producer_identifier_returned"]
                    & producer_facts["producer_explicit_siret_present"]
                    & ~producer_facts["producer_observed_siret_returned"]
                ).sum()
            ),
            "priority_counts": {
                str(key): int(value)
                for key, value in priority[
                    "adjudication_priority_code"
                ].value_counts().items()
            },
            "relation_observation_counts": {
                "TOP1_AND_INPUT": int(
                    priority["sector_top1_and_input_count"].sum()
                ),
                "TOP1_ONLY": int(priority["sector_top1_only_count"].sum()),
                "INPUT_ONLY": int(priority["sector_input_only_count"].sum()),
                "OTHER": int(priority["sector_other_count"].sum()),
            },
            "correctness_conclusions": {"NOT_DERIVED": int(len(priority))},
            "training_eligible_count": 0,
            "verdict": "FACTS_AND_REVIEW_PRIORITY_ONLY",
        }
        summary_path = staging / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_paths": {
                "sector_artifact": str(sector_artifact),
                "evidence_facts": str(evidence_facts_path),
            },
            "outputs": {
                path.name: file_sha256(path)
                for path in (
                    producer_facts_path,
                    priority_path,
                    summary_path,
                )
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
    parser.add_argument("--sector-artifact", type=Path, required=True)
    parser.add_argument("--evidence-facts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = derive(
        sector_artifact=args.sector_artifact,
        evidence_facts_path=args.evidence_facts,
        output_root=args.output_root,
    )
    print(target)


if __name__ == "__main__":
    main()
