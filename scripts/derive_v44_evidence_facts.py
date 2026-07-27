#!/usr/bin/env python3
"""Derive conservative facts from the frozen V4.4 official evidence.

This script never adjudicates whether a model prediction is correct.  The
official search endpoint exposes several views of the same SIRENE-derived
source; agreement between those views is recorded as a fact, not counted as
independent corroboration.
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
from typing import Any, Mapping
import unicodedata
from urllib.parse import urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.4-evidence-facts-1"
POLICY_VERSION = "official-facts-no-adjudication-v1"
OFFICIAL_SOURCE_FAMILY = "SIRENE_DERIVED_RECHERCHE_ENTREPRISES_API"
EXPECTED_AUTO_COUNT = 172
EXPECTED_QUERY_KINDS = {"TOP1_SIRET", "INPUT_SIRET", "CRM_NAME_GEO"}


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _siret(value: Any) -> str | None:
    digits = "".join(character for character in _text(value) if character.isdigit())
    return digits if len(digits) == 14 else None


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(_text(value) or "{}")
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _normalized_name(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", _text(value).upper())
    ascii_text = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[A-Z0-9]+", ascii_text))


def _result_names(result: Mapping[str, Any]) -> list[str]:
    values: list[Any] = [
        result.get("nom_complet"),
        result.get("nom_raison_sociale"),
        result.get("sigle"),
    ]
    for establishment in result.get("matching_etablissements") or []:
        values.append(establishment.get("nom_commercial"))
        values.extend(establishment.get("liste_enseignes") or [])
    return list(dict.fromkeys(_text(value) for value in values if _text(value)))


def _payload_index(payload: Mapping[str, Any]) -> dict[str, Any]:
    sirets: set[str] = set()
    sirens: set[str] = set()
    state_by_siret: dict[str, str] = {}
    insee_by_siret: dict[str, str] = {}
    postcode_by_siret: dict[str, str] = {}
    names_by_siren: dict[str, list[str]] = {}
    root_rank_by_siren: dict[str, int] = {}
    results = payload.get("results") or []
    for rank, result in enumerate(results, start=1):
        if not isinstance(result, Mapping):
            continue
        siren = _text(result.get("siren"))
        if len(siren) == 9 and siren.isdigit():
            sirens.add(siren)
            root_rank_by_siren.setdefault(siren, rank)
            names_by_siren.setdefault(siren, [])
            names_by_siren[siren].extend(_result_names(result))
        for establishment in result.get("matching_etablissements") or []:
            if not isinstance(establishment, Mapping):
                continue
            siret = _siret(establishment.get("siret"))
            if siret is None:
                continue
            sirets.add(siret)
            sirens.add(siret[:9])
            state_by_siret[siret] = _text(
                establishment.get("etat_administratif")
            ).upper()
            insee_by_siret[siret] = _text(establishment.get("commune"))
            postcode_by_siret[siret] = _text(establishment.get("code_postal"))
    return {
        "sirets": sirets,
        "sirens": sirens,
        "state_by_siret": state_by_siret,
        "insee_by_siret": insee_by_siret,
        "postcode_by_siret": postcode_by_siret,
        "names_by_siren": {
            siren: list(dict.fromkeys(names))
            for siren, names in names_by_siren.items()
        },
        "root_rank_by_siren": root_rank_by_siren,
        "payload_result_count": len(results),
        "total_results": int(payload.get("total_results") or 0),
    }


def _direct_facts(
    row: pd.Series | None,
    expected_siret: str | None,
) -> dict[str, Any]:
    if row is None or expected_siret is None:
        return {
            "queried": False,
            "http_ok": False,
            "exact_siret_returned": False,
            "siren_returned": False,
            "state": "",
            "names": [],
            "insee": "",
            "postcode": "",
            "result_count": 0,
        }
    payload = _json(row["payload_json"])
    index = _payload_index(payload)
    siren = expected_siret[:9]
    return {
        "queried": True,
        "http_ok": int(row["http_status"]) == 200,
        "exact_siret_returned": expected_siret in index["sirets"],
        "siren_returned": siren in index["sirens"],
        "state": index["state_by_siret"].get(expected_siret, ""),
        "names": index["names_by_siren"].get(siren, []),
        "insee": index["insee_by_siret"].get(expected_siret, ""),
        "postcode": index["postcode_by_siret"].get(expected_siret, ""),
        "result_count": int(row["result_count"]),
    }


def _max_name_overlap(crm_name: str, official_names: list[str]) -> float:
    crm_tokens = set(_normalized_name(crm_name).split())
    if not crm_tokens:
        return 0.0
    return max(
        (
            len(crm_tokens & set(_normalized_name(name).split()))
            / len(crm_tokens)
            for name in official_names
        ),
        default=0.0,
    )


def derive_case_facts(
    case: pd.Series,
    evidence: pd.DataFrame,
) -> dict[str, Any]:
    """Return observable facts only; never return a correctness label."""

    by_kind = {
        str(row["query_kind"]): row
        for _, row in evidence.sort_values("query_kind").iterrows()
    }
    top1 = _siret(case.get("top1_siret") or case.get("predicted_siret"))
    input_siret = _siret(case.get("input_siret"))
    top1_direct = _direct_facts(by_kind.get("TOP1_SIRET"), top1)
    input_query_kind = "INPUT_SIRET"
    input_evidence_row = by_kind.get("INPUT_SIRET")
    if input_siret is not None and input_siret == top1 and input_evidence_row is None:
        # The collector intentionally deduplicates identical direct queries.
        # Reuse the returned fact without pretending that a second view exists.
        input_evidence_row = by_kind.get("TOP1_SIRET")
        input_query_kind = "TOP1_SIRET_SHARED"
    input_direct = _direct_facts(input_evidence_row, input_siret)

    name_geo_row = by_kind.get("CRM_NAME_GEO")
    name_geo_payload = (
        _json(name_geo_row["payload_json"]) if name_geo_row is not None else {}
    )
    name_geo_index = _payload_index(name_geo_payload)
    query_params = (
        _json(name_geo_row["query_params_json"])
        if name_geo_row is not None
        else {}
    )
    top1_siren = top1[:9] if top1 else None
    input_siren = input_siret[:9] if input_siret else None
    crm_name = _text(case.get("SITE"))
    normalized_official_names = {
        _normalized_name(name)
        for name in top1_direct["names"]
        if _normalized_name(name)
    }
    exact_name_match = (
        bool(_normalized_name(crm_name))
        and _normalized_name(crm_name) in normalized_official_names
    )
    top1_name_geo_siren_rank = (
        name_geo_index["root_rank_by_siren"].get(top1_siren)
        if top1_siren
        else None
    )
    fact_codes: list[str] = []
    if top1_direct["exact_siret_returned"]:
        fact_codes.append("TOP1_DIRECT_IDENTITY_RETURNED")
    if input_direct["exact_siret_returned"]:
        fact_codes.append("INPUT_DIRECT_IDENTITY_RETURNED")
    if top1 and top1 in name_geo_index["sirets"]:
        fact_codes.append("NAME_GEO_RETURNS_TOP1_EXACT")
    elif top1_siren and top1_siren in name_geo_index["sirens"]:
        fact_codes.append("NAME_GEO_RETURNS_TOP1_SIREN")
    if input_siret and input_siret in name_geo_index["sirets"]:
        fact_codes.append("NAME_GEO_RETURNS_INPUT_EXACT")
    elif input_siren and input_siren in name_geo_index["sirens"]:
        fact_codes.append("NAME_GEO_RETURNS_INPUT_SIREN")
    if top1 and input_siret and top1 == input_siret:
        fact_codes.append("INPUT_EQUALS_TOP1")
    elif top1_siren and input_siren and top1_siren == input_siren:
        fact_codes.append("INPUT_TOP1_SAME_SIREN")

    successful_views = sum(
        int(row["http_status"]) == 200
        for row in by_kind.values()
    )
    return {
        "audit_case_id": _text(case.get("audit_case_id")),
        "service_id": _text(case.get("service_id")),
        "decision": _text(case.get("decision")),
        "top1_siret": top1,
        "top1_siren": top1_siren,
        "input_siret": input_siret,
        "input_siren": input_siren,
        "input_siret_state_queue": _text(case.get("input_siret_state")),
        "input_equals_top1": bool(top1 and input_siret and top1 == input_siret),
        "input_top1_same_siren": bool(
            top1_siren and input_siren and top1_siren == input_siren
        ),
        "top1_direct_query_present": top1_direct["queried"],
        "top1_direct_http_ok": top1_direct["http_ok"],
        "top1_direct_exact_siret_returned": top1_direct[
            "exact_siret_returned"
        ],
        "top1_direct_siren_returned": top1_direct["siren_returned"],
        "top1_direct_state": top1_direct["state"],
        "top1_direct_names_json": json.dumps(
            top1_direct["names"], ensure_ascii=False
        ),
        "top1_direct_insee": top1_direct["insee"],
        "top1_direct_postcode": top1_direct["postcode"],
        "top1_direct_insee_matches_crm": bool(
            top1_direct["insee"]
            and top1_direct["insee"] == _text(case.get("CODE_INSEE"))
        ),
        "top1_direct_postcode_matches_crm": bool(
            top1_direct["postcode"]
            and top1_direct["postcode"] == _text(case.get("CODE_POSTAL"))
        ),
        "input_direct_query_present": input_direct["queried"],
        "input_direct_evidence_query_kind": (
            input_query_kind if input_direct["queried"] else ""
        ),
        "input_direct_http_ok": input_direct["http_ok"],
        "input_direct_exact_siret_returned": input_direct[
            "exact_siret_returned"
        ],
        "input_direct_siren_returned": input_direct["siren_returned"],
        "input_direct_state": input_direct["state"],
        "input_direct_names_json": json.dumps(
            input_direct["names"], ensure_ascii=False
        ),
        "name_geo_query_present": name_geo_row is not None,
        "name_geo_http_ok": bool(
            name_geo_row is not None and int(name_geo_row["http_status"]) == 200
        ),
        "name_geo_filter_kind": (
            "INSEE"
            if query_params.get("code_commune")
            else ("POSTCODE" if query_params.get("code_postal") else "NONE")
        ),
        "name_geo_filter_value": _text(
            query_params.get("code_commune") or query_params.get("code_postal")
        ),
        "name_geo_total_results": name_geo_index["total_results"],
        "name_geo_payload_result_count": name_geo_index[
            "payload_result_count"
        ],
        "name_geo_payload_truncated": (
            name_geo_index["total_results"]
            > name_geo_index["payload_result_count"]
        ),
        "name_geo_top1_exact_hit": bool(
            top1 and top1 in name_geo_index["sirets"]
        ),
        "name_geo_top1_siren_hit": bool(
            top1_siren and top1_siren in name_geo_index["sirens"]
        ),
        "name_geo_top1_siren_rank": top1_name_geo_siren_rank,
        "name_geo_input_exact_hit": bool(
            input_siret and input_siret in name_geo_index["sirets"]
        ),
        "name_geo_input_siren_hit": bool(
            input_siren and input_siren in name_geo_index["sirens"]
        ),
        "crm_name_exact_top1_official_name": exact_name_match,
        "crm_name_token_coverage_top1_official_max": _max_name_overlap(
            crm_name, top1_direct["names"]
        ),
        "top1_input_share_official_name": bool(
            {
                _normalized_name(name)
                for name in top1_direct["names"]
                if _normalized_name(name)
            }
            & {
                _normalized_name(name)
                for name in input_direct["names"]
                if _normalized_name(name)
            }
        ),
        "fact_codes_json": json.dumps(fact_codes, ensure_ascii=False),
        "official_view_count": successful_views,
        # Direct SIRET and name/geo requests are different API views of one
        # SIRENE-derived source, never independent corroborating sources.
        "official_source_family": OFFICIAL_SOURCE_FAMILY,
        "independent_source_family_count": int(successful_views > 0),
        "independent_non_sirene_source_count": 0,
        "same_source_views_not_counted_as_corroboration": True,
        "model_score_used_for_facts": False,
        "address_used_as_correctness_proof": False,
        "correctness_conclusion": "NOT_DERIVED",
        "training_eligible": False,
        "requires_human_or_independent_evidence": True,
    }


def derive_facts(queue: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    auto = queue.loc[queue["decision"].astype(str).eq("AUTO_MATCH")].copy()
    if len(auto) != EXPECTED_AUTO_COUNT:
        raise ValueError(
            f"Frozen AUTO population changed: {len(auto)} != {EXPECTED_AUTO_COUNT}"
        )
    if auto["audit_case_id"].astype(str).duplicated().any():
        raise ValueError("AUTO audit_case_id values must be unique")
    evidence = evidence.copy()
    evidence["audit_case_id"] = evidence["audit_case_id"].astype(str)
    if evidence.duplicated(["audit_case_id", "query_kind"]).any():
        raise ValueError("Official evidence contains duplicate case/query views")
    unknown_kinds = set(evidence["query_kind"].astype(str)) - EXPECTED_QUERY_KINDS
    if unknown_kinds:
        raise ValueError(f"Unsupported official query kinds: {sorted(unknown_kinds)}")
    auto_ids = set(auto["audit_case_id"].astype(str))
    if set(evidence["audit_case_id"]) != auto_ids:
        raise ValueError("Official evidence and frozen AUTO cases do not align")
    for case_id, group in evidence.groupby("audit_case_id"):
        kinds = set(group["query_kind"].astype(str))
        if not {"TOP1_SIRET", "CRM_NAME_GEO"} <= kinds:
            raise ValueError(f"{case_id}: missing mandatory official views")
        for source_url in group["source_url"]:
            parsed = urlparse(_text(source_url))
            if (
                parsed.scheme != "https"
                or parsed.netloc != "recherche-entreprises.api.gouv.fr"
            ):
                raise ValueError(f"{case_id}: unexpected evidence source")

    grouped = {
        case_id: group
        for case_id, group in evidence.groupby("audit_case_id", sort=False)
    }
    records = [
        derive_case_facts(
            case,
            grouped[str(case["audit_case_id"])],
        )
        for _, case in auto.sort_values("audit_case_id").iterrows()
    ]
    facts = pd.DataFrame(records)
    if facts["correctness_conclusion"].ne("NOT_DERIVED").any():
        raise AssertionError("Evidence facts must not adjudicate correctness")
    if facts["training_eligible"].any():
        raise AssertionError("Evidence facts alone are never training labels")
    return facts


def _validate_input_manifests(
    queue_path: Path,
    evidence_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    queue_manifest_path = queue_path.parent / "manifest.json"
    evidence_manifest_path = evidence_path.parent / "manifest.json"
    queue_manifest = json.loads(queue_manifest_path.read_text(encoding="utf-8"))
    evidence_manifest = json.loads(
        evidence_manifest_path.read_text(encoding="utf-8")
    )
    queue_hash = file_sha256(queue_path)
    evidence_hash = file_sha256(evidence_path)
    if queue_manifest.get("outputs", {}).get(queue_path.name) != queue_hash:
        raise ValueError("Frozen hard-label queue hash mismatch")
    if evidence_manifest.get("outputs", {}).get(evidence_path.name) != evidence_hash:
        raise ValueError("Frozen official evidence hash mismatch")
    if evidence_manifest.get("queue_sha256") != queue_hash:
        raise ValueError("Official evidence was collected from another queue")
    return queue_manifest, evidence_manifest


def build_artifact(
    *,
    queue_path: Path,
    evidence_path: Path,
    output_root: Path,
) -> Path:
    queue_manifest, evidence_manifest = _validate_input_manifests(
        queue_path, evidence_path
    )
    queue_hash = file_sha256(queue_path)
    evidence_hash = file_sha256(evidence_path)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "queue_sha256": queue_hash,
        "official_evidence_sha256": evidence_hash,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.4 evidence facts exist: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent)
    )
    try:
        facts = derive_facts(
            pd.read_parquet(queue_path),
            pd.read_parquet(evidence_path),
        )
        facts_path = staging / "evidence_facts.parquet"
        facts.to_parquet(facts_path, index=False)
        summary = {
            "case_count": int(len(facts)),
            "correctness_conclusions": {
                str(key): int(value)
                for key, value in facts["correctness_conclusion"]
                .value_counts()
                .items()
            },
            "training_eligible_count": int(facts["training_eligible"].sum()),
            "top1_direct_identity_count": int(
                facts["top1_direct_exact_siret_returned"].sum()
            ),
            "name_geo_top1_exact_hit_count": int(
                facts["name_geo_top1_exact_hit"].sum()
            ),
            "input_direct_identity_count": int(
                facts["input_direct_exact_siret_returned"].sum()
            ),
            "independent_non_sirene_source_count_max": int(
                facts["independent_non_sirene_source_count"].max()
            ),
            "verdict": "FACTS_ONLY_NO_ADJUDICATION",
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
            "inputs": {
                "queue": {
                    "path": str(queue_path),
                    "sha256": queue_hash,
                    "manifest_sha256": file_sha256(
                        queue_path.parent / "manifest.json"
                    ),
                    "build_id": queue_manifest.get("build_id"),
                },
                "official_evidence": {
                    "path": str(evidence_path),
                    "sha256": evidence_hash,
                    "manifest_sha256": file_sha256(
                        evidence_path.parent / "manifest.json"
                    ),
                    "build_id": evidence_manifest.get("build_id"),
                },
            },
            "source_policy": {
                "source_family": OFFICIAL_SOURCE_FAMILY,
                "multiple_api_views_are_independent": False,
                "model_score_can_prove_correctness": False,
                "address_can_prove_correctness": False,
                "creates_training_labels": False,
            },
            "outputs": {
                facts_path.name: file_sha256(facts_path),
                summary_path.name: file_sha256(summary_path),
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
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--official-evidence", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        build_artifact(
            queue_path=args.queue,
            evidence_path=args.official_evidence,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
