from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.adapt_v44_known_contradictions import (
    EXPECTED_CASE_IDS,
    adapt_inputs,
    build_input_artifact,
    validate_input_artifact,
)
from scripts.build_v44_adjudications import (
    build_adjudications,
    candidate_pool_sha256,
)
from src.xgb_matcher.v9_dataset import file_sha256


def _source_artifact() -> dict:
    cases = []
    for index, case_id in enumerate(sorted(EXPECTED_CASE_IDS), start=1):
        service_id = f"service-{index}"
        top1 = f"{index:09d}00001"
        label = "UNRESOLVED" if index == 3 else "TOP1_WRONG"
        sources = [
            {
                "evidence_id": f"{case_id}-registry",
                "producer": "INSEE, diffusé par la Direction interministérielle du numérique",
                "source_family": "REGISTRY_CORE_SIRENE",
                "document_type": "API officielle",
                "url": f"https://recherche-entreprises.api.gouv.fr/{top1}",
                "collected_at": "2026-07-27T10:00:00+00:00",
                "archived_facts": ["Le registre identifie le top-1."],
            }
        ]
        if label != "UNRESOLVED":
            sources.append(
                {
                    "evidence_id": f"{case_id}-welcoop",
                    "producer": "WELCOOP LOGISTIQUE",
                    "source_family": "ENTITY_SELF_DECLARATION",
                    "document_type": "Site officiel",
                    "url": "https://www.welcoop-logistique.com/mentions-legales.html",
                    "collected_at": "2026-07-27",
                    "archived_facts": ["L'entité publie une identité différente."],
                }
            )
        cases.append(
            {
                "audit_case_id": case_id,
                "service_id": service_id,
                "frozen_top1": {"siret": top1, "siren": top1[:9]},
                "adjudication_label": label,
                "evidence_validated": label != "UNRESOLVED",
                "validated_correct_siret": None,
                "decision_reason": f"Décision figée {label}.",
                "sources": sources,
            }
        )
    return {
        "schema_version": "test",
        "created_at": "2026-07-27T23:59:00+02:00",
        "cases": cases,
    }


def _shadow(source: dict) -> tuple[dict, dict]:
    manifest = {
        "schema_version": "sireto-shadow-v4.1-1",
        "run_metadata": {
            "release_id": "release-exact",
            "ranker_bundle_id": "ranker-exact",
            "acceptor_bundle_id": "acceptor-exact",
            "retrieval_signature": "retrieval-exact",
        },
    }
    records = {}
    for line_number, case in enumerate(source["cases"], start=1):
        top1 = case["frozen_top1"]["siret"]
        alternative = f"{int(top1[:9]) + 10:09d}00002"
        records[case["service_id"]] = {
            "service_id": case["service_id"],
            "decision": {"predicted_siret": top1},
            "top_candidates": [
                {"rank": 1, "candidate_siret": top1},
                {"rank": 2, "candidate_siret": alternative},
            ],
            "_line_number": line_number,
        }
    return manifest, records


def test_adapter_uses_only_archived_order_and_frozen_manifest_identifiers():
    source = _source_artifact()
    manifest, shadow_cases = _shadow(source)
    facts, proofs, judgments = adapt_inputs(
        evidence_artifact=source,
        shadow_manifest=manifest,
        shadow_cases=shadow_cases,
        shadow_evidence_path=Path("/archive/evidence.jsonl"),
    )

    assert len(facts) == 5
    assert set(facts["audit_case_id"]) == EXPECTED_CASE_IDS
    assert facts["frozen_model_bundle_id"].eq("release-exact").all()
    assert facts["frozen_ranker_bundle_id"].eq("ranker-exact").all()
    assert facts["frozen_acceptor_bundle_id"].eq("acceptor-exact").all()
    assert facts["frozen_retrieval_signature"].eq("retrieval-exact").all()
    for row in facts.to_dict("records"):
        pool = json.loads(row["frozen_candidate_sirets_json"])
        assert pool[0] == row["frozen_top1_siret"]
        assert row["frozen_candidate_pool_sha256"] == candidate_pool_sha256(pool)
        assert row["positive_injection_by_adapter"] is False

    canonical = build_adjudications(facts, proofs, judgments)
    assert canonical["adjudication_label"].value_counts().to_dict() == {
        "TOP1_WRONG": 4,
        "UNRESOLVED": 1,
    }
    assert int(canonical["training_eligible"].sum()) == 4
    assert int(canonical["ranker_eligible"].sum()) == 0


def test_date_only_collection_is_encoded_with_explicit_day_precision():
    source = _source_artifact()
    manifest, shadow_cases = _shadow(source)
    _, proofs, _ = adapt_inputs(
        evidence_artifact=source,
        shadow_manifest=manifest,
        shadow_cases=shadow_cases,
        shadow_evidence_path=Path("/archive/evidence.jsonl"),
    )
    day_rows = proofs[proofs["collected_at_precision"].eq("DAY_EUROPE_PARIS")]

    assert not day_rows.empty
    assert day_rows["collected_at"].str.endswith("+00:00").all()
    assert day_rows["collected_at"].str.startswith("2026-07-26T22:00:00").all()


def test_adapter_refuses_candidate_or_decision_drift():
    source = _source_artifact()
    manifest, shadow_cases = _shadow(source)
    first = source["cases"][0]
    shadow_cases[first["service_id"]]["decision"]["predicted_siret"] = "99999999900009"

    with pytest.raises(ValueError, match="differs from shadow decision"):
        adapt_inputs(
            evidence_artifact=source,
            shadow_manifest=manifest,
            shadow_cases=shadow_cases,
            shadow_evidence_path=Path("/archive/evidence.jsonl"),
        )


def test_adapter_refuses_unmapped_proof_instead_of_inventing_mapping():
    source = _source_artifact()
    source["cases"][0]["sources"][1]["producer"] = "UNKNOWN PRODUCER"
    manifest, shadow_cases = _shadow(source)

    with pytest.raises(ValueError, match="unmapped proof source"):
        adapt_inputs(
            evidence_artifact=source,
            shadow_manifest=manifest,
            shadow_cases=shadow_cases,
            shadow_evidence_path=Path("/archive/evidence.jsonl"),
        )


def test_input_artifact_is_immutable_and_recomputable(tmp_path: Path):
    source = _source_artifact()
    evidence_json = tmp_path / "evidence.json"
    evidence_json.write_text(json.dumps(source), encoding="utf-8")
    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    manifest, shadow_cases = _shadow(source)
    evidence_jsonl = shadow_dir / "evidence.jsonl"
    with evidence_jsonl.open("w", encoding="utf-8") as stream:
        for record in shadow_cases.values():
            record = {key: value for key, value in record.items() if key != "_line_number"}
            stream.write(json.dumps(record) + "\n")
    manifest["outputs"] = {"evidence.jsonl": file_sha256(evidence_jsonl)}
    (shadow_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    artifact = build_input_artifact(
        evidence_json=evidence_json,
        shadow_dir=shadow_dir,
        output_root=tmp_path / "inputs",
    )
    validate_input_artifact(artifact)
    artifact_manifest = json.loads((artifact / "manifest.json").read_text())
    assert artifact_manifest["row_counts"] == {
        "facts": 5,
        "proofs": 9,
        "judgments": 5,
    }

    with pytest.raises(FileExistsError, match="Immutable"):
        build_input_artifact(
            evidence_json=evidence_json,
            shadow_dir=shadow_dir,
            output_root=tmp_path / "inputs",
        )

    with (artifact / "facts.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="output hash mismatch"):
        validate_input_artifact(artifact)
