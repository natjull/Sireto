from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.adapt_v44_adjudication_batches import (
    adapt_batches,
    build_input_artifact,
    validate_input_artifact,
)
from scripts.build_v44_adjudications import (
    build_adjudications,
    candidate_pool_sha256,
)
from src.xgb_matcher.v9_dataset import file_sha256


TOP1 = "11111111100011"
SECOND = "22222222200022"


def _case(
    *,
    case_id: str = "case-a",
    service_id: str = "service-a",
    label: str = "TOP1_CORRECT",
) -> dict:
    return {
        "audit_case_id": case_id,
        "service_id": service_id,
        "frozen_top1": {"siret": TOP1, "siren": TOP1[:9]},
        "adjudication_label": label,
        "evidence_validated": True,
        "training_eligible": True,
        "validated_correct_siret": TOP1 if label == "TOP1_CORRECT" else None,
        "independent_source_families": ["SIRENE_REGISTRY", "SECTOR_ALPHA"],
        "decision_reason": "Deux producteurs identifient explicitement le top-1.",
        "sources": [
            {
                "evidence_id": f"{case_id}-registry",
                "producer": "INSEE",
                "source_family": "REGISTRY_CORE_SIRENE",
                "canonical_source_family": "SIRENE_REGISTRY",
                "independence_group": "SIRENE_REGISTRY",
                "proof_kind": "IDENTITY_TOP1_MATCHES_CRM",
                "counts_for_independence": True,
                "document_type": "registre",
                "url": f"https://registry.example/{TOP1}",
                "collected_at": "2026-07-27T10:00:00+00:00",
                "archived_facts": ["Le registre relie le nom au SIRET."],
            },
            {
                "evidence_id": f"{case_id}-sector",
                "producer": "Producteur sectoriel Alpha",
                "source_family": "SECTOR_REVIEWED_ALPHA",
                "canonical_source_family": "SECTOR_ALPHA",
                "independence_group": "SECTOR_ALPHA",
                "proof_kind": "IDENTITY_SECTOR_IDENTIFIER_LINKS_TOP1_TO_CRM",
                "counts_for_independence": True,
                "document_type": "registre sectoriel",
                "urls": [
                    f"https://sector.example/{TOP1}/second",
                    f"https://sector.example/{TOP1}/first",
                ],
                "collected_at": "2026-07-27",
                "archived_facts": ["L'identifiant sectoriel relie le nom au SIRET."],
            },
        ],
    }


def _batch(case: dict | None = None) -> dict:
    return {
        "schema_version": "sireto-v4.4-sector-adjudications-test-1",
        "created_at": "2026-07-27T23:59:00+02:00",
        "cases": [case or _case()],
    }


def _queue(case: dict | None = None) -> pd.DataFrame:
    case = case or _case()
    return pd.DataFrame(
        [
            {
                "audit_case_id": case["audit_case_id"],
                "service_id": case["service_id"],
                "top1_siret": case["frozen_top1"]["siret"],
                "decision": "AUTO_MATCH",
                "sampling_stratum": "RANDOM_POPULATION",
                "priority_reason": "P1_OTHER_AUTO_UNRESOLVED",
            }
        ]
    )


def _shadow(case: dict | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    case = case or _case()
    manifest = {
        "schema_version": "sireto-shadow-v4.1-1",
        "run_metadata": {
            "release_id": "release-frozen",
            "ranker_bundle_id": "ranker-frozen",
            "acceptor_bundle_id": "acceptor-frozen",
            "retrieval_signature": "retrieval-frozen",
        },
    }
    top10 = pd.DataFrame(
        [
            {
                "service_id": case["service_id"],
                "rank": 1,
                "candidate_siret": case["frozen_top1"]["siret"],
            },
            {
                "service_id": case["service_id"],
                "rank": 2,
                "candidate_siret": SECOND,
            },
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "service_id": case["service_id"],
                "predicted_siret": case["frozen_top1"]["siret"],
            }
        ]
    )
    return manifest, top10, decisions


def _adapt(batch: dict | None = None):
    batch = batch or _batch()
    case = batch["cases"][0]
    manifest, top10, decisions = _shadow(case)
    return adapt_batches(
        batches=[batch],
        queue=_queue(case),
        top10=top10,
        decisions=decisions,
        shadow_manifest=manifest,
        top10_path=Path("/frozen/candidates_top10.parquet"),
    )


def test_generic_adapter_copies_taxonomy_and_uses_only_frozen_top10() -> None:
    facts, proofs, judgments = _adapt()

    assert json.loads(facts.iloc[0]["frozen_candidate_sirets_json"]) == [
        TOP1,
        SECOND,
    ]
    assert facts.iloc[0]["frozen_candidate_pool_sha256"] == candidate_pool_sha256(
        [TOP1, SECOND]
    )
    assert facts.iloc[0]["sampling_stratum"] == "RANDOM_POPULATION"
    assert bool(facts.iloc[0]["positive_injection_by_adapter"]) is False
    assert set(proofs["source_family"]) == {
        "SIRENE_REGISTRY",
        "SECTOR_ALPHA",
    }
    assert set(proofs["proof_kind"]) == {
        "IDENTITY_TOP1_MATCHES_CRM",
        "IDENTITY_SECTOR_IDENTIFIER_LINKS_TOP1_TO_CRM",
    }
    sector = proofs.loc[proofs["source_family"].eq("SECTOR_ALPHA")].iloc[0]
    assert sector["source_locator"].endswith("/first")
    assert json.loads(sector["source_locators_json"]) == sorted(
        [
            f"https://sector.example/{TOP1}/second",
            f"https://sector.example/{TOP1}/first",
        ]
    )
    canonical = build_adjudications(facts, proofs, judgments)
    assert canonical.iloc[0]["adjudication_label"] == "TOP1_CORRECT"
    assert bool(canonical.iloc[0]["training_eligible"]) is True


def test_adapter_accepts_unambiguous_mixed_source_aliases() -> None:
    batch = _batch()
    case = batch["cases"][0]
    case["sources"][1]["source_family"] = "ENTITY_SELF_PUBLICATION"
    case["sources"][1][
        "canonical_source_family"
    ] = "OFFICIAL_ALPHA_PUBLICATION"
    case["sources"][1]["independence_group"] = "ALPHA_ENTITY"
    case["independent_source_families"] = [
        "SIRENE_REGISTRY",
        "ENTITY_SELF_PUBLICATION",
    ]

    _, proofs, _ = _adapt(batch)

    assert set(proofs["independence_group"]) == {
        "SIRENE_REGISTRY",
        "ALPHA_ENTITY",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda batch: batch["cases"][0].update(training_eligible=False),
            "evidence_validated must equal training_eligible",
        ),
        (
            lambda batch: batch["cases"][0]["sources"][1].update(
                independence_group="SIRENE_REGISTRY"
            ),
            "requires two independent proofs",
        ),
        (
            lambda batch: batch["cases"][0]["sources"][1].pop("proof_kind"),
            "proof_kind must be explicit",
        ),
        (
            lambda batch: batch["cases"][0]["sources"][1].pop(
                "canonical_source_family"
            ),
            "canonical source family and group are required",
        ),
    ],
)
def test_adapter_refuses_invalid_review_contract(mutation, message: str) -> None:
    batch = _batch()
    mutation(batch)
    with pytest.raises(ValueError, match=message):
        _adapt(batch)


def test_adapter_refuses_duplicate_cases_services_and_proofs() -> None:
    with pytest.raises(ValueError, match="audit_case_id"):
        batch = _batch()
        batch["cases"].append(copy.deepcopy(batch["cases"][0]))
        _adapt(batch)

    batch = _batch()
    duplicate = copy.deepcopy(batch["cases"][0])
    duplicate["audit_case_id"] = "case-b"
    duplicate["sources"][0]["evidence_id"] = "case-b-registry"
    duplicate["sources"][1]["evidence_id"] = "case-b-sector"
    batch["cases"].append(duplicate)

    with pytest.raises(ValueError, match="service_id"):
        _adapt(batch)

    batch = _batch()
    batch["cases"][0]["sources"][1]["evidence_id"] = "case-a-registry"
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        _adapt(batch)


def test_validated_case_requires_two_counted_groups() -> None:
    batch = _batch()
    case = batch["cases"][0]
    case["sources"][1]["independence_group"] = "SIRENE_REGISTRY"
    case["independent_source_families"] = ["SIRENE_REGISTRY"]

    with pytest.raises(ValueError, match="requires two independent proofs"):
        _adapt(batch)


def test_adapter_refuses_queue_shadow_and_pool_drift() -> None:
    batch = _batch()
    case = batch["cases"][0]
    manifest, top10, decisions = _shadow(case)
    queue = _queue(case)
    queue.loc[0, "top1_siret"] = SECOND
    with pytest.raises(ValueError, match="top1 mismatch with V4.3 queue"):
        adapt_batches(
            batches=[batch],
            queue=queue,
            top10=top10,
            decisions=decisions,
            shadow_manifest=manifest,
            top10_path=Path("/frozen/top10.parquet"),
        )

    queue = _queue(case)
    queue.loc[0, "decision"] = "REVIEW"
    with pytest.raises(ValueError, match="accepts only frozen AUTO_MATCH"):
        adapt_batches(
            batches=[batch],
            queue=queue,
            top10=top10,
            decisions=decisions,
            shadow_manifest=manifest,
            top10_path=Path("/frozen/top10.parquet"),
        )

    queue = _queue(case)
    top10.loc[top10["rank"].eq(2), "rank"] = 3
    with pytest.raises(ValueError, match="ranks must be consecutive"):
        adapt_batches(
            batches=[batch],
            queue=queue,
            top10=top10,
            decisions=decisions,
            shadow_manifest=manifest,
            top10_path=Path("/frozen/top10.parquet"),
        )


def _write_source_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    batch = _batch()
    batch_path = tmp_path / "batch-a.json"
    batch_path.write_text(json.dumps(batch), encoding="utf-8")

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    queue_path = queue_dir / "hard_label_queue.parquet"
    _queue().to_parquet(queue_path, index=False)

    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    shadow_manifest, top10, decisions = _shadow()
    top10_path = shadow_dir / "candidates_top10.parquet"
    decisions_path = shadow_dir / "decisions.parquet"
    top10.to_parquet(top10_path, index=False)
    decisions.to_parquet(decisions_path, index=False)
    shadow_manifest["outputs"] = {
        top10_path.name: file_sha256(top10_path),
        decisions_path.name: file_sha256(decisions_path),
    }
    (shadow_dir / "manifest.json").write_text(
        json.dumps(shadow_manifest), encoding="utf-8"
    )

    queue_manifest = {
        "schema_version": "sireto-v4.3-hard-label-queue-1",
        "inputs": {
            "top10": {
                "path": str(top10_path),
                "sha256": file_sha256(top10_path),
            }
        },
        "outputs": {queue_path.name: file_sha256(queue_path)},
    }
    (queue_dir / "manifest.json").write_text(
        json.dumps(queue_manifest), encoding="utf-8"
    )
    return batch_path, queue_dir, shadow_dir


def test_input_artifact_is_immutable_recomputable_and_hash_bound(
    tmp_path: Path,
) -> None:
    batch_path, queue_dir, shadow_dir = _write_source_artifacts(tmp_path)
    output_root = tmp_path / "adapted"
    artifact = build_input_artifact(
        batch_jsons=[batch_path],
        queue_dir=queue_dir,
        shadow_dir=shadow_dir,
        output_root=output_root,
    )
    validate_input_artifact(artifact)
    manifest = json.loads((artifact / "manifest.json").read_text())
    assert manifest["row_counts"] == {
        "facts": 1,
        "proofs": 2,
        "judgments": 1,
    }
    assert manifest["invariants"]["case_allow_list"] is False
    assert manifest["invariants"]["source_mapping_table"] is False

    with pytest.raises(FileExistsError, match="Immutable"):
        build_input_artifact(
            batch_jsons=[batch_path],
            queue_dir=queue_dir,
            shadow_dir=shadow_dir,
            output_root=output_root,
        )

    with (artifact / "proofs.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="output hash mismatch"):
        validate_input_artifact(artifact)


def test_builder_refuses_queue_shadow_top10_hash_conflict(tmp_path: Path) -> None:
    batch_path, queue_dir, shadow_dir = _write_source_artifacts(tmp_path)
    queue_manifest_path = queue_dir / "manifest.json"
    queue_manifest = json.loads(queue_manifest_path.read_text())
    queue_manifest["inputs"]["top10"]["sha256"] = "0" * 64
    queue_manifest_path.write_text(json.dumps(queue_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="queue manifest input hash mismatch"):
        build_input_artifact(
            batch_jsons=[batch_path],
            queue_dir=queue_dir,
            shadow_dir=shadow_dir,
            output_root=tmp_path / "adapted",
        )
