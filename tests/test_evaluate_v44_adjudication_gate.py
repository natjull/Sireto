from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_v44_adjudications import (
    SIRENE_INDEPENDENCE_GROUP,
    build_adjudications,
    build_artifact as build_adjudication_artifact,
    candidate_pool_sha256,
)
from scripts.evaluate_v44_adjudication_gate import (
    build_gate_artifact,
    compute_gate_metrics,
    decide_gate,
    evaluate_adjudication_gate,
    load_adjudication_artifacts,
    validate_gate_artifact,
)
from src.xgb_matcher.v9_dataset import file_sha256


TOP1 = "11111111100001"
ALTERNATIVE = "22222222200002"


def _canonical_cases(
    labels: list[str],
    *,
    random_count: int = 0,
) -> pd.DataFrame:
    facts: list[dict] = []
    proofs: list[dict] = []
    judgments: list[dict] = []
    pool = [TOP1, ALTERNATIVE]
    for index, label in enumerate(labels):
        case_id = f"case-{index:04d}"
        facts.append(
            {
                "audit_case_id": case_id,
                "service_id": f"service-{index:04d}",
                "frozen_top1_siret": TOP1,
                "frozen_top1_siren": TOP1[:9],
                "frozen_model_bundle_id": "model-v41",
                "frozen_retrieval_signature": "retrieval-v42",
                "frozen_candidate_sirets_json": json.dumps(pool),
                "frozen_candidate_pool_sha256": candidate_pool_sha256(pool),
                "sampling_stratum": (
                    "RANDOM_POPULATION" if index < random_count else "HARD_AUTO"
                ),
                "priority_reason": "SYNTHETIC_TEST",
            }
        )
        refs: list[str] = []
        if label != "UNRESOLVED":
            refs = [f"{case_id}-sirene", f"{case_id}-site"]
            proofs.extend(
                [
                    {
                        "proof_id": refs[0],
                        "audit_case_id": case_id,
                        "producer": "INSEE",
                        "source_family": "SIRENE_SNAPSHOT",
                        "independence_group": SIRENE_INDEPENDENCE_GROUP,
                        "source_locator": (
                            f"https://registry.example/{case_id}"
                        ),
                        "collected_at": "2026-07-27T10:00:00+00:00",
                        "proof_kind": "IDENTITY_REGISTRY",
                        "supports_label": label,
                        "identity_consistent": True,
                        "contradiction_unresolved": False,
                    },
                    {
                        "proof_id": refs[1],
                        "audit_case_id": case_id,
                        "producer": "ENTITY_ALPHA",
                        "source_family": "OFFICIAL_ENTITY_SITE",
                        "independence_group": "ENTITY_ALPHA_OFFICIAL",
                        "source_locator": (
                            f"https://alpha.example/evidence/{case_id}"
                        ),
                        "collected_at": "2026-07-27T11:00:00+00:00",
                        "proof_kind": "IDENTITY_OFFICIAL_SITE",
                        "supports_label": label,
                        "identity_consistent": True,
                        "contradiction_unresolved": False,
                    },
                ]
            )
        judgments.append(
            {
                "audit_case_id": case_id,
                "adjudication_label": label,
                "validated_correct_siret": (
                    TOP1 if label == "TOP1_CORRECT" else None
                ),
                "evidence_ref_ids_json": json.dumps(refs),
                "adjudication_reason": "Synthetic explicit judgment.",
                "adjudication_rule_version": "synthetic-v1",
                "adjudicated_at": "2026-07-27T12:00:00+00:00",
            }
        )
    proof_columns = [
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
    ]
    return build_adjudications(
        pd.DataFrame(facts),
        pd.DataFrame(proofs, columns=proof_columns),
        pd.DataFrame(judgments),
    )


def _write_adjudication_artifact(
    root: Path,
    *,
    label: str = "TOP1_CORRECT",
) -> Path:
    source = root / "source"
    source.mkdir(parents=True)
    canonical = _canonical_cases([label], random_count=1)
    # Recreate the canonical builder inputs rather than spoofing its artifact.
    pool = [TOP1, ALTERNATIVE]
    facts = pd.DataFrame(
        [
            {
                "audit_case_id": "case-0000",
                "service_id": "service-0000",
                "frozen_top1_siret": TOP1,
                "frozen_top1_siren": TOP1[:9],
                "frozen_model_bundle_id": "model-v41",
                "frozen_retrieval_signature": "retrieval-v42",
                "frozen_candidate_sirets_json": json.dumps(pool),
                "frozen_candidate_pool_sha256": candidate_pool_sha256(pool),
                "sampling_stratum": "RANDOM_POPULATION",
                "priority_reason": "SYNTHETIC_TEST",
            }
        ]
    )
    proof_rows = []
    refs: list[str] = []
    if label != "UNRESOLVED":
        refs = ["case-0000-sirene", "case-0000-site"]
        proof_rows = [
            {
                "proof_id": refs[0],
                "audit_case_id": "case-0000",
                "producer": "INSEE",
                "source_family": "SIRENE_SNAPSHOT",
                "independence_group": SIRENE_INDEPENDENCE_GROUP,
                "source_locator": "https://registry.example/case-0000",
                "collected_at": "2026-07-27T10:00:00+00:00",
                "proof_kind": "IDENTITY_REGISTRY",
                "supports_label": label,
                "identity_consistent": True,
                "contradiction_unresolved": False,
            },
            {
                "proof_id": refs[1],
                "audit_case_id": "case-0000",
                "producer": "ENTITY_ALPHA",
                "source_family": "OFFICIAL_ENTITY_SITE",
                "independence_group": "ENTITY_ALPHA_OFFICIAL",
                "source_locator": "https://alpha.example/evidence/case-0000",
                "collected_at": "2026-07-27T11:00:00+00:00",
                "proof_kind": "IDENTITY_OFFICIAL_SITE",
                "supports_label": label,
                "identity_consistent": True,
                "contradiction_unresolved": False,
            },
        ]
    proofs = pd.DataFrame(
        proof_rows,
        columns=[
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
        ],
    )
    judgments = pd.DataFrame(
        [
            {
                "audit_case_id": "case-0000",
                "adjudication_label": label,
                "validated_correct_siret": (
                    TOP1 if label == "TOP1_CORRECT" else None
                ),
                "evidence_ref_ids_json": json.dumps(refs),
                "adjudication_reason": "Synthetic explicit judgment.",
                "adjudication_rule_version": "synthetic-v1",
                "adjudicated_at": "2026-07-27T12:00:00+00:00",
            }
        ]
    )
    # Ensure this helper remains aligned with its intended canonical row.
    assert len(canonical) == 1
    facts_path = source / "facts.parquet"
    proofs_path = source / "proofs.parquet"
    judgments_path = source / "judgments.parquet"
    facts.to_parquet(facts_path, index=False)
    proofs.to_parquet(proofs_path, index=False)
    judgments.to_parquet(judgments_path, index=False)
    return build_adjudication_artifact(
        facts_path=facts_path,
        proofs_path=proofs_path,
        judgments_path=judgments_path,
        output_root=root / "adjudications",
    )


def test_gate_goes_only_when_all_contract_counts_pass() -> None:
    adjudications = _canonical_cases(
        ["TOP1_CORRECT"] * 75 + ["TOP1_WRONG"] * 50,
        random_count=30,
    )
    report = evaluate_adjudication_gate(adjudications)

    assert report["verdict"] == "GO_RETRAIN_AUTO"
    assert report["metrics"]["top1_correct_evidence_validated_count"] == 75
    assert report["metrics"]["top1_wrong_evidence_validated_count"] == 50
    assert report["metrics"]["random_evidence_validated_count"] == 30
    assert report["metrics"]["acceptor_eligible_count"] == 125
    assert report["metrics"]["ranker_eligible_count"] == 75
    assert all(report["checks"].values())


@pytest.mark.parametrize(
    ("correct", "wrong", "random"),
    [(74, 50, 30), (75, 49, 30), (75, 50, 29)],
)
def test_any_gate_deficit_defaults_to_pivot(
    correct: int,
    wrong: int,
    random: int,
) -> None:
    report = evaluate_adjudication_gate(
        _canonical_cases(
            ["TOP1_CORRECT"] * correct + ["TOP1_WRONG"] * wrong,
            random_count=random,
        )
    )

    assert report["verdict"] == "PIVOT_MORE_EVIDENCE"
    assert report["source_status"] == "OPEN"
    assert not report["stop_requested"]


def test_stop_requires_explicit_option_and_reason() -> None:
    metrics = compute_gate_metrics(_canonical_cases(["TOP1_CORRECT"]))

    with pytest.raises(ValueError, match="explicit reason"):
        decide_gate(metrics, stop_requested=True)
    with pytest.raises(ValueError, match="requires stop_requested"):
        decide_gate(metrics, stop_reason="No source remains.")

    decision = decide_gate(
        metrics,
        stop_requested=True,
        stop_reason="All authorized public sources were exhausted.",
    )
    assert decision["verdict"] == "STOP_AUTONOMOUS_LABELING"
    assert decision["source_status"] == "EXPLICITLY_EXHAUSTED"


def test_metrics_publish_unresolved_and_model_eligibility_counts() -> None:
    metrics = compute_gate_metrics(
        _canonical_cases(
            [
                "TOP1_CORRECT",
                "TOP1_WRONG",
                "AMBIGUOUS",
                "UNRESOLVED",
            ],
            random_count=4,
        )
    )

    assert metrics["unresolved_count"] == 1
    assert metrics["evidence_validated_count"] == 3
    assert metrics["acceptor_eligible_count"] == 3
    assert metrics["ranker_eligible_count"] == 1
    assert metrics["random_evidence_validated_count"] == 3


def test_identical_cases_are_deduplicated_across_valid_artifacts(
    tmp_path: Path,
) -> None:
    first = _write_adjudication_artifact(tmp_path / "first")
    second = _write_adjudication_artifact(tmp_path / "second")

    combined, provenance = load_adjudication_artifacts([first, second, first])

    assert len(combined) == 1
    assert len(provenance) == 2


def test_conflicting_duplicate_case_is_blocking(tmp_path: Path) -> None:
    correct = _write_adjudication_artifact(tmp_path / "correct")
    wrong = _write_adjudication_artifact(
        tmp_path / "wrong",
        label="TOP1_WRONG",
    )

    with pytest.raises(ValueError, match="Conflicting adjudications"):
        load_adjudication_artifacts([correct, wrong])


def test_gate_report_is_immutable_hashed_json_and_markdown(
    tmp_path: Path,
) -> None:
    adjudication = _write_adjudication_artifact(tmp_path / "input")
    contract = tmp_path / "contract.md"
    contract.write_text("# Synthetic frozen contract\n", encoding="utf-8")

    gate = build_gate_artifact(
        adjudication_artifact_dirs=[adjudication],
        output_root=tmp_path / "gates",
        contract_path=contract,
    )
    manifest = json.loads((gate / "manifest.json").read_text())
    report = json.loads((gate / "gate_report.json").read_text())

    assert report["verdict"] == "PIVOT_MORE_EVIDENCE"
    assert manifest["verdict"] == report["verdict"]
    assert manifest["contract"]["sha256"] == file_sha256(contract)
    assert manifest["outputs"]["gate_report.json"] == file_sha256(
        gate / "gate_report.json"
    )
    assert manifest["outputs"]["gate_report.md"] == file_sha256(
        gate / "gate_report.md"
    )
    assert "PIVOT_MORE_EVIDENCE" in (gate / "gate_report.md").read_text()
    validate_gate_artifact(gate)

    with pytest.raises(FileExistsError, match="Immutable"):
        build_gate_artifact(
            adjudication_artifact_dirs=[adjudication],
            output_root=tmp_path / "gates",
            contract_path=contract,
        )

    with (gate / "gate_report.md").open("a", encoding="utf-8") as stream:
        stream.write("tampered")
    with pytest.raises(ValueError, match="output hash mismatch"):
        validate_gate_artifact(gate)
