from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_v44_adjudications import (
    SIRENE_INDEPENDENCE_GROUP,
    build_adjudications,
    build_artifact,
    candidate_pool_sha256,
    validate_artifact,
)
from src.xgb_matcher.v9_dataset import file_sha256


TOP1 = "11111111100001"
ALTERNATIVE = "22222222200002"


def _facts(case_id: str = "case-1") -> pd.DataFrame:
    pool = [TOP1, ALTERNATIVE]
    return pd.DataFrame(
        [
            {
                "audit_case_id": case_id,
                "service_id": f"service-{case_id}",
                "frozen_top1_siret": TOP1,
                "frozen_top1_siren": TOP1[:9],
                "frozen_model_bundle_id": "model-bundle-v41",
                "frozen_retrieval_signature": "retrieval-v42",
                "frozen_candidate_sirets_json": json.dumps(pool),
                "frozen_candidate_pool_sha256": candidate_pool_sha256(pool),
                "sampling_stratum": "RANDOM_POPULATION",
                "priority_reason": "P1_OTHER_AUTO_UNRESOLVED",
            }
        ]
    )


def _proof(
    proof_id: str,
    *,
    family: str,
    group: str,
    producer: str | None = None,
    label: str = "TOP1_CORRECT",
    case_id: str = "case-1",
    identity_consistent: bool = True,
    contradiction: bool = False,
) -> dict:
    hostname = (
        "registry.example"
        if group == SIRENE_INDEPENDENCE_GROUP
        else f"{group.lower().replace('_', '-')}.example"
    )
    return {
        "proof_id": proof_id,
        "audit_case_id": case_id,
        "producer": producer or family,
        "source_family": family,
        "independence_group": group,
        "source_locator": f"https://{hostname}/{proof_id}",
        "collected_at": "2026-07-27T10:00:00+00:00",
        "proof_kind": "IDENTITY_REGISTRY",
        "supports_label": label,
        "identity_consistent": identity_consistent,
        "contradiction_unresolved": contradiction,
    }


def _independent_proofs(
    label: str = "TOP1_CORRECT",
    case_id: str = "case-1",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _proof(
                "sirene",
                family="SIRENE_SNAPSHOT",
                group=SIRENE_INDEPENDENCE_GROUP,
                label=label,
                case_id=case_id,
            ),
            _proof(
                "entity-site",
                family="OFFICIAL_ENTITY_SITE",
                group="ENTITY_ALPHA_OFFICIAL",
                label=label,
                case_id=case_id,
            ),
        ]
    )


def _judgment(
    label: str = "TOP1_CORRECT",
    *,
    target: str | None = TOP1,
    refs: list[str] | None = None,
    case_id: str = "case-1",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_case_id": case_id,
                "adjudication_label": label,
                "validated_correct_siret": target,
                "evidence_ref_ids_json": json.dumps(
                    refs if refs is not None else ["sirene", "entity-site"]
                ),
                "adjudication_reason": "Two identity proofs agree.",
                "adjudication_rule_version": "v4.4-test-1",
                "adjudicated_at": "2026-07-27T12:00:00+00:00",
            }
        ]
    )


def test_two_independent_identity_proofs_create_training_targets() -> None:
    result = build_adjudications(
        _facts(),
        _independent_proofs(),
        _judgment(),
    ).iloc[0]

    assert bool(result["evidence_validated"])
    assert bool(result["training_eligible"])
    assert result["acceptor_target"] == 1
    assert bool(result["ranker_eligible"])
    assert result["ranker_target_siret"] == TOP1
    assert result["independent_evidence_group_count"] == 2
    assert result["sirene_correlated_proof_count"] == 1


def test_two_sirene_views_are_one_source_and_never_training_grade() -> None:
    proofs = pd.DataFrame(
        [
            _proof(
                "sirene-snapshot",
                family="SIRENE_SNAPSHOT",
                group=SIRENE_INDEPENDENCE_GROUP,
            ),
            _proof(
                "sirene-api",
                family="SIRENE_DERIVED_RECHERCHE_ENTREPRISES_API",
                group=SIRENE_INDEPENDENCE_GROUP,
            ),
        ]
    )
    result = build_adjudications(
        _facts(),
        proofs,
        _judgment(refs=["sirene-snapshot", "sirene-api"]),
    ).iloc[0]

    assert result["cited_proof_count"] == 2
    assert result["independent_evidence_group_count"] == 1
    assert result["sirene_correlated_proof_count"] == 2
    assert not bool(result["evidence_validated"])
    assert not bool(result["training_eligible"])


def test_sirene_source_cannot_claim_a_forged_independent_group() -> None:
    proofs = _independent_proofs()
    proofs.loc[proofs["proof_id"].eq("sirene"), "independence_group"] = "FAKE_SECOND"

    with pytest.raises(ValueError, match="SIRENE-correlated"):
        build_adjudications(_facts(), proofs, _judgment())


def test_one_producer_cannot_claim_two_independent_groups() -> None:
    proofs = _independent_proofs()
    proofs.loc[
        proofs["proof_id"].eq("entity-site"), "producer"
    ] = proofs.loc[proofs["proof_id"].eq("sirene"), "producer"].iloc[0]

    with pytest.raises(ValueError, match="One producer cannot claim"):
        build_adjudications(_facts(), proofs, _judgment())


@pytest.mark.parametrize(
    ("label", "target", "message"),
    [
        ("TOP1_CORRECT", ALTERNATIVE, "equal to top1"),
        ("TOP1_WRONG", TOP1, "cannot equal"),
        ("AMBIGUOUS", ALTERNATIVE, "cannot carry"),
        ("UNRESOLVED", ALTERNATIVE, "cannot carry"),
    ],
)
def test_exact_target_invariants_are_blocking(
    label: str,
    target: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_adjudications(
            _facts(),
            _independent_proofs(label),
            _judgment(label, target=target),
        )


def test_wrong_top1_without_known_replacement_trains_only_acceptor() -> None:
    result = build_adjudications(
        _facts(),
        _independent_proofs("TOP1_WRONG"),
        _judgment("TOP1_WRONG", target=None),
    ).iloc[0]

    assert bool(result["training_eligible"])
    assert result["acceptor_target"] == 0
    assert result["ranker_target_siret"] is None
    assert not bool(result["ranker_eligible"])


def test_wrong_top1_replacement_must_be_naturally_in_frozen_pool_for_ranker() -> None:
    result = build_adjudications(
        _facts(),
        _independent_proofs("TOP1_WRONG"),
        _judgment("TOP1_WRONG", target=ALTERNATIVE),
    ).iloc[0]

    assert result["acceptor_target"] == 0
    assert result["ranker_target_siret"] == ALTERNATIVE
    assert bool(result["ranker_eligible"])

    missing = "33333333300003"
    result_missing = build_adjudications(
        _facts(),
        _independent_proofs("TOP1_WRONG"),
        _judgment("TOP1_WRONG", target=missing),
    ).iloc[0]
    assert result_missing["ranker_target_siret"] == missing
    assert not bool(result_missing["ranker_eligible"])


def test_ambiguous_is_negative_for_acceptor_and_never_for_ranker() -> None:
    result = build_adjudications(
        _facts(),
        _independent_proofs("AMBIGUOUS"),
        _judgment("AMBIGUOUS", target=None),
    ).iloc[0]

    assert bool(result["training_eligible"])
    assert result["acceptor_target"] == 0
    assert result["ranker_target_siret"] is None
    assert not bool(result["ranker_eligible"])


def test_unresolved_is_never_training_eligible() -> None:
    result = build_adjudications(
        _facts(),
        pd.DataFrame(columns=list(_independent_proofs().columns)),
        _judgment("UNRESOLVED", target=None, refs=[]),
    ).iloc[0]

    assert not bool(result["evidence_validated"])
    assert not bool(result["training_eligible"])
    assert pd.isna(result["acceptor_target"])
    assert not bool(result["ranker_eligible"])


def test_derived_eligibility_cannot_be_supplied_by_judgment() -> None:
    judgment = _judgment()
    judgment["training_eligible"] = True

    with pytest.raises(ValueError, match="derived columns"):
        build_adjudications(_facts(), _independent_proofs(), judgment)


def test_address_only_or_model_score_cannot_be_an_identity_proof() -> None:
    for proof_kind in ("ADDRESS_ONLY", "MODEL_SCORE"):
        proofs = _independent_proofs()
        proofs.loc[
            proofs["proof_id"].eq("entity-site"), "proof_kind"
        ] = proof_kind
        with pytest.raises(ValueError, match="identity evidence"):
            build_adjudications(_facts(), proofs, _judgment())


def test_contradiction_or_identity_failure_prevents_training() -> None:
    for field in ("contradiction_unresolved", "identity_consistent"):
        proofs = _independent_proofs()
        proof_id = proofs["proof_id"].eq("entity-site")
        proofs.loc[proof_id, field] = field == "contradiction_unresolved"
        result = build_adjudications(_facts(), proofs, _judgment()).iloc[0]
        assert not bool(result["evidence_validated"])
        assert not bool(result["training_eligible"])


def test_immutable_artifact_has_hashes_and_recomputes_from_inputs(
    tmp_path: Path,
) -> None:
    facts_path = tmp_path / "facts.parquet"
    proofs_path = tmp_path / "proofs.parquet"
    judgments_path = tmp_path / "judgments.parquet"
    _facts().to_parquet(facts_path, index=False)
    _independent_proofs().to_parquet(proofs_path, index=False)
    _judgment().to_parquet(judgments_path, index=False)

    artifact = build_artifact(
        facts_path=facts_path,
        proofs_path=proofs_path,
        judgments_path=judgments_path,
        output_root=tmp_path / "artifacts",
    )
    manifest = json.loads((artifact / "manifest.json").read_text())

    assert manifest["invariants"]["training_eligible_is_derived"] is True
    assert manifest["invariants"]["sirene_views_share_one_independence_group"] is True
    assert manifest["outputs"]["adjudications.parquet"] == file_sha256(
        artifact / "adjudications.parquet"
    )
    validate_artifact(artifact)

    with pytest.raises(FileExistsError, match="Immutable"):
        build_artifact(
            facts_path=facts_path,
            proofs_path=proofs_path,
            judgments_path=judgments_path,
            output_root=tmp_path / "artifacts",
        )

    with (artifact / "adjudications.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="output hash mismatch"):
        validate_artifact(artifact)
