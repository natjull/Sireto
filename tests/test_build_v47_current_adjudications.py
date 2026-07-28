from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts import build_v47_current_adjudications as subject


def _docket() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_case_id": "case-drift",
                "query_id": "query-drift",
                "service_id": "service-drift",
                "sampling_stratum": "TARGETED",
                "siret_to_adjudicate": "12345678900011",
                "scene_status": "SCENE_DRIFT",
                "evidence_partition": "targeted",
            }
        ]
    )


def _official() -> pd.DataFrame:
    payload = json.dumps(
        {"results": [{"siege": {"siret": "12345678900011"}}]}
    )
    return pd.DataFrame(
        [
            {
                "audit_case_id": "case-drift",
                "siret_to_adjudicate": "12345678900011",
                "query_kind": "TOP1_SIRET",
                "http_status": 200,
                "result_count": 1,
                "payload_json": payload,
                "payload_sha256": "a" * 64,
                "collected_at": "2026-07-27T10:00:00+00:00",
                "source_url": "https://registry.test/siret",
                "source_family": "SIRENE_DERIVED_RECHERCHE_ENTREPRISES_API",
                "independence_group": "SIRENE_REGISTRY",
            }
        ]
    )


def _source(**overrides) -> pd.DataFrame:
    row = {
        "audit_case_id": "case-drift",
        "siret_to_adjudicate": "12345678900011",
        "source_url": "https://entity.test/proof",
        "producer": "ENTITY",
        "source_family": "ENTITY_OFFICIAL_SITE",
        "independence_group": "ENTITY_GROUP",
        "relationship": "CONTRADICTS_CURRENT_TOP1",
        "required_terms": ["alpha site", "10 rue cible"],
        "fact_summary": "La source officielle situe Alpha au 10 rue Cible.",
        "spec_source_index": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _scenes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "audit_case_id": "case-compatible",
                "service_id": "service-compatible",
                "sampling_stratum": "RANDOM_POPULATION",
                "replayed_top1_siret": "99999999900011",
                "scene_status": "SCENE_COMPATIBLE",
                "scene_adjudication_label": "TOP1_CORRECT",
                "scene_acceptor_target": 1,
                "scene_training_eligible": True,
            },
            {
                "audit_case_id": "case-drift",
                "service_id": "service-drift",
                "sampling_stratum": "TARGETED",
                "replayed_top1_siret": "12345678900011",
                "scene_status": "SCENE_DRIFT",
                "scene_adjudication_label": None,
                "scene_acceptor_target": None,
                "scene_training_eligible": False,
            },
        ]
    )


def test_public_proof_is_archived_and_label_is_derived(tmp_path):
    sources = _source()

    def fetcher(url, timeout):
        assert url == "https://entity.test/proof"
        return (
            200,
            "text/html",
            b"<html><body>Alpha Site - 10 rue Cible</body></html>",
            url,
        )

    public = subject.archive_public_sources(
        sources=sources,
        raw_dir=tmp_path / "raw",
        timeout_seconds=1,
        fetcher=fetcher,
    )
    evidence = pd.concat(
        [
            subject.build_registry_evidence(_docket(), _official()),
            subject.build_public_evidence_rows(public),
        ],
        ignore_index=True,
    )
    adjudications = subject.build_adjudications(_docket(), evidence)
    result = adjudications.iloc[0]

    assert bool(public.loc[0, "usable"])
    assert (tmp_path / "raw").iterdir()
    assert result["adjudication_label"] == "TOP1_WRONG"
    assert bool(result["evidence_validated"])
    assert result["independent_evidence_group_count"] == 2
    assert not bool(result["old_label_transported"])


def test_failed_required_term_stays_unresolved(tmp_path):
    sources = _source()

    def fetcher(url, timeout):
        return 200, "text/html", b"<p>Alpha Site sans adresse</p>", url

    public = subject.archive_public_sources(
        sources=sources,
        raw_dir=tmp_path / "raw",
        timeout_seconds=1,
        fetcher=fetcher,
    )
    evidence = pd.concat(
        [
            subject.build_registry_evidence(_docket(), _official()),
            subject.build_public_evidence_rows(public),
        ],
        ignore_index=True,
    )
    result = subject.build_adjudications(_docket(), evidence).iloc[0]

    assert not bool(public.loc[0, "terms_validated"])
    assert result["adjudication_label"] == "UNRESOLVED"
    assert not bool(result["evidence_validated"])
    assert json.loads(result["evidence_ref_ids_json"]) == []


def test_current_labels_never_transport_old_label_to_drift(tmp_path):
    sources = _source(relationship="SUPPORTS_CURRENT_TOP1")

    def fetcher(url, timeout):
        return (
            200,
            "text/html",
            b"<html><body>Alpha Site, 10 rue Cible</body></html>",
            url,
        )

    public = subject.archive_public_sources(
        sources=sources,
        raw_dir=tmp_path / "raw",
        timeout_seconds=1,
        fetcher=fetcher,
    )
    evidence = pd.concat(
        [
            subject.build_registry_evidence(_docket(), _official()),
            subject.build_public_evidence_rows(public),
        ],
        ignore_index=True,
    )
    adjudications = subject.build_adjudications(_docket(), evidence)
    current = subject.build_current_labels(_scenes(), adjudications)
    compatible = current.loc[
        current["audit_case_id"].eq("case-compatible")
    ].iloc[0]
    drift = current.loc[current["audit_case_id"].eq("case-drift")].iloc[0]

    assert compatible["current_label_origin"] == "V4.4_TRANSPORT_EXACT_TOP1"
    assert compatible["current_adjudication_label"] == "TOP1_CORRECT"
    assert drift["current_label_origin"] == "V4.7_CURRENT_TOP1"
    assert drift["current_adjudication_label"] == "TOP1_CORRECT"
    assert drift["current_top1_siret"] == "12345678900011"


def test_validate_inputs_rejects_evidence_pinned_to_another_siret():
    bad_sources = _source(siret_to_adjudicate="88888888800011")

    with pytest.raises(ValueError, match="pins wrong current SIRET"):
        subject.validate_inputs(
            docket=_docket(),
            official=_official(),
            scenes=_scenes().loc[
                lambda frame: frame["scene_status"].eq("SCENE_DRIFT")
            ],
            sources=bad_sources,
            enforce_canonical=False,
        )


def test_gate_follows_preregistered_thresholds():
    rows = []
    for index in range(172):
        random = index < 57
        reliable = index < 150
        if random:
            label = "TOP1_WRONG" if index < 3 else "TOP1_CORRECT"
        else:
            label = "TOP1_WRONG" if index < 77 else "TOP1_CORRECT"
        rows.append(
            {
                "audit_case_id": f"case-{index}",
                "sampling_stratum": (
                    "RANDOM_POPULATION" if random else "TARGETED"
                ),
                "scene_status": "SCENE_COMPATIBLE",
                "current_adjudication_label": label if reliable else "UNRESOLVED",
                "current_evidence_validated": reliable,
                "current_independent_evidence_group_count": 2 if reliable else 0,
                "current_label_origin": "V4.4_TRANSPORT_EXACT_TOP1",
            }
        )
    current = pd.DataFrame(rows)
    adjudications = pd.DataFrame(
        {
            "audit_case_id": [f"v47-{index}" for index in range(37)],
            "evidence_validated": [True] * 15 + [False] * 22,
            "adjudication_label": ["TOP1_WRONG"] * 15 + ["UNRESOLVED"] * 22,
        }
    )

    gate = subject.evaluate_gate(adjudications, current)

    assert gate["verdict"] == "GO_ACCEPTOR_FEASIBILITY"
    assert gate["quality_gate_passed"]
