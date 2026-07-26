import json

import pandas as pd
import pytest

from src.xgb_matcher.v41_shadow import (
    FeedbackOutcome,
    InputSiretState,
    ShadowDecision,
    ShadowReviewReason,
    V41ShadowDecision,
    append_feedback_event,
    build_pre_prediction_panel,
    build_shadow_inventory,
    enrich_inventory_siret_state,
    feedback_outcome,
    select_scoreable_rows,
    write_shadow_run,
)


def _source(size=8):
    return pd.DataFrame(
        {
            "SERVICE ID": [f"id-{index}" for index in range(size)],
            "SITE": [f"site-{index}" for index in range(size)],
            "SIRET": [f"12345678{index % 10}00011" for index in range(size)],
        }
    )


def test_inventory_excludes_missing_and_all_consumed_populations():
    source = _source(4)
    source.loc[3, "SERVICE ID"] = None
    inventory = build_shadow_inventory(
        source,
        denylists={"OLD_TEST": {"id-0"}, "FRESH_HOLDOUT": {"id-1"}},
    )
    assert inventory["eligible_for_shadow"].tolist() == [False, False, True, False]
    assert inventory["exclusion_reason"].tolist() == [
        "CONSUMED_OLD_TEST",
        "CONSUMED_FRESH_HOLDOUT",
        None,
        "MISSING_SERVICE_ID",
    ]
    scoreable = select_scoreable_rows(source, inventory)
    assert scoreable["service_id"].tolist() == ["id-2"]


def test_inventory_refuses_duplicate_non_empty_service_ids():
    source = _source(3)
    source.loc[2, "SERVICE ID"] = "id-1"
    with pytest.raises(ValueError, match="must be unique"):
        build_shadow_inventory(source, denylists={})


def test_short_numeric_input_is_invalid_not_zero_padded():
    source = pd.DataFrame({"SERVICE ID": ["short"], "SIRET": ["123"]})
    inventory = build_shadow_inventory(source, denylists={})
    assert inventory.loc[0, "input_siret"] is None
    assert inventory.loc[0, "input_siret_state"] == "INVALID"


def test_siret_state_enrichment_is_snapshot_relative():
    source = pd.DataFrame(
        {
            "SERVICE ID": ["active", "closed", "absent", "invalid"],
            "SIRET": [
                "12345678900011",
                "22222222200011",
                "33333333300011",
                "bad",
            ],
        }
    )
    inventory = build_shadow_inventory(source, denylists={})
    sirene = pd.DataFrame(
        {
            "siret": [
                "12345678900011",
                "12345678900029",
                "22222222200011",
                "22222222200029",
            ],
            "siren": ["123456789", "123456789", "222222222", "222222222"],
            "etatAdministratifEtablissement": ["A", "A", "F", "A"],
        }
    )
    enriched = enrich_inventory_siret_state(inventory, sirene)
    assert enriched["input_siret_state"].tolist() == [
        "ACTIVE",
        "CLOSED",
        "NOT_FOUND",
        "INVALID",
    ]
    assert enriched["active_sibling_count"].tolist() == [1, 1, 0, 0]


def test_panel_is_deterministic_disjoint_and_pre_prediction():
    records = []
    strata = [
        ("CLOSED", 1, 1),
        ("CLOSED", 0, 0),
        ("NOT_FOUND", 0, 0),
        ("ACTIVE", 1, 2),
    ]
    for group, (state, siblings, active_count) in enumerate(strata):
        for index in range(4):
            records.append(
                {
                    "service_id": f"s-{group}-{index}",
                    "eligible_for_shadow": True,
                    "input_siret_state": state,
                    "active_sibling_count": siblings,
                    "active_siret_count_for_siren": active_count,
                }
            )
    for index in range(8):
        records.append(
            {
                "service_id": f"r-{index}",
                "eligible_for_shadow": True,
                "input_siret_state": "ACTIVE",
                "active_sibling_count": 0,
                "active_siret_count_for_siren": 1,
            }
        )
    inventory = pd.DataFrame(records)
    quotas = {
        "REPRESENTATIVE": 4,
        "CLOSED_WITH_ACTIVE_SIBLING": 2,
        "CLOSED_WITHOUT_ACTIVE_SIBLING": 2,
        "ABSENT_OR_INVALID": 2,
        "ACTIVE_MULTI_SITE": 2,
    }
    first = build_pre_prediction_panel(inventory, seed=7, quotas=quotas)
    second = build_pre_prediction_panel(inventory, seed=7, quotas=quotas)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12
    assert first["service_id"].is_unique
    assert "decision" not in first


def _small_export_inputs(run_id):
    inventory = pd.DataFrame(
        {
            "service_id": ["ok", "excluded"],
            "eligible_for_shadow": [True, False],
            "exclusion_reason": [None, "CONSUMED_OLD_TEST"],
            "input_siret_state": ["ACTIVE", "ACTIVE"],
        }
    )
    decision = V41ShadowDecision(
        service_id="ok",
        decision=ShadowDecision.AUTO_MATCH,
        predicted_siret="12345678900011",
        predicted_siren="123456789",
        confidence=0.9,
        confidence_kind="ROUTING_SCORE_UNCALIBRATED",
        review_reason=None,
        model_bundle_id="bundle",
        dataset_manifest_id="dataset",
        shadow_run_id=run_id,
        input_siret_state=InputSiretState.ACTIVE,
        candidate_count=1,
    )
    candidates = pd.DataFrame(
        {
            "service_id": ["ok"],
            "candidate_siret": ["12345678900011"],
            "rank": [1],
            "etat_admin": ["A"],
            "extra_future_evidence": ["kept"],
        }
    )
    panel = inventory.loc[[0]].copy()
    return inventory, [decision], candidates, panel


def test_atomic_shadow_export_has_hash_manifest_and_legacy_status(tmp_path):
    inventory, decisions, candidates, panel = _small_export_inputs("run-1")
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    result = write_shadow_run(
        output_root=tmp_path / "shadow",
        run_id="run-1",
        inventory=inventory,
        decisions=decisions,
        candidates_top10=candidates,
        evidence=[{"service_id": "ok", "proof": {"kind": "name"}}],
        panel=panel,
        input_artifacts={"source": source},
        run_metadata={"model": "test"},
    )
    assert result.name == "run-1"
    assert not list((tmp_path / "shadow").glob(".run-1.tmp-*"))
    exported = pd.read_parquet(result / "decisions.parquet")
    assert exported.loc[0, "routing_status"] == "AUTO"
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["invariants"]["excluded_rows_scored"] == 0
    assert manifest["row_counts"]["decisions"] == 1
    assert manifest["outputs"]["decisions.parquet"]
    with pytest.raises(FileExistsError):
        write_shadow_run(
            output_root=tmp_path / "shadow",
            run_id="run-1",
            inventory=inventory,
            decisions=decisions,
            candidates_top10=candidates,
            evidence=[],
            panel=panel,
            input_artifacts={"source": source},
            run_metadata={},
        )


def test_shadow_export_rejects_any_excluded_decision(tmp_path):
    inventory, decisions, candidates, panel = _small_export_inputs("run-2")
    decisions.append(
        V41ShadowDecision(
            service_id="excluded",
            decision=ShadowDecision.REVIEW,
            predicted_siret=None,
            predicted_siren=None,
            confidence=0.1,
            confidence_kind="ROUTING_SCORE_UNCALIBRATED",
            review_reason=ShadowReviewReason.NO_CANDIDATE,
            model_bundle_id="bundle",
            dataset_manifest_id="dataset",
            shadow_run_id="run-2",
        )
    )
    with pytest.raises(ValueError, match="exactly match eligible"):
        write_shadow_run(
            output_root=tmp_path,
            run_id="run-2",
            inventory=inventory,
            decisions=decisions,
            candidates_top10=candidates,
            evidence=[],
            panel=panel,
            input_artifacts={},
            run_metadata={},
        )


def test_shadow_export_rejects_excluded_evidence(tmp_path):
    inventory, decisions, candidates, panel = _small_export_inputs("run-3")
    with pytest.raises(ValueError, match="Evidence contains an excluded"):
        write_shadow_run(
            output_root=tmp_path,
            run_id="run-3",
            inventory=inventory,
            decisions=decisions,
            candidates_top10=candidates,
            evidence=[{"service_id": "excluded", "proof": "forbidden"}],
            panel=panel,
            input_artifacts={},
            run_metadata={},
        )


def test_feedback_is_append_only_and_silence_is_unknown(tmp_path):
    assert (
        feedback_outcome(
            proposed_siret="12345678900011",
            later_crm_siret="12345678900011",
            explicit_crm_change=False,
        )
        == FeedbackOutcome.UNKNOWN
    )
    journal = tmp_path / "feedback" / "events.jsonl"
    first = append_feedback_event(
        journal,
        shadow_run_id="run",
        service_id="one",
        proposed_siret="12345678900011",
        later_crm_siret="12345678900011",
        explicit_crm_change=True,
        source_snapshot_id="crm-later-1",
    )
    second = append_feedback_event(
        journal,
        shadow_run_id="run",
        service_id="two",
        proposed_siret="12345678900011",
        later_crm_siret="99999999900011",
        explicit_crm_change=True,
        source_snapshot_id="crm-later-1",
    )
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [event["event_id"] for event in events] == [
        first["event_id"],
        second["event_id"],
    ]
    assert [event["outcome"] for event in events] == ["CONFIRMED", "CORRECTED"]
    assert all(not event["eligible_for_automatic_retraining"] for event in events)
