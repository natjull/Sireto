from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = REPOSITORY / "config/v4_13_fresh_labels_minimal_plan.json"


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def load_plan() -> dict:
    raw = PLAN_PATH.read_bytes()
    plan = json.loads(raw)
    assert raw == canonical_json(plan)
    return plan


def test_contract_and_existing_registries_are_exactly_pinned() -> None:
    plan = load_plan()
    contract = plan["contract"]
    assert hashlib.sha256(
        (REPOSITORY / contract["path"]).read_bytes()
    ).hexdigest() == contract["sha256"]
    for registry in plan["registries"].values():
        manifest = Path(registry["root"]) / "manifest.json"
        if not manifest.exists():
            manifest = Path(registry["root"]) / "payload_manifest.json"
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
            registry["manifest_sha256"]
        )


def test_gate_zero_checks_source_and_truth_before_any_model() -> None:
    plan = load_plan()
    assert plan["source"]["required_files"] == [
        "collection_manifest.json",
        "crm_source",
        "authoritative_mapping",
    ]
    assert plan["source"]["optional_stopping_allowed"] is False
    assert plan["source"]["all_source_rows_in_denominator"] is True
    assert plan["gate_zero"] == {
        "minimum_match_exact": 657,
        "minimum_coverage": 0.8,
        "maximum_forbidden_overlap": 0,
        "requires_complete_frame": True,
        "verdicts": {
            "waiting": "WAITING_FOR_NEW_SOURCE",
            "go": "GO_BUILD_FRESH_LABELS",
            "pivot": "PIVOT_SOURCE_EVIDENCE",
            "stop": "STOP_INTEGRITY_OR_CONTAMINATION",
        },
    }
    forbidden = set(plan["truth_policy"]["forbidden"])
    assert {
        "RETRIEVAL",
        "CANDIDATE",
        "HIT",
        "RANK",
        "SCORE",
        "PREDICTION",
        "MODEL_OUTPUT",
        "USER_VALIDATION",
    } <= forbidden


def test_query_oracle_separation_and_split_are_closed() -> None:
    plan = load_plan()
    assert plan["outputs"]["trees"] == ["queries", "oracle", "audit"]
    assert {"SIRET", "SIREN", "EVIDENCE", "LABEL"} <= set(
        plan["outputs"]["query_forbidden_fields"]
    )
    split = plan["split"]
    assert split["unit"] == "AUTHORITATIVE_SIREN_CONNECTED_COMPONENT"
    assert split["proportions"] == {"fit": 0.7, "dev": 0.15, "test": 0.15}
    assert sum(split["proportions"].values()) == 1.0
    assert split["frozen_before_retrieval"] is True
    assert split["test_open_once_after_full_freeze"] is True


def test_retrieval_and_model_order_preserve_active_directive() -> None:
    plan = load_plan()
    retrieval = plan["retrieval_gate"]
    assert retrieval["candidate_maximum"] == 100
    assert retrieval["candidate_maximum_is_absolute"] is True
    assert retrieval["truth_absent_is_error"] is True
    assert retrieval["minimum_identifiable_coverage"] == 0.8
    assert retrieval["minimum_exact_siret_recall_at_100_dev"] == 0.99
    assert set(retrieval["models_frozen_until_go"]) == {
        "RANKER",
        "DECIDER",
        "RISK_MODEL",
        "ACCEPTOR",
    }
    sequence = plan["model_sequence"]
    assert sequence.index("RETRIEVAL_GATE_FIT_DEV") < sequence.index(
        "RANKER_FIT_ON_REAL_POOLS"
    )
    assert sequence.index("RANKER_SIREN_GROUPED_OOF") < sequence.index(
        "QUERY_LEVEL_ACCEPTOR_ON_OOF_SCENES"
    )
    assert sequence[-1] == "OPEN_TEST_ONCE"


def test_local_pki_is_explicitly_out_of_the_north_star_path() -> None:
    plan = load_plan()
    assert plan["purpose"].endswith("WITHOUT_LOCAL_PKI")
    assert plan["threat_model"] == {
        "trusted": "COOPERATIVE_LOCAL_OPERATOR_SINGLE_UID",
        "excluded": "HOSTILE_PROCESS_ALREADY_CONTROLLING_SAME_UID",
        "claims_external_non_repudiation": False,
    }
    serialized = canonical_json(plan).decode("utf-8").lower()
    assert "ksecusedataprotectionkeychain" not in serialized
    assert "secitemadd" not in serialized
    assert plan["supersedes_for_north_star_only"]["mutation_policy"] == (
        "PRESERVE_ALL_V1_AND_S1_ARTIFACTS_IMMUTABLY"
    )


def test_no_implementation_or_source_open_is_authorized_yet() -> None:
    plan = load_plan()
    assert plan["status"] == (
        "PREREGISTERED_WAITING_FOR_TWO_AUDITS_AND_NEW_SOURCE"
    )
    assert set(plan["future_implementation"].values()) == {"UNIMPLEMENTED"}
    assert plan["gates"]["preregistration"] == {
        "independent_audits_required": 2,
        "verdict": "GO_V413_FRESH_LABELS_PREREG",
    }
    assert plan["current_evidence"] == {
        "retrieval_dev_match_exact": 1217,
        "retrieval_dev_recall_at_100_success": 1217,
        "guard_comparison_dev_auto": 614,
        "guard_comparison_dev_errors": 0,
        "local_historical_rows_all_consumed": 23609,
        "fresh_collection_present": False,
    }
