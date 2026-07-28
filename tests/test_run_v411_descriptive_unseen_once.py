from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.run_v411_descriptive_unseen_once as runner
from scripts.build_v411_input_blind_dataset import RANKER_C_FEATURE_ORDER
from scripts.build_v411_unseen_qualification import LABEL_COLUMNS
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy


TAXONOMY = SiteFunctionTaxonomy.load(
    Path("config/v4_9_site_function_taxonomy.json")
)


class _Ranker:
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return matrix[:, 0]


class _Acceptor:
    def __init__(self, scores: list[float]) -> None:
        self.scores = np.asarray(scores, dtype=float)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        assert matrix.shape[0] == len(self.scores)
        return np.column_stack([1.0 - self.scores, self.scores])


def _candidate(
    *,
    query_id: str,
    siret: str,
    retrieval_rank: int,
    score_feature: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        name: 0.0 for name in RANKER_C_FEATURE_ORDER
    }
    row.update(
        {
            "query_id": query_id,
            "candidate_siret": siret,
            "candidate_siren": siret[:9],
            "candidate_state": "A",
            "retrieval_rank": retrieval_rank,
            "enseigne1": None,
            "enseigne2": None,
            "enseigne3": None,
            "denomination_usuelle": None,
            "activity_code": None,
            "has_any_name": score_feature,
        }
    )
    return row


def _query(query_id: str) -> dict[str, object]:
    return {
        "query_id": query_id,
        "crm_record_id": f"crm-{query_id}",
        "crm_name": "ENTREPRISE TEST",
        "crm_address": "1 RUE DE PARIS",
        "crm_postcode": "75001",
        "crm_city": "PARIS",
        "crm_insee": "75056",
    }


def test_ledger_is_exclusive_and_labels_are_phase_gated(tmp_path: Path) -> None:
    ledger = tmp_path / "one-shot.json"
    predictions_path = tmp_path / "predictions.parquet"
    prediction = {
        column: None for column in runner.RAW_PREDICTION_COLUMNS
    }
    prediction.update(
        {
            "query_id": "q1",
            "crm_record_id": "crm-q1",
            "candidate_count": 0,
            "threshold": runner.FIXED_THRESHOLD,
            "decision": "REVIEW",
            "review_reason": "NO_CANDIDATE",
        }
    )
    runner.create_exclusive_ledger(
        ledger, {"run_id": "run-1", "phase": "PREFLIGHT"}
    )
    with pytest.raises(FileExistsError):
        runner.create_exclusive_ledger(
            ledger, {"run_id": "run-2", "phase": "PREFLIGHT"}
        )

    calls: list[tuple[Path, list[str]]] = []

    def loader(path: Path, *, columns: list[str]) -> pd.DataFrame:
        calls.append((path, columns))
        return pd.DataFrame(columns=columns)

    with pytest.raises(ValueError, match="before predictions seal"):
        runner.open_labels_after_seal(
            tmp_path / "labels.parquet",
            ledger,
            predictions_path,
            expected_query_ids={"q1"},
            loader=loader,
        )
    assert calls == []

    pd.DataFrame(
        [prediction], columns=runner.RAW_PREDICTION_COLUMNS
    ).to_parquet(predictions_path, index=False)
    runner.update_ledger(
        ledger,
        expected_run_id="run-1",
        phase="PREDICTIONS_SEALED",
        predictions_sha256=runner.file_sha256(predictions_path),
    )
    result = runner.open_labels_after_seal(
        tmp_path / "labels.parquet",
        ledger,
        predictions_path,
        expected_query_ids={"q1"},
        loader=loader,
    )
    assert result.empty
    assert calls == [(tmp_path / "labels.parquet", LABEL_COLUMNS)]


def test_labels_refuse_tampered_or_incomplete_sealed_predictions(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "one-shot.json"
    predictions_path = tmp_path / "predictions.parquet"
    prediction = {
        column: None for column in runner.RAW_PREDICTION_COLUMNS
    }
    prediction.update(
        {
            "query_id": "q1",
            "crm_record_id": "crm-q1",
            "candidate_count": 0,
            "threshold": runner.FIXED_THRESHOLD,
            "decision": "REVIEW",
            "review_reason": "NO_CANDIDATE",
        }
    )
    frame = pd.DataFrame(
        [prediction], columns=runner.RAW_PREDICTION_COLUMNS
    )
    frame.to_parquet(predictions_path, index=False)
    runner.create_exclusive_ledger(
        ledger,
        {
            "run_id": "run-1",
            "phase": "PREDICTIONS_SEALED",
            "predictions_sha256": runner.file_sha256(predictions_path),
        },
    )

    with pytest.raises(ValueError, match="population changed"):
        runner.open_labels_after_seal(
            tmp_path / "labels.parquet",
            ledger,
            predictions_path,
            expected_query_ids={"q1", "q2"},
        )

    frame.assign(decision="AUTO_MATCH").to_parquet(
        predictions_path, index=False
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.open_labels_after_seal(
            tmp_path / "labels.parquet",
            ledger,
            predictions_path,
            expected_query_ids={"q1"},
        )


def test_ranker_forbids_truth_and_uses_exact_tie_break() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(
                query_id="q1",
                siret="33333333300003",
                retrieval_rank=3,
                score_feature=0.5,
            ),
            _candidate(
                query_id="q1",
                siret="22222222200002",
                retrieval_rank=2,
                score_feature=0.8,
            ),
            _candidate(
                query_id="q1",
                siret="11111111100001",
                retrieval_rank=1,
                score_feature=0.8,
            ),
        ]
    )
    candidates["is_ground_truth"] = 0
    with pytest.raises(ValueError, match="truth metadata"):
        runner.score_ranker_candidates(_Ranker(), candidates)

    scored = runner.score_ranker_candidates(
        _Ranker(), candidates.drop(columns="is_ground_truth")
    )
    assert scored["candidate_siret"].tolist() == [
        "11111111100001",
        "22222222200002",
        "33333333300003",
    ]
    assert scored["ranker_rank"].tolist() == [1, 2, 3]
    assert not {
        "is_ground_truth",
        "ground_truth_siret",
        "ground_truth_siren",
        "label_kind",
    } & set(scored.columns)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: pd.concat(
                [frame, frame.iloc[[0]]], ignore_index=True
            ),
            "duplicate candidate",
        ),
        (
            lambda frame: frame.assign(candidate_state="F"),
            "non-active",
        ),
        (
            lambda frame: frame.assign(retrieval_rank=[1, 3]),
            "not contiguous",
        ),
    ],
)
def test_candidate_pool_schema_fails_closed(mutate, message: str) -> None:
    valid = pd.DataFrame(
        [
            _candidate(
                query_id="q1",
                siret="11111111100001",
                retrieval_rank=1,
                score_feature=0.9,
            ),
            _candidate(
                query_id="q1",
                siret="22222222200002",
                retrieval_rank=2,
                score_feature=0.8,
            ),
        ]
    )
    with pytest.raises(ValueError, match=message):
        runner.validate_candidate_pools(mutate(valid))


def test_blind_predictions_keep_empty_pool_in_review() -> None:
    queries = pd.DataFrame([_query("q1"), _query("q2")])
    candidates = pd.DataFrame(
        [
            _candidate(
                query_id="q1",
                siret="11111111100001",
                retrieval_rank=1,
                score_feature=0.9,
            )
        ]
    )
    scored = runner.score_ranker_candidates(_Ranker(), candidates)
    predictions, scenes = runner.build_blind_predictions(
        queries=queries,
        scored_candidates=scored,
        taxonomy=TAXONOMY,
        acceptor=_Acceptor([0.99, 0.99]),
    )

    assert list(predictions.columns) == runner.RAW_PREDICTION_COLUMNS
    assert list(scenes["query_id"]) == ["q1", "q2"]
    q1 = predictions.set_index("query_id").loc["q1"]
    q2 = predictions.set_index("query_id").loc["q2"]
    assert q1["decision"] == "AUTO_MATCH"
    assert q1["predicted_siret"] == "11111111100001"
    assert q2["decision"] == "REVIEW"
    assert q2["review_reason"] == "NO_CANDIDATE"
    assert pd.isna(q2["predicted_siret"])
    assert (predictions["threshold"] == runner.FIXED_THRESHOLD).all()


def test_evaluation_preserves_raw_decision_and_separates_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(
        runner,
        "EXPECTED_COHORT_COUNTS",
        {"DESCRIPTIVE_UNSEEN_BLIND_222": 2},
    )
    predictions = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "crm_record_id": "crm-q1",
                "predicted_siret": "11111111100001",
                "predicted_siren": "111111111",
                "candidate_count": 1,
                "ranker_score_top1": 0.9,
                "acceptor_score": 0.99,
                "threshold": runner.FIXED_THRESHOLD,
                "decision": "AUTO_MATCH",
                "review_reason": None,
            },
            {
                "query_id": "q2",
                "crm_record_id": "crm-q2",
                "predicted_siret": "22222222200002",
                "predicted_siren": "222222222",
                "candidate_count": 1,
                "ranker_score_top1": 0.8,
                "acceptor_score": 0.98,
                "threshold": runner.FIXED_THRESHOLD,
                "decision": "AUTO_MATCH",
                "review_reason": None,
            },
        ],
        columns=runner.RAW_PREDICTION_COLUMNS,
    )
    label_rows = []
    for query_id, kind, siret in [
        ("q1", "MATCH_EXACT", "11111111100001"),
        ("q2", "UNRESOLVED", None),
    ]:
        row = {column: None for column in LABEL_COLUMNS}
        row.update(
            {
                "query_id": query_id,
                "label_kind": kind,
                "ground_truth_siret": siret,
                "ground_truth_siren": siret[:9] if siret else None,
            }
        )
        label_rows.append(row)
    labels = pd.DataFrame(label_rows, columns=LABEL_COLUMNS)
    mapping = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "source_row_number": 1,
                "cohort": "DESCRIPTIVE_UNSEEN_BLIND_222",
            },
            {
                "query_id": "q2",
                "source_row_number": 2,
                "cohort": "DESCRIPTIVE_UNSEEN_BLIND_222",
            },
        ]
    )
    scored = pd.DataFrame(
        [
            {"query_id": "q1", "candidate_siret": "11111111100001"},
            {"query_id": "q2", "candidate_siret": "22222222200002"},
        ]
    )

    evaluated, overlay, metrics = runner.evaluate_after_seal(
        predictions=predictions,
        scored_candidates=scored,
        labels=labels,
        mapping=mapping,
    )

    assert evaluated.set_index("query_id").loc["q2", "decision"] == "AUTO_MATCH"
    q2_overlay = overlay.set_index("query_id").loc["q2"]
    assert q2_overlay["raw_decision"] == "AUTO_MATCH"
    assert q2_overlay["overlay_decision"] == "REVIEW"
    assert q2_overlay["overlay_reason"] == "UNRESOLVED_LABEL_OVERLAY"
    assert metrics["ALL_225"]["unresolved_auto"] == 1
    assert metrics["ALL_225"]["confirmed_error_auto"] == 0
    assert metrics["ALL_225"]["unverifiable_auto"] == 1
    assert metrics["ALL_225"]["precision_evaluable"] == 1.0
    assert metrics["ALL_225"]["precision_conservative_lower_bound"] == 0.5
    assert metrics["ALL_225"]["identifiable_count"] == 1
    assert metrics["ALL_225"]["coverage_identifiable"] == 1.0
    assert metrics["ALL_225"]["non_unresolved_count"] == 1


def test_evaluation_refuses_population_or_cohort_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(
        runner,
        "EXPECTED_COHORT_COUNTS",
        {"DESCRIPTIVE_UNSEEN_BLIND_222": 1},
    )
    prediction = {
        column: None for column in runner.RAW_PREDICTION_COLUMNS
    }
    prediction.update(
        {
            "query_id": "q1",
            "crm_record_id": "crm-q1",
            "candidate_count": 0,
            "threshold": runner.FIXED_THRESHOLD,
            "decision": "REVIEW",
            "review_reason": "NO_CANDIDATE",
        }
    )
    predictions = pd.DataFrame(
        [prediction], columns=runner.RAW_PREDICTION_COLUMNS
    )
    label = {column: None for column in LABEL_COLUMNS}
    label.update({"query_id": "q2", "label_kind": "UNRESOLVED"})
    labels = pd.DataFrame([label], columns=LABEL_COLUMNS)
    mapping = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "source_row_number": 1,
                "cohort": "DESCRIPTIVE_UNSEEN_BLIND_222",
            }
        ]
    )

    with pytest.raises(ValueError, match="populations differ"):
        runner.evaluate_after_seal(
            predictions=predictions,
            scored_candidates=pd.DataFrame(
                columns=["query_id", "candidate_siret"]
            ),
            labels=labels,
            mapping=mapping,
        )

    labels.loc[0, "query_id"] = "q1"
    mapping.loc[0, "cohort"] = "EXPOSED_3"
    with pytest.raises(ValueError, match="cohort counts changed"):
        runner.evaluate_after_seal(
            predictions=predictions,
            scored_candidates=pd.DataFrame(
                columns=["query_id", "candidate_siret"]
            ),
            labels=labels,
            mapping=mapping,
        )


def test_execution_lock_pins_transitive_retrieval_and_similarity_runtime() -> None:
    assert {
        "src/xgb_matcher/candidates.py",
        "src/xgb_matcher/fusion.py",
        "src/xgb_matcher/retrieval_config.py",
    }.issubset(runner.INFERENCE_SOURCE_PATHS)
    package_sources = {
        str(path)
        for path in Path("src/xgb_matcher").glob("*.py")
    }
    assert package_sources.issubset(runner.INFERENCE_SOURCE_PATHS)
    assert runner.EXPECTED_RUNTIME["rapidfuzz"] == "3.14.3"
    assert runner.EXPECTED_RUNTIME["scipy"] == "1.16.3"
    assert runner.GLOBAL_OPENING_LEDGER.is_absolute()
    assert runner.GLOBAL_OPENING_LEDGER.name == "OPENING_LEDGER.json"


def test_execution_lock_rejects_extra_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner_path = Path(runner.__file__).resolve()
    lock = {
        "schema_version": runner.LOCK_SCHEMA_VERSION,
        "purpose": "DESCRIPTIVE_UNSEEN_225_ONCE",
        "runner_sha256": runner.file_sha256(runner_path),
        "runner_commit": "deadbeef",
        "sanitized_manifest_sha256": "sanitized",
        "qualification_manifest_sha256": "qualification",
        "labels_sha256": "labels",
        "stack_manifest_sha256": runner.EXPECTED_STACK_SHA256,
        "ranker_model_sha256": runner.EXPECTED_RANKER_MODEL_SHA256,
        "acceptor_model_sha256": runner.EXPECTED_ACCEPTOR_MODEL_SHA256,
        "threshold": runner.FIXED_THRESHOLD,
        "snapshot_sha256": runner.EXPECTED_SNAPSHOT_SHA256,
        "partitions_signature": runner.EXPECTED_PARTITIONS_SIGNATURE,
        "tfidf_cache_namespace": runner.EXPECTED_TFIDF_NAMESPACE,
        "runtime": runner.EXPECTED_RUNTIME,
        "source_hashes": {},
        "unexpected": True,
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="fields changed"):
        runner.validate_execution_lock(
            path,
            sanitized_manifest_sha256="sanitized",
            qualification_manifest_sha256="qualification",
            labels_sha256="labels",
            verify_git=False,
        )
