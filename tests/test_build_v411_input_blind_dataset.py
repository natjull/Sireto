from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import ANY

import pandas as pd
import pytest

from scripts import build_v411_input_blind_dataset as subject


def _query(query_id: str = "q1") -> dict:
    return {
        "query_id": query_id,
        "crm_record_id": f"crm-{query_id}",
        "crm_name": "Ecole Alpha",
        "crm_address": "1 rue Alpha",
        "crm_postcode": "75001",
        "crm_city": "Paris",
        "crm_insee": "75056",
        "crm_name_norm": "ecole alpha",
        "crm_address_norm": "1 rue alpha",
        "crm_city_norm": "paris",
    }


def _candidate(siret: str, *, score: float = 0.5, state: str = "A") -> dict:
    return {
        "siret": siret,
        "siren": siret[:9],
        "etat_admin": state,
        "nom_etablissement": "ECOLE ALPHA",
        "denomination_unite_legale": "ECOLE ALPHA",
        "adresse": "1 RUE ALPHA",
        "postcode": "75001",
        "city": "PARIS",
        "insee": "75056",
        "rrf_score": score,
        "retrieval_source": "sparse_name+sparse_address",
        "retrieval_channel_count": 2,
    }


@dataclass
class _SparseResult:
    candidates: list[dict]
    gt_was_injected: bool = False
    idf_map: dict[str, float] = field(default_factory=dict)
    default_idf: float = 1.0


class _StateStore:
    def __init__(self, states: dict[str, str] | None = None):
        self.states = states or {}

    def get_candidate_scene_details(self, sirets):
        return {
            siret: {
                "candidate_state": self.states.get(siret, "A"),
                "enseigne1": "ECOLE ALPHA",
                "enseigne2": None,
                "enseigne3": None,
                "denomination_usuelle": "ECOLE ALPHA",
                "activity_code": "85.20Z",
            }
            for siret in sirets
        }


class _SparseBuilder:
    def __init__(self, candidates, *, injected=False):
        self.candidates = candidates
        self.injected = injected
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _SparseResult(
            candidates=self.candidates,
            gt_was_injected=self.injected,
        )


def _candidate_row(
    *,
    query_id: str = "q1",
    siret: str = "11111111100011",
    rank: int = 1,
) -> dict:
    row = {
        "query_id": query_id,
        "candidate_siret": siret,
        "candidate_siren": siret[:9],
        "candidate_state": "A",
        "is_ground_truth": 0,
        "retrieval_rank": rank,
        "retrieval_source": "v4.11-sparse",
        "retrieval_channel_count": 1,
        "retrieval_agreement": 0,
        "enseigne1": "ECOLE ALPHA",
        "enseigne2": None,
        "enseigne3": None,
        "denomination_usuelle": "ECOLE ALPHA",
        "activity_code": "85.20Z",
    }
    row.update({name: 0.0 for name in subject.RANKER_C_FEATURE_ORDER})
    row["retrieval_rank_recip"] = 1.0 / rank
    return row


def _population():
    queries = pd.DataFrame([_query("q1"), _query("q2")])
    labels = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": "11111111100011",
                "ground_truth_siren": "111111111",
            },
            {
                "query_id": "q2",
                "label_kind": "AMBIGUOUS",
                "ground_truth_siret": None,
                "ground_truth_siren": None,
            },
        ]
    )
    assignments = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "siren_component_id": "component-1",
                "split": "fit",
                "oof_fold": 0.0,
            },
            {
                "query_id": "q2",
                "siren_component_id": "component-2",
                "split": "dev",
                "oof_fold": 1,
            },
        ]
    )
    return queries, labels, assignments


def test_ranker_c_contract_is_exact_and_contains_no_identifier_feature():
    assert len(subject.RANKER_C_FEATURE_ORDER) == 45
    assert subject.RANKER_C_FEATURE_ORDER[-1] == "retrieval_rank_recip"
    assert not (
        subject.FORBIDDEN_PREDICTION_COLUMNS
        & set(subject.RANKER_C_FEATURE_ORDER)
    )
    assert subject.input_blind_retrieval_config().variant.value == "A"
    assert (
        subject.input_blind_retrieval_config().sparse_config().include_closed
        is False
    )


def test_frozen_folds_cover_fit_and_dev_without_splitting_components():
    queries = pd.DataFrame([_query(f"q{index}") for index in range(5)])
    assignments = pd.DataFrame(
        [
            {
                "query_id": f"q{index}",
                "siren_component_id": f"component-{index}",
                "split": "dev" if index == 1 else "fit",
                "oof_fold": index,
            }
            for index in range(5)
        ]
    )
    subject.validate_assignments(queries, assignments)

    broken = pd.concat(
        [
            assignments,
            pd.DataFrame(
                [
                    {
                        "query_id": "q5",
                        "siren_component_id": "component-0",
                        "split": "fit",
                        "oof_fold": 2,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    expanded_queries = pd.concat(
        [queries, pd.DataFrame([_query("q5")])],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="component spans multiple"):
        subject.validate_assignments(expanded_queries, broken)


def test_retrieval_interface_passes_no_identifier_or_truth_and_ties_by_siret():
    larger = "22222222200022"
    smaller = "11111111100011"
    builder = _SparseBuilder(
        [_candidate(larger), _candidate(smaller)]
    )

    rows, diagnostic = subject.retrieve_input_blind_query(
        query=_query(),
        partitioned_store=object(),
        current_state_store=_StateStore(),
        config=subject.input_blind_retrieval_config(),
        tfidf_cache={},
        persistent_cache=object(),
        sparse_pool_builder=builder,
    )

    args, kwargs = builder.calls[0]
    crm_row = args[1]
    assert args[5] is None
    assert kwargs == {"persistent_cache": ANY}
    assert not {"input_siret", "input_siren"} & set(crm_row)
    assert [row["candidate_siret"] for row in rows] == [smaller, larger]
    assert [row["retrieval_rank_recip"] for row in rows] == [1.0, 0.5]
    assert list(rows[0]) == subject.CANDIDATE_COLUMNS
    assert rows[0]["activity_code"] == "85.20Z"
    assert diagnostic["candidate_count"] == 2


def test_retrieval_rejects_forbidden_boundary_fields_and_injection():
    leaked = {**_query(), "input_siret": "11111111100011"}
    with pytest.raises(ValueError, match="forbidden fields"):
        subject.retrieve_input_blind_query(
            query=leaked,
            partitioned_store=object(),
            current_state_store=_StateStore(),
            config=subject.input_blind_retrieval_config(),
            tfidf_cache={},
            persistent_cache=object(),
            sparse_pool_builder=_SparseBuilder([]),
        )

    with pytest.raises(ValueError, match="Positive injection"):
        subject.retrieve_input_blind_query(
            query=_query(),
            partitioned_store=object(),
            current_state_store=_StateStore(),
            config=subject.input_blind_retrieval_config(),
            tfidf_cache={},
            persistent_cache=object(),
            sparse_pool_builder=_SparseBuilder([], injected=True),
        )


def test_authoritative_state_filter_never_emits_closed_candidate():
    active = "11111111100011"
    closed = "22222222200022"
    rows, diagnostic = subject.retrieve_input_blind_query(
        query=_query(),
        partitioned_store=object(),
        current_state_store=_StateStore({active: "A", closed: "F"}),
        config=subject.input_blind_retrieval_config(),
        tfidf_cache={},
        persistent_cache=object(),
        sparse_pool_builder=_SparseBuilder(
            [_candidate(active), _candidate(closed)]
        ),
    )
    assert [row["candidate_siret"] for row in rows] == [active]
    assert diagnostic["authoritative_non_active_removed"] == 1


def test_truth_is_joined_only_after_retrieval_file_is_closed(tmp_path):
    _, labels, _ = _population()
    unlabelled = tmp_path / "unlabelled.parquet"
    labelled = tmp_path / "labelled.parquet"
    writer = subject.CandidateWriter(unlabelled)
    writer.write(
        [
            _candidate_row(),
            _candidate_row(query_id="q2", siret="22222222200022"),
        ]
    )
    writer.close()

    count = subject.label_closed_candidate_file(
        unlabelled_path=unlabelled,
        output_path=labelled,
        labels=labels,
    )
    observed = pd.read_parquet(labelled)
    assert count == 2
    assert observed["is_ground_truth"].tolist() == [1, 0]


def test_integrity_counts_retrieval_miss_end_to_end():
    queries, labels, assignments = _population()
    candidates = pd.DataFrame(
        [_candidate_row(siret="99999999900099")],
        columns=subject.CANDIDATE_COLUMNS,
    )
    report = subject.compute_integrity(
        queries=queries,
        labels=labels,
        assignments=assignments,
        candidates=candidates,
    )
    assert report["retrieval"]["fit"]["recall_siret_at_100"] == {
        "successes": 0,
        "total": 1,
        "rate": 0.0,
    }
    assert report["retrieval"]["fit"]["miss_query_ids_at_100"] == ["q1"]
    assert report["verdict"] == "PIVOT_INPUT_BLIND_RETRIEVAL"


def test_integrity_rejects_non_contiguous_rank_and_wrong_rank_recip():
    queries, labels, assignments = _population()
    broken_rank = pd.DataFrame(
        [_candidate_row(rank=2)],
        columns=subject.CANDIDATE_COLUMNS,
    )
    with pytest.raises(ValueError, match="not contiguous"):
        subject.compute_integrity(
            queries=queries,
            labels=labels,
            assignments=assignments,
            candidates=broken_rank,
        )

    wrong_recip = pd.DataFrame(
        [_candidate_row()],
        columns=subject.CANDIDATE_COLUMNS,
    )
    wrong_recip.loc[0, "retrieval_rank_recip"] = 0.25
    with pytest.raises(ValueError, match="inconsistent"):
        subject.compute_integrity(
            queries=queries,
            labels=labels,
            assignments=assignments,
            candidates=wrong_recip,
        )
