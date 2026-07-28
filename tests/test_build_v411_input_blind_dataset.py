from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY

import duckdb
import pandas as pd
import pytest
import pyarrow.parquet as pq

from scripts import build_v411_input_blind_dataset as subject
from src.xgb_matcher.features import (
    make_feature_rows_from_preprocessed,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache


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
    def __init__(
        self,
        candidates,
        *,
        injected=False,
        idf_map=None,
        default_idf=1.0,
    ):
        self.candidates = candidates
        self.injected = injected
        self.idf_map = idf_map or {}
        self.default_idf = default_idf
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _SparseResult(
            candidates=self.candidates,
            gt_was_injected=self.injected,
            idf_map=self.idf_map,
            default_idf=self.default_idf,
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


def test_source_query_reader_projects_safe_columns_physically():
    calls = []

    def reader(path, *, columns):
        calls.append((path, list(columns)))
        return pd.DataFrame(columns=columns)

    path = Path("/tmp/queries.parquet")
    observed = subject.read_projected_source_queries(path, reader=reader)
    assert calls == [(path, subject.SOURCE_QUERY_READ_COLUMNS)]
    assert list(observed.columns) == subject.SOURCE_QUERY_READ_COLUMNS
    assert not {"input_siret", "input_siren"} & set(calls[0][1])


def test_verified_cache_namespace_binds_config_and_partitions(tmp_path):
    namespace = subject.tfidf_cache_namespace(
        sparse_config_hash="config-a",
        tfidf_artifact_hash="artifact-a",
        partitions_signature="partitions-a",
    )
    assert namespace != subject.tfidf_cache_namespace(
        sparse_config_hash="config-b",
        tfidf_artifact_hash="artifact-a",
        partitions_signature="partitions-a",
    )
    assert namespace != subject.tfidf_cache_namespace(
        sparse_config_hash="config-a",
        tfidf_artifact_hash="artifact-a",
        partitions_signature="partitions-b",
    )
    directory = subject.verified_tfidf_cache_dir(
        cache_root=tmp_path,
        namespace=namespace,
    )
    assert directory.is_relative_to(tmp_path.resolve())
    with pytest.raises(ValueError, match="escaped"):
        subject.verified_tfidf_cache_dir(
            cache_root=tmp_path,
            namespace="../../outside",
        )


def test_strict_tfidf_cache_verifies_hash_before_unpickling(
    tmp_path,
    monkeypatch,
):
    cache = TfidfPersistentCache(
        "verified-namespace",
        cache_dir=tmp_path,
        require_verified=True,
    )
    artifacts = ("name", "matrix", [], None, None, None, None)
    cache.put("75056_", artifacts)
    path = cache._key_path("75056_")
    sidecar = cache._sidecar_path(path)
    assert path.exists() and sidecar.exists()
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["config_hash"] == "verified-namespace"
    assert record["partition_key"] == "75056_"
    assert cache.get("75056_") == artifacts

    path.write_bytes(path.read_bytes() + b"tampered")
    called = []

    def forbidden_load(_handle):
        called.append(True)
        raise AssertionError("unverified pickle must never be deserialized")

    monkeypatch.setattr("src.xgb_matcher.tfidf_cache.pickle.load", forbidden_load)
    assert cache.get("75056_") is None
    assert called == []
    assert cache.stats() == {
        "hits": 1,
        "misses": 1,
        "verification_rejections": 1,
    }


def test_strict_tfidf_cache_rejects_legacy_pickle_without_sidecar(
    tmp_path,
    monkeypatch,
):
    cache = TfidfPersistentCache(
        "verified-namespace",
        cache_dir=tmp_path,
        require_verified=True,
    )
    path = cache._key_path("75056_")
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        pickle.dump(("legacy",), handle)
    called = []
    monkeypatch.setattr(
        "src.xgb_matcher.tfidf_cache.pickle.load",
        lambda _handle: called.append(True),
    )
    assert cache.get("75056_") is None
    assert called == []
    assert cache.stats()["verification_rejections"] == 1


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


def test_tfidf_cache_is_deterministically_bounded_to_v41_ceiling():
    cache = {
        ("main", f"partition-{index}"): (index,)
        for index in range(27)
    }
    subject._trim_tfidf_cache(cache)
    assert len(cache) == subject.MAX_IN_MEMORY_TFIDF_PARTITIONS == 20
    assert list(cache) == [
        ("main", f"partition-{index}")
        for index in range(7, 27)
    ]


class _RecordingConnection:
    def __init__(self, connection):
        self.connection = connection
        self.queries = []

    def execute(self, query, parameters=None):
        self.queries.append(str(query))
        if parameters is None:
            return self.connection.execute(query)
        return self.connection.execute(query, parameters)


def _pre_optimization_reference(
    *,
    query,
    candidates,
    snapshot_details,
    idf_map,
    default_idf,
):
    """Independent transcription of the original per-query algorithm."""

    crm_row = {
        "query_id": query["query_id"],
        "crm_id": query["query_id"],
        "crm_name": query["crm_name"],
        "crm_address": query["crm_address"],
        "crm_city": query["crm_city"],
        "postcode": query["crm_postcode"],
        "insee": query["crm_insee"],
    }
    crm_pre = preprocess_crm_row(crm_row)
    by_siret = {}
    for original in candidates:
        siret = str(original.get("siret") or "")
        if siret and siret not in by_siret:
            by_siret[siret] = dict(original)
    active = []
    for siret, candidate in by_siret.items():
        details = snapshot_details.get(siret, {})
        if str(details.get("candidate_state") or "").upper() != "A":
            continue
        candidate["etat_admin"] = "A"
        candidate["enseigne1"] = details.get("enseigne1")
        candidate["enseigne2"] = details.get("enseigne2")
        candidate["enseigne3"] = details.get("enseigne3")
        candidate["denomination_usuelle"] = details.get(
            "denomination_usuelle"
        )
        candidate["activity_code"] = details.get("activity_code")
        active.append(candidate)
    active.sort(
        key=lambda candidate: (
            -float(candidate.get("rrf_score") or 0.0),
            str(candidate.get("siret") or ""),
        )
    )
    active = active[:100]
    set_global_name_idf_map(idf_map, default_idf)
    features = make_feature_rows_from_preprocessed(
        crm_pre,
        active,
        include_semantic=False,
    )
    rows = []
    for rank, (candidate, feature_row) in enumerate(
        zip(active, features, strict=True),
        start=1,
    ):
        siret = str(candidate["siret"])
        channel_count = int(candidate.get("retrieval_channel_count") or 0)
        row = {
            "query_id": str(query["query_id"]),
            "candidate_siret": siret,
            "candidate_siren": str(candidate.get("siren") or siret[:9]),
            "candidate_state": "A",
            "is_ground_truth": 0,
            "retrieval_rank": rank,
            "retrieval_source": str(
                candidate.get("retrieval_source") or "v4.11-sparse"
            ),
            "retrieval_channel_count": channel_count,
            "retrieval_agreement": int(channel_count >= 2),
            "enseigne1": candidate.get("enseigne1"),
            "enseigne2": candidate.get("enseigne2"),
            "enseigne3": candidate.get("enseigne3"),
            "denomination_usuelle": candidate.get("denomination_usuelle"),
            "activity_code": candidate.get("activity_code"),
        }
        for name in subject.RANKER_C_FEATURE_ORDER[:-1]:
            value = float(feature_row.get(name, 0.0))
            row[name] = value if math.isfinite(value) else 0.0
        row["retrieval_rank_recip"] = 1.0 / rank
        rows.append(row)
    return rows


def test_bulk_hydration_matches_indexed_path_when_snapshot_enseigne_differs(
    tmp_path,
):
    first_siret = "11111111100011"
    second_siret = "22222222200022"
    closed_siret = "33333333300033"
    first = {
        **_candidate(first_siret, score=0.5),
        "denomination": "NOM PARTITION",
        "enseigne1": "ENSEIGNE PARTITION FAUSSE",
        "enseigne2": None,
        "enseigne3": None,
        "is_siege": False,
        "etablissementSiege": False,
        "numeroVoie": "1",
        "typeVoie": "RUE",
        "libelleVoie": "ALPHA",
        "complementAdresse": None,
        "cj_ul": "5710",
        "sigle_ul": None,
        "denomination_ul": "NOM PARTITION",
        "denomination_usuelle_ul": None,
        "nom_ul": None,
        "prenom_usuel_ul": None,
        "pm_dirigeant_names": None,
        "_xgb_addr_density_insee": 1,
        "_xgb_addr_density_cp": 1,
    }
    second = {
        **first,
        "siret": second_siret,
        "siren": second_siret[:9],
        "enseigne1": "AUTRE PARTITION",
        "numeroVoie": "99",
        "libelleVoie": "BETA",
        "rrf_score": 0.5,
        "retrieval_channel_count": 1,
    }
    closed = {
        **first,
        "siret": closed_siret,
        "siren": closed_siret[:9],
        "rrf_score": 0.9,
    }
    partition_candidates = [closed, second, first]
    # ALPHA deliberately exercises default_idf: it is present in both the CRM
    # and snapshot enseigne, but absent from the partition IDF dictionary.
    idf_map = {"ECOLE": 3.0}
    builder = _SparseBuilder(
        partition_candidates,
        idf_map=idf_map,
        default_idf=7.0,
    )
    raw_rows, _ = subject.retrieve_raw_input_blind_query(
        query=_query(),
        partitioned_store=object(),
        config=subject.input_blind_retrieval_config(),
        tfidf_cache={},
        persistent_cache=object(),
        sparse_pool_builder=builder,
    )
    raw_path = tmp_path / "raw.parquet"
    hydrated_path = tmp_path / "hydrated.parquet"
    final_path = tmp_path / "final.parquet"
    writer = subject.RawCandidateWriter(raw_path)
    writer.write(raw_rows)
    writer.close()

    snapshot_path = tmp_path / "snapshot.parquet"
    details = {
        first_siret: {
            "candidate_state": "A",
            "enseigne1": "ECOLE ALPHA",
            "enseigne2": "PRIMAIRE",
            "enseigne3": None,
            "denomination_usuelle": "ECOLE ALPHA",
            "activity_code": "85.20Z",
        },
        second_siret: {
            "candidate_state": "A",
            "enseigne1": "BETA SERVICES",
            "enseigne2": None,
            "enseigne3": None,
            "denomination_usuelle": "BETA",
            "activity_code": "62.01Z",
        },
        closed_siret: {
            "candidate_state": "F",
            "enseigne1": "ECOLE ALPHA FERMEE",
            "enseigne2": None,
            "enseigne3": None,
            "denomination_usuelle": "FERMEE",
            "activity_code": "85.20Z",
        },
    }
    pd.DataFrame(
        [
            {
                "siret": siret,
                "etatAdministratifEtablissement": detail["candidate_state"],
                "enseigne1Etablissement": detail["enseigne1"],
                "enseigne2Etablissement": detail["enseigne2"],
                "enseigne3Etablissement": detail["enseigne3"],
                "denominationUsuelleEtablissement": detail[
                    "denomination_usuelle"
                ],
                "activitePrincipaleEtablissement": detail["activity_code"],
            }
            for siret, detail in details.items()
        ]
    ).to_parquet(snapshot_path, index=False)
    recording = _RecordingConnection(duckdb.connect())
    hydration = subject.bulk_hydrate_snapshot(
        connection=recording,
        raw_candidates_path=raw_path,
        state_snapshot_path=snapshot_path,
        output_path=hydrated_path,
    )
    count, _ = subject.finalize_hydrated_pools(
        hydrated_path=hydrated_path,
        output_path=final_path,
        queries=pd.DataFrame([_query()]),
    )
    bulk_rows = pd.read_parquet(final_path)
    expected_rows = _pre_optimization_reference(
        query=_query(),
        candidates=partition_candidates,
        snapshot_details=details,
        idf_map=idf_map,
        default_idf=7.0,
    )
    expected = pd.DataFrame(
        expected_rows,
        columns=subject.CANDIDATE_COLUMNS,
    )
    expected[subject.RANKER_C_FEATURE_ORDER] = expected[
        subject.RANKER_C_FEATURE_ORDER
    ].astype("float32")
    pd.testing.assert_frame_equal(
        bulk_rows.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    assert count == 2
    assert hydration["snapshot_full_scan_count"] == 1
    snapshot_mentions = sum(
        str(snapshot_path.resolve()) in query
        for query in recording.queries
    )
    assert snapshot_mentions == 1
    assert bulk_rows["candidate_siret"].tolist() == [
        first_siret,
        second_siret,
    ]
    assert bulk_rows.loc[0, "enseigne1"] == "ECOLE ALPHA"
    assert bulk_rows.loc[0, "activity_code"] == "85.20Z"
    assert bulk_rows.loc[0, "name_jaro_max"] == 1.0
    assert bulk_rows.loc[0, "idf_name"] == 5.0
    assert bulk_rows.loc[0, "addr_jaro"] > bulk_rows.loc[1, "addr_jaro"]


def test_label_loader_requires_a_physically_closed_final_pool(tmp_path):
    final_path = tmp_path / "final.parquet"
    labels_path = tmp_path / "labels.parquet"
    writer = subject.CandidateWriter(final_path)
    writer.write([_candidate_row()])
    writer.close()
    pd.DataFrame({"query_id": ["q1"]}).to_parquet(labels_path, index=False)
    observed = []

    def loader(path):
        # A readable footer proves CandidateWriter.close() happened first.
        observed.append(
            pq.ParquetFile(final_path).metadata.num_rows
        )
        return pd.DataFrame({"query_id": ["q1"]})

    labels = subject.load_labels_after_final_pool_closed(
        final_unlabelled_path=final_path,
        labels_path=labels_path,
        expected_sha256=subject.file_sha256(labels_path),
        loader=loader,
    )
    assert observed == [1]
    assert labels["query_id"].tolist() == ["q1"]


def test_pinned_comparison_reader_never_calls_loader_on_wrong_hash(tmp_path):
    path = tmp_path / "candidates_v42b.parquet"
    pd.DataFrame({"candidate_siret": ["11111111100011"]}).to_parquet(
        path,
        index=False,
    )
    calls = []
    with pytest.raises(ValueError, match="hash mismatch"):
        subject.read_sha256_pinned_parquet(
            path,
            expected_sha256="0" * 64,
            reader=lambda _path: calls.append(_path),
        )
    assert calls == []


def test_pinned_label_reader_never_calls_loader_on_wrong_hash(tmp_path):
    final_path = tmp_path / "final.parquet"
    labels_path = tmp_path / "labels.parquet"
    writer = subject.CandidateWriter(final_path)
    writer.write([_candidate_row()])
    writer.close()
    pd.DataFrame({"query_id": ["q1"]}).to_parquet(labels_path, index=False)
    calls = []
    with pytest.raises(ValueError, match="hash mismatch"):
        subject.load_labels_after_final_pool_closed(
            final_unlabelled_path=final_path,
            labels_path=labels_path,
            expected_sha256="f" * 64,
            loader=lambda _path: calls.append(_path),
        )
    assert calls == []
