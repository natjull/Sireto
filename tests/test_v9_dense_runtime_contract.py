from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.precompute_embeddings import prepare_candidates_for_index
from src.xgb_matcher.dense_retrieval import PartitionEmbeddingStore
from src.xgb_matcher.retrieval import build_candidate_pool
from src.xgb_matcher.retrieval_config import RetrievalConfigV1


class _FakeIndex:
    def __init__(self, count: int) -> None:
        self.n = count

    def search(self, _query: np.ndarray, k: int):
        indices = np.arange(k, dtype=np.int64)
        scores = np.linspace(1.0, 0.0, k, dtype=np.float32)
        return scores, indices


class _FakeCandidateStore:
    def __init__(self, candidates: list[dict]) -> None:
        self.candidates = candidates

    def load_by_insee_then_postcode_with_key(self, *_args, **_kwargs):
        return self.candidates, "01001_"


class _FakeDenseStore:
    def __init__(self, count: int) -> None:
        self.index = _FakeIndex(count)

    def has_embeddings(self, _key: str) -> bool:
        return True

    def get_index(self, _key: str):
        return self.index

    def validates_candidate_order(self, _key, candidates, index) -> bool:
        return len(candidates) == index.n


def test_partition_dense_manifest_validates_exact_siret_order(
    tmp_path: Path,
) -> None:
    candidates = [
        {"siret": "01001000000001"},
        {"siret": "01001000000002"},
    ]
    payload = "\n".join(candidate["siret"] for candidate in candidates)
    (tmp_path / "01001__manifest.json").write_text(
        json.dumps(
            {
                "candidate_count": 2,
                "siret_order_sha256": hashlib.sha256(
                    payload.encode("ascii")
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    store = PartitionEmbeddingStore(tmp_path)
    index = _FakeIndex(2)

    assert store.validates_candidate_order("01001_", candidates, index)
    assert not store.validates_candidate_order(
        "01001_",
        list(reversed(candidates)),
        index,
    )
    assert not store.validates_candidate_order(
        "01001_",
        candidates[:1],
        index,
    )


def test_precompute_uses_canonical_filter_and_dedupe_order() -> None:
    candidates = [
        {"siret": "1", "denomination": "ALPHA", "etat_admin": "A"},
        {"siret": "2", "denomination": None, "etat_admin": "A"},
        {"siret": "1", "denomination": "ALPHA BIS", "etat_admin": "A"},
        {"siret": "3", "denomination": "FERMEE", "etat_admin": "F"},
    ]

    prepared = prepare_candidates_for_index(
        candidates,
        drop_unnamed=True,
        include_closed=False,
    )

    assert [candidate["siret"] for candidate in prepared] == ["1"]
    assert prepared[0]["denomination"] == "ALPHA BIS"


def test_dense_only_rrf_has_defined_sparse_scores(monkeypatch) -> None:
    candidates = [
        {
            "siret": f"0100100000{index:04d}",
            "siren": "010010000",
            "denomination": f"ENTREPRISE {index}",
            "etat_admin": "A",
        }
        for index in range(60)
    ]
    monkeypatch.setattr(
        "src.xgb_matcher.dense_retrieval.encode_query",
        lambda _text: np.ones(4, dtype=np.float32),
    )
    config = RetrievalConfigV1(
        sparse_retrieval_enabled=False,
        dense_retrieval_enabled=True,
        dense_top_k=50,
        fusion_mode="rrf",
        retrieval_budget=50,
        prefilter_k=50,
        min_candidates=50,
    )
    result = build_candidate_pool(
        _FakeCandidateStore(candidates),
        {
            "crm_name": "ENTREPRISE",
            "insee": "01001",
            "postcode": "01000",
        },
        {
            "crm_addr": "",
            "crm_street_num": "",
            "crm_street_name": "",
        },
        config,
        {},
        dense_store=_FakeDenseStore(len(candidates)),
    )

    assert len(result.candidates) == 50
    assert all(
        candidate["retrieval_source"] == "dense"
        for candidate in result.candidates
    )


def test_separate_sparse_rrf_passes_name_and_address_as_distinct_channels(
    monkeypatch,
) -> None:
    candidates = [
        {
            "siret": f"0100100000{index:04d}",
            "siren": "010010000",
            "denomination": f"ENTREPRISE {index}",
            "etat_admin": "A",
        }
        for index in range(60)
    ]
    monkeypatch.setattr(
        "src.xgb_matcher.retrieval._get_tfidf_artifacts",
        lambda *_args, **_kwargs: (
            object(),
            object(),
            [],
            None,
            None,
            object(),
            object(),
        ),
    )
    monkeypatch.setattr(
        "src.xgb_matcher.retrieval.prefilter_candidates_tfidf_scored",
        lambda *_args, **_kwargs: [(0, 0.9), (1, 0.8)],
    )
    monkeypatch.setattr(
        "src.xgb_matcher.retrieval.prefilter_candidates_address_tfidf_scored",
        lambda *_args, **_kwargs: [(1, 90.0), (2, 80.0)],
    )
    config = RetrievalConfigV1(
        fusion_mode="rrf",
        sparse_channel_fusion_mode="separate_rrf",
        retrieval_budget=10,
        prefilter_k=50,
        prefilter_trigger_size=1,
        min_candidates=10,
    )

    result = build_candidate_pool(
        _FakeCandidateStore(candidates),
        {
            "crm_name": "ENTREPRISE",
            "insee": "01001",
            "postcode": "01000",
        },
        {
            "crm_addr": "1 RUE TEST",
            "crm_street_num": "",
            "crm_street_name": "",
        },
        config,
        {},
    )

    agreed = next(
        candidate
        for candidate in result.candidates
        if candidate["siret"] == candidates[1]["siret"]
    )
    assert agreed["retrieval_source"] == "sparse_address+sparse_name"
    assert agreed["sparse_name_rank"] == 2
    assert agreed["sparse_address_rank"] == 1
    assert agreed["sparse_rank"] == 1
