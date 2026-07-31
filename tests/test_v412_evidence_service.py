from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.xgb_matcher.v412_evidence_service import (
    V412DirectEvidenceService,
)


@dataclass
class _Index:
    active_count: int


def _service(*, cache_entries: int = 2):
    loads: list[str] = []

    def route(query, store):
        return str(query["crm_insee"])

    def load(key, store):
        loads.append(key)
        return [{"key": key}]

    def build_index(rows):
        return _Index(active_count=10)

    def search(query, index, *, partition_key):
        if partition_key == "A":
            return [
                {
                    "candidate_siret": "00000000100001",
                    "candidate_siren": "000000001",
                    "candidate_state": "A",
                    "exact_name_anchor": True,
                    "exact_address_anchor": True,
                    "direct_evidence_class": "NAME_AND_ADDRESS",
                }
            ]
        return []

    return (
        V412DirectEvidenceService(
            partition_store=object(),
            max_index_cache_entries=cache_entries,
            route=route,
            load_partition=load,
            build_index=build_index,
            search=search,
        ),
        loads,
    )


def test_full_universe_evidence_is_aggregated_and_cached() -> None:
    service, loads = _service()
    first = service.build(
        {
            "query_id": "q0",
            "crm_name": "One",
            "crm_address": "1 rue exemple",
            "crm_postcode": "75001",
            "crm_city": "Paris",
            "crm_insee": "A",
        }
    )
    second = service.build(
        {
            "query_id": "q1",
            "crm_name": "One",
            "crm_address": "1 rue exemple",
            "crm_postcode": "75001",
            "crm_city": "Paris",
            "crm_insee": "A",
        }
    )

    assert first.query["active_universe_count"] == 10
    assert first.query["direct_candidate_count"] == 1
    assert first.query["sole_direct_siret"] == "00000000100001"
    assert len(first.candidates) == 1
    assert second.query["direct_candidate_count"] == 1
    assert loads == ["A"]
    assert service.cache_miss_count == 1
    assert service.cache_hit_count == 1


def test_evidence_index_cache_is_bounded_lru() -> None:
    service, loads = _service(cache_entries=1)
    for index, key in enumerate(("A", "B", "A")):
        service.build({"query_id": f"q{index}", "crm_insee": key})
    assert loads == ["A", "B", "A"]
    assert service.cache_miss_count == 3
    assert service.cache_eviction_count == 2


def test_forbidden_truth_is_rejected_before_routing() -> None:
    service, loads = _service()
    with pytest.raises(ValueError, match="forbidden"):
        service.build(
            {
                "query_id": "q0",
                "crm_insee": "A",
                "ground_truth_siret": "00000000100001",
            }
        )
    assert loads == []


class _StrictOnlyStore:
    def __init__(self, rows):
        self.partition_keys = ("75056_", "_75001")
        self.rows = rows
        self.loads: list[str] = []

    def load(self, key):
        self.loads.append(key)
        return [dict(row) for row in self.rows.get(key, [])]


def test_frozen_defaults_use_only_strict_store_api() -> None:
    store = _StrictOnlyStore({"75056_": [], "_75001": []})
    service = V412DirectEvidenceService(partition_store=store)

    output = service.build(
        {
            "query_id": "strict",
            "crm_name": "One",
            "crm_address": "1 rue exemple",
            "crm_postcode": "75001",
            "crm_city": "Paris",
            "crm_insee": "75056",
        }
    )

    assert output.query["partition_key"] == "insee:75056"
    assert output.query["active_universe_count"] == 0
    assert store.loads == ["75056_"]


def test_frozen_mega_route_filters_verified_insee_partition_by_postcode() -> None:
    from src.xgb_matcher.v412_evidence_service import (
        _load_strict_partition,
        _strict_partition_key,
    )

    store = _StrictOnlyStore(
        {
            "75056_": [],
            "_75001": [],
            "75115_": [
                {"siret": "00000000100001", "postcode": "75015"},
                {"siret": "00000000200002", "postcode": "75016"},
            ],
        }
    )
    store.partition_keys = (*store.partition_keys, "75115_")
    query = {"insee": "75115", "postcode": "75015"}

    key = _strict_partition_key(query, store)
    rows = _load_strict_partition(key, store)

    assert key == "insee_cp:75115:75015"
    assert [row["siret"] for row in rows] == ["00000000100001"]
    assert store.loads == ["75115_"]


def test_frozen_mega_route_does_not_depend_on_strict_subset_row_count() -> None:
    from src.xgb_matcher.v412_evidence_service import _strict_partition_key

    mega = _StrictOnlyStore({"76540_": []})
    mega.partition_keys = ("76540_",)
    normal = _StrictOnlyStore({"93048_": [object()] * 100_001})
    normal.partition_keys = ("93048_",)

    assert (
        _strict_partition_key(
            {"insee": "76540", "postcode": "76600"},
            mega,
        )
        == "insee_cp:76540:76600"
    )
    assert (
        _strict_partition_key(
            {"insee": "93048", "postcode": "93100"},
            normal,
        )
        == "insee:93048"
    )
    assert mega.loads == []
    assert normal.loads == []


def test_partial_evidence_function_injection_is_rejected() -> None:
    with pytest.raises(ValueError, match="fully injected"):
        V412DirectEvidenceService(
            partition_store=object(),
            route=lambda query, store: "none",
        )
