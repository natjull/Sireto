from __future__ import annotations

from dataclasses import dataclass

from src.xgb_matcher.hierarchical_retrieval import (
    InMemoryBackend,
    IndexedEstablishment,
    RetrievalQuery,
)
from src.xgb_matcher.official_evidence_retrieval import (
    OfficialEvidenceCandidate,
    OfficialEvidenceRetrievalConfig,
    OfficialEvidenceRetrievalResult,
    OfficialEvidenceRetriever,
    OfficialEvidenceSignal,
)
from src.xgb_matcher.retrieval_all_human_v11 import (
    AllHumanV11Retriever,
    RetrievalV11Config,
)
from src.xgb_matcher.retrieval_sidecars_v11 import LocalSite, SiteQueryResult


def _record(siret: str, *, name: str = "ALPHA", state: str = "A") -> IndexedEstablishment:
    return IndexedEstablishment.from_mapping(
        {
            "siret": siret,
            "siren": siret[:9],
            "insee": "75056",
            "postcode": "75001",
            "names": [name],
            "addresses": ["10 RUE DE LA PAIX"],
            "number": "10",
            "state": state,
        }
    )


def _site(siret: str, *, state: str = "A") -> LocalSite:
    return LocalSite(
        siret=siret,
        siren=siret[:9],
        insee="75056",
        postcode="75001",
        street_number="10",
        street_number_suffix="",
        street_normalized="RUE DE LA PAIX",
        current_address_normalized="10 RUE DE LA PAIX",
        administrative_state=state,
        is_headquarters=False,
    )


class _Canonical:
    def __init__(self, records: list[IndexedEstablishment]) -> None:
        self.records = {record.siret: record for record in records}

    def by_siret(self, siret: str) -> IndexedEstablishment | None:
        return self.records.get(siret)


class _Official:
    def __init__(self, records: list[IndexedEstablishment]) -> None:
        self.backends = [("fixture", _Canonical(records))]
        self.records = records
        self.requested_cap = 0

    def retrieve(
        self, query: RetrievalQuery, *, candidate_cap: int | None = None
    ) -> OfficialEvidenceRetrievalResult:
        self.requested_cap = int(candidate_cap or 0)
        candidates = tuple(
            OfficialEvidenceCandidate(
                union_rank=rank,
                siret=record.siret,
                siren=record.siren,
                record=record,
                signals=(
                    OfficialEvidenceSignal(
                        source="fixture", channel="fielded_name_bm25", rank=rank, score=1 / rank
                    ),
                ),
            )
            for rank, record in enumerate(self.records[: self.requested_cap], start=1)
        )
        return OfficialEvidenceRetrievalResult(
            query=query,
            candidates=candidates,
            raw_union_size=len(self.records),
            union_truncated=len(self.records) > len(candidates),
            retrieval_latency_ms=0.1,
        )


@dataclass
class _Sidecars:
    sites: tuple[LocalSite, ...]

    def geo_candidates(self, **kwargs: object) -> SiteQueryResult:
        cap = int(kwargs["max_sites"])
        return SiteQueryResult(self.sites[:cap], "INSEE_ALL", len(self.sites), cap, len(self.sites) > cap)

    def sites_by_siren_geo(self, **kwargs: object) -> SiteQueryResult:
        siren = str(kwargs["siren"])
        cap = int(kwargs["max_sites"])
        matches = tuple(site for site in self.sites if site.siren == siren)
        return SiteQueryResult(matches[:cap], "SIREN_GEO", len(matches), cap, len(matches) > cap)


@dataclass
class _ParentOnlySidecars(_Sidecars):
    def geo_candidates(self, **kwargs: object) -> SiteQueryResult:
        cap = int(kwargs["max_sites"])
        return SiteQueryResult(
            (), "INSUFFICIENT_LARGE_COMMUNE_GEO", 50_000, cap, True
        )


def test_retrieval_is_label_blind_and_keeps_active_and_closed() -> None:
    official_record = _record("11111111100001")
    closed_site = _site("22222222200002", state="F")
    official = _Official([official_record])
    retriever = AllHumanV11Retriever(
        official,
        _Sidecars((_site(official_record.siret), closed_site)),
        RetrievalV11Config(raw_reference_cap=10, internal_cap=10, parent_candidate_budget=10),
    )
    base = {
        "crm_name": "ALPHA",
        "crm_address": "10 RUE DE LA PAIX",
        "crm_insee": "75056",
        "crm_cp": "75001",
    }
    result = retriever.retrieve(base)
    with_label = retriever.retrieve({**base, "ground_truth_siret": closed_site.siret})

    assert official.requested_cap == 10
    assert [item.siret for item in result.candidates] == [item.siret for item in with_label.candidates]
    assert {item.siret for item in result.candidates} == {official_record.siret, closed_site.siret}
    assert next(item for item in result.candidates if item.siret == closed_site.siret).record.active is False
    serialized = result.candidate_rows("q1", {"fold": 2, "crm_name": "ALPHA"})
    assert len(serialized) == 2
    assert all("ground_truth_siret" not in row for row in serialized)
    assert any(row["geo_insee_census_rank"] is not None for row in serialized)
    assert {row["pool_size"] for row in serialized} == {2}
    assert {row["mega_base_pool"] for row in serialized} == {False}


def test_parent_round_robin_prevents_chain_monopoly() -> None:
    sites = tuple(
        site
        for index in range(1, 4)
        for site in (
            _site(f"111111111{index:05d}"),
            _site(f"222222222{index:05d}"),
        )
    )
    records = [_record(site.siret) for site in sites]
    retriever = AllHumanV11Retriever(
        _Official(records[:2]),
        _Sidecars(sites),
        RetrievalV11Config(
            raw_reference_cap=10,
            internal_cap=6,
            parent_candidate_budget=6,
            parent_siren_limit=2,
        ),
    )
    result = retriever.retrieve(
        {"crm_name": "ALPHA", "crm_address": "10 RUE DE LA PAIX", "crm_insee": "75056", "crm_cp": "75001"}
    )

    assert len(result.candidates) == 6
    assert {item.siren for item in result.candidates} == {"111111111", "222222222"}
    assert result.strategy_counts["selected"] == 6


def test_complete_geo_census_survives_final_union_cap() -> None:
    geo_sites = tuple(_site(f"999999999{index:05d}") for index in range(1, 5))
    official_records = [_record(f"{index:09d}00001") for index in range(1, 9)]
    retriever = AllHumanV11Retriever(
        _Official(official_records),
        _Sidecars(geo_sites),
        RetrievalV11Config(
            raw_reference_cap=20,
            internal_cap=6,
            parent_candidate_budget=2,
            parent_siren_limit=2,
        ),
    )

    result = retriever.retrieve(
        {
            "crm_name": "ALPHA",
            "crm_address": "10 RUE DE LA PAIX",
            "crm_insee": "75056",
            "crm_cp": "75001",
        }
    )

    selected = {candidate.siret for candidate in result.candidates}
    assert {site.siret for site in geo_sites}.issubset(selected)
    assert result.strategy_counts["geo_reserved"] == 4


def test_parent_expansion_budget_survives_final_union_round_robin() -> None:
    parent_sites = (
        _site("11111111100001"),
        _site("11111111100002", state="F"),
        _site("22222222200001"),
        _site("22222222200002", state="F"),
    )
    official_records = [
        _record("11111111100001"),
        _record("22222222200001"),
        *[_record(f"{index:09d}00001") for index in range(3, 9)],
    ]
    retriever = AllHumanV11Retriever(
        _Official(official_records),
        _ParentOnlySidecars(parent_sites),
        RetrievalV11Config(
            raw_reference_cap=20,
            internal_cap=6,
            parent_candidate_budget=4,
            parent_siren_limit=2,
        ),
    )

    result = retriever.retrieve(
        {
            "crm_name": "ALPHA",
            "crm_address": "10 RUE DE LA PAIX",
            "crm_insee": "75056",
            "crm_cp": "75001",
        }
    )

    selected = {candidate.siret for candidate in result.candidates}
    assert {site.siret for site in parent_sites}.issubset(selected)
    assert result.strategy_counts["parent_reserved"] == 4


def test_v11_config_rejects_invalid_caps() -> None:
    try:
        RetrievalV11Config(raw_reference_cap=2_001, internal_cap=2_001)
    except ValueError as error:
        assert "internal cap" in str(error)
    else:
        raise AssertionError("invalid v11 cap accepted")


def test_official_reference_prefix_override_does_not_change_legacy_default() -> None:
    records = [_record(f"{index:09d}00001") for index in range(1, 6)]
    retriever = OfficialEvidenceRetriever(
        InMemoryBackend(records),
        config=OfficialEvidenceRetrievalConfig(
            max_union_candidates=2,
            exact_limit=10,
            word_limit=10,
            character_limit=10,
            historical_limit=10,
            supporting_limit=10,
            siren_limit=3,
            sites_per_siren=3,
            relation_seed_limit=2,
            relation_limit=10,
            search_workers=1,
            require_temporal_complete=False,
        ),
    )
    query = RetrievalQuery(name="ALPHA", insee="75056", postcode="75001")

    assert len(retriever.retrieve(query).candidates) == 2
    assert len(retriever.retrieve(query, candidate_cap=4).candidates) == 4
