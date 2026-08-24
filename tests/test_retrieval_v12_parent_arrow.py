from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.xgb_matcher.retrieval_fold_fused_v12 import (
    compile_label_blind_scene_v12,
)
from src.xgb_matcher.retrieval_sidecars_v11 import LocalSite, SiteQueryResult
from src.xgb_matcher.retrieval_top100_cascade_v12 import (
    RetrievalTop100CascadeV12Config,
)
from src.xgb_matcher.retrieval_v12_parent_arrow import (
    compile_parent_arrow_scene_v12,
    parent_signal_complete_projection_v12,
)
from src.xgb_matcher.retrieval_v12_parent_signal_capability import (
    build_parent_signal_capability_v12,
    load_parent_signal_capability_v12,
)
from src.xgb_matcher.retrieval_v12_scene_ir import sha256_file


class _Sidecars:
    def all_sites_by_insee_if_fits(
        self, insee: str, *, max_sites: int
    ) -> SiteQueryResult:
        assert insee == "75056"
        site = LocalSite(
            siret="99999999900001",
            siren="999999999",
            insee=insee,
            postcode="75001",
            street_number="10",
            street_number_suffix="",
            street_normalized="RUE PAIX",
            current_address_normalized="10 RUE PAIX",
            administrative_state="A",
            is_headquarters=False,
        )
        return SiteQueryResult((site,), "INSEE_ALL", 1, max_sites, False)


def _signal(
    channel: str,
    rank: int,
    *,
    source: str = "SIRENE_CURRENT:fixture",
    parent: str = "",
) -> dict[str, object]:
    return {
        "source": source,
        "channel": channel,
        "rank": rank,
        "score": 1.0 / rank,
        "parent_identifier": parent,
        "parent_rank": rank,
        "parent_support_count": 1,
        "site_rank": 1,
        "relation_direction": "FORWARD" if parent else "",
    }


def _rows() -> list[dict[str, object]]:
    common = {
        "query_id": "q2",
        "fold": 2,
        "crm_name": "ALPHA",
        "crm_address": "10 RUE PAIX",
        "crm_insee": "75056",
        "crm_postcode": "75001",
    }
    return [
        {
            **common,
            "union_rank": 1,
            "siret": "11111111100001",
            "siren": "111111111",
            "channels": ["legal_name_exact"],
            "sources": ["SIRENE_CURRENT:fixture"],
            "signals": [_signal("legal_name_exact", 1)],
            "candidate_insee": "75056",
            "candidate_postcode": "75001",
            "legal_name_exact_rank": 1,
            "legal_name_exact_score": 1.0,
        },
        # Duplicate SIRET exercises first-geography and min-rank merge.
        {
            **common,
            "union_rank": 4,
            "siret": "11111111100001",
            "siren": "111111111",
            "channels": ["address_exact"],
            "sources": ["RNE:fixture"],
            "signals": [_signal("address_exact", 4, source="RNE:fixture")],
            "candidate_insee": "75056",
            "candidate_postcode": "75001",
            "address_exact_rank": 4,
            "address_exact_score": 0.25,
        },
        {
            **common,
            "union_rank": 2,
            "siret": "22222222200002",
            "siren": "222222222",
            "channels": ["bodacc_relation", "name_word"],
            "sources": ["BODACC_STRUCTURED:fixture"],
            "signals": [
                _signal(
                    "bodacc_relation",
                    2,
                    source="BODACC_STRUCTURED:fixture",
                    parent="222222222",
                ),
                _signal("name_word", 2),
            ],
            "candidate_insee": "75056",
            "candidate_postcode": "75001",
            "bodacc_relation_rank": 2,
            "bodacc_relation_score": 0.5,
            "name_word_rank": 2,
            "name_word_score": 0.5,
        },
    ]


def _config() -> RetrievalTop100CascadeV12Config:
    return RetrievalTop100CascadeV12Config(
        master_reference_cap=20,
        hydration_shortlist_cap=10,
        parent_shortlist_cap=3,
        small_insee_threshold=5,
        parent_expansion_budget=10,
        max_sites_per_parent=20,
        relation_seed_limit=4,
        relation_reference_budget=4,
        rescue_parent_budget=2,
        rescue_parent_reserve=1,
        rescue_min_name_parents=2,
        rescue_min_strong_parents=1,
    )


def test_parent_arrow_is_exact_for_duplicates_relations_and_census() -> None:
    table = pa.Table.from_pylist(_rows())
    old = compile_label_blind_scene_v12(
        table.to_pylist(), query_id="q2", sidecars=_Sidecars(), config=_config()
    )
    new = compile_parent_arrow_scene_v12(
        table, query_id="q2", sidecars=_Sidecars(), config=_config()
    )
    assert old.parents == new.parents
    for name in (
        "query_id",
        "query_fold",
        "input_reference_count",
        "parent_count_before_rescue",
        "rescue_triggered",
        "small_insee_sites",
        "small_insee_pool",
        "small_insee_strategy",
        "structured_relation_references",
    ):
        assert getattr(old, name) == getattr(new, name), name

    projection = parent_signal_complete_projection_v12(table.column_names)
    projected = compile_parent_arrow_scene_v12(
        table.select(projection),
        query_id="q2",
        sidecars=_Sidecars(),
        config=_config(),
        signals_complete=True,
    )
    assert projected.parents == old.parents


@dataclass
class _Cache:
    source: Path
    manifest: dict[str, object]


def _cache(path: Path, rows: list[dict[str, object]]) -> _Cache:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return _Cache(
        source=path,
        manifest={
            "build_id": "a" * 64,
            "source_candidate_sha256": sha256_file(path),
            "source_manifest_sha256": "b" * 64,
            "candidate_rows": len(rows),
        },
    )


def test_signal_capability_is_exhaustive_content_addressed_and_bound(
    tmp_path: Path,
) -> None:
    cache = _cache(tmp_path / "complete.parquet", _rows())
    result = build_parent_signal_capability_v12(
        cache, output_root=tmp_path / "capabilities", threads=1  # type: ignore[arg-type]
    )
    loaded = load_parent_signal_capability_v12(
        result.manifest_path, cache  # type: ignore[arg-type]
    )
    assert loaded == result
    assert result.manifest_path.parent.name == result.build_id[:16]
    assert result.candidate_rows == len(_rows())

    forged = _rows()
    forged[0] = dict(forged[0])
    forged[0]["signals"] = []
    bad = _cache(tmp_path / "incomplete.parquet", forged)
    with pytest.raises(ValueError, match="not signal-complete"):
        build_parent_signal_capability_v12(
            bad,
            output_root=tmp_path / "bad-capabilities",
            threads=1,  # type: ignore[arg-type]
        )
