from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.run_retrieval_top100_cascade_v12 as cascade_cli
from scripts.run_retrieval_hierarchical_admission_v12 import (
    verify_parent_candidates,
)
from scripts.run_retrieval_top100_cascade_v12 import (
    _CANDIDATE_SCHEMA,
    _PARENT_SCHEMA,
    _candidate_rows,
    _iter_query_groups,
    _load_parent_model_recipe,
    _percentiles,
    sha256_file,
)
from src.xgb_matcher.retrieval_sidecars_v11 import LocalSite, SiteQueryResult
from src.xgb_matcher.retrieval_top100_cascade_v12 import (
    CASCADE_PARENT_FEATURES,
    RescueParentHitV12,
    RetrievalTop100CascadeV12,
    RetrievalTop100CascadeV12Config,
    production_projection,
)


ROOT = Path(__file__).resolve().parents[1]


def _official_v11_manifest(
    source: Path, *, run_kind: str = "LABEL_BLIND_RETRIEVAL"
) -> dict[str, Any]:
    table = pq.read_table(source)
    rows = table.to_pylist()
    query_counts: dict[str, int] = {}
    for row in rows:
        query_id = str(row.get("query_id") or "")
        query_counts[query_id] = query_counts.get(query_id, 0) + 1
    folds = sorted({int(row.get("fold", row.get("oof_fold", -1))) for row in rows})
    return {
        "schema_version": cascade_cli.V11_SOURCE_MANIFEST_SCHEMA,
        "candidate_schema_version": cascade_cli.V11_CANDIDATE_SCHEMA,
        "run_kind": run_kind,
        "candidate_parquet_sha256": sha256_file(source),
        "source_input_sha256": "a" * 64,
        "source_manifest_sha256": "e" * 64,
        "config_sha256": "b" * 64,
        "official_config_sha256": "c" * 64,
        "sidecar_manifest_sha256": "d" * 64,
        "folds": folds,
        "queries": len(query_counts),
        "candidate_rows": len(rows),
        "empty_queries": 0,
        "maximum_union_candidates": max(query_counts.values()),
        "positive_injection": False,
        "human_labels_opened_by_retrieval": False,
    }


def _write_official_combined_v11_manifest(source: Path) -> Path:
    rows = pq.read_table(source).to_pylist()
    source_shards: list[dict[str, str]] = []
    for fold in (2, 3, 4):
        shard_path = source.parent / f"natural_union_fold{fold}.parquet"
        shard_rows = [
            row
            for row in rows
            if int(row.get("fold", row.get("oof_fold", -1))) == fold
        ]
        pq.write_table(pa.Table.from_pylist(shard_rows), shard_path)
        shard_manifest = _official_v11_manifest(shard_path)
        shard_manifest_path = shard_path.with_suffix(
            shard_path.suffix + ".manifest.json"
        )
        shard_manifest_path.write_text(
            json.dumps(shard_manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        source_shards.append(
            {
                "path": str(shard_path.resolve()),
                "sha256": sha256_file(shard_path),
                "manifest_sha256": sha256_file(shard_manifest_path),
            }
        )
    combined = _official_v11_manifest(
        source, run_kind="LABEL_BLIND_RETRIEVAL_COMBINED"
    )
    combined["source_shards"] = source_shards
    manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(combined, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _write_parent_model_bundle(
    tmp_path: Path,
    *,
    source_retrieval_lineage: Mapping[str, str] | None = None,
    source_labels_sha256: str = "4" * 64,
    source_labels_manifest_sha256: str = "5" * 64,
) -> Path:
    models: dict[str, dict[str, Any]] = {}
    for suffix in ("2", "3", "4", "23", "24", "34"):
        model_path = tmp_path / f"fit_{suffix}.json"
        model_path.write_text(f"sealed-model-{suffix}\n", encoding="utf-8")
        models[f"fit_{suffix}"] = {
            "kind": "XGBOOST_BOOSTER",
            "path": model_path.name,
            "sha256": sha256_file(model_path),
            "fit_folds": [int(value) for value in suffix],
        }
    outer_maps = {
        "outer2": {
            "outer_heldout_fold": 2,
            "scorer_by_query_fold": {
                "2": {"model": "fit_34", "role": "OUTER_HELDOUT"},
                "3": {"model": "fit_4", "role": "INNER_TRAIN"},
                "4": {"model": "fit_3", "role": "INNER_TRAIN"},
            },
        },
        "outer3": {
            "outer_heldout_fold": 3,
            "scorer_by_query_fold": {
                "2": {"model": "fit_4", "role": "INNER_TRAIN"},
                "3": {"model": "fit_24", "role": "OUTER_HELDOUT"},
                "4": {"model": "fit_2", "role": "INNER_TRAIN"},
            },
        },
        "outer4": {
            "outer_heldout_fold": 4,
            "scorer_by_query_fold": {
                "2": {"model": "fit_3", "role": "INNER_TRAIN"},
                "3": {"model": "fit_2", "role": "INNER_TRAIN"},
                "4": {"model": "fit_23", "role": "OUTER_HELDOUT"},
            },
        },
    }
    bundle = {
        "schema_version": "sireto-parent-oof-model-bundle-v12-1",
        "run_kind": "PARENT_NESTED_OOF_MODEL_BUNDLE",
        "gate_eligible_for_nested_candidate_generation": True,
        "parent_feature_origin": "CASCADE_PRE_EXPANSION",
        "parent_feature_names": list(CASCADE_PARENT_FEATURES),
        "opened_folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "config_sha256": sha256_file(
            Path("config/retrieval_all_human_v12.json")
        ),
        "source_retrieval_lineage": source_retrieval_lineage
        or {
            "source_candidate_sha256": "1" * 64,
            "source_manifest_sha256": "2" * 64,
            "sidecar_manifest_sha256": "3" * 64,
        },
        "source_labels_sha256": source_labels_sha256,
        "source_labels_manifest_sha256": source_labels_manifest_sha256,
        "models": models,
        "outer_maps": outer_maps,
    }
    path = tmp_path / "parent_oof_bundle.json"
    path.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _site(
    siret: str,
    *,
    insee: str = "75056",
    postcode: str = "75001",
    number: str = "10",
    state: str = "A",
) -> LocalSite:
    return LocalSite(
        siret=siret,
        siren=siret[:9],
        insee=insee,
        postcode=postcode,
        street_number=number,
        street_number_suffix="",
        street_normalized="RUE DE LA PAIX",
        current_address_normalized=f"{number} RUE DE LA PAIX",
        administrative_state=state,
        is_headquarters=False,
    )


def _signal(
    channel: str,
    *,
    rank: int = 1,
    source: str = "SIRENE_CURRENT:fixture",
    parent_identifier: str = "",
) -> dict[str, Any]:
    return {
        "source": source,
        "channel": channel,
        "rank": rank,
        "score": 1.0 / rank,
        "parent_identifier": parent_identifier,
        "parent_rank": None,
        "site_rank": None,
        "relation_direction": "SUCCESSOR" if parent_identifier else "",
    }


def _row(
    siret: str,
    channels: tuple[str, ...] = ("fielded_name_bm25",),
    *,
    rank: int = 1,
    insee: str = "75056",
    postcode: str = "75001",
    signals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    values = signals or [_signal(channel, rank=rank) for channel in channels]
    return {
        "query_id": "q1",
        "crm_name": "ALPHA",
        "crm_address": "10 RUE DE LA PAIX",
        "crm_insee": "75056",
        "crm_postcode": "75001",
        "union_rank": rank,
        "siret": siret,
        "siren": siret[:9],
        "channels": list(channels),
        "sources": ["SIRENE_CURRENT:fixture"],
        "signals": values,
        "candidate_names": ["ALPHA"],
        "candidate_addresses": ["10 RUE DE LA PAIX"],
        "candidate_insee": insee,
        "candidate_postcode": postcode,
        "candidate_number": "10",
        "candidate_state": "A",
    }


@dataclass
class _Sidecars:
    sites: tuple[LocalSite, ...]
    commune_count: int = 50_000

    def all_sites_by_insee_if_fits(
        self, insee: str, *, max_sites: int
    ) -> SiteQueryResult:
        matches = tuple(site for site in self.sites if site.insee == insee)
        count = self.commune_count if self.commune_count > len(matches) else len(matches)
        if count > max_sites:
            return SiteQueryResult((), "INSEE_TOO_LARGE", count, max_sites, True)
        return SiteQueryResult(matches, "INSEE_ALL", count, max_sites, False)

    def sites_by_sirens_geo(self, **kwargs: Any) -> dict[str, SiteQueryResult]:
        cap = int(kwargs["max_sites_per_siren"])
        insee = str(kwargs.get("insee") or "")
        postcode = str(kwargs.get("postcode") or "")
        output: dict[str, SiteQueryResult] = {}
        for siren in kwargs["sirens"]:
            matches = tuple(
                site
                for site in self.sites
                if site.siren == siren
                and (site.insee == insee if insee else site.postcode == postcode)
            )
            output[siren] = SiteQueryResult(
                matches[:cap], "SIREN_GEO_BATCH", len(matches), cap, len(matches) > cap
            )
        return output


class _Rescue:
    def __init__(self, hits: list[RescueParentHitV12]) -> None:
        self.hits = hits
        self.calls = 0

    def retrieve(self, query: object, *, limit: int, query_id: str = ""):
        self.calls += 1
        return self.hits[:limit]


def _config(**changes: Any) -> RetrievalTop100CascadeV12Config:
    values = {
        "master_reference_cap": 30,
        "hydration_shortlist_cap": 20,
        "parent_shortlist_cap": 8,
        "small_insee_threshold": 10,
        "parent_expansion_budget": 20,
        "max_sites_per_parent": 20,
        "relation_seed_limit": 4,
        "relation_reference_budget": 4,
        "rescue_parent_budget": 4,
        "rescue_parent_reserve": 2,
        "rescue_min_name_parents": 2,
        "rescue_min_strong_parents": 1,
    }
    values.update(changes)
    if "parent_expansion_budget" not in changes:
        values["parent_expansion_budget"] = min(
            values["parent_expansion_budget"], values["master_reference_cap"]
        )
    return RetrievalTop100CascadeV12Config(**values)


def test_truth_and_alias_columns_do_not_change_generation() -> None:
    rows = [_row("11111111100001")]
    cascade = RetrievalTop100CascadeV12(_Sidecars((_site("11111111100001"),)), _config())
    baseline = cascade.cascade(rows, query_id="q1")
    forbidden = {
        "ground_truth_siret": "9" * 14,
        "historical_ground_truth_siret": "9" * 14,
        "human_ground_truth_siret": "9" * 14,
        "human_ground_truth_siren": "9" * 9,
        "human_siret": "9" * 14,
        "gt_siret": "9" * 14,
        "target_siret": "9" * 14,
        "operational_sirets_json": '["99999999999999"]',
        "acceptable_sirets_operational_json": '["99999999999999"]',
        "crm_alias": "PRIVATE MEMORY",
    }
    polluted = cascade.cascade(
        [{**rows[0], **forbidden}],
        query_id="q1",
    )

    assert [item.siret for item in baseline.master_references] == [
        item.siret for item in polluted.master_references
    ]
    assert [item.siret for item in baseline.hydration_shortlist] == [
        item.siret for item in polluted.hydration_shortlist
    ]
    projected = production_projection(["query_id", *forbidden, "siret"])
    assert projected == ("query_id", "siret")
    assert all(
        "ground_truth" not in key
        for row in polluted.admission_rows("q1")
        for key in row
    )
    master_row = baseline.master_rows("q1")[0]
    admission_row = baseline.admission_rows("q1")[0]
    assert master_row["union_rank"] == master_row["master_rank"]
    assert admission_row["union_rank"] == admission_row["hydration_rank"]
    assert admission_row["fielded_name_bm25_rank"] == 1
    assert admission_row["fielded_name_bm25_score"] == 1.0


def test_small_insee_complete_census_is_protected() -> None:
    sites = tuple(_site(f"{index:09d}00001") for index in range(1, 7))
    cascade = RetrievalTop100CascadeV12(
        _Sidecars(sites, commune_count=6),
        _config(master_reference_cap=8, hydration_shortlist_cap=8),
    )
    result = cascade.cascade([_row("99999999900001")], query_id="q1")

    assert {site.siret for site in sites}.issubset(
        {candidate.siret for candidate in result.master_references}
    )
    assert {site.siret for site in sites}.issubset(
        {candidate.siret for candidate in result.hydration_shortlist}
    )
    assert result.strategy_counts["small_insee_sites"] == 6


def test_injected_parent_scorer_ranks_before_targeted_expansion() -> None:
    strong = "11111111100001"
    weak = "22222222200001"
    expanded = "11111111100002"
    observed: list[Mapping[str, float]] = []

    def scorer(features: Mapping[str, float]) -> float:
        observed.append(dict(features))
        assert not any(
            token in name.lower()
            for name in features
            for token in ("siren", "siret", "query_id")
        )
        return 100.0 * features["exact_name_evidence"] + features["channel_rrf"]

    cascade = RetrievalTop100CascadeV12(
        _Sidecars((_site(strong), _site(expanded), _site(weak))),
        _config(parent_shortlist_cap=1, rescue_parent_budget=1, rescue_parent_reserve=1),
        parent_scorer=scorer,
    )
    result = cascade.cascade(
        [
            _row(strong, ("legal_name_exact", "address_word"), rank=10),
            _row(weak, ("fielded_name_bm25", "address_word"), rank=1),
        ],
        query_id="q1",
    )

    assert observed
    assert {parent.siren for parent in result.parent_candidates} == {
        "111111111",
        "222222222",
    }
    assert result.parent_shortlist[0].siren == "111111111"
    candidate = next(item for item in result.master_references if item.siret == expanded)
    assert "targeted_parent_expansion" in candidate.lanes
    assert candidate.parent_rank == 1
    unexpanded = next(item for item in result.master_references if item.siret == weak)
    assert unexpanded.parent_rank == 2
    assert unexpanded.parent_score > 0.0
    assert unexpanded.parent_feature_names == CASCADE_PARENT_FEATURES
    assert len(unexpanded.parent_feature_values) == len(CASCADE_PARENT_FEATURES)
    assert tuple(result.parent_rows("q1")[0]["feature_names"]) == CASCADE_PARENT_FEATURES
    assert len(result.parent_rows("q1")[0]["feature_values"]) == len(
        CASCADE_PARENT_FEATURES
    )


def test_parent_expansion_has_soft_round_robin_and_no_three_site_cap() -> None:
    sites = tuple(_site(f"111111111{index:05d}") for index in range(1, 8))
    cascade = RetrievalTop100CascadeV12(
        _Sidecars(sites),
        _config(
            hydration_shortlist_cap=10,
            parent_shortlist_cap=1,
            parent_expansion_budget=10,
            max_sites_per_parent=10,
            rescue_parent_budget=1,
            rescue_parent_reserve=1,
        ),
    )
    result = cascade.cascade(
        [_row(sites[0].siret, ("legal_name_exact", "address_exact"))],
        query_id="q1",
    )

    same_parent = [
        item for item in result.hydration_shortlist if item.siren == "111111111"
    ]
    assert len(same_parent) == 7
    assert result.strategy_counts["hard_sites_per_siren_cap"] is None


def test_conditional_rescue_runs_once_only_when_sparse_is_insufficient() -> None:
    rescued_site = _site("33333333300001")
    rescue = _Rescue([RescueParentHitV12("333333333", 1, 0.9)])
    sparse = RetrievalTop100CascadeV12(
        _Sidecars((rescued_site,)), _config(), rescue_provider=rescue
    )
    result = sparse.cascade(
        [_row("11111111100001", ("address_word",))], query_id="q1"
    )

    assert rescue.calls == 1
    assert result.rescue_triggered is True
    assert result.rescue_executed is True
    rescued = next(
        item for item in result.hydration_shortlist if item.siret == rescued_site.siret
    )
    assert "conditional_siren_rescue" in rescued.lanes

    dense_rescue = _Rescue([RescueParentHitV12("333333333", 1, 0.9)])
    dense = RetrievalTop100CascadeV12(
        _Sidecars(()), _config(), rescue_provider=dense_rescue
    )
    dense_result = dense.cascade(
        [
            _row(
                "11111111100001",
                ("legal_name_exact", "address_exact"),
                rank=1,
            ),
            _row(
                "22222222200001",
                ("legal_name_exact", "address_exact"),
                rank=2,
            ),
        ],
        query_id="q1",
    )
    assert dense_result.rescue_triggered is False
    assert dense_rescue.calls == 0


@pytest.mark.parametrize(
    "channel", ("contrastive_siren_rescue", "v12_conditional_siren_rescue")
)
def test_initial_official_stack_rejects_deferred_upstream_rescue(
    channel: str,
) -> None:
    cascade = RetrievalTop100CascadeV12(_Sidecars(()), _config())
    with pytest.raises(ValueError, match="refuses deferred upstream rescue"):
        cascade.cascade([_row("11111111100001", (channel,))], query_id="q1")


def test_only_structured_one_hop_geocompatible_relations_survive() -> None:
    structured = _signal(
        "official_successor",
        source="SIRENE_SUCCESSION:structured_siret_relation",
        parent_identifier="11111111100001",
    )
    structured_siren = _signal(
        "bodacc_relation",
        source="bodacc_graph:siren_relation->sirene_current:site",
        parent_identifier="111111111",
    )
    unstructured = _signal(
        "bodacc_relation",
        source="BODACC:free_text_relation",
        parent_identifier="11111111100001",
    )
    rows = [
        _row("22222222200001", ("official_successor",), signals=[structured]),
        _row(
            "33333333300001",
            ("official_successor",),
            insee="13055",
            postcode="13001",
            signals=[structured],
        ),
        _row("44444444400001", ("bodacc_relation",), signals=[unstructured]),
        _row("55555555500001", ("bodacc_relation",), signals=[structured_siren]),
    ]
    result = RetrievalTop100CascadeV12(_Sidecars(()), _config()).cascade(
        rows, query_id="q1"
    )
    ids = {candidate.siret for candidate in result.master_references}

    assert "22222222200001" in ids
    assert "55555555500001" in ids
    assert "33333333300001" not in ids
    assert "44444444400001" not in ids
    assert result.strategy_counts["relation_hops"] == 1


def test_caps_tie_break_and_config_loader_are_deterministic() -> None:
    rows = [
        _row(f"{index:09d}00001", ("fielded_name_bm25",), rank=1)
        for index in range(1, 10)
    ]
    config = _config(
        master_reference_cap=6,
        hydration_shortlist_cap=4,
        parent_shortlist_cap=3,
        rescue_parent_budget=2,
        rescue_parent_reserve=1,
    )
    cascade = RetrievalTop100CascadeV12(_Sidecars(()), config)
    first = cascade.cascade(rows, query_id="q1")
    second = cascade.cascade(list(reversed(rows)), query_id="q1")

    assert len(first.master_references) == 6
    assert len(first.hydration_shortlist) == 4
    assert len(first.parent_candidates) == 6
    assert len(first.parent_shortlist) == 3
    assert [item.siret for item in first.master_references] == [
        item.siret for item in second.master_references
    ]
    loaded = RetrievalTop100CascadeV12Config.load(
        ROOT / "config/retrieval_all_human_v12.json"
    )
    assert loaded.master_reference_cap == 2_000
    assert loaded.hydration_shortlist_cap == 500
    assert loaded.parent_shortlist_cap == 64
    assert loaded.small_insee_threshold == 100
    assert loaded.parent_expansion_budget == 700
    assert loaded.max_sites_per_parent == 2_000
    assert loaded.rescue_parent_budget == 16
    assert loaded.rescue_parent_reserve == 8


def test_cli_scanner_never_loads_truth_and_rejects_locked_folds(
    tmp_path: Path,
) -> None:
    safe_path = tmp_path / "safe.parquet"
    rows = [
        {**_row("11111111100001"), "fold": 2, "human_ground_truth_siret": "9" * 14},
        {**_row("22222222200001"), "query_id": "q2", "fold": 3, "crm_alias": "SECRET"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), safe_path)
    groups = list(_iter_query_groups(safe_path, batch_size=1))

    assert [query_id for query_id, _rows in groups] == ["q1", "q2"]
    assert all(
        "human_ground_truth_siret" not in row and "crm_alias" not in row
        for _query_id, values in groups
        for row in values
    )

    locked_path = tmp_path / "locked.parquet"
    pq.write_table(
        pa.Table.from_pylist([{**_row("11111111100001"), "fold": 1}]),
        locked_path,
    )
    with pytest.raises(ValueError, match="outside 2/3/4"):
        list(_iter_query_groups(locked_path))


def test_cli_output_schemas_accept_cascade_and_admission_views() -> None:
    cascade = RetrievalTop100CascadeV12(_Sidecars(()), _config())
    result = cascade.cascade(
        [_row("11111111100001", ("legal_name_exact", "address_word"))],
        query_id="q1",
    )
    metadata = {
        "fold": 2,
        "crm_name": "ALPHA",
        "crm_address": "10 RUE DE LA PAIX",
        "crm_insee": "75056",
        "crm_postcode": "75001",
    }
    master = _candidate_rows(result.master_rows("q1", metadata))
    admission = _candidate_rows(result.admission_rows("q1", metadata))
    parents = [{**row, "fold": 2} for row in result.parent_rows("q1")]
    parent_candidates = [
        {**row, "fold": 2} for row in result.parent_candidate_rows("q1")
    ]

    assert pa.Table.from_pylist(master, schema=_CANDIDATE_SCHEMA).num_rows == 1
    assert pa.Table.from_pylist(admission, schema=_CANDIDATE_SCHEMA).num_rows == 1
    assert pa.Table.from_pylist(parents, schema=_PARENT_SCHEMA).num_rows == 1
    assert (
        pa.Table.from_pylist(parent_candidates, schema=_PARENT_SCHEMA).num_rows
        == 1
    )
    assert admission[0]["union_rank"] == 1
    assert admission[0]["legal_name_exact_rank"] == 1


def test_sealed_parent_bundle_routes_nested_models_by_query_fold(
    tmp_path: Path,
) -> None:
    bundle = _write_parent_model_bundle(tmp_path)
    loaded: list[str] = []

    def loader(path: Path) -> str:
        loaded.append(path.name)
        return path.name

    recipe = _load_parent_model_recipe(
        bundle,
        recipe_name="outer2",
        required_folds={2, 3, 4},
        scorer_loader=loader,
    )

    assert recipe.scorer_for_fold(2) == "fit_34.json"
    assert recipe.scorer_for_fold(3) == "fit_4.json"
    assert recipe.scorer_for_fold(4) == "fit_3.json"
    assert sorted(loaded) == ["fit_3.json", "fit_34.json", "fit_4.json"]
    payload = recipe.manifest_payload()
    assert payload["gate_eligible"] is True
    assert payload["outer_heldout_fold"] == 2
    assert payload["parent_feature_names"] == list(CASCADE_PARENT_FEATURES)
    assert payload["training_lineage"]["label_ledger_sha256"] == "4" * 64
    assert payload["scorer_by_query_fold"]["2"]["fit_folds"] == [3, 4]
    assert payload["scorer_by_query_fold"]["3"]["fit_folds"] == [4]
    assert payload["scorer_by_query_fold"]["4"]["fit_folds"] == [3]
    assert set(payload["model_sha256_by_name"]) == {
        "fit_2",
        "fit_3",
        "fit_4",
        "fit_23",
        "fit_24",
        "fit_34",
    }
    assert all(
        payload["scorer_by_query_fold"][str(fold)]["sha256"]
        for fold in (2, 3, 4)
    )


def test_parent_bundle_rejects_hash_schema_and_outer_fold_leakage(
    tmp_path: Path,
) -> None:
    bundle = _write_parent_model_bundle(tmp_path)
    raw = json.loads(bundle.read_text(encoding="utf-8"))

    raw["models"]["fit_34"]["sha256"] = "0" * 64
    bundle.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        _load_parent_model_recipe(
            bundle,
            recipe_name="outer2",
            required_folds={2},
            scorer_loader=lambda path: path,
        )

    bundle = _write_parent_model_bundle(tmp_path)
    raw = json.loads(bundle.read_text(encoding="utf-8"))
    raw["parent_feature_names"] = list(reversed(CASCADE_PARENT_FEATURES))
    bundle.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="schema/order drift"):
        _load_parent_model_recipe(
            bundle,
            recipe_name="outer2",
            required_folds={2},
            scorer_loader=lambda path: path,
        )

    bundle = _write_parent_model_bundle(tmp_path)
    raw = json.loads(bundle.read_text(encoding="utf-8"))
    raw["outer_maps"]["outer2"]["scorer_by_query_fold"]["3"]["model"] = "fit_2"
    bundle.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="leaky parent model"):
        _load_parent_model_recipe(
            bundle,
            recipe_name="outer2",
            required_folds={2, 3, 4},
            scorer_loader=lambda path: path,
        )


def test_latency_is_carried_in_candidate_schema_and_percentiles() -> None:
    cascade = RetrievalTop100CascadeV12(_Sidecars(()), _config())
    result = cascade.cascade([_row("11111111100001")], query_id="q1")
    metadata = {
        "fold": 2,
        "crm_name": "ALPHA",
        "crm_address": "10 RUE DE LA PAIX",
        "crm_insee": "75056",
        "crm_postcode": "75001",
        "retrieval_latency_ms": 12.5,
    }
    master = _candidate_rows(result.master_rows("q1", metadata))
    admission = _candidate_rows(result.admission_rows("q1", metadata))

    assert "retrieval_latency_ms" in _CANDIDATE_SCHEMA.names
    assert master[0]["retrieval_latency_ms"] == 12.5
    assert admission[0]["retrieval_latency_ms"] == 12.5
    assert _percentiles([1.0, 2.0, 3.0, 4.0]) == {
        "count": 4,
        "p50": 2.5,
        "p95": pytest.approx(3.85),
        "p99": pytest.approx(3.97),
    }


def test_source_manifest_requires_candidate_content_hash(tmp_path: Path) -> None:
    source = tmp_path / "unsealed.parquet"
    pq.write_table(
        pa.Table.from_pylist([{**_row("12345678900001"), "fold": 2}]), source
    )
    manifest = _official_v11_manifest(source)
    manifest.pop("candidate_parquet_sha256")
    source.with_suffix(source.suffix + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="candidate_parquet_sha256"):
        cascade_cli._load_source_manifest(source, allow_unsealed_fixture=False)


def test_source_manifest_rejects_self_declared_sealed_fixture(
    tmp_path: Path,
) -> None:
    source = tmp_path / "forged.parquet"
    pq.write_table(pa.Table.from_pylist([_row("12345678900001")]), source)
    manifest = _official_v11_manifest(
        source, run_kind="SEALED_NATURAL_V11_FIXTURE"
    )
    source.with_suffix(source.suffix + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="official label-blind v11 run"):
        cascade_cli._load_source_manifest(source, allow_unsealed_fixture=True)


def test_cli_executes_outer_specific_fold_routing_and_seals_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "natural_union.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    **_row(f"{fold:09d}00001"),
                    "query_id": f"q{fold}",
                    "fold": fold,
                    "retrieval_latency_ms": 5.0 + fold,
                }
                for fold in (2, 3, 4)
            ]
        ),
        source,
    )
    _write_official_combined_v11_manifest(source)
    sidecars = tmp_path / "sidecars"
    sidecars.mkdir()
    (sidecars / "manifest.json").write_text("{}\n", encoding="utf-8")
    bundle = _write_parent_model_bundle(
        tmp_path,
        source_retrieval_lineage={
            "source_candidate_sha256": sha256_file(source),
            "source_manifest_sha256": sha256_file(
                source.with_suffix(source.suffix + ".manifest.json")
            ),
            "sidecar_manifest_sha256": sha256_file(sidecars / "manifest.json"),
        },
    )
    calls: dict[str, int] = {}

    class Scorer:
        def __init__(self, name: str) -> None:
            self.name = name

        def score(self, features: Mapping[str, float]) -> float:
            calls[self.name] = calls.get(self.name, 0) + 1
            return float(sum(features.values()))

    monkeypatch.setattr(
        cascade_cli.XGBoostParentScorerV12,
        "from_model_path",
        lambda path: Scorer(Path(path).name),
    )

    class Reader:
        def __init__(self, _path: Path) -> None:
            self.sidecars = _Sidecars(())

        def __enter__(self) -> _Sidecars:
            return self.sidecars

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(cascade_cli, "RetrievalSidecarReader", Reader)
    output = tmp_path / "cascade_outer2"
    args = cascade_cli.parse_args(
        [
            "--input",
            str(source),
            "--sidecars",
            str(sidecars),
            "--output",
            str(output),
            "--parent-model-bundle",
            str(bundle),
            "--parent-outer-recipe",
            "outer2",
        ]
    )
    cascade_cli.run(args)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parent_scorer_kind"] == "NESTED_OOF_FOLD_MODEL_MAPPING"
    assert manifest["gate_eligible"] is True
    assert manifest["bootstrap"] is False
    assert manifest["outer_heldout_fold"] == 2
    assert manifest["candidate_generation"] == {
        "mode": "NESTED_OUTER_OOF",
        "heldout_parent_fit_folds": [3, 4],
        "training_fold_parent_fit_folds": {"3": [4], "4": [3]},
    }
    assert manifest["nested_parent_oof"]["gate_eligible"] is True
    assert manifest["candidate_parent_scores"] == {
        "fields": ["parent_score", "parent_rank"],
        "parent_feature_names": list(CASCADE_PARENT_FEATURES),
        "parent_feature_values_embedded": True,
        "scorer_kind": "NESTED_OOF_FOLD_MODEL_MAPPING",
        "computed_before_parent_expansion": True,
        "complete_for_all_master_and_hydration_candidates": True,
        "sealed_by_nested_model_bundle": True,
    }
    assert manifest["views"]["master_references.parquet"] == {
        "union_rank_semantics": "MASTER_RANK_WITHIN_MAX_2000",
        "hydration_rank_preserved": True,
        "parent_score_rank_preserved": True,
        "site_lambdamart_input": True,
        "admission_input": True,
    }
    assert manifest["views"]["hydration_shortlist.parquet"] == {
        "union_rank_semantics": "HEURISTIC_HYDRATION_RANK_WITHIN_MAX_500",
        "master_rank_preserved": True,
        "parent_score_rank_preserved": True,
        "heuristic_only": True,
        "admission_input": False,
    }
    assert manifest["parent_model_hashes"] == {
        str(fold): manifest["nested_parent_oof"]["scorer_by_query_fold"][str(fold)][
            "sha256"
        ]
        for fold in (2, 3, 4)
    }
    assert set(calls) == {"fit_34.json", "fit_4.json", "fit_3.json"}
    assert all(count > 0 for count in calls.values())
    assert manifest["retrieval_latency_ms"]["complete_end_to_end"] is True
    assert manifest["retrieval_latency_ms"]["total"]["count"] == 3
    assert manifest["upstream_rescue_reference_count"] == 0
    assert manifest["outputs"] == manifest["files"]
    verified_manifest, _ = verify_parent_candidates(
        output / "parent_candidates.parquet",
        require_nested_oof=True,
        expected_outer_fold=2,
    )
    assert verified_manifest == output / "manifest.json"
    forged = dict(manifest)
    forged["upstream_rescue_reference_count"] = 1
    (output / "manifest.json").write_text(
        json.dumps(forged), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exclude every deferred rescue"):
        verify_parent_candidates(
            output / "parent_candidates.parquet",
            require_nested_oof=True,
            expected_outer_fold=2,
        )
    hydrated = pq.read_table(output / "hydration_shortlist.parquet")
    assert all(value >= 7.0 for value in hydrated["retrieval_latency_ms"].to_pylist())

    mismatched_bundle = json.loads(bundle.read_text(encoding="utf-8"))
    mismatched_bundle["source_retrieval_lineage"][
        "source_candidate_sha256"
    ] = "0" * 64
    bundle.write_text(json.dumps(mismatched_bundle), encoding="utf-8")
    mismatched_args = cascade_cli.parse_args(
        [
            "--input",
            str(source),
            "--sidecars",
            str(sidecars),
            "--output",
            str(tmp_path / "cascade_mismatched_bundle"),
            "--parent-model-bundle",
            str(bundle),
            "--parent-outer-recipe",
            "outer2",
        ]
    )
    with pytest.raises(ValueError, match="another retrieval/sidecar snapshot"):
        cascade_cli.run(mismatched_args)
