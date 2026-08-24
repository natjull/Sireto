from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from scripts import run_retrieval_hierarchical_admission_v12 as admission_cli
from scripts import run_retrieval_fold_fused_parent_v12 as parent_cli
from scripts import run_retrieval_top100_cascade_v12 as cascade_cli
from src.xgb_matcher.retrieval_fold_fused_parent_bundle_v12 import (
    OUTER_PARENT_MAPS_V12,
    PARENT_FIT_FOLDS_V12,
    PARENT_SAMPLE_COLUMNS_V12,
    ParentBundleLineageV12,
    build_fold_fused_parent_bundle_v12,
    collect_parent_fit_sample_v12,
    deterministic_bootstrap_parent_pool_v12,
    sample_bootstrap_parent_pool_v12,
    validate_strict_parent_labels_v12,
)
from src.xgb_matcher.retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    fit_parent_ranker_from_cascade_rows,
)
from src.xgb_matcher import retrieval_fold_fused_runtime_v12 as fused_runtime
from src.xgb_matcher.retrieval_sidecars_v11 import SiteQueryResult
from src.xgb_matcher.retrieval_top100_cascade_v12 import (
    CASCADE_PARENT_FEATURES,
    RetrievalTop100CascadeV12,
    RetrievalTop100CascadeV12Config,
)
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIR


def _admission_config(*, cap: int = 4) -> AdmissionConfigV12:
    raw = deepcopy(dict(AdmissionConfigV12.default().raw))
    raw["admission_v12"]["fit_candidate_sampling"][
        "maximum_candidates_per_query"
    ] = cap
    return AdmissionConfigV12.from_mapping(raw)


def _cascade_config() -> RetrievalTop100CascadeV12Config:
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


def _signal(siren: str, rank: int) -> dict[str, Any]:
    return {
        "source": "SIRENE_CURRENT:fixture",
        "channel": "legal_name_exact",
        "rank": rank,
        "score": 1.0 / rank,
        "parent_identifier": siren,
        "parent_rank": rank,
        "site_rank": 1,
        "relation_direction": "",
    }


def _scene_rows(fold: int) -> tuple[list[dict[str, Any]], str]:
    query_id = f"query-{fold}"
    rows: list[dict[str, Any]] = []
    for rank in range(1, 7):
        siren = f"{fold}{rank:08d}"
        rows.append(
            {
                "query_id": query_id,
                "fold": fold,
                "crm_name": "ALPHA",
                "crm_address": "10 RUE DE LA PAIX",
                "crm_insee": "75056",
                "crm_postcode": "75001",
                "union_rank": rank,
                "siret": siren + "00001",
                "siren": siren,
                "channels": ["legal_name_exact"],
                "sources": ["SIRENE_CURRENT:fixture"],
                "signals": [_signal(siren, rank)],
                "candidate_names": ["ALPHA"],
                "candidate_addresses": ["10 RUE DE LA PAIX"],
                "candidate_insee": "75056",
                "candidate_postcode": "75001",
                "candidate_number": "10",
                "candidate_state": "A",
            }
        )
    # Rank six is intentionally outside the <=4 head.  The old and fused
    # samplers must both replace row four with this naturally present parent.
    return rows, f"{fold}{6:08d}" + "00001"


@dataclass
class _FixtureSceneCache:
    scenes: tuple[SceneIR, ...]

    def __post_init__(self) -> None:
        self.manifest = {
            "query_count": len(self.scenes),
            "fold_counts": {
                str(fold): sum(scene.fold == fold for scene in self.scenes)
                for fold in (2, 3, 4)
            },
        }

    def iter_scenes(
        self,
        *,
        columns: Any = None,
        batch_size: int = 262_144,
    ) -> Iterator[SceneIR]:
        del columns, batch_size
        yield from self.scenes


class _NoExpansionSidecars:
    def all_sites_by_insee_if_fits(
        self, insee: str, *, max_sites: int
    ) -> SiteQueryResult:
        del insee
        return SiteQueryResult((), "INSEE_TOO_LARGE", 50_000, max_sites, True)

    def sites_by_sirens_geo(self, **kwargs: Any) -> dict[str, SiteQueryResult]:
        cap = int(kwargs["max_sites_per_siren"])
        return {
            str(siren): SiteQueryResult((), "SIREN_GEO_BATCH", 0, cap, False)
            for siren in kwargs["sirens"]
        }


def _fixture() -> tuple[_FixtureSceneCache, pd.DataFrame]:
    scenes: list[SceneIR] = []
    labels: list[dict[str, Any]] = []
    row_start = 0
    for fold in (2, 3, 4):
        rows, human_siret = _scene_rows(fold)
        scenes.append(
            SceneIR(
                query_id=f"query-{fold}",
                fold=fold,
                row_start=row_start,
                rows=pa.Table.from_pylist(rows),
            )
        )
        row_start += len(rows)
        labels.append(
            {
                "query_id": f"query-{fold}",
                "fold": fold,
                "human_siret": human_siret,
            }
        )
    return _FixtureSceneCache(tuple(scenes)), pd.DataFrame(labels)


def _legacy_pass_a_sample(
    tmp_path: Path,
    cache: _FixtureSceneCache,
    labels: pd.DataFrame,
    *,
    cascade_config: RetrievalTop100CascadeV12Config,
    admission_config: AdmissionConfigV12,
) -> tuple[pd.DataFrame, dict[str, int]]:
    sidecars = _NoExpansionSidecars()
    cascade = RetrievalTop100CascadeV12(
        sidecars,
        cascade_config,
        parent_scorer=None,
        rescue_provider=None,
    )
    parent_rows: list[dict[str, Any]] = []
    for scene in cache.iter_scenes():
        result = cascade.cascade(scene.as_mappings(), query_id=scene.query_id)
        parent_rows.extend(
            {**row, "fold": scene.fold}
            for row in result.parent_candidate_rows(scene.query_id)
        )
    path = tmp_path / "legacy-pass-a-parents.parquet"
    pd.DataFrame(parent_rows).to_parquet(path, index=False)
    sample, diagnostics = admission_cli._load_parent_fit_sample(
        path, labels, admission_config
    )
    return sample, dict(diagnostics)


def test_fused_sample_is_differentially_identical_to_pass_a_and_fit_loader(
    tmp_path: Path,
) -> None:
    cache, labels = _fixture()
    cascade_config = _cascade_config()
    admission_config = _admission_config(cap=4)
    legacy, legacy_diagnostics = _legacy_pass_a_sample(
        tmp_path,
        cache,
        labels,
        cascade_config=cascade_config,
        admission_config=admission_config,
    )
    fused, fused_diagnostics = collect_parent_fit_sample_v12(
        cache,
        sidecars=_NoExpansionSidecars(),
        cascade_config=cascade_config,
        admission_config=admission_config,
        labels=labels,
    )

    assert list(fused.columns) == list(legacy.columns) == list(
        PARENT_SAMPLE_COLUMNS_V12
    )
    pd.testing.assert_frame_equal(
        fused.reset_index(drop=True),
        legacy.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    assert fused_diagnostics.rows_before_sampling == legacy_diagnostics[
        "rows_before_sampling"
    ]
    assert fused_diagnostics.rows_after_sampling == legacy_diagnostics[
        "rows_after_sampling"
    ]
    assert (
        fused_diagnostics.natural_parent_positive_queries
        == legacy_diagnostics["natural_parent_positive_queries"]
        == 3
    )
    assert fused.groupby("query_id").size().eq(4).all()
    for row in labels.to_dict("records"):
        candidates = fused.loc[
            fused["query_id"].eq(row["query_id"]), "candidate_parent"
        ]
        assert str(row["human_siret"])[:9] in set(candidates)


def test_sampling_never_injects_an_absent_positive() -> None:
    cache, _labels = _fixture()
    scene = next(cache.iter_scenes())
    from src.xgb_matcher.retrieval_fold_fused_v12 import (  # local: fixture only
        compile_label_blind_scene_v12,
    )

    compiled = compile_label_blind_scene_v12(
        scene.as_mappings(),
        query_id=scene.query_id,
        sidecars=_NoExpansionSidecars(),
        config=_cascade_config(),
    )
    pool = deterministic_bootstrap_parent_pool_v12(
        compiled, config=_cascade_config()
    )
    missing = "29999999900001"
    sample = sample_bootstrap_parent_pool_v12(
        pool,
        human_siret=missing,
        maximum_candidates_per_query=4,
    )
    assert missing[:9] not in set(sample["candidate_parent"])
    assert list(sample["sealed_parent_rank"]) == [1.0, 2.0, 3.0, 4.0]


def test_label_cannot_rescue_truth_removed_by_label_blind_master_cap() -> None:
    cache, labels = _fixture()
    cascade_config = replace(
        _cascade_config(),
        master_reference_cap=5,
        hydration_shortlist_cap=5,
        parent_expansion_budget=5,
    )
    sample, diagnostics = collect_parent_fit_sample_v12(
        cache,
        sidecars=_NoExpansionSidecars(),
        cascade_config=cascade_config,
        admission_config=_admission_config(cap=4),
        labels=labels,
    )
    assert diagnostics.natural_positive_queries_before_master_cap == 3
    assert diagnostics.natural_parent_positive_queries == 0
    assert diagnostics.maximum_natural_parents_before_master_cap == 6
    assert diagnostics.maximum_bounded_parents == 5
    for row in labels.to_dict("records"):
        candidates = sample.loc[
            sample["query_id"].eq(row["query_id"]), "candidate_parent"
        ]
        assert str(row["human_siret"])[:9] not in set(candidates)


def test_strict_labels_reject_fold_1_and_cross_fold_siren_overlap() -> None:
    _cache, labels = _fixture()
    polluted = labels.copy()
    polluted.loc[0, "fold"] = 1
    try:
        validate_strict_parent_labels_v12(polluted)
    except ValueError as error:
        assert "folds 2/3/4" in str(error)
    else:
        raise AssertionError("fold 1 label was accepted")

    overlap = labels.copy()
    overlap.loc[1, "human_siret"] = str(overlap.loc[0, "human_siret"])
    try:
        validate_strict_parent_labels_v12(overlap)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("cross-fold truth SIREN overlap was accepted")


def test_parent_output_is_fail_closed_outside_external_scratch(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "bundle"
    resolved, free = parent_cli._require_external_parent_output(
        allowed,
        scratch_root=tmp_path,
        maximum_temporary_gib=1,
    )
    assert resolved == allowed.resolve()
    assert free >= 1024**3
    assert not allowed.exists()

    outside = tmp_path.parent / "outside-parent-bundle"
    with pytest.raises(parent_cli.FoldFusedInvariantError, match="external scratch"):
        parent_cli._require_external_parent_output(
            outside,
            scratch_root=tmp_path,
            maximum_temporary_gib=1,
        )


class _FixtureModel:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def save_model(self, path: Path | str) -> None:
        Path(path).write_text(self.payload + "\n", encoding="utf-8")


def test_bundle_fits_same_six_subsets_and_is_legacy_loader_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache, labels = _fixture()
    cascade_config = _cascade_config()
    admission_config = _admission_config(cap=4)
    expected_sample, expected_diagnostics = collect_parent_fit_sample_v12(
        cache,
        sidecars=_NoExpansionSidecars(),
        cascade_config=cascade_config,
        admission_config=admission_config,
        labels=labels,
    )
    fitted: list[pd.DataFrame] = []

    def recording_fitter(
        rows: pd.DataFrame, _config: AdmissionConfigV12
    ) -> tuple[_FixtureModel, dict[str, int]]:
        fitted.append(rows.copy())
        digest = hashlib.sha256(
            rows[["query_id", "candidate_parent"]].to_csv(index=False).encode()
        ).hexdigest()
        return _FixtureModel(digest), {
            "missing_natural_parent": 0,
            "without_parent_negative": 0,
        }

    lineage = ParentBundleLineageV12(
        source_candidate_sha256="1" * 64,
        source_manifest_sha256="2" * 64,
        sidecar_manifest_sha256="3" * 64,
        source_labels_sha256="4" * 64,
        source_labels_manifest_sha256="5" * 64,
        config_sha256="6" * 64,
        scene_ir_manifest_sha256="7" * 64,
        scene_ir_build_id="8" * 64,
    )
    output = tmp_path / "parent-bundle"
    build = build_fold_fused_parent_bundle_v12(
        cache,
        sidecars=_NoExpansionSidecars(),
        cascade_config=cascade_config,
        admission_config=admission_config,
        labels=labels,
        lineage=lineage,
        output_dir=output,
        model_fitter=recording_fitter,
    )
    manifest = json.loads(build.manifest_path.read_text(encoding="utf-8"))

    assert set(manifest["models"]) == set(PARENT_FIT_FOLDS_V12)
    assert manifest["outer_maps"] == {
        f"outer{outer}": {
            "outer_heldout_fold": int(outer),
            "scorer_by_query_fold": mapping,
        }
        for outer, mapping in OUTER_PARENT_MAPS_V12.items()
    }
    assert OUTER_PARENT_MAPS_V12 == admission_cli._OUTER_PARENT_MAPS
    assert manifest["physical_plan"] == {
        "id": "SCENE_IR_FUSED_BOOTSTRAP_SAMPLE_128_NO_FULL_PARENT_PARQUET",
        "scene_compilations_per_query": 1,
        "bootstrap_parent_score": "DETERMINISTIC_V12_DEFAULT",
        "label_attached_after_complete_natural_parent_pool": True,
        "fit_rows_maximum_per_query": 4,
        "full_parent_candidate_parquet_materialized": False,
        "temporary_fit_sample_removed_before_publication": True,
        "post_expansion_parent_rescore": False,
    }
    assert build.diagnostics.fit_sample_semantic_sha256 == (
        expected_diagnostics.fit_sample_semantic_sha256
    )
    assert not list(output.glob(".parent-fit-fold*.parquet"))
    assert set(path.name for path in output.iterdir()) == {
        "manifest.json",
        *(f"parent_{name}.json" for name in PARENT_FIT_FOLDS_V12),
    }

    for (name, folds), observed in zip(PARENT_FIT_FOLDS_V12.items(), fitted):
        expected = expected_sample[expected_sample["fold"].isin(folds)].reset_index(
            drop=True
        )
        expected.loc[:, list(CASCADE_PARENT_FEATURES)] = expected.loc[
            :, list(CASCADE_PARENT_FEATURES)
        ].astype("float32")
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=0.0,
            atol=0.0,
        )
        assert set(observed["fold"].astype(int)) == set(folds), name

    # Existing cascade code accepts the new manifest without a compatibility
    # branch.  A fixture scorer loader avoids interpreting the sealed dummy
    # model bytes as XGBoost while retaining every hash/route validation.
    recipe = cascade_cli._load_parent_model_recipe(
        build.manifest_path,
        recipe_name="outer2",
        required_folds={2, 3, 4},
        scorer_loader=lambda path: path,
    )
    assert recipe.outer_fold == 2
    assert set(recipe.scorers_by_fold) == {2, 3, 4}
    assert recipe.source_retrieval_lineage == lineage.source_retrieval_lineage

    monkeypatch.setattr(
        fused_runtime.XGBoostParentMatrixScorerV12,
        "from_model_path",
        classmethod(lambda _cls, path: Path(path).resolve()),
    )
    catalog = fused_runtime.load_parent_scorer_catalog_v12(
        build.manifest_path,
        config_sha256=lineage.config_sha256,
        source_candidate_sha256=lineage.source_candidate_sha256,
        source_manifest_sha256=lineage.source_manifest_sha256,
        sidecar_manifest_sha256=lineage.sidecar_manifest_sha256,
        label_ledger_sha256=lineage.source_labels_sha256,
        label_ledger_manifest_sha256=lineage.source_labels_manifest_sha256,
    )
    assert set(catalog.scorers) == set(PARENT_FIT_FOLDS_V12)
    assert catalog.source_retrieval_lineage == lineage.source_retrieval_lineage


def test_default_fitter_seals_six_loadable_xgboost_models(tmp_path: Path) -> None:
    cache, labels = _fixture()
    cascade_config = _cascade_config()
    admission_config = _admission_config(cap=4)
    legacy_sample, _legacy_diagnostics = _legacy_pass_a_sample(
        tmp_path,
        cache,
        labels,
        cascade_config=cascade_config,
        admission_config=admission_config,
    )
    lineage = ParentBundleLineageV12(
        source_candidate_sha256="a" * 64,
        source_manifest_sha256="b" * 64,
        sidecar_manifest_sha256="c" * 64,
        source_labels_sha256="d" * 64,
        source_labels_manifest_sha256="e" * 64,
        config_sha256="f" * 64,
        scene_ir_manifest_sha256="0" * 64,
        scene_ir_build_id="1" * 64,
    )
    build = build_fold_fused_parent_bundle_v12(
        cache,
        sidecars=_NoExpansionSidecars(),
        cascade_config=cascade_config,
        admission_config=admission_config,
        labels=labels,
        lineage=lineage,
        output_dir=tmp_path / "real-xgboost-parent-bundle",
    )
    catalog = fused_runtime.load_parent_scorer_catalog_v12(
        build.manifest_path,
        config_sha256=lineage.config_sha256,
        source_candidate_sha256=lineage.source_candidate_sha256,
        source_manifest_sha256=lineage.source_manifest_sha256,
        sidecar_manifest_sha256=lineage.sidecar_manifest_sha256,
        label_ledger_sha256=lineage.source_labels_sha256,
        label_ledger_manifest_sha256=lineage.source_labels_manifest_sha256,
    )
    matrix = np.zeros((2, len(CASCADE_PARENT_FEATURES)), dtype=np.float32)
    assert set(catalog.scorers) == set(PARENT_FIT_FOLDS_V12)
    for scorer in catalog.scorers.values():
        assert scorer.score_matrix(matrix).shape == (2,)

    # The default fused fit is byte-identical to invoking the current parent
    # fitter on the current pass-A + _load_parent_fit_sample fixture rows.
    for name, folds in PARENT_FIT_FOLDS_V12.items():
        legacy_model, _diagnostics = fit_parent_ranker_from_cascade_rows(
            legacy_sample[legacy_sample["fold"].isin(folds)],
            admission_config,
        )
        legacy_path = tmp_path / f"legacy-parent-{name}.json"
        legacy_model.save_model(legacy_path)
        assert legacy_path.read_bytes() == build.model_paths[name].read_bytes()
