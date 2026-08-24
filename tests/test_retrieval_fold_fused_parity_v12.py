from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pytest

import src.xgb_matcher.retrieval_fold_fused_parity_v12 as parity_module
from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (
    FittedSiteModel,
    FoldFusedInvariantError,
    HumanLabel,
)
from src.xgb_matcher.retrieval_fold_fused_heldout_spool_v12 import (
    validate_spool_calibration_v12,
)
from src.xgb_matcher.retrieval_fold_fused_parity_v12 import (
    FoldFusedParityInputsV12,
    SiteModelCatalogV12,
    run_fold_fused_parity_v12,
)
from src.xgb_matcher.retrieval_fold_fused_v12 import (
    LinearParentMatrixScorerV12,
    PARENT_MODEL_KEYS_V12,
)
from src.xgb_matcher.retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    site_feature_names,
)
from src.xgb_matcher.retrieval_sidecars_v11 import SiteQueryResult
from src.xgb_matcher.retrieval_top100_cascade_v12 import (
    CASCADE_PARENT_FEATURES,
    RetrievalTop100CascadeV12Config,
)
from src.xgb_matcher.retrieval_v12_scene_ir import SceneIR


class _OneSceneCache:
    def __init__(self, scene: SceneIR) -> None:
        self.scene = scene

    def iter_scenes(self, *, batch_size: int):
        assert batch_size > 0
        yield self.scene


class _OneLabelLedger:
    def __init__(self, label: HumanLabel) -> None:
        self.label = label

    def fold_by_query(self):
        return {self.label.query_id: self.label.fold}

    def get(self, query_id: str) -> HumanLabel:
        assert query_id == self.label.query_id
        return self.label


class _EmptyOfficialSidecars:
    def all_sites_by_insee_if_fits(self, insee: str, *, max_sites: int):
        return SiteQueryResult((), "INSEE_TOO_LARGE", 50_000, max_sites, True)

    def sites_by_sirens_geo(self, **kwargs):
        return {
            str(siren): SiteQueryResult(
                (), "SIREN_GEO_BATCH", 0, int(kwargs["max_sites_per_siren"]), False
            )
            for siren in kwargs["sirens"]
        }

    def sites_by_siren_geo(self, **kwargs):
        return SiteQueryResult(
            (), "SIREN_GEO", 0, int(kwargs["max_sites"]), False
        )


class _Predictor:
    def predict(self, values):
        return values[:, 0]


def _row(siret: str, rank: int, channels: list[str]) -> dict[str, object]:
    return {
        "query_id": "q-parity",
        "fold": 2,
        "crm_name": "ALPHA",
        "crm_address": "10 RUE DE LA PAIX",
        "crm_number": "10",
        "crm_insee": "75056",
        "crm_postcode": "75001",
        "union_rank": rank,
        "siret": siret,
        "siren": siret[:9],
        "channels": channels,
        "sources": ["SIRENE_CURRENT:fixture"],
        "signals": [
            {
                "source": "SIRENE_CURRENT:fixture",
                "channel": channel,
                "rank": rank,
                "score": 1.0 / rank,
            }
            for channel in channels
        ],
        "candidate_names": ["ALPHA" if rank == 1 else f"CONCURRENT {rank}"],
        "candidate_addresses": ["10 RUE DE LA PAIX"],
        "candidate_number": "10",
        "candidate_insee": "75056",
        "candidate_postcode": "75001",
        "candidate_state": "A",
        "official_catalog_present": True,
    }


def test_prefit_parity_harness_runs_actual_legacy_and_fused_paths(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(parity_module, "PARITY_ROWS", 1)
    rows = [
        _row("11111111100001", 1, ["legal_name_exact", "address_word"]),
        _row("22222222200001", 2, ["name_word"]),
        _row("33333333300001", 3, ["address_exact"]),
    ]
    table = pa.Table.from_pylist(rows)
    scene = SceneIR("q-parity", 2, 0, table)
    weights = [0.0] * len(CASCADE_PARENT_FEATURES)
    weights[CASCADE_PARENT_FEATURES.index("exact_name_evidence")] = 1.0
    scorers = {
        key: LinearParentMatrixScorerV12(tuple(weights))
        for key in PARENT_MODEL_KEYS_V12
    }
    label = HumanLabel(
        query_id="q-parity",
        fold=2,
        human_siret="11111111100001",
        operational_sirets=("11111111100001",),
    )
    config = RetrievalTop100CascadeV12Config(
        master_reference_cap=20,
        hydration_shortlist_cap=10,
        parent_shortlist_cap=3,
        small_insee_threshold=5,
        parent_expansion_budget=5,
        max_sites_per_parent=10,
        relation_seed_limit=4,
        relation_reference_budget=4,
        rescue_parent_budget=1,
        rescue_parent_reserve=1,
        rescue_min_name_parents=1,
        rescue_min_strong_parents=1,
    )
    digest = hashlib.sha256(b"lineage").hexdigest()
    inputs = FoldFusedParityInputsV12(
            scene_cache=_OneSceneCache(scene),  # type: ignore[arg-type]
            query_sample=pd.DataFrame({"query_id": ["q-parity"], "fold": [2]}),
            query_sample_path=tmp_path / "sample.parquet",
            ledger=_OneLabelLedger(label),  # type: ignore[arg-type]
            sidecars=_EmptyOfficialSidecars(),
            parent_scorers=scorers,
            admission_config=AdmissionConfigV12.load(
                "config/retrieval_all_human_v12.json"
            ),
            cascade_config=config,
            lineage={"fixture_sha256": digest},
            national_query_count=1,
        )
    manifest = run_fold_fused_parity_v12(
        inputs,
        output_dir=tmp_path / "parity",
    )
    result = json.loads(manifest.read_text())
    assert result["status"] == "PASS"
    assert result["phase"] == "PREFIT_NON_CIRCULAR"
    assert result["context_count"] == 3
    assert all(result["checks"].values())
    calibration = validate_spool_calibration_v12(
        manifest,
        result["heldout_spool_calibration"],
        expected_sample_queries=1,
        expected_national_queries=1,
        expected_maximum_bytes=20 * 1024**3,
    )
    assert calibration["actual_sample_size_bytes"] > 0
    assert calibration["projected_national_size_bytes"] == calibration[
        "actual_sample_size_bytes"
    ]

    model_features = tuple(site_feature_names(inputs.admission_config))
    model_hash = hashlib.sha256(b"model").hexdigest()
    catalog = SiteModelCatalogV12(
        manifest_path=tmp_path / "models.json",
        manifest_sha256=hashlib.sha256(b"models").hexdigest(),
        models={
            outer: FittedSiteModel(
                outer_fold=outer,
                handle=_Predictor(),
                path=tmp_path / f"model-{outer}.json",
                sha256=model_hash,
                recipe_sha256=hashlib.sha256(b"recipe").hexdigest(),
                feature_names=model_features,
            )
            for outer in (2, 3, 4)
        },
    )
    complete_manifest = run_fold_fused_parity_v12(
        replace(inputs, site_models=catalog),
        output_dir=tmp_path / "parity-complete",
    )
    complete = json.loads(complete_manifest.read_text())
    assert complete["schema_version"] == "sireto-retrieval-v12-fold-fused-parity-1"
    assert complete["phase"] == "FULL_POSTFIT"
    assert set(complete["checks"]) == {
        "parent_scores_bit_identical",
        "master_candidates_and_ranks_byte_identical",
        "site_features_float32_bit_identical",
        "fit_row_ids_and_order_byte_identical",
        "prerank500_byte_identical",
        "top100_byte_identical",
    }
    assert all(complete["checks"].values())

    calibration_path = (
        complete_manifest.parent
        / "heldout_spool_calibration"
        / "heldout_master.parquet"
    )
    with calibration_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(FoldFusedInvariantError, match="bytes/hash drifted"):
        validate_spool_calibration_v12(
            complete_manifest,
            complete["heldout_spool_calibration"],
            expected_sample_queries=1,
            expected_national_queries=1,
            expected_maximum_bytes=20 * 1024**3,
        )
