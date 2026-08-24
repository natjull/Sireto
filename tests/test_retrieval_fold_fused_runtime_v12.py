from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (
    HumanLabel,
    MasterContext,
)
from src.xgb_matcher.retrieval_fold_fused_runtime_v12 import (
    PREFIT_PARITY_CHECKS,
    PREFIT_PARITY_SCHEMA_VERSION,
    ExistingV12FitSamplerAdapter,
    FoldFusedContextAdapterV12,
    _MasterPayload,
    load_parity_evidence_v12,
    load_prefit_parity_evidence_v12,
)
from src.xgb_matcher.retrieval_fold_fused_oof_v12 import REQUIRED_PARITY_CHECKS
from src.xgb_matcher.retrieval_fold_fused_certification_v12 import (
    certify_fold_fused_oof_postfit_v12,
)
from src.xgb_matcher.retrieval_fold_fused_v12 import PARENT_MODEL_KEYS_V12
from src.xgb_matcher.retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    canonicalize_union,
    sample_precomputed_candidates,
    site_feature_names,
)
from src.xgb_matcher.retrieval_top100_cascade_v12 import (
    CASCADE_PARENT_FEATURES,
    RetrievalTop100CascadeV12Config,
)


def _master_rows(count: int = 140) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        parent_index = index // 2
        parent = f"{410_000_000 + parent_index:09d}"
        siret = f"{parent}{index % 2 + 1:05d}"
        channels = ["legal_name_word"]
        if index % 7 == 0:
            channels.append("address_word")
        if index % 19 == 0:
            channels.append("geo_exact_cell")
        parent_values = [
            float(parent_index + feature_index / 100.0)
            for feature_index, _name in enumerate(CASCADE_PARENT_FEATURES)
        ]
        row: dict[str, object] = {
            "query_id": "q-lazy-parity",
            "fold": 2,
            "crm_name": "ALPHA SERVICES",
            "crm_address": "10 RUE DE LA PAIX 75001 PARIS",
            "crm_number": "10",
            "crm_postcode": "75001",
            "crm_insee": "75056",
            "siret": siret,
            "siren": parent,
            "union_rank": index + 1,
            "master_rank": index + 1,
            "hydration_rank": index + 1 if index < 100 else None,
            "parent_rank": parent_index + 1,
            "parent_score": 10.0 - parent_index / 10.0,
            "parent_feature_names": list(CASCADE_PARENT_FEATURES),
            "parent_feature_values": parent_values,
            "candidate_names": [
                "ALPHA SERVICES" if index % 11 == 0 else f"ALPHA CONCURRENT {index}"
            ],
            "candidate_addresses": [
                "10 RUE DE LA PAIX 75001 PARIS"
                if index % 13 == 0
                else f"{20 + index} RUE AUTRE 75001 PARIS"
            ],
            "candidate_number": "10" if index % 13 == 0 else str(20 + index),
            "candidate_postcode": "75001",
            "candidate_insee": "75056",
            "candidate_state": "A" if index % 5 else "F",
            "candidate_is_headquarters": index % 2 == 0,
            "channels": channels,
            "sources": ["SIRENE_CURRENT:fixture"],
            "official_evidence_sources": ["SIRENE_CURRENT:fixture"],
            "signals": (
                [
                    {
                        "channel": "bodacc_relation",
                        "rank": 2,
                        "score": 0.75,
                        "source": "BODACC",
                        "relation_direction": "PREDECESSOR",
                        "event_date": "2020-01-02",
                    }
                ]
                if index == 37
                else []
            ),
            "official_catalog_present": True,
        }
        for channel_index, channel in enumerate(channels):
            row[f"{channel}_rank"] = index + 1 + channel_index
            row[f"{channel}_score"] = 5.0 - index / 100.0 - channel_index / 10.0
        rows.append(row)
    return rows


def _adapter(rows: list[dict[str, object]]):
    admission = AdmissionConfigV12.load("config/retrieval_all_human_v12.json")
    adapter = FoldFusedContextAdapterV12(
        sidecars=object(),
        parent_scorers={key: object() for key in PARENT_MODEL_KEYS_V12},
        cascade_config=RetrievalTop100CascadeV12Config.load(
            "config/retrieval_all_human_v12.json"
        ),
        admission_config=admission,
    )
    context = MasterContext(
        query_id="q-lazy-parity",
        scene_fold=2,
        outer_fold=3,
        candidate_sirets=tuple(str(row["siret"]) for row in rows),
        candidate_parents=tuple(str(row["siren"]) for row in rows),
        opaque=_MasterPayload(tuple(rows)),
    )
    return admission, adapter, context


def _float32_bits(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    values = np.ascontiguousarray(
        frame[list(features)].fillna(0.0).to_numpy(dtype=np.float32),
        dtype=np.float32,
    )
    return values.view(np.uint32)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parity_bundle(
    root: Path,
    *,
    lineage: dict[str, str],
    complete: bool,
    site_model_manifest_sha: str | None = None,
    site_model_hashes: dict[str, str] | None = None,
    site_model_recipe_hashes: dict[str, str] | None = None,
) -> Path:
    root.mkdir()
    checks = REQUIRED_PARITY_CHECKS if complete else PREFIT_PARITY_CHECKS
    row_records: list[dict[str, object]] = []
    digest = hashlib.sha256(b"row-digest").hexdigest()
    for query_index in range(512):
        for outer_fold in (2, 3, 4):
            row: dict[str, object] = {
                "schema_version": "sireto-retrieval-v12-fold-fused-parity-rows-1",
                "query_id": f"q-{query_index:04d}",
                "fold": 2 + query_index % 3,
                "outer_fold": outer_fold,
                "legacy_master_sha256": digest,
                "fused_master_sha256": digest,
                "legacy_features_sha256": digest,
                "fused_features_sha256": digest,
                "legacy_parent_sha256": digest,
                "fused_parent_sha256": digest,
                "legacy_fit_rows_sha256": digest,
                "fused_fit_rows_sha256": digest,
                "legacy_prerank500_sha256": digest,
                "fused_prerank500_sha256": digest,
                "legacy_top100_sha256": digest,
                "fused_top100_sha256": digest,
            }
            row.update({name: True for name in checks})
            row_records.append(row)
    rows_path = root / "parity_rows.parquet"
    pd.DataFrame(row_records).to_parquet(rows_path, index=False)
    model_sha = (
        site_model_manifest_sha or hashlib.sha256(b"site-models").hexdigest()
        if complete
        else None
    )
    model_hashes = site_model_hashes or (
        {
            str(outer): hashlib.sha256(f"site-{outer}".encode()).hexdigest()
            for outer in (2, 3, 4)
        }
        if complete
        else None
    )
    recipe_hash = hashlib.sha256(b"site-recipe").hexdigest()
    recipe_hashes = site_model_recipe_hashes or (
        {str(outer): recipe_hash for outer in (2, 3, 4)} if complete else None
    )
    reference = {
        "schema_version": "sireto-retrieval-v12-fold-fused-reference-1",
        "engine": "LEGACY_SINGLE_CONTEXT_V12",
        "query_count": 512,
        "context_count": 1536,
        "folds": [2, 3, 4],
        "lineage": lineage,
        "site_model_manifest_sha256": model_sha,
        "site_model_sha256": model_hashes,
        "site_model_recipe_sha256": recipe_hashes,
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
    }
    reference_path = root / "reference_manifest.json"
    reference_path.write_text(json.dumps(reference, sort_keys=True), encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": (
            "sireto-retrieval-v12-fold-fused-parity-1"
            if complete
            else PREFIT_PARITY_SCHEMA_VERSION
        ),
        "status": "PASS",
        "phase": "FULL_POSTFIT" if complete else "PREFIT_NON_CIRCULAR",
        "reference_manifest_sha256": _sha256_file(reference_path),
        "query_count": 512,
        "context_count": 1536,
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "checks": {name: True for name in checks},
        "stop_on_any_divergence": True,
        "postfit_checks_status": "PASS" if complete else "PENDING_SITE_MODELS",
        "lineage": lineage,
        "artifacts": {
            "reference_manifest": {
                "path": reference_path.name,
                "sha256": _sha256_file(reference_path),
            },
            "parity_rows": {
                "path": rows_path.name,
                "sha256": _sha256_file(rows_path),
                "rows": len(row_records),
            },
        },
    }
    if complete:
        manifest["required_checks"] = list(REQUIRED_PARITY_CHECKS)
        manifest["site_model_manifest_sha256"] = model_sha
        manifest["site_model_sha256"] = model_hashes
        manifest["site_model_recipe_sha256"] = recipe_hashes
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


def test_lazy_fit_materialization_is_bit_identical_to_full_then_sample() -> None:
    rows = _master_rows()
    admission, adapter, context = _adapter(rows)
    statistics = adapter.compute_full_master_statistics(context)
    state = statistics.opaque

    # Computing complete-Master sufficient statistics must not accidentally
    # perform the expensive 319-feature canonicalization.
    assert state.full_canonical is None
    assert state.full_matrix is None
    assert tuple(state.feature_names) == site_feature_names(admission)

    truth = str(rows[133]["siret"])
    label = HumanLabel(
        query_id=context.query_id,
        fold=2,
        human_siret=truth,
        operational_sirets=(truth,),
    )
    plan = ExistingV12FitSamplerAdapter(admission).choose(
        context,
        statistics,
        label,
        maximum_rows=128,
    )
    lazy = adapter.materialize(
        context,
        statistics,
        candidate_indices=plan.candidate_indices,
    )

    complete = canonicalize_union(
        pd.DataFrame(rows), admission, labels_allowed=False
    )
    complete["human_siret"] = truth
    complete["operational_sirets_json"] = json.dumps([truth])
    expected = sample_precomputed_candidates(complete, admission)

    assert lazy.candidate_sirets == tuple(expected["candidate_siret"].astype(str))
    assert np.array_equal(
        lazy.values.view(np.uint32),
        _float32_bits(expected, tuple(site_feature_names(admission))),
    )
    # The fit-only call stays lazy even after materialising its selected rows.
    assert state.full_canonical is None
    assert state.full_matrix is None


def test_heldout_materialization_keeps_full_master_and_canonical_parity() -> None:
    rows = _master_rows(40)
    admission, adapter, context = _adapter(rows)
    statistics = adapter.compute_full_master_statistics(context)
    heldout = adapter.materialize(context, statistics, candidate_indices=None)
    expected = canonicalize_union(
        pd.DataFrame(rows), admission, labels_allowed=False
    )

    assert heldout.candidate_sirets == tuple(expected["candidate_siret"].astype(str))
    assert np.array_equal(
        heldout.values.view(np.uint32),
        _float32_bits(expected, tuple(site_feature_names(admission))),
    )
    assert statistics.opaque.full_canonical is not None
    assert statistics.opaque.full_matrix is not None


def test_prefit_and_complete_parity_schemas_are_distinct_and_lineage_bound(
    tmp_path,
) -> None:
    expected_lineage = {
        "scientific_config_sha256": hashlib.sha256(b"config").hexdigest(),
        "scene_ir_manifest_sha256": hashlib.sha256(b"scene").hexdigest(),
    }
    prefit_path = _write_parity_bundle(
        tmp_path / "prefit", lineage=expected_lineage, complete=False
    )
    loaded, digest = load_prefit_parity_evidence_v12(
        prefit_path, expected_lineage=expected_lineage
    )
    assert loaded["status"] == "PASS"
    assert len(digest) == 64

    complete_path = _write_parity_bundle(
        tmp_path / "complete", lineage=expected_lineage, complete=True
    )
    evidence, _digest = load_parity_evidence_v12(
        complete_path, expected_lineage=expected_lineage
    )
    assert evidence.as_manifest()["status"] == "PASS"

    wrong = dict(expected_lineage)
    wrong["scene_ir_manifest_sha256"] = hashlib.sha256(b"other").hexdigest()
    import pytest

    with pytest.raises(Exception, match="lineage mismatch"):
        load_prefit_parity_evidence_v12(prefit_path, expected_lineage=wrong)


def test_parity_gate_rejects_manifest_only_claim_and_tampered_rows(tmp_path) -> None:
    import pytest

    lineage = {"scientific_config_sha256": hashlib.sha256(b"config").hexdigest()}
    forged = tmp_path / "forged.json"
    forged.write_text(
        json.dumps(
            {
                "schema_version": PREFIT_PARITY_SCHEMA_VERSION,
                "status": "PASS",
                "phase": "PREFIT_NON_CIRCULAR",
                "reference_manifest_sha256": hashlib.sha256(b"fake").hexdigest(),
                "query_count": 512,
                "context_count": 1536,
                "fold0_opened": False,
                "fold1_opened": False,
                "positive_injection": False,
                "stop_on_any_divergence": True,
                "postfit_checks_status": "PENDING_SITE_MODELS",
                "checks": {name: True for name in PREFIT_PARITY_CHECKS},
                "lineage": lineage,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="artifact set"):
        load_prefit_parity_evidence_v12(forged, expected_lineage=lineage)

    manifest_path = _write_parity_bundle(
        tmp_path / "tampered", lineage=lineage, complete=False
    )
    rows_path = manifest_path.parent / "parity_rows.parquet"
    rows = pd.read_parquet(rows_path)
    rows.loc[0, PREFIT_PARITY_CHECKS[0]] = False
    rows.to_parquet(rows_path, index=False)
    with pytest.raises(Exception, match="artifact hash mismatch"):
        load_prefit_parity_evidence_v12(manifest_path, expected_lineage=lineage)


def test_parity_gate_recomputes_row_checks_after_valid_rehash(tmp_path) -> None:
    import pytest

    lineage = {"scientific_config_sha256": hashlib.sha256(b"config").hexdigest()}
    manifest_path = _write_parity_bundle(
        tmp_path / "false-row", lineage=lineage, complete=True
    )
    rows_path = manifest_path.parent / "parity_rows.parquet"
    rows = pd.read_parquet(rows_path)
    rows.loc[0, REQUIRED_PARITY_CHECKS[-1]] = False
    rows.to_parquet(rows_path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["parity_rows"]["sha256"] = _sha256_file(rows_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(Exception, match="row check failed"):
        load_parity_evidence_v12(manifest_path, expected_lineage=lineage)


def test_parity_gate_recomputes_digest_equality_after_valid_rehash(tmp_path) -> None:
    import pytest

    lineage = {"scientific_config_sha256": hashlib.sha256(b"config").hexdigest()}
    manifest_path = _write_parity_bundle(
        tmp_path / "digest-divergence", lineage=lineage, complete=True
    )
    rows_path = manifest_path.parent / "parity_rows.parquet"
    rows = pd.read_parquet(rows_path)
    assert bool(rows.loc[0, "parent_scores_bit_identical"]) is True
    rows.loc[0, "fused_parent_sha256"] = hashlib.sha256(b"forged").hexdigest()
    rows.to_parquet(rows_path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["parity_rows"]["sha256"] = _sha256_file(rows_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(Exception, match="row check failed"):
        load_parity_evidence_v12(manifest_path, expected_lineage=lineage)


def _write_fake_prefit_only_oof(root: Path) -> tuple[Path, dict[str, str], str]:
    root.mkdir()
    lineage_keys = (
        "scene_ir_sha256",
        "scene_ir_manifest_sha256",
        "source_candidate_sha256",
        "source_manifest_sha256",
        "label_ledger_sha256",
        "label_ledger_manifest_sha256",
        "label_sidecar_sha256",
        "scientific_config_sha256",
        "scientific_contract_sha256",
        "executor_config_sha256",
        "official_sidecar_manifest_sha256",
        "mmap_sidecar_manifest_sha256",
        "reader_parity_evidence_sha256",
        "parent_model_bundle_sha256",
        "query_sample_sha256",
        "prefit_parity_evidence_sha256",
    )
    lineage = {
        key: hashlib.sha256(f"lineage-{key}".encode()).hexdigest()
        for key in lineage_keys
    }
    lineage["parity_evidence_sha256"] = None
    recipe = hashlib.sha256(b"site-recipe").hexdigest()
    site_models: dict[str, dict[str, object]] = {}
    for outer in (2, 3, 4):
        path = root / f"site_{outer}.json"
        path.write_text(f"model-{outer}", encoding="utf-8")
        site_models[str(outer)] = {
            "path": path.name,
            "sha256": _sha256_file(path),
            "recipe_sha256": recipe,
        }
    train_matrices: dict[str, dict[str, object]] = {}
    for outer in (2, 3, 4):
        path = root / f"train_{outer}.parquet"
        path.write_bytes(f"train-{outer}".encode())
        train_matrices[str(outer)] = {
            "path": path.name,
            "sha256": _sha256_file(path),
            "rows": 1,
        }
    selected = root / "selected.parquet"
    selected.write_bytes(b"selected")
    manifest = {
        "schema_version": "sireto-retrieval-v12-fold-fused-oof-manifest-1",
        "engine_id": "FOLD_FUSED_CROSSFIT_COMPILER_V12_1",
        "run_kind": "NESTED_OUTER_OOF_FOLD_FUSED",
        "status": "EXECUTED_NO_READY_CLAIM",
        "opened_folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "logical_recipe_changes": False,
        "lineage": lineage,
        "train_matrices": train_matrices,
        "site_models": site_models,
        "outputs": {
            "selected": {
                "path": selected.name,
                "sha256": _sha256_file(selected),
                "rows": 1,
            }
        },
        "parity": {"status": "NOT_RUN"},
        "gates": {
            "parity_gate": "NOT_RUN",
            "decision": "NO_READY_DECISION",
            "gate_eligible": False,
            "master_recall_at_2000": {"rate": 0.99},
            "master_headroom_target_met": False,
            "prerank_recall_at_500": {"rate": 0.995},
            "exact_recall_at_100": {"rate": 0.993},
            "exact_recall_at_100_by_outer_fold": {},
            "thresholds": {"prerank_recall_at_500": 0.994},
            "recall_gates_pass": True,
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path, {key: value["sha256"] for key, value in site_models.items()}, recipe


def test_postfit_certificate_binds_exact_oof_and_three_models_without_refit(
    tmp_path,
) -> None:
    import pytest

    oof_path, model_hashes, recipe = _write_fake_prefit_only_oof(tmp_path / "oof")
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    parity_keys = (
        "scene_ir_sha256",
        "scene_ir_manifest_sha256",
        "source_candidate_sha256",
        "source_manifest_sha256",
        "label_ledger_sha256",
        "label_ledger_manifest_sha256",
        "scientific_config_sha256",
        "executor_config_sha256",
        "official_sidecar_manifest_sha256",
        "mmap_sidecar_manifest_sha256",
        "reader_parity_evidence_sha256",
        "parent_model_bundle_sha256",
    )
    parity_lineage = {key: oof["lineage"][key] for key in parity_keys}
    parity_lineage["query_sample_sha256"] = oof["lineage"]["query_sample_sha256"]
    parity_path = _write_parity_bundle(
        tmp_path / "postfit",
        lineage=parity_lineage,
        complete=True,
        site_model_manifest_sha=_sha256_file(oof_path),
        site_model_hashes=model_hashes,
        site_model_recipe_hashes={str(outer): recipe for outer in (2, 3, 4)},
    )
    certificate_path = certify_fold_fused_oof_postfit_v12(
        oof_manifest_path=oof_path,
        postfit_parity_manifest_path=parity_path,
        output_path=tmp_path / "certificate.json",
    )
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    assert certificate["decision"] == "PENDING_RUNTIME_PROFILE"
    assert certificate["gate_eligible"] is False
    assert certificate["refit_performed"] is False
    assert certificate["lineage"]["site_model_sha256"] == model_hashes

    wrong_hashes = dict(model_hashes)
    wrong_hashes["2"] = hashlib.sha256(b"different-model").hexdigest()
    wrong_parity = _write_parity_bundle(
        tmp_path / "wrong-postfit",
        lineage=parity_lineage,
        complete=True,
        site_model_manifest_sha=_sha256_file(oof_path),
        site_model_hashes=wrong_hashes,
        site_model_recipe_hashes={str(outer): recipe for outer in (2, 3, 4)},
    )
    with pytest.raises(Exception, match="binding mismatch"):
        certify_fold_fused_oof_postfit_v12(
            oof_manifest_path=oof_path,
            postfit_parity_manifest_path=wrong_parity,
            output_path=tmp_path / "wrong-certificate.json",
        )

    wrong_sample_lineage = dict(parity_lineage)
    wrong_sample_lineage["query_sample_sha256"] = hashlib.sha256(
        b"different-sample"
    ).hexdigest()
    wrong_sample_parity = _write_parity_bundle(
        tmp_path / "wrong-sample-postfit",
        lineage=wrong_sample_lineage,
        complete=True,
        site_model_manifest_sha=_sha256_file(oof_path),
        site_model_hashes=model_hashes,
        site_model_recipe_hashes={str(outer): recipe for outer in (2, 3, 4)},
    )
    with pytest.raises(Exception, match="lineage mismatch: query_sample_sha256"):
        certify_fold_fused_oof_postfit_v12(
            oof_manifest_path=oof_path,
            postfit_parity_manifest_path=wrong_sample_parity,
            output_path=tmp_path / "wrong-sample-certificate.json",
        )
