from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.train_v41_models as training
from src.xgb_matcher.v41_acceptor import (
    V41_CONFIDENCE_KIND,
    V41RawLogisticAcceptor,
)
from src.xgb_matcher.v41_features import V41_CANDIDATE_FEATURE_NAMES
from src.xgb_matcher.v41_release import V41ReleaseManifest
from src.xgb_matcher.v9_dataset import file_sha256


class _PickleableProbabilityModel:
    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        positive = np.full(len(matrix), 0.999)
        return np.column_stack([1.0 - positive, positive])


def _write_dataset(
    root: Path,
    *,
    source_split: str | None = None,
    positive_injection: bool = False,
    injected_source: bool = False,
) -> Path:
    root.mkdir()
    query_rows = []
    label_rows = []
    candidate_rows = []
    # Exact rows have a trivially learnable two-candidate ordering.
    for index in range(80):
        siren = f"{100_000_000 + index:09d}"
        truth = f"{siren}00001"
        query = {
            "query_id": f"exact-{index}",
            "input_siret": truth,
            "input_siren": siren,
            "input_siret_state": "A",
        }
        label = {
            "query_id": query["query_id"],
            "label_kind": "MATCH_EXACT",
            "ground_truth_siret": truth,
            "ground_truth_siren": siren,
        }
        if source_split is not None:
            query["split"] = source_split
            label["split"] = source_split
        query_rows.append(query)
        label_rows.append(label)
        for is_truth in (1, 0):
            candidate_siren = siren if is_truth else f"{800_000_000 + index:09d}"
            candidate_rows.append(
                {
                    "query_id": query["query_id"],
                    "candidate_siret": f"{candidate_siren}00001",
                    "candidate_siren": candidate_siren,
                    "is_ground_truth": is_truth,
                    "candidate_state": "A",
                    "retrieval_rank": 1 if is_truth else 2,
                    "retrieval_source": (
                        "injected_training_positive"
                        if injected_source and is_truth and index == 0
                        else "sparse"
                    ),
                    "retrieval_channel_count": 1,
                    "retrieval_agreement": 1,
                    "legacy_signal": float(is_truth),
                    **{feature: 0.0 for feature in V41_CANDIDATE_FEATURE_NAMES},
                }
            )

    # Ambiguous retrieval misses must survive as explicit incorrect scenes.
    for index in range(20):
        siren = f"{300_000_000 + index:09d}"
        query = {
            "query_id": f"ambiguous-{index}",
            "input_siret": f"{siren}00001",
            "input_siren": siren,
            "input_siret_state": "UNKNOWN",
        }
        label = {
            "query_id": query["query_id"],
            "label_kind": "AMBIGUOUS",
            "ground_truth_siret": None,
            "ground_truth_siren": None,
        }
        if source_split is not None:
            query["split"] = source_split
            label["split"] = source_split
        query_rows.append(query)
        label_rows.append(label)

    frames = {
        "queries.parquet": pd.DataFrame(query_rows),
        "labels.parquet": pd.DataFrame(label_rows),
        "candidates.parquet": pd.DataFrame(candidate_rows),
    }
    outputs = {}
    for filename, frame in frames.items():
        path = root / filename
        frame.to_parquet(path, index=False)
        outputs[filename] = file_sha256(path)
    manifest = {
        "schema_version": "v4.1-canonical-test",
        "build_id": "canonical-build-1",
        "retrieval_signature": "retrieval-active-A",
        "feature_order": ["legacy_signal", *V41_CANDIDATE_FEATURE_NAMES],
        "critical_segment_columns": ["input_siret_state"],
        "positive_injection": positive_injection,
        "outputs": outputs,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_loader_refuses_final_boundaries_and_positive_injection(tmp_path: Path) -> None:
    final_dataset = _write_dataset(tmp_path / "final", source_split="test")
    with pytest.raises(ValueError, match="forbidden final boundary"):
        training.load_v41_canonical_dataset(final_dataset)

    injected_manifest = _write_dataset(
        tmp_path / "manifest-injected",
        positive_injection=True,
    )
    with pytest.raises(ValueError, match="positive_injection=false"):
        training.load_v41_canonical_dataset(injected_manifest)

    injected_row = _write_dataset(
        tmp_path / "row-injected",
        injected_source=True,
    )
    with pytest.raises(ValueError, match="injection marker"):
        training.load_v41_canonical_dataset(injected_row)


def test_training_emits_component_safe_fit_dev_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _write_dataset(tmp_path / "dataset")
    calls: dict[str, object] = {}

    def fake_acceptor(
        scenes: pd.DataFrame,
        *,
        feature_order,
        dataset_manifest_id,
        retrieval_signature,
        target_precision,
        min_auto_count,
        seed,
    ):
        calls.update(
            {
                "scene_count": len(scenes),
                "miss_count": int(scenes["retrieval_miss"].sum()),
                "all_out_of_sample": bool(
                    scenes["ranker_prediction_is_out_of_sample"].all()
                ),
                "target_precision": target_precision,
                "min_auto_count": min_auto_count,
                "seed": seed,
            }
        )
        bundle = V41RawLogisticAcceptor(
            model=_PickleableProbabilityModel(),
            threshold=0.95,
            feature_order=list(feature_order),
            model_bundle_id="acceptor-synthetic",
            dataset_manifest_id=dataset_manifest_id,
            retrieval_signature=retrieval_signature,
        )
        return bundle, {
            "schema_version": "synthetic",
            "calibration_method": "raw",
            "final_test_evaluated": False,
        }

    monkeypatch.setattr(training, "fit_v41_raw_logistic_acceptor", fake_acceptor)
    output = tmp_path / "output"
    report = training.train_v41_models(
        dataset_dir=dataset,
        output_dir=output,
        ranker_params={
            "n_estimators": 3,
            "max_depth": 2,
            "learning_rate": 0.3,
            "n_jobs": 1,
        },
    )

    assert calls == {
        "scene_count": 100,
        "miss_count": 20,
        "all_out_of_sample": True,
        "target_precision": 0.998,
        "min_auto_count": 100,
        "seed": 42,
    }
    assert report["test_or_holdout_consumed"] is False
    assert report["positive_injection"] is False
    assert report["ranker_comparison"]["r1_hit_at_1"] >= 0.96
    assert report["ranker_comparison"]["selected_variant"] == "R1"

    assignments = pd.read_parquet(output / "split_assignments.parquet")
    assert set(assignments["split"]) == {"fit", "dev"}
    assert assignments.groupby("siren_component_id")["split"].nunique().max() == 1
    assert assignments.groupby("siren_component_id")["oof_fold"].nunique().max() == 1
    predictions = pd.read_parquet(output / "ranker_predictions.parquet")
    scenes = pd.read_parquet(output / "acceptor_scenes.parquet")
    assert set(predictions["prediction_origin"].dropna()) == {
        "oof",
        "out_of_sample_dev",
    }
    assert int(scenes["retrieval_miss"].sum()) == 20

    release = V41ReleaseManifest.load(output / "release_manifest.json")
    assert release.ranker_dataset_manifest_id == "canonical-build-1"
    assert release.acceptor_dataset_manifest_id != release.ranker_dataset_manifest_id
    acceptor_metadata = json.loads(
        (output / "acceptor" / "metadata.json").read_text()
    )
    assert acceptor_metadata["calibration_method"] == "raw"
    assert acceptor_metadata["confidence_kind"] == V41_CONFIDENCE_KIND
    assert (output / "ranker" / "ranker.json").exists()
    assert (output / "ranker" / "metadata.json").exists()
