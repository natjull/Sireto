from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import xgb_matcher.retrieval_fold_fused_sharded_runtime_v12 as runtime
from xgb_matcher.retrieval_fold_fused_oof_v12 import FoldFusedInvariantError


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _lineage() -> dict[str, str]:
    return {
        "scene_ir_sha256": _digest("scene-build"),
        "scene_ir_manifest_sha256": _digest("scene-manifest"),
        "source_candidate_sha256": _digest("source"),
        "source_manifest_sha256": _digest("source-manifest"),
        "label_ledger_sha256": _digest("ledger"),
        "label_ledger_manifest_sha256": _digest("ledger-manifest"),
        "scientific_config_sha256": _digest("config"),
        "executor_config_sha256": _digest("executor"),
        "official_sidecar_manifest_sha256": _digest("sidecars"),
        "mmap_sidecar_manifest_sha256": _digest("mmap"),
        "reader_parity_evidence_sha256": _digest("reader"),
        "parent_model_bundle_sha256": _digest("parent"),
        "query_sample_sha256": _digest("sample"),
    }


def _authority_paths(tmp_path: Path) -> dict[str, Path]:
    values = {
        "workspace_root": tmp_path / "workspace",
        "scene_ir": tmp_path / "scene",
        "labels": tmp_path / "labels.parquet",
        "sidecars": tmp_path / "sidecars",
        "mmap_sidecar": tmp_path / "mmap",
        "reader_parity_manifest": tmp_path / "reader.json",
        "parent_model_bundle": tmp_path / "parent.json",
        "scientific_config": tmp_path / "scientific.json",
        "executor_config": tmp_path / "executor.json",
        "sharded_execution_plan": tmp_path / "plan.json",
        "prefit_parity_manifest": tmp_path / "prefit.json",
    }
    for name in ("workspace_root", "scene_ir", "sidecars", "mmap_sidecar"):
        values[name].mkdir()
    for name, path in values.items():
        if not path.exists():
            path.write_text(f"{name}\n", encoding="utf-8")
    return values


def test_common_payload_is_json_safe_and_binds_plan_and_lineage(
    tmp_path: Path,
) -> None:
    paths = _authority_paths(tmp_path)
    plan_sha = runtime.sha256_file(paths["sharded_execution_plan"])
    lineage = _lineage()
    payload = runtime.make_runtime_common_payload_v12(
        **paths,
        sharded_execution_plan_sha256=plan_sha,
        prefit_parity_manifest_sha256=runtime.sha256_file(
            paths["prefit_parity_manifest"]
        ),
        lineage=lineage,
    )
    assert json.loads(json.dumps(payload))["lineage"] == lineage
    assert payload["replay_generation_inputs_opened"] is False
    paths["sharded_execution_plan"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FoldFusedInvariantError, match="unsealed"):
        runtime.make_runtime_common_payload_v12(
            **paths,
            sharded_execution_plan_sha256=plan_sha,
            prefit_parity_manifest_sha256=runtime.sha256_file(
                paths["prefit_parity_manifest"]
            ),
            lineage=lineage,
        )


def test_replay_common_never_touches_generation_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _authority_paths(tmp_path)
    plan_sha = runtime.sha256_file(paths["sharded_execution_plan"])
    lineage = _lineage()
    common = runtime.make_runtime_common_payload_v12(
        **paths,
        sharded_execution_plan_sha256=plan_sha,
        prefit_parity_manifest_sha256=runtime.sha256_file(
            paths["prefit_parity_manifest"]
        ),
        lineage=lineage,
    )
    executor = SimpleNamespace(
        scientific_config_sha256=lineage["scientific_config_sha256"],
        sha256=lineage["executor_config_sha256"],
    )
    admission = SimpleNamespace(
        train_folds=(2, 3, 4),
        internal_cap=2_000,
        prerank_cap=500,
        final_cap=100,
    )
    cascade = SimpleNamespace(
        master_reference_cap=2_000,
        hydration_shortlist_cap=500,
    )
    ledger = SimpleNamespace(
        ledger_sha256=lineage["label_ledger_sha256"],
        manifest_sha256=lineage["label_ledger_manifest_sha256"],
        manifest={"runtime_query_sample_sha256": lineage["query_sample_sha256"]},
    )
    monkeypatch.setattr(runtime, "load_executor_config_v12", lambda *_a, **_k: executor)
    monkeypatch.setattr(runtime.AdmissionConfigV12, "load", lambda *_a, **_k: admission)
    monkeypatch.setattr(
        runtime.RetrievalTop100CascadeV12Config, "load", lambda *_a, **_k: cascade
    )
    monkeypatch.setattr(runtime.StrictV12LabelLedger, "load", lambda *_a, **_k: ledger)
    monkeypatch.setattr(runtime, "site_feature_names", lambda _config: tuple(f"f{i}" for i in range(319)))

    def forbidden_manifest(_path: Path) -> Path:
        raise AssertionError("replay touched a generation manifest")

    monkeypatch.setattr(runtime, "_manifest_file", forbidden_manifest)
    observed = runtime._require_common(
        common, lineage, open_generation_inputs=False
    )
    assert observed[1] is executor
    assert observed[4] is ledger


@pytest.mark.parametrize(
    ("entrypoint", "expected"),
    [
        (runtime.build_worker_entrypoint_v12, True),
        (runtime.replay_worker_entrypoint_v12, False),
    ],
)
def test_worker_entrypoints_select_the_correct_authority_surface(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint,
    expected: bool,
) -> None:
    observed: list[bool] = []

    class StopHere(RuntimeError):
        pass

    def stop(_common, _lineage, *, open_generation_inputs: bool):
        observed.append(open_generation_inputs)
        raise StopHere

    monkeypatch.setattr(runtime, "_require_common", stop)
    phase = "BUILD" if expected else "REPLAY"
    with pytest.raises(StopHere):
        entrypoint({"phase": phase, "common": {}, "lineage": {}})
    assert observed == [expected]


def test_lazy_site_backend_does_not_precreate_core_output(tmp_path: Path) -> None:
    output = tmp_path / "trial" / "site_models"
    runtime.LazyXGBoostSiteLambdaMARTBackendV12(
        params={"objective": "rank:ndcg"}, output_dir=output
    )
    assert not output.exists()
