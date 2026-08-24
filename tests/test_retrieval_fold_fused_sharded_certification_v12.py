from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.xgb_matcher.retrieval_fold_fused_certification_v12 as certification
from src.xgb_matcher.retrieval_fold_fused_certification_v12 import (
    certify_fold_fused_oof_postfit_v12,
)
from src.xgb_matcher.retrieval_fold_fused_oof_v12 import (
    ExecutionCaps,
    FeatureBatch,
    sha256_file,
)
from src.xgb_matcher.retrieval_fold_fused_physical_benchmark_v12 import (
    current_national_worker_guard_v12,
)
from src.xgb_matcher.retrieval_fold_fused_sharded_oof_v12 import (
    ShardSpecV12,
    ShardedPhysicalInputsV12,
    build_shard_v12,
    load_fitted_site_model_records_v12,
    load_heldout_master_record_v12,
    plan_query_shards,
    replay_shard_v12,
    run_sharded_physical_sample,
)
from tests.test_retrieval_fold_fused_sharded_oof_v12 import (
    _Backend,
    _Contexts,
    _LazyBackend,
    _Ledger,
    _Sampler,
    _Selector,
    _Source,
    _labels,
    _scene_index,
)
from tests.test_retrieval_fold_fused_runtime_v12 import _write_parity_bundle


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _Contexts319(_Contexts):
    feature_names = tuple(f"feature_{index:03d}" for index in range(319))

    def materialize(self, context, statistics, *, candidate_indices):
        indices = (
            tuple(range(len(context.candidate_sirets)))
            if candidate_indices is None
            else tuple(candidate_indices)
        )
        values = np.zeros((len(indices), 319), dtype=np.float32)
        for position, index in enumerate(indices):
            truth = context.candidate_sirets[index] == self.truth[context.query_id]
            values[position, :3] = (10.0 if truth else 5.0 - index, truth, 0.2)
        raw = context.opaque.raw_rows
        return FeatureBatch(
            query_id=context.query_id,
            scene_fold=context.scene_fold,
            outer_fold=context.outer_fold,
            candidate_indices=indices,
            candidate_sirets=tuple(context.candidate_sirets[index] for index in indices),
            candidate_parents=tuple(
                context.candidate_parents[index] for index in indices
            ),
            feature_names=self.feature_names,
            values=values,
            statistics_sha256=statistics.statistics_sha256,
            row_payloads=tuple(raw[index] for index in indices),
        )


def _national_lineage() -> dict[str, str]:
    keys = (
        *certification._SCIENTIFIC_LINEAGE_KEYS,
        *certification._SHARDED_EXECUTION_LINEAGE_KEYS,
        *certification._NATIONAL_EXECUTION_LINEAGE_KEYS,
    )
    return {key: _sha(f"lineage-{key}") for key in keys}


def _runtime_guard(
    lineage: dict[str, str],
    *,
    maximum_workers: int,
    physical_authorized_workers: int | None = None,
    memory_used_bytes: int = 12 * 1024**3,
    swapouts: tuple[int, int] = (123, 123),
) -> dict[str, object]:
    observed_swapouts = iter(swapouts)
    timestamps = iter((1_000.0, 1_005.0))
    guard = dict(
        current_national_worker_guard_v12(
            physical_authorized_workers or maximum_workers,
            Path(__file__).resolve().parents[1]
            / "config/retrieval_all_human_v12_sharded_execution_plan.json",
            memory_reader=lambda: SimpleNamespace(used=memory_used_bytes),
            swap_reader=lambda: next(observed_swapouts),
            sleeper=lambda _seconds: None,
            clock=lambda: next(timestamps),
        )
    )
    assert guard["selected_maximum_concurrent_workers"] == maximum_workers
    lineage["runtime_resource_guard_sha256"] = str(guard["record_sha256"])
    return guard


def certification_build_worker(payload: dict[str, object]) -> str:
    spec = ShardSpecV12.from_manifest(dict(payload["spec"]))
    labels = _labels()
    truth = {query_id: value.human_siret for query_id, value in labels.items()}
    artifact = build_shard_v12(
        spec,
        source=_Source(spec, []),
        labels=_Ledger(labels, []),
        contexts=_Contexts319(truth, []),
        sampler=_Sampler(),
        admission_config=SimpleNamespace(channels=()),
        output_dir=Path(str(payload["output_dir"])),
        lineage=dict(payload["lineage"]),
        caps=ExecutionCaps(**dict(payload["caps"])),
        maximum_spool_bytes=int(payload["maximum_spool_bytes"]),
    )
    return str(artifact.manifest_path)


def certification_replay_worker(payload: dict[str, object]) -> str:
    spec = ShardSpecV12.from_manifest(dict(payload["spec"]))
    heldout = load_heldout_master_record_v12(dict(payload["heldout"]))
    models = load_fitted_site_model_records_v12(
        dict(payload["models"]),
        handle_loader=lambda model_path, _outer: json.loads(
            model_path.read_text(encoding="utf-8")
        ),
    )
    labels = _labels()
    truth = {query_id: value.human_siret for query_id, value in labels.items()}
    output = Path(str(payload["output_dir"]))
    artifact = replay_shard_v12(
        spec,
        heldout=heldout,
        labels=_Ledger(labels, []),
        contexts=_Contexts319(truth, []),
        model_backend=_Backend(output.parent / f"score-{spec.shard_id}"),
        models=models,
        selector=_Selector(),
        output_dir=output,
        lineage=dict(payload["lineage"]),
        build_manifest_sha256=str(payload["build_manifest_sha256"]),
        caps=ExecutionCaps(**dict(payload["caps"])),
    )
    return str(artifact.manifest_path)


def _write_sharded_oof(
    root: Path,
    *,
    maximum_workers: int = 2,
    physical_authorized_workers: int | None = None,
    memory_used_bytes: int = 12 * 1024**3,
    swapouts: tuple[int, int] = (123, 123),
) -> tuple[Path, dict[str, str], str]:
    index = _scene_index()
    query_ids = tuple(str(value) for value in index["query_id"].to_pylist())
    lineage = _national_lineage()
    guard = _runtime_guard(
        lineage,
        maximum_workers=maximum_workers,
        physical_authorized_workers=physical_authorized_workers,
        memory_used_bytes=memory_used_bytes,
        swapouts=swapouts,
    )
    evidence = {
        "phase": "NATIONAL_OOF_FOLDS_2_3_4",
        "physical_benchmark_manifest_sha256": lineage[
            "physical_benchmark_manifest_sha256"
        ],
        "execution_authorization_sha256": lineage[
            "execution_authorization_sha256"
        ],
        "runtime_resource_guard": guard,
        "fixed_shard_count": 3,
        "maximum_concurrent_workers": maximum_workers,
    }
    entrypoint = "tests.test_retrieval_fold_fused_sharded_certification_v12"
    artifact = run_sharded_physical_sample(
        ShardedPhysicalInputsV12(
            scene_index=index,
            query_ids=query_ids,
            lineage=lineage,
            build_entrypoint=f"{entrypoint}:certification_build_worker",
            build_payload={},
            replay_entrypoint=f"{entrypoint}:certification_replay_worker",
            replay_payload={},
            model_backend=_LazyBackend(root / "site_models"),
            caps=ExecutionCaps(fit_sample=4),
            maximum_spool_bytes=32 * 1024**2,
            maximum_workers=maximum_workers,
            execution_evidence=evidence,
        ),
        plan_query_shards(index, query_ids, 3),
        root,
        miniature_fit=False,
    )
    model_hashes = {
        str(outer): artifact.models[outer].sha256 for outer in (2, 3, 4)
    }
    recipe = artifact.models[2].recipe_sha256
    return artifact.manifest_path, model_hashes, recipe


def _use_core_fixture_shape(
    monkeypatch: pytest.MonkeyPatch, manifest: dict[str, object]
) -> None:
    specs = [ShardSpecV12.from_manifest(value) for value in manifest["specs"]]
    monkeypatch.setattr(certification, "_NATIONAL_QUERY_COUNT", 6)
    monkeypatch.setattr(
        certification,
        "_NATIONAL_SHARDS",
        tuple(
            {
                "shard_id": spec.shard_id,
                "fold": spec.folds[0],
                "query_start": spec.sequence_start,
                "query_stop": spec.sequence_stop,
                "candidate_start": spec.source_ranges[0][0],
                "candidate_stop": spec.source_ranges[0][1],
            }
            for spec in specs
        ),
    )


def _write_postfit_for_sharded(
    root: Path,
    oof_path: Path,
    model_hashes: dict[str, str],
    recipe: str,
) -> Path:
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    return _write_parity_bundle(
        root,
        lineage=dict(oof["lineage"]),
        complete=True,
        site_model_manifest_sha=sha256_file(oof_path),
        site_model_hashes=model_hashes,
        site_model_recipe_hashes={str(outer): recipe for outer in (2, 3, 4)},
    )


@pytest.mark.parametrize(
    (
        "maximum_workers",
        "physical_authorized_workers",
        "memory_used_bytes",
        "swapouts",
    ),
    (
        (2, 2, 12 * 1024**3, (123, 123)),
        (3, 3, 12 * 1024**3, (123, 123)),
        (2, 3, 15 * 1024**3, (123, 123)),
        (2, 3, 12 * 1024**3, (123, 124)),
    ),
)
def test_postfit_certifier_accepts_strict_sharded_oof_without_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    maximum_workers: int,
    physical_authorized_workers: int,
    memory_used_bytes: int,
    swapouts: tuple[int, int],
) -> None:
    oof_path, model_hashes, recipe = _write_sharded_oof(
        tmp_path / "oof",
        maximum_workers=maximum_workers,
        physical_authorized_workers=physical_authorized_workers,
        memory_used_bytes=memory_used_bytes,
        swapouts=swapouts,
    )
    manifest = json.loads(oof_path.read_text(encoding="utf-8"))
    _use_core_fixture_shape(monkeypatch, manifest)
    parity_path = _write_postfit_for_sharded(
        tmp_path / "postfit", oof_path, model_hashes, recipe
    )
    certificate_path = certify_fold_fused_oof_postfit_v12(
        oof_manifest_path=oof_path,
        postfit_parity_manifest_path=parity_path,
        output_path=tmp_path / "certificate.json",
    )
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    assert certificate["decision"] == "PIVOT_ADMISSION"
    assert certificate["gate_eligible"] is False
    assert certificate["refit_performed"] is False
    assert certificate["lineage"]["site_model_sha256"] == model_hashes


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda manifest: manifest.update({"miniature_fit": True}), "not certifiable"),
        (lambda manifest: manifest.update({"folds": [2, 3]}), "not certifiable"),
        (
            lambda manifest: manifest["gates"]["exact_recall_at_100"].update(
                {"successes": 0}
            ),
            "not recomputed",
        ),
        (
            lambda manifest: manifest["site_models"]["2"].update(
                {"size_bytes": 999}
            ),
            "model content/schema drift",
        ),
    ),
)
def test_postfit_certifier_rejects_unsafe_sharded_manifest_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    oof_path, model_hashes, recipe = _write_sharded_oof(tmp_path / "oof")
    original = json.loads(oof_path.read_text(encoding="utf-8"))
    _use_core_fixture_shape(monkeypatch, original)
    parity_path = _write_postfit_for_sharded(
        tmp_path / "postfit", oof_path, model_hashes, recipe
    )
    manifest = dict(original)
    mutation(manifest)
    oof_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match=message):
        certify_fold_fused_oof_postfit_v12(
            oof_manifest_path=oof_path,
            postfit_parity_manifest_path=parity_path,
            output_path=tmp_path / "certificate.json",
        )


def test_sharded_certifier_requires_full_national_authorization_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert certification._NATIONAL_QUERY_COUNT == 23_571
    assert [value["query_stop"] for value in certification._NATIONAL_SHARDS] == [
        7_767,
        15_734,
        23_571,
    ]
    oof_path, _model_hashes, _recipe = _write_sharded_oof(tmp_path / "oof")
    original = json.loads(oof_path.read_text(encoding="utf-8"))
    _use_core_fixture_shape(monkeypatch, original)

    mutations = (
        (
            lambda value: value.update({"maximum_concurrent_workers": 1}),
            "population/concurrency",
        ),
        (
            lambda value: value.pop("execution_evidence"),
            "execution evidence",
        ),
        (
            lambda value: value.update({"run_id": _sha("wrong-run")}),
            "run identity",
        ),
        (
            lambda value: value["execution_evidence"].update(
                {"execution_authorization_sha256": _sha("wrong-auth")}
            ),
            "authorization/guard lineage",
        ),
        (
            lambda value: value["worker_build_manifest_sha256"].__setitem__(
                0, _sha("wrong-build")
            ),
            "worker lineage",
        ),
        (
            lambda value: value["specs"][0]["source_ranges"][0].__setitem__(
                0, 1
            ),
            "national range",
        ),
    )
    for mutate, message in mutations:
        candidate = copy.deepcopy(original)
        mutate(candidate)
        oof_path.write_text(
            json.dumps(candidate, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(Exception, match=message):
            certification._validate_oof_manifest(oof_path)

    oof_path.write_text(
        json.dumps(original, sort_keys=True) + "\n", encoding="utf-8"
    )
    child = oof_path.parent / "worker_build" / "fold2" / "manifest.json"
    child_bytes = child.read_bytes()
    child_raw = json.loads(child_bytes)
    child_raw["fold0_opened"] = True
    child.write_text(json.dumps(child_raw, sort_keys=True), encoding="utf-8")
    try:
        with pytest.raises(Exception, match="unsafe"):
            certification._validate_oof_manifest(oof_path)
    finally:
        child.write_bytes(child_bytes)
