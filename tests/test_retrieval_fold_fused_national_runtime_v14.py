from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import xgb_matcher.retrieval_fold_fused_national_runtime_v14 as national
from xgb_matcher.retrieval_fold_fused_oof_v12 import sha256_file


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest_keyset_fixture() -> dict[str, object]:
    raw: dict[str, object] = {
        key: None for key in national._NATIONAL_MANIFEST_KEYS_V14
    }
    raw["authorization"] = {
        "path": "auth",
        "sha256": _digest("auth"),
        "benchmark_manifest_path": "benchmark",
        "benchmark_manifest_sha256": _digest("benchmark"),
    }
    for name in ("source_candidate", "sharded_execution_plan", "prefit_parity"):
        raw[name] = {"path": name, "sha256": _digest(name)}
    raw["national_attempt"] = {
        "path": "marker",
        "sha256": _digest("marker"),
        "attempt_id": _digest("attempt"),
        "stage_directory_name": ".stage",
    }
    return raw


def test_v14_national_manifest_and_authority_keysets_are_exact() -> None:
    raw = _manifest_keyset_fixture()
    national._validate_national_manifest_keysets_v14(raw)
    with_root_extra = dict(raw, unregistered=True)
    with pytest.raises(national.PhysicalBenchmarkInvariantErrorV14):
        national._validate_national_manifest_keysets_v14(with_root_extra)
    for name in (
        "authorization",
        "source_candidate",
        "sharded_execution_plan",
        "prefit_parity",
        "national_attempt",
    ):
        drifted = dict(raw)
        drifted[name] = {**dict(raw[name]), "unregistered": True}  # type: ignore[arg-type]
        with pytest.raises(national.PhysicalBenchmarkInvariantErrorV14):
            national._validate_national_manifest_keysets_v14(drifted)


def test_v14_national_stage_uses_full_content_identity_and_rejects_alternate(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "national-v14"
    marker = tmp_path / ".national-v14-marker.json"
    prefix = ".national-v14.stage-"
    config = {
        "national_execution_protocol": {
            "canonical_output_dir": str(canonical),
            "attempt_consumed_marker_path": str(marker),
            "stage_directory_prefix": prefix,
        }
    }
    attempt_id = _digest("full-content-identity")
    observed_canonical, stage, observed_marker = national._execution_paths(
        config, requested=canonical, attempt_id=attempt_id
    )
    assert observed_canonical == canonical.resolve()
    assert observed_marker == marker.resolve()
    assert stage.name == prefix + attempt_id
    assert len(stage.name.removeprefix(prefix)) == 64
    with pytest.raises(national.PhysicalBenchmarkInvariantErrorV14):
        national._execution_paths(
            config, requested=tmp_path / "alternate", attempt_id=attempt_id
        )


def test_v14_national_marker_is_one_shot_and_strict(tmp_path: Path) -> None:
    marker = tmp_path / ".national-marker.json"
    payload = {
        "schema_version": "test-v14-attempt",
        "attempt_id": _digest("attempt"),
    }
    digest = national._consume_attempt(marker, payload)
    national._validate_attempt(
        marker, expected_sha256=digest, expected_payload=payload
    )
    with pytest.raises(national.PhysicalBenchmarkInvariantErrorV14):
        national._consume_attempt(marker, payload)


def test_v14_national_post_o_excl_failure_keeps_marker_and_durable_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / ".national-marker.json"
    stage = tmp_path / ".national.stage-" / _digest("attempt")
    canonical = tmp_path / "national"
    payload = {"attempt_id": _digest("attempt")}
    real_fsync = national._fsync_directory
    monkeypatch.setattr(
        national,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("fsync fault")),
    )
    with pytest.raises(national._PostOExclNationalMarkerErrorV14) as failure:
        national._consume_attempt(marker, payload)
    assert marker.exists()
    monkeypatch.setattr(national, "_fsync_directory", real_fsync)
    manifest = national._publish_exception_stop(
        stage=stage,
        canonical=canonical,
        marker=marker,
        marker_sha256=national._observed_marker_sha256(marker),
        attempt_id=_digest("attempt"),
        v14_implementation_sha256=_digest("v14"),
        implementation_files=(),
        authorization=tmp_path / "authorization.json",
        authorization_sha256=_digest("authorization"),
        benchmark=tmp_path / "benchmark.json",
        benchmark_sha256=_digest("benchmark"),
        phase="ATOMIC_ATTEMPT_MARKER_CREATION",
        error=failure.value,
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    assert raw["status"] == "STOP_V14_NATIONAL_EXECUTION_EXCEPTION"
    assert raw["automatic_retry_allowed"] is False
    assert raw["fold0_opened"] is False and raw["fold1_opened"] is False
    assert marker.exists() and not canonical.exists()


def test_v14_national_failed_strict_reload_rolls_promotion_back_to_stop_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canonical = tmp_path / "national-v14"
    marker = tmp_path / ".national-v14-marker.json"
    prefix = ".national-v14.stage-"
    config = {
        "national_execution_protocol": {
            "canonical_output_dir": str(canonical),
            "attempt_consumed_marker_path": str(marker),
            "stage_directory_prefix": prefix,
        },
        "input_lineage": {"source_candidate_sha256": _digest("source")},
    }
    benchmark = tmp_path / "benchmark.json"
    authorization = tmp_path / "authorization.json"
    benchmark.write_text("{}\n", encoding="utf-8")
    authorization.write_text("{}\n", encoding="utf-8")
    prepared = SimpleNamespace(
        sharded_execution_plan_sha256=_digest("plan"),
        prefit_parity_manifest_sha256=_digest("prefit"),
        national_query_ids=tuple(f"query-{index}" for index in range(23571)),
    )
    monkeypatch.setattr(national, "load_physical_config_v14", lambda _p: config)
    monkeypatch.setattr(
        national,
        "load_execution_authorization_v14",
        lambda *a, **k: {
            "authorization_identity": {
                "authorized_scope": "NATIONAL_OOF_FOLDS_2_3_4_V14_ONLY"
            }
        },
    )
    monkeypatch.setattr(
        national,
        "v14_implementation_lineage",
        lambda _root: (_digest("v14"), ()),
    )
    monkeypatch.setattr(
        national,
        "v13_implementation_lineage",
        lambda _root: (_digest("v13"), ()),
    )
    monkeypatch.setattr(
        national, "_validated_authorized_base_lineage", lambda **k: {}
    )

    def execute_stage(**kwargs: object) -> national.NationalFragmentOOFV14:
        stage = Path(kwargs["stage"])  # type: ignore[arg-type]
        manifest = stage / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n", encoding="utf-8")
        return national.NationalFragmentOOFV14(
            manifest_path=manifest,
            manifest_sha256=sha256_file(manifest),
            run_id=_digest("run"),
            query_count=23571,
            lineage={},
            gates={},
        )

    monkeypatch.setattr(national, "_execute_stage", execute_stage)
    monkeypatch.setattr(
        national,
        "load_national_fragment_oof_v14",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("strict reopen fault")),
    )
    with pytest.raises(RuntimeError, match="strict reopen fault"):
        national.execute_national_fragment_oof_v14(
            prepared=prepared,
            workspace_root=tmp_path,
            output_dir=canonical,
            physical_benchmark_manifest=benchmark,
            physical_benchmark_manifest_sha256=_digest("benchmark"),
            execution_authorization=authorization,
            execution_authorization_sha256=_digest("authorization"),
        )
    stages = list(tmp_path.glob(prefix + "*"))
    assert len(stages) == 1
    assert not canonical.exists()
    stop = json.loads((stages[0] / "manifest.json").read_text(encoding="utf-8"))
    assert stop["status"] == "STOP_V14_NATIONAL_EXECUTION_EXCEPTION"
    assert marker.exists()
