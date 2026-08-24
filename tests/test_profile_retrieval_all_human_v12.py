from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from scripts import profile_retrieval_all_human_v12 as profiler
from scripts import build_retrieval_all_human_v12_runtime_queries as query_builder
from scripts import run_retrieval_hierarchical_admission_v12 as admission_cli


def _sample_512() -> pd.DataFrame:
    values = [f"query-{index:04d}" for index in range(640)]
    ordered = sorted(
        values,
        key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value),
    )[:512]
    # Ensure all three outer folds are represented without affecting the
    # query-id-only deterministic selection/order rule.
    return pd.DataFrame(
        {
            "query_id": ordered,
            "fold": [2 + (index % 3) for index in range(512)],
        }
    )


def test_selection_bytes_are_rank_canonical_and_order_sensitive() -> None:
    first = pd.DataFrame(
        {
            "candidate_siret": ["00000000000002", "00000000000001"],
            "selected_rank": [2, 1],
        }
    )
    same_different_row_order = first.iloc[::-1].reset_index(drop=True)
    changed_candidate_order = pd.DataFrame(
        {
            "candidate_siret": ["00000000000002", "00000000000001"],
            "selected_rank": [1, 2],
        }
    )
    assert profiler.selection_bytes(first) == profiler.selection_bytes(
        same_different_row_order
    )
    assert profiler.selection_sha256(first) != profiler.selection_sha256(
        changed_candidate_order
    )


def test_query_sample_contract_and_truth_source_fail_closed(tmp_path) -> None:
    sample = _sample_512()
    sample_path = tmp_path / "sample.parquet"
    sample.to_parquet(sample_path, index=False)
    loaded = profiler._validate_query_sample(sample_path)
    assert loaded.equals(sample)

    query_source = sample.assign(
        crm_name="fixture",
        crm_address="1 RUE TEST",
        human_ground_truth_siret="00000000000001",
    )
    source_path = tmp_path / "queries.parquet"
    query_source.to_parquet(source_path, index=False)
    with pytest.raises(ValueError, match="human-label columns"):
        profiler._load_label_blind_queries(source_path, loaded)

    opened_fold = sample.copy()
    opened_fold.loc[0, "fold"] = 1
    opened_path = tmp_path / "opened-fold.parquet"
    opened_fold.to_parquet(opened_path, index=False)
    with pytest.raises(ValueError, match="folds 2/3/4"):
        profiler._validate_query_sample(opened_path)


def test_stub_replay_measures_every_query_and_preserves_parity() -> None:
    sample = _sample_512().head(6).copy()
    queries = {
        row.query_id: {"query_id": row.query_id, "fold": int(row.fold)}
        for row in sample.itertuples(index=False)
    }
    calls: list[tuple[str, int]] = []

    def replay(row: dict[str, object], fold: int) -> pd.DataFrame:
        calls.append((str(row["query_id"]), fold))
        return pd.DataFrame(
            {
                "query_id": [row["query_id"]],
                "candidate_siret": [f"{fold:014d}"],
                "selected_rank": [1],
            }
        )

    warm, warm_latency = profiler._run_replay_pass(
        sample, queries, replay, measure=False
    )
    measured, latencies = profiler._run_replay_pass(
        sample, queries, replay, measure=True
    )
    assert len(calls) == 12
    assert warm_latency == []
    assert len(latencies) == len(sample)
    assert {
        query_id: profiler.selection_sha256(value)
        for query_id, value in warm.items()
    } == {
        query_id: profiler.selection_sha256(value)
        for query_id, value in measured.items()
    }


def test_parity_aggregate_digest_matches_admission_validator() -> None:
    sample = _sample_512()
    rows = sample.assign(
        reference_sha256=[hashlib.sha256(value.encode()).hexdigest() for value in sample["query_id"]],
        replay_sha256=[hashlib.sha256(value.encode()).hexdigest() for value in sample["query_id"]],
        byte_identical=True,
    )
    assert profiler.parity_aggregate_digest(
        rows, "reference_sha256"
    ) == admission_cli._parity_aggregate_digest(rows, "reference_sha256")


def test_non_m4_fixture_can_never_claim_gate_hardware() -> None:
    fixture = {
        "computer": "Linux",
        "chip": "fixture",
        "os": "fixture",
        "python": "fixture",
        "xgboost": "fixture",
    }
    assert profiler._is_target_hardware(fixture) is False
    assert profiler._is_target_hardware(
        {**fixture, "computer": "Mac", "chip": "Apple M4 Pro"}
    ) is True


def test_hardware_identity_recognizes_m4_pro_from_system_profiler(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return profiler.subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "Hardware:\n"
                "    Hardware Overview:\n"
                "      Model Identifier: Mac16,7\n"
                "      Chip: Apple M4 Pro\n"
                "      Serial Number (system): MUST_NOT_LEAK\n"
                "      Hardware UUID: MUST_NOT_LEAK\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(profiler.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(profiler.platform, "platform", lambda: "macOS-test")
    monkeypatch.setattr(
        profiler.shutil, "which", lambda name: "/usr/sbin/system_profiler"
    )
    monkeypatch.setattr(profiler.subprocess, "run", fake_run)

    identity = profiler._hardware_identity()
    assert identity["chip"] == "Apple M4 Pro"
    assert profiler._is_target_hardware(identity) is True
    assert len(calls) == 1
    assert calls[0][0] == "/usr/sbin/system_profiler"
    assert "MUST_NOT_LEAK" not in json.dumps(identity)


def test_hardware_identity_falls_back_to_safe_sysctl_brand(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        if "system_profiler" in command[0]:
            raise FileNotFoundError(command[0])
        return profiler.subprocess.CompletedProcess(
            command, 0, stdout="Apple M4 Pro\n", stderr=""
        )

    monkeypatch.setattr(profiler.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(profiler.platform, "platform", lambda: "macOS-test")
    monkeypatch.setattr(profiler.shutil, "which", lambda _name: None)
    monkeypatch.setattr(profiler.platform, "processor", lambda: "arm")
    monkeypatch.setattr(profiler.subprocess, "run", fake_run)

    identity = profiler._hardware_identity()
    assert identity["chip"] == "Apple M4 Pro"
    assert profiler._is_target_hardware(identity) is True
    assert calls == [
        ["/usr/sbin/system_profiler", "SPHardwareDataType", "-detailLevel", "mini"],
        ["sysctl", "-n", "machdep.cpu.brand_string"],
    ]


def test_system_profiler_chip_accepts_localized_label_only(monkeypatch) -> None:
    monkeypatch.setattr(
        profiler.shutil, "which", lambda _name: "/usr/sbin/system_profiler"
    )
    monkeypatch.setattr(
        profiler.subprocess,
        "run",
        lambda command, **_kwargs: profiler.subprocess.CompletedProcess(
            command,
            0,
            stdout="  Puce : Apple M4 Pro\n  UUID du matériel : MUST_NOT_LEAK\n",
            stderr="",
        ),
    )
    assert profiler._system_profiler_chip() == "Apple M4 Pro"


def test_query_projection_builder_never_materializes_truth_or_folds_0_1(
    tmp_path,
) -> None:
    source = tmp_path / "truth-bearing.parquet"
    pd.DataFrame(
        {
            "query_id": [f"q{fold}" for fold in (0, 1, 2, 3, 4)],
            "oof_fold": [0, 1, 2, 3, 4],
            "crm_name": [f"name {fold}" for fold in (0, 1, 2, 3, 4)],
            "crm_address": [f"{fold} rue test" for fold in (0, 1, 2, 3, 4)],
            "human_ground_truth_siret": [f"{fold:014d}" for fold in (0, 1, 2, 3, 4)],
            "acceptable_sirets_operational_json": ["[]"] * 5,
        }
    ).to_parquet(source, index=False)
    v11_manifest_path = tmp_path / "v11.manifest.json"
    v11_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": query_builder.V11_SOURCE_MANIFEST_SCHEMA,
                "candidate_schema_version": query_builder.V11_CANDIDATE_SCHEMA,
                "run_kind": "LABEL_BLIND_RETRIEVAL_COMBINED",
                "folds": [2, 3, 4],
                "positive_injection": False,
                "human_labels_opened_by_retrieval": False,
                "source_input_sha256": query_builder.sha256_file(source),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "projection"
    query_builder.build(
        Namespace(
            source=source,
            v11_candidate_manifest=v11_manifest_path,
            output_dir=output,
        )
    )
    projected = pd.read_parquet(output / "queries.parquet")
    assert set(projected["oof_fold"]) == {2, 3, 4}
    assert "human_ground_truth_siret" not in projected
    assert "acceptable_sirets_operational_json" not in projected
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_input_sha256"] == query_builder.sha256_file(source)
    assert manifest["v11_candidate_manifest_sha256"] == query_builder.sha256_file(
        v11_manifest_path
    )
    assert manifest["projection_code_sha256"] == query_builder.sha256_file(
        Path(query_builder.__file__)
    )
    profiler._validate_query_source_manifest(
        output / "manifest.json",
        query_path=output / "queries.parquet",
        query_columns=tuple(projected.columns),
        query_rows=len(projected),
        v11_candidate_manifest_path=v11_manifest_path,
        v11_candidate_manifest=json.loads(v11_manifest_path.read_text()),
    )


def test_query_projection_rejects_source_hash_substitution(tmp_path) -> None:
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        {
            "query_id": ["q2", "q3", "q4"],
            "fold": [2, 3, 4],
            "crm_name": ["a", "b", "c"],
        }
    ).to_parquet(source, index=False)
    manifest = tmp_path / "v11.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": query_builder.V11_SOURCE_MANIFEST_SCHEMA,
                "candidate_schema_version": query_builder.V11_CANDIDATE_SCHEMA,
                "run_kind": "LABEL_BLIND_RETRIEVAL",
                "folds": [2, 3, 4],
                "positive_injection": False,
                "human_labels_opened_by_retrieval": False,
                "source_input_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact folds-2/3/4 input"):
        query_builder.build(
            Namespace(
                source=source,
                v11_candidate_manifest=manifest,
                output_dir=tmp_path / "out",
            )
        )


def test_minimal_self_declared_label_ledger_sample_manifest_is_rejected(
    tmp_path,
) -> None:
    directory = tmp_path / "forged"
    directory.mkdir()
    sample = _sample_512()
    sample_path = directory / "runtime_query_sample.parquet"
    sample.to_parquet(sample_path, index=False)
    sample_sha = profiler.sha256_file(sample_path)
    manifest = {
        "schema_version": profiler.LABEL_LEDGER_SCHEMA,
        "build_id": "forged",
        "population": "ALL_USABLE_HUMAN_LABELS",
        "folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "runtime_query_sample_rule": profiler.SAMPLE_SELECTION_RULE,
        "runtime_query_sample_rows": 512,
        "runtime_query_sample_path": sample_path.name,
        "runtime_query_sample_sha256": sample_sha,
        "artifacts": {
            "label_ledger": {"path": "label_ledger.parquet"},
            "runtime_query_sample": {
                "path": sample_path.name,
                "rows": 512,
                "sha256": sample_sha,
            }
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="build_identity"):
        profiler._validate_label_ledger_sample_manifest(
            manifest_path,
            sample_path=sample_path,
            config_path=profiler.DEFAULT_CONFIG,
        )


def test_profile_rejects_another_label_ledger_before_pipeline_replay(
    tmp_path, monkeypatch
) -> None:
    manifest_path = tmp_path / "ledger-manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    sealed = profiler.SealedOOFInputs(
        oof_manifest_path=tmp_path / "oof-manifest.json",
        oof_manifest={
            "source_labels_sha256": "a" * 64,
            "source_labels_manifest_sha256": profiler.sha256_file(manifest_path),
        },
        cascade_manifests={},
        selected_paths={},
        site_model_paths={},
        lineage={},
    )
    monkeypatch.setattr(profiler, "_validate_oof_inputs", lambda _args: sealed)
    monkeypatch.setattr(
        profiler,
        "_validate_label_ledger_sample_manifest",
        lambda *_args, **_kwargs: {"label_ledger_sha256": "b" * 64},
    )
    sidecars = tmp_path / "sidecars"
    sidecars.mkdir()
    calls: list[str] = []

    def pipeline_factory(_args, _sealed):
        calls.append("opened")
        raise AssertionError("pipeline must not be constructed")

    args = Namespace(
        output_dir=tmp_path / "runtime-output",
        search_workers=1,
        allow_non_m4_test_fixture=False,
        config=profiler.DEFAULT_CONFIG,
        v11_config=profiler.DEFAULT_CONFIG,
        official_config=profiler.DEFAULT_CONFIG,
        sidecars=sidecars,
        label_ledger_manifest=manifest_path,
        query_sample=tmp_path / "not-opened-sample.parquet",
    )
    with pytest.raises(ValueError, match="another label-ledger snapshot"):
        profiler.profile(
            args,
            pipeline_factory=pipeline_factory,
            hardware_provider=lambda: {
                "computer": "Mac",
                "chip": "Apple M4 Pro",
                "os": "fixture",
                "python": "fixture",
                "xgboost": "fixture",
            },
        )
    assert calls == []
