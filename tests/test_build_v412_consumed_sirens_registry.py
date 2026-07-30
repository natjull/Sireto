from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_v412_consumed_sirens_registry.py"
SPEC = importlib.util.spec_from_file_location(
    "build_v412_consumed_sirens_registry", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)
RegistryStop = subject.RegistryStop
build_registry = subject.build_registry
canonical_bytes = subject.canonical_bytes
file_sha256 = subject.file_sha256
validate_inputs = subject.validate_inputs
validate_plan = subject.validate_plan


def _git_commit(root: Path = ROOT) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _audited_builder(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "builder-repository"
    script = repository / "scripts" / SCRIPT.name
    tests = repository / "tests" / Path(__file__).name
    script.parent.mkdir(parents=True)
    tests.parent.mkdir(parents=True)
    script.write_bytes(SCRIPT.read_bytes())
    tests.write_bytes(Path(__file__).read_bytes())
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@sireto.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "SIRETO Tests"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "add", "scripts", "tests"], cwd=repository, check=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "audit builder"],
        cwd=repository,
        check=True,
    )
    return script, _git_commit(repository)


def _schema_spec(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {"name": field.name, "nullable": field.nullable, "type": str(field.type)}
        for field in schema
    ]


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_bytes(value))


def _fixture(
    tmp_path: Path,
    *,
    candidate_mapping: bool = False,
    forbidden_siren_field: str | None = None,
    mismatch: bool = False,
    duplicate_observation: bool = False,
) -> tuple[Path, Path, Path, Path]:
    builder_script, audited_commit = _audited_builder(tmp_path)
    contract = tmp_path / "contract.md"
    contract.write_text("fixture contract\n", encoding="utf-8")
    source_manifest = tmp_path / "source-manifest.json"
    event_manifest = tmp_path / "event-manifest.json"
    _write_canonical(source_manifest, {"kind": "source"})
    _write_canonical(event_manifest, {"kind": "event"})

    schema = pa.schema(
        [
            pa.field("query_id", pa.string(), False),
            pa.field("label_kind", pa.string(), False),
            pa.field("ground_truth_siret", pa.string(), True),
            pa.field("ground_truth_siren", pa.string(), True),
        ],
        metadata=None,
    )
    second_siren = "999999999" if mismatch else "987654321"
    rows = [
        {
            "query_id": "q1",
            "label_kind": "MATCH_EXACT",
            "ground_truth_siret": "12345678900012",
            "ground_truth_siren": "123456789",
        },
        {
            "query_id": "q2",
            "label_kind": "MATCH_EXACT",
            "ground_truth_siret": "98765432100019",
            "ground_truth_siren": second_siren,
        },
        {
            "query_id": "q3",
            "label_kind": "UNRESOLVED",
            "ground_truth_siret": None,
            "ground_truth_siren": None,
        },
    ]
    if duplicate_observation:
        rows.append(dict(rows[0]))
    source_path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), source_path)
    mapping = {
        "derivation": "DIRECT_OR_SIRET_PREFIX_WITH_CONSISTENCY",
        "identity_role": "GROUND_TRUTH_CURRENT",
        "label_kind_field": "label_kind",
        "siren_field": (
            forbidden_siren_field
            or ("candidate_siren" if candidate_mapping else "ground_truth_siren")
        ),
        "siret_field": "ground_truth_siret",
        "source_field": (
            "candidate_siren|ground_truth_siret"
            if candidate_mapping
            else "ground_truth_siren|ground_truth_siret"
        ),
    }
    output_schemas = {
        "observations": [
            ("siren", pa.string(), False),
            ("identity_role", pa.string(), False),
            ("consumption_scope", pa.string(), False),
            ("source_id", pa.string(), False),
            ("source_path", pa.string(), False),
            ("source_sha256", pa.string(), False),
            ("source_manifest_sha256", pa.string(), False),
            ("source_record_locator", pa.string(), False),
            ("source_field", pa.string(), False),
            ("label_kind", pa.string(), True),
            ("derivation", pa.string(), False),
            ("observation_key_sha256", pa.string(), False),
        ],
        "consumed_sirens": [
            ("siren", pa.string(), False),
            ("provenance_count", pa.uint32(), False),
            ("identity_roles_json", pa.string(), False),
            ("consumption_scopes_json", pa.string(), False),
            ("source_ids_json", pa.string(), False),
            ("provenance_payload_sha256", pa.string(), False),
        ],
        "rejected_values": [
            ("source_id", pa.string(), False),
            ("source_record_locator", pa.string(), False),
            ("source_field", pa.string(), False),
            ("identity_role", pa.string(), False),
            ("rejection_reason", pa.string(), False),
            ("raw_value_sha256", pa.string(), True),
        ],
    }

    def output_spec(name: str, sort: list[str]) -> dict[str, object]:
        fields = [
            pa.field(field_name, field_type, nullable)
            for field_name, field_type, nullable in output_schemas[name]
        ]
        return {
            "filename": f"{name}.parquet",
            "schema": _schema_spec(pa.schema(fields)),
            "sort": sort,
        }

    plan = {
        "build": {
            "build_id_length": 16,
            "builder": {
                "git_commit": audited_commit,
                "source_path": "scripts/build_v412_consumed_sirens_registry.py",
                "source_sha256": file_sha256(builder_script),
                "status": "PINNED",
                "tests_path": "tests/test_build_v412_consumed_sirens_registry.py",
                "tests_sha256": file_sha256(
                    builder_script.parent.parent
                    / "tests"
                    / "test_build_v412_consumed_sirens_registry.py"
                ),
            },
        },
        "canonical_json": "UTF8_SORT_KEYS_COMPACT_ALLOW_NAN_FALSE_SINGLE_LF",
        "contract": {"path": str(contract), "sha256": file_sha256(contract)},
        "durability": {
            "permissions": {
                "directories": "0700",
                "files": "0600",
                "umask": "0077",
            }
        },
        "event_only_manifests": [
            {
                "event_role": "FIXTURE_EVENT",
                "path": str(event_manifest),
                "projection": "FULL_FILE_BYTES",
                "sha256": file_sha256(event_manifest),
                "size_bytes": event_manifest.stat().st_size,
            }
        ],
        "identity_sources": [
            {
                "consumption_scopes": ["FIXTURE_SCOPE"],
                "id": "FIXTURE_TRUTH",
                "identity_mappings": [mapping],
                "manifest_path": str(source_manifest),
                "manifest_sha256": file_sha256(source_manifest),
                "path": str(source_path),
                "projection": [field.name for field in schema],
                "projection_schema": _schema_spec(schema),
                "record_locator": "query_id",
                "required_filter": "NON_NULL_IDENTITY_ONLY_MATCH_EXACT_REQUIRES_VALID_IDENTITY",
            "row_count": len(rows),
                "sha256": file_sha256(source_path),
                "size_bytes": source_path.stat().st_size,
            }
        ],
        "invariants": {
            "candidate_or_prediction_identity_count": 0,
            "expected_event_manifest_count": 1,
            "expected_identity_source_count": 1,
        },
        "normalization": {"version": "fixture-normalization"},
        "outputs": {
            "consumed_sirens": output_spec("consumed_sirens", ["siren ASC"]),
            "exact_files": [
                "sources.json",
                "observations.parquet",
                "consumed_sirens.parquet",
                "rejected_values.parquet",
                "manifest.json",
            ],
            "manifest": {
                "required_fields": [
                    "schema_version",
                    "build_id",
                    "builder_git_commit",
                    "builder_source_sha256",
                    "builder_tests_sha256",
                    "contract_sha256",
                    "plan_sha256",
                    "input_source_hashes",
                    "event_manifest_hashes",
                    "source_row_counts",
                    "identity_role_counts",
                    "rejection_counts",
                    "observation_count",
                    "unique_siren_count",
                    "observations_logical_sha256",
                    "sirens_logical_sha256",
                    "files",
                    "tree_payload_sha256",
                ]
            },
            "observations": output_spec(
                "observations", ["siren ASC", "observation_key_sha256 ASC"]
            ),
            "rejected_values": output_spec(
                "rejected_values",
                [
                    "source_id ASC",
                    "source_record_locator ASC",
                    "source_field ASC",
                    "identity_role ASC",
                    "rejection_reason ASC",
                ],
            ),
        },
        "runtime": {
            "architecture": os.uname().machine,
            "models_allowed": False,
            "network_allowed": False,
            "os": "macOS",
            "pandas_serialization_allowed": False,
            "pyarrow": pa.__version__,
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "retrieval_outputs_allowed": False,
        },
        "schema_version": "fixture-consumed-sirens-1",
        "writer": {
            "compression": "zstd",
            "compression_level": 9,
            "data_page_version": "1.0",
            "format_version": "2.6",
            "rechunk_one_chunk_per_column": True,
            "row_group_size": 65536,
            "store_schema": True,
            "use_dictionary": False,
            "write_statistics": True,
        },
    }
    plan_path = tmp_path / "plan.json"
    _write_canonical(plan_path, plan)
    return plan_path, contract, source_path, builder_script


def test_build_is_private_sealed_and_byte_reproducible(tmp_path: Path) -> None:
    plan, contract, _source, builder_script = _fixture(tmp_path)
    first = build_registry(plan, contract, tmp_path / "out-a", builder_script)
    second = build_registry(plan, contract, tmp_path / "out-b", builder_script)

    assert sorted(path.name for path in first.iterdir()) == [
        "consumed_sirens.parquet",
        "manifest.json",
        "observations.parquet",
        "rejected_values.parquet",
        "sources.json",
    ]
    assert first.name == second.name
    for filename in (
        "sources.json",
        "observations.parquet",
        "consumed_sirens.parquet",
        "rejected_values.parquet",
        "manifest.json",
    ):
        if filename != "manifest.json":
            assert (first / filename).read_bytes() == (second / filename).read_bytes()
        assert (first / filename).stat().st_mode & 0o777 == 0o600
    assert first.stat().st_mode & 0o777 == 0o700

    manifest = json.loads((first / "manifest.json").read_bytes())
    assert manifest["observation_count"] == 2
    assert manifest["unique_siren_count"] == 2
    assert set(manifest) == set(
        json.loads(plan.read_text())["outputs"]["manifest"]["required_fields"]
    )
    consumed = pq.read_table(first / "consumed_sirens.parquet")
    assert consumed.column("siren").to_pylist() == ["123456789", "987654321"]
    assert consumed.schema.metadata is None

    with pytest.raises(FileExistsError):
        build_registry(plan, contract, tmp_path / "out-a", builder_script)


def test_validation_fails_closed_on_input_drift(tmp_path: Path) -> None:
    plan_path, contract, source, builder_script = _fixture(tmp_path)
    source.write_bytes(source.read_bytes() + b"drift")
    plan, *_ = validate_plan(
        plan_path, contract, builder_script, require_builder_pin=True
    )
    with pytest.raises(RegistryStop, match="STOP_INPUT_DRIFT"):
        validate_inputs(plan)


def test_candidate_mapping_is_forbidden_before_source_read(tmp_path: Path) -> None:
    plan_path, contract, _source, builder_script = _fixture(
        tmp_path, candidate_mapping=True
    )
    with pytest.raises(RegistryStop, match="candidate/prediction identity mapping"):
        validate_plan(
            plan_path, contract, builder_script, require_builder_pin=True
        )


@pytest.mark.parametrize(
    "field",
    [
        "top1_siren",
        "retrieval_output_siret",
        "snapshot_neighbor_siret",
        "snapshot_universe_siren",
    ],
)
def test_semantic_candidate_families_are_forbidden(
    tmp_path: Path, field: str
) -> None:
    plan_path, contract, _source, builder_script = _fixture(
        tmp_path, forbidden_siren_field=field
    )
    with pytest.raises(RegistryStop, match="candidate/prediction identity mapping"):
        validate_plan(
            plan_path, contract, builder_script, require_builder_pin=True
        )


def test_duplicate_observation_key_is_actually_deduplicated(
    tmp_path: Path,
) -> None:
    plan_path, contract, _source, builder_script = _fixture(
        tmp_path, duplicate_observation=True
    )
    output = build_registry(
        plan_path, contract, tmp_path / "out", builder_script
    )
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["observation_count"] == 2
    assert pq.read_table(output / "observations.parquet").num_rows == 2


def test_siret_siren_mismatch_stops_build(tmp_path: Path) -> None:
    plan_path, contract, _source, builder_script = _fixture(
        tmp_path, mismatch=True
    )
    with pytest.raises(
        RegistryStop,
        match=r"^STOP_SIRET_SIREN_MISMATCH: SIRET/SIREN mismatch",
    ):
        build_registry(
            plan_path, contract, tmp_path / "out", builder_script
        )


def test_validate_only_cli_does_not_create_output(tmp_path: Path) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(builder_script),
            "--plan",
            str(plan_path),
            "--contract",
            str(contract),
            "--output-root",
            str(output),
            "--validate-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "VALIDATED_NO_BUILD"
    assert not output.exists()


def test_plan_must_be_canonical_and_contract_pinned(tmp_path: Path) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    parsed = json.loads(plan_path.read_text())
    plan_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    with pytest.raises(RegistryStop, match="plan JSON is not canonical"):
        validate_plan(
            plan_path, contract, builder_script, require_builder_pin=True
        )


def test_builder_pin_rejects_worktree_matching_plan_but_not_audited_blob(
    tmp_path: Path,
) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    builder_script.write_bytes(builder_script.read_bytes() + b"\n# drift\n")
    plan = json.loads(plan_path.read_bytes())
    plan["build"]["builder"]["source_sha256"] = file_sha256(builder_script)
    _write_canonical(plan_path, plan)
    with pytest.raises(
        RegistryStop,
        match="builder worktree/blob/ancestry/path/status pin is invalid",
    ):
        validate_plan(
            plan_path, contract, builder_script, require_builder_pin=True
        )


def test_tests_pin_rejects_worktree_matching_plan_but_not_audited_blob(
    tmp_path: Path,
) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    tests_path = (
        builder_script.parent.parent
        / "tests"
        / "test_build_v412_consumed_sirens_registry.py"
    )
    tests_path.write_bytes(tests_path.read_bytes() + b"\n# tests drift\n")
    plan = json.loads(plan_path.read_bytes())
    plan["build"]["builder"]["tests_sha256"] = file_sha256(tests_path)
    _write_canonical(plan_path, plan)
    with pytest.raises(
        RegistryStop,
        match="builder worktree/blob/ancestry/path/status pin is invalid",
    ):
        validate_plan(
            plan_path, contract, builder_script, require_builder_pin=True
        )


def test_audited_builder_commit_may_be_ancestor_of_later_head(
    tmp_path: Path,
) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    repository = builder_script.parent.parent
    audited_commit = json.loads(plan_path.read_bytes())["build"]["builder"][
        "git_commit"
    ]
    (repository / "README.md").write_text("later metadata\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "later head"],
        cwd=repository,
        check=True,
    )
    assert _git_commit(repository) != audited_commit
    _plan, _plan_hash, _contract_hash, accepted_commit = validate_plan(
        plan_path, contract, builder_script, require_builder_pin=True
    )
    assert accepted_commit == audited_commit


def test_builder_pin_ignores_replace_object_and_hex_named_ref(
    tmp_path: Path,
) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    audited_commit = plan["build"]["builder"]["git_commit"]
    repository = builder_script.parent.parent
    original_bytes = builder_script.read_bytes()

    builder_script.write_bytes(original_bytes + b"\n# replacement blob\n")
    subprocess.run(["git", "add", "scripts"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "replacement commit"],
        cwd=repository,
        check=True,
    )
    replacement_commit = _git_commit(repository)
    builder_script.write_bytes(original_bytes)
    subprocess.run(["git", "add", "scripts"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "restore audited blob"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "replace", audited_commit, replacement_commit],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "update-ref",
            f"refs/heads/{audited_commit}",
            replacement_commit,
        ],
        cwd=repository,
        check=True,
    )
    _plan, _plan_hash, _contract_hash, accepted_commit = validate_plan(
        plan_path, contract, builder_script, require_builder_pin=True
    )
    assert accepted_commit == audited_commit


def test_annotated_tag_object_is_not_an_audited_commit(tmp_path: Path) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    repository = builder_script.parent.parent
    subprocess.run(
        ["git", "tag", "-a", "audited-tag", "-m", "not a commit"],
        cwd=repository,
        check=True,
    )
    tag_oid = subprocess.run(
        ["git", "rev-parse", "audited-tag"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plan = json.loads(plan_path.read_bytes())
    plan["build"]["builder"]["git_commit"] = tag_oid
    _write_canonical(plan_path, plan)
    with pytest.raises(
        RegistryStop,
        match="builder worktree/blob/ancestry/path/status pin is invalid",
    ):
        validate_plan(
            plan_path, contract, builder_script, require_builder_pin=True
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("git_commit", "HEAD^{commit}:scripts/build_v412_consumed_sirens_registry.py"),
        ("source_path", "scripts/../scripts/build_v412_consumed_sirens_registry.py"),
        ("source_path", "--upload-pack=malicious"),
        ("tests_path", "tests/../tests/test_build_v412_consumed_sirens_registry.py"),
        ("tests_path", "--config-env=malicious"),
        ("status", "READY"),
    ],
)
def test_builder_pin_rejects_injection_path_and_non_pinned_status(
    tmp_path: Path, field: str, value: str
) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    plan["build"]["builder"][field] = value
    _write_canonical(plan_path, plan)
    with pytest.raises(
        RegistryStop,
        match="builder worktree/blob/ancestry/path/status pin is invalid",
    ):
        validate_plan(
            plan_path, contract, builder_script, require_builder_pin=True
        )


def test_shallow_history_missing_audited_commit_is_rejected(
    tmp_path: Path,
) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    repository = builder_script.parent.parent
    (repository / "README.md").write_text("later head\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "later head"],
        cwd=repository,
        check=True,
    )
    shallow = tmp_path / "shallow"
    subprocess.run(
        [
            "git",
            "clone",
            "-q",
            "--depth",
            "1",
            repository.as_uri(),
            str(shallow),
        ],
        check=True,
    )
    shallow_script = shallow / "scripts" / builder_script.name
    with pytest.raises(
        RegistryStop,
        match="builder worktree/blob/ancestry/path/status pin is invalid",
    ):
        validate_plan(
            plan_path, contract, shallow_script, require_builder_pin=True
        )


def test_git_validation_ignores_path_binary_and_inherited_git_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    marker = tmp_path / "fake-git-invoked"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        f"touch '{marker}'\n"
        "exit 97\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o700)
    lure = tmp_path / "lure"
    lure.mkdir()
    monkeypatch.setenv("PATH", str(fake_bin))
    monkeypatch.setenv("GIT_DIR", str(lure / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(lure))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(lure / "objects"))
    monkeypatch.setenv(
        "GIT_ALTERNATE_OBJECT_DIRECTORIES", str(lure / "alternate")
    )
    monkeypatch.setenv("GIT_INDEX_FILE", str(lure / "index"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(lure / "config"))
    _plan, _plan_hash, _contract_hash, accepted_commit = validate_plan(
        plan_path, contract, builder_script, require_builder_pin=True
    )
    assert accepted_commit == json.loads(plan_path.read_bytes())["build"][
        "builder"
    ]["git_commit"]
    assert not marker.exists()


def test_build_api_rejects_duplicate_source_id_and_path(tmp_path: Path) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    plan["identity_sources"].append(dict(plan["identity_sources"][0]))
    plan["invariants"]["expected_identity_source_count"] = 2
    _write_canonical(plan_path, plan)
    with pytest.raises(RegistryStop, match="duplicate identity source id/path"):
        build_registry(
            plan_path, contract, tmp_path / "out", builder_script
        )


def test_complete_crash_tree_is_promoted_without_payload_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    output_root = tmp_path / "out"
    completed = build_registry(
        plan_path, contract, output_root, builder_script
    )
    build_id = completed.name
    crash_tree = output_root / f".tmp-{build_id}-dead"
    completed.rename(crash_tree)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("complete recovery must not reread payload sources")

    monkeypatch.setattr(subject, "validate_inputs", forbidden)
    monkeypatch.setattr(subject, "_extract", forbidden)
    recovered = build_registry(
        plan_path, contract, output_root, builder_script
    )
    assert recovered == output_root / build_id
    assert recovered.is_dir()
    assert not crash_tree.exists()


def test_recovery_rejects_physically_rehashed_but_semantically_forged_tree(
    tmp_path: Path,
) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    output_root = tmp_path / "out"
    completed = build_registry(
        plan_path, contract, output_root, builder_script
    )
    build_id = completed.name
    crash_tree = output_root / f".tmp-{build_id}-forged"
    completed.rename(crash_tree)

    consumed_path = crash_tree / "consumed_sirens.parquet"
    consumed_table = pq.read_table(consumed_path)
    forged_rows = consumed_table.to_pylist()
    forged_rows[0]["siren"] = "555555555"
    pq.write_table(
        pa.Table.from_pylist(forged_rows, schema=consumed_table.schema),
        consumed_path,
        row_group_size=65536,
        version="2.6",
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        store_schema=True,
    )
    consumed_path.chmod(0o600)
    manifest_path = crash_tree / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["sirens_logical_sha256"] = subject.canonical_hash(forged_rows)
    manifest["files"]["consumed_sirens.parquet"] = {
        "size_bytes": consumed_path.stat().st_size,
        "sha256": file_sha256(consumed_path),
    }
    manifest["tree_payload_sha256"] = subject.canonical_hash(manifest["files"])
    manifest_path.write_bytes(canonical_bytes(manifest))
    manifest_path.chmod(0o600)

    with pytest.raises(
        RegistryStop,
        match="consumed registry differs from observation aggregation",
    ):
        build_registry(plan_path, contract, output_root, builder_script)
    assert crash_tree.is_dir()
    assert not (output_root / build_id).exists()


def test_output_operations_remain_bound_to_anchored_dirfd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, contract, _source, builder_script = _fixture(tmp_path)
    output_root = tmp_path / "out"
    moved_root = tmp_path / "out-original"
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    original_write = subject._write_exclusive_bytes_at
    swapped = False

    def swap_then_write(
        parent_fd: int, name: str, payload: bytes
    ) -> None:
        nonlocal swapped
        if not swapped:
            output_root.rename(moved_root)
            output_root.symlink_to(decoy, target_is_directory=True)
            swapped = True
        original_write(parent_fd, name, payload)

    monkeypatch.setattr(subject, "_write_exclusive_bytes_at", swap_then_write)
    built = build_registry(plan_path, contract, output_root, builder_script)
    assert swapped
    assert not any(decoy.iterdir())
    assert (moved_root / built.name / "manifest.json").is_file()
    output_root.unlink()
    moved_root.rename(output_root)
