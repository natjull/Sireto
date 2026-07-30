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


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
) -> tuple[Path, Path, Path]:
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
                "git_commit": _git_commit(),
                "source_path": str(SCRIPT),
                "source_sha256": file_sha256(SCRIPT),
                "status": "PINNED",
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
    return plan_path, contract, source_path


def test_build_is_private_sealed_and_byte_reproducible(tmp_path: Path) -> None:
    plan, contract, _source = _fixture(tmp_path)
    first = build_registry(plan, contract, tmp_path / "out-a", SCRIPT)
    second = build_registry(plan, contract, tmp_path / "out-b", SCRIPT)

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
        build_registry(plan, contract, tmp_path / "out-a", SCRIPT)


def test_validation_fails_closed_on_input_drift(tmp_path: Path) -> None:
    plan_path, contract, source = _fixture(tmp_path)
    source.write_bytes(source.read_bytes() + b"drift")
    plan, *_ = validate_plan(
        plan_path, contract, SCRIPT, require_builder_pin=True
    )
    with pytest.raises(RegistryStop, match="STOP_INPUT_DRIFT"):
        validate_inputs(plan)


def test_candidate_mapping_is_forbidden_before_source_read(tmp_path: Path) -> None:
    plan_path, contract, _source = _fixture(tmp_path, candidate_mapping=True)
    with pytest.raises(RegistryStop, match="candidate/prediction identity mapping"):
        validate_plan(plan_path, contract, SCRIPT, require_builder_pin=True)


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
    plan_path, contract, _source = _fixture(
        tmp_path, forbidden_siren_field=field
    )
    with pytest.raises(RegistryStop, match="candidate/prediction identity mapping"):
        validate_plan(plan_path, contract, SCRIPT, require_builder_pin=True)


def test_duplicate_observation_key_is_actually_deduplicated(
    tmp_path: Path,
) -> None:
    plan_path, contract, _source = _fixture(
        tmp_path, duplicate_observation=True
    )
    output = build_registry(plan_path, contract, tmp_path / "out", SCRIPT)
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["observation_count"] == 2
    assert pq.read_table(output / "observations.parquet").num_rows == 2


def test_siret_siren_mismatch_stops_build(tmp_path: Path) -> None:
    plan_path, contract, _source = _fixture(tmp_path, mismatch=True)
    with pytest.raises(
        RegistryStop,
        match=r"^STOP_SIRET_SIREN_MISMATCH: SIRET/SIREN mismatch",
    ):
        build_registry(plan_path, contract, tmp_path / "out", SCRIPT)


def test_validate_only_cli_does_not_create_output(tmp_path: Path) -> None:
    plan_path, contract, _source = _fixture(tmp_path)
    output = tmp_path / "must-not-exist"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
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
    plan_path, contract, _source = _fixture(tmp_path)
    parsed = json.loads(plan_path.read_text())
    plan_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    with pytest.raises(RegistryStop, match="plan JSON is not canonical"):
        validate_plan(plan_path, contract, SCRIPT, require_builder_pin=True)


def test_build_api_rejects_duplicate_source_id_and_path(tmp_path: Path) -> None:
    plan_path, contract, _source = _fixture(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    plan["identity_sources"].append(dict(plan["identity_sources"][0]))
    plan["invariants"]["expected_identity_source_count"] = 2
    _write_canonical(plan_path, plan)
    with pytest.raises(RegistryStop, match="duplicate identity source id/path"):
        build_registry(plan_path, contract, tmp_path / "out", SCRIPT)


def test_complete_crash_tree_is_promoted_without_payload_reread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, contract, _source = _fixture(tmp_path)
    output_root = tmp_path / "out"
    completed = build_registry(plan_path, contract, output_root, SCRIPT)
    build_id = completed.name
    crash_tree = output_root / f".tmp-{build_id}-dead"
    completed.rename(crash_tree)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("complete recovery must not reread payload sources")

    monkeypatch.setattr(subject, "validate_inputs", forbidden)
    monkeypatch.setattr(subject, "_extract", forbidden)
    recovered = build_registry(plan_path, contract, output_root, SCRIPT)
    assert recovered == output_root / build_id
    assert recovered.is_dir()
    assert not crash_tree.exists()


def test_recovery_rejects_physically_rehashed_but_semantically_forged_tree(
    tmp_path: Path,
) -> None:
    plan_path, contract, _source = _fixture(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    output_root = tmp_path / "out"
    completed = build_registry(plan_path, contract, output_root, SCRIPT)
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
        build_registry(plan_path, contract, output_root, SCRIPT)
    assert crash_tree.is_dir()
    assert not (output_root / build_id).exists()


def test_output_operations_remain_bound_to_anchored_dirfd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, contract, _source = _fixture(tmp_path)
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
    built = build_registry(plan_path, contract, output_root, SCRIPT)
    assert swapped
    assert not any(decoy.iterdir())
    assert (moved_root / built.name / "manifest.json").is_file()
    output_root.unlink()
    moved_root.rename(output_root)
