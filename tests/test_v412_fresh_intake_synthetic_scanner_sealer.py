from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import sys
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "config/v4_12_fresh_intake_synthetic_scanner_sealer_plan.json"
CONTRACT_PATH = ROOT / "docs/v4_12_fresh_intake_synthetic_scanner_sealer_contract.md"
FRESH_PLAN_PATH = ROOT / "config/v4_12_fresh_holdout_intake_plan.json"
PRODUCER_PATH = ROOT / "scripts/build_v412_fresh_intake_synthetic_fixture.py"
SCANNER_PATH = ROOT / "scripts/run_v412_fresh_intake_synthetic_scanner_sealer.py"
CATNAT_ROOT = Path("/Volumes/CATNAT_DATA")
PYTEST_V412_ROOT = CATNAT_ROOT / "SIRETO_RECALL100/tmp/pytest_v412"

EXPECTED_PLAN_SHA256 = "e8a55a999035183363c0bf7711280b09553a305434173286e41c696ea3e4772f"
EXPECTED_CONTRACT_SHA256 = "ad8eed1bf5d8d8a280ea8b212d3d308eb5c8b048efb3ebefb567956b3eb60ca8"
EXPECTED_FIXTURE_SHA256 = "6f917f98b7a8b42e34af390b21e63ef4cb33051aa5a0bf7154f5351c4337d33e"
EXPECTED_GOLDEN_SHA256 = "9af0547f33d4caf6aab89655b5a1e357068e0adc519e70135eab9a850425b64f"
AUDIT_TIME = "2026-07-30T00:00:01Z"
ID_PATTERN = re.compile(r"^[a-p]{64}$")
HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
QUERY_ID_DOMAIN = "SIRETO-V412-FRESH-QUERY-ID\0"

INPUT_FILES = {
    "collection_source_manifest.json",
    "source_manifest.json",
    "crm_safe.csv",
    "evidence_source_manifest.json",
    "evidence_source.parquet",
}
SEALED_INPUT_TREE = INPUT_FILES | {"payload_manifest.json", "seal.json"}
SCAN_PAYLOADS = {
    "safe_queries_preidentity.parquet",
    "quarantine_proofs.parquet",
    "source_identity_map.parquet",
    "scan_integrity.json",
    "scan_provenance.json",
}
SCAN_TREE = SCAN_PAYLOADS | {"payload_manifest.json", "seal.json"}
BATCH_QUARANTINE_TREE = {
    "batch_quarantine_proof.json",
    "payload_manifest.json",
    "seal.json",
}

BATCH_REASONS = [
    "CSV_ENCODING_INVALID",
    "CSV_BOM_FORBIDDEN",
    "CSV_NUL_FORBIDDEN",
    "CSV_MIXED_LINE_ENDINGS",
    "CSV_HEADER_DRIFT",
    "CSV_ROW_SHAPE_DRIFT",
]
ROW_REASONS = [
    "DUPLICATE_SOURCE_RECORD_ID",
    "EMPTY_REQUIRED_PROVENANCE",
    "PROVENANCE_MISMATCH",
    "LOCATION_RULE_FAILED",
    "UNICODE_DECIMAL_9_OR_14",
]


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical_without_lf(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical(value: Any) -> bytes:
    return _canonical_without_lf(value) + b"\n"


def _write_canonical(path: Path, value: Any) -> None:
    path.write_bytes(_canonical(value))


def _load_strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise AssertionError(f"duplicate JSON key {key!r} in {path}")
            output[key] = value
        return output

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate)
    assert isinstance(value, dict)
    assert path.read_bytes() == _canonical(value)
    return value


def _load_module(name: str, path: Path):
    if not path.is_file():
        pytest.skip(f"future S0 module not present yet: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def plan() -> dict[str, Any]:
    value = _load_strict_json(PLAN_PATH)
    assert _sha(PLAN_PATH) == EXPECTED_PLAN_SHA256
    assert _sha(CONTRACT_PATH) == EXPECTED_CONTRACT_SHA256
    assert value["contract"]["sha256"] == EXPECTED_CONTRACT_SHA256
    return value


@pytest.fixture(scope="session")
def producer():
    module = _load_module("build_v412_fresh_intake_synthetic_fixture", PRODUCER_PATH)
    assert callable(getattr(module, "build_fixture", None))
    return module


@pytest.fixture(scope="session")
def scanner():
    module = _load_module("run_v412_fresh_intake_synthetic_scanner_sealer", SCANNER_PATH)
    assert callable(getattr(module, "run_scanner", None))
    return module


@pytest.fixture
def catnat_tmp_path(tmp_path: Path) -> Path:
    if not CATNAT_ROOT.is_dir():
        pytest.skip("CATNAT_DATA is not mounted; S0 tests never fall back to another volume")
    resolved = tmp_path.resolve()
    required = PYTEST_V412_ROOT.resolve()
    if not resolved.is_relative_to(required):
        pytest.skip(
            "S0 tests require pytest --basetemp="
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/tmp/pytest_v412/<run>"
        )
    assert resolved != required
    return resolved


def _result_path(result: dict[str, Any], keys: tuple[str, ...]) -> Path | None:
    for key in keys:
        value = result.get(key)
        if isinstance(value, (str, os.PathLike)):
            return Path(value)
    return None


def _only(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    assert len(matches) == 1, f"expected exactly one {name} below {root}, got {matches}"
    return matches[0]


def _build_fixture(producer: Any, root: Path) -> tuple[dict[str, Any], Path, str]:
    assert root.parent.is_dir()
    assert not root.exists()
    result = producer.build_fixture(plan_path=PLAN_PATH, root=root)
    assert isinstance(result, dict)
    control_path = _result_path(
        result,
        ("control_manifest_path", "fixture_control_manifest_path", "control_path"),
    )
    if control_path is None:
        control_path = _only(root, "fixture_control_manifest.json")
    assert control_path.is_file()
    control = _load_strict_json(control_path)
    run_id = control["synthetic_run_id"]
    assert ID_PATTERN.fullmatch(run_id)
    return result, control_path, run_id


def _run(
    scanner: Any,
    root: Path,
    control_path: Path,
    *,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    calls: list[float] = []

    def zero_wait(seconds: float) -> None:
        calls.append(seconds)
        if sleep_fn is not None:
            sleep_fn(seconds)

    result = scanner.run_scanner(
        plan_path=PLAN_PATH,
        control_manifest_path=control_path,
        root=root,
        stability_wait_seconds=0.0,
        sleep_fn=zero_wait,
        audit_now_fn=lambda: AUDIT_TIME,
        _test_mode=True,
    )
    assert isinstance(result, dict)
    assert all(seconds == 0.0 for seconds in calls)
    return result


def _verdict(result: dict[str, Any]) -> str:
    for key in ("verdict", "status", "result"):
        value = result.get(key)
        if isinstance(value, str) and (
            "INGESTED" in value or "QUARANTINED" in value or "STOP" in value
        ):
            return value
    raise AssertionError(f"no S0 verdict in result: {result}")


def _assert_stop(call: Callable[[], dict[str, Any]]) -> None:
    try:
        result = call()
    except Exception as exc:  # fail-closed exception is an authorised STOP surface
        assert "STOP" in f"{type(exc).__name__}: {exc}".upper()
    else:
        assert "STOP" in _verdict(result)


def _paths(root: Path, run_id: str) -> dict[str, Path]:
    return {
        "inbox": root / "inbox" / run_id / "package",
        "sealed": root / "sealed" / run_id / "input",
        "scan": root / "scan" / run_id / "output",
        "quarantine": root / "quarantine" / run_id / "batch",
        "audit": root / "audit" / run_id,
        "temp": root / "tmp" / run_id,
    }


def _tree_names(root: Path) -> set[str]:
    assert root.is_dir()
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_modes(root: Path) -> None:
    for path in root.rglob("*"):
        mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
        if path.is_dir():
            assert mode == 0o700, (path, oct(mode))
        elif path.is_file():
            assert mode == 0o600, (path, oct(mode))


def _assert_arrow_schema(path: Path, spec: list[list[Any]]) -> pa.Table:
    table = pq.read_table(path)
    actual = [(field.name, str(field.type), field.nullable) for field in table.schema]
    expected = [(name, arrow_type, nullable) for name, arrow_type, nullable in spec]
    assert actual == expected
    return table


def _map_id(domain: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(domain.encode("utf-8") + _canonical_without_lf(payload)).hexdigest()
    return digest.translate(str.maketrans("0123456789abcdef", "abcdefghijklmnop"))


def _has_forbidden_decimal(value: str) -> bool:
    projected = unicodedata.normalize("NFKC", value)
    index = 0
    while index < len(projected):
        if not projected[index].isdecimal():
            index += 1
            continue
        end = index + 1
        while end < len(projected) and projected[end].isdecimal():
            end += 1
        if end - index in {9, 14}:
            return True
        index = end
    return False


def _raw_row_hash(row: dict[str, str], columns: list[str]) -> str:
    return _sha_bytes(_canonical_without_lf([row[column] for column in columns]))


def _assert_sealed_package(
    directory: Path,
    expected_tree: set[str],
    expected_kind: str,
    expected_payloads: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert _tree_names(directory) == expected_tree
    manifest_path = directory / "payload_manifest.json"
    seal_path = directory / "seal.json"
    manifest = _load_strict_json(manifest_path)
    seal = _load_strict_json(seal_path)
    assert manifest["package_kind"] == expected_kind
    assert seal["package_kind"] == expected_kind
    assert manifest["payload_count"] == len(expected_payloads)
    records = manifest["ordered_payload_records"]
    assert [record["relative_path"] for record in records] == expected_payloads
    assert set(records[0]) == {"relative_path", "size_bytes", "sha256"}
    for record in records:
        payload = directory / record["relative_path"]
        assert payload.is_file()
        assert record["size_bytes"] == payload.stat().st_size
        assert record["sha256"] == _sha(payload)
    tree_hash = _sha_bytes(_canonical_without_lf(records))
    assert manifest["payload_tree_sha256"] == tree_hash
    assert seal["payload_tree_sha256"] == tree_hash
    assert seal["payload_manifest_size_bytes"] == manifest_path.stat().st_size
    assert seal["payload_manifest_sha256"] == _sha(manifest_path)
    return manifest, seal


def _audit_files(paths: dict[str, Path]) -> tuple[list[Path], list[Path]]:
    events = sorted(
        path
        for path in (paths["audit"] / "events").glob("*.json")
        if path.name[:8] in {"00000001", "00000002", "00000003"}
    )
    generations = sorted((paths["audit"] / "events_manifests").glob("*.json"))
    return events, generations


def _assert_event_chain(paths: dict[str, Path], terminal: str) -> None:
    events, generations = _audit_files(paths)
    assert len(events) == 3
    assert len(generations) == 3
    previous_event: str | None = None
    previous_generation: str | None = None
    event_records: list[dict[str, Any]] = []
    for sequence, path in enumerate(events, 1):
        assert path.name.startswith(f"{sequence:08d}-")
        event = _load_strict_json(path)
        assert event["sequence"] == sequence
        assert event["previous_event_sha256"] == previous_event
        assert path.name == f"{sequence:08d}-{_sha(path)}.json"
        assert set(event["manifest_hashes"]) == {
            "collection_source_manifest_sha256",
            "source_manifest_sha256",
            "evidence_source_manifest_sha256",
        }
        assert set(event["tree_hashes"]) == {
            "sealed_input_payload_manifest_sha256",
            "sealed_input_seal_sha256",
            "scan_output_payload_manifest_sha256",
            "scan_output_seal_sha256",
            "batch_quarantine_payload_manifest_sha256",
            "batch_quarantine_seal_sha256",
        }
        previous_event = _sha(path)
        event_records.append(
            {
                "relative_path": f"events/{path.name}",
                "sequence": sequence,
                "size_bytes": path.stat().st_size,
                "sha256": previous_event,
            }
        )
        generation_path = generations[sequence - 1]
        generation = _load_strict_json(generation_path)
        assert generation["generation"] == sequence
        assert generation["event_count"] == sequence
        assert generation["ordered_event_records"] == event_records
        assert generation["head_event_sha256"] == previous_event
        assert generation["previous_manifest_sha256"] == previous_generation
        assert generation_path.name == f"{sequence:08d}-{_sha(generation_path)}.json"
        previous_generation = _sha(generation_path)
    assert _load_strict_json(events[0])["new_state"] == "RECEIPTED"
    assert _load_strict_json(events[1])["new_state"] == terminal
    assert _load_strict_json(events[2])["new_state"] == terminal


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    rows = [dict(row) for row in reader]
    assert reader.fieldnames is not None
    return list(reader.fieldnames), rows


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=columns,
        delimiter=",",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8", newline="")


def _repin_source(control_path: Path, source_path: Path) -> None:
    inbox = source_path.parent
    source_manifest_path = inbox / "source_manifest.json"
    source_manifest = _load_strict_json(source_manifest_path)
    source_manifest["source_size_bytes"] = source_path.stat().st_size
    source_manifest["source_sha256"] = _sha(source_path)
    _write_canonical(source_manifest_path, source_manifest)
    control = _load_strict_json(control_path)
    control["crm_safe_csv_sha256"] = _sha(source_path)
    control["source_manifest_sha256"] = _sha(source_manifest_path)
    _write_canonical(control_path, control)


def _repin_evidence(control_path: Path, evidence_path: Path) -> None:
    inbox = evidence_path.parent
    manifest_path = inbox / "evidence_source_manifest.json"
    manifest = _load_strict_json(manifest_path)
    manifest["evidence_size_bytes"] = evidence_path.stat().st_size
    manifest["evidence_sha256"] = _sha(evidence_path)
    _write_canonical(manifest_path, manifest)
    control = _load_strict_json(control_path)
    control["evidence_source_parquet_sha256"] = _sha(evidence_path)
    control["evidence_source_manifest_sha256"] = _sha(manifest_path)
    _write_canonical(control_path, control)


def _rewrite_parquet_with_application_metadata(
    path: Path,
    plan: dict[str, Any],
) -> None:
    table = pq.read_table(path).replace_schema_metadata(
        {b"sireto-test-application-metadata": b"forbidden"}
    )
    writer = plan["parquet_writer"]
    pq.write_table(
        table,
        path,
        compression=writer["compression"],
        compression_level=writer["compression_level"],
        version=writer["format_version"],
        data_page_version=writer["data_page_version"],
        use_dictionary=writer["use_dictionary"],
        write_statistics=writer["write_statistics"],
        row_group_size=writer["row_group_size"],
        store_schema=writer["store_schema"],
    )
    assert (
        pq.ParquetFile(path).schema_arrow.metadata
        == {b"sireto-test-application-metadata": b"forbidden"}
    )


def _build_and_repin_csv(
    producer: Any,
    root: Path,
    mutate: Callable[[Path], None],
) -> tuple[Path, str]:
    _, control_path, run_id = _build_fixture(producer, root)
    source = _paths(root, run_id)["inbox"] / "crm_safe.csv"
    mutate(source)
    _repin_source(control_path, source)
    return control_path, run_id


def test_contract_plan_pins_and_golden_vectors(
    catnat_tmp_path: Path, plan: dict[str, Any]
) -> None:
    del catnat_tmp_path
    fixture_hash = _sha_bytes(_canonical_without_lf(plan["fixture"]))
    assert fixture_hash == EXPECTED_FIXTURE_SHA256
    assert plan["control_manifest"]["fixture_spec_sha256"] == fixture_hash

    fresh = _load_strict_json(FRESH_PLAN_PATH)
    golden = {
        "opaque": fresh["opaque_ids"]["golden_vectors"],
        "unicode": fresh["outputs"]["safe_queries"]["anti_leak_scan"]["golden_vectors"],
    }
    assert _sha_bytes(_canonical_without_lf(golden)) == EXPECTED_GOLDEN_SHA256
    assert plan["references"]["golden"]["canonical_projection_sha256"] == EXPECTED_GOLDEN_SHA256

    domains = fresh["opaque_ids"]["domains"]
    for vector in golden["opaque"]:
        assert _map_id(domains[vector["kind"]], vector["payload"]) == vector["expected_id"]
    for vector in golden["unicode"]:
        assert _has_forbidden_decimal(vector["input"]) is vector["expected_match"]


def test_nominal_six_rows_exact_outputs(
    catnat_tmp_path: Path, plan: dict[str, Any], producer: Any, scanner: Any
) -> None:
    root = catnat_tmp_path / "nominal"
    _, control_path, run_id = _build_fixture(producer, root)
    control = _load_strict_json(control_path)

    fixture_hash = _sha_bytes(_canonical_without_lf(plan["fixture"]))
    expected_run = _map_id(
        plan["ids"]["run"]["domain"],
        {"fixture_spec_sha256": fixture_hash, "plan_sha256": _sha(PLAN_PATH)},
    )
    assert run_id == expected_run
    assert control["fixture_spec_sha256"] == fixture_hash

    paths = _paths(root, run_id)
    assert _tree_names(paths["inbox"]) == INPUT_FILES
    assert (paths["inbox"] / "crm_safe.csv").read_text() == plan["fixture"]["csv"][
        "exact_utf8_text"
    ]
    evidence_file = pq.ParquetFile(paths["inbox"] / "evidence_source.parquet")
    assert evidence_file.metadata.num_rows == 0
    assert evidence_file.metadata.num_row_groups == 1

    result = _run(scanner, root, control_path)
    assert _verdict(result) == plan["verdicts"]["ingested"]

    _assert_sealed_package(
        paths["sealed"],
        SEALED_INPUT_TREE,
        "SEALED_INPUT",
        plan["outputs"]["sealed_input"]["payloads"],
    )
    _assert_sealed_package(
        paths["scan"],
        SCAN_TREE,
        "SCAN_OUTPUT",
        plan["outputs"]["scan_output"]["payloads"],
    )
    assert not paths["quarantine"].exists()

    schemas = plan["scan"]["tree"]["schemas"]
    safe = _assert_arrow_schema(
        paths["scan"] / "safe_queries_preidentity.parquet",
        schemas["safe_queries_preidentity.parquet"],
    )
    proofs = _assert_arrow_schema(
        paths["scan"] / "quarantine_proofs.parquet",
        schemas["quarantine_proofs.parquet"],
    )
    identities = _assert_arrow_schema(
        paths["scan"] / "source_identity_map.parquet",
        schemas["source_identity_map.parquet"],
    )
    assert safe.num_rows == 2
    assert proofs.num_rows == 4
    assert identities.num_rows == 6
    safe_rows = safe.to_pylist()
    assert {row["crm_name_raw"] for row in safe_rows} == {"Mairie de Test", "École Démo"}
    assert [row["source_row_number"] for row in proofs.to_pylist()] == [3, 4, 5, 6]
    assert {row["reason_code"] for row in proofs.to_pylist()} == {
        "UNICODE_DECIMAL_9_OR_14"
    }
    for row in safe_rows:
        assert all(not _has_forbidden_decimal(value) for value in row.values() if isinstance(value, str))
        assert ID_PATTERN.fullmatch(row["query_id"])
        assert ID_PATTERN.fullmatch(row["opaque_batch_id"])
        assert ID_PATTERN.fullmatch(row["opaque_stratum_id"])

    identity_by_number = {
        row["source_row_number"]: row for row in identities.to_pylist()
    }
    source_columns, source_rows = _read_csv(paths["sealed"] / "crm_safe.csv")
    source_sha = _sha(paths["sealed"] / "crm_safe.csv")
    for number, row in enumerate(source_rows, 1):
        raw_hash = _raw_row_hash(row, source_columns)
        identity = identity_by_number[number]
        assert identity["raw_row_sha256"] == raw_hash
        query_payload = {
            "collection_id": plan["fixture"]["collection_id"],
            "source_batch_id": row["source_batch_id"],
            "source_record_id": row["source_record_id"],
            "source_sha256": source_sha,
            "raw_row_sha256": raw_hash,
        }
        assert identity["query_id"] == _map_id(
            QUERY_ID_DOMAIN, query_payload
        )

    integrity = _load_strict_json(paths["scan"] / "scan_integrity.json")
    assert integrity["source_row_count"] == 6
    assert integrity["safe_row_count"] == 2
    assert integrity["quarantined_row_count"] == 4
    assert integrity["quarantine_proof_count"] == 4
    assert integrity["reason_counts"] == plan["fixture"]["expected"]["reason_counts"]
    assert integrity["all_safe_output_strings_rescanned"] is True
    assert integrity["all_ids_pattern_valid"] is True
    assert set(integrity["logical_hashes"]) == SCAN_PAYLOADS
    assert all(HEX64_PATTERN.fullmatch(value) for value in integrity["logical_hashes"].values())

    receipt_collection = _only(paths["audit"] / "receipts" / "collections", "receipt.json")
    receipt_batch = _only(paths["audit"] / "receipts" / "batches", "receipt.json")
    collection = _load_strict_json(receipt_collection)
    batch = _load_strict_json(receipt_batch)
    assert set(collection) == set(plan["receipts"]["collection_exact_fields"])
    assert set(batch) == set(plan["receipts"]["batch_exact_fields"])
    assert batch["sealed_source_relative_path"] == "crm_safe.csv"
    assert batch["sealed_source_sha256"] == source_sha
    assert batch["sealed_source_size_bytes"] == (paths["sealed"] / "crm_safe.csv").stat().st_size
    assert ID_PATTERN.fullmatch(collection["receipt_id"])
    assert ID_PATTERN.fullmatch(batch["receipt_id"])
    _assert_event_chain(paths, "INGESTED")
    _assert_modes(root)


def test_producer_builds_exact_pinned_fixture(
    catnat_tmp_path: Path, plan: dict[str, Any], producer: Any
) -> None:
    root = catnat_tmp_path / "producer-only"
    result, control_path, run_id = _build_fixture(producer, root)
    paths = _paths(root, run_id)
    assert Path(result["package_path"]) == paths["inbox"]
    assert _tree_names(paths["inbox"]) == INPUT_FILES
    assert paths["inbox"].joinpath("crm_safe.csv").read_bytes() == plan["fixture"][
        "csv"
    ]["exact_utf8_text"].encode("utf-8")
    control = _load_strict_json(control_path)
    assert set(control) == set(plan["control_manifest"]["exact_fields"])
    assert control["fixture_spec_sha256"] == EXPECTED_FIXTURE_SHA256
    for key, filename in (
        ("collection_source_manifest_sha256", "collection_source_manifest.json"),
        ("source_manifest_sha256", "source_manifest.json"),
        ("crm_safe_csv_sha256", "crm_safe.csv"),
        ("evidence_source_manifest_sha256", "evidence_source_manifest.json"),
        ("evidence_source_parquet_sha256", "evidence_source.parquet"),
    ):
        assert control[key] == _sha(paths["inbox"] / filename)
    assert result["fixture_control_manifest_sha256"] == _sha(control_path)
    evidence = pq.ParquetFile(paths["inbox"] / "evidence_source.parquet")
    assert evidence.metadata.num_rows == 0
    assert evidence.metadata.num_row_groups == 1
    _assert_modes(root)


def _batch_mutator(reason: str) -> Callable[[Path], None]:
    def mutate(path: Path) -> None:
        payload = path.read_bytes()
        if reason == "CSV_ENCODING_INVALID":
            path.write_bytes(payload.replace("Mairie".encode(), b"Mair\xffie", 1))
        elif reason == "CSV_BOM_FORBIDDEN":
            path.write_bytes(b"\xef\xbb\xbf" + payload)
        elif reason == "CSV_NUL_FORBIDDEN":
            path.write_bytes(payload.replace(b"Mairie", b"Mai\x00rie", 1))
        elif reason == "CSV_MIXED_LINE_ENDINGS":
            path.write_bytes(payload.replace(b"\n", b"\r\n", 1))
        elif reason == "CSV_HEADER_DRIFT":
            path.write_bytes(payload.replace(b"crm_name_raw", b"crm_name_changed", 1))
        elif reason == "CSV_ROW_SHAPE_DRIFT":
            lines = payload.splitlines(keepends=True)
            lines[1] = lines[1].replace(b",75101", b"", 1)
            path.write_bytes(b"".join(lines))
        else:  # pragma: no cover - protects the parametrisation itself
            raise AssertionError(reason)

    return mutate


@pytest.mark.parametrize("reason", BATCH_REASONS)
def test_each_batch_quarantine_reason(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    plan: dict[str, Any],
    reason: str,
) -> None:
    root = catnat_tmp_path / f"batch-{reason.lower()}"
    control_path, run_id = _build_and_repin_csv(producer, root, _batch_mutator(reason))
    result = _run(scanner, root, control_path)
    assert _verdict(result) == plan["verdicts"]["quarantined"]
    paths = _paths(root, run_id)
    assert not paths["scan"].exists()
    _assert_sealed_package(
        paths["quarantine"],
        BATCH_QUARANTINE_TREE,
        "BATCH_QUARANTINE",
        ["batch_quarantine_proof.json"],
    )
    proof = _load_strict_json(paths["quarantine"] / "batch_quarantine_proof.json")
    assert proof["reason_code"] == reason
    assert proof["expected_source_row_count"] == 6
    assert "raw" not in "".join(proof).lower()
    assert "locator" not in "".join(proof).lower()
    _assert_event_chain(paths, "QUARANTINED")


def _row_mutator(reason: str) -> Callable[[Path], None]:
    def mutate(path: Path) -> None:
        columns, rows = _read_csv(path)
        if reason == "DUPLICATE_SOURCE_RECORD_ID":
            rows[1]["source_record_id"] = rows[0]["source_record_id"]
        elif reason == "EMPTY_REQUIRED_PROVENANCE":
            rows[0]["source_system"] = ""
        elif reason == "PROVENANCE_MISMATCH":
            rows[0]["source_system"] = "OTHER"
        elif reason == "LOCATION_RULE_FAILED":
            for key in (
                "crm_name_raw",
                "crm_address_raw",
                "crm_postcode_raw",
                "crm_city_raw",
                "crm_insee_raw",
            ):
                rows[0][key] = ""
        elif reason == "UNICODE_DECIMAL_9_OR_14":
            rows[0]["crm_name_raw"] += " 123456789"
        else:  # pragma: no cover
            raise AssertionError(reason)
        _write_csv(path, columns, rows)

    return mutate


@pytest.mark.parametrize("reason", ROW_REASONS)
def test_each_row_quarantine_reason(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    plan: dict[str, Any],
    reason: str,
) -> None:
    root = catnat_tmp_path / f"row-{reason.lower()}"
    control_path, run_id = _build_and_repin_csv(producer, root, _row_mutator(reason))
    result = _run(scanner, root, control_path)
    assert _verdict(result) == plan["verdicts"]["ingested"]
    paths = _paths(root, run_id)
    proofs = pq.read_table(paths["scan"] / "quarantine_proofs.parquet").to_pylist()
    targeted = [proof for proof in proofs if proof["reason_code"] == reason]
    assert targeted
    if reason == "DUPLICATE_SOURCE_RECORD_ID":
        assert [proof["source_row_number"] for proof in targeted] == [1, 2]
    elif reason == "EMPTY_REQUIRED_PROVENANCE":
        assert [proof["source_row_number"] for proof in targeted] == [1]
        assert not any(
            proof["source_row_number"] == 1
            and proof["reason_code"] == "PROVENANCE_MISMATCH"
            for proof in proofs
        )
    else:
        assert any(proof["source_row_number"] == 1 for proof in targeted)


@pytest.mark.parametrize("kind", ["MUTATION_SAME_INODE", "PATH_SUBSTITUTION_NEW_INODE"])
def test_stability_rejects_mutation_and_substitution(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    kind: str,
) -> None:
    root = catnat_tmp_path / f"stability-{kind.lower()}"
    _, control_path, run_id = _build_fixture(producer, root)
    source = _paths(root, run_id)["inbox"] / "crm_safe.csv"

    def mutate_during_wait(_: float) -> None:
        if kind == "MUTATION_SAME_INODE":
            with source.open("ab") as handle:
                handle.write(b" ")
                handle.flush()
                os.fsync(handle.fileno())
        else:
            replacement = source.with_name("replacement.tmp")
            replacement.write_bytes(source.read_bytes())
            os.replace(replacement, source)

    _assert_stop(lambda: _run(scanner, root, control_path, sleep_fn=mutate_during_wait))
    paths = _paths(root, run_id)
    assert not (paths["audit"] / "receipts").exists()


@pytest.mark.parametrize(
    "attack",
    ["INPUT_SYMLINK", "INPUT_HARDLINK"],
)
def test_rejects_symlink_and_hardlink_in_sealed_input(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    attack: str,
) -> None:
    root = catnat_tmp_path / f"path-attack-{attack.lower()}"
    _, control_path, run_id = _build_fixture(producer, root)
    paths = _paths(root, run_id)
    _run(scanner, root, control_path)
    source = paths["sealed"] / "crm_safe.csv"
    if attack == "INPUT_SYMLINK":
        target = root / "synthetic-target.csv"
        target.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(target)
    elif attack == "INPUT_HARDLINK":
        os.link(source, root / "synthetic-hardlink.csv")
    _assert_stop(lambda: _run(scanner, root, control_path))


def test_production_api_rejects_short_stability_interval(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
) -> None:
    root = catnat_tmp_path / "production-short-wait"
    _, control_path, _run_id = _build_fixture(producer, root)
    _assert_stop(
        lambda: scanner.run_scanner(
            plan_path=PLAN_PATH,
            control_manifest_path=control_path,
            root=root,
            stability_wait_seconds=0.0,
        )
    )


def test_production_api_is_unconditionally_disabled_without_lock_and_sandbox(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
) -> None:
    root = catnat_tmp_path / "production-disabled"
    _, control_path, _run_id = _build_fixture(producer, root)

    def forbidden_sleep(_seconds: float) -> None:
        raise AssertionError("production-disabled invocation reached stability sleep")

    _assert_stop(
        lambda: scanner.run_scanner(
            plan_path=PLAN_PATH,
            control_manifest_path=control_path,
            root=root,
            stability_wait_seconds=60.0,
            sleep_fn=forbidden_sleep,
        )
    )


@pytest.mark.parametrize("violation", ["OUTSIDE_PYTEST_V412_ROOT", "MISSING_PYTEST_ENV"])
def test_test_mode_is_restricted_to_pytest_v412_root_and_live_pytest(
    catnat_tmp_path: Path,
    scanner: Any,
    monkeypatch: pytest.MonkeyPatch,
    violation: str,
) -> None:
    if violation == "OUTSIDE_PYTEST_V412_ROOT":
        attempted_root = (
            CATNAT_ROOT / "SIRETO_RECALL100/tmp/not_pytest_v412/test-mode-denied"
        )
        assert not attempted_root.is_relative_to(PYTEST_V412_ROOT)
    else:
        attempted_root = catnat_tmp_path / "missing-pytest-env"
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    _assert_stop(
        lambda: scanner.run_scanner(
            plan_path=PLAN_PATH,
            control_manifest_path=attempted_root / "not-opened.json",
            root=attempted_root,
            stability_wait_seconds=0.0,
            sleep_fn=lambda _seconds: None,
            _test_mode=True,
        )
    )
    assert not attempted_root.exists()


def test_test_mode_rejects_symlink_ancestor_escaping_pytest_v412_before_io(
    catnat_tmp_path: Path,
    scanner: Any,
) -> None:
    outside_target = (
        CATNAT_ROOT
        / "SIRETO_RECALL100/tmp"
        / f"v412-external-target-{catnat_tmp_path.name}"
    )
    assert not outside_target.is_relative_to(PYTEST_V412_ROOT)
    assert not outside_target.exists()
    ancestor_link = catnat_tmp_path / "ancestor-link"
    ancestor_link.symlink_to(outside_target, target_is_directory=True)
    attempted_root = ancestor_link / "must-not-be-created-or-read"

    _assert_stop(
        lambda: scanner.run_scanner(
            plan_path=PLAN_PATH,
            control_manifest_path=attempted_root / "not-opened.json",
            root=attempted_root,
            stability_wait_seconds=0.0,
            sleep_fn=lambda _seconds: None,
            _test_mode=True,
        )
    )
    assert ancestor_link.is_symlink()
    assert not outside_target.exists()
    assert not attempted_root.exists()


def test_cli_rejects_any_short_wait_override(
    catnat_tmp_path: Path,
    scanner: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCANNER_PATH),
            "--plan",
            str(PLAN_PATH),
            "--control-manifest",
            str(catnat_tmp_path / "not-opened.json"),
            "--root",
            str(catnat_tmp_path / "not-opened-root"),
            "--stability-wait-seconds",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as error:
        scanner.main()
    assert error.value.code == 2


@pytest.mark.parametrize("surface", ["CONTROL", "SOURCE_MANIFEST"])
def test_non_finite_json_is_rejected_as_stop(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    surface: str,
) -> None:
    root = catnat_tmp_path / f"nan-{surface.lower()}"
    _, control_path, run_id = _build_fixture(producer, root)
    if surface == "CONTROL":
        payload = control_path.read_text(encoding="utf-8").replace(
            '"expected_source_row_count":6',
            '"expected_source_row_count":NaN',
        )
        control_path.write_text(payload, encoding="utf-8", newline="")
    else:
        source_manifest = _paths(root, run_id)["inbox"] / "source_manifest.json"
        payload = source_manifest.read_text(encoding="utf-8").replace(
            '"source_row_count":6',
            '"source_row_count":NaN',
        )
        source_manifest.write_text(payload, encoding="utf-8", newline="")
        control = _load_strict_json(control_path)
        control["source_manifest_sha256"] = _sha(source_manifest)
        _write_canonical(control_path, control)
    _assert_stop(lambda: _run(scanner, root, control_path))


@pytest.mark.parametrize(
    "attack",
    ["CONTROL_WRONG_TYPE", "SOURCE_EXTRA_FIELD", "INPUT_MODE", "INPUT_HARDLINK"],
)
def test_control_manifest_and_input_metadata_are_strict(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    attack: str,
) -> None:
    root = catnat_tmp_path / f"strict-{attack.lower()}"
    _, control_path, run_id = _build_fixture(producer, root)
    paths = _paths(root, run_id)
    if attack == "CONTROL_WRONG_TYPE":
        control = _load_strict_json(control_path)
        control["expected_source_row_count"] = "6"
        _write_canonical(control_path, control)
    elif attack == "SOURCE_EXTRA_FIELD":
        source_manifest = paths["inbox"] / "source_manifest.json"
        source = _load_strict_json(source_manifest)
        source["unexpected"] = True
        _write_canonical(source_manifest, source)
        control = _load_strict_json(control_path)
        control["source_manifest_sha256"] = _sha(source_manifest)
        _write_canonical(control_path, control)
    elif attack == "INPUT_MODE":
        os.chmod(paths["inbox"] / "crm_safe.csv", 0o644)
    else:
        os.link(paths["inbox"] / "crm_safe.csv", root / "extra-hardlink.csv")
    _assert_stop(lambda: _run(scanner, root, control_path))


def test_evidence_parquet_application_metadata_is_rejected(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    plan: dict[str, Any],
) -> None:
    root = catnat_tmp_path / "evidence-application-metadata"
    _, control_path, run_id = _build_fixture(producer, root)
    evidence = _paths(root, run_id)["inbox"] / "evidence_source.parquet"
    _rewrite_parquet_with_application_metadata(evidence, plan)
    metadata = pq.ParquetFile(evidence).metadata
    assert metadata.num_rows == 0
    assert metadata.num_row_groups == 1
    _repin_evidence(control_path, evidence)
    _assert_stop(lambda: _run(scanner, root, control_path))


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _remove_files_after(directory: Path, maximum: int) -> None:
    if not directory.exists():
        return
    for path in directory.glob("*.json"):
        sequence = int(path.name.split("-", 1)[0])
        if sequence > maximum:
            path.unlink()


def _reseal_existing_tree(
    directory: Path,
    *,
    package_kind: str | None = None,
    reverse_records: bool = False,
) -> None:
    manifest_path = directory / "payload_manifest.json"
    seal_path = directory / "seal.json"
    manifest = _load_strict_json(manifest_path)
    if package_kind is not None:
        manifest["package_kind"] = package_kind
    records = list(manifest["ordered_payload_records"])
    if reverse_records:
        records.reverse()
    for record in records:
        payload = directory / record["relative_path"]
        record["size_bytes"] = payload.stat().st_size
        record["sha256"] = _sha(payload)
    manifest["ordered_payload_records"] = records
    manifest["payload_tree_sha256"] = _sha_bytes(_canonical_without_lf(records))
    _write_canonical(manifest_path, manifest)
    seal = _load_strict_json(seal_path)
    if package_kind is not None:
        seal["package_kind"] = package_kind
    seal["payload_manifest_size_bytes"] = manifest_path.stat().st_size
    seal["payload_manifest_sha256"] = _sha(manifest_path)
    seal["payload_tree_sha256"] = manifest["payload_tree_sha256"]
    _write_canonical(seal_path, seal)


def _prune_to_prefix(paths: dict[str, Path], prefix: str) -> None:
    audit = paths["audit"]
    if prefix == "SEALED_INPUT_ONLY":
        _remove_tree(audit)
        _remove_tree(paths["scan"])
        _remove_tree(paths["quarantine"])
    elif prefix == "COLLECTION_RECEIPT_ONLY":
        _remove_tree(audit / "receipts" / "batches")
        _remove_tree(audit / "events")
        _remove_tree(audit / "events_manifests")
        _remove_tree(paths["scan"])
        _remove_tree(paths["quarantine"])
    elif prefix == "BOTH_RECEIPTS_NO_EVENT":
        _remove_tree(audit / "events")
        _remove_tree(audit / "events_manifests")
        _remove_tree(paths["scan"])
        _remove_tree(paths["quarantine"])
    elif prefix == "SEQ1_ONLY":
        _remove_files_after(audit / "events", 1)
        _remove_files_after(audit / "events_manifests", 1)
        _remove_tree(paths["scan"])
        _remove_tree(paths["quarantine"])
    elif prefix in {"SEQ1_PLUS_SCAN_OUTPUT_NO_SEQ2", "SEQ1_PLUS_BATCH_QUARANTINE_NO_SEQ2"}:
        _remove_files_after(audit / "events", 1)
        _remove_files_after(audit / "events_manifests", 1)
    elif prefix == "SEQ2_NO_SEQ3":
        _remove_files_after(audit / "events", 2)
        _remove_files_after(audit / "events_manifests", 2)
    elif prefix == "SEQ3_COMPLETE":
        return
    else:  # pragma: no cover
        raise AssertionError(prefix)


@pytest.mark.parametrize(
    "prefix",
    [
        "SEALED_INPUT_ONLY",
        "COLLECTION_RECEIPT_ONLY",
        "BOTH_RECEIPTS_NO_EVENT",
        "SEQ1_ONLY",
        "SEQ1_PLUS_SCAN_OUTPUT_NO_SEQ2",
        "SEQ2_NO_SEQ3",
        "SEQ3_COMPLETE",
    ],
)
def test_recovery_each_ingested_prefix(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    plan: dict[str, Any],
    prefix: str,
) -> None:
    root = catnat_tmp_path / f"recover-{prefix.lower()}"
    _, control_path, run_id = _build_fixture(producer, root)
    first = _run(scanner, root, control_path)
    assert _verdict(first) == plan["verdicts"]["ingested"]
    paths = _paths(root, run_id)
    _prune_to_prefix(paths, prefix)
    second = _run(scanner, root, control_path)
    assert _verdict(second) == plan["verdicts"]["ingested"]
    _assert_event_chain(paths, "INGESTED")


def test_recovery_rejects_resealed_scan_parquet_application_metadata(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    plan: dict[str, Any],
) -> None:
    root = catnat_tmp_path / "scan-application-metadata"
    _, control_path, run_id = _build_fixture(producer, root)
    _run(scanner, root, control_path)
    paths = _paths(root, run_id)
    _prune_to_prefix(paths, "SEQ1_PLUS_SCAN_OUTPUT_NO_SEQ2")
    parquet_path = paths["scan"] / "safe_queries_preidentity.parquet"
    original_rows = pq.read_table(parquet_path).to_pylist()
    _rewrite_parquet_with_application_metadata(parquet_path, plan)
    assert pq.read_table(parquet_path).to_pylist() == original_rows
    _reseal_existing_tree(paths["scan"])
    _assert_stop(lambda: _run(scanner, root, control_path))


@pytest.mark.parametrize("receipt_kind", ["COLLECTION", "BATCH"])
def test_recovery_completes_partial_expected_receipt_directory_without_extras(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    plan: dict[str, Any],
    receipt_kind: str,
) -> None:
    root = catnat_tmp_path / f"partial-{receipt_kind.lower()}-receipt"
    _, control_path, run_id = _build_fixture(producer, root)
    first = _run(scanner, root, control_path)
    assert _verdict(first) == plan["verdicts"]["ingested"]
    paths = _paths(root, run_id)
    collection_dir = next(
        path
        for path in (paths["audit"] / "receipts" / "collections").iterdir()
        if path.is_dir()
    )
    batch_dir = next(
        path
        for path in (paths["audit"] / "receipts" / "batches").iterdir()
        if path.is_dir()
    )
    collection_id = collection_dir.name
    batch_id = batch_dir.name
    if receipt_kind == "COLLECTION":
        _prune_to_prefix(paths, "SEALED_INPUT_ONLY")
        partial = (
            paths["audit"] / "receipts" / "collections" / collection_id
        )
    else:
        _prune_to_prefix(paths, "COLLECTION_RECEIPT_ONLY")
        partial = paths["audit"] / "receipts" / "batches" / batch_id
    partial.mkdir(parents=True)
    current = partial
    while current != root:
        os.chmod(current, 0o700)
        current = current.parent

    second = _run(scanner, root, control_path)
    assert _verdict(second) == plan["verdicts"]["ingested"]
    receipt_root = paths["audit"] / "receipts"
    expected_ids = {
        "collections": collection_id,
        "batches": batch_id,
    }
    assert {path.name for path in receipt_root.iterdir()} == {
        "collections",
        "batches",
    }
    for plural, receipt_id in expected_ids.items():
        kind_root = receipt_root / plural
        assert [path.name for path in kind_root.iterdir()] == [receipt_id]
        assert [path.name for path in (kind_root / receipt_id).iterdir()] == [
            "receipt.json"
        ]
    _assert_event_chain(paths, "INGESTED")


@pytest.mark.parametrize("receipt_kind", ["COLLECTION", "BATCH"])
@pytest.mark.parametrize("conflict", ["DIRECTORY_EXTRA", "RECEIPT_CONTENT"])
def test_recovery_rejects_receipt_directory_extras_and_conflicts(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    receipt_kind: str,
    conflict: str,
) -> None:
    root = (
        catnat_tmp_path
        / f"receipt-{receipt_kind.lower()}-{conflict.lower()}"
    )
    _, control_path, run_id = _build_fixture(producer, root)
    _run(scanner, root, control_path)
    plural = "collections" if receipt_kind == "COLLECTION" else "batches"
    receipt = _only(
        _paths(root, run_id)["audit"] / "receipts" / plural,
        "receipt.json",
    )
    if conflict == "DIRECTORY_EXTRA":
        extra = receipt.parent / "unexpected"
        extra.write_bytes(b"forbidden")
        os.chmod(extra, 0o600)
    else:
        value = _load_strict_json(receipt)
        value["receipt_id"] = "a" * 64
        _write_canonical(receipt, value)
    _assert_stop(lambda: _run(scanner, root, control_path))


@pytest.mark.parametrize("receipt_kind", ["COLLECTION", "BATCH"])
def test_recovery_rejects_impossible_but_regex_compatible_receipt_timestamp(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    receipt_kind: str,
) -> None:
    root = catnat_tmp_path / f"receipt-{receipt_kind.lower()}-bad-date"
    _, control_path, run_id = _build_fixture(producer, root)
    _run(scanner, root, control_path)
    plural = "collections" if receipt_kind == "COLLECTION" else "batches"
    receipt = _only(
        _paths(root, run_id)["audit"] / "receipts" / plural,
        "receipt.json",
    )
    value = _load_strict_json(receipt)
    value["created_at_utc"] = "2026-99-99T25:61:61Z"
    _write_canonical(receipt, value)
    _assert_stop(lambda: _run(scanner, root, control_path))


@pytest.mark.parametrize("receipt_kind", ["COLLECTION", "BATCH"])
def test_recovery_rejects_unexpected_sibling_receipt_id(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    receipt_kind: str,
) -> None:
    root = catnat_tmp_path / f"receipt-{receipt_kind.lower()}-sibling"
    _, control_path, run_id = _build_fixture(producer, root)
    _run(scanner, root, control_path)
    plural = "collections" if receipt_kind == "COLLECTION" else "batches"
    kind_root = _paths(root, run_id)["audit"] / "receipts" / plural
    existing_ids = {path.name for path in kind_root.iterdir()}
    unexpected_id = "a" * 64
    if unexpected_id in existing_ids:
        unexpected_id = "b" * 64
    unexpected = kind_root / unexpected_id
    unexpected.mkdir(mode=0o700)
    assert {path.name for path in kind_root.iterdir()} == existing_ids | {
        unexpected_id
    }
    _assert_stop(lambda: _run(scanner, root, control_path))


def test_recovery_batch_quarantine_tree_without_seq2(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    plan: dict[str, Any],
) -> None:
    root = catnat_tmp_path / "recover-batch-quarantine"
    control_path, run_id = _build_and_repin_csv(
        producer, root, _batch_mutator("CSV_BOM_FORBIDDEN")
    )
    first = _run(scanner, root, control_path)
    assert _verdict(first) == plan["verdicts"]["quarantined"]
    paths = _paths(root, run_id)
    _prune_to_prefix(paths, "SEQ1_PLUS_BATCH_QUARANTINE_NO_SEQ2")
    second = _run(scanner, root, control_path)
    assert _verdict(second) == plan["verdicts"]["quarantined"]
    _assert_event_chain(paths, "QUARANTINED")


def test_fault_after_seq1_is_journalled_as_batch_and_collection_stopped(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = catnat_tmp_path / "fault-after-seq1"
    _, control_path, run_id = _build_fixture(producer, root)

    def injected_fault(**_kwargs: Any) -> dict[str, bytes]:
        raise scanner.ScannerStop("STOP injected after seq1")

    monkeypatch.setattr(scanner, "_build_scan_payloads", injected_fault)
    _assert_stop(lambda: _run(scanner, root, control_path))
    paths = _paths(root, run_id)
    events, generations = _audit_files(paths)
    assert len(events) == 3
    assert len(generations) == 3
    assert _load_strict_json(events[0])["new_state"] == "RECEIPTED"
    assert _load_strict_json(events[1])["new_state"] == "STOPPED"
    assert _load_strict_json(events[2])["new_state"] == "STOPPED"


def test_corrupt_journal_stops_without_any_append(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
) -> None:
    root = catnat_tmp_path / "corrupt-journal-no-append"
    _, control_path, run_id = _build_fixture(producer, root)
    _run(scanner, root, control_path)
    paths = _paths(root, run_id)
    generation_two = sorted(
        (paths["audit"] / "events_manifests").glob("00000002-*.json")
    )[0]
    generation_two.write_bytes(generation_two.read_bytes() + b" ")
    before = {
        path.relative_to(paths["audit"]).as_posix(): path.read_bytes()
        for path in paths["audit"].rglob("*")
        if path.is_file()
    }
    _assert_stop(lambda: _run(scanner, root, control_path))
    after = {
        path.relative_to(paths["audit"]).as_posix(): path.read_bytes()
        for path in paths["audit"].rglob("*")
        if path.is_file()
    }
    assert after == before


def test_unmanifested_orphan_is_preserved_and_ignored(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    plan: dict[str, Any],
) -> None:
    root = catnat_tmp_path / "orphan"
    _, control_path, run_id = _build_fixture(producer, root)
    result = _run(scanner, root, control_path)
    assert _verdict(result) == plan["verdicts"]["ingested"]
    paths = _paths(root, run_id)
    orphan = paths["audit"] / "events" / f"00000099-{'a' * 64}.json"
    orphan.write_bytes(_canonical({"orphan": True}))
    os.chmod(orphan, 0o600)
    rerun = _run(scanner, root, control_path)
    assert _verdict(rerun) == plan["verdicts"]["ingested"]
    assert orphan.read_bytes() == _canonical({"orphan": True})
    _assert_event_chain(paths, "INGESTED")


@pytest.mark.parametrize(
    "conflict",
    [
        "PAYLOAD_BYTE_MUTATED",
        "PAYLOAD_MANIFEST_SUBSTITUTED",
        "SEAL_SUBSTITUTED",
        "REFERENCED_EVENT_OR_GENERATION_HASH_CONFLICT",
    ],
)
def test_recovery_rejects_referenced_conflicts(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    conflict: str,
) -> None:
    root = catnat_tmp_path / f"conflict-{conflict.lower()}"
    _, control_path, run_id = _build_fixture(producer, root)
    _run(scanner, root, control_path)
    paths = _paths(root, run_id)
    if conflict == "PAYLOAD_BYTE_MUTATED":
        target = paths["scan"] / "scan_provenance.json"
    elif conflict == "PAYLOAD_MANIFEST_SUBSTITUTED":
        target = paths["scan"] / "payload_manifest.json"
    elif conflict == "SEAL_SUBSTITUTED":
        target = paths["scan"] / "seal.json"
    else:
        target = sorted((paths["audit"] / "events").glob("00000001-*.json"))[0]
    target.write_bytes(target.read_bytes() + b" ")
    _assert_stop(lambda: _run(scanner, root, control_path))


def test_exact_tree_rejects_added_payload(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
) -> None:
    root = catnat_tmp_path / "extra-payload"
    _, control_path, run_id = _build_fixture(producer, root)
    _run(scanner, root, control_path)
    paths = _paths(root, run_id)
    extra = paths["scan"] / "unexpected.txt"
    extra.write_text("unexpected", encoding="utf-8")
    os.chmod(extra, 0o600)
    _assert_stop(lambda: _run(scanner, root, control_path))


@pytest.mark.parametrize("conflict", ["PAYLOAD_RECORD_ORDER_CHANGED", "PACKAGE_KIND_MISMATCH"])
def test_recovery_rejects_well_formed_package_conflicts(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
    conflict: str,
) -> None:
    root = catnat_tmp_path / f"well-formed-{conflict.lower()}"
    _, control_path, run_id = _build_fixture(producer, root)
    _run(scanner, root, control_path)
    paths = _paths(root, run_id)
    _prune_to_prefix(paths, "SEQ1_PLUS_SCAN_OUTPUT_NO_SEQ2")
    _reseal_existing_tree(
        paths["scan"],
        package_kind="BATCH_QUARANTINE" if conflict == "PACKAGE_KIND_MISMATCH" else None,
        reverse_records=conflict == "PAYLOAD_RECORD_ORDER_CHANGED",
    )
    _assert_stop(lambda: _run(scanner, root, control_path))


def test_recovery_rejects_well_formed_scan_input_binding_mismatch(
    catnat_tmp_path: Path,
    producer: Any,
    scanner: Any,
) -> None:
    root = catnat_tmp_path / "input-binding-mismatch"
    _, control_path, run_id = _build_fixture(producer, root)
    _run(scanner, root, control_path)
    paths = _paths(root, run_id)
    _prune_to_prefix(paths, "SEQ1_PLUS_SCAN_OUTPUT_NO_SEQ2")

    provenance_path = paths["scan"] / "scan_provenance.json"
    provenance = _load_strict_json(provenance_path)
    provenance["sealed_input_seal_sha256"] = "0" * 64
    _write_canonical(provenance_path, provenance)

    integrity_path = paths["scan"] / "scan_integrity.json"
    integrity = _load_strict_json(integrity_path)
    integrity["logical_hashes"]["scan_provenance.json"] = _sha_bytes(
        _canonical_without_lf(provenance)
    )
    _write_canonical(integrity_path, integrity)
    _reseal_existing_tree(paths["scan"])

    _assert_stop(lambda: _run(scanner, root, control_path))


def test_rename_exclusive_syncs_both_source_and_destination_parents(
    catnat_tmp_path: Path,
    scanner: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = catnat_tmp_path / "rename-parent-sync"
    source_parent = root / "source"
    destination_parent = root / "destination"
    source_parent.mkdir(parents=True)
    destination_parent.mkdir()
    for directory in (root, source_parent, destination_parent):
        os.chmod(directory, 0o700)
    source = source_parent / "payload"
    source.write_bytes(b"payload")
    os.chmod(source, 0o600)

    class FakeRename:
        argtypes: Any = None
        restype: Any = None

        def __call__(self, *_args: Any) -> int:
            return 0

    class FakeLibC:
        renameatx_np = FakeRename()

    monkeypatch.setattr(
        scanner.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: FakeLibC(),
    )
    syncs: list[tuple[int, bool]] = []

    def record_sync(fd: int, *, full_required: bool) -> None:
        syncs.append((os.fstat(fd).st_ino, full_required))

    monkeypatch.setattr(scanner, "_sync_fd", record_sync)
    authority = scanner._RootFD(root)
    try:
        authority.rename_exclusive(
            "source/payload",
            "destination/payload",
        )
    finally:
        authority.close()
    assert syncs == [
        (source_parent.stat().st_ino, False),
        (destination_parent.stat().st_ino, False),
    ]
