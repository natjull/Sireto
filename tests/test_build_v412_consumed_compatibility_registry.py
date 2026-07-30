from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

import pyarrow.parquet as pq
import pyarrow as pa
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = REPO_ROOT / "config/v4_12_consumed_compatibility_registry_plan.json"
SCRIPT_PATH = (
    REPO_ROOT / "scripts/build_v412_consumed_compatibility_registry.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_v412_consumed_compatibility_registry",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys_modules_name = SPEC.name
sys.modules[sys_modules_name] = builder
SPEC.loader.exec_module(builder)
TEST_KEY = bytes.fromhex(
    "000102030405060708090a0b0c0d0e0f"
    "101112131415161718191a1b1c1d1e1f"
)
HEX_A = "a" * 64
HEX_B = "b" * 64


def _rows() -> list[dict[str, object]]:
    return [
        {
            "source_row_number": 1,
            "SITE": "  École   Élémentaire ",
            "CODE_POSTAL": "75001",
            "CODE_INSEE": "75101",
            "SERVICE ID": " abc-42 ",
            "COMMUNE": " Paris ",
            "SIRET": "12345678901234",
            "SITE_CLI_ADRESSE": " 1, Rue de l’Église ",
            "SITE_CLI_COMMUNE": "Paris",
        },
        {
            "source_row_number": 2,
            "SITE": "Société Test",
            "CODE_POSTAL": "69002",
            "CODE_INSEE": "69382",
            "SERVICE ID": "",
            "COMMUNE": "Lyon 2e",
            "SIRET": "١٢٣٤٥٦٧٨٩٠١٢٣٤",
            "SITE_CLI_ADRESSE": "10 rue Victor-Hugo",
            "SITE_CLI_COMMUNE": "Lyon",
        },
        {
            "source_row_number": 3,
            "SITE": "",
            "CODE_POSTAL": "",
            "CODE_INSEE": None,
            "SERVICE ID": "",
            "COMMUNE": None,
            "SIRET": None,
            "SITE_CLI_ADRESSE": None,
            "SITE_CLI_COMMUNE": "",
        },
    ]


def _tables() -> builder.RegistryTables:
    return builder.build_registry_tables(
        _rows(),
        hmac_key=TEST_KEY,
        challenge_source_rows={2},
        expected_rows=3,
        expected_empty_service_ids=2,
    )


def _spec(plan: dict[str, object]) -> dict[str, object]:
    return builder.build_specification(
        plan,
        builder_git_commit="test-commit",
        builder_source_sha256=HEX_A,
        tests_sha256=HEX_B,
        hmac_key_sha256=hashlib.sha256(TEST_KEY).hexdigest(),
    )


def _fixture_plan() -> dict[str, object]:
    plan, _raw = builder.load_plan(PLAN_PATH)
    fixture = deepcopy(plan)
    fixture["outputs"]["files"]["compatibility_rows.parquet"]["rows"] = 3
    fixture["outputs"]["files"]["fuzzy_historical_observations.parquet"][
        "row_bounds"
    ] = [4, 4]
    fixture["invariants"].update(
        {
            "challenge_rows": 1,
            "compatibility_rows": 3,
            "effective_consumed_rows": 3,
            "expected_empty_service_id": 2,
            "expected_fuzzy_observations_min": 4,
            "expected_fuzzy_observations_max": 4,
            "expected_nonempty_service_id": 1,
        }
    )
    return fixture


def _reseal_for_test(root: Path, build_id: str) -> None:
    records = builder._payload_records(root)
    manifest = {
        "schema_version": builder.PAYLOAD_MANIFEST_SCHEMA,
        "payload_files": records,
    }
    manifest_bytes = builder.canonical_json(manifest)
    (root / "payload_manifest.json").write_bytes(manifest_bytes)
    seal = {
        "schema_version": builder.SEAL_SCHEMA,
        "build_id": build_id,
        "build_spec_sha256": build_id,
        "payload_manifest_size_bytes": len(manifest_bytes),
        "payload_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "payload_logical_head_sha256": builder.logical_payload_head(records),
    }
    (root / "seal.json").write_bytes(builder.canonical_json(seal))
    os.chmod(root / "payload_manifest.json", 0o600)
    os.chmod(root / "seal.json", 0o600)


def test_committed_plan_is_canonical_and_all_golden_vectors_pass() -> None:
    plan, raw = builder.load_plan(PLAN_PATH)
    assert raw == builder.canonical_json(plan)
    builder.validate_runtime(plan)
    builder.validate_golden_vectors(plan)
    superscript = next(
        item
        for item in plan["golden_vectors"]["vectors"]
        if item["id"] == "ISDIGIT_NOT_ISDECIMAL_PIN"
    )
    assert superscript["expected"]["normalized_input_siret"] == "2" * 14
    assert builder.normalize_siret(superscript["input_siret"]) == "2" * 14


def test_fuzzy_emits_one_future_compatible_hash_per_distinct_city() -> None:
    observed = builder.fuzzy_singletons(_rows()[1])
    assert [(ordinal, city) for ordinal, city, _digest in observed] == [
        (0, "lyon"),
        (1, "lyon 2e"),
    ]
    future_lyon = {
        **_rows()[1],
        "COMMUNE": "Lyon",
        "SITE_CLI_COMMUNE": "Lyon",
    }
    future_lyon_2e = {
        **_rows()[1],
        "COMMUNE": "Lyon 2e",
        "SITE_CLI_COMMUNE": "Lyon 2e",
    }
    assert builder.fuzzy_singletons(future_lyon)[0][2] == observed[0][2]
    assert builder.fuzzy_singletons(future_lyon_2e)[0][2] == observed[1][2]
    assert len(builder.fuzzy_singletons(_rows()[2])) == 1
    assert builder.fuzzy_singletons(_rows()[2])[0][1] == ""


def test_tables_have_exact_schemas_hmac_multiplicities_and_no_cleartext() -> None:
    tables = _tables()
    assert tables.compatibility_rows.schema == builder.COMPATIBILITY_SCHEMA
    assert tables.fuzzy_observations.schema == builder.FUZZY_OBSERVATIONS_SCHEMA
    rows = tables.compatibility_rows.to_pylist()
    assert [row["consumption_reason"] for row in rows] == [
        "V411_HISTORICAL_CONSUMED",
        "V411_CHALLENGE_225",
        "V411_HISTORICAL_CONSUMED",
    ]
    assert [row["fuzzy_fingerprint_count"] for row in rows] == [1, 2, 1]
    assert rows[0]["service_id_lineage_hmac_sha256"] == (
        "37a12718157ce0e185788598178900658914a194b5156e7aebceaecfa9676489"
    )
    assert rows[1]["input_siret_lineage_hmac_sha256"] == (
        "fa31c3786c0a7eb7d54a4c092998c107953ae7030dd7d78fb258099654cf3e49"
    )
    assert sum(
        row["row_count"] for row in tables.masked_keyset.to_pylist()
    ) == 3
    assert sum(
        row["row_count"] for row in tables.fuzzy_keyset.to_pylist()
    ) == 4
    serialized = b"".join(
        builder.canonical_json(table.to_pylist())
        for table in (
            tables.compatibility_rows,
            tables.service_keyset,
            tables.input_siret_keyset,
        )
    )
    assert b"abc-42" not in serialized.lower()
    assert b"12345678901234" not in serialized


def test_short_duplicate_or_non_contiguous_inputs_fail_closed() -> None:
    with pytest.raises(builder.CompatibilityRegistryError, match="256 bits"):
        builder.build_registry_tables(_rows(), hmac_key=b"x" * 31)
    duplicate = _rows()
    duplicate[1]["source_row_number"] = 1
    with pytest.raises(builder.CompatibilityRegistryError, match="duplicate"):
        builder.build_registry_tables(duplicate, hmac_key=TEST_KEY)
    gap = _rows()
    gap[2]["source_row_number"] = 4
    with pytest.raises(builder.CompatibilityRegistryError, match="contiguous"):
        builder.build_registry_tables(gap, hmac_key=TEST_KEY)


def test_v411_registry_parity_and_consumed_unseen_partition_are_exact() -> None:
    source_rows = []
    for row in _rows():
        source_rows.append(
            {
                **row,
                "service_id_norm": builder.canonical_text(row["SERVICE ID"]),
                "input_siret_norm": builder.normalize_siret(row["SIRET"]),
                "row_fingerprint_sha256": builder.v411_row_fingerprint(row),
            }
        )
    source = pa.Table.from_pylist(source_rows)
    consumed = pa.table({"source_row_number": [1, 3]})
    unseen = pa.table({"source_row_number": [2]})
    assert builder._validate_v411_parity(
        _rows(), source, consumed, unseen
    ) == {2}
    drifted = source.to_pylist()
    drifted[0]["SITE"] = "substituted"
    with pytest.raises(builder.CompatibilityRegistryError, match="raw field drift"):
        builder._validate_v411_parity(
            _rows(),
            pa.Table.from_pylist(drifted),
            consumed,
            unseen,
        )
    with pytest.raises(builder.CompatibilityRegistryError, match="partition"):
        builder._validate_v411_parity(
            _rows(),
            source,
            consumed,
            pa.table({"source_row_number": [1, 2]}),
        )


def test_hmac_key_is_read_only_cloexec_single_link_and_hash_pinned(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "key"
    key_path.write_bytes(TEST_KEY)
    os.chmod(key_path, 0o600)
    read_fd = os.open(
        key_path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        provider = builder.FdHmacKeyProvider(
            key_id="SIRETO_V412_COMPATIBILITY_LINEAGE_HMAC_V1",
            descriptor=read_fd,
            expected_sha256=hashlib.sha256(TEST_KEY).hexdigest(),
        )
        assert provider.load(
            expected_key_id="SIRETO_V412_COMPATIBILITY_LINEAGE_HMAC_V1"
        ) == TEST_KEY
        with pytest.raises(builder.CompatibilityRegistryError, match="key ID"):
            provider.load(expected_key_id="OTHER")
        with pytest.raises(builder.CompatibilityRegistryError, match="hash"):
            builder.read_hmac_key_from_fd(read_fd, expected_sha256=HEX_A)
    finally:
        os.close(read_fd)
    writable_fd = os.open(key_path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
    try:
        with pytest.raises(builder.CompatibilityRegistryError, match="read-only"):
            builder.read_hmac_key_from_fd(
                writable_fd,
                expected_sha256=hashlib.sha256(TEST_KEY).hexdigest(),
            )
    finally:
        os.close(writable_fd)
    short_path = tmp_path / "short"
    short_path.write_bytes(b"x" * 31)
    os.chmod(short_path, 0o600)
    short_fd = os.open(short_path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        with pytest.raises(builder.CompatibilityRegistryError, match="256 bits"):
            builder.read_hmac_key_from_fd(short_fd, expected_sha256=None)
    finally:
        os.close(short_fd)


def test_pinned_reader_rejects_symlink_and_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"stable")
    snapshot = builder.read_pinned_file(
        source,
        expected_sha256=hashlib.sha256(b"stable").hexdigest(),
        expected_size=6,
        expected_uid=os.getuid(),
        expected_device=os.stat(source).st_dev,
    )
    assert snapshot.data == b"stable"
    with pytest.raises(builder.CompatibilityRegistryError, match="INPUT_DRIFT"):
        builder.read_pinned_file(
            source,
            expected_sha256=HEX_A,
            expected_size=6,
        )
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(OSError):
        builder.read_pinned_file(
            link,
            expected_sha256=hashlib.sha256(b"stable").hexdigest(),
            expected_size=6,
        )


def test_identity_pin_lock_is_canonical_and_path_exact(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"x")
    payload = {
        "schema_version": (
            "sireto-v4.12-consumed-compatibility-identity-pins-1"
        ),
        "files": {
            str(source): {
                "uid": os.getuid(),
                "device": os.stat(source).st_dev,
                "volume_uuid": "TEST-UUID",
            }
        },
        "output_root": {
            "uid": os.getuid(),
            "device": os.stat(tmp_path).st_dev,
            "volume_uuid": "TEST-UUID",
        },
    }
    lock = tmp_path / "identity-lock.json"
    lock.write_bytes(builder.canonical_json(payload))
    pins, output = builder.load_identity_pins(
        lock,
        expected_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
    )
    assert pins[str(source.absolute())].volume_uuid == "TEST-UUID"
    assert output.uid == os.getuid()
    lock.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(builder.CompatibilityRegistryError, match="canonical"):
        builder.load_identity_pins(
            lock,
            expected_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        )


def test_single_canonical_execution_lock_contains_all_execution_pins(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": builder.EXECUTION_LOCK_SCHEMA,
        "plan": {"path": str(PLAN_PATH), "sha256": HEX_A},
        "builder": {
            "git_commit": "deadbeef",
            "path": str(SCRIPT_PATH),
            "source_sha256": HEX_B,
        },
        "tests": {
            "path": str(Path(__file__).resolve()),
            "sha256": "c" * 64,
        },
        "hmac": {
            "key_id": "SIRETO_V412_COMPATIBILITY_LINEAGE_HMAC_V1",
            "key_sha256": "d" * 64,
        },
        "attempt_id": "attempt-locked",
        "identity": {
            "files": {
                str(PLAN_PATH): {
                    "uid": os.getuid(),
                    "device": PLAN_PATH.stat().st_dev,
                    "volume_uuid": "TEST-UUID",
                }
            },
            "output_root": {
                "uid": os.getuid(),
                "device": tmp_path.stat().st_dev,
                "volume_uuid": "TEST-UUID",
            },
        },
    }
    lock_path = tmp_path / "execution-lock.json"
    lock_path.write_bytes(builder.canonical_json(payload))
    lock = builder.load_execution_lock(
        lock_path,
        expected_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    )
    assert lock.attempt_id == "attempt-locked"
    assert lock.hmac_key_sha256 == "d" * 64
    parsed = builder._parse_args(
        [
            "--hmac-key-fd",
            "7",
            "--execution-lock",
            str(lock_path),
            "--execution-lock-sha256",
            hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        ]
    )
    assert not hasattr(parsed, "builder_source_sha256")
    payload["unexpected"] = True
    lock_path.write_bytes(builder.canonical_json(payload))
    with pytest.raises(builder.CompatibilityRegistryError, match="fields"):
        builder.load_execution_lock(
            lock_path,
            expected_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        )


def test_payload_is_byte_reproducible_private_and_non_self_referential(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan()
    spec = _spec(plan)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = builder.write_payload_tree(
        first,
        _tables(),
        plan=plan,
        build_spec=spec,
        require_fullfsync=False,
    )
    second_result = builder.write_payload_tree(
        second,
        _tables(),
        plan=plan,
        build_spec=spec,
        require_fullfsync=False,
    )
    assert first_result == second_result
    builder.compare_complete_trees(first, second)
    for name in builder.PAYLOAD_FILES + ("payload_manifest.json", "seal.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert stat.S_IMODE((first / name).stat().st_mode) == 0o600
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    manifest = json.loads((first / "payload_manifest.json").read_bytes())
    assert [item["relative_path"] for item in manifest["payload_files"]] == list(
        builder.PAYLOAD_FILES
    )
    assert "payload_manifest.json" not in {
        item["relative_path"] for item in manifest["payload_files"]
    }
    seal = json.loads((first / "seal.json").read_bytes())
    assert "seal_sha256" not in seal
    builder.validate_complete_tree(
        first,
        expected_build_id=first_result["build_id"],
        plan=plan,
        build_spec=spec,
    )
    for name, schema in builder.TABLE_SCHEMAS.items():
        parquet = pq.ParquetFile(first / name)
        assert parquet.schema_arrow == schema
        assert parquet.metadata.num_row_groups == 1
    (second / "seal.json").write_bytes(
        (second / "seal.json").read_bytes() + b"x"
    )
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="differs byte-for-byte",
    ):
        builder.compare_complete_trees(first, second)


def test_production_plan_refuses_three_row_fixture(tmp_path: Path) -> None:
    plan, _raw = builder.load_plan(PLAN_PATH)
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="written row count mismatch",
    ):
        builder.write_payload_tree(
            tmp_path / "must-not-seal",
            _tables(),
            plan=plan,
            build_spec=_spec(plan),
            require_fullfsync=False,
        )


def test_tree_validation_detects_extra_or_modified_payload(tmp_path: Path) -> None:
    plan = _fixture_plan()
    spec = _spec(plan)
    root = tmp_path / "tree"
    result = builder.write_payload_tree(
        root,
        _tables(),
        plan=plan,
        build_spec=spec,
        require_fullfsync=False,
    )
    (root / "extra").write_bytes(b"x")
    with pytest.raises(builder.CompatibilityRegistryError, match="tree mismatch"):
        builder.validate_complete_tree(
            root,
            expected_build_id=result["build_id"],
            plan=plan,
            build_spec=spec,
        )


def test_resealed_forged_keyset_is_rejected_by_exact_counter(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan()
    spec = _spec(plan)
    root = tmp_path / "forged"
    result = builder.write_payload_tree(
        root,
        _tables(),
        plan=plan,
        build_spec=spec,
        require_fullfsync=False,
    )
    keyset_path = root / "siret_masked_keyset.parquet"
    rows = pq.read_table(keyset_path).to_pylist()
    rows[0]["siret_masked_fingerprint_sha256"] = "0" * 64
    forged = pa.Table.from_pylist(rows, schema=builder.MASKED_KEYSET_SCHEMA)
    keyset_path.unlink()
    pq.write_table(
        forged,
        keyset_path,
        version="2.6",
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        data_page_version="1.0",
        row_group_size=65536,
        store_schema=True,
    )
    os.chmod(keyset_path, 0o600)
    _reseal_for_test(root, result["build_id"])
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="keyset exact content mismatch",
    ):
        builder.validate_complete_tree(
            root,
            expected_build_id=result["build_id"],
            plan=plan,
            build_spec=spec,
        )


def test_attempt_receipt_event_chain_is_immutable_and_detects_tampering(
    tmp_path: Path,
) -> None:
    attempt = builder.create_attempt_receipt(
        tmp_path / "attempts",
        attempt_id="attempt-1",
        plan_sha256=HEX_A,
        input_pins_sha256=HEX_B,
        require_fullfsync=False,
    )
    first = builder.append_attempt_event(
        attempt,
        event_type="FIRST",
        fields={"value": 1},
        require_fullfsync=False,
    )
    second = builder.append_attempt_event(
        attempt,
        event_type="SECOND",
        fields={"value": 2},
        require_fullfsync=False,
    )
    assert first != second
    builder.validate_attempt_chain(attempt)
    with pytest.raises(FileExistsError):
        builder.create_attempt_receipt(
            tmp_path / "attempts",
            attempt_id="attempt-1",
            plan_sha256=HEX_A,
            input_pins_sha256=HEX_B,
            require_fullfsync=False,
        )
    event = sorted((attempt / "events").iterdir())[0]
    os.chmod(event, 0o600)
    event.write_bytes(event.read_bytes() + b"x")
    with pytest.raises(builder.CompatibilityRegistryError, match="event"):
        builder.validate_attempt_chain(attempt)


def test_orphan_event_is_preserved_but_never_applied(
    tmp_path: Path,
) -> None:
    attempt = builder.create_attempt_receipt(
        tmp_path / "attempts",
        attempt_id="orphan-1",
        plan_sha256=HEX_A,
        input_pins_sha256=HEX_B,
        require_fullfsync=False,
    )
    builder.append_attempt_event(
        attempt,
        event_type="COMPLETE",
        fields={"value": 1},
        require_fullfsync=False,
    )
    complete = builder.validate_attempt_chain(attempt)
    previous = hashlib.sha256(
        builder.canonical_json(complete.complete_events[-1])
    ).hexdigest()
    orphan = {
        "schema_version": builder.EVENT_SCHEMA,
        "sequence": 2,
        "event_type": "ORPHAN",
        "previous_event_sha256": previous,
        "fields": {"must_not_apply": True},
    }
    orphan_bytes = builder.canonical_json(orphan)
    orphan_hash = hashlib.sha256(orphan_bytes).hexdigest()
    (attempt / "events" / f"00000002-{orphan_hash}.json").write_bytes(
        orphan_bytes
    )
    os.chmod(attempt / "events" / f"00000002-{orphan_hash}.json", 0o600)
    state = builder.validate_attempt_chain(attempt)
    assert len(state.complete_events) == 1
    assert len(state.orphan_event_paths) == 1
    with pytest.raises(builder.CompatibilityRegistryError, match="orphan"):
        builder.append_attempt_event(
            attempt,
            event_type="MUST_STOP",
            fields={},
            require_fullfsync=False,
        )


def test_recovery_validates_complete_tree_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _fixture_plan()
    spec = _spec(plan)
    staging = tmp_path / "staging"
    result = builder.write_payload_tree(
        staging,
        _tables(),
        plan=plan,
        build_spec=spec,
        require_fullfsync=False,
    )
    destination = tmp_path / "published"
    plan_hash = builder.sha256_bytes(builder.canonical_json(plan))
    pins_hash = builder.sha256_bytes(
        builder.canonical_json(plan["inputs"])
    )
    attempt = builder.create_attempt_receipt(
        tmp_path / "attempts",
        attempt_id="recover-1",
        plan_sha256=plan_hash,
        input_pins_sha256=pins_hash,
        require_fullfsync=False,
    )
    builder.append_attempt_event(
        attempt,
        event_type="TREE_VALIDATED",
        fields=result,
        require_fullfsync=False,
    )
    promoted: list[tuple[Path, Path]] = []

    def fake_rename(
        source: Path,
        target: Path,
        *,
        expected_root_device: int,
        expected_root_inode: int,
    ) -> None:
        assert (expected_root_device, expected_root_inode) == (
            staging.stat().st_dev,
            staging.stat().st_ino,
        )
        promoted.append((source, target))

    monkeypatch.setattr(builder, "_rename_exclusive", fake_rename)
    builder.recover_validated_tree(
        staging,
        destination,
        expected_build_id=result["build_id"],
        plan=plan,
        build_spec=spec,
        attempt_root=attempt,
        expected_attempt_id="recover-1",
        expected_plan_sha256=plan_hash,
        expected_input_pins_sha256=pins_hash,
    )
    assert promoted == [(staging, destination)]
    (staging / "integrity.json").unlink()
    with pytest.raises(builder.CompatibilityRegistryError):
        builder.recover_validated_tree(
            staging,
            destination,
            expected_build_id=result["build_id"],
            plan=plan,
            build_spec=spec,
            attempt_root=attempt,
            expected_attempt_id="recover-1",
            expected_plan_sha256=plan_hash,
            expected_input_pins_sha256=pins_hash,
        )


def test_promotion_rejects_staging_root_substituted_after_validation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    actual = source.stat()
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="identity changed",
    ):
        builder._rename_exclusive(
            source,
            destination,
            expected_root_device=actual.st_dev,
            expected_root_inode=actual.st_ino + 1,
        )
    assert source.is_dir()
    assert not destination.exists()


def test_retained_output_dirfd_defeats_output_root_path_swap(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan()
    spec = _spec(plan)
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    output_fd = os.open(
        output,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    retained = tmp_path / "retained-original"
    try:
        output.rename(retained)
        output.mkdir(mode=0o700)
        result = builder.write_payload_tree(
            output / "staging",
            _tables(),
            plan=plan,
            build_spec=spec,
            require_fullfsync=False,
            parent_fd=output_fd,
        )
        assert (retained / "staging" / "seal.json").is_file()
        assert not (output / "staging").exists()
        validation = builder.validate_complete_tree(
            output / "staging",
            expected_build_id=result["build_id"],
            plan=plan,
            build_spec=spec,
            parent_fd=output_fd,
        )
        builder._rename_exclusive_at(
            output_fd,
            "staging",
            "published",
            expected_root_device=validation.root_device,
            expected_root_inode=validation.root_inode,
        )
        assert (retained / "published" / "seal.json").is_file()
        assert not (output / "published").exists()
    finally:
        os.close(output_fd)


def test_existing_attempt_recovers_closed_tree_without_source_rebuild(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan()
    spec = _spec(plan)
    build_id = builder.build_id_for_spec(spec)
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    output_fd = os.open(
        output,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    plan_hash = builder.sha256_bytes(builder.canonical_json(plan))
    pins_hash = builder.sha256_bytes(
        builder.canonical_json(plan["inputs"])
    )
    attempt = builder.create_attempt_receipt(
        tmp_path / "attempts",
        attempt_id="resume-1",
        plan_sha256=plan_hash,
        input_pins_sha256=pins_hash,
        require_fullfsync=False,
    )
    staging = output / f".tmp-{build_id}-resume-1-primary"
    publication = builder.write_payload_tree(
        staging,
        _tables(),
        plan=plan,
        build_spec=spec,
        require_fullfsync=False,
        parent_fd=output_fd,
    )
    builder.append_attempt_event(
        attempt,
        event_type="TREE_VALIDATED",
        fields=publication,
        require_fullfsync=False,
    )
    try:
        destination = builder.recover_existing_attempt(
            output_root=output,
            output_root_fd=output_fd,
            attempt_root=attempt,
            attempt_id="resume-1",
            plan=plan,
            plan_sha256=plan_hash,
            build_spec=spec,
            input_pins_sha256=pins_hash,
            require_fullfsync=False,
        )
        assert destination == output / build_id
        assert destination.is_dir()
        assert not staging.exists()
        # A second invocation validates and returns the already promoted tree:
        # no O_EXCL rebuild and no source input is part of this API.
        assert builder.recover_existing_attempt(
            output_root=output,
            output_root_fd=output_fd,
            attempt_root=attempt,
            attempt_id="resume-1",
            plan=plan,
            plan_sha256=plan_hash,
            build_spec=spec,
            input_pins_sha256=pins_hash,
            require_fullfsync=False,
        ) == destination
    finally:
        os.close(output_fd)


def test_private_umask_sets_0077_and_restores_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_umask(value: int) -> int:
        calls.append(value)
        return 0o022

    monkeypatch.setattr(builder.os, "umask", fake_umask)
    with builder.private_umask():
        pass
    assert calls == [0o077, 0o022]
