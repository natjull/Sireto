from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import plistlib
import stat
import subprocess
import sys

import pyarrow.parquet as pq
import pyarrow as pa
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN_PATH = REPO_ROOT / "config/v4_12_consumed_compatibility_registry_plan.json"
EXECUTION_LOCK_PATH = (
    REPO_ROOT / "config/v4_12_consumed_compatibility_execution_lock.json"
)
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


def test_keychain_provider_pins_logical_id_hash_and_never_discloses_secret(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = b"do-not-log-this-key-material-0123456789"
    expected = hashlib.sha256(secret).hexdigest()
    monkeypatch.setattr(
        builder,
        "_copy_keychain_generic_password_no_ui",
        lambda: secret,
    )
    provider = builder.MacOSKeychainHmacKeyProvider(
        logical_key_id="LOGICAL-ID",
        expected_sha256=expected,
    )
    assert provider.load(expected_key_id="LOGICAL-ID") == secret
    wrong_hash = builder.MacOSKeychainHmacKeyProvider(
        logical_key_id="LOGICAL-ID",
        expected_sha256="0" * 64,
    )
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="hash differs",
    ) as failure:
        wrong_hash.load(expected_key_id="LOGICAL-ID")
    output = capsys.readouterr()
    observed = output.out + output.err + str(failure.value)
    assert secret.decode() not in observed


def test_keychain_provider_rejects_wrong_key_id_before_keychain_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accessed = False

    def forbidden() -> bytes:
        nonlocal accessed
        accessed = True
        raise AssertionError("Keychain must not be read for a wrong key ID")

    monkeypatch.setattr(
        builder,
        "_copy_keychain_generic_password_no_ui",
        forbidden,
    )
    provider = builder.MacOSKeychainHmacKeyProvider(
        logical_key_id="LOCK-ID",
        expected_sha256="0" * 64,
    )
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="plan and execution lock",
    ):
        provider.load(expected_key_id="PLAN-ID")
    assert not accessed


def test_keychain_boundary_has_no_cli_and_authentication_ui_is_fail_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_subprocess(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Keychain boundary must not spawn a process")

    monkeypatch.setattr(builder.subprocess, "run", forbidden_subprocess)
    source = inspect.getsource(builder._copy_keychain_generic_password_no_ui)
    assert "subprocess" not in source
    assert "kSecUseAuthenticationUIFail" in source

    def auth_denied() -> bytes:
        builder._stop(
            "Keychain HMAC unavailable without UI (OSStatus -25308)"
        )

    monkeypatch.setattr(
        builder,
        "_copy_keychain_generic_password_no_ui",
        auth_denied,
    )
    provider = builder.MacOSKeychainHmacKeyProvider(
        logical_key_id="ID",
        expected_sha256="0" * 64,
    )
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="unavailable without UI",
    ):
        provider.load(expected_key_id="ID")


def test_run_boundary_revalidates_injected_provider_and_zeroizes_failures() -> None:
    class IgnoringProvider:
        def __init__(self, value: bytearray) -> None:
            self.value = value

        def load(self, *, expected_key_id: str) -> bytearray:
            return self.value

    valid = bytearray(b"v" * 32)
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="plan and execution lock",
    ):
        builder.validate_provider_key(
            IgnoringProvider(valid),
            plan_key_id="PLAN",
            lock_key_id="LOCK",
            lock_key_sha256=hashlib.sha256(valid).hexdigest(),
        )
    short = bytearray(b"s" * 31)
    with pytest.raises(builder.CompatibilityRegistryError, match="256 bits"):
        builder.validate_provider_key(
            IgnoringProvider(short),
            plan_key_id="ID",
            lock_key_id="ID",
            lock_key_sha256=hashlib.sha256(short).hexdigest(),
        )
    assert short == bytearray(len(short))
    wrong = bytearray(b"w" * 32)
    with pytest.raises(builder.CompatibilityRegistryError, match="hash differs"):
        builder.validate_provider_key(
            IgnoringProvider(wrong),
            plan_key_id="ID",
            lock_key_id="ID",
            lock_key_sha256="0" * 64,
        )
    assert wrong == bytearray(len(wrong))


def test_keychain_cf_resources_query_and_abi_are_exact_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Function:
        def __init__(self, implementation: object) -> None:
            self.implementation = implementation
            self.argtypes: object = "unset"
            self.restype: object = "unset"

        def __call__(self, *args: object) -> object:
            return self.implementation(*args)  # type: ignore[operator]

    released: list[int] = []
    pairs: list[tuple[int, int]] = []
    strings = iter([201, 202])
    secret = (builder.ctypes.c_uint8 * 32)(*range(32))

    class Core:
        CFDictionaryCreateMutable = Function(lambda *_args: 101)
        CFDictionarySetValue = Function(
            lambda query, key, value: pairs.append((int(key), int(value)))
        )
        CFStringCreateWithCString = Function(
            lambda *_args: next(strings)
        )
        CFGetTypeID = Function(lambda value: 77 if value == 303 else 0)
        CFDataGetTypeID = Function(lambda: 77)
        CFDataGetLength = Function(lambda value: 32 if value == 303 else -1)
        CFDataGetBytePtr = Function(
            lambda _value: builder.ctypes.cast(
                secret,
                builder.ctypes.POINTER(builder.ctypes.c_uint8),
            )
        )
        CFRelease = Function(lambda value: released.append(int(value)))

    class Security:
        @staticmethod
        def copy(_query: int, output: object) -> int:
            output._obj.value = 303
            return 0

        SecItemCopyMatching = Function(copy)

    constants = {
        "kSecClass": 1,
        "kSecClassGenericPassword": 2,
        "kSecAttrService": 3,
        "kSecAttrAccount": 4,
        "kSecReturnData": 5,
        "kCFBooleanTrue": 6,
        "kSecMatchLimit": 7,
        "kSecMatchLimitOne": 8,
        "kSecUseAuthenticationUI": 9,
        "kSecUseAuthenticationUIFail": 10,
    }
    requested: list[str] = []

    def constant(_library: object, name: str) -> builder.ctypes.c_void_p:
        requested.append(name)
        return builder.ctypes.c_void_p(constants[name])

    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        builder,
        "_load_keychain_frameworks",
        lambda: (Security(), Core()),
    )
    monkeypatch.setattr(builder, "_framework_constant", constant)
    assert builder._copy_keychain_generic_password_no_ui() == bytearray(
        range(32)
    )
    assert pairs == [
        (1, 2),
        (3, 201),
        (4, 202),
        (5, 6),
        (7, 8),
        (9, 10),
    ]
    assert requested == list(constants)
    assert released == [303, 201, 202, 101]
    assert Core.CFDictionarySetValue.restype is None
    assert Core.CFRelease.restype is None
    assert Core.CFDataGetTypeID.argtypes == []


def test_keychain_cf_resources_are_released_on_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Function:
        def __init__(self, implementation: object) -> None:
            self.implementation = implementation

        def __call__(self, *args: object) -> object:
            return self.implementation(*args)  # type: ignore[operator]

    released: list[int] = []
    strings = iter([201, 202])

    class Core:
        CFDictionaryCreateMutable = Function(lambda *_args: 101)
        CFDictionarySetValue = Function(lambda *_args: None)
        CFStringCreateWithCString = Function(lambda *_args: next(strings))
        CFGetTypeID = Function(lambda _value: 0)
        CFDataGetTypeID = Function(lambda: 0)
        CFDataGetLength = Function(lambda _value: 0)
        CFDataGetBytePtr = Function(lambda _value: None)
        CFRelease = Function(lambda value: released.append(int(value)))

    class Security:
        SecItemCopyMatching = Function(lambda _query, _output: -25308)

    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        builder,
        "_load_keychain_frameworks",
        lambda: (Security(), Core()),
    )
    monkeypatch.setattr(
        builder,
        "_framework_constant",
        lambda _library, name: builder.ctypes.c_void_p(
            {
                "kSecClass": 1,
                "kSecClassGenericPassword": 2,
                "kSecAttrService": 3,
                "kSecAttrAccount": 4,
                "kSecReturnData": 5,
                "kCFBooleanTrue": 6,
                "kSecMatchLimit": 7,
                "kSecMatchLimitOne": 8,
                "kSecUseAuthenticationUI": 9,
                "kSecUseAuthenticationUIFail": 10,
            }[name]
        ),
    )
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="OSStatus -25308",
    ):
        builder._copy_keychain_generic_password_no_ui()
    assert released == [201, 202, 101]


def test_keychain_cf_partial_string_allocation_releases_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Function:
        def __init__(self, implementation: object) -> None:
            self.implementation = implementation

        def __call__(self, *args: object) -> object:
            return self.implementation(*args)  # type: ignore[operator]

    released: list[int] = []
    strings = iter([201, 0])

    class Core:
        CFDictionaryCreateMutable = Function(lambda *_args: 101)
        CFDictionarySetValue = Function(lambda *_args: None)
        CFStringCreateWithCString = Function(lambda *_args: next(strings))
        CFGetTypeID = Function(lambda _value: 0)
        CFDataGetTypeID = Function(lambda: 0)
        CFDataGetLength = Function(lambda _value: 0)
        CFDataGetBytePtr = Function(lambda _value: None)
        CFRelease = Function(lambda value: released.append(int(value)))

    class Security:
        SecItemCopyMatching = Function(
            lambda _query, _output: pytest.fail(
                "API must not run after allocation failure"
            )
        )

    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        builder,
        "_load_keychain_frameworks",
        lambda: (Security(), Core()),
    )
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="allocate pinned Keychain locator",
    ):
        builder._copy_keychain_generic_password_no_ui()
    assert released == [201, 101]


def test_failed_key_validation_precedes_attempt_and_historical_source_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _fixture_plan()
    plan["build"]["output_root"] = str(tmp_path / "output")
    plan["outputs"]["attempts_root"] = str(tmp_path / "attempts")
    plan_bytes = builder.canonical_json(plan)
    plan_path = Path("/private/test/plan.json")
    tests_path = Path("/private/test/tests.py")
    contract_path = Path(plan["contract"]["path"])
    builder_path = Path(builder.__file__).absolute()
    allowed_before_key = {
        str(plan_path.absolute()),
        str(contract_path.absolute()),
        str(builder_path),
        str(tests_path.absolute()),
    }
    observed_reads: list[str] = []
    pin = builder.IdentityPin(uid=os.getuid(), device=1, volume_uuid="UUID")
    lock = builder.ExecutionLock(
        plan_path=plan_path,
        plan_sha256="0" * 64,
        builder_git_commit="a" * 40,
        builder_source_sha256="1" * 64,
        tests_path=tests_path,
        tests_sha256="2" * 64,
        hmac_key_id=plan["hmac_lineage"]["key_id"],
        hmac_key_sha256="3" * 64,
        attempt_id="attempt",
        identity_pins={path: pin for path in allowed_before_key},
        output_identity_pin=pin,
    )

    def snapshot(path: Path, **_kwargs: object) -> builder.FileSnapshot:
        normalized = str(path.absolute())
        observed_reads.append(normalized)
        assert normalized in allowed_before_key
        data = plan_bytes if normalized == str(plan_path.absolute()) else b"x"
        return builder.FileSnapshot(
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
            device=1,
            inode=1,
            uid=os.getuid(),
            nlink=1,
        )

    class DeniedProvider:
        def load(self, *, expected_key_id: str) -> bytes:
            assert expected_key_id == plan["hmac_lineage"]["key_id"]
            raise builder.CompatibilityRegistryError("KEY_DENIED")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("private attempt/source I/O occurred before key")

    monkeypatch.setattr(builder, "read_pinned_file", snapshot)
    monkeypatch.setattr(builder, "verify_identity_pin", lambda *_args: None)
    monkeypatch.setattr(builder, "load_plan", lambda _path: (plan, plan_bytes))
    monkeypatch.setattr(builder, "validate_runtime", lambda _plan: None)
    monkeypatch.setattr(builder, "validate_golden_vectors", lambda _plan: None)
    monkeypatch.setattr(
        builder,
        "validate_audited_git_state",
        lambda **_kwargs: "head",
    )
    monkeypatch.setattr(
        builder,
        "_mkdirs_anchored",
        lambda path, **_kwargs: Path(path).mkdir(
            parents=True, exist_ok=True, mode=0o700
        ),
    )
    monkeypatch.setattr(builder, "create_attempt_receipt", forbidden)
    monkeypatch.setattr(builder, "_read_private_regular", forbidden)
    monkeypatch.setattr(builder, "parse_historical_csv", forbidden)
    with pytest.raises(builder.CompatibilityRegistryError, match="KEY_DENIED"):
        builder.run_build(
            execution_lock=lock,
            key_provider=DeniedProvider(),
            require_fullfsync=False,
        )
    assert set(observed_reads) == allowed_before_key


def test_existing_attempt_recovery_uses_neither_keychain_nor_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _fixture_plan()
    output = tmp_path / "output"
    attempts = tmp_path / "attempts"
    plan["build"]["output_root"] = str(output)
    plan["outputs"]["attempts_root"] = str(attempts)
    plan_bytes = builder.canonical_json(plan)
    plan_path = Path("/private/test/recovery-plan.json")
    tests_path = Path("/private/test/recovery-tests.py")
    control_paths = {
        str(plan_path.absolute()),
        str(Path(plan["contract"]["path"]).absolute()),
        str(Path(builder.__file__).absolute()),
        str(tests_path.absolute()),
    }
    pin = builder.IdentityPin(uid=os.getuid(), device=1, volume_uuid="UUID")
    lock = builder.ExecutionLock(
        plan_path=plan_path,
        plan_sha256="0" * 64,
        builder_git_commit="a" * 40,
        builder_source_sha256="1" * 64,
        tests_path=tests_path,
        tests_sha256="2" * 64,
        hmac_key_id="INTENTIONALLY-UNAVAILABLE-DURING-RECOVERY",
        hmac_key_sha256="3" * 64,
        attempt_id="existing",
        identity_pins={path: pin for path in control_paths},
        output_identity_pin=pin,
    )

    def snapshot(path: Path, **_kwargs: object) -> builder.FileSnapshot:
        normalized = str(path.absolute())
        assert normalized in control_paths
        data = plan_bytes if normalized == str(plan_path.absolute()) else b"x"
        return builder.FileSnapshot(
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
            device=1,
            inode=1,
            uid=os.getuid(),
            nlink=1,
        )

    class ForbiddenProvider:
        def load(self, *, expected_key_id: str) -> bytearray:
            raise AssertionError("recovery accessed Keychain")

    recovered = output / "recovered"
    recovery_calls: list[dict[str, object]] = []

    def recover(**kwargs: object) -> Path:
        recovery_calls.append(kwargs)
        return recovered

    def mkdir(path: Path, **_kwargs: object) -> None:
        Path(path).mkdir(parents=True, exist_ok=True, mode=0o700)
        if Path(path) == attempts:
            (attempts / lock.attempt_id).mkdir(mode=0o700)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("recovery accessed fresh receipt or source")

    monkeypatch.setattr(builder, "read_pinned_file", snapshot)
    monkeypatch.setattr(builder, "verify_identity_pin", lambda *_args: None)
    monkeypatch.setattr(builder, "load_plan", lambda _path: (plan, plan_bytes))
    monkeypatch.setattr(builder, "validate_runtime", lambda _plan: None)
    monkeypatch.setattr(builder, "validate_golden_vectors", lambda _plan: None)
    monkeypatch.setattr(
        builder,
        "validate_audited_git_state",
        lambda **_kwargs: "head",
    )
    monkeypatch.setattr(builder, "_mkdirs_anchored", mkdir)
    monkeypatch.setattr(builder, "recover_existing_attempt", recover)
    monkeypatch.setattr(builder, "create_attempt_receipt", forbidden)
    monkeypatch.setattr(builder, "parse_historical_csv", forbidden)
    assert builder.run_build(
        execution_lock=lock,
        key_provider=ForbiddenProvider(),
        require_fullfsync=False,
    ) == recovered
    assert len(recovery_calls) == 1


def test_fresh_attempt_orders_key_receipt_source_and_zeroizes_after_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _fixture_plan()
    output = tmp_path / "output"
    attempts = tmp_path / "attempts"
    plan["build"]["output_root"] = str(output)
    plan["outputs"]["attempts_root"] = str(attempts)
    plan_bytes = builder.canonical_json(plan)
    plan_path = Path("/private/test/fresh-plan.json")
    tests_path = Path("/private/test/fresh-tests.py")
    control_paths = {
        str(plan_path.absolute()),
        str(Path(plan["contract"]["path"]).absolute()),
        str(Path(builder.__file__).absolute()),
        str(tests_path.absolute()),
    }
    private_paths = {
        str(Path(plan["inputs"]["historical_raw"]["path"]).absolute()),
        *(
            str(
                Path(plan["inputs"]["v411_registry"][name]["path"]).absolute()
            )
            for name in ("source_registry", "consumed", "unseen", "manifest")
        ),
        *(
            str(Path(spec["manifest_path"]).absolute())
            for spec in plan["inputs"]["challenge_225"].values()
        ),
    }
    all_paths = control_paths | private_paths
    pin = builder.IdentityPin(uid=os.getuid(), device=1, volume_uuid="UUID")
    secret = bytearray(b"k" * 32)
    lock = builder.ExecutionLock(
        plan_path=plan_path,
        plan_sha256="0" * 64,
        builder_git_commit="a" * 40,
        builder_source_sha256="1" * 64,
        tests_path=tests_path,
        tests_sha256="2" * 64,
        hmac_key_id=plan["hmac_lineage"]["key_id"],
        hmac_key_sha256=hashlib.sha256(secret).hexdigest(),
        attempt_id="fresh",
        identity_pins={path: pin for path in all_paths},
        output_identity_pin=pin,
    )
    order: list[str] = []
    expected_tables = _tables()

    def snapshot(path: Path, **_kwargs: object) -> builder.FileSnapshot:
        normalized = str(path.absolute())
        assert normalized in all_paths
        if normalized in private_paths:
            order.append("source")
        data = (
            plan_bytes
            if normalized == str(plan_path.absolute())
            else b"{}\n"
        )
        return builder.FileSnapshot(
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
            device=1,
            inode=1,
            uid=os.getuid(),
            nlink=1,
        )

    class Provider:
        def load(self, *, expected_key_id: str) -> bytearray:
            order.append("key")
            return secret

    def receipt(root: Path, **_kwargs: object) -> Path:
        order.append("receipt")
        attempt = root / lock.attempt_id
        attempt.mkdir(mode=0o700)
        return attempt

    def projected(*_args: object, **kwargs: object) -> builder.RegistryTables:
        order.append("project")
        assert kwargs["hmac_key"] == bytearray(b"k" * 32)
        return expected_tables

    monkeypatch.setattr(builder, "read_pinned_file", snapshot)
    monkeypatch.setattr(builder, "verify_identity_pin", lambda *_args: None)
    monkeypatch.setattr(builder, "load_plan", lambda _path: (plan, plan_bytes))
    monkeypatch.setattr(builder, "validate_runtime", lambda _plan: None)
    monkeypatch.setattr(builder, "validate_golden_vectors", lambda _plan: None)
    monkeypatch.setattr(
        builder,
        "validate_audited_git_state",
        lambda **_kwargs: "head",
    )
    monkeypatch.setattr(
        builder,
        "_mkdirs_anchored",
        lambda path, **_kwargs: Path(path).mkdir(
            parents=True, exist_ok=True, mode=0o700
        ),
    )
    monkeypatch.setattr(builder, "create_attempt_receipt", receipt)
    monkeypatch.setattr(builder, "append_attempt_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "parse_historical_csv", lambda _data: _rows())
    monkeypatch.setattr(builder, "_load_pinned_parquet", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(builder, "_validate_v411_parity", lambda *_args: {2})
    monkeypatch.setattr(builder, "build_registry_tables", projected)
    monkeypatch.setattr(
        builder,
        "write_payload_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            builder.CompatibilityRegistryError("STOP_AFTER_PROJECTION")
        ),
    )
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="STOP_AFTER_PROJECTION",
    ):
        builder.run_build(
            execution_lock=lock,
            key_provider=Provider(),
            require_fullfsync=False,
        )
    assert order.index("key") < order.index("receipt") < order.index("source")
    assert order.index("source") < order.index("project")
    assert secret == bytearray(32)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="read-only Keychain integration requires macOS",
)
def test_keychain_native_read_only_integration_no_prompt() -> None:
    plan, plan_bytes = builder.load_plan(PLAN_PATH)
    lock_bytes = EXECUTION_LOCK_PATH.read_bytes()
    lock = json.loads(lock_bytes)
    assert lock_bytes == builder.canonical_json(lock)
    assert lock["plan"]["sha256"] == hashlib.sha256(plan_bytes).hexdigest()
    assert lock["hmac"]["key_id"] == plan["hmac_lineage"]["key_id"]
    provider = builder.MacOSKeychainHmacKeyProvider(
        logical_key_id=lock["hmac"]["key_id"],
        expected_sha256=lock["hmac"]["key_sha256"],
    )
    secret = provider.load(expected_key_id=plan["hmac_lineage"]["key_id"])
    try:
        assert isinstance(secret, bytearray)
        assert len(secret) >= 32
        assert hashlib.sha256(secret).hexdigest() == lock["hmac"]["key_sha256"]
    finally:
        builder.zeroize_secret(secret)
    assert secret == bytearray(len(secret))


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


def test_macos_volume_uuid_uses_anchored_fd_device_and_sanitized_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"stable")
    info = source.stat()
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        builder,
        "_darwin_fstatfs",
        lambda _fd: ("/dev/disk99s1", "/Volumes/Trusted"),
    )

    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((arguments, kwargs))
        payload = {
            "DeviceNode": "/dev/disk99s1",
            "MountPoint": "/Volumes/Trusted",
            "VolumeUUID": "abc-def",
        }
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=plistlib.dumps(payload),
            stderr=b"",
        )

    monkeypatch.setattr(builder.subprocess, "run", run)
    monkeypatch.setenv("PATH", str(tmp_path / "attacker-bin"))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker.git"))
    assert builder.macos_volume_uuid(
        source,
        expected_device=info.st_dev,
        expected_inode=info.st_ino,
    ) == "ABC-DEF"
    assert len(calls) == 1
    arguments, kwargs = calls[0]
    assert arguments == [
        "/usr/sbin/diskutil",
        "info",
        "-plist",
        "/dev/disk99s1",
    ]
    assert kwargs["env"] == {"LANG": "C", "LC_ALL": "C"}
    assert kwargs["close_fds"] is True
    assert "PATH" not in kwargs["env"]
    assert "GIT_DIR" not in kwargs["env"]


def test_macos_volume_uuid_rejects_symlink_and_prelookup_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"stable")
    link = tmp_path / "link"
    link.symlink_to(source)
    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")
    with pytest.raises(OSError):
        builder.macos_volume_uuid(link)
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("diskutil must not run after identity drift")

    monkeypatch.setattr(builder.subprocess, "run", forbidden)
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="identity changed before",
    ):
        builder.macos_volume_uuid(
            source,
            expected_device=source.stat().st_dev,
            expected_inode=source.stat().st_ino + 1,
        )
    assert not called


@pytest.mark.parametrize(
    ("mounted_device", "payload_override", "message"),
    [
        ("relative-device", {}, "unexpected mounted device"),
        (
            "/dev/disk99s1",
            {"DeviceNode": "/dev/disk88s1"},
            "diskutil device mismatch",
        ),
        (
            "/dev/disk99s1",
            {"MountPoint": "/Volumes/Substituted"},
            "diskutil mount-point mismatch",
        ),
    ],
)
def test_macos_volume_uuid_rejects_untrusted_mount_or_diskutil_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mounted_device: str,
    payload_override: dict[str, str],
    message: str,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"x")
    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        builder,
        "_darwin_fstatfs",
        lambda _fd: (mounted_device, "/Volumes/Trusted"),
    )
    payload = {
        "DeviceNode": "/dev/disk99s1",
        "MountPoint": "/Volumes/Trusted",
        "VolumeUUID": "UUID",
        **payload_override,
    }
    monkeypatch.setattr(
        builder.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments,
            0,
            stdout=plistlib.dumps(payload),
            stderr=b"",
        ),
    )
    with pytest.raises(builder.CompatibilityRegistryError, match=message):
        builder.macos_volume_uuid(source)


def test_macos_volume_uuid_detects_mount_change_during_diskutil_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"x")
    identities = iter(
        [
            ("/dev/disk99s1", "/Volumes/Trusted"),
            ("/dev/disk88s1", "/Volumes/Substituted"),
        ]
    )
    monkeypatch.setattr(builder.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        builder,
        "_darwin_fstatfs",
        lambda _fd: next(identities),
    )
    payload = {
        "DeviceNode": "/dev/disk99s1",
        "MountPoint": "/Volumes/Trusted",
        "VolumeUUID": "UUID",
    }
    monkeypatch.setattr(
        builder.subprocess,
        "run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments,
            0,
            stdout=plistlib.dumps(payload),
            stderr=b"",
        ),
    )
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="mount changed during",
    ):
        builder.macos_volume_uuid(source)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="read-only integration check requires macOS",
)
def test_macos_volume_uuid_real_internal_and_catnat_data() -> None:
    external = Path("/Volumes/CATNAT_DATA")
    if not external.is_dir():
        pytest.skip("CATNAT_DATA is not mounted")
    internal_uuid = builder.macos_volume_uuid(SCRIPT_PATH)
    external_uuid = builder.macos_volume_uuid(external)
    assert internal_uuid
    assert external_uuid
    assert internal_uuid != external_uuid


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
            "--execution-lock",
            str(lock_path),
            "--execution-lock-sha256",
            hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        ]
    )
    assert not hasattr(parsed, "builder_source_sha256")
    assert not hasattr(parsed, "hmac_key_fd")
    with pytest.raises(SystemExit):
        builder._parse_args(
            [
                "--hmac-key-fd",
                "7",
                "--execution-lock",
                str(lock_path),
                "--execution-lock-sha256",
                hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            ]
        )
    payload["unexpected"] = True
    lock_path.write_bytes(builder.canonical_json(payload))
    with pytest.raises(builder.CompatibilityRegistryError, match="fields"):
        builder.load_execution_lock(
            lock_path,
            expected_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        )


def test_audited_git_blobs_match_and_later_head_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("config", "user.name", "SIRETO Test")
    git("config", "user.email", "sireto-test@example.invalid")
    script = repo / "scripts" / "builder.py"
    tests = repo / "tests" / "test_builder.py"
    script.parent.mkdir()
    tests.parent.mkdir()
    script.write_bytes(b"audited builder\n")
    tests.write_bytes(b"audited tests\n")
    git("add", "scripts/builder.py", "tests/test_builder.py")
    git("commit", "-q", "-m", "audited code")
    audited_commit = git("rev-parse", "HEAD").stdout.decode().strip()
    (repo / "execution-lock.json").write_text("{}\n", encoding="utf-8")
    git("add", "execution-lock.json")
    git("commit", "-q", "-m", "later lock commit")
    later_head = git("rev-parse", "HEAD").stdout.decode().strip()
    assert later_head != audited_commit
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    marker = tmp_path / "fake-git-was-called"
    fake_git.write_text(
        f"#!/bin/sh\n/usr/bin/touch '{marker}'\nexit 99\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o700)
    lure = tmp_path / "lure.git"
    subprocess.run(
        ["/usr/bin/git", "init", "--bare", "-q", str(lure)],
        check=True,
    )
    monkeypatch.setenv("PATH", str(fake_bin))
    monkeypatch.setenv("GIT_DIR", str(lure))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "false-worktree"))
    assert builder.validate_audited_git_state(
        repo_root=repo,
        audited_commit=audited_commit,
        artifact_hashes={
            script: hashlib.sha256(script.read_bytes()).hexdigest(),
            tests: hashlib.sha256(tests.read_bytes()).hexdigest(),
        },
    ) == later_head
    assert not marker.exists()


def test_audited_git_blob_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["/usr/bin/git", "init", "-q"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "config", "user.name", "SIRETO Test"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "config",
            "user.email",
            "sireto-test@example.invalid",
        ],
        cwd=repo,
        check=True,
    )
    artifact = repo / "builder.py"
    artifact.write_bytes(b"committed bytes\n")
    subprocess.run(
        ["/usr/bin/git", "add", "builder.py"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "commit", "-q", "-m", "audited"],
        cwd=repo,
        check=True,
    )
    commit = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="blob hash mismatch",
    ):
        builder.validate_audited_git_state(
            repo_root=repo,
            audited_commit=commit,
            artifact_hashes={artifact: "0" * 64},
        )


def test_audited_git_ignores_chained_replace_refs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "SIRETO Test")
    git("config", "user.email", "sireto-test@example.invalid")
    artifact = repo / "builder.py"
    artifact.write_bytes(b"audited bytes\n")
    audited_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    git("add", "builder.py")
    git("commit", "-q", "-m", "audited")
    audited_commit = git("rev-parse", "HEAD")
    artifact.write_bytes(b"replacement bytes\n")
    git("add", "builder.py")
    git("commit", "-q", "-m", "replacement one")
    replacement_one = git("rev-parse", "HEAD")
    artifact.write_bytes(b"second replacement bytes\n")
    git("add", "builder.py")
    git("commit", "-q", "-m", "replacement two")
    replacement_two = git("rev-parse", "HEAD")
    git("replace", audited_commit, replacement_one)
    git("replace", replacement_one, replacement_two)
    # The real worktree verification is a separate pinned-FD check in the
    # runner. Restore audited bytes here to isolate Git object semantics.
    artifact.write_bytes(b"audited bytes\n")
    assert builder.validate_audited_git_state(
        repo_root=repo,
        audited_commit=audited_commit,
        artifact_hashes={artifact: audited_hash},
    ) == replacement_two


def test_annotated_tag_object_is_rejected_as_audited_commit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["/usr/bin/git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["/usr/bin/git", "config", "user.name", "SIRETO Test"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "config",
            "user.email",
            "sireto-test@example.invalid",
        ],
        cwd=repo,
        check=True,
    )
    artifact = repo / "builder.py"
    artifact.write_bytes(b"builder\n")
    subprocess.run(
        ["/usr/bin/git", "add", "builder.py"], cwd=repo, check=True
    )
    subprocess.run(
        ["/usr/bin/git", "commit", "-q", "-m", "commit"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "tag", "-a", "audited", "-m", "annotated"],
        cwd=repo,
        check=True,
    )
    tag_object = subprocess.run(
        ["/usr/bin/git", "rev-parse", "audited"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="not a commit",
    ):
        builder.validate_audited_git_state(
            repo_root=repo,
            audited_commit=tag_object,
            artifact_hashes={
                artifact: hashlib.sha256(artifact.read_bytes()).hexdigest()
            },
        )


def test_git_commit_injection_outside_path_and_shallow_history_fail_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()

    def source_git(*arguments: str) -> str:
        return subprocess.run(
            ["/usr/bin/git", *arguments],
            cwd=source,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    source_git("init", "-q")
    source_git("config", "user.name", "SIRETO Test")
    source_git("config", "user.email", "sireto-test@example.invalid")
    artifact = source / "builder.py"
    artifact.write_bytes(b"old\n")
    source_git("add", "builder.py")
    source_git("commit", "-q", "-m", "old")
    old_commit = source_git("rev-parse", "HEAD")
    artifact.write_bytes(b"new\n")
    source_git("add", "builder.py")
    source_git("commit", "-q", "-m", "new")
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="full lowercase SHA-1",
    ):
        builder.validate_audited_git_state(
            repo_root=source,
            audited_commit="a" * 39 + ";",
            artifact_hashes={artifact: hashlib.sha256(b"new\n").hexdigest()},
        )
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside\n")
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="outside repository",
    ):
        builder.validate_audited_git_state(
            repo_root=source,
            audited_commit=source_git("rev-parse", "HEAD"),
            artifact_hashes={
                outside: hashlib.sha256(outside.read_bytes()).hexdigest()
            },
        )
    shallow = tmp_path / "shallow"
    subprocess.run(
        [
            "/usr/bin/git",
            "clone",
            "-q",
            "--depth",
            "1",
            source.as_uri(),
            str(shallow),
        ],
        check=True,
    )
    with pytest.raises(
        builder.CompatibilityRegistryError,
        match="not a commit",
    ):
        builder.validate_audited_git_state(
            repo_root=shallow,
            audited_commit=old_commit,
            artifact_hashes={
                shallow / "builder.py": hashlib.sha256(b"new\n").hexdigest()
            },
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
