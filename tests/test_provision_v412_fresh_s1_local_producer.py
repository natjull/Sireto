from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import importlib.metadata
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


REPOSITORY = Path(__file__).resolve().parents[1]
SUBJECT_PATH = (
    REPOSITORY / "scripts/provision_v412_fresh_s1_local_producer.py"
)
SPEC = importlib.util.spec_from_file_location("s1_provisioner", SUBJECT_PATH)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def _canonical(value: object, *, final_lf: bool = True) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return raw + (b"\n" if final_lf else b"")


def _bundle(tmp_path: Path) -> subject.ControlBundle:
    plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s1_local_producer_authority_plan.json"
        ).read_bytes()
    )
    plan["paths"]["root"] = str(tmp_path / "authority")
    plan_raw = _canonical(plan)
    implementation = {
        "git_commit": "a" * 40,
        "provisioner_path": "scripts/provision_v412_fresh_s1_local_producer.py",
        "provisioner_sha256": hashlib.sha256(
            SUBJECT_PATH.read_bytes()
        ).hexdigest(),
        "tests_path": "tests/test_provision_v412_fresh_s1_local_producer.py",
        "tests_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "plan_path": "config/v4_12_fresh_s1_local_producer_authority_plan.json",
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "contract_path": plan["authorities"]["contract"]["path"],
        "contract_sha256": plan["authorities"]["contract"]["sha256"],
    }
    runtime = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable_path": str(Path(sys.executable).resolve()),
        "python_executable_sha256": hashlib.sha256(
            Path(sys.executable).resolve().read_bytes()
        ).hexdigest(),
        "cryptography_version": importlib.metadata.version("cryptography"),
        "security_framework_path": (
            "/System/Library/Frameworks/Security.framework/Security"
        ),
        "corefoundation_framework_path": (
            "/System/Library/Frameworks/CoreFoundation.framework/"
            "CoreFoundation"
        ),
        "os_build": platform.release(),
    }
    keychain_policy = {
        "item_class": "GENERIC_PASSWORD",
        "service": plan["keychain"]["service"],
        "account": plan["keychain"]["account"],
        "label": plan["producer"]["producer_key_id"],
        "binding_attribute": "KSECATTRGENERIC_CLAIM_SHA256",
        "synchronizable": False,
        "accessible": "AFTER_FIRST_UNLOCK_THIS_DEVICE_ONLY",
        "data_protection_keychain": True,
        "authentication_ui": "FAIL",
        "secret_length_bytes": 32,
    }
    lock = {
        "schema_version": plan["schemas"]["execution_lock"]["schema_version"],
        "purpose": "LOCAL_PRODUCER_AUTHORITY_WITHOUT_CRM",
        "plan_path": (
            "config/v4_12_fresh_s1_local_producer_authority_plan.json"
        ),
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "contract_path": plan["authorities"]["contract"]["path"],
        "contract_sha256": plan["authorities"]["contract"]["sha256"],
        "implementation": implementation,
        "runtime": runtime,
        "keychain_policy": keychain_policy,
        "expected_uid": os.getuid(),
        "volume_device": tmp_path.stat().st_dev,
        "volume_uuid": "",
        "output_root": plan["paths"]["root"],
        "logical_time_utc": plan["identity"]["logical_time_utc"],
        "authorization_path": plan["paths"]["authorization"],
    }
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        lock["volume_uuid"] = subject.volume_uuid_for_fd(parent_fd)
    finally:
        os.close(parent_fd)
    lock_raw = _canonical(lock)
    authorization = {
        "schema_version": plan["schemas"]["authorization"]["schema_version"],
        "authorization": "AUTHORIZED_S1_LOCAL_PRODUCER_PROVISION",
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "execution_lock_sha256": hashlib.sha256(lock_raw).hexdigest(),
        "implementation_commit": implementation["git_commit"],
        "logical_time_utc": plan["identity"]["logical_time_utc"],
    }
    return subject.ControlBundle(
        plan,
        plan_raw,
        lock,
        lock_raw,
        authorization,
        _canonical(authorization),
    )


def _run(
    tmp_path: Path,
    *,
    backend: subject.MemoryKeychain | None = None,
    checkpoint=lambda _name: None,
) -> tuple[dict, subject.MemoryKeychain, subject.ControlBundle]:
    bundle = _bundle(tmp_path)
    keychain = backend or subject.MemoryKeychain()
    values = iter([bytes(range(32)), bytes(range(32, 64))])
    receipt = subject.provision(
        bundle,
        keychain,
        trusted_parent=tmp_path,
        root=Path(bundle.plan["paths"]["root"]),
        random32=lambda: next(values),
        checkpoint=checkpoint,
    )
    return receipt, keychain, bundle


def _write_claim(
    tmp_path: Path, bundle: subject.ControlBundle, raw: bytes
) -> None:
    store = subject.AnchoredStore(
        tmp_path, Path(bundle.plan["paths"]["root"])
    )
    try:
        store.create_claim(raw)
    finally:
        store.close()


def test_rfc8032_vector_one() -> None:
    seed = bytearray.fromhex(
        "9d61b19deffd5a60ba844af492ec2cc4"
        "4449c5697b326919703bac031cae7f60"
    )
    public = subject._public_from_seed(seed)
    signature = subject._sign(seed, b"")
    assert public.hex() == (
        "d75a980182b10ab7d54bfed3c964073a"
        "0ee172f3daa62325af021a68f707511a"
    )
    assert signature.hex() == (
        "e5564300c360ac729086e2cc806e828a"
        "84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46b"
        "d25bf5f0595bbe24655141438e7a100b"
    )


def test_happy_path_builds_signed_closed_tree(tmp_path: Path) -> None:
    receipt, keychain, bundle = _run(tmp_path)
    root = Path(bundle.plan["paths"]["root"])
    claim_raw = (root / "claims/provision.claim.json").read_bytes()
    claim = json.loads(claim_raw)
    assert claim_raw == _canonical(claim)
    nonce = base64.b64decode(claim["attempt_binding_nonce_base64"])
    assert len(nonce) == 32
    assert hashlib.sha256(nonce).hexdigest() == claim[
        "attempt_binding_nonce_sha256"
    ]
    assert keychain.add_calls == 1
    assert keychain.read_calls == 2
    assert keychain.item is not None
    assert keychain.item.claim_sha256_raw == hashlib.sha256(claim_raw).digest()
    authority = root / "authorities" / receipt["authority_id"]
    assert {path.name for path in authority.iterdir()} == {
        "ledger_genesis.json",
        "producer_authority_payload.json",
        "producer_authority_seal.json",
        "provision_receipt.json",
    }
    payload = json.loads((authority / "producer_authority_payload.json").read_bytes())
    genesis = json.loads((authority / "ledger_genesis.json").read_bytes())
    unsigned = dict(genesis)
    signature = base64.b64decode(unsigned.pop("signature_base64"))
    public = base64.b64decode(payload["public_key_base64"])
    Ed25519PublicKey.from_public_bytes(public).verify(
        signature, _canonical(unsigned, final_lf=False)
    )
    assert payload["producer_export_ledger_head_sha256"] == hashlib.sha256(
        _canonical(unsigned, final_lf=False)
    ).hexdigest()
    assert payload["ledger_genesis_sha256"] == hashlib.sha256(
        (authority / "ledger_genesis.json").read_bytes()
    ).hexdigest()
    assert receipt["terminal_state"] == "PROVISIONED"
    for directory in (root, root / "claims", root / "authorities", authority):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for path in [root / "claims/provision.claim.json", *authority.iterdir()]:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_idempotent_receipt_does_not_touch_keychain_or_random(
    tmp_path: Path,
) -> None:
    first, keychain, bundle = _run(tmp_path)
    before_reads, before_adds = keychain.read_calls, keychain.add_calls
    second = subject.provision(
        bundle,
        keychain,
        trusted_parent=tmp_path,
        root=Path(bundle.plan["paths"]["root"]),
        random32=lambda: pytest.fail("random forbidden"),
    )
    assert second == first
    assert (keychain.read_calls, keychain.add_calls) == (
        before_reads,
        before_adds,
    )


def test_new_claim_never_adopts_existing_item(tmp_path: Path) -> None:
    keychain = subject.MemoryKeychain(
        subject.KeychainRecord(bytearray(b"x" * 32), b"y" * 32)
    )
    with pytest.raises(subject.ProvisionError, match="FOREIGN_ITEM_NEW_CLAIM"):
        _run(tmp_path, backend=keychain)
    assert keychain.add_calls == 0


def test_existing_claim_with_missing_item_creates_key(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    root = Path(bundle.plan["paths"]["root"])
    nonce = b"n" * 32
    claim = subject._claim_value(bundle, nonce)
    _write_claim(tmp_path, bundle, _canonical(claim))
    keychain = subject.MemoryKeychain()
    receipt = subject.provision(
        bundle,
        keychain,
        trusted_parent=tmp_path,
        root=root,
        random32=lambda: b"s" * 32,
    )
    assert receipt["terminal_state"] == "PROVISIONED"
    assert keychain.add_calls == 1


def test_existing_claim_only_accepts_exact_binding(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    root = Path(bundle.plan["paths"]["root"])
    claim_raw = _canonical(subject._claim_value(bundle, b"n" * 32))
    _write_claim(tmp_path, bundle, claim_raw)
    wrong = subject.MemoryKeychain(
        subject.KeychainRecord(bytearray(b"s" * 32), b"x" * 32)
    )
    with pytest.raises(subject.ProvisionError, match="FOREIGN_ITEM_BINDING"):
        subject.provision(
            bundle, wrong, trusted_parent=tmp_path, root=root
        )
    exact = subject.MemoryKeychain(
        subject.KeychainRecord(
            bytearray(b"s" * 32), hashlib.sha256(claim_raw).digest()
        )
    )
    receipt = subject.provision(
        bundle,
        exact,
        trusted_parent=tmp_path,
        root=root,
        random32=lambda: pytest.fail("random forbidden"),
    )
    assert receipt["terminal_state"] == "PROVISIONED"
    assert exact.add_calls == 0


def test_duplicate_add_stops_without_success_receipt(tmp_path: Path) -> None:
    keychain = subject.MemoryKeychain(duplicate_on_add=True)
    with pytest.raises(subject.ProvisionError, match="KEYCHAIN_ADD_DUPLICATE"):
        _run(tmp_path, backend=keychain)
    root = tmp_path / "authority"
    assert not list((root / "authorities").iterdir())


@pytest.mark.parametrize(
    "checkpoint",
    [
        "CLAIM_DURABLE",
        "KEYCHAIN_QUERIED",
        "SEED_GENERATED",
        "KEYCHAIN_ADDED",
        "ledger_genesis.json_DURABLE",
        "producer_authority_payload.json_DURABLE",
        "producer_authority_seal.json_DURABLE",
        "provision_receipt.json_DURABLE",
    ],
)
def test_crash_recovery_converges_to_one_authority(
    tmp_path: Path, checkpoint: str
) -> None:
    backend = subject.MemoryKeychain()
    bundle = _bundle(tmp_path)
    values = iter([b"n" * 32, b"s" * 32])

    def crash(name: str) -> None:
        if name == checkpoint:
            raise subject.InjectedCrash(name)

    with pytest.raises(subject.InjectedCrash):
        subject.provision(
            bundle,
            backend,
            trusted_parent=tmp_path,
            root=Path(bundle.plan["paths"]["root"]),
            random32=lambda: next(values),
            checkpoint=crash,
        )
    result = subject.provision(
        bundle,
        backend,
        trusted_parent=tmp_path,
        root=Path(bundle.plan["paths"]["root"]),
        random32=lambda: b"r" * 32,
    )
    assert result["terminal_state"] == "PROVISIONED"
    authorities = list(
        (Path(bundle.plan["paths"]["root"]) / "authorities").iterdir()
    )
    assert len(authorities) == 1
    assert backend.add_calls == 1


def test_two_concurrent_launchers_converge_to_one_authority(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)

    class AtomicMemoryKeychain(subject.MemoryKeychain):
        def __init__(self) -> None:
            super().__init__()
            self.lock = threading.Lock()

        def copy_item(self):
            with self.lock:
                return super().copy_item()

        def add_item(self, *, seed, claim_sha256_raw):
            with self.lock:
                return super().add_item(
                    seed=seed, claim_sha256_raw=claim_sha256_raw
                )

    backend = AtomicMemoryKeychain()
    barrier = threading.Barrier(2)

    def launch(fill: int):
        calls = 0

        def random32() -> bytes:
            nonlocal calls
            calls += 1
            if calls == 1:
                barrier.wait(timeout=5)
            return bytes([fill + calls]) * 32

        try:
            return subject.provision(
                bundle,
                backend,
                trusted_parent=tmp_path,
                root=Path(bundle.plan["paths"]["root"]),
                random32=random32,
            )
        except subject.ProvisionError as exc:
            return exc

    original_umask = os.umask(0o022)
    os.umask(original_umask)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(launch, (10, 20)))
    finally:
        os.umask(original_umask)
    successes = [
        value for value in outcomes if isinstance(value, dict)
    ]
    assert len(successes) == 1
    assert len(outcomes) - len(successes) == 1
    root = Path(bundle.plan["paths"]["root"])
    assert len(list((root / "authorities").iterdir())) == 1
    assert len(list((root / "claims").iterdir())) == 1
    assert backend.add_calls == 1
    assert backend.item is not None
    authority = next((root / "authorities").iterdir())
    assert (authority / "provision_receipt.json").exists()


def test_claim_logical_time_must_equal_plan_before_keychain(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    claim = subject._claim_value(bundle, b"n" * 32)
    claim["logical_time_utc"] = "2026-07-30T00:00:00Z"
    _write_claim(tmp_path, bundle, _canonical(claim))
    backend = subject.MemoryKeychain()
    with pytest.raises(subject.ProvisionError, match="CLAIM_LOGICAL_TIME"):
        subject.provision(
            bundle,
            backend,
            trusted_parent=tmp_path,
            root=Path(bundle.plan["paths"]["root"]),
        )
    assert backend.read_calls == backend.add_calls == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("terminal_state", "REVIEW"),
        ("reason_code", "NOT_OK"),
    ],
)
def test_receipt_terminal_contract_is_checked_before_keychain(
    tmp_path: Path, field: str, value: str
) -> None:
    _receipt, backend, bundle = _run(tmp_path)
    authority = next(
        (Path(bundle.plan["paths"]["root"]) / "authorities").iterdir()
    )
    receipt_path = authority / "provision_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt[field] = value
    receipt_path.write_bytes(_canonical(receipt))
    reads = backend.read_calls
    with pytest.raises(subject.ProvisionError):
        subject.provision(
            bundle,
            backend,
            trusted_parent=tmp_path,
            root=Path(bundle.plan["paths"]["root"]),
        )
    assert backend.read_calls == reads


@pytest.mark.parametrize(
    "filename",
    [
        "ledger_genesis.json",
        "producer_authority_payload.json",
        "producer_authority_seal.json",
    ],
)
def test_corrupt_public_tree_stops_before_keychain(
    tmp_path: Path, filename: str
) -> None:
    _receipt, keychain, bundle = _run(tmp_path)
    authority = next(
        (Path(bundle.plan["paths"]["root"]) / "authorities").iterdir()
    )
    path = authority / filename
    path.chmod(0o600)
    with path.open("ab") as stream:
        stream.write(b"x")
    reads = keychain.read_calls
    with pytest.raises(subject.ProvisionError):
        subject.provision(
            bundle,
            keychain,
            trusted_parent=tmp_path,
            root=Path(bundle.plan["paths"]["root"]),
        )
    assert keychain.read_calls == reads


def test_keychain_operation_contracts_are_exact(tmp_path: Path) -> None:
    plan = _bundle(tmp_path).plan
    seed = bytes(range(32))
    claim_hash = bytes(range(32, 64))
    add = subject.MacOSDataProtectionKeychain.add_contract(
        plan, seed, claim_hash
    )
    assert set(add) == {
        "kSecClass",
        "kSecAttrService",
        "kSecAttrAccount",
        "kSecAttrLabel",
        "kSecAttrGeneric",
        "kSecAttrSynchronizable",
        "kSecAttrAccessible",
        "kSecUseDataProtectionKeychain",
        "kSecValueData",
    }
    assert add["kSecUseDataProtectionKeychain"] is True
    assert add["kSecAttrGeneric"] == claim_hash
    assert add["kSecValueData"] == seed
    query = subject.MacOSDataProtectionKeychain.query_contract(plan)
    assert query == plan["keychain"]["secitemcopymatching_query_exact"]
    assert query["kSecUseDataProtectionKeychain"] is True
    assert query["kSecUseAuthenticationUI"] == "kSecUseAuthenticationUIFail"


def test_native_backend_uses_exact_query_and_projects_secret(
    tmp_path: Path,
) -> None:
    plan = _bundle(tmp_path).plan

    class API:
        def __init__(self) -> None:
            self.query = None
            self.required = None

        def copy_matching(self, query, required):
            self.query = query
            self.required = required
            projection = dict(required)
            projection["kSecAttrGeneric"] = b"g" * 32
            projection["kSecValueData"] = bytearray(b"s" * 32)
            return subject.NativeCopyResult(0, projection)

        def add(self, attributes):
            pytest.fail("add forbidden")

    api = API()
    backend = subject.MacOSDataProtectionKeychain(plan, api)
    item = backend.copy_item()
    assert item is not None
    assert item.claim_sha256_raw == b"g" * 32
    assert item.seed == bytearray(b"s" * 32)
    assert api.query == plan["keychain"]["secitemcopymatching_query_exact"]
    assert set(api.required) == set(
        plan["keychain"][
            "secitemcopymatching_result_policy"
        ]["required_persisted_attributes_verified_exactly"]
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (0, "ADDED"),
        (subject.ERR_SEC_DUPLICATE_ITEM, "DUPLICATE"),
    ],
)
def test_native_backend_add_contract_and_status(
    tmp_path: Path, status: int, expected: str
) -> None:
    plan = _bundle(tmp_path).plan

    class API:
        attributes = None

        def copy_matching(self, query, required):
            pytest.fail("copy forbidden")

        def add(self, attributes):
            self.attributes = dict(attributes)
            self.attributes["kSecValueData"] = bytes(
                attributes["kSecValueData"]
            )
            return status

    api = API()
    backend = subject.MacOSDataProtectionKeychain(plan, api)
    result = backend.add_item(
        seed=bytearray(b"s" * 32), claim_sha256_raw=b"g" * 32
    )
    assert result == expected
    assert api.attributes == (
        subject.MacOSDataProtectionKeychain.add_contract(
            plan, b"s" * 32, b"g" * 32
        )
    )
    assert api.attributes["kSecUseDataProtectionKeychain"] is True
    assert "kSecUseAuthenticationUI" not in api.attributes


def test_native_backend_copy_failures_are_closed(tmp_path: Path) -> None:
    plan = _bundle(tmp_path).plan

    class API:
        def __init__(self, result):
            self.result = result

        def copy_matching(self, query, required):
            return self.result

        def add(self, attributes):
            pytest.fail("add forbidden")

    missing = subject.MacOSDataProtectionKeychain(
        plan,
        API(subject.NativeCopyResult(subject.ERR_SEC_ITEM_NOT_FOUND, None)),
    )
    assert missing.copy_item() is None
    bad_status = subject.MacOSDataProtectionKeychain(
        plan, API(subject.NativeCopyResult(-1, None))
    )
    with pytest.raises(subject.ProvisionError, match="COPY_STATUS"):
        bad_status.copy_item()
    required = dict(
        plan["keychain"][
            "secitemcopymatching_result_policy"
        ]["required_persisted_attributes_verified_exactly"]
    )
    required["kSecAttrGeneric"] = b"g" * 31
    required["kSecValueData"] = bytearray(b"s" * 32)
    bad_length = subject.MacOSDataProtectionKeychain(
        plan, API(subject.NativeCopyResult(0, required))
    )
    with pytest.raises(subject.ProvisionError, match="BINDING_LENGTH"):
        bad_length.copy_item()


def test_native_backend_rejects_projection_shape_value_seed_and_add_status(
    tmp_path: Path,
) -> None:
    plan = _bundle(tmp_path).plan
    base = dict(
        plan["keychain"][
            "secitemcopymatching_result_policy"
        ]["required_persisted_attributes_verified_exactly"]
    )
    base["kSecAttrGeneric"] = b"g" * 32
    base["kSecValueData"] = bytearray(b"s" * 32)

    class API:
        def __init__(self, projection=None, add_status=0):
            self.projection = projection
            self.add_status = add_status

        def copy_matching(self, query, required):
            return subject.NativeCopyResult(0, self.projection)

        def add(self, attributes):
            return self.add_status

    missing = dict(base)
    missing.pop("kSecAttrLabel")
    with pytest.raises(subject.ProvisionError, match="RESULT_FIELDS"):
        subject.MacOSDataProtectionKeychain(
            plan, API(missing)
        ).copy_item()
    extra = dict(base, unexpected="ignored-by-os-layer-but-not-wrapper")
    with pytest.raises(subject.ProvisionError, match="RESULT_FIELDS"):
        subject.MacOSDataProtectionKeychain(plan, API(extra)).copy_item()
    mismatch = dict(base)
    mismatch["kSecAttrAccount"] = "WRONG"
    with pytest.raises(subject.ProvisionError, match="RESULT_PROJECTION"):
        subject.MacOSDataProtectionKeychain(
            plan, API(mismatch)
        ).copy_item()
    short_seed = dict(base)
    short_seed["kSecValueData"] = bytearray(b"s" * 31)
    with pytest.raises(subject.ProvisionError, match="SEED_LENGTH"):
        subject.MacOSDataProtectionKeychain(
            plan, API(short_seed)
        ).copy_item()
    with pytest.raises(subject.ProvisionError, match="ADD_LENGTH"):
        subject.MacOSDataProtectionKeychain(plan, API()).add_item(
            seed=bytearray(b"s" * 31), claim_sha256_raw=b"g" * 32
        )
    with pytest.raises(subject.ProvisionError, match="ADD_STATUS_-1"):
        subject.MacOSDataProtectionKeychain(
            plan, API(add_status=-1)
        ).add_item(seed=bytearray(b"s" * 32), claim_sha256_raw=b"g" * 32)


class _FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(*args)


class _FakeCoreFoundation:
    def __init__(self) -> None:
        self.next_pointer = 1000
        self.objects: dict[int, tuple[str, object]] = {}
        self.data_arrays: dict[int, object] = {}
        self.released: list[int] = []
        self.constants = {
            "kCFBooleanTrue": 101,
            "kCFBooleanFalse": 102,
        }
        self.objects[101] = ("constant", True)
        self.objects[102] = ("constant", False)
        self.CFDictionaryCreateMutable = _FakeFunction(
            lambda *_args: self._new("dictionary", {})
        )
        self.CFDictionarySetValue = _FakeFunction(self._set)
        self.CFDictionaryGetValue = _FakeFunction(self._get)
        self.CFStringCreateWithCString = _FakeFunction(self._string)
        self.CFDataCreate = _FakeFunction(self._data)
        self.CFGetTypeID = _FakeFunction(self._type_id)
        self.CFDictionaryGetTypeID = _FakeFunction(lambda: 1)
        self.CFDataGetTypeID = _FakeFunction(lambda: 2)
        self.CFDataGetLength = _FakeFunction(
            lambda pointer: len(self.objects[int(pointer)][1])
        )
        self.CFDataGetBytePtr = _FakeFunction(self._data_pointer)
        self.CFEqual = _FakeFunction(self._equal)
        self.CFRelease = _FakeFunction(
            lambda pointer: self.released.append(int(pointer))
        )

    def _new(self, kind: str, value: object) -> int:
        self.next_pointer += 1
        self.objects[self.next_pointer] = (kind, value)
        return self.next_pointer

    def _set(self, dictionary: int, key: int, value: int) -> None:
        self.objects[int(dictionary)][1][int(key)] = int(value)

    def _get(self, dictionary: int, key: int) -> int:
        return self.objects[int(dictionary)][1].get(int(key), 0)

    def _string(self, _allocator, raw: bytes, _encoding: int) -> int:
        return self._new("string", raw.decode())

    def _data(self, _allocator, pointer, length: int) -> int:
        raw = bytes(pointer[:length])
        result = self._new("data", raw)
        array = (subject.ctypes.c_uint8 * len(raw))(*raw)
        self.data_arrays[result] = array
        return result

    def _type_id(self, pointer: int) -> int:
        return {"dictionary": 1, "data": 2}.get(
            self.objects[int(pointer)][0], 3
        )

    def _data_pointer(self, pointer: int):
        return subject.ctypes.cast(
            self.data_arrays[int(pointer)],
            subject.ctypes.POINTER(subject.ctypes.c_uint8),
        )

    def _equal(self, left: int, right: int) -> bool:
        return self.objects[int(left)] == self.objects[int(right)]

    def make_value(self, value: object, symbols: dict[str, int]) -> int:
        if type(value) is bool:
            return self.constants[
                "kCFBooleanTrue" if value else "kCFBooleanFalse"
            ]
        if type(value) is bytes:
            array = (subject.ctypes.c_uint8 * len(value))(*value)
            return self._data(None, array, len(value))
        if type(value) is str and value in symbols:
            return symbols[value]
        if type(value) is str:
            return self._new("string", value)
        raise AssertionError(value)


class _FakeSecurityFramework:
    def __init__(self, core: _FakeCoreFoundation) -> None:
        self.core = core
        names = [
            "kSecClass",
            "kSecClassGenericPassword",
            "kSecAttrService",
            "kSecAttrAccount",
            "kSecAttrLabel",
            "kSecAttrGeneric",
            "kSecAttrSynchronizable",
            "kSecAttrAccessible",
            "kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly",
            "kSecUseDataProtectionKeychain",
            "kSecValueData",
            "kSecReturnData",
            "kSecReturnAttributes",
            "kSecMatchLimit",
            "kSecMatchLimitOne",
            "kSecUseAuthenticationUI",
            "kSecUseAuthenticationUIFail",
        ]
        self.constants = {
            name: 200 + index for index, name in enumerate(names)
        }
        for name, pointer in self.constants.items():
            core.objects[pointer] = ("constant", name)
        self.copy_status = 0
        self.add_status = 0
        self.result_projection: dict[str, object] = {}
        self.copy_dictionary: dict[int, int] | None = None
        self.add_dictionary: dict[int, int] | None = None
        self.SecItemCopyMatching = _FakeFunction(self._copy)
        self.SecItemAdd = _FakeFunction(self._add)

    def _result(self) -> int:
        result = self.core._new("dictionary", {})
        values = self.core.objects[result][1]
        for key, value in self.result_projection.items():
            values[self.constants[key]] = self.core.make_value(
                value, self.constants
            )
        return result

    def _copy(self, dictionary: int, output) -> int:
        self.copy_dictionary = dict(
            self.core.objects[int(dictionary)][1]
        )
        if self.copy_status == 0:
            output._obj.value = self._result()
        return self.copy_status

    def _add(self, dictionary: int, _output) -> int:
        self.add_dictionary = dict(
            self.core.objects[int(dictionary)][1]
        )
        return self.add_status


def _fake_native_api(monkeypatch: pytest.MonkeyPatch):
    core = _FakeCoreFoundation()
    security = _FakeSecurityFramework(core)

    def load(path: str):
        return security if "Security.framework" in path else core

    def constant(library, name: str):
        return subject.ctypes.c_void_p(library.constants[name])

    monkeypatch.setattr(subject.ctypes, "CDLL", load)
    monkeypatch.setattr(subject, "_framework_constant", constant)
    return subject.CoreFoundationKeychainAPI(), security, core


def test_corefoundation_bridge_builds_and_releases_exact_copy_and_add(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _bundle(tmp_path).plan
    api, security, core = _fake_native_api(monkeypatch)
    required = dict(
        plan["keychain"][
            "secitemcopymatching_result_policy"
        ]["required_persisted_attributes_verified_exactly"]
    )
    required["kSecAttrGeneric"] = b"g" * 32
    required["kSecValueData"] = b"s" * 32
    security.result_projection = required
    before = len(core.released)
    result = api.copy_matching(
        plan["keychain"]["secitemcopymatching_query_exact"], required
    )
    assert result == subject.NativeCopyResult(0, required)
    assert len(security.copy_dictionary) == 9
    assert set(security.copy_dictionary) == {
        security.constants[key]
        for key in plan["keychain"]["secitemcopymatching_query_exact"]
    }
    for key, expected in plan["keychain"][
        "secitemcopymatching_query_exact"
    ].items():
        pointer = security.copy_dictionary[security.constants[key]]
        assert core.objects[pointer][1] == expected
    assert len(core.released) > before
    copy_releases = list(core.released)
    attributes = subject.MacOSDataProtectionKeychain.add_contract(
        plan, b"s" * 32, b"g" * 32
    )
    assert api.add(attributes) == 0
    assert len(security.add_dictionary) == 9
    assert set(security.add_dictionary) == {
        security.constants[key] for key in attributes
    }
    for key, expected in attributes.items():
        pointer = security.add_dictionary[security.constants[key]]
        observed = core.objects[pointer][1]
        assert observed == expected
    assert len(core.released) > len(copy_releases)
    assert api.core.CFDictionarySetValue.restype is None
    assert api.security.SecItemCopyMatching.restype is subject.ctypes.c_int32
    assert api.security.SecItemAdd.restype is subject.ctypes.c_int32


def test_framework_constant_resolution_fails_closed() -> None:
    library = subject.ctypes.CDLL(None)
    with pytest.raises(subject.ProvisionError, match="KEYCHAIN_CONSTANT"):
        subject._framework_constant(
            library, "SIRETO_MISSING_SECURITY_CONSTANT_FOR_TEST"
        )


@pytest.mark.parametrize(
    "status",
    [subject.ERR_SEC_ITEM_NOT_FOUND, -1],
)
def test_corefoundation_bridge_releases_query_on_copy_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    plan = _bundle(tmp_path).plan
    api, security, core = _fake_native_api(monkeypatch)
    security.copy_status = status
    result = api.copy_matching(
        plan["keychain"]["secitemcopymatching_query_exact"], {}
    )
    assert result == subject.NativeCopyResult(status, None)
    assert security.copy_dictionary is not None
    assert core.released


def test_corefoundation_bridge_releases_add_dictionary_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _bundle(tmp_path).plan
    api, security, core = _fake_native_api(monkeypatch)
    security.add_status = -1
    attributes = subject.MacOSDataProtectionKeychain.add_contract(
        plan, b"s" * 32, b"g" * 32
    )
    assert api.add(attributes) == -1
    assert security.add_dictionary is not None
    assert core.released


def test_corefoundation_bridge_rejects_missing_projection_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _bundle(tmp_path).plan
    api, security, core = _fake_native_api(monkeypatch)
    required = dict(
        plan["keychain"][
            "secitemcopymatching_result_policy"
        ]["required_persisted_attributes_verified_exactly"]
    )
    security.result_projection = {}
    with pytest.raises(subject.ProvisionError, match="RESULT_MISSING"):
        api.copy_matching(
            plan["keychain"]["secitemcopymatching_query_exact"], required
        )
    assert core.released


def test_control_mutations_stop_before_keychain(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    bundle.authorization["execution_lock_sha256"] = "0" * 64
    bundle = subject.ControlBundle(
        bundle.plan,
        bundle.plan_raw,
        bundle.lock,
        bundle.lock_raw,
        bundle.authorization,
        _canonical(bundle.authorization),
    )
    backend = subject.MemoryKeychain()
    with pytest.raises(subject.ProvisionError, match="AUTH_LOCK_HASH"):
        subject.provision(
            bundle,
            backend,
            trusted_parent=tmp_path,
            root=Path(bundle.plan["paths"]["root"]),
        )
    assert backend.read_calls == backend.add_calls == 0


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda bundle: bundle.lock.__setitem__("purpose", "WRONG"),
            "execution_lock_purpose_TYPE",
        ),
        (
            lambda bundle: bundle.lock["runtime"].__setitem__(
                "python_version", "0"
            ),
            "LOCK_RUNTIME",
        ),
        (
            lambda bundle: bundle.lock["keychain_policy"].__setitem__(
                "data_protection_keychain", False
            ),
            "execution_lock_keychain_policy_TYPE",
        ),
        (
            lambda bundle: bundle.lock["implementation"].__setitem__(
                "provisioner_sha256", "0" * 64
            ),
            "IMPLEMENTATION_SOURCE_HASH",
        ),
        (
            lambda bundle: bundle.lock.__setitem__(
                "output_root", "/private/wrong"
            ),
            "LOCK_OUTPUT_ROOT",
        ),
        (
            lambda bundle: bundle.lock.__setitem__(
                "volume_device", bundle.lock["volume_device"] + 1
            ),
            "LOCK_VOLUME_DEVICE",
        ),
        (
            lambda bundle: bundle.lock.__setitem__(
                "volume_uuid", "00000000-0000-0000-0000-000000000001"
            ),
            "LOCK_VOLUME_UUID",
        ),
    ],
)
def test_all_control_families_are_enforced_before_keychain(
    tmp_path: Path, mutate, reason: str
) -> None:
    bundle = _bundle(tmp_path)
    mutate(bundle)
    lock_raw = _canonical(bundle.lock)
    bundle.authorization["execution_lock_sha256"] = hashlib.sha256(
        lock_raw
    ).hexdigest()
    bundle = subject.ControlBundle(
        bundle.plan,
        bundle.plan_raw,
        bundle.lock,
        lock_raw,
        bundle.authorization,
        _canonical(bundle.authorization),
    )
    backend = subject.MemoryKeychain()
    with pytest.raises(subject.ProvisionError, match=reason):
        subject.provision(
            bundle,
            backend,
            trusted_parent=tmp_path,
            root=Path(bundle.plan["paths"]["root"]),
        )
    assert backend.read_calls == backend.add_calls == 0


def test_self_consistent_payload_rewrite_is_rejected_without_keychain(
    tmp_path: Path,
) -> None:
    _receipt, backend, bundle = _run(tmp_path)
    authority = next(
        (Path(bundle.plan["paths"]["root"]) / "authorities").iterdir()
    )
    payload_path = authority / "producer_authority_payload.json"
    seal_path = authority / "producer_authority_seal.json"
    receipt_path = authority / "provision_receipt.json"
    payload = json.loads(payload_path.read_bytes())
    payload["producer"]["source_system"] = "ATTACKER"
    payload_raw = _canonical(payload)
    payload_path.write_bytes(payload_raw)
    seal = json.loads(seal_path.read_bytes())
    seal["payload_size_bytes"] = len(payload_raw)
    seal["payload_sha256"] = hashlib.sha256(payload_raw).hexdigest()
    seal_raw = _canonical(seal)
    seal_path.write_bytes(seal_raw)
    receipt = json.loads(receipt_path.read_bytes())
    receipt["payload_sha256"] = hashlib.sha256(payload_raw).hexdigest()
    receipt["seal_sha256"] = hashlib.sha256(seal_raw).hexdigest()
    receipt_path.write_bytes(_canonical(receipt))
    reads = backend.read_calls
    with pytest.raises(
        subject.ProvisionError, match="PAYLOAD_CONTROL_DIVERGENCE"
    ):
        subject.provision(
            bundle,
            backend,
            trusted_parent=tmp_path,
            root=Path(bundle.plan["paths"]["root"]),
        )
    assert backend.read_calls == reads


def test_symlinked_root_is_rejected_before_keychain(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    target = tmp_path / "redirect"
    target.mkdir(mode=0o700)
    Path(bundle.plan["paths"]["root"]).symlink_to(target, target_is_directory=True)
    backend = subject.MemoryKeychain()
    with pytest.raises(subject.ProvisionError, match="PRIVATE_DIRECTORY_OPEN"):
        subject.provision(
            bundle,
            backend,
            trusted_parent=tmp_path,
            root=Path(bundle.plan["paths"]["root"]),
        )
    assert backend.read_calls == backend.add_calls == 0


def test_private_mode_drift_stops_idempotent_read_before_keychain(
    tmp_path: Path,
) -> None:
    _receipt, backend, bundle = _run(tmp_path)
    authority = next(
        (Path(bundle.plan["paths"]["root"]) / "authorities").iterdir()
    )
    (authority / "producer_authority_payload.json").chmod(0o640)
    reads = backend.read_calls
    with pytest.raises(subject.ProvisionError, match="PAYLOAD_IDENTITY"):
        subject.provision(
            bundle,
            backend,
            trusted_parent=tmp_path,
            root=Path(bundle.plan["paths"]["root"]),
        )
    assert backend.read_calls == reads


def test_hardlinked_artifact_stops_idempotent_read_before_keychain(
    tmp_path: Path,
) -> None:
    _receipt, backend, bundle = _run(tmp_path)
    authority = next(
        (Path(bundle.plan["paths"]["root"]) / "authorities").iterdir()
    )
    os.link(
        authority / "producer_authority_payload.json",
        tmp_path / "external-hardlink",
    )
    reads = backend.read_calls
    with pytest.raises(subject.ProvisionError, match="PAYLOAD_IDENTITY"):
        subject.provision(
            bundle,
            backend,
            trusted_parent=tmp_path,
            root=Path(bundle.plan["paths"]["root"]),
        )
    assert backend.read_calls == reads


def test_partial_or_noncanonical_claim_is_never_rewritten(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    root = Path(bundle.plan["paths"]["root"])
    _write_claim(tmp_path, bundle, b"{")
    claim = root / "claims/provision.claim.json"
    with pytest.raises(subject.ProvisionError, match="CLAIM_INVALID_JSON"):
        subject.provision(
            bundle,
            subject.MemoryKeychain(),
            trusted_parent=tmp_path,
            root=root,
        )
    assert claim.read_bytes() == b"{"


def test_main_rejects_arguments_before_control_or_keychain(capsys) -> None:
    assert subject.main(["anything"]) == 64
    assert capsys.readouterr().err == "STOP:ARGS_FORBIDDEN\n"


def test_main_without_real_lock_stops_before_native_backend(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        subject,
        "MacOSDataProtectionKeychain",
        lambda _plan: pytest.fail("native backend must not be constructed"),
    )
    assert subject.main([]) == 65
    assert capsys.readouterr().err == "STOP:EXECUTION_LOCK_OPEN\n"


def test_source_has_no_forbidden_secret_or_external_channels() -> None:
    source = SUBJECT_PATH.read_text()
    for forbidden in (
        "subprocess",
        "tempfile",
        "requests",
        "urllib",
        "os.system",
        "Popen",
        "input(",
    ):
        assert forbidden not in source
    assert "CoreFoundationKeychainAPI" in source
    assert "SecItemCopyMatching" in source
    assert "SecItemAdd" in source
