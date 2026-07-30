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


def test_source_has_no_forbidden_secret_or_external_channels() -> None:
    source = SUBJECT_PATH.read_text()
    for forbidden in (
        "subprocess",
        "tempfile",
        "requests",
        "urllib",
        "security ",
        "input(",
    ):
        assert forbidden not in source
    assert "NATIVE_KEYCHAIN_NOT_PINNED" in source
