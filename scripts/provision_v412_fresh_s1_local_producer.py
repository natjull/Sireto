#!/usr/bin/env python3
"""One-shot S1 local producer authority.

The pure provisioning engine is intentionally injectable so every destructive
and secret-bearing transition can be tested without touching the macOS
Keychain.  ``main`` stays closed until the separately sealed execution lock
and authorization exist.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping, Protocol

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    REPOSITORY
    / "config/v4_12_fresh_s1_local_producer_authority_plan.json"
)
ERR_SEC_ITEM_NOT_FOUND = -25300
ERR_SEC_DUPLICATE_ITEM = -25299


class ProvisionError(RuntimeError):
    """Closed, non-secret provisioning failure."""


class InjectedCrash(RuntimeError):
    """Synthetic crash used only by tests."""


def _stop(reason: str) -> None:
    raise ProvisionError(reason)


def canonical_json(value: Any, *, final_lf: bool = True) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError):
        _stop("NON_CANONICAL_VALUE")
    return raw + (b"\n" if final_lf else b"")


def parse_canonical_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _stop(f"{label}_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _stop(f"{label}_INVALID_JSON")
    if type(value) is not dict or raw != canonical_json(value):
        _stop(f"{label}_NON_CANONICAL")
    return value


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _opaque_id(domain: str, projection: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        domain.encode("utf-8") + canonical_json(dict(projection), final_lf=False)
    ).hexdigest()
    return "".join(chr(ord("a") + int(nibble, 16)) for nibble in digest)


def derive_authority_id(plan: Mapping[str, Any], public_sha256: str) -> str:
    producer = plan["producer"]
    identity = plan["identity"]["authority"]
    projection = {
        "producer_id": producer["producer_id"],
        "producer_key_id": producer["producer_key_id"],
        "public_key_sha256": public_sha256,
        "producer_export_ledger_id": producer["producer_export_ledger_id"],
    }
    if list(projection) != identity["projection"]:
        _stop("AUTHORITY_ID_PROJECTION")
    return _opaque_id(identity["domain"], projection)


def derive_attempt_id(
    plan: Mapping[str, Any], plan_sha256: str, authority_id: str
) -> str:
    identity = plan["identity"]["attempt"]
    projection = {
        "plan_sha256": plan_sha256,
        "authority_id": authority_id,
        "logical_time_utc": plan["identity"]["logical_time_utc"],
    }
    if list(projection) != identity["projection"]:
        _stop("ATTEMPT_ID_PROJECTION")
    return _opaque_id(identity["domain"], projection)


def _validate_schema(
    value: Mapping[str, Any],
    schema_name: str,
    plan: Mapping[str, Any],
) -> None:
    schema = plan["schemas"][schema_name]
    if list(value) != schema["exact_fields"] and set(value) != set(
        schema["exact_fields"]
    ):
        _stop(f"{schema_name}_FIELDS")
    if set(value) != set(schema["types"]):
        _stop(f"{schema_name}_TYPES")
    for field in schema.get("nullable", []):
        if field not in value:
            _stop(f"{schema_name}_NULLABLE")
    expected_version = schema.get("schema_version")
    if expected_version is not None and value.get("schema_version") != expected_version:
        _stop(f"{schema_name}_VERSION")


def _read_regular(path: Path, label: str, *, private: bool = False) -> bytes:
    try:
        before = path.lstat()
    except OSError:
        _stop(f"{label}_MISSING")
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        _stop(f"{label}_NOT_PRIVATE_REGULAR")
    if private and (
        stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.getuid()
    ):
        _stop(f"{label}_MODE_OR_OWNER")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        _stop(f"{label}_OPEN")
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            _stop(f"{label}_IDENTITY")
        if private and (
            stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
        ):
            _stop(f"{label}_MODE_OR_OWNER")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def load_plan(path: Path = PLAN_PATH) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, "PLAN")
    return parse_canonical_object(raw, "PLAN"), raw


def _sync(fd: int) -> None:
    os.fsync(fd)
    full = getattr(fcntl, "F_FULLFSYNC", None)
    if full is None:
        _stop("F_FULLFSYNC_UNAVAILABLE")
    try:
        fcntl.fcntl(fd, full)
    except OSError:
        _stop("F_FULLFSYNC_FAILED")


def _check_private_dir(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        _stop("DIRECTORY_MODE")
    if info.st_uid != os.getuid():
        _stop("DIRECTORY_OWNER")


def ensure_private_tree(trusted_parent: Path, root: Path) -> None:
    try:
        relative = root.relative_to(trusted_parent)
    except ValueError:
        _stop("ROOT_OUTSIDE_TRUSTED_PARENT")
    current = trusted_parent
    if current.is_symlink() or not current.is_dir():
        _stop("TRUSTED_PARENT")
    for part in relative.parts:
        if part in ("", ".", ".."):
            _stop("ROOT_COMPONENT")
        current = current / part
        try:
            os.mkdir(current, 0o700)
            parent_fd = os.open(
                current.parent,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except FileExistsError:
            pass
        except OSError:
            _stop("DIRECTORY_CREATE")
        _check_private_dir(current)


def write_exclusive_durable(path: Path, raw: bytes) -> bool:
    """Create exact bytes; validate an identical existing file on recovery."""
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        if _read_regular(path, "EXISTING_ARTIFACT", private=True) != raw:
            _stop("EXISTING_ARTIFACT_DIVERGENCE")
        return False
    except OSError:
        _stop("ARTIFACT_CREATE")
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                _stop("SHORT_WRITE")
            view = view[count:]
        os.fchmod(fd, 0o600)
        _sync(fd)
    finally:
        os.close(fd)
    parent_fd = os.open(
        path.parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    if _read_regular(path, "CREATED_ARTIFACT", private=True) != raw:
        _stop("ARTIFACT_VERIFY")
    return True


@dataclass
class KeychainRecord:
    seed: bytearray
    claim_sha256_raw: bytes
    projection_valid: bool = True


class KeychainBackend(Protocol):
    read_calls: int
    add_calls: int

    def copy_item(self) -> KeychainRecord | None: ...

    def add_item(
        self, *, seed: bytearray, claim_sha256_raw: bytes
    ) -> str: ...


class MemoryKeychain:
    """In-memory backend used by synthetic tests only."""

    def __init__(
        self,
        item: KeychainRecord | None = None,
        *,
        duplicate_on_add: bool = False,
    ) -> None:
        self.item = item
        self.duplicate_on_add = duplicate_on_add
        self.read_calls = 0
        self.add_calls = 0

    def copy_item(self) -> KeychainRecord | None:
        self.read_calls += 1
        if self.item is None:
            return None
        return KeychainRecord(
            bytearray(self.item.seed),
            bytes(self.item.claim_sha256_raw),
            self.item.projection_valid,
        )

    def add_item(self, *, seed: bytearray, claim_sha256_raw: bytes) -> str:
        self.add_calls += 1
        if self.duplicate_on_add or self.item is not None:
            return "DUPLICATE"
        self.item = KeychainRecord(bytearray(seed), bytes(claim_sha256_raw))
        return "ADDED"


class MacOSDataProtectionKeychain:
    """Production backend boundary.

    The CoreFoundation bridge is deliberately activated only after the future
    provision gate.  Its exact dictionaries are public and testable now; the
    native call remains fail-closed until it is pinned by the execution lock.
    """

    read_calls = 0
    add_calls = 0

    @staticmethod
    def add_contract(plan: Mapping[str, Any], seed: bytes, claim_hash: bytes) -> dict[str, Any]:
        expected = dict(plan["keychain"]["secitemadd_dictionary_exact"])
        expected["kSecValueData"] = seed
        expected["kSecAttrGeneric"] = claim_hash
        return expected

    @staticmethod
    def query_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
        return dict(plan["keychain"]["secitemcopymatching_query_exact"])

    def copy_item(self) -> KeychainRecord | None:
        _stop("NATIVE_KEYCHAIN_NOT_PINNED")

    def add_item(self, *, seed: bytearray, claim_sha256_raw: bytes) -> str:
        _stop("NATIVE_KEYCHAIN_NOT_PINNED")


def zeroize(secret: bytearray | None) -> None:
    if secret is not None:
        secret[:] = b"\0" * len(secret)


def _public_from_seed(seed: bytearray) -> bytes:
    private = Ed25519PrivateKey.from_private_bytes(bytes(seed))
    return private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


def _sign(seed: bytearray, message: bytes) -> bytes:
    return Ed25519PrivateKey.from_private_bytes(bytes(seed)).sign(message)


def _verify_signature(public: bytes, message: bytes, signature: bytes) -> None:
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(signature, message)
    except Exception:
        _stop("GENESIS_SIGNATURE")


@dataclass(frozen=True)
class ControlBundle:
    plan: dict[str, Any]
    plan_raw: bytes
    lock: dict[str, Any]
    lock_raw: bytes
    authorization: dict[str, Any]
    authorization_raw: bytes

    @property
    def plan_sha256(self) -> str:
        return sha256_bytes(self.plan_raw)

    @property
    def lock_sha256(self) -> str:
        return sha256_bytes(self.lock_raw)

    @property
    def authorization_sha256(self) -> str:
        return sha256_bytes(self.authorization_raw)


def validate_control_bundle(bundle: ControlBundle) -> None:
    plan = bundle.plan
    _validate_schema(bundle.lock, "execution_lock", plan)
    _validate_schema(bundle.authorization, "authorization", plan)
    if bundle.lock["plan_sha256"] != bundle.plan_sha256:
        _stop("LOCK_PLAN_HASH")
    if bundle.authorization["plan_sha256"] != bundle.plan_sha256:
        _stop("AUTH_PLAN_HASH")
    if bundle.authorization["execution_lock_sha256"] != bundle.lock_sha256:
        _stop("AUTH_LOCK_HASH")
    if bundle.authorization["authorization"] != "AUTHORIZED_S1_LOCAL_PRODUCER_PROVISION":
        _stop("AUTHORIZATION_VALUE")
    if bundle.authorization["implementation_commit"] != bundle.lock["implementation"]["git_commit"]:
        _stop("AUTH_IMPLEMENTATION")
    if bundle.lock["logical_time_utc"] != plan["identity"]["logical_time_utc"]:
        _stop("LOCK_LOGICAL_TIME")
    if bundle.authorization["logical_time_utc"] != plan["identity"]["logical_time_utc"]:
        _stop("AUTH_LOGICAL_TIME")
    if bundle.lock["expected_uid"] != os.getuid():
        _stop("LOCK_UID")


def _claim_value(bundle: ControlBundle, nonce: bytes) -> dict[str, Any]:
    return {
        "schema_version": bundle.plan["schemas"]["claim"]["schema_version"],
        "plan_sha256": bundle.plan_sha256,
        "execution_lock_sha256": bundle.lock_sha256,
        "authorization_sha256": bundle.authorization_sha256,
        "attempt_binding_nonce_base64": base64.b64encode(nonce).decode("ascii"),
        "attempt_binding_nonce_sha256": sha256_bytes(nonce),
        "logical_time_utc": bundle.plan["identity"]["logical_time_utc"],
        "claim_state": "CLAIMED_BEFORE_KEYCHAIN",
    }


def _validate_claim(claim: dict[str, Any], bundle: ControlBundle) -> None:
    _validate_schema(claim, "claim", bundle.plan)
    if claim["plan_sha256"] != bundle.plan_sha256:
        _stop("CLAIM_PLAN")
    if claim["execution_lock_sha256"] != bundle.lock_sha256:
        _stop("CLAIM_LOCK")
    if claim["authorization_sha256"] != bundle.authorization_sha256:
        _stop("CLAIM_AUTHORIZATION")
    try:
        nonce = base64.b64decode(
            claim["attempt_binding_nonce_base64"], validate=True
        )
    except Exception:
        _stop("CLAIM_NONCE_BASE64")
    if len(nonce) != 32 or sha256_bytes(nonce) != claim["attempt_binding_nonce_sha256"]:
        _stop("CLAIM_NONCE")


def _s1_authorities(plan: Mapping[str, Any]) -> dict[str, Any]:
    authorities = plan["authorities"]
    return {
        "s1_contract_path": authorities["s1_contract"]["path"],
        "s1_contract_sha256": authorities["s1_contract"]["sha256"],
        "s1_plan_path": authorities["s1_plan"]["path"],
        "s1_plan_sha256": authorities["s1_plan"]["sha256"],
        "s1_authoritative_commit": authorities["s1_authoritative_commit"],
        "s1_certification_commit": authorities["s1_certification_commit"],
    }


def _build_artifacts(
    bundle: ControlBundle,
    claim_raw: bytes,
    seed: bytearray,
) -> tuple[str, dict[str, bytes], dict[str, Any]]:
    plan = bundle.plan
    public = _public_from_seed(seed)
    public_sha = sha256_bytes(public)
    authority_id = derive_authority_id(plan, public_sha)
    attempt_id = derive_attempt_id(plan, bundle.plan_sha256, authority_id)
    producer = plan["producer"]
    genesis_unsigned = {
        "schema_version": plan["schemas"]["ledger_genesis"]["schema_version"],
        "producer_id": producer["producer_id"],
        "producer_export_ledger_id": producer["producer_export_ledger_id"],
        "producer_export_sequence": 0,
        "producer_export_previous_entry_sha256": None,
        "entry_kind": "GENESIS",
        "producer_key_id": producer["producer_key_id"],
        "public_key_sha256": public_sha,
        "logical_time_utc": plan["identity"]["logical_time_utc"],
    }
    signed_raw = canonical_json(genesis_unsigned, final_lf=False)
    signature = _sign(seed, signed_raw)
    genesis = dict(genesis_unsigned)
    genesis["signature_base64"] = base64.b64encode(signature).decode("ascii")
    genesis_raw = canonical_json(genesis)
    genesis_sha = sha256_bytes(genesis_raw)
    ledger_head = sha256_bytes(signed_raw)
    locator = {
        "item_class": plan["keychain"]["item_class"],
        "service": plan["keychain"]["service"],
        "account": plan["keychain"]["account"],
        "producer_key_id": producer["producer_key_id"],
        "binding_attribute": "KSECATTRGENERIC_CLAIM_SHA256",
        "synchronizable": False,
        "accessible": "AFTER_FIRST_UNLOCK_THIS_DEVICE_ONLY",
        "data_protection_keychain": True,
    }
    payload = {
        "schema_version": plan["schemas"]["payload"]["schema_version"],
        "authority_id": authority_id,
        "producer": dict(producer),
        "public_key_base64": base64.b64encode(public).decode("ascii"),
        "public_key_sha256": public_sha,
        "keychain_locator": locator,
        "ledger_genesis_sha256": genesis_sha,
        "producer_export_ledger_head_sha256": ledger_head,
        "next_expected_export_sequence": 1,
        "s1_authorities": _s1_authorities(plan),
        "implementation": dict(bundle.lock["implementation"]),
        "runtime": dict(bundle.lock["runtime"]),
    }
    payload_raw = canonical_json(payload)
    seal = {
        "schema_version": plan["schemas"]["seal"]["schema_version"],
        "authority_id": authority_id,
        "payload_size_bytes": len(payload_raw),
        "payload_sha256": sha256_bytes(payload_raw),
        "ledger_genesis_size_bytes": len(genesis_raw),
        "ledger_genesis_sha256": genesis_sha,
    }
    seal_raw = canonical_json(seal)
    receipt = {
        "schema_version": plan["schemas"]["receipt"]["schema_version"],
        "authority_id": authority_id,
        "attempt_id": attempt_id,
        "plan_sha256": bundle.plan_sha256,
        "execution_lock_sha256": bundle.lock_sha256,
        "claim_sha256": sha256_bytes(claim_raw),
        "payload_sha256": sha256_bytes(payload_raw),
        "seal_sha256": sha256_bytes(seal_raw),
        "ledger_genesis_sha256": genesis_sha,
        "public_key_sha256": public_sha,
        "terminal_state": "PROVISIONED",
        "reason_code": "OK",
    }
    receipt_raw = canonical_json(receipt)
    for name, value in (
        ("ledger_genesis", genesis),
        ("payload", payload),
        ("seal", seal),
        ("receipt", receipt),
    ):
        _validate_schema(value, name, plan)
    _verify_signature(public, signed_raw, signature)
    return authority_id, {
        plan["paths"]["ledger_genesis_filename"]: genesis_raw,
        plan["paths"]["payload_filename"]: payload_raw,
        plan["paths"]["seal_filename"]: seal_raw,
        plan["paths"]["receipt_filename"]: receipt_raw,
    }, receipt


def _existing_receipt(
    root: Path, bundle: ControlBundle, claim_path: Path
) -> dict[str, Any] | None:
    authorities = root / "authorities"
    if not authorities.exists():
        return None
    entries = list(authorities.iterdir())
    if len(entries) > 1:
        _stop("MULTIPLE_AUTHORITIES")
    if not entries:
        return None
    authority_dir = entries[0]
    _check_private_dir(authority_dir)
    if (
        len(authority_dir.name) != 64
        or any(character not in "abcdefghijklmnop" for character in authority_dir.name)
    ):
        _stop("AUTHORITY_DIRECTORY_ID")
    receipt_path = authority_dir / bundle.plan["paths"]["receipt_filename"]
    if not receipt_path.exists():
        return None
    claim_raw = _read_regular(claim_path, "CLAIM", private=True)
    claim = parse_canonical_object(claim_raw, "CLAIM")
    _validate_claim(claim, bundle)
    receipt = parse_canonical_object(
        _read_regular(receipt_path, "RECEIPT", private=True), "RECEIPT"
    )
    _validate_schema(receipt, "receipt", bundle.plan)
    if receipt["claim_sha256"] != sha256_bytes(claim_raw):
        _stop("RECEIPT_CLAIM")
    if receipt["plan_sha256"] != bundle.plan_sha256 or receipt["execution_lock_sha256"] != bundle.lock_sha256:
        _stop("RECEIPT_CONTROL")
    if receipt["authority_id"] != authority_dir.name:
        _stop("RECEIPT_AUTHORITY")
    expected_files = {
        bundle.plan["paths"]["ledger_genesis_filename"],
        bundle.plan["paths"]["payload_filename"],
        bundle.plan["paths"]["seal_filename"],
        bundle.plan["paths"]["receipt_filename"],
    }
    if {entry.name for entry in authority_dir.iterdir()} != expected_files:
        _stop("AUTHORITY_TREE")
    genesis_raw = _read_regular(
        authority_dir / bundle.plan["paths"]["ledger_genesis_filename"],
        "GENESIS",
        private=True,
    )
    payload_raw = _read_regular(
        authority_dir / bundle.plan["paths"]["payload_filename"],
        "PAYLOAD",
        private=True,
    )
    seal_raw = _read_regular(
        authority_dir / bundle.plan["paths"]["seal_filename"],
        "SEAL",
        private=True,
    )
    if receipt["ledger_genesis_sha256"] != sha256_bytes(genesis_raw):
        _stop("RECEIPT_GENESIS")
    if receipt["payload_sha256"] != sha256_bytes(payload_raw):
        _stop("RECEIPT_PAYLOAD")
    if receipt["seal_sha256"] != sha256_bytes(seal_raw):
        _stop("RECEIPT_SEAL")
    payload = parse_canonical_object(payload_raw, "PAYLOAD")
    genesis = parse_canonical_object(genesis_raw, "GENESIS")
    seal = parse_canonical_object(seal_raw, "SEAL")
    for name, value in (
        ("payload", payload),
        ("ledger_genesis", genesis),
        ("seal", seal),
    ):
        _validate_schema(value, name, bundle.plan)
    if payload["authority_id"] != receipt["authority_id"]:
        _stop("PAYLOAD_AUTHORITY")
    if (
        seal["authority_id"] != receipt["authority_id"]
        or seal["payload_sha256"] != sha256_bytes(payload_raw)
        or seal["payload_size_bytes"] != len(payload_raw)
        or seal["ledger_genesis_sha256"] != sha256_bytes(genesis_raw)
        or seal["ledger_genesis_size_bytes"] != len(genesis_raw)
    ):
        _stop("SEAL_HASH")
    try:
        public = base64.b64decode(payload["public_key_base64"], validate=True)
        signature = base64.b64decode(
            genesis["signature_base64"], validate=True
        )
    except Exception:
        _stop("PUBLIC_ENCODING")
    public_sha = sha256_bytes(public)
    if (
        len(public) != 32
        or len(signature) != 64
        or payload["public_key_sha256"] != public_sha
        or receipt["public_key_sha256"] != public_sha
        or derive_authority_id(bundle.plan, public_sha) != receipt["authority_id"]
        or derive_attempt_id(
            bundle.plan, bundle.plan_sha256, receipt["authority_id"]
        )
        != receipt["attempt_id"]
    ):
        _stop("PUBLIC_IDENTITY")
    unsigned = dict(genesis)
    del unsigned["signature_base64"]
    unsigned_raw = canonical_json(unsigned, final_lf=False)
    if (
        payload["producer_export_ledger_head_sha256"]
        != sha256_bytes(unsigned_raw)
    ):
        _stop("LEDGER_HEAD")
    _verify_signature(
        public, unsigned_raw, signature
    )
    return receipt


def provision(
    bundle: ControlBundle,
    backend: KeychainBackend,
    *,
    trusted_parent: Path,
    root: Path,
    random32: Callable[[], bytes] = lambda: os.urandom(32),
    checkpoint: Callable[[str], None] = lambda _name: None,
) -> dict[str, Any]:
    """Provision or recover one authority using only injected resources."""
    validate_control_bundle(bundle)
    old_umask = os.umask(0o077)
    seed: bytearray | None = None
    item_seed: bytearray | None = None
    try:
        ensure_private_tree(trusted_parent, root)
        ensure_private_tree(trusted_parent, root / "claims")
        ensure_private_tree(trusted_parent, root / "authorities")
        claim_path = root / "claims" / "provision.claim.json"
        existing = _existing_receipt(root, bundle, claim_path)
        if existing is not None:
            return existing
        if claim_path.exists():
            claim_raw = _read_regular(claim_path, "CLAIM", private=True)
            claim = parse_canonical_object(claim_raw, "CLAIM")
            _validate_claim(claim, bundle)
            claim_created = False
        else:
            nonce = random32()
            if type(nonce) is not bytes or len(nonce) != 32:
                _stop("RANDOM_NONCE")
            claim = _claim_value(bundle, nonce)
            _validate_schema(claim, "claim", bundle.plan)
            claim_raw = canonical_json(claim)
            claim_created = write_exclusive_durable(claim_path, claim_raw)
        checkpoint("CLAIM_DURABLE")
        persisted_claim = parse_canonical_object(
            _read_regular(claim_path, "CLAIM", private=True), "CLAIM"
        )
        _validate_claim(persisted_claim, bundle)
        claim_hash_raw = hashlib.sha256(claim_raw).digest()
        item = backend.copy_item()
        checkpoint("KEYCHAIN_QUERIED")
        if claim_created and item is not None:
            zeroize(item.seed)
            _stop("FOREIGN_ITEM_NEW_CLAIM")
        if item is not None:
            item_seed = item.seed
            if (
                not item.projection_valid
                or item.claim_sha256_raw != claim_hash_raw
                or len(item_seed) != 32
            ):
                _stop("FOREIGN_ITEM_BINDING")
            seed = item_seed
        else:
            seed = bytearray(random32())
            if len(seed) != 32:
                _stop("RANDOM_SEED")
            checkpoint("SEED_GENERATED")
            result = backend.add_item(
                seed=seed, claim_sha256_raw=claim_hash_raw
            )
            if result != "ADDED":
                _stop("KEYCHAIN_ADD_DUPLICATE")
            checkpoint("KEYCHAIN_ADDED")
            reread = backend.copy_item()
            if reread is None:
                _stop("KEYCHAIN_ADD_NOT_READABLE")
            item_seed = reread.seed
            if (
                not reread.projection_valid
                or reread.claim_sha256_raw != claim_hash_raw
                or item_seed != seed
            ):
                _stop("KEYCHAIN_ADD_VERIFY")
        authority_id, artifacts, receipt = _build_artifacts(
            bundle, claim_raw, seed
        )
        authority_dir = root / "authorities" / authority_id
        ensure_private_tree(trusted_parent, authority_dir)
        for filename in (
            bundle.plan["paths"]["ledger_genesis_filename"],
            bundle.plan["paths"]["payload_filename"],
            bundle.plan["paths"]["seal_filename"],
            bundle.plan["paths"]["receipt_filename"],
        ):
            write_exclusive_durable(authority_dir / filename, artifacts[filename])
            checkpoint(f"{filename}_DURABLE")
        validated = _existing_receipt(root, bundle, claim_path)
        if validated != receipt:
            _stop("FINAL_PUBLIC_VALIDATION")
        return receipt
    finally:
        zeroize(seed)
        if item_seed is not seed:
            zeroize(item_seed)
        os.umask(old_umask)


def load_control_bundle(
    plan_path: Path = PLAN_PATH,
) -> ControlBundle:
    plan, plan_raw = load_plan(plan_path)
    lock_path = REPOSITORY / plan["paths"]["execution_lock"]
    authorization_path = REPOSITORY / plan["paths"]["authorization"]
    lock_raw = _read_regular(lock_path, "EXECUTION_LOCK")
    auth_raw = _read_regular(authorization_path, "AUTHORIZATION")
    return ControlBundle(
        plan,
        plan_raw,
        parse_canonical_object(lock_raw, "EXECUTION_LOCK"),
        lock_raw,
        parse_canonical_object(auth_raw, "AUTHORIZATION"),
        auth_raw,
    )


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        print("STOP:ARGS_FORBIDDEN", file=sys.stderr)
        return 64
    try:
        bundle = load_control_bundle()
        backend = MacOSDataProtectionKeychain()
        receipt = provision(
            bundle,
            backend,
            trusted_parent=Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100"),
            root=Path(bundle.plan["paths"]["root"]),
        )
    except ProvisionError as exc:
        print(f"STOP:{exc}", file=sys.stderr)
        return 65
    print(canonical_json(receipt).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
