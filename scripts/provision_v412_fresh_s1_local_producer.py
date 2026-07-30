#!/usr/bin/env python3
"""One-shot S1 local producer authority.

The pure provisioning engine is intentionally injectable so every destructive
and secret-bearing transition can be tested without touching the macOS
Keychain.  ``main`` stays closed until the separately sealed execution lock
and authorization exist.
"""

from __future__ import annotations

import base64
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import struct
import sys
from typing import Any, Callable, Mapping, Protocol
import uuid

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
    if set(value) != set(schema["exact_fields"]):
        _stop(f"{schema_name}_FIELDS")
    if set(value) != set(schema["types"]):
        _stop(f"{schema_name}_TYPES")
    nullable = set(schema.get("nullable", []))
    for field, type_name in schema["types"].items():
        field_value = value[field]
        if field_value is None and field not in nullable and type_name != "null":
            _stop(f"{schema_name}_{field}_NULL")
        if not _matches_type(field_value, type_name, plan):
            _stop(f"{schema_name}_{field}_TYPE")
    expected_version = schema.get("schema_version")
    if expected_version is not None and value.get("schema_version") != expected_version:
        _stop(f"{schema_name}_VERSION")


_ENUM_VALUES = {
    "enum_AFTER_FIRST_UNLOCK_THIS_DEVICE_ONLY": (
        "AFTER_FIRST_UNLOCK_THIS_DEVICE_ONLY"
    ),
    "enum_AUTHORIZED_S1_LOCAL_PRODUCER_PROVISION": (
        "AUTHORIZED_S1_LOCAL_PRODUCER_PROVISION"
    ),
    "enum_CLAIMED_BEFORE_KEYCHAIN": "CLAIMED_BEFORE_KEYCHAIN",
    "enum_Darwin": "Darwin",
    "enum_FAIL": "FAIL",
    "enum_GENERIC_PASSWORD": "GENERIC_PASSWORD",
    "enum_GENESIS": "GENESIS",
    "enum_KSECATTRGENERIC_CLAIM_SHA256": (
        "KSECATTRGENERIC_CLAIM_SHA256"
    ),
    "enum_LOCAL_PRODUCER_AUTHORITY_WITHOUT_CRM": (
        "LOCAL_PRODUCER_AUTHORITY_WITHOUT_CRM"
    ),
    "enum_OK": "OK",
    "enum_PROVISIONED": "PROVISIONED",
}


def _is_base64_size(value: Any, size: int) -> bool:
    if type(value) is not str:
        return False
    try:
        decoded = base64.b64decode(value, validate=True)
    except Exception:
        return False
    return len(decoded) == size and base64.b64encode(decoded).decode() == value


def _matches_type(value: Any, type_name: str, plan: Mapping[str, Any]) -> bool:
    if type_name in plan["schemas"]:
        return type(value) is dict and _schema_matches(value, type_name, plan)
    if type_name in _ENUM_VALUES:
        return value == _ENUM_VALUES[type_name]
    if type_name == "null":
        return value is None
    if type_name == "boolean_false":
        return value is False
    if type_name == "boolean_true":
        return value is True
    if type_name == "integer_zero":
        return type(value) is int and value == 0
    if type_name == "integer_one":
        return type(value) is int and value == 1
    if type_name == "integer_32":
        return type(value) is int and value == 32
    if type_name == "integer_nonnegative":
        return type(value) is int and value >= 0
    if type_name == "integer_positive":
        return type(value) is int and value > 0
    if type_name == "sha256":
        return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    if type_name == "git_commit_40_hex":
        return type(value) is str and re.fullmatch(r"[0-9a-f]{40}", value) is not None
    if type_name == "opaque_id_64_a_to_p":
        return type(value) is str and re.fullmatch(r"[a-p]{64}", value) is not None
    if type_name == "base64_ed25519_public_key_32_bytes":
        return _is_base64_size(value, 32)
    if type_name == "base64_ed25519_signature_64_bytes":
        return _is_base64_size(value, 64)
    if type_name == "base64_random_32_bytes":
        return _is_base64_size(value, 32)
    if type_name == "timestamp_rfc3339_utc":
        return type(value) is str and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
        ) is not None
    if type_name == "uuid_string":
        if type(value) is not str:
            return False
        try:
            return str(uuid.UUID(value)).lower() == value.lower()
        except ValueError:
            return False
    if type_name in {
        "string_nonempty",
        "string_equal_plan",
        "string_equal_producer_key_id",
    }:
        return type(value) is str and bool(value)
    if type_name in {"repo_relative_path", "repo_relative_path_equal_plan"}:
        return (
            type(value) is str
            and bool(value)
            and not Path(value).is_absolute()
            and ".." not in Path(value).parts
        )
    if type_name in {"absolute_path", "absolute_path_equal_plan"}:
        return type(value) is str and Path(value).is_absolute()
    return False


def _schema_matches(
    value: Mapping[str, Any], schema_name: str, plan: Mapping[str, Any]
) -> bool:
    try:
        _validate_schema(value, schema_name, plan)
    except ProvisionError:
        return False
    return True


def _read_regular(path: Path, label: str, *, private: bool = False) -> bytes:
    parent_fd = _open_absolute_dir_anchored(path.parent)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError:
        os.close(parent_fd)
        _stop(f"{label}_OPEN")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _stop(f"{label}_NOT_PRIVATE_REGULAR")
        if private and (
            stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
        ):
            _stop(f"{label}_MODE_OR_OWNER")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _stop(f"{label}_DRIFT")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            _stop(f"{label}_SIZE")
        return raw
    finally:
        os.close(fd)
        os.close(parent_fd)


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


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_absolute_dir_anchored(path: Path) -> int:
    if not path.is_absolute():
        _stop("ANCHOR_NOT_ABSOLUTE")
    try:
        current = os.open("/", _directory_flags())
    except OSError:
        _stop("ANCHOR_ROOT_OPEN")
    try:
        for component in path.parts[1:]:
            if component in ("", ".", ".."):
                _stop("ANCHOR_COMPONENT")
            try:
                following = os.open(
                    component, _directory_flags(), dir_fd=current
                )
            except OSError:
                _stop("ANCHOR_COMPONENT_OPEN")
            os.close(current)
            current = following
        return current
    except Exception:
        os.close(current)
        raise


def _check_dir_fd(fd: int, label: str, *, private: bool) -> None:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        _stop(f"{label}_NOT_DIRECTORY")
    if private and (
        stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.getuid()
    ):
        _stop(f"{label}_MODE_OR_OWNER")


def _open_or_create_private_dir(parent_fd: int, name: str) -> int:
    if "/" in name or name in ("", ".", ".."):
        _stop("PRIVATE_DIRECTORY_NAME")
    try:
        fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        except OSError:
            _stop("PRIVATE_DIRECTORY_CREATE")
        try:
            fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
        except OSError:
            _stop("PRIVATE_DIRECTORY_OPEN")
    except OSError:
        _stop("PRIVATE_DIRECTORY_OPEN")
    _check_dir_fd(fd, "PRIVATE_DIRECTORY", private=True)
    return fd


def _read_private_at(parent_fd: int, name: str, label: str) -> bytes:
    if "/" in name or name in ("", ".", ".."):
        _stop(f"{label}_NAME")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError:
        _stop(f"{label}_OPEN")
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
        ):
            _stop(f"{label}_IDENTITY")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _stop(f"{label}_DRIFT")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            _stop(f"{label}_SIZE")
        return raw
    finally:
        os.close(fd)


def _exists_at(parent_fd: int, name: str) -> bool:
    try:
        fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return False
    except OSError:
        _stop("EXISTENCE_CHECK")
    os.close(fd)
    return True


def _write_private_at(parent_fd: int, name: str, raw: bytes) -> bool:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(name, flags, 0o600, dir_fd=parent_fd)
    except FileExistsError:
        if _read_private_at(parent_fd, name, "EXISTING_ARTIFACT") != raw:
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
    os.fsync(parent_fd)
    if _read_private_at(parent_fd, name, "CREATED_ARTIFACT") != raw:
        _stop("ARTIFACT_VERIFY")
    return True


class AnchoredStore:
    def __init__(self, trusted_parent: Path, root: Path) -> None:
        try:
            relative = root.relative_to(trusted_parent)
        except ValueError:
            _stop("ROOT_OUTSIDE_TRUSTED_PARENT")
        self._fds: list[int] = []
        parent_fd = _open_absolute_dir_anchored(trusted_parent)
        self._fds.append(parent_fd)
        current = parent_fd
        for component in relative.parts:
            following = _open_or_create_private_dir(current, component)
            self._fds.append(following)
            current = following
        self.root_fd = current
        self.claims_fd = _open_or_create_private_dir(self.root_fd, "claims")
        self._fds.append(self.claims_fd)
        self.authorities_fd = _open_or_create_private_dir(
            self.root_fd, "authorities"
        )
        self._fds.append(self.authorities_fd)

    def close(self) -> None:
        while self._fds:
            os.close(self._fds.pop())

    def verify_volume(self, device: int, volume_uuid: str) -> None:
        for fd in (self.root_fd, self.claims_fd, self.authorities_fd):
            if os.fstat(fd).st_dev != device:
                _stop("STORE_VOLUME_DEVICE")
            if volume_uuid_for_fd(fd) != volume_uuid.lower():
                _stop("STORE_VOLUME_UUID")

    def claim_exists(self) -> bool:
        return _exists_at(self.claims_fd, "provision.claim.json")

    def read_claim(self) -> bytes:
        return _read_private_at(
            self.claims_fd, "provision.claim.json", "CLAIM"
        )

    def create_claim(self, raw: bytes) -> bool:
        return _write_private_at(
            self.claims_fd, "provision.claim.json", raw
        )

    def authority_names(self) -> list[str]:
        names = os.listdir(self.authorities_fd)
        if any("/" in name or name in (".", "..") for name in names):
            _stop("AUTHORITY_NAME")
        return sorted(names)

    def open_authority(self, name: str, *, create: bool) -> int:
        if (
            len(name) != 64
            or any(character not in "abcdefghijklmnop" for character in name)
        ):
            _stop("AUTHORITY_DIRECTORY_ID")
        if create:
            fd = _open_or_create_private_dir(self.authorities_fd, name)
        else:
            try:
                fd = os.open(
                    name, _directory_flags(), dir_fd=self.authorities_fd
                )
            except OSError:
                _stop("AUTHORITY_DIRECTORY_OPEN")
            _check_dir_fd(fd, "AUTHORITY_DIRECTORY", private=True)
        self._fds.append(fd)
        return fd

    @staticmethod
    def read_authority_file(fd: int, name: str, label: str) -> bytes:
        return _read_private_at(fd, name, label)

    @staticmethod
    def write_authority_file(fd: int, name: str, raw: bytes) -> bool:
        return _write_private_at(fd, name, raw)


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


class _AttrList(ctypes.Structure):
    _fields_ = [
        ("bitmapcount", ctypes.c_ushort),
        ("reserved", ctypes.c_uint16),
        ("commonattr", ctypes.c_uint32),
        ("volattr", ctypes.c_uint32),
        ("dirattr", ctypes.c_uint32),
        ("fileattr", ctypes.c_uint32),
        ("forkattr", ctypes.c_uint32),
    ]


def volume_uuid_for_fd(fd: int) -> str:
    if platform.system() != "Darwin":
        _stop("VOLUME_UUID_PLATFORM")
    before = os.fstat(fd)
    attributes = _AttrList(5, 0, 0, 0x80000000 | 0x00040000, 0, 0, 0)
    buffer = ctypes.create_string_buffer(20)
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "fgetattrlist", None)
    if function is None:
        _stop("VOLUME_UUID_API")
    function.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(_AttrList),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_ulong,
    ]
    function.restype = ctypes.c_int
    if function(
        fd,
        ctypes.byref(attributes),
        ctypes.byref(buffer),
        ctypes.sizeof(buffer),
        0,
    ):
        _stop("VOLUME_UUID_READ")
    if struct.unpack_from("=I", buffer.raw, 0)[0] < 20:
        _stop("VOLUME_UUID_SHORT")
    result = str(uuid.UUID(bytes=buffer.raw[4:20])).lower()
    after = os.fstat(fd)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        _stop("VOLUME_UUID_DRIFT")
    return result


def _expected_keychain_policy(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "item_class": plan["keychain"]["item_class"],
        "service": plan["keychain"]["service"],
        "account": plan["keychain"]["account"],
        "label": plan["producer"]["producer_key_id"],
        "binding_attribute": "KSECATTRGENERIC_CLAIM_SHA256",
        "synchronizable": False,
        "accessible": "AFTER_FIRST_UNLOCK_THIS_DEVICE_ONLY",
        "data_protection_keychain": True,
        "authentication_ui": plan["keychain"]["authentication_ui"],
        "secret_length_bytes": plan["keychain"]["secret_length_bytes"],
    }


def _expected_runtime() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable_path": str(executable),
        "python_executable_sha256": sha256_bytes(
            _read_regular(executable, "PYTHON_EXECUTABLE")
        ),
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


def _verify_repo_pin(path_text: str, expected_sha256: str, label: str) -> None:
    path = REPOSITORY / path_text
    if not path.is_relative_to(REPOSITORY):
        _stop(f"{label}_PATH")
    if sha256_bytes(_read_regular(path, label)) != expected_sha256:
        _stop(f"{label}_HASH")


def validate_control_bundle(
    bundle: ControlBundle,
    *,
    trusted_parent: Path,
    root: Path,
) -> None:
    plan = bundle.plan
    if bundle.plan_raw != canonical_json(plan):
        _stop("PLAN_CANONICAL")
    if bundle.lock_raw != canonical_json(bundle.lock):
        _stop("LOCK_CANONICAL")
    if bundle.authorization_raw != canonical_json(bundle.authorization):
        _stop("AUTHORIZATION_CANONICAL")
    _validate_schema(bundle.lock, "execution_lock", plan)
    _validate_schema(bundle.authorization, "authorization", plan)
    lock = bundle.lock
    authorization = bundle.authorization
    if bundle.lock["plan_sha256"] != bundle.plan_sha256:
        _stop("LOCK_PLAN_HASH")
    if bundle.authorization["plan_sha256"] != bundle.plan_sha256:
        _stop("AUTH_PLAN_HASH")
    if bundle.authorization["execution_lock_sha256"] != bundle.lock_sha256:
        _stop("AUTH_LOCK_HASH")
    if authorization["authorization"] != "AUTHORIZED_S1_LOCAL_PRODUCER_PROVISION":
        _stop("AUTHORIZATION_VALUE")
    if authorization["implementation_commit"] != lock["implementation"]["git_commit"]:
        _stop("AUTH_IMPLEMENTATION")
    if lock["logical_time_utc"] != plan["identity"]["logical_time_utc"]:
        _stop("LOCK_LOGICAL_TIME")
    if authorization["logical_time_utc"] != plan["identity"]["logical_time_utc"]:
        _stop("AUTH_LOGICAL_TIME")
    if lock["expected_uid"] != os.getuid():
        _stop("LOCK_UID")
    if lock["purpose"] != "LOCAL_PRODUCER_AUTHORITY_WITHOUT_CRM":
        _stop("LOCK_PURPOSE")
    if lock["plan_path"] != str(PLAN_PATH.relative_to(REPOSITORY)):
        _stop("LOCK_PLAN_PATH")
    if (
        lock["contract_path"] != plan["authorities"]["contract"]["path"]
        or lock["contract_sha256"]
        != plan["authorities"]["contract"]["sha256"]
    ):
        _stop("LOCK_CONTRACT")
    if lock["authorization_path"] != plan["paths"]["authorization"]:
        _stop("LOCK_AUTHORIZATION_PATH")
    if lock["output_root"] != plan["paths"]["root"] or root != Path(
        plan["paths"]["root"]
    ):
        _stop("LOCK_OUTPUT_ROOT")
    implementation = lock["implementation"]
    expected_implementation_paths = {
        "provisioner_path": plan["future_implementation"]["provisioner"][
            "path"
        ],
        "tests_path": plan["future_implementation"]["tests"]["path"],
        "plan_path": str(PLAN_PATH.relative_to(REPOSITORY)),
        "contract_path": plan["authorities"]["contract"]["path"],
    }
    for field, expected in expected_implementation_paths.items():
        if implementation[field] != expected:
            _stop(f"IMPLEMENTATION_{field.upper()}")
    if (
        implementation["plan_sha256"] != bundle.plan_sha256
        or implementation["contract_sha256"]
        != plan["authorities"]["contract"]["sha256"]
    ):
        _stop("IMPLEMENTATION_CONTROL_HASH")
    for path_field, hash_field, label in (
        ("provisioner_path", "provisioner_sha256", "IMPLEMENTATION_SOURCE"),
        ("tests_path", "tests_sha256", "IMPLEMENTATION_TESTS"),
        ("contract_path", "contract_sha256", "CONTRACT"),
    ):
        _verify_repo_pin(
            implementation[path_field], implementation[hash_field], label
        )
    for role, label in (("s1_contract", "S1_CONTRACT"), ("s1_plan", "S1_PLAN")):
        authority = plan["authorities"][role]
        _verify_repo_pin(authority["path"], authority["sha256"], label)
    if lock["runtime"] != _expected_runtime():
        _stop("LOCK_RUNTIME")
    if lock["keychain_policy"] != _expected_keychain_policy(plan):
        _stop("LOCK_KEYCHAIN_POLICY")
    parent_fd = _open_absolute_dir_anchored(trusted_parent)
    try:
        parent_info = os.fstat(parent_fd)
        if parent_info.st_dev != lock["volume_device"]:
            _stop("LOCK_VOLUME_DEVICE")
        if volume_uuid_for_fd(parent_fd) != lock["volume_uuid"].lower():
            _stop("LOCK_VOLUME_UUID")
    finally:
        os.close(parent_fd)


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
    store: AnchoredStore, bundle: ControlBundle
) -> dict[str, Any] | None:
    names = store.authority_names()
    if len(names) > 1:
        _stop("MULTIPLE_AUTHORITIES")
    if not names:
        return None
    authority_name = names[0]
    authority_fd = store.open_authority(authority_name, create=False)
    if os.fstat(authority_fd).st_dev != bundle.lock["volume_device"]:
        _stop("AUTHORITY_VOLUME_DEVICE")
    if (
        volume_uuid_for_fd(authority_fd)
        != bundle.lock["volume_uuid"].lower()
    ):
        _stop("AUTHORITY_VOLUME_UUID")
    receipt_name = bundle.plan["paths"]["receipt_filename"]
    if not _exists_at(authority_fd, receipt_name):
        return None
    if not store.claim_exists():
        _stop("RECEIPT_WITHOUT_CLAIM")
    claim_raw = store.read_claim()
    claim = parse_canonical_object(claim_raw, "CLAIM")
    _validate_claim(claim, bundle)
    receipt = parse_canonical_object(
        store.read_authority_file(authority_fd, receipt_name, "RECEIPT"),
        "RECEIPT",
    )
    _validate_schema(receipt, "receipt", bundle.plan)
    if receipt["claim_sha256"] != sha256_bytes(claim_raw):
        _stop("RECEIPT_CLAIM")
    if receipt["plan_sha256"] != bundle.plan_sha256 or receipt["execution_lock_sha256"] != bundle.lock_sha256:
        _stop("RECEIPT_CONTROL")
    if receipt["authority_id"] != authority_name:
        _stop("RECEIPT_AUTHORITY")
    expected_files = {
        bundle.plan["paths"]["ledger_genesis_filename"],
        bundle.plan["paths"]["payload_filename"],
        bundle.plan["paths"]["seal_filename"],
        bundle.plan["paths"]["receipt_filename"],
    }
    if set(os.listdir(authority_fd)) != expected_files:
        _stop("AUTHORITY_TREE")
    genesis_raw = store.read_authority_file(
        authority_fd,
        bundle.plan["paths"]["ledger_genesis_filename"],
        "GENESIS",
    )
    payload_raw = store.read_authority_file(
        authority_fd,
        bundle.plan["paths"]["payload_filename"],
        "PAYLOAD",
    )
    seal_raw = store.read_authority_file(
        authority_fd,
        bundle.plan["paths"]["seal_filename"],
        "SEAL",
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
    plan = bundle.plan
    producer = plan["producer"]
    expected_locator = {
        "item_class": plan["keychain"]["item_class"],
        "service": plan["keychain"]["service"],
        "account": plan["keychain"]["account"],
        "producer_key_id": producer["producer_key_id"],
        "binding_attribute": "KSECATTRGENERIC_CLAIM_SHA256",
        "synchronizable": False,
        "accessible": "AFTER_FIRST_UNLOCK_THIS_DEVICE_ONLY",
        "data_protection_keychain": True,
    }
    if (
        payload["producer"] != producer
        or payload["keychain_locator"] != expected_locator
        or payload["s1_authorities"] != _s1_authorities(plan)
        or payload["implementation"] != bundle.lock["implementation"]
        or payload["runtime"] != bundle.lock["runtime"]
        or payload["next_expected_export_sequence"] != 1
        or payload["ledger_genesis_sha256"] != sha256_bytes(genesis_raw)
    ):
        _stop("PAYLOAD_CONTROL_DIVERGENCE")
    expected_genesis = {
        "schema_version": plan["schemas"]["ledger_genesis"][
            "schema_version"
        ],
        "producer_id": producer["producer_id"],
        "producer_export_ledger_id": producer["producer_export_ledger_id"],
        "producer_export_sequence": 0,
        "producer_export_previous_entry_sha256": None,
        "entry_kind": "GENESIS",
        "producer_key_id": producer["producer_key_id"],
        "public_key_sha256": payload["public_key_sha256"],
        "logical_time_utc": plan["identity"]["logical_time_utc"],
    }
    if {key: genesis[key] for key in expected_genesis} != expected_genesis:
        _stop("GENESIS_CONTROL_DIVERGENCE")
    if (
        receipt["terminal_state"] != "PROVISIONED"
        or receipt["reason_code"] != "OK"
    ):
        _stop("RECEIPT_TERMINAL")
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
    validate_control_bundle(
        bundle, trusted_parent=trusted_parent, root=root
    )
    old_umask = os.umask(0o077)
    seed: bytearray | None = None
    item_seed: bytearray | None = None
    store: AnchoredStore | None = None
    try:
        store = AnchoredStore(trusted_parent, root)
        store.verify_volume(
            bundle.lock["volume_device"], bundle.lock["volume_uuid"]
        )
        existing = _existing_receipt(store, bundle)
        if existing is not None:
            return existing
        if store.claim_exists():
            claim_raw = store.read_claim()
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
            claim_created = store.create_claim(claim_raw)
        checkpoint("CLAIM_DURABLE")
        persisted_claim = parse_canonical_object(
            store.read_claim(), "CLAIM"
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
        authority_fd = store.open_authority(authority_id, create=True)
        if os.fstat(authority_fd).st_dev != bundle.lock["volume_device"]:
            _stop("AUTHORITY_VOLUME_DEVICE")
        if (
            volume_uuid_for_fd(authority_fd)
            != bundle.lock["volume_uuid"].lower()
        ):
            _stop("AUTHORITY_VOLUME_UUID")
        for filename in (
            bundle.plan["paths"]["ledger_genesis_filename"],
            bundle.plan["paths"]["payload_filename"],
            bundle.plan["paths"]["seal_filename"],
            bundle.plan["paths"]["receipt_filename"],
        ):
            store.write_authority_file(
                authority_fd, filename, artifacts[filename]
            )
            checkpoint(f"{filename}_DURABLE")
        validated = _existing_receipt(store, bundle)
        if validated != receipt:
            _stop("FINAL_PUBLIC_VALIDATION")
        return receipt
    finally:
        zeroize(seed)
        if item_seed is not seed:
            zeroize(item_seed)
        if store is not None:
            store.close()
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
