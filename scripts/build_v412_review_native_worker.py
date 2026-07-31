#!/usr/bin/env python3
"""Build the pinned V4.12 native worker deterministically and without network."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/native/v412_review_worker.c"
CLANG = Path("/Library/Developer/CommandLineTools/usr/bin/clang")
LD = Path("/Library/Developer/CommandLineTools/usr/bin/ld")
SDK = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
SDK_SETTINGS = SDK / "SDKSettings.json"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
CODESIGN = Path("/usr/bin/codesign")
CANONICAL_BASENAME = "v412_review_native_worker_r31"
SCHEMA = "sireto-v4.12-r30-native-worker-build-2"
ARCH = "arm64"
BUILD_FLAGS = (
    "-isysroot", os.fspath(SDK),
    "-arch", ARCH,
    "-std=c11",
    "-O2",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wno-deprecated-declarations",
    "-Wl,-no_adhoc_codesign",
)
SIGN_FLAGS = (
    "--force", "--sign", "-", "--identifier", "com.sireto.v412.review-worker",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build(destination: Path) -> dict[str, object]:
    destination = destination.resolve()
    if destination.name != CANONICAL_BASENAME:
        raise RuntimeError(f"native worker basename must be {CANONICAL_BASENAME!r}")
    if platform.machine() != ARCH:
        raise RuntimeError(f"native worker build requires {ARCH}")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".building")
    receipt_path = destination.with_suffix(destination.suffix + ".json")
    if temporary.exists() or destination.exists() or receipt_path.exists():
        raise RuntimeError("native worker destination or receipt already exists")

    builder_path = Path(__file__).resolve()
    inputs_before = {
        "builder_sha256": sha256(builder_path),
        "clang_sha256": sha256(CLANG),
        "codesign_sha256": sha256(CODESIGN),
        "ld_sha256": sha256(LD),
        "sandbox_exec_sha256": sha256(SANDBOX_EXEC),
        "sdk_settings_sha256": sha256(SDK_SETTINGS),
        "source_sha256": sha256(SOURCE),
    }
    subprocess.run(
        [os.fspath(CLANG), *BUILD_FLAGS, os.fspath(SOURCE), "-o", os.fspath(temporary)],
        check=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "ZERO_AR_DATE": "1"},
    )
    subprocess.run(
        [os.fspath(CODESIGN), *SIGN_FLAGS, os.fspath(temporary)],
        check=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    inputs_after = {
        "builder_sha256": sha256(builder_path),
        "clang_sha256": sha256(CLANG),
        "codesign_sha256": sha256(CODESIGN),
        "ld_sha256": sha256(LD),
        "sandbox_exec_sha256": sha256(SANDBOX_EXEC),
        "sdk_settings_sha256": sha256(SDK_SETTINGS),
        "source_sha256": sha256(SOURCE),
    }
    if inputs_before != inputs_after:
        raise RuntimeError("build input changed during native compilation")

    os.chmod(temporary, 0o500)
    _fsync_file(temporary)
    temporary.rename(destination)
    _fsync_directory(destination.parent)
    receipt: dict[str, object] = {
        "architecture": ARCH,
        "artifact_sha256": sha256(destination),
        "basename": CANONICAL_BASENAME,
        "build_flags": list(BUILD_FLAGS),
        "sign_flags": list(SIGN_FLAGS),
        **inputs_after,
        "schema_version": SCHEMA,
    }
    descriptor = os.open(
        receipt_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeError("short receipt write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(destination.parent)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.destination), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
