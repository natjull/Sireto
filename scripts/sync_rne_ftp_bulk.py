#!/usr/bin/env python3
"""Resume and seal explicitly authorized bulk files from the INPI RNE FTP."""

from __future__ import annotations

import argparse
import ftplib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.official_source_sync import (  # noqa: E402
    Credentials,
    KeychainLocator,
    OfficialSyncError,
    canonical_json,
    read_keychain_secret,
    sha256_file,
)

FTP_SYNC_ERRORS = (OfficialSyncError, OSError, *ftplib.all_errors)


def _safe_remote_name(value: str) -> str:
    path = PurePosixPath(value)
    if not value.startswith("/") or path.name in {"", ".", ".."}:
        raise OfficialSyncError("remote bulk path must be an absolute FTP file path")
    return str(path)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _connect(host: str, port: int, credentials: Credentials) -> ftplib.FTP:
    client = ftplib.FTP(timeout=180)
    try:
        client.connect(host=host, port=port)
        client.login(user=credentials.username, passwd=credentials.password)
        client.set_pasv(True)
        return client
    except Exception as exc:
        try:
            client.close()
        except Exception:
            pass
        raise OfficialSyncError("RNE bulk FTP connection failed") from exc


def _remote_metadata(
    client: ftplib.FTP, remote_paths: list[str]
) -> list[dict[str, Any]]:
    result = []
    for remote_path in remote_paths:
        try:
            size = client.size(remote_path)
        except ftplib.all_errors as exc:
            raise OfficialSyncError(f"RNE bulk SIZE unavailable for {PurePosixPath(remote_path).name}") from exc
        if size is None or size < 1:
            raise OfficialSyncError(f"RNE bulk remote file is empty: {PurePosixPath(remote_path).name}")
        try:
            modified = client.sendcmd(f"MDTM {remote_path}").removeprefix("213 ").strip()
        except ftplib.all_errors:
            modified = ""
        result.append(
            {
                "name": PurePosixPath(remote_path).name,
                "remote_path": remote_path,
                "size_bytes": int(size),
                "modified": modified,
            }
        )
    return result


def _download_resumable(
    client: ftplib.FTP,
    *,
    remote_path: str,
    destination: Path,
    expected_size: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    offset = destination.stat().st_size if destination.exists() else 0
    if destination.suffix.lower() == ".zip" and offset >= expected_size:
        try:
            with zipfile.ZipFile(destination) as archive:
                if archive.testzip() is None:
                    return
        except (OSError, zipfile.BadZipFile):
            pass
    elif offset > expected_size:
        raise OfficialSyncError(
            f"partial RNE file exceeds advertised remote size: {destination.name}"
        )
    if offset == expected_size and destination.suffix.lower() != ".zip":
        return
    mode = "ab" if offset else "wb"
    with destination.open(mode) as output:
        client.retrbinary(
            f"RETR {remote_path}",
            output.write,
            blocksize=4 * 1024 * 1024,
            rest=offset or None,
        )
        output.flush()
        os.fsync(output.fileno())
    actual = destination.stat().st_size
    zip_complete = False
    if destination.suffix.lower() == ".zip" and actual >= expected_size:
        try:
            with zipfile.ZipFile(destination) as archive:
                zip_complete = archive.testzip() is None
        except (OSError, zipfile.BadZipFile):
            zip_complete = False
    if actual != expected_size and not zip_complete:
        raise OfficialSyncError(
            f"RNE bulk transfer incomplete for {destination.name}: "
            f"{actual}/{expected_size} advertised bytes"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="www.inpi.net")
    parser.add_argument("--port", type=int, default=21)
    parser.add_argument("--remote", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--keychain-service", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--allow-insecure-plaintext", action="store_true")
    args = parser.parse_args()
    if not args.allow_insecure_plaintext:
        print("RNE bulk sync refused: explicit plaintext acknowledgement is required", file=sys.stderr)
        return 2
    remote_paths = [_safe_remote_name(value) for value in args.remote]
    if len({PurePosixPath(value).name for value in remote_paths}) != len(remote_paths):
        print("RNE bulk sync refused: duplicate output filenames", file=sys.stderr)
        return 2

    secret = bytearray()
    client: ftplib.FTP | None = None
    try:
        locator = KeychainLocator(args.keychain_service, args.account)
        secret = read_keychain_secret(locator)
        credentials = Credentials.from_keychain_payload(secret, username=args.account)
        client = _connect(args.host, args.port, credentials)
        remote = _remote_metadata(client, remote_paths)
        advertised_remote = [dict(item) for item in remote]
        identity = {
            "schema_version": "sireto-rne-ftp-bulk-manifest-v1",
            "source": "rne-ftp-bulk",
            "host": args.host,
            "port": args.port,
            "remote": advertised_remote,
            "plaintext_explicitly_authorized": True,
            "credential_material_recorded": False,
        }
        build_id = hashlib.sha256(canonical_json(identity)).hexdigest()
        source_root = args.output_root / "rne-ftp-bulk"
        final = source_root / build_id[:16]
        if final.exists():
            print(final)
            return 0
        stage = source_root / f".{build_id[:16]}.partial"
        stage.mkdir(parents=True, exist_ok=True, mode=0o700)
        required = sum(
            max(0, int(item["size_bytes"]) - (stage / item["name"]).stat().st_size)
            if (stage / item["name"]).exists()
            else int(item["size_bytes"])
            for item in remote
        )
        free = shutil.disk_usage(args.output_root).free
        reserve = int(args.minimum_free_gib * 1024**3)
        if free - required < reserve:
            raise OfficialSyncError(
                f"insufficient disk for RNE bulk files: need {required} bytes plus {reserve} reserve, have {free}"
            )
        for item in remote:
            partial = stage / item["name"]
            advertised_size = int(item["size_bytes"])
            _download_resumable(
                client,
                remote_path=item["remote_path"],
                destination=partial,
                expected_size=advertised_size,
            )
            item["advertised_size_bytes"] = advertised_size
            item["size_bytes"] = partial.stat().st_size
            item["sha256"] = sha256_file(partial)
            print(
                json.dumps(
                    {
                        "downloaded": item["name"],
                        "size_bytes": item["size_bytes"],
                        "sha256": item["sha256"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        manifest = {
            **identity,
            "observed_payload": remote,
            "payload": [
                {
                    "name": item["name"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                }
                for item in remote
            ],
            "build_id": build_id,
            "complete": True,
        }
        (stage / "manifest.json").write_bytes(canonical_json(manifest))
        os.chmod(stage / "manifest.json", 0o600)
        for item in remote:
            _fsync_file(stage / item["name"])
        _fsync_file(stage / "manifest.json")
        source_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.rename(stage, final)
        descriptor = os.open(source_root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        print(final)
        return 0
    except FTP_SYNC_ERRORS as exc:
        print(f"RNE bulk sync failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            try:
                client.quit()
            except Exception:
                client.close()
        for index in range(len(secret)):
            secret[index] = 0


if __name__ == "__main__":
    raise SystemExit(main())
