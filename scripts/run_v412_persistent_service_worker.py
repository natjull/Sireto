#!/usr/bin/env python3
"""Run one frozen V4.11 or V4.12-G persistent worker process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v412_service_bundle import (  # noqa: E402
    _capture_exact,
    _path_chain,
    load_frozen_v412_service_bundle,
    successful_load_counts,
)
from src.xgb_matcher.v412_service_run import (  # noqa: E402
    CollectedServiceRun,
    SAFE_QUERY_COLUMNS,
    collect_persistent_run,
    peak_rss_bytes,
)
from src.xgb_matcher.v412_service_worker import (  # noqa: E402
    PersistentV412Worker,
)
from src.xgb_matcher.v412_service_execution_lock import (  # noqa: E402
    validate_execution_lock,
    validate_loaded_repository_modules,
)


SAFE_QUERIES = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/references/"
    "v4_12_service_parity/b4b7fef24c5e7036/queries.parquet"
)
SAFE_QUERIES_SHA256 = (
    "70ded26776bfd56c96501c6033e0e322a6dd11ed296c3309ad89bd9deec84cf9"
)
PARQUET_OPTIONS = {
    "compression": "zstd",
    "compression_level": 9,
    "data_page_version": "1.0",
    "row_group_size": 65_536,
    "use_dictionary": False,
    "version": "2.6",
    "write_statistics": True,
}


def _prepare_output(path: Path) -> Path:
    output = Path(path)
    if not output.is_absolute():
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: output path must be absolute"
        )
    parent = output.parent
    parent_chain = _path_chain(parent)
    metadata = os.stat(parent, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: output parent is not a directory"
        )
    os.mkdir(output, mode=0o700)
    current = os.stat(output, follow_symlinks=False)
    if not stat.S_ISDIR(current.st_mode):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: output is not a directory"
        )
    if _path_chain(parent) != parent_chain:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: output parent changed"
        )
    return output


def _write_parquet(path: Path, frame) -> dict[str, Any]:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        before = os.fstat(descriptor)
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            pq.write_table(table, handle, **PARQUET_OPTIONS)
            handle.flush()
            os.fsync(handle.fileno())
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or (current.st_dev, current.st_ino)
            != (before.st_dev, before.st_ino)
            or size != after.st_size
        ):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: output Parquet changed"
            )
    finally:
        os.close(descriptor)
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": size,
        "row_count": len(frame),
        "columns": list(frame.columns),
    }


def _write_run(
    output: Path,
    run: CollectedServiceRun,
    *,
    phase: str,
    run_nonce: str,
    execution_lock_sha256: str,
) -> None:
    frames = {
        "candidates_features.parquet": run.candidates,
        "ranker.parquet": run.ranker,
        "scenes.parquet": run.scenes,
        "acceptor.parquet": run.acceptor,
        "timings.parquet": run.timings,
    }
    if run.mode == "v412g":
        frames.update(
            {
                "query_evidence.parquet": run.query_evidence,
                "candidate_evidence.parquet": run.candidate_evidence,
                "guard.parquet": run.guard,
            }
        )
    files = {
        name: _write_parquet(output / name, frame)
        for name, frame in frames.items()
    }
    manifest = {
        **run.manifest,
        "peak_rss_bytes": peak_rss_bytes(),
        "network_denied": (
            os.environ.get("SIRETO_NETWORK_AUDIT_DENY") == "1"
        ),
        "phase": phase,
        "run_nonce": run_nonce,
        "execution_lock_sha256": execution_lock_sha256,
        "safe_queries_path": str(SAFE_QUERIES),
        "safe_queries_sha256": SAFE_QUERIES_SHA256,
        "files": files,
    }
    manifest_path = output / "manifest.json"
    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    manifest_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        manifest_flags |= os.O_NOFOLLOW
    descriptor = os.open(manifest_path, manifest_flags, 0o600)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: short manifest write"
            )
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        observed = os.read(descriptor, len(payload) + 1)
        if observed != payload:
            raise ValueError(
                "STOP_V412_SERVICE_INTEGRITY: manifest reseal mismatch"
            )
    finally:
        os.close(descriptor)


def run(
    mode: str,
    output_dir: Path,
    *,
    phase: str,
    run_nonce: str,
    execution_lock_sha256: str,
) -> None:
    if mode not in {"v411", "v412g"}:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: invalid worker mode"
        )
    if phase not in {"diagnostic", "gate"}:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: invalid worker phase"
        )
    if (
        len(run_nonce) != 64
        or any(character not in "0123456789abcdef" for character in run_nonce)
    ):
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: invalid parent run nonce"
        )
    if os.environ.get("SIRETO_NETWORK_AUDIT_DENY") != "1":
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: network-deny bootstrap absent"
        )
    lock, observed_lock_sha256 = validate_execution_lock(
        expected_sha256=execution_lock_sha256,
        verify_git=False,
    )
    if observed_lock_sha256 != execution_lock_sha256:
        raise AssertionError("validated execution-lock hash changed")
    validate_loaded_repository_modules(lock["source_hashes"])
    query_bytes = _capture_exact(SAFE_QUERIES, SAFE_QUERIES_SHA256)
    table = pq.read_table(pa.BufferReader(query_bytes))
    if table.column_names != list(SAFE_QUERY_COLUMNS) or table.num_rows != 1456:
        raise ValueError(
            "STOP_V412_SERVICE_INTEGRITY: safe query table changed"
        )
    queries = table.to_pandas()
    output = _prepare_output(output_dir)
    loads_before = successful_load_counts()
    with load_frozen_v412_service_bundle(
        include_evidence=(mode == "v412g")
    ) as bundle:
        loads_after = successful_load_counts()
        worker = PersistentV412Worker(bundle=bundle, mode=mode)
        collected = collect_persistent_run(
            worker=worker,
            queries=queries,
            model_load_count=loads_after[0] - loads_before[0],
            store_load_count=loads_after[1] - loads_before[1],
        )
    _write_run(
        output,
        collected,
        phase=phase,
        run_nonce=run_nonce,
        execution_lock_sha256=execution_lock_sha256,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("v411", "v412g"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("diagnostic", "gate"),
        required=True,
    )
    parser.add_argument("--run-nonce", required=True)
    parser.add_argument("--execution-lock-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(
            args.mode,
            args.output_dir,
            phase=args.phase,
            run_nonce=args.run_nonce,
            execution_lock_sha256=args.execution_lock_sha256,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 65
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
