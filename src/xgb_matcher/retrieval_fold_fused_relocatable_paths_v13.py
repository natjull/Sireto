"""Fail-closed relocatable paths for atomic retrieval-v13 run promotion.

An OOF run is first built below an immutable staging root and the complete
directory is then renamed atomically to its canonical location.  Absolute
paths embedded in child manifests would retain the staging prefix and make the
promoted tree unreadable.  These helpers encode only paths whose source and
record roots are both descendants of the same sealed run root.

The helpers deliberately do not provide an absolute-path fallback.  External
authorities (configuration, source candidate, PREFIT and authorization) keep
their dedicated absolute records and must never flow through this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class RelocatablePathInvariantErrorV13(ValueError):
    """Raised when a purported run-internal path escapes its run root."""


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    sealed_root = Path(root).resolve()
    try:
        resolved.relative_to(sealed_root)
    except ValueError as error:
        raise RelocatablePathInvariantErrorV13(
            f"{label} escaped the retrieval-v13 run root"
        ) from error
    return resolved


def relocatable_record_path_v13(
    path: Path,
    *,
    record_root: Path,
    run_root: Path,
) -> str:
    """Encode one run-internal target relative to its manifest directory."""

    sealed_root = Path(run_root).resolve()
    base = _inside(record_root, sealed_root, label="record root")
    target = _inside(path, sealed_root, label="record target")
    value = os.path.relpath(target, start=base)
    if Path(value).is_absolute() or (base / value).resolve() != target:
        raise RelocatablePathInvariantErrorV13(
            "retrieval-v13 relocatable path did not round-trip"
        )
    return value


def resolve_relocatable_record_path_v13(
    value: Any,
    *,
    record_root: Path,
    run_root: Path,
) -> Path:
    """Resolve a run-relative manifest path and reject every escape/absolute."""

    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise RelocatablePathInvariantErrorV13(
            "retrieval-v13 internal record path must be non-empty and relative"
        )
    sealed_root = Path(run_root).resolve()
    base = _inside(record_root, sealed_root, label="record root")
    return _inside(base / value, sealed_root, label="record target")


__all__ = [
    "RelocatablePathInvariantErrorV13",
    "relocatable_record_path_v13",
    "resolve_relocatable_record_path_v13",
]
