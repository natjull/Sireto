from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
LOCK_PATH = (
    REPOSITORY / "config/v4_13_fresh_labels_preregistration_lock.json"
)


def test_preregistration_lock_pins_exact_double_go_commit_and_blobs() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["schema_version"] == (
        "sireto-v4.13-fresh-labels-preregistration-lock-1"
    )
    assert lock["status"] == (
        "GO_V413_PREREGISTRATION_IMPLEMENT_SYNTHETIC_ONLY"
    )
    assert lock["real_collection_open_authorized"] is False
    assert lock["git_commit"] == (
        "bf4ed261ba80e0242bda8b46884fcf484c6c8e1b"
    )
    assert [audit["verdict"] for audit in lock["independent_audits"]] == [
        "GO_V413_PREREGISTRATION",
        "GO_V413_PREREGISTRATION",
    ]
    for key in ("plan", "plan_schema", "contract", "tests", "authority_catalog"):
        pin = lock[key]
        assert hashlib.sha256(
            (REPOSITORY / pin["path"]).read_bytes()
        ).hexdigest() == pin["sha256"]
    assert lock["authority_catalog"]["status"] == (
        "EMPTY_NO_REAL_AUTHORITY_REGISTERED"
    )


def test_locked_files_are_exactly_from_locked_commit() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    commit = lock["git_commit"]
    for key in ("plan", "plan_schema", "contract", "tests", "authority_catalog"):
        pin = lock[key]
        blob = subprocess.run(
            ["git", "show", f"{commit}:{pin['path']}"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == pin["sha256"]
