from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
LOCK_PATH = (
    REPOSITORY / "config/v4_13_synthetic_intake_implementation_lock.json"
)
HEX64 = set("0123456789abcdef")


def _git_file(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout


def test_synthetic_intake_lock_is_scoped_and_pins_double_go_commit() -> None:
    lock = json.loads(LOCK_PATH.read_bytes())
    assert set(lock) == {
        "schema_version",
        "status",
        "implementation_commit",
        "implementation_blobs",
        "audit_verdicts",
        "runtime",
        "scope",
        "test_evidence",
    }
    assert (
        lock["schema_version"]
        == "sireto-v4.13-synthetic-intake-implementation-lock-1"
    )
    assert lock["status"] == "GO_V413_SYNTHETIC_INTAKE_IMPLEMENTATION_ONLY"
    commit = lock["implementation_commit"]
    assert len(commit) == 40 and set(commit) <= HEX64
    assert lock["audit_verdicts"] == [
        {
            "agent": "audit_r3_gate_code",
            "audited_commit": commit,
            "verdict": "GO_V413_SYNTHETIC_INTAKE_IMPLEMENTATION",
        },
        {
            "agent": "audit_r3_gate_tests",
            "audited_commit": commit,
            "verdict": "GO_V413_SYNTHETIC_INTAKE_IMPLEMENTATION",
        },
    ]
    assert "real_collection_gate_0a" in lock["scope"]["forbidden"]
    assert "retrieval_dev_or_test" in lock["scope"]["forbidden"]
    assert "ranker_or_acceptor_training" in lock["scope"]["forbidden"]
    assert "final_test" in lock["scope"]["forbidden"]
    assert lock["test_evidence"] == {
        "command": "pytest -q tests/test_v413_*.py",
        "passed": 105,
    }

    for path, expected_sha256 in lock["implementation_blobs"].items():
        assert len(expected_sha256) == 64 and set(expected_sha256) <= HEX64
        assert hashlib.sha256(_git_file(commit, path)).hexdigest() == expected_sha256
