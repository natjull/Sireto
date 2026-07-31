from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
LOCK_PATH = (
    REPOSITORY / "config/v4_13_synthetic_intake_implementation_lock.json"
)
HEX64 = set("0123456789abcdef")
IMPLEMENTATION_COMMIT = "280043d46ac6ccf0f1ee1c16c56be97a68c971a6"
EXPECTED_BLOBS = {
    "docs/v4_13_fresh_labels_minimal_contract.md": (
        "ec11f9466029a19f394fe8b1c93997ebff8b617ad45c617c22cecd02fcc8c7b4"
    ),
    "config/v4_13_fresh_labels_preregistration_lock.json": (
        "b0648ae9ccdce25b9d0562328ce237a121503b95dcbc88f1e117c755186b6878"
    ),
    "scripts/audit_v413_fresh_source_availability.py": (
        "e8a6016803536d1679ad1720f575955dac819513bc908a7c40e0962d36eff24f"
    ),
    "scripts/audit_v413_synthetic_contamination.py": (
        "73977f6c5e0a9a7b4870e61303ad44acb3dcc742a5d527c3c43430fb483cdde0"
    ),
    "scripts/build_v413_fresh_qualification.py": (
        "221b336646d80bcc0a40a48c6f32e5e64bbc11226bcd944190616abcc0ce75ac"
    ),
    "scripts/open_v413_synthetic_qualification.py": (
        "b64d4865ac85d26e58f5cd548f46e415ec4ec1fac2e79813d631a8385a6daa0c"
    ),
    "scripts/seal_v413_fresh_splits.py": (
        "d7ab400543b541d36f4d8a6ce600bddfa1293fbd6c0af752c0c4054ec91aa711"
    ),
    "scripts/validate_v413_fresh_artifacts.py": (
        "8e4418407e1a3072aa4c5d09da3c004fd1d41fedeb168ed715ff52efdc2cffa0"
    ),
    "tests/test_v413_fresh_artifact_validator.py": (
        "39a5c003212b8736298550d34e38920d312157e359d411d084a377e9a8312960"
    ),
    "tests/test_v413_fresh_labels_minimal_plan.py": (
        "af96aef5e8597e3a8577607915b2efc956738fec76b97f253e81f43ee3226f42"
    ),
    "tests/test_v413_fresh_labels_preregistration_lock.py": (
        "757263e20741b2e6049ba49f177496060f8542eb44e853413053feb91a69c1e5"
    ),
    "tests/test_v413_fresh_qualification.py": (
        "e58e9a0cef5b264bba230f7a20baffb792a2a74973ba6d503bce41e6cf9778a9"
    ),
    "tests/test_v413_fresh_source_availability.py": (
        "a6eab43a6aa82914b9f2de54929d83fee133e5dc0456a6f978d4f589dfb39fc9"
    ),
    "tests/test_v413_fresh_split_sealer.py": (
        "e2e9e4b0c307b1b60f6cebd2e14f11d20753353de691a8e98d702ecb651d6d78"
    ),
    "tests/test_v413_fresh_synthetic_pipeline.py": (
        "1dd2c9d0beb3a705c4d436c48d12a7557deaf8250bedbd1a9d8a0d83bafb352b"
    ),
    "tests/test_v413_synthetic_contamination.py": (
        "9dc279a132e9acec355bd2f13539ac0be558b6cb983c6710b1afbd6a1c96d5d2"
    ),
    "tests/test_v413_synthetic_gate0b.py": (
        "b57238b45efd4f77dda88e7d40864ef23c2ed3c852bc8993ae3d726906c8dce6"
    ),
}
EXPECTED_AUTHORIZED = [
    "synthetic_fixture_intake_tests",
    "synthetic_gate_0a",
    "synthetic_gate_0b",
    "synthetic_qualification",
    "synthetic_contamination_audit",
    "synthetic_split_sealing",
]
EXPECTED_FORBIDDEN = [
    "real_collection_gate_0a",
    "real_payload_gate_0b",
    "real_keychain_or_historical_registry_access",
    "retrieval_dev_or_test",
    "ranker_or_acceptor_training",
    "final_test",
]
EXPECTED_RUNTIME = {
    "architecture": "arm64",
    "operating_system": "macOS-26.5.2-arm64-arm-64bit-Mach-O",
    "python": "3.14.3",
}


def _git_file(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=REPOSITORY,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
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
    assert lock["implementation_commit"] == IMPLEMENTATION_COMMIT
    assert lock["implementation_blobs"] == EXPECTED_BLOBS
    assert lock["audit_verdicts"] == [
        {
            "agent": "audit_r3_gate_code",
            "audited_commit": IMPLEMENTATION_COMMIT,
            "verdict": "GO_V413_SYNTHETIC_INTAKE_IMPLEMENTATION",
        },
        {
            "agent": "audit_r3_gate_tests",
            "audited_commit": IMPLEMENTATION_COMMIT,
            "verdict": "GO_V413_SYNTHETIC_INTAKE_IMPLEMENTATION",
        },
    ]
    assert lock["runtime"] == EXPECTED_RUNTIME
    assert lock["scope"] == {
        "authorized": EXPECTED_AUTHORIZED,
        "forbidden": EXPECTED_FORBIDDEN,
    }
    assert lock["test_evidence"] == {
        "command": "pytest -q tests/test_v413_*.py",
        "passed": 105,
    }

    for path, expected_sha256 in EXPECTED_BLOBS.items():
        assert len(expected_sha256) == 64 and set(expected_sha256) <= HEX64
        assert (
            hashlib.sha256(_git_file(IMPLEMENTATION_COMMIT, path)).hexdigest()
            == expected_sha256
        )
