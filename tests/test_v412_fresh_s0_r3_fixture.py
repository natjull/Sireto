from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts/build_v412_fresh_s0_r3_fixture.py"
PLAN_PATH = REPOSITORY / "config/v4_12_fresh_s0_r3_plan.json"
CORE_BUILDER_PATH = (
    REPOSITORY / "scripts/build_v412_fresh_intake_synthetic_fixture.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = _load("v412_s0_r3_fixture", SCRIPT)
core = _load("v412_s0_core_fixture_for_r3_tests", CORE_BUILDER_PATH)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _test_plan(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN_PATH.read_bytes())
    predecessor = plan["predecessor"]
    source = Path(predecessor["receipt_path"])
    receipt = tmp_path / "r2-receipt.json"
    receipt.write_bytes(source.read_bytes())
    predecessor["receipt_path"] = str(receipt)
    root = tmp_path / "r3"
    plan["paths"]["allowed_root"] = str(root)
    plan["r3_successor"]["root"] = str(root)
    plan_path = tmp_path / "r3-plan.json"
    plan_path.write_bytes(_canonical(plan))
    return plan_path, root


def _tree(result: dict) -> dict[str, bytes]:
    package = Path(result["package_path"])
    control = Path(result["control_manifest_path"])
    return {
        **{path.name: path.read_bytes() for path in package.iterdir()},
        control.name: control.read_bytes(),
    }


def test_preregistered_r3_identity_and_control_are_exact() -> None:
    plan = json.loads(PLAN_PATH.read_bytes())
    identity = plan["execution_identity"]
    assert subject.opaque_digest(
        identity["run"]["domain"], identity["run"]["values"]
    ) == subject.EXPECTED_RUN_ID
    assert subject.opaque_digest(
        identity["attempt"]["domain"], identity["attempt"]["values"]
    ) == subject.EXPECTED_ATTEMPT_ID
    assert identity["attempt"]["values"][
        "fixture_control_manifest_sha256"
    ] == subject.EXPECTED_CONTROL_SHA256


def test_r3_build_uses_only_the_pinned_core_fixture(tmp_path: Path) -> None:
    plan_path, root = _test_plan(tmp_path)
    result = subject.build_fixture(root=root, plan_path=plan_path)
    tree = _tree(result)
    core_plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_intake_synthetic_scanner_sealer_plan.json"
        ).read_bytes()
    )
    csv_raw = core_plan["fixture"]["csv"]["exact_utf8_text"].encode("utf-8")
    evidence_raw = core._empty_evidence_parquet(core_plan)
    expected = core._manifest_objects(
        core_plan,
        subject.EXPECTED_RUN_ID,
        csv_raw,
        evidence_raw,
    )
    for name, raw in expected.items():
        assert tree[name] == raw
    assert tree["crm_safe.csv"] == csv_raw
    assert tree["evidence_source.parquet"] == evidence_raw
    assert _sha(tree["fixture_control_manifest.json"]) == (
        subject.EXPECTED_CONTROL_SHA256
    )
    assert result["synthetic_run_id"] == subject.EXPECTED_RUN_ID
    assert result["attempt_id"] == subject.EXPECTED_ATTEMPT_ID
    assert result["predecessor_receipt_sha256"] == (
        json.loads(plan_path.read_bytes())["predecessor"]["receipt_sha256"]
    )


def test_wrong_root_and_existing_root_stop_before_output(tmp_path: Path) -> None:
    plan_path, root = _test_plan(tmp_path)
    wrong = tmp_path / "wrong"
    with pytest.raises(
        subject.R3FixtureBuildError,
        match="requested R3 root differs",
    ):
        subject.build_fixture(root=wrong, plan_path=plan_path)
    assert not root.exists()
    assert not wrong.exists()

    root.mkdir()
    marker = root / "marker"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(Exception):
        subject.build_fixture(root=root, plan_path=plan_path)
    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("component", "field", "replacement", "message"),
    [
        ("run", "domain", "SIRETO-WRONG\0", "run identity"),
        (
            "run",
            "projection",
            [
                "core_plan_sha256",
                "fixture_spec_sha256",
                "predecessor_receipt_sha256",
            ],
            "run identity",
        ),
        ("run", "result", "a" * 64, "run identity"),
        ("attempt", "domain", "SIRETO-WRONG\0", "attempt identity"),
        (
            "attempt",
            "projection",
            [
                "fixture_control_manifest_sha256",
                "synthetic_run_id",
                "logical_time_utc",
            ],
            "attempt identity",
        ),
        ("attempt", "result", "a" * 64, "attempt identity"),
        (
            "run",
            "values",
            {
                "fixture_spec_sha256": "0" * 64,
                "core_plan_sha256": "0" * 64,
                "predecessor_receipt_sha256": "0" * 64,
            },
            "run identity",
        ),
        (
            "attempt",
            "values",
            {
                "synthetic_run_id": subject.EXPECTED_RUN_ID,
                "fixture_control_manifest_sha256": (
                    subject.EXPECTED_CONTROL_SHA256
                ),
                "logical_time_utc": "2026-07-31T00:00:00Z",
            },
            "attempt identity",
        ),
    ],
)
def test_identity_mutations_stop_before_root(
    tmp_path: Path,
    component: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    plan_path, root = _test_plan(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    plan["execution_identity"][component][field] = replacement
    plan_path.write_bytes(_canonical(plan))
    with pytest.raises(subject.R3FixtureBuildError, match=message):
        subject.build_fixture(root=root, plan_path=plan_path)
    assert not root.exists()


@pytest.mark.parametrize(
    ("component", "value_key", "replacement"),
    [
        ("run", "fixture_spec_sha256", "0" * 64),
        ("run", "core_plan_sha256", "0" * 64),
        ("run", "predecessor_receipt_sha256", "0" * 64),
        ("attempt", "synthetic_run_id", "a" * 64),
        ("attempt", "fixture_control_manifest_sha256", "0" * 64),
        ("attempt", "logical_time_utc", "2026-07-31T00:00:00Z"),
    ],
)
def test_each_identity_value_mutation_stops_before_root(
    tmp_path: Path,
    component: str,
    value_key: str,
    replacement: str,
) -> None:
    plan_path, root = _test_plan(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    plan["execution_identity"][component]["values"][value_key] = replacement
    plan_path.write_bytes(_canonical(plan))
    with pytest.raises(
        subject.R3FixtureBuildError,
        match=f"{component} identity",
    ):
        subject.build_fixture(root=root, plan_path=plan_path)
    assert not root.exists()


def test_identity_schema_mutation_stops_before_root(tmp_path: Path) -> None:
    plan_path, root = _test_plan(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    plan["execution_identity"]["schema_version"] = "invalid"
    plan_path.write_bytes(_canonical(plan))
    with pytest.raises(
        subject.R3FixtureBuildError,
        match="execution identity schema mismatch",
    ):
        subject.build_fixture(root=root, plan_path=plan_path)
    assert not root.exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("synthetic_run_id", "a" * 64),
        ("attempt_id", "a" * 64),
        ("phase", "WORKER"),
        ("reason_code", "OK"),
        ("terminal_result", "INGESTED_SYNTHETIC_SCANNER_SEALER_V412"),
        ("verdict", "INGESTED"),
        ("schema_version", "invalid"),
    ],
)
def test_hostile_predecessor_receipt_fields_are_rejected(
    tmp_path: Path, field: str, replacement: object
) -> None:
    plan_path, _root = _test_plan(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    predecessor = plan["predecessor"]
    receipt_path = Path(predecessor["receipt_path"])
    receipt = json.loads(receipt_path.read_bytes())
    receipt[field] = replacement
    raw = _canonical(receipt)
    receipt_path.write_bytes(raw)
    predecessor = copy.deepcopy(predecessor)
    predecessor["receipt_sha256"] = _sha(raw)
    r2_plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s0_authoritative_run_plan.json"
        ).read_bytes()
    )
    with pytest.raises(
        subject.R3FixtureBuildError,
        match="receipt (schema mismatch|terminal authority mismatch)",
    ):
        subject._verify_predecessor(predecessor, r2_plan=r2_plan)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("run_id", "a" * 64),
        ("attempt_id", "a" * 64),
        ("reason_code", "OK"),
        ("terminal_result", "INGESTED_SYNTHETIC_SCANNER_SEALER_V412"),
        ("verdict", "INGESTED"),
    ],
)
def test_hostile_predecessor_authority_fields_are_rejected(
    tmp_path: Path, field: str, replacement: object
) -> None:
    plan_path, _root = _test_plan(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    predecessor = copy.deepcopy(plan["predecessor"])
    predecessor[field] = replacement
    r2_plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s0_authoritative_run_plan.json"
        ).read_bytes()
    )
    with pytest.raises(
        subject.R3FixtureBuildError,
        match="predecessor (receipt terminal authority|constants) mismatch",
    ):
        subject._verify_predecessor(predecessor, r2_plan=r2_plan)


def test_hostile_predecessor_path_hash_permissions_and_json(
    tmp_path: Path,
) -> None:
    plan_path, root = _test_plan(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    receipt = Path(plan["predecessor"]["receipt_path"])

    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(
        subject.R3FixtureBuildError, match="receipt hash mismatch"
    ):
        subject.build_fixture(root=root, plan_path=plan_path)
    assert not root.exists()

    plan_path, root = _test_plan(tmp_path / "unsafe")
    plan = json.loads(plan_path.read_bytes())
    receipt = Path(plan["predecessor"]["receipt_path"])
    receipt.chmod(0o666)
    with pytest.raises(
        subject.R3FixtureBuildError, match="permissions are unsafe"
    ):
        subject.build_fixture(root=root, plan_path=plan_path)
    assert not root.exists()

    plan_path, _root = _test_plan(tmp_path / "noncanonical")
    plan = json.loads(plan_path.read_bytes())
    predecessor = plan["predecessor"]
    receipt = Path(predecessor["receipt_path"])
    raw = receipt.read_bytes() + b" "
    receipt.write_bytes(raw)
    predecessor = copy.deepcopy(predecessor)
    predecessor["receipt_sha256"] = _sha(raw)
    r2_plan = json.loads(
        (
            REPOSITORY
            / "config/v4_12_fresh_s0_authoritative_run_plan.json"
        ).read_bytes()
    )
    with pytest.raises(
        subject.R3FixtureBuildError, match="not canonical JSON"
    ):
        subject._verify_predecessor(predecessor, r2_plan=r2_plan)

    missing = copy.deepcopy(predecessor)
    missing["receipt_path"] = str(tmp_path / "absent-receipt.json")
    with pytest.raises(
        subject.R3FixtureBuildError, match="cannot be opened safely"
    ):
        subject._verify_predecessor(missing, r2_plan=r2_plan)


def test_mutated_core_or_contract_pin_stops_before_root(tmp_path: Path) -> None:
    plan_path, root = _test_plan(tmp_path)
    plan = json.loads(plan_path.read_bytes())
    plan["base_authorities"]["core_plan"]["sha256"] = "0" * 64
    plan_path.write_bytes(_canonical(plan))
    with pytest.raises(
        subject.R3FixtureBuildError, match="base core_plan hash mismatch"
    ):
        subject.build_fixture(root=root, plan_path=plan_path)
    assert not root.exists()

    plan_path, root = _test_plan(tmp_path / "contract")
    plan = json.loads(plan_path.read_bytes())
    plan["contract"]["sha256"] = "f" * 64
    plan_path.write_bytes(_canonical(plan))
    with pytest.raises(
        subject.R3FixtureBuildError, match="R3 contract hash mismatch"
    ):
        subject.build_fixture(root=root, plan_path=plan_path)
    assert not root.exists()


def test_authoritative_r3_root_is_never_created_by_tests() -> None:
    assert not subject.R3_ROOT.exists()
