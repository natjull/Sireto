from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts/build_v412_fresh_s0_r2_fixture.py"
CORE_BUILDER_PATH = (
    REPOSITORY_ROOT
    / "scripts/build_v412_fresh_intake_synthetic_fixture.py"
)
CORE_PLAN_PATH = (
    REPOSITORY_ROOT
    / "config/v4_12_fresh_intake_synthetic_scanner_sealer_plan.json"
)
AUTHORITATIVE_PLAN_PATH = (
    REPOSITORY_ROOT
    / "config/v4_12_fresh_s0_authoritative_run_plan.json"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = _load("v412_s0_r2_fixture", SCRIPT)
core = _load("v412_s0_core_fixture_for_r2_tests", CORE_BUILDER_PATH)


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


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _test_plan(tmp_path: Path) -> tuple[Path, Path]:
    plan = json.loads(AUTHORITATIVE_PLAN_PATH.read_bytes())
    source_receipt = Path(
        plan["r2_successor"]["predecessor_receipt_absolute_path"]
    )
    receipt = tmp_path / "r1-receipt.json"
    receipt_bytes = source_receipt.read_bytes()
    assert _sha(receipt_bytes) == (
        plan["r2_successor"]["predecessor_receipt_sha256"]
    )
    receipt.write_bytes(receipt_bytes)
    plan["r2_successor"]["predecessor_receipt_absolute_path"] = str(receipt)
    planned_root = tmp_path / "r2"
    plan["r2_successor"]["root"] = str(planned_root)
    path = tmp_path / "authoritative-plan.json"
    path.write_bytes(_canonical(plan))
    return path, planned_root


def _read_tree(package: Path, control: Path) -> dict[str, bytes]:
    return {
        **{path.name: path.read_bytes() for path in package.iterdir()},
        "fixture_control_manifest.json": control.read_bytes(),
    }


def _without_run_id(value: dict) -> dict:
    projected = copy.deepcopy(value)
    projected.pop("synthetic_run_id")
    return projected


def test_preregistered_production_run_id_is_exact_and_distinct():
    plan = json.loads(AUTHORITATIVE_PLAN_PATH.read_bytes())
    successor = plan["r2_successor"]
    assert subject.opaque_digest(
        successor["run_derivation"]["domain"],
        successor["run_derivation"]["values"],
    ) == (
        "bjpoibmapghmeklagcnddeamijgmlfijmifdobbmmanmohkknplbpolonjfjahlo"
    )
    assert subject.EXPECTED_R2_RUN_ID != successor["predecessor_run_id"]


def test_r2_adapter_reuses_core_payloads_and_only_changes_allowlisted_fields(
    tmp_path,
):
    r1_root = tmp_path / "r1"
    r1 = core.build_fixture(CORE_PLAN_PATH, r1_root)
    test_plan, r2_root = _test_plan(tmp_path)
    r2 = subject.build_fixture(
        root=r2_root, authoritative_plan_path=test_plan
    )

    r1_tree = _read_tree(
        Path(r1["package_path"]), Path(r1["control_manifest_path"])
    )
    r2_tree = _read_tree(
        Path(r2["package_path"]), Path(r2["control_manifest_path"])
    )
    assert set(r1_tree) == set(r2_tree)
    assert r1_tree["crm_safe.csv"] == r2_tree["crm_safe.csv"]
    assert (
        r1_tree["evidence_source.parquet"]
        == r2_tree["evidence_source.parquet"]
    )

    for name in (
        "collection_source_manifest.json",
        "source_manifest.json",
        "evidence_source_manifest.json",
    ):
        r1_manifest = json.loads(r1_tree[name])
        r2_manifest = json.loads(r2_tree[name])
        assert _without_run_id(r1_manifest) == _without_run_id(r2_manifest)
        assert r1_manifest["synthetic_run_id"] == r1["synthetic_run_id"]
        assert r2_manifest["synthetic_run_id"] == r2["synthetic_run_id"]

    r1_control = json.loads(r1_tree["fixture_control_manifest.json"])
    r2_control = json.loads(r2_tree["fixture_control_manifest.json"])
    allowed_control_changes = {
        "synthetic_run_id",
        "collection_source_manifest_sha256",
        "source_manifest_sha256",
        "evidence_source_manifest_sha256",
    }
    assert {
        key
        for key in r1_control
        if r1_control[key] != r2_control[key]
    } == allowed_control_changes
    assert r2["synthetic_run_id"] != r1["synthetic_run_id"]
    assert (
        r2["synthetic_run_id"]
        != json.loads(test_plan.read_bytes())["r2_successor"][
            "predecessor_run_id"
        ]
    )
    assert (
        r2["attempt_id"]
        != json.loads(test_plan.read_bytes())["r2_successor"][
            "predecessor_attempt_id"
        ]
    )


def test_r2_adapter_rejects_mutated_receipt_and_core_pins(
    tmp_path,
):
    test_plan, planned_root = _test_plan(tmp_path)
    plan = json.loads(test_plan.read_bytes())
    receipt = Path(
        plan["r2_successor"]["predecessor_receipt_absolute_path"]
    )
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(
        subject.R2FixtureBuildError, match="receipt hash mismatch"
    ):
        subject.build_fixture(
            root=planned_root, authoritative_plan_path=test_plan
        )

    plan = json.loads(test_plan.read_bytes())
    plan["core"]["pins"]["fixture_builder"]["sha256"] = "0" * 64
    test_plan.write_bytes(_canonical(plan))
    with pytest.raises(
        subject.R2FixtureBuildError, match="core fixture builder hash mismatch"
    ):
        subject.build_fixture(
            root=planned_root, authoritative_plan_path=test_plan
        )


def test_r2_adapter_requires_the_exact_root_from_the_plan(tmp_path):
    test_plan, planned_root = _test_plan(tmp_path)
    wrong_root = tmp_path / "wrong-r2"
    with pytest.raises(
        subject.R2FixtureBuildError,
        match="requested R2 root differs from the authoritative plan",
    ):
        subject.build_fixture(
            root=wrong_root, authoritative_plan_path=test_plan
        )
    assert not planned_root.exists()
    assert not wrong_root.exists()


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        (
            "run_derivation",
            "domain",
            "SIRETO-V412-WRONG-RUN-DOMAIN\0",
            "run derivation authority mismatch",
        ),
        (
            "run_derivation",
            "projection",
            [
                "core_plan_sha256",
                "fixture_spec_sha256",
                "predecessor_receipt_sha256",
            ],
            "run derivation authority mismatch",
        ),
        (
            "attempt_derivation",
            "domain",
            "SIRETO-V412-WRONG-ATTEMPT-DOMAIN\0",
            "attempt derivation authority mismatch",
        ),
        (
            "attempt_derivation",
            "projection",
            [
                "fixture_control_manifest_sha256",
                "synthetic_run_id",
                "logical_time_utc",
            ],
            "attempt derivation authority mismatch",
        ),
    ],
)
def test_r2_adapter_rejects_mutated_domains_and_projections(
    tmp_path, section, field, replacement, message
):
    test_plan, planned_root = _test_plan(tmp_path)
    plan = json.loads(test_plan.read_bytes())
    plan["r2_successor"][section][field] = replacement
    test_plan.write_bytes(_canonical(plan))
    with pytest.raises(subject.R2FixtureBuildError, match=message):
        subject.build_fixture(
            root=planned_root, authoritative_plan_path=test_plan
        )
    assert not planned_root.exists()


def test_r2_adapter_rejects_unsafe_receipt_and_any_mutated_core_pin(
    tmp_path,
):
    test_plan, planned_root = _test_plan(tmp_path)
    plan = json.loads(test_plan.read_bytes())
    receipt = Path(
        plan["r2_successor"]["predecessor_receipt_absolute_path"]
    )
    receipt.chmod(0o666)
    with pytest.raises(
        subject.R2FixtureBuildError,
        match="pinned R1 receipt identity or permissions are unsafe",
    ):
        subject.build_fixture(
            root=planned_root, authoritative_plan_path=test_plan
        )
    assert not planned_root.exists()

    receipt.chmod(0o600)
    plan["core"]["pins"]["scanner"]["sha256"] = "f" * 64
    test_plan.write_bytes(_canonical(plan))
    with pytest.raises(
        subject.R2FixtureBuildError, match="pinned core scanner hash mismatch"
    ):
        subject.build_fixture(
            root=planned_root, authoritative_plan_path=test_plan
        )
    assert not planned_root.exists()
