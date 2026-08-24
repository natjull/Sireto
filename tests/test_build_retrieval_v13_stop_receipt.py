from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_retrieval_v13_stop_receipt as receipt_module


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(receipt_module.canonical_json_bytes(value) + b"\n")
    path.chmod(0o644)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    tree_root = tmp_path / "v13-stop"
    tree_root.mkdir(parents=True)
    marker_path = tmp_path / ".attempt.json"
    config_path = tmp_path / "config.json"
    _write_json(config_path, {"recipe": "frozen-v13-test"})
    config_sha256 = receipt_module.sha256_file(config_path)
    _write_json(
        marker_path,
        {
            "canonical_output_dir": str(tree_root.resolve()),
            "config_sha256": config_sha256,
            "marker_path": str(marker_path.resolve()),
            "retry_allowed": False,
            "schema_version": "sireto-retrieval-v13-one-shot-attempt-1",
            "status": "ATTEMPT_IRREVERSIBLY_CONSUMED_BEFORE_CALLBACK",
        },
    )
    marker_sha256 = receipt_module.sha256_file(marker_path)
    implementation = [
        {
            "path": f"src/frozen_{index:02d}.py",
            "sha256": hashlib.sha256(f"impl-{index}".encode()).hexdigest(),
            "size_bytes": index + 1,
        }
        for index in range(15)
    ]
    implementation_sha256 = receipt_module.sha256_json(implementation)
    monkeypatch.setattr(
        receipt_module, "EXPECTED_IMPLEMENTATION_SHA256", implementation_sha256
    )
    _write_json(
        tree_root / "fold_2" / "manifest.json",
        {
            "folds": [2],
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "execution_authorization_published": False,
        },
    )
    data_path = tree_root / "fold_2" / "outcomes.parquet"
    data_path.write_bytes(b"synthetic-test-evidence\x00\x01")
    data_path.chmod(0o640)
    _write_json(
        tree_root / "manifest.json",
        {
            "attempt_marker": {
                "path": str(marker_path.resolve()),
                "sha256": marker_sha256,
            },
            "automatic_retry_allowed": False,
            "config": {
                "path": str(config_path.resolve()),
                "sha256": config_sha256,
            },
            "exception": {
                "message_sanitized": "compact output logical content drift",
                "type": "FoldFusedInvariantError",
            },
            "execution_authorization_published": False,
            "fold0_opened": False,
            "fold1_opened": False,
            "folds": [2, 3, 4],
            "implementation_files": implementation,
            "phase": "MEASURED_W3_CALLBACK_OR_POSTCHECK",
            "positive_injection": False,
            "post_error_current_implementation_sha256": implementation_sha256,
            "schema_version": "sireto-retrieval-v13-execution-exception-1",
            "status": "STOP_V13_EXECUTION_EXCEPTION",
            "v13_implementation_sha256": implementation_sha256,
        },
    )
    return {
        "tree_root": tree_root,
        "marker_path": marker_path,
        "config_path": config_path,
        "marker_sha256": marker_sha256,
        "config_sha256": config_sha256,
        "implementation_sha256": implementation_sha256,
    }


def _build_for_inventory(
    fixture: dict[str, object],
    inventory: tuple[dict[str, object], ...],
) -> dict[str, object]:
    tree_root = Path(fixture["tree_root"])
    marker_path = Path(fixture["marker_path"])
    config_path = Path(fixture["config_path"])
    stop_path = tree_root / "manifest.json"
    return receipt_module.build_receipt(
        tree_root=tree_root,
        marker_path=marker_path,
        config_path=config_path,
        expected_tree_file_count=len(inventory),
        expected_tree_size_bytes=sum(int(row["size_bytes"]) for row in inventory),
        expected_tree_mode=receipt_module._mode(tree_root.lstat()),
        expected_inventory_sha256=receipt_module.sha256_json(list(inventory)),
        expected_inventory_merkle_root=receipt_module.inventory_merkle_root(
            inventory
        ),
        expected_marker_sha256=str(fixture["marker_sha256"]),
        expected_marker_size_bytes=marker_path.stat().st_size,
        expected_marker_mode=receipt_module._mode(marker_path.lstat()),
        expected_stop_sha256=receipt_module.sha256_file(stop_path),
        expected_stop_size_bytes=stop_path.stat().st_size,
        expected_stop_mode=receipt_module._mode(stop_path.lstat()),
        expected_config_sha256=str(fixture["config_sha256"]),
        expected_config_size_bytes=config_path.stat().st_size,
        expected_config_mode=receipt_module._mode(config_path.lstat()),
        expected_json_file_count=sum(
            str(row["relative_path"]).endswith(".json") for row in inventory
        ),
        expected_protected_manifest_count=2,
    )


def _build_current(fixture: dict[str, object]) -> dict[str, object]:
    tree_root = Path(fixture["tree_root"])
    return _build_for_inventory(fixture, receipt_module.inventory_tree(tree_root))


def _independent_merkle(inventory: list[dict[str, object]]) -> str:
    level = [
        hashlib.sha256(
            b"\x00" + receipt_module.canonical_json_bytes(record)
        ).digest()
        for record in inventory
    ]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(b"\x01" + level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def test_receipt_is_deterministic_exhaustive_and_content_addressed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    first = _build_current(fixture)
    second = _build_current(fixture)

    assert first == second
    inventory = first["inventory"]
    assert isinstance(inventory, list)
    assert [row["relative_path"] for row in inventory] == [
        "fold_2/manifest.json",
        "fold_2/outcomes.parquet",
        "manifest.json",
    ]
    assert set(inventory[0]) == {
        "relative_path",
        "size_bytes",
        "sha256",
        "mode",
    }
    assert inventory[1]["mode"] == "0640"
    assert first["source_tree"]["inventory_merkle_root"] == _independent_merkle(
        inventory
    )
    body = {key: value for key, value in first.items() if key != "receipt_id"}
    assert first["receipt_id"] == receipt_module.sha256_json(body)
    assert first["safety"] == {
        "json_file_count": 2,
        "protected_manifest_count": 2,
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "execution_authorization_present": False,
        "execution_authorization_published": False,
    }
    receipt_module.validate_receipt(first)


def test_publish_is_external_atomic_read_only_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    source_root = Path(fixture["tree_root"])
    before = {
        path.relative_to(source_root).as_posix(): (
            path.read_bytes(),
            receipt_module._mode(path.lstat()),
        )
        for path in source_root.rglob("*")
        if path.is_file()
    }
    receipt = _build_current(fixture)
    output_root = tmp_path / "receipts"

    manifest = receipt_module.publish_receipt(receipt, output_root=output_root)
    try:
        assert manifest == output_root / str(receipt["receipt_id"])[:16] / "manifest.json"
        assert receipt_module._mode(manifest.lstat()) == "0444"
        assert receipt_module._mode(manifest.parent.lstat()) == "0555"
        assert json.loads(manifest.read_text(encoding="utf-8")) == receipt
        assert (
            receipt_module.publish_receipt(receipt, output_root=output_root)
            == manifest
        )
        after = {
            path.relative_to(source_root).as_posix(): (
                path.read_bytes(),
                receipt_module._mode(path.lstat()),
            )
            for path in source_root.rglob("*")
            if path.is_file()
        }
        assert after == before
    finally:
        if manifest.exists():
            manifest.parent.chmod(0o755)
            manifest.chmod(0o644)


def test_tree_extra_file_and_linked_evidence_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    tree_root = Path(fixture["tree_root"])
    original = receipt_module.inventory_tree(tree_root)
    extra = tree_root / "unexpected.bin"
    extra.write_bytes(b"unexpected")
    with pytest.raises(receipt_module.V13StopReceiptError, match="count"):
        _build_for_inventory(fixture, original)

    extra.unlink()
    symlink = tree_root / "linked.bin"
    symlink.symlink_to(tree_root / "fold_2" / "outcomes.parquet")
    with pytest.raises(receipt_module.V13StopReceiptError, match="symlink"):
        receipt_module.inventory_tree(tree_root)
    symlink.unlink()

    hardlink = tree_root / "hardlinked.bin"
    os.link(tree_root / "fold_2" / "outcomes.parquet", hardlink)
    with pytest.raises(receipt_module.V13StopReceiptError, match="linked evidence"):
        receipt_module.inventory_tree(tree_root)


def test_same_count_and_size_content_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    tree_root = Path(fixture["tree_root"])
    original = receipt_module.inventory_tree(tree_root)
    data_path = tree_root / "fold_2" / "outcomes.parquet"
    original_bytes = data_path.read_bytes()
    data_path.write_bytes(b"X" * len(original_bytes))

    with pytest.raises(receipt_module.V13StopReceiptError, match="inventory"):
        _build_for_inventory(fixture, original)


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        (
            "fold_2/manifest.json",
            {
                "folds": [0, 2],
                "fold0_opened": True,
                "fold1_opened": False,
                "positive_injection": False,
                "execution_authorization_published": False,
            },
            "fold0",
        ),
        (
            "fold_2/manifest.json",
            {
                "folds": [2],
                "fold0_opened": False,
                "fold1_opened": False,
                "positive_injection": True,
                "execution_authorization_published": False,
            },
            "positive injection",
        ),
        (
            "execution_authorization.json",
            {"folds": [2, 3, 4]},
            "execution authorization",
        ),
    ],
)
def test_closed_folds_injection_and_authorization_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    payload: dict[str, object],
    message: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _write_json(Path(fixture["tree_root"]) / filename, payload)
    with pytest.raises(receipt_module.V13StopReceiptError, match=message):
        _build_current(fixture)


def test_implementation_projection_and_config_lineage_drift_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    stop_path = Path(fixture["tree_root"]) / "manifest.json"
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    stop["implementation_files"][0]["sha256"] = "f" * 64
    drifted = receipt_module.sha256_json(stop["implementation_files"])
    stop["v13_implementation_sha256"] = drifted
    stop["post_error_current_implementation_sha256"] = drifted
    _write_json(stop_path, stop)
    with pytest.raises(receipt_module.V13StopReceiptError, match="aggregate"):
        _build_current(fixture)

    fixture = _fixture(tmp_path / "config-drift", monkeypatch)
    stop_path = Path(fixture["tree_root"]) / "manifest.json"
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    stop["config"]["path"] = str(tmp_path / "wrong-config.json")
    _write_json(stop_path, stop)
    with pytest.raises(receipt_module.V13StopReceiptError, match="STOP contract"):
        _build_current(fixture)


@pytest.mark.parametrize("mutation", ("order", "duplicate", "merkle"))
def test_receipt_order_duplicate_and_merkle_drift_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = _build_current(fixture)
    if mutation == "order":
        receipt["inventory"] = list(reversed(receipt["inventory"]))
    elif mutation == "duplicate":
        receipt["inventory"] = [*receipt["inventory"], receipt["inventory"][0]]
    else:
        receipt["source_tree"]["inventory_merkle_root"] = "0" * 64
    body = {key: value for key, value in receipt.items() if key != "receipt_id"}
    receipt["receipt_id"] = receipt_module.sha256_json(body)

    with pytest.raises(receipt_module.V13StopReceiptError):
        receipt_module.validate_receipt(receipt)


def test_publish_refuses_content_address_collision_and_source_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    receipt = _build_current(fixture)
    output_root = tmp_path / "receipts"
    manifest = receipt_module.publish_receipt(receipt, output_root=output_root)
    try:
        manifest.parent.chmod(0o755)
        manifest.chmod(0o644)
        manifest.write_text("{}\n", encoding="utf-8")
        manifest.chmod(0o444)
        manifest.parent.chmod(0o555)
        with pytest.raises(receipt_module.V13StopReceiptError, match="collision"):
            receipt_module.publish_receipt(receipt, output_root=output_root)
        with pytest.raises(receipt_module.V13StopReceiptError, match="separate"):
            receipt_module.publish_receipt(
                receipt, output_root=Path(fixture["tree_root"]) / "receipt"
            )
    finally:
        manifest.parent.chmod(0o755)
        manifest.chmod(0o644)
