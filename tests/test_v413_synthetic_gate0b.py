from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from scripts import audit_v413_fresh_source_availability as gate0a
from scripts.build_v413_fresh_qualification import CRM_COLUMNS, MAPPING_COLUMNS
from scripts.open_v413_synthetic_qualification import (
    Gate0BStop,
    MARKER_FILENAME,
    RECEIPT_FILENAME,
    open_synthetic_qualification,
)
from scripts.audit_v413_synthetic_contamination import ContaminationStop
from scripts.validate_v413_fresh_artifacts import ValidationStopped


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True)
    path.chmod(0o700)


def _csv(columns: list[str], rows: list[list[object]]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return stream.getvalue().encode()


def _manifest(crm_raw: bytes, mapping_raw: bytes) -> dict:
    return {
        "authority_catalog_id": "v413-independent-authorities-1",
        "collection_id": "TEST_GATE_0B",
        "created_at_utc": "2026-08-02T00:00:00Z",
        "crm_file": "crm_source.csv",
        "crm_format": "CSV",
        "crm_row_count": 1,
        "crm_sha256": hashlib.sha256(crm_raw).hexdigest(),
        "crm_size_bytes": len(crm_raw),
        "export_cutoff_utc": "2026-08-01T12:00:00Z",
        "export_id": "TEST_GATE_0B_EXPORT",
        "mapping_file": "authoritative_mapping.csv",
        "mapping_format": "CSV",
        "mapping_row_count": 1,
        "mapping_sha256": hashlib.sha256(mapping_raw).hexdigest(),
        "mapping_size_bytes": len(mapping_raw),
        "matching_based_exclusions": False,
        "period_end_utc": "2026-08-01T10:00:00Z",
        "period_start_utc": "2026-08-01T00:00:00Z",
        "plan_git_commit": gate0a.EXPECTED_PREREGISTRATION_COMMIT,
        "plan_sha256": gate0a.EXPECTED_PLAN_SHA256,
        "population_definition": "synthetic complete frame",
        "population_exclusions": [],
        "population_is_exhaustive": True,
        "preregistration_lock_sha256": gate0a.EXPECTED_LOCK_SHA256,
        "producer_id": "SYNTHETIC_TEST_ONLY",
        "reference_date": "2026-08-01",
        "schema_version": "sireto-v4.13-collection-manifest-1",
        "source_record_id_semantics": "SYNTHETIC_UNIQUE_ROW_ID",
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    inbox, control, output = (
        tmp_path / "inbox",
        tmp_path / "control",
        tmp_path / "output",
    )
    _private_dir(inbox)
    _private_dir(control)
    crm_raw = _csv(
        CRM_COLUMNS,
        [[
            "row-1",
            "",
            "2026-08-01T01:00:00Z",
            "2026-08-01",
            "Boulangerie des Lilas",
            "1 rue des Lilas",
            "75001",
            "Paris",
            "75101",
            "true",
        ]],
    )
    mapping_raw = _csv(
        MAPPING_COLUMNS,
        [[
            "row-1",
            "CONTRACT_OR_BILLING_SIRET",
            "TEST_ISSUER",
            "TEST_BILLING",
            "proof-1",
            "12345678900012",
            "123456789",
            "2026-01-01",
            "",
            "2026-07-01T00:00:00Z",
            "a" * 64,
            "false",
        ]],
    )
    manifest = _manifest(crm_raw, mapping_raw)
    manifest_raw = gate0a.canonical_json(manifest)
    digest = hashlib.sha256(manifest_raw).hexdigest()
    collection = inbox / f"{1:020d}_{digest}"
    _private_dir(collection)
    for name, raw in (
        ("collection_manifest.json", manifest_raw),
        ("crm_source.csv", crm_raw),
        ("authoritative_mapping.csv", mapping_raw),
    ):
        path = collection / name
        path.write_bytes(raw)
        path.chmod(0o600)
    gate0a.audit_synthetic_availability(
        inbox=inbox,
        control_root=control,
        stability_observer=lambda _: 60.0,
        synthetic_only=True,
    )
    return inbox, control, output, collection


def _catalog() -> dict:
    return {
        "real_collection_open_authorized": False,
        "synthetic_test_authorities": [
            {
                "authority_type": "CONTRACT_OR_BILLING_SIRET",
                "authority_issuer_id": "TEST_ISSUER",
                "authority_system_id": "TEST_BILLING",
                "test_only": True,
            }
        ],
    }


def _run(inbox: Path, control: Path, output: Path, **kwargs):
    keysets = kwargs.pop(
        "contamination_keysets",
        {
            "service_id": set(),
            "siret_masked": set(),
            "fuzzy_historical": set(),
            "consumed_sirens": set(),
        },
    )
    return open_synthetic_qualification(
        inbox=inbox,
        control_root=control,
        output_root=output,
        authority_catalog=_catalog(),
        contamination_keysets=keysets,
        synthetic_hmac_key=bytes(range(32)),
        synthetic_only=True,
        **kwargs,
    )


def test_real_gate0a_to_gate0b_outputs_and_idempotence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox, control, output, _ = _fixture(tmp_path)
    receipt = _run(inbox, control, output)
    assert receipt["terminal_state"] == "QUALIFICATION_SEALED"
    assert receipt["counts"]["MATCH_EXACT"] == 1
    assert (control / MARKER_FILENAME).is_file()
    assert (control / RECEIPT_FILENAME).is_file()
    assert (output / "queries/queries.csv").is_file()
    assert (output / "oracle/oracle.csv").is_file()
    assert (output / "audit/contamination.json").is_file()
    for split in ("fit", "dev", "test"):
        assert (output / f"splits/{split}/split_manifest.json").is_file()

    original_open = os.open

    def forbid_payload(path, *args, **kwargs):
        if os.fspath(path) in {"crm_source.csv", "authoritative_mapping.csv"}:
            raise AssertionError("idempotent receipt must not reopen payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(
        __import__(
            "scripts.open_v413_synthetic_qualification", fromlist=["os"]
        ).os,
        "open",
        forbid_payload,
    )
    assert _run(inbox, control, output) == receipt


def test_marker_is_durable_before_first_payload_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox, control, output, _ = _fixture(tmp_path)
    subject = __import__(
        "scripts.open_v413_synthetic_qualification", fromlist=["os"]
    )
    original_open = subject.os.open

    def inspect_open(path, *args, **kwargs):
        if os.fspath(path) in {"crm_source.csv", "authoritative_mapping.csv"}:
            assert (control / MARKER_FILENAME).is_file()
            with (control / MARKER_FILENAME).open("rb") as handle:
                os.fsync(handle.fileno())
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(subject.os, "open", inspect_open)
    _run(inbox, control, output)


def test_crash_after_marker_forbids_second_payload_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inbox, control, output, _ = _fixture(tmp_path)

    with pytest.raises(Gate0BStop, match="SYNTHETIC_CRASH_AFTER_MARKER"):
        _run(inbox, control, output, crash_stage="after_marker")

    subject = __import__(
        "scripts.open_v413_synthetic_qualification", fromlist=["os"]
    )
    original_open = subject.os.open

    def forbid_payload(path, *args, **kwargs):
        if os.fspath(path) in {"crm_source.csv", "authoritative_mapping.csv"}:
            raise AssertionError("incomplete prior opening must stop before payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(subject.os, "open", forbid_payload)
    with pytest.raises(Gate0BStop, match="INCOMPLETE_PRIOR_PAYLOAD_OPEN"):
        _run(inbox, control, output)


@pytest.mark.parametrize("mutation", ["hash", "symlink"])
def test_mutated_or_symlink_payload_stops(
    tmp_path: Path, mutation: str
) -> None:
    inbox, control, output, collection = _fixture(tmp_path)
    payload = collection / "crm_source.csv"
    if mutation == "hash":
        raw = bytearray(payload.read_bytes())
        raw[-1] = ord("X")
        payload.write_bytes(raw)
        payload.chmod(0o600)
    else:
        outside = tmp_path / "outside.csv"
        outside.write_bytes(payload.read_bytes())
        outside.chmod(0o600)
        payload.unlink()
        payload.symlink_to(outside)
    with pytest.raises((Gate0BStop, gate0a.AvailabilityStop)):
        _run(inbox, control, output)


def test_claim_metadata_tamper_is_rejected_before_marker(tmp_path: Path) -> None:
    inbox, control, output, _ = _fixture(tmp_path)
    claim_path = control / gate0a.CLAIM_FILENAME
    claim = json.loads(claim_path.read_bytes())
    claim["arrival_epoch_ns"] = 999
    claim_path.unlink()
    claim_path.write_bytes(gate0a.canonical_json(claim))
    claim_path.chmod(0o600)
    with pytest.raises(Gate0BStop, match="CLAIM_DIRECTORY_BINDING"):
        _run(inbox, control, output)
    assert not (control / MARKER_FILENAME).exists()


def test_post_receipt_claim_tamper_is_not_masked(tmp_path: Path) -> None:
    inbox, control, output, _ = _fixture(tmp_path)
    _run(inbox, control, output)
    claim_path = control / gate0a.CLAIM_FILENAME
    claim = json.loads(claim_path.read_bytes())
    claim["collection_id"] = "TAMPERED"
    claim_path.unlink()
    claim_path.write_bytes(gate0a.canonical_json(claim))
    claim_path.chmod(0o600)
    with pytest.raises(Gate0BStop, match="CLAIM_COLLECTION_BINDING"):
        _run(inbox, control, output)


def test_input_control_output_roots_must_be_pairwise_disjoint(
    tmp_path: Path,
) -> None:
    inbox, control, _, collection = _fixture(tmp_path)
    embedded = collection / "embedded-output"
    with pytest.raises(Gate0BStop, match="ROOTS_NOT_PAIRWISE_DISJOINT"):
        _run(inbox, control, embedded)
    assert len(list(collection.iterdir())) == 3
    assert not (control / MARKER_FILENAME).exists()


def test_forbidden_siren_overlap_stops_before_outputs(tmp_path: Path) -> None:
    inbox, control, output, _ = _fixture(tmp_path)
    with pytest.raises(ContaminationStop, match="forbidden historical overlap"):
        _run(
            inbox,
            control,
            output,
            contamination_keysets={
                "service_id": set(),
                "siret_masked": set(),
                "fuzzy_historical": set(),
                "consumed_sirens": {"123456789"},
            },
        )
    assert (control / MARKER_FILENAME).is_file()
    assert not (control / RECEIPT_FILENAME).exists()
    assert not output.exists()


@pytest.mark.parametrize(
    "target", ["query", "audit", "contamination", "split", "manifest"]
)
def test_final_semantic_revalidation_rejects_late_output_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    inbox, control, output, collection = _fixture(tmp_path)
    subject = __import__(
        "scripts.open_v413_synthetic_qualification",
        fromlist=["seal_manifests"],
    )
    original = subject.seal_manifests

    def seal_then_mutate(rows, root):
        result = original(rows, root)
        if target == "query":
            path = output / "queries/queries.csv"
            path.write_text(
                path.read_text().replace(
                    "Boulangerie des Lilas", "LEAK 12345678901234"
                )
            )
        elif target == "audit":
            path = output / "audit/qualification.json"
            value = json.loads(path.read_text())
            value["synthetic_fixtures_only"] = False
            value["unexpected"] = "same counts"
            path.write_bytes(gate0a.canonical_json(value))
        elif target == "contamination":
            path = output / "audit/contamination.json"
            value = json.loads(path.read_text())
            value["verdict"] = "BYPASSED"
            path.write_bytes(gate0a.canonical_json(value))
        elif target == "split":
            (output / "splits/fit/split_manifest.json").write_text("{}\n")
        else:
            path = collection / "collection_manifest.json"
            value = json.loads(path.read_text())
            value["created_at_utc"] = "2026-08-03T00:00:00Z"
            path.write_bytes(gate0a.canonical_json(value))
            path.chmod(0o600)
        return result

    monkeypatch.setattr(subject, "seal_manifests", seal_then_mutate)
    with pytest.raises((Gate0BStop, ValidationStopped)) as caught:
        _run(inbox, control, output)
    assert "FINAL_" in str(caught.value) or "digit leak" in str(caught.value)
    assert not (control / RECEIPT_FILENAME).exists()


def test_arbitrary_crash_callback_is_not_an_execution_surface(
    tmp_path: Path,
) -> None:
    inbox, control, output, _ = _fixture(tmp_path)
    with pytest.raises(TypeError, match="crash_hook"):
        _run(inbox, control, output, crash_hook=lambda _: None)
    assert not (control / MARKER_FILENAME).exists()
