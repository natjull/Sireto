from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY / "scripts/audit_v413_fresh_source_availability.py"
SPEC = importlib.util.spec_from_file_location("v413_availability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def _write_private(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)
    path.chmod(0o600)


def _mkdir_private(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def _manifest(
    *,
    collection_id: str,
    crm_raw: bytes,
    mapping_raw: bytes,
) -> dict[str, object]:
    return {
        "authority_catalog_id": "v413-independent-authorities-1",
        "collection_id": collection_id,
        "created_at_utc": "2026-08-02T00:00:00Z",
        "crm_file": "crm_source.csv",
        "crm_format": "CSV",
        "crm_row_count": 2,
        "crm_sha256": hashlib.sha256(crm_raw).hexdigest(),
        "crm_size_bytes": len(crm_raw),
        "export_cutoff_utc": "2026-08-01T12:00:00Z",
        "export_id": f"export-{collection_id}",
        "mapping_file": "authoritative_mapping.csv",
        "mapping_format": "CSV",
        "mapping_row_count": 2,
        "mapping_sha256": hashlib.sha256(mapping_raw).hexdigest(),
        "mapping_size_bytes": len(mapping_raw),
        "matching_based_exclusions": False,
        "period_end_utc": "2026-08-01T10:00:00Z",
        "period_start_utc": "2026-08-01T00:00:00Z",
        "plan_git_commit": subject.EXPECTED_PREREGISTRATION_COMMIT,
        "plan_sha256": subject.EXPECTED_PLAN_SHA256,
        "population_definition": "synthetic exhaustive fixture",
        "population_exclusions": [],
        "population_is_exhaustive": True,
        "preregistration_lock_sha256": subject.EXPECTED_LOCK_SHA256,
        "producer_id": "SYNTHETIC_TEST_ONLY",
        "reference_date": "2026-08-01",
        "schema_version": "sireto-v4.13-collection-manifest-1",
        "source_record_id_semantics": "SYNTHETIC_UNIQUE_ROW_ID",
    }


def _collection(
    inbox: Path,
    *,
    arrival: int,
    collection_id: str,
) -> Path:
    crm_raw = b"source_record_id,crm_name_raw\n1,alpha\n"
    mapping_raw = b"source_record_id,authoritative_siret\n1,00000000000000\n"
    manifest = _manifest(
        collection_id=collection_id,
        crm_raw=crm_raw,
        mapping_raw=mapping_raw,
    )
    manifest_raw = subject.canonical_json(manifest)
    digest = hashlib.sha256(manifest_raw).hexdigest()
    root = inbox / f"{arrival:020d}_{digest}"
    _mkdir_private(root)
    _write_private(root / "collection_manifest.json", manifest_raw)
    _write_private(root / "crm_source.csv", crm_raw)
    _write_private(root / "authoritative_mapping.csv", mapping_raw)
    return root


@pytest.fixture
def synthetic_roots(tmp_path: Path) -> tuple[Path, Path]:
    inbox = tmp_path / "inbox"
    control = tmp_path / "control"
    _mkdir_private(inbox)
    _mkdir_private(control)
    return inbox, control


def _stable(_: float) -> float:
    return 60.0


def _run(inbox: Path, control: Path, **kwargs: object) -> dict[str, object]:
    return subject.audit_synthetic_availability(
        inbox=inbox,
        control_root=control,
        stability_observer=kwargs.pop("stability_observer", _stable),
        minimum_stability_seconds=60.0,
        synthetic_only=True,
        **kwargs,
    )


def test_selects_minimum_tuple_and_creates_durable_private_controls(
    synthetic_roots: tuple[Path, Path],
) -> None:
    inbox, control = synthetic_roots
    later = _collection(inbox, arrival=12, collection_id="later")
    earlier = _collection(inbox, arrival=7, collection_id="earlier")

    result = _run(inbox, control)

    assert result["directory_name"] == earlier.name
    assert result["directory_name"] != later.name
    assert result["claim_state"] == "MANIFEST_ONLY_NO_PAYLOAD_OPEN"
    assert result["synthetic_only"] is True
    for name in (subject.LEDGER_FILENAME, subject.CLAIM_FILENAME):
        path = control / name
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        raw = path.read_bytes()
        assert raw == subject.canonical_json(json.loads(raw))


def test_same_arrival_uses_manifest_hash_tie_break(
    synthetic_roots: tuple[Path, Path],
) -> None:
    inbox, control = synthetic_roots
    first = _collection(inbox, arrival=5, collection_id="same-a")
    second = _collection(inbox, arrival=5, collection_id="same-b")
    expected = min(first.name, second.name)

    result = _run(inbox, control)

    assert result["directory_name"] == expected


def test_two_concurrent_auditors_converge_on_one_claim(
    synthetic_roots: tuple[Path, Path],
) -> None:
    inbox, control = synthetic_roots
    _collection(inbox, arrival=1, collection_id="concurrent")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run, inbox, control) for _ in range(2)]
    results = [future.result() for future in futures]

    assert results[0] == results[1]
    assert len(list(control.glob(subject.CLAIM_FILENAME))) == 1
    assert len(list(control.glob(subject.LEDGER_FILENAME))) == 1


def test_payload_files_are_never_opened(
    synthetic_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inbox, control = synthetic_roots
    _collection(inbox, arrival=1, collection_id="only")
    opened: list[str] = []
    original_open = subject.os.open

    def recording_open(path: object, *args: object, **kwargs: object) -> int:
        opened.append(os.fspath(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(subject.os, "open", recording_open)
    _run(inbox, control)

    assert "collection_manifest.json" in opened
    assert "crm_source.csv" not in opened
    assert "authoritative_mapping.csv" not in opened


@pytest.mark.parametrize("kind", ["extra", "manifest_symlink", "payload_symlink", "hardlink"])
def test_rejects_unsafe_collection_entries(
    synthetic_roots: tuple[Path, Path],
    kind: str,
) -> None:
    inbox, control = synthetic_roots
    root = _collection(inbox, arrival=1, collection_id=kind)
    if kind == "extra":
        _write_private(root / "unexpected.txt", b"x")
    elif kind == "manifest_symlink":
        manifest = root / "collection_manifest.json"
        target = root / "manifest-target.json"
        manifest.rename(target)
        manifest.symlink_to(target.name)
    elif kind == "payload_symlink":
        payload = root / "crm_source.csv"
        target = inbox.parent / "payload-target.csv"
        payload.replace(target)
        payload.symlink_to(target)
    else:
        payload = root / "crm_source.csv"
        outside = inbox.parent / "crm-hardlink-source.csv"
        payload.replace(outside)
        os.link(outside, payload)

    with pytest.raises(subject.AvailabilityStop):
        _run(inbox, control)
    assert not (control / subject.CLAIM_FILENAME).exists()


def test_rejects_symlink_collection_directory(
    synthetic_roots: tuple[Path, Path],
) -> None:
    inbox, control = synthetic_roots
    outside = inbox.parent / "outside"
    _mkdir_private(outside)
    name = "00000000000000000001_" + "0" * 64
    (inbox / name).symlink_to(outside, target_is_directory=True)

    with pytest.raises(subject.AvailabilityStop, match="INBOX_CHILD_OPEN"):
        _run(inbox, control)


def test_rejects_noncanonical_or_wrongly_named_manifest(
    synthetic_roots: tuple[Path, Path],
) -> None:
    inbox, control = synthetic_roots
    root = _collection(inbox, arrival=1, collection_id="noncanonical")
    manifest_path = root / "collection_manifest.json"
    value = json.loads(manifest_path.read_bytes())
    _write_private(
        manifest_path,
        (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8"),
    )

    with pytest.raises(subject.AvailabilityStop):
        _run(inbox, control)


def test_stability_is_injectable_and_detects_manifest_mutation(
    synthetic_roots: tuple[Path, Path],
) -> None:
    inbox, control = synthetic_roots
    root = _collection(inbox, arrival=1, collection_id="unstable")
    manifest_path = root / "collection_manifest.json"

    def mutate(_: float) -> float:
        raw = bytearray(manifest_path.read_bytes())
        raw[-2] = ord(" ")
        _write_private(manifest_path, bytes(raw))
        return 60.0

    with pytest.raises(subject.AvailabilityStop, match="MANIFEST_UNSTABLE"):
        _run(inbox, control, stability_observer=mutate)


def test_refuses_short_stability_and_real_mode(
    synthetic_roots: tuple[Path, Path],
) -> None:
    inbox, control = synthetic_roots
    _collection(inbox, arrival=1, collection_id="short")
    with pytest.raises(subject.AvailabilityStop, match="STABILITY_INTERVAL"):
        _run(inbox, control, stability_observer=lambda _: 59.999)
    with pytest.raises(subject.AvailabilityStop, match="REAL_COLLECTION_OPEN_FORBIDDEN"):
        subject.audit_synthetic_availability(
            inbox=inbox,
            control_root=control,
            stability_observer=_stable,
            synthetic_only=False,
        )


def test_exact_rerun_is_idempotent_and_does_not_reobserve_payloads(
    synthetic_roots: tuple[Path, Path],
) -> None:
    inbox, control = synthetic_roots
    _collection(inbox, arrival=1, collection_id="idempotent")
    first = _run(inbox, control)

    def forbidden(_: float) -> float:
        raise AssertionError("idempotent claim must return before observation")

    second = _run(inbox, control, stability_observer=forbidden)
    assert second == first


def test_rejects_corrupt_existing_claim(
    synthetic_roots: tuple[Path, Path],
) -> None:
    inbox, control = synthetic_roots
    _collection(inbox, arrival=1, collection_id="corrupt")
    _run(inbox, control)
    claim_path = control / subject.CLAIM_FILENAME
    claim = json.loads(claim_path.read_bytes())
    claim["plan_sha256"] = "0" * 64
    claim_path.unlink()
    _write_private(claim_path, subject.canonical_json(claim))

    with pytest.raises(subject.AvailabilityStop, match="CLAIM_POLICY"):
        _run(inbox, control)


@pytest.mark.parametrize(
    ("field", "value"),
    [("arrival_epoch_ns", 999), ("collection_id", "tampered")],
)
def test_claim_identity_is_bound_to_selected_ledger_record(
    synthetic_roots: tuple[Path, Path],
    field: str,
    value: object,
) -> None:
    inbox, control = synthetic_roots
    _collection(inbox, arrival=1, collection_id="bound")
    _run(inbox, control)
    claim_path = control / subject.CLAIM_FILENAME
    claim = json.loads(claim_path.read_bytes())
    claim[field] = value
    claim_path.unlink()
    _write_private(claim_path, subject.canonical_json(claim))
    with pytest.raises(subject.AvailabilityStop, match="CLAIM_LEDGER_BINDING"):
        _run(inbox, control)


def test_synthetic_api_rejects_non_tmp_roots() -> None:
    with pytest.raises(subject.AvailabilityStop, match="NOT_SYNTHETIC_TMP"):
        subject.audit_synthetic_availability(
            inbox=Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_labels_v4_13/inbox"),
            control_root=Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_labels_v4_13/control"),
            stability_observer=_stable,
            synthetic_only=True,
        )


def test_rejects_corrupt_existing_ledger(
    synthetic_roots: tuple[Path, Path],
) -> None:
    inbox, control = synthetic_roots
    _collection(inbox, arrival=1, collection_id="ledger-corrupt")
    _run(inbox, control)
    ledger_path = control / subject.LEDGER_FILENAME
    ledger = json.loads(ledger_path.read_bytes())
    ledger["selected_manifest_sha256"] = "0" * 64
    ledger_path.unlink()
    _write_private(ledger_path, subject.canonical_json(ledger))

    with pytest.raises(subject.AvailabilityStop):
        _run(inbox, control)


def test_waiting_does_not_create_ledger_or_claim(
    synthetic_roots: tuple[Path, Path],
) -> None:
    inbox, control = synthetic_roots
    result = _run(inbox, control)
    assert result["verdict"] == "WAITING_FOR_NEW_SOURCE"
    assert not (control / subject.LEDGER_FILENAME).exists()
    assert not (control / subject.CLAIM_FILENAME).exists()


def test_plan_or_lock_hash_drift_stops_before_inbox_open(
    synthetic_roots: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    inbox, control = synthetic_roots
    copied_plan = tmp_path / "plan.json"
    copied_plan.write_bytes(subject.PLAN_PATH.read_bytes() + b" ")
    copied_plan.chmod(0o600)

    with pytest.raises(subject.AvailabilityStop):
        _run(inbox, control, plan_path=copied_plan)
