from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket

import pytest

from scripts import build_v412_review_m3_synthetic_fixture as fixture


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("m3-offline") / "run"
    fixture.build(destination)
    return destination


def test_static_postcondition_has_no_comparison_or_truth_tokens() -> None:
    source = Path(fixture.__file__).read_text(encoding="utf-8").casefold()
    config = fixture.CONFIG.read_text(encoding="utf-8").casefold()
    forbidden = ("docket", "candidate", "ground_truth", "predicted", "top1", "label")
    assert all(token not in source for token in forbidden)
    assert all(token not in config for token in forbidden)
    assert "comparison" not in fixture.IDENTITY_ROOT.as_posix().casefold()


def test_config_is_external_not_evidence_and_identifiers_are_generated() -> None:
    raw = fixture.CONFIG.read_bytes()
    config = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == fixture.CONFIG_SHA256
    assert config["role"] == "NOT_EVIDENCE"
    assert config["transport_sha256"] == "d11e62f3bd9f1ce4242ed6fcb8fc9124b911d06e9764ed153439539b9d5ce457"
    assert config["snapshot_sha256"] == "cbf40be7c05158557bad4dd2bde6787d1d340f96bb04b179155c734e1e5bb0b6"
    assert not any(str(fixture._site_id(seed)) in raw.decode() for seed in range(1, 5))
    assert all(fixture.replay.luhn_valid(fixture._site_id(seed)) for seed in range(1, 5))


def test_search_empty_top5_dedup_inadmissible_and_quota(built: Path) -> None:
    searches = load(built / "search_outcomes.json")
    assert len(searches) == 90
    assert searches[0]["status"] == "SUCCESS" and searches[0]["result_count"] == 5
    assert searches[5]["status"] == "SUCCESS" and searches[5]["result_count"] == 0
    pages = load(built / "page_outcomes.json")
    first = [row for row in pages if row["query_ordinal"] == 1 and row["query_id"] == searches[0]["query_id"]]
    assert [row["decision"] for row in first] == [
        "OPEN_ATTEMPT", "SKIP_DUPLICATE_DOMAIN", "OPEN_ATTEMPT",
        "SKIP_INADMISSIBLE", "SKIP_QUERY_QUOTA",
    ]


def test_html_plain_pdf_triples_distance_and_dates(built: Path) -> None:
    pages = load(built / "page_outcomes.json")
    eligible = [row for row in pages if row["facts_eligible"]]
    assert {row["postopen_family"] for row in eligible} >= {
        "ENTITY_OFFICIAL_SITE", "PUBLIC_ADMINISTRATION", "DATED_PUBLIC_DOCUMENT",
    }
    valid_ids = {value for row in eligible for value in row["qualified_site_ids"]}
    assert fixture._site_id(1) in valid_ids
    assert fixture._site_id(2) in valid_ids
    assert fixture._site_id(3) in valid_ids
    assert fixture._site_id(4) not in valid_ids
    pdf_rows = [row for row in pages if row["query_ordinal"] == 2]
    assert [(row["postopen_family"], row["facts_eligible"]) for row in pdf_rows] == [
        ("DATED_PUBLIC_DOCUMENT", True), ("INADMISSIBLE_AFTER_OPEN", False),
    ]
    bodies = [path.read_bytes() for path in (built / "raw/pages").glob("*.bin")]
    assert any(body.startswith(b"%PDF-1.4") for body in bodies)
    assert any(b"<p>" in body for body in bodies)
    assert any(not body.startswith((b"<", b"%PDF")) for body in bodies)


def test_registry_unique_closed_nonunique_and_global_cache(built: Path) -> None:
    rows = load(built / "lookup_outcomes.json")
    assert len(rows) == 4
    active = [row for row in rows if row["site_id"] == fixture._site_id(1)]
    assert len(active) == 2
    assert [row["served_from_global_cache"] for row in active] == [False, True]
    assert all(row["found_exactly_once"] and row["state"] == "A" for row in active)
    closed = next(row for row in rows if row["site_id"] == fixture._site_id(2))
    assert closed["found_exactly_once"] and closed["state"] == "F"
    repeated = next(row for row in rows if row["site_id"] == fixture._site_id(3))
    assert repeated["found_exactly_once"] is False and repeated["state"] is None


def test_timeout_oversize_and_error_paths_are_present(built: Path) -> None:
    searches = load(built / "search_outcomes.json")
    assert searches[2]["status"] == "TIMEOUT"
    assert searches[2]["error_type"] == "READ_TIMEOUT"
    assert searches[4]["status"] == "HTTP_ERROR"
    assert searches[4]["error_type"] == "TOO_LARGE"
    pages = load(built / "page_outcomes.json")
    timeout = next(row for row in pages if row["status"] == "TIMEOUT")
    assert timeout["decision"] == "OPEN_ATTEMPT" and timeout["facts_eligible"] is False


def test_summary_manifest_seal_and_not_evidence_claim(built: Path) -> None:
    summary = load(built / "summary.json")
    manifest = load(built / "manifest.json")
    seal = load(built / "seal.json")
    assert summary["role"] == "NOT_EVIDENCE"
    assert summary["search_count"] == 90
    assert set(summary["scenario_ids"]) == set(load(fixture.CONFIG)["scenario_ids"])
    assert manifest["input_scope"] == "IDENTITY_PARQUETS_ONLY"
    assert manifest["output_claim"] == "NOT_EVIDENCE_NOT_M3_WORKER_NOT_SECTION_5"
    observed = fixture._tree(built)
    assert manifest["payload_files"] == observed
    assert seal["manifest_sha256"] == fixture.file_digest(built / "manifest.json")
    assert seal["payload_tree_sha256"] == fixture.digest(
        fixture.TREE_DOMAIN + fixture.canonical(observed)
    )


def test_no_socket_is_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_socket(*args, **kwargs):
        raise AssertionError("socket construction attempted")

    monkeypatch.setattr(socket, "socket", forbidden_socket)
    fixture.build(tmp_path / "offline")


def test_external_seal_mutation_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    changed = json.loads(fixture.CONFIG.read_text())
    changed["transport_sha256"] = "0" * 64
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n")
    monkeypatch.setattr(fixture, "CONFIG", path)
    with pytest.raises(fixture.FixtureStop, match="configuration hash"):
        fixture.build(tmp_path / "rejected")


def test_two_builds_are_byte_identical_and_not_integrated(tmp_path: Path) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    fixture.build(first)
    fixture.build(second)
    assert files(first) == files(second)
    for path in (
        fixture.ROOT / "scripts/build_v412_review_collection_synthetic_fixture.py",
        fixture.ROOT / "scripts/v412_review_collection_offline_runtime.py",
        fixture.ROOT / "scripts/v412_review_collection_broker.py",
    ):
        assert "build_v412_review_m3_synthetic_fixture" not in path.read_text(encoding="utf-8")
