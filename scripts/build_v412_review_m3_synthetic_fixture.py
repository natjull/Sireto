#!/usr/bin/env python3
"""Build an independent NOT_EVIDENCE offline fixture for M3 transformations."""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import replay_v412_review_collection_policy as replay
from scripts import v412_review_collection_broker as broker
from scripts import v412_review_collection_offline_runtime as runtime


CONFIG = ROOT / "config/v4_12_review_m3_synthetic_fixture.json"
POLICY = ROOT / "config/v4_12_review_collection_policy.json"
SUFFIXES = ROOT / "config/v4_12_review_public_suffixes.txt"
DNS = ROOT / "config/v4_12_review_dns_security_vectors.json"
IDENTITY_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/"
    "v4_12_review_adjudication_pilot/c7a9feecaf2d3c2a/identity"
)
CONFIG_SHA256 = "edaadb2dea576bf36b3060095e9dfd48891de34d4528e6d48a98daa37b638167"
TREE_DOMAIN = b"SIRETO-V412-R30-M3-SYNTHETIC-TREE\0"
_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar("m3_fixture_no_network", default=False)
_HOOK_INSTALLED = False


class FixtureStop(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _audit(event: str, args: tuple[Any, ...]) -> None:
    if not _ACTIVE.get() or event != "socket.__new__":
        return
    family = args[1] if len(args) > 1 else None
    if family in {socket.AF_INET, socket.AF_INET6}:
        raise FixtureStop("network socket forbidden in M3 offline fixture")


class no_network:
    def __enter__(self):
        global _HOOK_INSTALLED
        if not _HOOK_INSTALLED:
            sys.addaudithook(_audit)
            _HOOK_INSTALLED = True
        self._token = _ACTIVE.set(True)
        return self

    def __exit__(self, *_):
        _ACTIVE.reset(self._token)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = canonical(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            if count <= 0:
                raise FixtureStop("short fixture write")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _site_id(seed: int) -> str:
    prefix = f"90000{seed:08d}"
    if len(prefix) != 13:
        raise FixtureStop("synthetic identifier seed overflow")
    for check in range(10):
        value = prefix + str(check)
        if replay.luhn_valid(value):
            return value
    raise FixtureStop("no deterministic Luhn check digit")


def _pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 3 Tf 20 780 Td ({escaped}) Tj ET".encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for ordinal, item in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{ordinal} 0 obj\n".encode("ascii") + item + b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(payload)


def _chunks(payload: bytes) -> tuple[bytes, ...]:
    return tuple(
        payload[offset : offset + broker.MAX_ARCHIVE_CHUNK]
        for offset in range(0, len(payload), broker.MAX_ARCHIVE_CHUNK)
    )


def _results(row: broker.PlannedQuery, urls: Sequence[str]) -> bytes:
    parts = []
    for url in urls:
        parts.append(
            '<div class="result"><a class="result__a" href="'
            + url
            + '">'
            + row.crm_name
            + '</a><div class="result__snippet">'
            + row.crm_name
            + "</div></div>"
        )
    return "".join(parts).encode("utf-8")


def _search(row: broker.PlannedQuery, body: bytes = b"<html></html>", terminal: str = "RESPONSE"):
    attempt = runtime.search_attempt_id(row.query_id, row.query_ordinal, row.search_query)
    return broker.FixtureExchange(
        "SEARCH", attempt, broker.SEARCH_HOST, ("1.1.1.1",), terminal,
        200 if terminal == "RESPONSE" else None,
        (("Content-Type", "text/html; charset=utf-8"),) if terminal == "RESPONSE" else (),
        _chunks(body) if terminal == "RESPONSE" else (),
    )


def _page(
    row: broker.PlannedQuery,
    search_body: bytes,
    rank: int,
    query_slot: int,
    dossier_slot: int,
    body: bytes = b"",
    mime: str = "text/plain; charset=utf-8",
    terminal: str = "RESPONSE",
):
    parsed = replay.parse_ddg_results(search_body, "utf-8")[rank - 1]
    host = replay.normalize_hostname(replay.urlparse(parsed["resolved_url"]).hostname)
    attempt = runtime.page_attempt_id(
        row.query_id, row.query_ordinal, rank, parsed["resolved_url"], query_slot, dossier_slot
    )
    return broker.FixtureExchange(
        "PAGE", attempt, host, ("1.1.1.1",), terminal,
        200 if terminal == "RESPONSE" else None,
        (("Content-Type", mime),) if terminal == "RESPONSE" else (),
        _chunks(body) if terminal == "RESPONSE" else (),
    )


def _materialize(plan: broker.CanonicalPlan):
    rows = plan.rows
    active, closed, repeated, distant = (_site_id(index) for index in range(1, 5))
    exchanges = []
    page_expectations: dict[tuple[str, int, int], dict[str, Any]] = {}

    top_urls = (
        "https://proof-one.fr/a",
        "https://www.proof-one.fr/duplicate",
        "https://service.gouv.fr/record",
        "https://annuaire-entreprises.data.gouv.fr/record",
        "https://proof-five.fr/quota",
        "https://ignored-six.fr/not-logged",
    )
    top_body = _results(rows[0], top_urls)
    exchanges.append(_search(rows[0], top_body))
    html_body = (
        distant + " " + ("bruit " * 150)
        + f"<p>{rows[0].crm_name} {rows[0].crm_address} {rows[0].crm_postcode} {active}</p>"
    ).encode("utf-8")
    exchanges.append(_page(rows[0], top_body, 1, 1, 1, html_body, "text/html; charset=utf-8"))
    plain_body = f"{rows[0].crm_name} {rows[0].crm_address} {rows[0].crm_postcode} {closed}".encode()
    exchanges.append(_page(rows[0], top_body, 3, 2, 2, plain_body))
    page_expectations[(rows[0].query_id, 1, 1)] = {"decision": "OPEN_ATTEMPT", "qualified": [active]}
    page_expectations[(rows[0].query_id, 1, 2)] = {"decision": "SKIP_DUPLICATE_DOMAIN", "qualified": []}
    page_expectations[(rows[0].query_id, 1, 3)] = {"decision": "OPEN_ATTEMPT", "qualified": [closed]}
    page_expectations[(rows[0].query_id, 1, 4)] = {"decision": "SKIP_INADMISSIBLE", "qualified": []}
    page_expectations[(rows[0].query_id, 1, 5)] = {"decision": "SKIP_QUERY_QUOTA", "qualified": []}

    pdf_urls = ("https://proof-pdf-valid.fr/valid.pdf", "https://proof-pdf-invalid.fr/invalid.pdf")
    pdf_search = _results(rows[1], pdf_urls)
    exchanges.append(_search(rows[1], pdf_search))
    valid_text = f"{rows[1].crm_name} {rows[1].crm_address} {rows[1].crm_postcode} {repeated} 2026-07-31"
    invalid_text = f"{rows[1].crm_name} {rows[1].crm_address} {rows[1].crm_postcode} {active} 2026-02-31"
    exchanges.append(_page(rows[1], pdf_search, 1, 1, 3, _pdf(valid_text), "application/pdf"))
    exchanges.append(_page(rows[1], pdf_search, 2, 2, 4, _pdf(invalid_text), "application/pdf"))
    page_expectations[(rows[1].query_id, 2, 1)] = {"decision": "OPEN_ATTEMPT", "qualified": [repeated]}
    page_expectations[(rows[1].query_id, 2, 2)] = {"decision": "OPEN_ATTEMPT", "qualified": []}

    exchanges.append(_search(rows[2], terminal="READ_TIMEOUT"))

    cache_urls = ("https://proof-cache.fr/a", "https://proof-timeout.fr/a")
    cache_search = _results(rows[3], cache_urls)
    exchanges.append(_search(rows[3], cache_search))
    cache_body = f"{rows[3].crm_name} {rows[3].crm_address} {rows[3].crm_postcode} {active}".encode()
    exchanges.append(_page(rows[3], cache_search, 1, 1, 1, cache_body))
    exchanges.append(_page(rows[3], cache_search, 2, 2, 2, terminal="READ_TIMEOUT"))
    page_expectations[(rows[3].query_id, 1, 1)] = {"decision": "OPEN_ATTEMPT", "qualified": [active]}
    page_expectations[(rows[3].query_id, 1, 2)] = {"decision": "OPEN_ATTEMPT", "qualified": []}

    exchanges.append(_search(rows[4], b"x" * (2 * 1024 * 1024 + 1)))
    exchanges.append(_search(rows[5]))
    for row in rows[6:]:
        exchanges.append(_search(row))

    records = (
        broker.SireneFixtureRecord(active, active[:9], etat_administratif="A", enseigne_1="SYNTH ACTIVE"),
        broker.SireneFixtureRecord(closed, closed[:9], etat_administratif="F", enseigne_1="SYNTH CLOSED"),
        broker.SireneFixtureRecord(repeated, repeated[:9], etat_administratif="A", enseigne_1="SYNTH DUP A"),
        broker.SireneFixtureRecord(repeated, repeated[:9], etat_administratif="A", enseigne_1="SYNTH DUP B"),
    )
    return tuple(exchanges), records, page_expectations, {
        "active": active, "closed": closed, "distant": distant, "repeated": repeated,
    }


def _transport_hash(exchanges) -> str:
    return digest(runtime.canonical_json({
        "exchanges": [item.seal_projection() for item in exchanges],
        "schema_version": broker.FIXTURE_SCHEMA,
    }))


def _snapshot_hash(records) -> str:
    return digest(runtime.canonical_json([
        item.seal_projection() for item in sorted(records, key=lambda item: item.siret)
    ]))


def _tree(root: Path, excluded=frozenset({"manifest.json", "seal.json"})):
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_digest(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    ]


def _build(root: Path) -> dict[str, Any]:
    raw_config = CONFIG.read_bytes()
    if digest(raw_config) != CONFIG_SHA256:
        raise FixtureStop("M3 fixture configuration hash mismatch")
    config = json.loads(raw_config)
    if config.get("role") != "NOT_EVIDENCE" or config.get("schema_version") != "sireto-v4.12-r30-m3-offline-fixture-1":
        raise FixtureStop("M3 fixture role or schema mismatch")
    plan = broker.load_canonical_plan_from_parquets(
        IDENTITY_ROOT / "identity_discovery.parquet", IDENTITY_ROOT / "collection_plan.parquet"
    )
    exchanges, records, expected_pages, identifiers = _materialize(plan)
    observed_transport = _transport_hash(exchanges)
    observed_snapshot = _snapshot_hash(records)
    if observed_transport != config["transport_sha256"] or observed_snapshot != config["snapshot_sha256"]:
        raise FixtureStop("external fixture seal mismatch")
    transport = broker.SealedFixtureTransport(exchanges, records, observed_transport, observed_snapshot)
    events = []

    def hook(operation: str, target: str) -> None:
        events.append({"operation": operation, "target": target})

    journal = runtime.AccessJournal(root / "journal")
    page_rows = []
    search_rows = []
    lookup_rows = []
    try:
        journal.state_transition(phase="PREFLIGHT", target_state="IDENTITY_NETWORK_OPEN")
        value = broker.InjectedOfflineBroker(
            journal=journal, plan=plan.rows, transport=transport,
            expected_transport_fixture_sha256=observed_transport,
            expected_sirene_snapshot_sha256=observed_snapshot,
            policy_bytes=POLICY.read_bytes(), public_suffix_bytes=SUFFIXES.read_bytes(),
            dns_vectors_bytes=DNS.read_bytes(),
            archive_store=broker.ExclusiveArchiveStore(root / "raw", hook), audit_hook=hook,
        )
        for row in plan.rows:
            response = value.search(row.query_id, row.query_ordinal)
            search_rows.append({
                "error_type": None if response.error is None else response.error.error_type,
                "query_id": row.query_id,
                "query_ordinal": row.query_ordinal,
                "result_count": len(response.results),
                "status": response.status,
            })
            for result in response.results:
                opened = value.open_page(row.query_id, row.query_ordinal, result.result_rank)
                page_rows.append({
                    "decision": opened.decision,
                    "facts_eligible": opened.facts_eligible,
                    "postopen_family": opened.postopen_family,
                    "qualified_site_ids": list(opened.qualified_sirets),
                    "query_id": row.query_id,
                    "query_ordinal": row.query_ordinal,
                    "result_rank": result.result_rank,
                    "status": opened.status,
                })
                expected = expected_pages.get((row.query_id, row.query_ordinal, result.result_rank))
                if expected is not None and (
                    opened.decision != expected["decision"]
                    or list(opened.qualified_sirets) != expected["qualified"]
                ):
                    raise FixtureStop("scenario page outcome diverges from its external expectation")
            for query_id, site_id in value.sirene_lookup_plan:
                if query_id != row.query_id or any(
                    item["query_id"] == query_id and item["site_id"] == site_id for item in lookup_rows
                ):
                    continue
                looked = value.lookup_sirene(query_id, row.query_ordinal, site_id)
                lookup_rows.append({
                    "found_exactly_once": looked.found_exactly_once,
                    "query_id": query_id,
                    "served_from_global_cache": looked.served_from_global_cache,
                    "site_id": site_id,
                    "state": None if looked.record is None else looked.record.etat_administratif,
                })
        value.revoke()
        if value.state != "IDENTITY_SEALED_NETWORK_REVOKED":
            raise FixtureStop("fixture broker did not revoke")
        head = journal.verify_complete()
        projection = journal.project_jsonl(root / "journal" / "access_journal.jsonl")
    finally:
        journal.close()

    _write_json(root / "search_outcomes.json", search_rows)
    _write_json(root / "page_outcomes.json", page_rows)
    _write_json(root / "lookup_outcomes.json", lookup_rows)
    _write_json(root / "audit_events.json", events)
    summary = {
        "identifier_generation": "LOCAL_DETERMINISTIC_LUHN",
        "journal_head_sha256": head,
        "journal_jsonl_sha256": projection,
        "lookup_count": len(lookup_rows),
        "page_decision_count": len(page_rows),
        "role": "NOT_EVIDENCE",
        "scenario_ids": config["scenario_ids"],
        "search_count": len(search_rows),
        "synthetic_identifier_sha256s": {key: digest(value.encode()) for key, value in identifiers.items()},
        "transport_sha256": observed_transport,
    }
    _write_json(root / "summary.json", summary)
    payload = _tree(root)
    manifest = {
        "broker_sha256": file_digest(Path(broker.__file__)),
        "config_sha256": digest(raw_config),
        "input_scope": "IDENTITY_PARQUETS_ONLY",
        "output_claim": "NOT_EVIDENCE_NOT_M3_WORKER_NOT_SECTION_5",
        "payload_files": payload,
        "schema_version": "sireto-v4.12-r30-m3-offline-fixture-manifest-1",
    }
    _write_json(root / "manifest.json", manifest)
    seal = {
        "manifest_sha256": file_digest(root / "manifest.json"),
        "payload_tree_sha256": digest(TREE_DOMAIN + canonical(payload)),
        "schema_version": "sireto-v4.12-r30-m3-offline-fixture-seal-1",
    }
    _write_json(root / "seal.json", seal)
    return {"manifest": manifest, "seal": seal, "summary": summary}


def build(destination: Path) -> dict[str, Any]:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        with no_network():
            result = _build(temporary)
        temporary.rename(destination)
        return result
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    print(canonical(build(args.destination)["seal"]).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
