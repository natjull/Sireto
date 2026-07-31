from __future__ import annotations

import hashlib
import importlib.util
import json
import inspect
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v412_review_collection_broker.py"
SPEC = importlib.util.spec_from_file_location("v412_review_collection_broker", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
broker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = broker
SPEC.loader.exec_module(broker)

POLICY_BYTES = (ROOT / "config/v4_12_review_collection_policy.json").read_bytes()
PSL_BYTES = (ROOT / "config/v4_12_review_public_suffixes.txt").read_bytes()
DNS_BYTES = (ROOT / "config/v4_12_review_dns_security_vectors.json").read_bytes()
SIRET = "55210055400013"
OTHER_SIRET = "78983652500020"
DOCKET_ROOT = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_review_adjudication_pilot/c7a9feecaf2d3c2a")
REAL_PLAN = broker.load_canonical_plan_from_parquets(
    DOCKET_ROOT / "identity/identity_discovery.parquet",
    DOCKET_ROOT / "identity/collection_plan.parquet",
)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def plans() -> tuple[broker.PlannedQuery, ...]:
    return REAL_PLAN.rows


def canonical_plan(rows=None) -> bytes:
    rows = plans() if rows is None else tuple(rows)
    return broker.runtime.canonical_json([row.projection() for row in rows])


def search_exchange(plan: broker.PlannedQuery, body: bytes = b"<html></html>", **changes):
    attempt = broker.runtime.search_attempt_id(plan.query_id, plan.query_ordinal, plan.search_query)
    values = {
        "request_kind": "SEARCH", "attempt_id": attempt, "hostname": broker.SEARCH_HOST,
        "addresses": ("1.1.1.1",), "terminal": "RESPONSE", "http_status": 200,
        "headers": (("Content-Type", "text/html; charset=utf-8"),),
        "body_chunks": tuple(body[index:index + broker.MAX_ARCHIVE_CHUNK] for index in range(0, len(body), broker.MAX_ARCHIVE_CHUNK)),
    }
    values.update(changes)
    return broker.FixtureExchange(**values)


def fixture_hash(exchanges) -> str:
    projection = {
        "exchanges": [item.seal_projection() for item in exchanges],
        "schema_version": broker.FIXTURE_SCHEMA,
    }
    return sha(broker.runtime.canonical_json(projection))


def snapshot_hash(records) -> str:
    projection = [item.seal_projection() for item in sorted(records, key=lambda item: item.siret)]
    return sha(broker.runtime.canonical_json(projection))


def transport(exchanges=(), records=()):
    exchanges, records = tuple(exchanges), tuple(records)
    return broker.SealedFixtureTransport(
        exchanges, records, fixture_hash(exchanges), snapshot_hash(records),
    )


def journal(path: Path):
    value = broker.runtime.AccessJournal(path)
    value.state_transition(phase="PREFLIGHT", target_state="IDENTITY_NETWORK_OPEN")
    return value


def make_broker(tmp_path: Path, exchanges=(), records=(), rows=None, *, hook=None, custom_transport=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = plans() if rows is None else tuple(rows)
    plan_bytes = canonical_plan(rows)
    audit = hook if hook is not None else (lambda operation, target: None)
    injected = custom_transport or transport(exchanges, records)
    return broker.InjectedOfflineBroker(
        journal=journal(tmp_path / "journal"), plan=rows,
        transport=injected, expected_transport_fixture_sha256=injected.fixture_sha256,
        expected_sirene_snapshot_sha256=injected.sirene_snapshot_sha256,
        policy_bytes=POLICY_BYTES, public_suffix_bytes=PSL_BYTES, dns_vectors_bytes=DNS_BYTES,
        archive_store=broker.ExclusiveArchiveStore(tmp_path / "archive", audit), audit_hook=audit,
    )


def result_html(url="https://societe-exemple.fr/a", row=None) -> bytes:
    row = plans()[0] if row is None else row
    return (
        f'<div class="result"><a class="result__a" href="{url}">{row.crm_name}</a>'
        '<div class="result__snippet">Lyon</div></div>'
    ).encode()


def page_exchange(plan, search_body, page_body, *, rank=1, terminal="RESPONSE", http_status=200):
    parsed = broker.replay.parse_ddg_results(search_body, "utf-8")[rank - 1]
    hostname = broker.replay.normalize_hostname(broker.replay.urlparse(parsed["resolved_url"]).hostname)
    page_id = broker.runtime.page_attempt_id(
        plan.query_id, plan.query_ordinal, rank, parsed["resolved_url"], rank, rank,
    )
    return broker.FixtureExchange(
        "PAGE", page_id, hostname, ("1.1.1.1",), terminal,
        http_status if terminal == "RESPONSE" else None,
        (("Content-Type", "text/plain; charset=utf-8"),) if terminal == "RESPONSE" else (),
        tuple(page_body[index:index + broker.MAX_ARCHIVE_CHUNK] for index in range(0, len(page_body), broker.MAX_ARCHIVE_CHUNK)) if terminal == "RESPONSE" else (),
    )


def test_real_identity_parquets_define_exact_authenticated_30_90_plan(tmp_path: Path) -> None:
    base = DOCKET_ROOT
    loaded = broker.load_canonical_plan_from_parquets(
        base / "identity/identity_discovery.parquet",
        base / "identity/collection_plan.parquet",
    )
    assert len(loaded.rows) == 90
    assert len({row.query_id for row in loaded.rows}) == 30
    assert [(row.selection_ordinal, row.query_ordinal) for row in loaded.rows] == [
        (selection, ordinal) for selection in range(1, 31) for ordinal in range(1, 4)
    ]
    assert sha(loaded.canonical_bytes) == loaded.canonical_sha256
    assert loaded.canonical_sha256 == broker.CANONICAL_PLAN_SHA256
    mutated_identity = tmp_path / "identity.parquet"
    payload = (base / "identity/identity_discovery.parquet").read_bytes()
    mutated_identity.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(broker.BrokerIntegrityStop, match="hash mismatch"):
        broker.load_canonical_plan_from_parquets(
            mutated_identity,
            base / "identity/collection_plan.parquet",
        )
    mutated_plan = tmp_path / "collection.parquet"
    payload = (base / "identity/collection_plan.parquet").read_bytes()
    mutated_plan.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
    with pytest.raises(broker.BrokerIntegrityStop, match="hash mismatch"):
        broker.load_canonical_plan_from_parquets(
            base / "identity/identity_discovery.parquet", mutated_plan,
        )


def test_plan_loader_has_no_docket_argument_or_open(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "docket" not in inspect.signature(broker.load_canonical_plan_from_parquets).parameters
    opened = []
    original = broker._read_hashed_file
    def observed(path, expected):
        opened.append(Path(path).name)
        return original(path, expected)
    monkeypatch.setattr(broker, "_read_hashed_file", observed)
    broker.load_canonical_plan_from_parquets(
        DOCKET_ROOT / "identity/identity_discovery.parquet",
        DOCKET_ROOT / "identity/collection_plan.parquet",
    )
    assert opened == ["identity_discovery.parquet", "collection_plan.parquet"]


def test_policy_psl_cidr_and_plan_pins_are_not_caller_parameterized(tmp_path: Path) -> None:
    rows = plans()
    plan_bytes = canonical_plan(rows)
    audit = lambda *_: None
    injected = transport()
    common = dict(
        journal=journal(tmp_path / "journal"), plan=rows,
        transport=injected, expected_transport_fixture_sha256=injected.fixture_sha256,
        expected_sirene_snapshot_sha256=injected.sirene_snapshot_sha256,
        policy_bytes=POLICY_BYTES, public_suffix_bytes=PSL_BYTES, dns_vectors_bytes=DNS_BYTES,
        archive_store=broker.ExclusiveArchiveStore(tmp_path / "archive", audit),
        audit_hook=audit,
    )
    signature = inspect.signature(broker.InjectedOfflineBroker)
    assert not any(name.startswith("expected_policy") or name.startswith("expected_plan") for name in signature.parameters)
    for field in ("policy_bytes", "public_suffix_bytes", "dns_vectors_bytes"):
        mutated = dict(common)
        mutated[field] = mutated[field] + b" "
        with pytest.raises(broker.BrokerIntegrityStop, match="authenticated|canonical plan"):
            broker.InjectedOfflineBroker(**mutated)
    changed_rows = list(rows)
    first = changed_rows[0]
    changed_rows[0] = broker.PlannedQuery(
        first.query_id, first.selection_ordinal, first.query_ordinal,
        first.search_query + " altered", first.crm_name, first.crm_address, first.crm_postcode,
    )
    changed = dict(common, plan=tuple(changed_rows))
    with pytest.raises(broker.BrokerIntegrityStop, match="canonical plan"):
        broker.InjectedOfflineBroker(**changed)


def test_transport_requires_external_hash_has_no_self_seal_and_detects_mutation() -> None:
    row = plans()[0]
    exchange = search_exchange(row)
    assert not hasattr(broker.SealedFixtureTransport, "seal")
    valid = fixture_hash((exchange,))
    broker.SealedFixtureTransport((exchange,), (), valid, snapshot_hash(()))
    changed = search_exchange(row, b"mutated")
    with pytest.raises(broker.BrokerIntegrityStop, match="seal mismatch"):
        broker.SealedFixtureTransport((changed,), (), valid, snapshot_hash(()))


def test_search_then_result_ranks_are_strict_and_fail_terminal(tmp_path: Path) -> None:
    rows = plans()
    first = search_exchange(rows[0], result_html())
    second = search_exchange(rows[1])
    value = make_broker(tmp_path, (first, second))
    response = value.search(rows[0].query_id, 1)
    assert response.archive is not None and response.archive.byte_count == len(result_html())
    archived_results = [
        event for event in value._journal.records
        if event["event_kind"] == "RESULT" and event["operation"] == "SEARCH_REQUEST"
    ]
    assert archived_results[-1]["byte_count"] == response.archive.byte_count
    assert archived_results[-1]["content_sha256"] == response.archive.content_sha256
    with pytest.raises(broker.BrokerIntegrityStop, match="prior result ranks"):
        value.search(rows[1].query_id, 2)
    assert value.state == "STOP_INTEGRITY"

    two_results = result_html() + result_html("https://societe-exemple-2.fr/b")
    first_two = search_exchange(rows[0], two_results)
    value = make_broker(tmp_path / "ranks", (first_two, page_exchange(rows[0], two_results, b"irrelevant")))
    value.search(rows[0].query_id, 1)
    with pytest.raises(broker.BrokerIntegrityStop, match="strict archived"):
        value.open_page(rows[0].query_id, 1, 2)


def test_raw_archive_precedes_parser_is_64k_bounded_and_o_excl(tmp_path: Path) -> None:
    calls = []
    store = broker.ExclusiveArchiveStore(tmp_path, lambda operation, target: calls.append((operation, target)))
    payload = b"x" * (broker.MAX_ARCHIVE_CHUNK * 2 + 7)
    receipt = store.archive("pages", "a" * 64, payload)
    assert receipt.byte_count == len(payload) and receipt.content_sha256 == sha(payload)
    assert receipt.chunk_count == 3 and receipt.maximum_chunk_size == broker.MAX_ARCHIVE_CHUNK
    assert calls == [("WRITE_LOCAL", "pages/" + "a" * 64 + ".bin")]
    with pytest.raises(broker.BrokerIntegrityStop, match="already exists"):
        store.archive("pages", "a" * 64, payload)


def test_oversize_body_is_never_archived_or_parsed(tmp_path: Path) -> None:
    row = plans()[0]
    oversized = b"<html>" + b"x" * (2 * 1024 * 1024)
    value = make_broker(tmp_path, (search_exchange(row, oversized),))
    response = value.search(row.query_id, row.query_ordinal)
    assert response.status == "HTTP_ERROR"
    assert response.error is not None and response.error.error_type == "TOO_LARGE"
    assert response.archive is None and not (tmp_path / "archive/search").exists()


def test_broker_owns_stream_limit_before_materialization() -> None:
    chunks = [b"a" * broker.MAX_ARCHIVE_CHUNK for _ in range(100)]
    class ChunkProbe:
        def __init__(self): self.yielded = 0
        def iter_body_chunks(self, exchange):
            for chunk in chunks:
                self.yielded += 1
                yield chunk
    probe = ChunkProbe()
    exchange = search_exchange(plans()[0])
    threshold_chunks = 10
    streamed = broker._consume_bounded_body(
        probe, exchange, broker.MAX_ARCHIVE_CHUNK * threshold_chunks,
    )
    assert probe.yielded == threshold_chunks + 1
    assert streamed.too_large and streamed.body is None
    assert streamed.wire_byte_count == (threshold_chunks + 1) * broker.MAX_ARCHIVE_CHUNK
    assert streamed.maximum_chunk_size == broker.MAX_ARCHIVE_CHUNK
    with pytest.raises(broker.BrokerIntegrityStop, match="64 KiB"):
        broker.FixtureExchange(
            "SEARCH", "a" * 64, broker.SEARCH_HOST, ("1.1.1.1",), "RESPONSE", 200,
            (("Content-Type", "text/html"),), (b"x" * (broker.MAX_ARCHIVE_CHUNK + 1),),
        )


def test_oversized_transport_chunk_closes_intent_and_poison_broker(tmp_path: Path) -> None:
    row = plans()[0]
    delegate = transport((search_exchange(row),))

    class InvalidChunkTransport:
        @property
        def fixture_sha256(self): return delegate.fixture_sha256
        @property
        def sirene_snapshot_sha256(self): return delegate.sirene_snapshot_sha256
        def consume(self, *args): return delegate.consume(*args)
        def iter_body_chunks(self, exchange):
            yield b"x" * (broker.MAX_ARCHIVE_CHUNK + 1)
        def records_for(self, siret): return delegate.records_for(siret)
        def assert_fully_consumed(self): return delegate.assert_fully_consumed()

    value = make_broker(tmp_path, custom_transport=InvalidChunkTransport())
    with pytest.raises(broker.BrokerIntegrityStop, match="64 KiB"):
        value.search(row.query_id, row.query_ordinal)
    assert value.state == "STOP_INTEGRITY"
    search_intents = [
        event for event in value._journal.records
        if event["operation"] == "SEARCH_REQUEST" and event["event_kind"] == "INTENT"
    ]
    search_results = [
        event for event in value._journal.records
        if event["operation"] == "SEARCH_REQUEST" and event["event_kind"] == "RESULT"
    ]
    assert len(search_intents) == len(search_results) == 1
    assert search_results[0]["parent_intent_ordinal"] == search_intents[0]["event_ordinal"]
    assert search_results[0]["outcome"] == "STOP_INTEGRITY"
    assert search_results[0]["error_type"] == "IO_INTEGRITY"
    value._journal.verify_complete()
    with pytest.raises(broker.BrokerIntegrityStop, match="terminal"):
        value.search(row.query_id, row.query_ordinal)


def test_direct_triple_is_recomputed_and_unrelated_siret_is_not_qualified(tmp_path: Path) -> None:
    row = plans()[0]
    search_body = result_html()
    page_body = (
        OTHER_SIRET + " " + ("bruit " * 150) +
        f"{row.crm_name} {row.crm_address} {row.crm_postcode} {SIRET}"
    ).encode()
    page = page_exchange(row, search_body, page_body)
    record = broker.SireneFixtureRecord(SIRET, SIRET[:9], etat_administratif="A")
    value = make_broker(tmp_path, (search_exchange(row, search_body), page), (record,))
    value.search(row.query_id, row.query_ordinal)
    opened = value.open_page(row.query_id, row.query_ordinal, 1)
    assert opened.archive is not None and opened.postopen_family == "ENTITY_OFFICIAL_SITE"
    assert opened.facts_eligible and opened.qualified_sirets == (SIRET,)
    assert value.sirene_lookup_plan == ((row.query_id, SIRET),)
    with pytest.raises(broker.BrokerIntegrityStop, match="qualified"):
        value.lookup_sirene(row.query_id, row.query_ordinal, OTHER_SIRET)


def test_dated_family_and_calendar_date_are_recomputed_from_raw(tmp_path: Path) -> None:
    row = plans()[0]
    search_body = result_html("https://societe-exemple.fr/rapport.pdf")
    base = f"{row.crm_name} {row.crm_address} {row.crm_postcode} {SIRET} "
    for suffix, date_value, eligible in (("valid", "2026-07-31", True), ("invalid", "2026-02-31", False)):
        page = page_exchange(row, search_body, (base + date_value).encode())
        value = make_broker(
            tmp_path / suffix,
            (search_exchange(row, search_body), page),
        )
        value.search(row.query_id, row.query_ordinal)
        opened = value.open_page(row.query_id, row.query_ordinal, 1)
        assert opened.facts_eligible is eligible
        assert opened.postopen_family == (
            "DATED_PUBLIC_DOCUMENT" if eligible else "INADMISSIBLE_AFTER_OPEN"
        )
        assert opened.qualified_sirets == ((SIRET,) if eligible else ())


class CountingTransport:
    def __init__(self, delegate):
        self.delegate = delegate
        self.lookups = 0

    @property
    def fixture_sha256(self): return self.delegate.fixture_sha256

    @property
    def sirene_snapshot_sha256(self): return self.delegate.sirene_snapshot_sha256

    def consume(self, *args): return self.delegate.consume(*args)

    def iter_body_chunks(self, *args): return self.delegate.iter_body_chunks(*args)

    def records_for(self, siret):
        self.lookups += 1
        return self.delegate.records_for(siret)

    def assert_fully_consumed(self): return self.delegate.assert_fully_consumed()


def test_sirene_plan_is_derived_and_snapshot_cache_is_global_by_siret(tmp_path: Path) -> None:
    rows = plans()
    body1 = result_html()
    body2 = result_html("https://societe-exemple.fr/b", rows[3])
    proof1 = f"{rows[0].crm_name} {rows[0].crm_address} {rows[0].crm_postcode} {SIRET}".encode()
    proof2 = f"{rows[3].crm_name} {rows[3].crm_address} {rows[3].crm_postcode} {SIRET}".encode()
    exchanges = (
        search_exchange(rows[0], body1), page_exchange(rows[0], body1, proof1),
        search_exchange(rows[1]), search_exchange(rows[2]),
        search_exchange(rows[3], body2), page_exchange(rows[3], body2, proof2),
    )
    record = broker.SireneFixtureRecord(SIRET, SIRET[:9], etat_administratif="A")
    counted = CountingTransport(transport(exchanges, (record,)))
    value = make_broker(tmp_path, custom_transport=counted)
    for row in (rows[0],):
        value.search(row.query_id, row.query_ordinal)
        value.open_page(row.query_id, row.query_ordinal, 1)
        looked = value.lookup_sirene(row.query_id, row.query_ordinal, SIRET)
        assert looked.record_sha256 == sha(broker.runtime.canonical_json([record.seal_projection()]))
        assert looked.sirene_snapshot_sha256 == counted.sirene_snapshot_sha256
    value.search(rows[1].query_id, rows[1].query_ordinal)
    value.search(rows[2].query_id, rows[2].query_ordinal)
    row = rows[3]
    value.search(row.query_id, row.query_ordinal)
    value.open_page(row.query_id, row.query_ordinal, 1)
    looked = value.lookup_sirene(row.query_id, row.query_ordinal, SIRET)
    assert looked.served_from_global_cache
    assert counted.lookups == 1


def test_audit_hook_is_fail_closed_before_transport(tmp_path: Path) -> None:
    row = plans()[0]
    injected = transport((search_exchange(row),))
    calls = []
    def deny(operation, target):
        calls.append((operation, target))
        if operation == "SEARCH_REQUEST":
            raise RuntimeError("audit unavailable")
    value = make_broker(tmp_path, custom_transport=injected, hook=deny)
    with pytest.raises(RuntimeError, match="audit unavailable"):
        value.search(row.query_id, row.query_ordinal)
    assert value.state == "STOP_INTEGRITY"
    assert calls[0][0] == "SEARCH_REQUEST"
    assert injected._consumed == set()


def test_complete_empty_run_revokes_irreversibly(tmp_path: Path) -> None:
    rows = plans()
    value = make_broker(tmp_path, tuple(search_exchange(row) for row in rows))
    for row in rows:
        assert value.search(row.query_id, row.query_ordinal).results == ()
    value.revoke()
    assert value.state == "IDENTITY_SEALED_NETWORK_REVOKED"
    with pytest.raises(broker.BrokerIntegrityStop, match="irreversible"):
        value.revoke()


def test_module_has_no_live_network_implementation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = ("import socket", "from socket", "requests", "http.client", "urllib.request")
    assert all(token not in source for token in forbidden)
    assert isinstance(transport(), broker.BrokerTransport)
