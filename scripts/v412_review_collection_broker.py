#!/usr/bin/env python3
"""Closed, fixture-injected broker primitives for V4.12-R30.

This module has deliberately no live transport.  Every DNS and HTTP outcome is
consumed once from a sealed in-memory fixture.  It reuses the frozen replay
policy for URL classification, DNS CIDR decisions and DDG parsing, and the
offline runtime for canonical identifiers and the primary journal.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from lxml import html

from scripts import replay_v412_review_collection_policy as replay
from scripts import v412_review_collection_offline_runtime as runtime


BROKER_SCHEMA = "sireto-v4.12-r30-fixture-broker-response-1"
FIXTURE_SCHEMA = "sireto-v4.12-r30-sealed-fixture-transport-1"
SEARCH_HOST = "html.duckduckgo.com"
SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
SEARCH_HEADERS = (
    ("Host", SEARCH_HOST),
    ("User-Agent", "SIRETO-V4.12-R30/1.0 (+offline-entity-resolution-audit)"),
    ("Accept", "text/html"),
    ("Accept-Language", "fr-FR,fr;q=0.9"),
    ("Accept-Encoding", "identity"),
    ("Connection", "close"),
)
PAGE_HEADER_TAIL = (
    ("User-Agent", "SIRETO-V4.12-R30/1.0 (+offline-entity-resolution-audit)"),
    ("Accept", "text/html,text/plain,application/pdf"),
    ("Accept-Language", "fr-FR,fr;q=0.9"),
    ("Accept-Encoding", "identity"),
    ("Connection", "close"),
)
SEARCH_TERMINALS = frozenset(
    {
        "RESPONSE", "DNS_ERROR", "CONNECT_ERROR", "CONNECT_TIMEOUT",
        "TLS_ERROR", "WRITE_ERROR", "NO_RESPONSE_HEADERS", "READ_TIMEOUT",
        "NETWORK_ERROR", "PARSE_ERROR",
    }
)
ERROR_TYPES = frozenset(
    {
        "DNS", "PRIVATE_ADDRESS", "NETWORK", "CONNECT_TIMEOUT", "READ_TIMEOUT",
        "TLS", "HTTP_STATUS", "REDIRECT_FORBIDDEN", "TOO_LARGE",
        "CONTENT_ENCODING", "UNSUPPORTED_MIME", "MALFORMED_RESPONSE", "PARSE",
        "IO_INTEGRITY",
    }
)
ERROR_STAGES = frozenset({"SEARCH", "PAGE_OPEN", "DNS", "TLS", "HTTP", "ARCHIVE", "PARSE"})
ADMISSIBLE_FAMILIES = frozenset(
    {
        "PUBLIC_ADMINISTRATION", "ENTITY_OFFICIAL_SITE_CANDIDATE",
        "OFFICIAL_SECTOR_DIRECTORY", "DATED_PUBLIC_DOCUMENT_CANDIDATE",
    }
)
POLICY_SHA256 = "1238eb957f84c811ac64375c66a0d62e1bef977a139c0a685e669a5d18c63b88"
PUBLIC_SUFFIXES_SHA256 = "10fe038631c2a3dd619370e368be3dbd9b6cb8daf2bd4203ced236cf6226c823"
DNS_VECTORS_SHA256 = "a1f460b1c8e51a2e9bcf33e06c512cba7d1ab4854f34321d2429cff887b1fb61"
CANONICAL_IDENTITY_SHA256 = "e5b62b5a614420d3d8260c2c0daa744043f956cd3aedcb05ca16301ea816b872"
CANONICAL_COLLECTION_PLAN_SHA256 = "d52ed433ee00a70bd95b5b453558d4432e83be4099f551e7d68c84602b0bbfe0"
CANONICAL_PLAN_SHA256 = "73b877ea1fcc138f606bc1e9d7f49c9beab4b0694ee61d5c66116dc3137e8ed4"
MAX_ARCHIVE_CHUNK = 64 * 1024


class BrokerIntegrityStop(runtime.IntegrityStop):
    """The closed broker contract was violated."""


def _require_uint(value: Any, maximum: int, label: str) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise BrokerIntegrityStop(f"{label} is outside its closed uint range")
    return value


def _quote_plus_utf8(value: str) -> str:
    if type(value) is not str:
        raise BrokerIntegrityStop("search query must be a string")
    output: list[str] = []
    for byte in value.encode("utf-8", errors="strict"):
        if (
            65 <= byte <= 90
            or 97 <= byte <= 122
            or 48 <= byte <= 57
            or byte in b"-._~"
        ):
            output.append(chr(byte))
        elif byte == 32:
            output.append("+")
        else:
            output.append(f"%{byte:02X}")
    return "".join(output)


def _compact_fixture_json(value: Any) -> bytes:
    return runtime.canonical_json(value)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_pinned_bytes(payload: bytes, expected_sha256: str, label: str) -> None:
    if type(payload) is not bytes or _sha256(payload) != expected_sha256:
        raise BrokerIntegrityStop(f"{label} bytes do not match their authenticated SHA-256")


def _read_hashed_file(path: Path, expected_sha256: str) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, MAX_ARCHIVE_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise BrokerIntegrityStop(f"pinned input mutated while reading: {path}")
        if digest.hexdigest() != expected_sha256:
            raise BrokerIntegrityStop(f"pinned input hash mismatch: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _header_map(headers: tuple[tuple[str, str], ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in headers:
        if type(name) is not str or type(value) is not str:
            raise BrokerIntegrityStop("fixture header is not a string pair")
        normalized = name.strip().casefold()
        if not normalized or normalized in result:
            raise BrokerIntegrityStop("fixture headers are empty or duplicated")
        result[normalized] = value.strip()
    return result


def _mime_type(headers: Mapping[str, str]) -> str | None:
    value = headers.get("content-type")
    if value is None:
        return None
    essence = value.split(";", 1)[0].strip().casefold()
    return essence or None


def _charset(headers: Mapping[str, str]) -> str | None:
    value = headers.get("content-type", "")
    for part in value.split(";")[1:]:
        key, separator, raw = part.partition("=")
        if separator and key.strip().casefold() == "charset":
            return raw.strip().strip('"\'')
    return None


def _siret(value: str) -> bool:
    return (
        type(value) is str
        and len(value) == 14
        and value.isascii()
        and value.isdigit()
        and replay.luhn_valid(value)
    )


@dataclass(frozen=True)
class PlannedQuery:
    query_id: str
    selection_ordinal: int
    query_ordinal: int
    search_query: str
    crm_name: str
    crm_address: str
    crm_postcode: str

    def __post_init__(self) -> None:
        if any(type(value) is not str or not value for value in (
            self.query_id, self.search_query, self.crm_name
        )) or any(type(value) is not str for value in (self.crm_address, self.crm_postcode)):
            raise BrokerIntegrityStop("planned query strings are invalid")
        if not 1 <= self.selection_ordinal <= 30:
            raise BrokerIntegrityStop("selection_ordinal must be 1..30")
        if not 1 <= self.query_ordinal <= 3:
            raise BrokerIntegrityStop("query_ordinal must be 1..3")

    def projection(self) -> list[Any]:
        return [
            self.query_id, self.selection_ordinal, self.query_ordinal,
            self.search_query, self.crm_name, self.crm_address, self.crm_postcode,
        ]


@dataclass(frozen=True)
class CanonicalPlan:
    rows: tuple[PlannedQuery, ...]
    canonical_bytes: bytes
    canonical_sha256: str


def load_canonical_plan_from_parquets(
    identity_path: Path,
    collection_plan_path: Path,
) -> CanonicalPlan:
    """Build the only admissible 30/90 plan from the two identity capabilities."""

    identity_bytes = _read_hashed_file(Path(identity_path), CANONICAL_IDENTITY_SHA256)
    collection_bytes = _read_hashed_file(Path(collection_plan_path), CANONICAL_COLLECTION_PLAN_SHA256)
    import pandas as pd

    identity = pd.read_parquet(io.BytesIO(identity_bytes))
    collection = pd.read_parquet(io.BytesIO(collection_bytes))
    if len(identity) != 30 or len(collection) != 90:
        raise BrokerIntegrityStop("pinned identity Parquets do not contain the exact 30/90 plan")
    identity_columns = {"query_id", "stratum", "crm_name", "crm_address", "crm_postcode"}
    collection_columns = {
        "query_id", "stratum", "query_ordinal", "search_query", "max_results_logged",
        "max_admissible_pages_opened", "max_admissible_pages_total_for_dossier",
    }
    if not identity_columns <= set(identity) or not collection_columns <= set(collection):
        raise BrokerIntegrityStop("pinned Parquet schema is incomplete")
    if identity["query_id"].astype(str).duplicated().any():
        raise BrokerIntegrityStop("identity query_id is duplicated")
    identity_map = {str(row.query_id): row for row in identity.itertuples(index=False)}
    selection_by_query = {
        str(row.query_id): index
        for index, row in enumerate(identity.itertuples(index=False), start=1)
    }
    rows: list[PlannedQuery] = []
    observed_collection: set[tuple[str, int]] = set()
    for raw in collection.itertuples(index=False):
        query_id = str(raw.query_id)
        query_ordinal = int(raw.query_ordinal)
        key = (query_id, query_ordinal)
        if key in observed_collection or query_id not in identity_map:
            raise BrokerIntegrityStop("collection plan has duplicate or unknown query rows")
        observed_collection.add(key)
        identity_row = identity_map[query_id]
        if str(raw.stratum) != str(identity_row.stratum):
            raise BrokerIntegrityStop("collection plan stratum differs from identity")
        if (int(raw.max_results_logged), int(raw.max_admissible_pages_opened), int(raw.max_admissible_pages_total_for_dossier)) != (5, 2, 6):
            raise BrokerIntegrityStop("collection plan quotas differ from 5/2/6")
        rows.append(PlannedQuery(
            query_id, selection_by_query[query_id], query_ordinal, str(raw.search_query),
            str(identity_row.crm_name), str(identity_row.crm_address), str(identity_row.crm_postcode),
        ))
    ordered = tuple(sorted(rows, key=lambda row: (row.selection_ordinal, row.query_ordinal)))
    canonical = runtime.canonical_json([row.projection() for row in ordered])
    observed_hash = _sha256(canonical)
    if observed_hash != CANONICAL_PLAN_SHA256:
        raise BrokerIntegrityStop("identity-derived canonical plan hash mismatch")
    return CanonicalPlan(ordered, canonical, observed_hash)


@dataclass(frozen=True)
class FixtureExchange:
    request_kind: str
    attempt_id: str
    hostname: str
    addresses: tuple[str, ...]
    terminal: str
    http_status: int | None = None
    headers: tuple[tuple[str, str], ...] = ()
    body_chunks: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        if self.request_kind not in {"SEARCH", "PAGE"}:
            raise BrokerIntegrityStop("fixture request kind is invalid")
        if type(self.attempt_id) is not str or not runtime.HEX64.fullmatch(self.attempt_id):
            raise BrokerIntegrityStop("fixture attempt id is invalid")
        if type(self.hostname) is not str or not self.hostname:
            raise BrokerIntegrityStop("fixture hostname is invalid")
        if type(self.addresses) is not tuple or any(type(value) is not str for value in self.addresses):
            raise BrokerIntegrityStop("fixture addresses are not a closed tuple")
        if self.terminal not in SEARCH_TERMINALS:
            raise BrokerIntegrityStop("fixture terminal is invalid")
        if type(self.headers) is not tuple or any(
            type(pair) is not tuple or len(pair) != 2 for pair in self.headers
        ):
            raise BrokerIntegrityStop("fixture headers are not closed")
        _header_map(self.headers)
        if self.terminal == "RESPONSE":
            _require_uint(self.http_status, 65535, "http_status")
            if type(self.body_chunks) is not tuple or any(
                type(chunk) is not bytes or len(chunk) > MAX_ARCHIVE_CHUNK
                for chunk in self.body_chunks
            ):
                raise BrokerIntegrityStop("fixture RESPONSE chunks are not bounded to 64 KiB")
        elif any(
            value not in {None, ()}
            for value in (self.http_status, self.headers, self.body_chunks)
        ):
            raise BrokerIntegrityStop("non-response fixture contains HTTP fields")

    def seal_projection(self) -> dict[str, Any]:
        return {
            "addresses": list(self.addresses),
            "attempt_id": self.attempt_id,
            "body_chunks_base64": [base64.b64encode(chunk).decode("ascii") for chunk in self.body_chunks],
            "headers": [list(pair) for pair in self.headers],
            "hostname": self.hostname,
            "http_status": self.http_status,
            "request_kind": self.request_kind,
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class SireneFixtureRecord:
    siret: str
    siren: str
    etat_administratif: str | None = None
    enseigne_1: str | None = None
    enseigne_2: str | None = None
    enseigne_3: str | None = None
    denomination_usuelle: str | None = None
    numero_voie: str | None = None
    type_voie: str | None = None
    libelle_voie: str | None = None
    code_postal: str | None = None
    libelle_commune: str | None = None
    code_commune: str | None = None

    def __post_init__(self) -> None:
        if not _siret(self.siret) or self.siren != self.siret[:9]:
            raise BrokerIntegrityStop("SIRENE fixture identifiers are invalid")
        optional = (
            self.etat_administratif, self.enseigne_1, self.enseigne_2,
            self.enseigne_3, self.denomination_usuelle, self.numero_voie,
            self.type_voie, self.libelle_voie, self.code_postal,
            self.libelle_commune, self.code_commune,
        )
        if any(value is not None and type(value) is not str for value in optional):
            raise BrokerIntegrityStop("SIRENE fixture contains a non-string scalar")

    def seal_projection(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@runtime_checkable
class BrokerTransport(Protocol):
    """Injected transport boundary; production implementations live elsewhere."""

    @property
    def fixture_sha256(self) -> str: ...

    @property
    def sirene_snapshot_sha256(self) -> str: ...

    def consume(self, request_kind: str, attempt_id: str, hostname: str) -> FixtureExchange: ...

    def iter_body_chunks(self, exchange: FixtureExchange) -> Iterator[bytes]: ...

    def records_for(self, siret: str) -> tuple[SireneFixtureRecord, ...]: ...

    def assert_fully_consumed(self) -> None: ...


@dataclass(frozen=True)
class ArchiveReceipt:
    relative_path: str
    byte_count: int
    content_sha256: str
    chunk_count: int
    maximum_chunk_size: int


@dataclass(frozen=True)
class StreamedBody:
    body: bytes | None
    wire_byte_count: int
    content_sha256: str
    chunk_count: int
    maximum_chunk_size: int
    too_large: bool

    def __post_init__(self) -> None:
        if (
            self.maximum_chunk_size > MAX_ARCHIVE_CHUNK
            or self.too_large != (self.body is None)
            or (self.body is not None and len(self.body) != self.wire_byte_count)
        ):
            raise BrokerIntegrityStop("streamed body violates bounded materialization invariants")


def _consume_bounded_body(
    transport: BrokerTransport, exchange: FixtureExchange, maximum_bytes: int,
) -> StreamedBody:
    """Consume every bounded chunk while materializing only bodies within the limit."""

    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise BrokerIntegrityStop("fixture body limit is invalid")
    digest = hashlib.sha256()
    materialized = bytearray()
    total = 0
    chunks = 0
    maximum = 0
    too_large = False
    for chunk in transport.iter_body_chunks(exchange):
        if type(chunk) is not bytes or len(chunk) > MAX_ARCHIVE_CHUNK:
            raise BrokerIntegrityStop("fixture transport yielded a chunk above 64 KiB")
        chunks += 1
        maximum = max(maximum, len(chunk))
        total += len(chunk)
        digest.update(chunk)
        if total > maximum_bytes:
            too_large = True
            materialized.clear()
            break
        materialized.extend(chunk)
    return StreamedBody(
        None if too_large else bytes(materialized), total, digest.hexdigest(),
        chunks, maximum, too_large,
    )


class ExclusiveArchiveStore:
    """Durably archives complete raw payloads with O_EXCL before any parser runs."""

    def __init__(self, root: Path, audit_hook: Callable[[str, str], None]):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._audit_hook = audit_hook

    @property
    def audit_hook(self) -> Callable[[str, str], None]:
        return self._audit_hook

    def archive(self, category: str, attempt_id: str, payload: bytes) -> ArchiveReceipt:
        if type(payload) is not bytes or not runtime.HEX64.fullmatch(attempt_id):
            raise BrokerIntegrityStop("archive input is not closed")
        relative = f"{category}/{attempt_id}.bin"
        self._audit_hook("WRITE_LOCAL", relative)
        directory = self._root / category
        directory.mkdir(mode=0o700, exist_ok=True)
        target = directory / f"{attempt_id}.bin"
        try:
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                0o400,
            )
        except FileExistsError as exc:
            raise BrokerIntegrityStop("raw archive already exists or was pre-created") from exc
        digest = hashlib.sha256()
        chunks = 0
        maximum = 0
        try:
            view = memoryview(payload)
            for offset in range(0, len(payload), MAX_ARCHIVE_CHUNK):
                chunk = view[offset : offset + MAX_ARCHIVE_CHUNK]
                maximum = max(maximum, len(chunk))
                chunks += 1
                written = 0
                while written < len(chunk):
                    count = os.write(descriptor, chunk[written:])
                    if count <= 0:
                        raise BrokerIntegrityStop("archive write made no progress")
                    digest.update(chunk[written : written + count])
                    written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return ArchiveReceipt(relative, len(payload), digest.hexdigest(), chunks, maximum)


class SealedFixtureTransport:
    """One-shot in-memory exchange store authenticated by caller-supplied hashes."""

    def __init__(
        self,
        exchanges: Sequence[FixtureExchange],
        sirene_records: Sequence[SireneFixtureRecord],
        expected_fixture_sha256: str,
        expected_sirene_snapshot_sha256: str,
    ):
        if type(expected_fixture_sha256) is not str or not runtime.HEX64.fullmatch(expected_fixture_sha256):
            raise BrokerIntegrityStop("fixture transport seal is invalid")
        if type(expected_sirene_snapshot_sha256) is not str or not runtime.HEX64.fullmatch(expected_sirene_snapshot_sha256):
            raise BrokerIntegrityStop("SIRENE snapshot seal is invalid")
        exchange_tuple = tuple(exchanges)
        record_tuple = tuple(sirene_records)
        keys = [(item.request_kind, item.attempt_id) for item in exchange_tuple]
        if len(keys) != len(set(keys)):
            raise BrokerIntegrityStop("fixture exchange key is duplicated")
        projection = {
            "exchanges": [item.seal_projection() for item in exchange_tuple],
            "schema_version": FIXTURE_SCHEMA,
        }
        observed = hashlib.sha256(_compact_fixture_json(projection)).hexdigest()
        if observed != expected_fixture_sha256:
            raise BrokerIntegrityStop("fixture transport seal mismatch")
        record_projection = [item.seal_projection() for item in sorted(record_tuple, key=lambda item: item.siret)]
        observed_snapshot = hashlib.sha256(_compact_fixture_json(record_projection)).hexdigest()
        if observed_snapshot != expected_sirene_snapshot_sha256:
            raise BrokerIntegrityStop("SIRENE snapshot seal mismatch")
        self._fixture_sha256 = observed
        self._sirene_snapshot_sha256 = observed_snapshot
        self._exchanges = dict(zip(keys, exchange_tuple, strict=True))
        self._consumed: set[tuple[str, str]] = set()
        self._streamed: set[tuple[str, str]] = set()
        records: dict[str, list[SireneFixtureRecord]] = {}
        for record in record_tuple:
            records.setdefault(record.siret, []).append(record)
        self._sirene_records = {key: tuple(value) for key, value in records.items()}

    @property
    def fixture_sha256(self) -> str:
        return self._fixture_sha256

    @property
    def sirene_snapshot_sha256(self) -> str:
        return self._sirene_snapshot_sha256

    def consume(self, request_kind: str, attempt_id: str, hostname: str) -> FixtureExchange:
        key = (request_kind, attempt_id)
        if key in self._consumed:
            raise BrokerIntegrityStop("fixture exchange retry is forbidden")
        exchange = self._exchanges.get(key)
        if exchange is None or exchange.hostname != hostname:
            raise BrokerIntegrityStop("fixture exchange is absent or hostname-bound elsewhere")
        self._consumed.add(key)
        return exchange

    def iter_body_chunks(self, exchange: FixtureExchange) -> Iterator[bytes]:
        key = (exchange.request_kind, exchange.attempt_id)
        if key not in self._consumed or key in self._streamed or exchange.terminal != "RESPONSE":
            raise BrokerIntegrityStop("fixture body stream is absent, premature, or repeated")
        self._streamed.add(key)
        for chunk in exchange.body_chunks:
            if len(chunk) > MAX_ARCHIVE_CHUNK:
                raise BrokerIntegrityStop("fixture transport yielded a chunk above 64 KiB")
            yield chunk

    def records_for(self, siret: str) -> tuple[SireneFixtureRecord, ...]:
        return tuple(sorted(
            self._sirene_records.get(siret, ()),
            key=lambda row: runtime.canonical_json(row.seal_projection()),
        ))

    def assert_fully_consumed(self) -> None:
        if self._consumed != set(self._exchanges):
            raise BrokerIntegrityStop("sealed fixture contains unconsumed HTTP exchanges")
        expected_streams = {
            key for key, exchange in self._exchanges.items() if exchange.terminal == "RESPONSE"
        }
        if self._streamed != expected_streams:
            raise BrokerIntegrityStop("sealed fixture contains an unconsumed response body stream")


@dataclass(frozen=True)
class BrokerError:
    stage: str
    error_type: str
    code: str
    errno: int | None = None
    http_status: int | None = None

    def __post_init__(self) -> None:
        if self.stage not in ERROR_STAGES or self.error_type not in ERROR_TYPES or type(self.code) is not str or not self.code:
            raise BrokerIntegrityStop("broker error schema is invalid")
        if self.errno is not None and type(self.errno) is not int:
            raise BrokerIntegrityStop("broker errno is invalid")
        if self.http_status is not None:
            _require_uint(self.http_status, 65535, "error http_status")


@dataclass(frozen=True)
class DNSResolution:
    dns_attempt_id: str
    parent_attempt_id: str
    request_kind: str
    normalized_hostname: str
    addresses: tuple[str, ...]
    all_addresses_permitted: bool
    chosen_ip: str | None
    error_type: str | None

    def __post_init__(self) -> None:
        if (
            self.request_kind not in {"SEARCH", "PAGE"}
            or type(self.addresses) is not tuple
            or type(self.all_addresses_permitted) is not bool
            or self.error_type not in {None, "DNS", "PRIVATE_ADDRESS"}
            or (self.all_addresses_permitted != (self.chosen_ip is not None and self.error_type is None))
        ):
            raise BrokerIntegrityStop("DNS response schema is invalid")


@dataclass(frozen=True)
class SearchRequest:
    method: str
    url: str
    hostname: str
    port: int
    headers: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (self.method, self.hostname, self.port, self.headers) != ("GET", SEARCH_HOST, 443, SEARCH_HEADERS):
            raise BrokerIntegrityStop("search request schema differs from the frozen request")


@dataclass(frozen=True)
class OrganicResult:
    query_id: str
    query_ordinal: int
    result_rank: int
    title: str
    snippet: str
    observed_href: str
    resolved_url: str
    normalized_hostname: str | None
    registrable_domain: str | None
    preopen_family: str
    inadmissible_reason: str

    def __post_init__(self) -> None:
        if not 1 <= self.query_ordinal <= 3:
            raise BrokerIntegrityStop("result query_ordinal must be 1..3")
        if not 1 <= self.result_rank <= 5:
            raise BrokerIntegrityStop("organic result rank exceeds top five")


@dataclass(frozen=True)
class SearchResponse:
    schema_version: str
    operation: str
    query_id: str
    selection_ordinal: int
    query_ordinal: int
    search_attempt_id: str
    request: SearchRequest
    dns: DNSResolution
    status: str
    http_status: int | None
    mime_type: str | None
    content_encoding: str | None
    archive: ArchiveReceipt | None
    results: tuple[OrganicResult, ...]
    error: BrokerError | None

    def __post_init__(self) -> None:
        if self.schema_version != BROKER_SCHEMA or self.operation != "SEARCH_RESPONSE":
            raise BrokerIntegrityStop("search response envelope is invalid")
        if self.status not in {"SUCCESS", "NETWORK_ERROR", "TIMEOUT", "HTTP_ERROR", "PARSE_ERROR"}:
            raise BrokerIntegrityStop("search response status is invalid")
        if (self.status == "SUCCESS") != (self.error is None):
            raise BrokerIntegrityStop("search response error presence is invalid")
        if type(self.results) is not tuple or len(self.results) > 5 or (self.status != "SUCCESS" and self.results):
            raise BrokerIntegrityStop("search response results violate top-five/status rules")


@dataclass(frozen=True)
class PageRequest:
    method: str
    url: str
    hostname: str
    port: int
    sni: str
    connected_ip: str | None
    headers: tuple[tuple[str, str], ...]
    follow_redirects: bool
    retry_count: int

    def __post_init__(self) -> None:
        expected_headers = (("Host", self.hostname), *PAGE_HEADER_TAIL)
        if (
            self.method != "GET" or self.port != 443 or self.sni != self.hostname
            or self.headers != expected_headers or self.follow_redirects is not False
            or self.retry_count != 0
        ):
            raise BrokerIntegrityStop("page request schema differs from the frozen request")


@dataclass(frozen=True)
class PageResponse:
    schema_version: str
    operation: str
    query_id: str
    query_ordinal: int
    result_rank: int
    decision: str
    domain_first_seen_query_ordinal: int | None
    domain_first_seen_result_rank: int | None
    query_open_slot: int | None
    dossier_open_ordinal: int | None
    page_attempt_id: str | None
    request: PageRequest | None
    dns: DNSResolution | None
    status: str
    http_status: int | None
    mime_type: str | None
    content_encoding: str | None
    archive: ArchiveReceipt | None
    postopen_family: str
    facts_eligible: bool
    qualified_sirets: tuple[str, ...]
    error: BrokerError | None

    def __post_init__(self) -> None:
        if self.schema_version != BROKER_SCHEMA or self.operation != "PAGE_RESPONSE":
            raise BrokerIntegrityStop("page response envelope is invalid")
        decisions = {
            "OPEN_ATTEMPT", "SKIP_INADMISSIBLE", "SKIP_DUPLICATE_DOMAIN",
            "SKIP_QUERY_QUOTA", "SKIP_DOSSIER_QUOTA",
        }
        if self.decision not in decisions:
            raise BrokerIntegrityStop("page decision is invalid")
        open_fields = (
            self.query_open_slot, self.dossier_open_ordinal, self.page_attempt_id,
            self.request, self.dns,
        )
        if self.decision == "OPEN_ATTEMPT":
            if any(value is None for value in open_fields) or self.status == "SKIPPED":
                raise BrokerIntegrityStop("OPEN_ATTEMPT response fields are incomplete")
        elif any(value is not None for value in open_fields) or self.status != "SKIPPED":
            raise BrokerIntegrityStop("skipped page response contains open-attempt fields")
        if self.status == "SUCCESS" and self.error is not None:
            raise BrokerIntegrityStop("successful page response contains an error")
        if self.status in {"NETWORK_ERROR", "TIMEOUT", "HTTP_ERROR", "PARSE_ERROR"} and self.error is None:
            raise BrokerIntegrityStop("failed page response lacks a typed error")
        if self.facts_eligible != bool(self.qualified_sirets) or (self.facts_eligible and self.status != "SUCCESS"):
            raise BrokerIntegrityStop("page qualification fields are inconsistent")
        if self.archive is not None and self.archive.maximum_chunk_size > MAX_ARCHIVE_CHUNK:
            raise BrokerIntegrityStop("archive chunk exceeded 64 KiB")


@dataclass(frozen=True)
class SireneLookupResponse:
    schema_version: str
    operation: str
    query_id: str
    siret: str
    found_exactly_once: bool
    record: SireneFixtureRecord | None
    sirene_snapshot_sha256: str
    record_sha256: str
    served_from_global_cache: bool

    def __post_init__(self) -> None:
        if (
            self.schema_version != BROKER_SCHEMA
            or self.operation != "SIRENE_LOOKUP_RESPONSE"
            or self.found_exactly_once != (self.record is not None)
            or not _siret(self.siret)
            or not runtime.HEX64.fullmatch(self.sirene_snapshot_sha256)
            or not runtime.HEX64.fullmatch(self.record_sha256)
            or type(self.served_from_global_cache) is not bool
        ):
            raise BrokerIntegrityStop("SIRENE lookup response schema is invalid")


def _transport_error(terminal: str, stage: str) -> BrokerError:
    mapping = {
        "DNS_ERROR": "DNS",
        "CONNECT_ERROR": "NETWORK",
        "CONNECT_TIMEOUT": "CONNECT_TIMEOUT",
        "TLS_ERROR": "TLS",
        "WRITE_ERROR": "NETWORK",
        "NO_RESPONSE_HEADERS": "NETWORK",
        "READ_TIMEOUT": "READ_TIMEOUT",
        "NETWORK_ERROR": "NETWORK",
        "PARSE_ERROR": "PARSE",
    }
    return BrokerError(stage, mapping[terminal], terminal)


class InjectedOfflineBroker:
    """Monotone V4.12 broker whose only transport is a sealed fixture."""

    def __init__(
        self,
        *,
        journal: runtime.AccessJournal,
        plan: Sequence[PlannedQuery],
        transport: BrokerTransport,
        expected_transport_fixture_sha256: str,
        expected_sirene_snapshot_sha256: str,
        policy_bytes: bytes,
        public_suffix_bytes: bytes,
        dns_vectors_bytes: bytes,
        archive_store: ExclusiveArchiveStore,
        audit_hook: Callable[[str, str], None],
    ):
        if type(journal) is not runtime.AccessJournal or journal.state != "IDENTITY_NETWORK_OPEN":
            raise BrokerIntegrityStop("broker requires the live runtime journal")
        if not isinstance(transport, BrokerTransport):
            raise BrokerIntegrityStop("broker transport does not implement the injected protocol")
        if (
            type(archive_store) is not ExclusiveArchiveStore or not callable(audit_hook)
            or archive_store.audit_hook is not audit_hook
        ):
            raise BrokerIntegrityStop("archive store or audit hook is invalid")
        if (
            not runtime.HEX64.fullmatch(expected_transport_fixture_sha256)
            or not runtime.HEX64.fullmatch(expected_sirene_snapshot_sha256)
            or transport.fixture_sha256 != expected_transport_fixture_sha256
            or transport.sirene_snapshot_sha256 != expected_sirene_snapshot_sha256
        ):
            raise BrokerIntegrityStop("transport or SIRENE snapshot external hash mismatch")
        self._plan = tuple(plan)
        plan_canonical_bytes = runtime.canonical_json([row.projection() for row in self._plan])
        _require_pinned_bytes(plan_canonical_bytes, CANONICAL_PLAN_SHA256, "canonical plan")
        self._validate_plan()
        _require_pinned_bytes(policy_bytes, POLICY_SHA256, "policy")
        _require_pinned_bytes(public_suffix_bytes, PUBLIC_SUFFIXES_SHA256, "public suffix list")
        _require_pinned_bytes(dns_vectors_bytes, DNS_VECTORS_SHA256, "DNS CIDR vectors")
        try:
            policy_copy = json.loads(policy_bytes)
            dns_vectors = json.loads(dns_vectors_bytes)
            suffixes = tuple(line.strip() for line in public_suffix_bytes.decode("utf-8").splitlines() if line.strip())
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise BrokerIntegrityStop("authenticated policy inputs are malformed") from exc
        if policy_copy.get("schema_version") != "sireto-v4.12-r30-collection-policy-1":
            raise BrokerIntegrityStop("collection policy schema mismatch")
        search_policy = policy_copy.get("search_engine", {})
        page_policy = policy_copy.get("page_fetch", {})
        state_policy = policy_copy.get("collection_state_machine", {})
        expected_limits = (
            search_policy.get("maximum_results_logged") == 5,
            search_policy.get("retry_count") == 0,
            search_policy.get("maximum_wire_bytes") == 2 * 1024 * 1024,
            search_policy.get("maximum_decoded_bytes") == 2 * 1024 * 1024,
            page_policy.get("maximum_open_attempts_per_query") == 2,
            page_policy.get("maximum_open_attempts_per_dossier") == 6,
            page_policy.get("retry_count") == 0,
            page_policy.get("maximum_redirects") == 0,
            page_policy.get("maximum_wire_bytes") == 10 * 1024 * 1024,
            page_policy.get("maximum_decoded_bytes") == 10 * 1024 * 1024,
            state_policy.get("search_attempts_per_dossier") == 3,
            state_policy.get("pagination_allowed") is False,
            state_policy.get("query_reformulation_allowed") is False,
            state_policy.get("duplicate_domain_consumes_open_slot") is False,
        )
        if not all(expected_limits):
            raise BrokerIntegrityStop("collection quotas or no-retry policy differ from the frozen contract")
        self._policy = policy_copy
        self._suffixes = suffixes
        if not self._suffixes or any(type(value) is not str or not value for value in self._suffixes):
            raise BrokerIntegrityStop("public suffix set is invalid")
        try:
            self._networks = tuple(ipaddress.ip_network(value, strict=True) for value in dns_vectors["forbidden_cidrs"])
        except ValueError as exc:
            raise BrokerIntegrityStop("forbidden CIDR table is invalid") from exc
        self._journal = journal
        self._gate = runtime.OfflineBroker(journal)
        self._transport = transport
        self._archive_store = archive_store
        self._audit_hook = audit_hook
        self._next_search = 0
        self._responses: dict[tuple[str, int], SearchResponse] = {}
        self._page_responses: dict[tuple[str, int, int], PageResponse] = {}
        self._query_open_count: dict[tuple[str, int], int] = {}
        self._dossier_open_count: dict[str, int] = {}
        self._opened_domains: dict[str, dict[str, tuple[int, int]]] = {}
        self._qualified: set[tuple[str, str]] = set()
        self._looked_up: set[tuple[str, str]] = set()
        self._sirene_cache: dict[str, tuple[tuple[SireneFixtureRecord, ...], str]] = {}
        self._next_result_rank: dict[tuple[str, int], int] = {}
        self._poisoned = False

    @property
    def state(self) -> str:
        return "STOP_INTEGRITY" if self._poisoned else self._gate.state

    @property
    def sirene_lookup_plan(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self._qualified))

    def _fail(self, message: str) -> None:
        self._poisoned = True
        raise BrokerIntegrityStop(message)

    def _validate_plan(self) -> None:
        if len(self._plan) != 90 or any(type(value) is not PlannedQuery for value in self._plan):
            raise BrokerIntegrityStop("plan must contain exactly 90 closed queries")
        observed: dict[str, list[PlannedQuery]] = {}
        for item in self._plan:
            observed.setdefault(item.query_id, []).append(item)
        if len(observed) != 30:
            raise BrokerIntegrityStop("plan must contain exactly 30 dossiers")
        expected_order: list[tuple[int, int]] = []
        for query_id, rows in observed.items():
            selections = {row.selection_ordinal for row in rows}
            ordinals = {row.query_ordinal for row in rows}
            if len(rows) != 3 or len(selections) != 1 or ordinals != {1, 2, 3}:
                raise BrokerIntegrityStop(f"dossier {query_id!r} does not have three exact queries")
            expected_order.extend((rows[0].selection_ordinal, ordinal) for ordinal in range(1, 4))
        if sorted({item.selection_ordinal for item in self._plan}) != list(range(1, 31)):
            raise BrokerIntegrityStop("selection ordinals are not contiguous 1..30")
        if [(item.selection_ordinal, item.query_ordinal) for item in self._plan] != sorted(expected_order):
            raise BrokerIntegrityStop("plan order differs from selection/query ordinal order")

    def _ensure_open(self, operation: str, query_id: str, query_ordinal: int, target: str) -> None:
        if self._poisoned:
            raise BrokerIntegrityStop("broker is terminal after STOP_INTEGRITY")
        if self.state == "IDENTITY_NETWORK_OPEN":
            return
        intent = self._journal.intent(
            phase="IDENTITY_DISCOVERY",
            operation=operation,
            target_kind="SIRET" if operation == "SIRENE_LOOKUP" else "URL",
            target_canonical=target,
            query_id=query_id,
            query_ordinal=query_ordinal,
        )
        self._journal.result(intent, outcome="STOP_INTEGRITY", error_type="IO_INTEGRITY")
        raise BrokerIntegrityStop("broker capability was irreversibly revoked")

    def _dns(self, parent_attempt_id: str, request_kind: str, plan: PlannedQuery, hostname: str, addresses: tuple[str, ...]) -> DNSResolution:
        dns_id = runtime.dns_attempt_id(parent_attempt_id, hostname)
        intent = self._journal.intent(
            phase="IDENTITY_DISCOVERY",
            operation="DNS_RESOLUTION",
            target_kind="HOSTNAME",
            target_canonical=hostname,
            query_id=plan.query_id,
            query_ordinal=plan.query_ordinal,
        )
        try:
            self._audit_hook("DNS_RESOLUTION", hostname)
            decision = replay.evaluate_resolution(list(addresses), list(self._networks))
        except ValueError as exc:
            self._journal.result(intent, outcome="NETWORK_ERROR", error_type="DNS")
            raise BrokerIntegrityStop("fixture DNS address is malformed") from exc
        if decision["permitted"]:
            self._journal.result(intent, outcome="SUCCESS")
        else:
            self._journal.result(intent, outcome="NETWORK_ERROR", error_type=decision["error_type"])
        return DNSResolution(
            dns_id,
            parent_attempt_id,
            request_kind,
            hostname,
            tuple(decision["expected_addresses"]),
            decision["permitted"],
            decision["chosen_ip"],
            decision["error_type"],
        )

    def _journal_terminal(
        self, intent: runtime.Intent, status: str, error: BrokerError | None,
        http_status: int | None, archive: ArchiveReceipt | None = None,
    ) -> None:
        if error is None:
            self._journal.result(
                intent, outcome="SUCCESS",
                byte_count=None if archive is None else archive.byte_count,
                content_sha256=None if archive is None else archive.content_sha256,
            )
        elif status == "TIMEOUT":
            self._journal.result(intent, outcome="TIMEOUT", error_type=error.error_type)
        elif status == "HTTP_ERROR":
            self._journal.result(intent, outcome="HTTP_ERROR", http_status=http_status or 0)
        elif status == "PARSE_ERROR":
            self._journal.result(intent, outcome="PARSE_ERROR", error_type="PARSE")
        else:
            journal_error = error.error_type if error.error_type in {"DNS", "PRIVATE_ADDRESS", "NETWORK", "TLS"} else "NETWORK"
            self._journal.result(intent, outcome="NETWORK_ERROR", error_type=journal_error)

    def search(self, query_id: str, query_ordinal: int) -> SearchResponse:
        self._ensure_open("SEARCH_REQUEST", query_id, query_ordinal, SEARCH_ENDPOINT)
        if self._next_search:
            previous = self._responses[(self._plan[self._next_search - 1].query_id, self._plan[self._next_search - 1].query_ordinal)]
            expected = {(previous.query_id, previous.query_ordinal, row.result_rank) for row in previous.results}
            if not expected <= set(self._page_responses):
                self._fail("search cannot advance before all prior result ranks are decided")
        if self._next_search >= len(self._plan):
            self._fail("all 90 searches were already consumed")
        plan = self._plan[self._next_search]
        if (query_id, query_ordinal) != (plan.query_id, plan.query_ordinal):
            self._fail("search differs from the next sealed plan row")
        key = (query_id, query_ordinal)
        if key in self._responses:
            self._fail("search retry is forbidden")
        attempt_id = runtime.search_attempt_id(query_id, query_ordinal, plan.search_query)
        url = SEARCH_ENDPOINT + "?q=" + _quote_plus_utf8(plan.search_query)
        request = SearchRequest("GET", url, SEARCH_HOST, 443, SEARCH_HEADERS)
        intent = self._journal.intent(
            phase="IDENTITY_DISCOVERY",
            operation="SEARCH_REQUEST",
            target_kind="URL",
            target_canonical=url,
            query_id=query_id,
            query_ordinal=query_ordinal,
        )
        try:
            self._audit_hook("SEARCH_REQUEST", url)
            exchange = self._transport.consume("SEARCH", attempt_id, SEARCH_HOST)
        except Exception:
            self._journal.result(intent, outcome="STOP_INTEGRITY", error_type="IO_INTEGRITY")
            self._poisoned = True
            raise
        try:
            dns = self._dns(attempt_id, "SEARCH", plan, SEARCH_HOST, exchange.addresses)
        except Exception:
            self._journal.result(intent, outcome="STOP_INTEGRITY", error_type="IO_INTEGRITY")
            self._poisoned = True
            raise
        status = "NETWORK_ERROR"
        http_status: int | None = None
        mime: str | None = None
        encoding: str | None = None
        archive: ArchiveReceipt | None = None
        results: tuple[OrganicResult, ...] = ()
        error: BrokerError | None
        if not dns.all_addresses_permitted:
            error = BrokerError("DNS", dns.error_type or "DNS", dns.error_type or "DNS")
        elif exchange.terminal != "RESPONSE":
            error = _transport_error(exchange.terminal, "SEARCH")
            status = "TIMEOUT" if error.error_type in {"CONNECT_TIMEOUT", "READ_TIMEOUT"} else (
                "PARSE_ERROR" if error.error_type == "PARSE" else "NETWORK_ERROR"
            )
        else:
            http_status = exchange.http_status
            headers = _header_map(exchange.headers)
            mime = _mime_type(headers)
            encoding = headers.get("content-encoding", "identity").casefold()
            try:
                streamed = _consume_bounded_body(self._transport, exchange, 2 * 1024 * 1024)
            except Exception:
                self._journal.result(intent, outcome="STOP_INTEGRITY", error_type="IO_INTEGRITY")
                self._poisoned = True
                raise
            body = streamed.body or b""
            if 300 <= http_status <= 399:
                status, error = "HTTP_ERROR", BrokerError("HTTP", "REDIRECT_FORBIDDEN", "REDIRECT_FORBIDDEN", http_status=http_status)
            elif not 200 <= http_status <= 299:
                status, error = "HTTP_ERROR", BrokerError("HTTP", "HTTP_STATUS", "HTTP_STATUS", http_status=http_status)
            elif streamed.too_large:
                status, error = "HTTP_ERROR", BrokerError("HTTP", "TOO_LARGE", "TOO_LARGE")
            elif encoding != "identity":
                status, error = "HTTP_ERROR", BrokerError("HTTP", "CONTENT_ENCODING", "CONTENT_ENCODING")
            elif mime != "text/html":
                status, error = "HTTP_ERROR", BrokerError("HTTP", "UNSUPPORTED_MIME", "UNSUPPORTED_MIME")
            else:
                try:
                    archive = self._archive_store.archive("search", attempt_id, body)
                except Exception:
                    self._journal.result(intent, outcome="STOP_INTEGRITY", error_type="IO_INTEGRITY")
                    self._poisoned = True
                    raise
                try:
                    parsed = replay.parse_ddg_results(body, _charset(headers))
                except Exception as exc:
                    status, error = "PARSE_ERROR", BrokerError("PARSE", "PARSE", "PARSE")
                    del exc
                else:
                    rows: list[OrganicResult] = []
                    for rank, raw_result in enumerate(parsed[:5], start=1):
                        classified = replay.classify_preopen(
                            raw_result["resolved_url"], raw_result["title"], raw_result["snippet"],
                            plan.crm_name, self._policy, list(self._suffixes),
                        )
                        rows.append(
                            OrganicResult(
                                query_id, query_ordinal, rank, raw_result["title"], raw_result["snippet"],
                                raw_result["observed_href"], raw_result["resolved_url"],
                                classified["normalized_hostname"], classified["registrable_domain"],
                                classified["family"], classified["reason"],
                            )
                        )
                    status, error, results = "SUCCESS", None, tuple(rows)
        response = SearchResponse(
            BROKER_SCHEMA, "SEARCH_RESPONSE", query_id, plan.selection_ordinal,
            query_ordinal, attempt_id, request, dns, status, http_status, mime,
            encoding, archive, results, error,
        )
        self._journal_terminal(intent, status, error, http_status, archive)
        self._responses[key] = response
        self._next_result_rank[key] = 1
        self._next_search += 1
        return response

    def open_page(self, query_id: str, query_ordinal: int, result_rank: int) -> PageResponse:
        self._ensure_open("PAGE_REQUEST", query_id, query_ordinal, "fixture://page")
        key = (query_id, query_ordinal, result_rank)
        if key in self._page_responses:
            self._fail("page retry is forbidden")
        search = self._responses.get((query_id, query_ordinal))
        if search is None or search.status != "SUCCESS":
            self._fail("page does not reference a successful archived search")
        if type(result_rank) is not int or not 1 <= result_rank <= len(search.results):
            self._fail("page result rank is outside archived top five")
        expected_rank = self._next_result_rank.get((query_id, query_ordinal), 1)
        if result_rank != expected_rank:
            self._fail("page decisions must follow strict archived result-rank order")
        result = search.results[result_rank - 1]
        first_query: int | None = None
        first_rank: int | None = None
        query_slot: int | None = None
        dossier_slot: int | None = None
        page_id: str | None = None
        request: PageRequest | None = None
        dns: DNSResolution | None = None
        status = "SKIPPED"
        http_status: int | None = None
        mime: str | None = None
        encoding: str | None = None
        archive: ArchiveReceipt | None = None
        final_family = "INADMISSIBLE_AFTER_OPEN"
        facts_eligible = False
        qualified: tuple[str, ...] = ()
        error: BrokerError | None = None
        domain = result.registrable_domain
        opened = self._opened_domains.setdefault(query_id, {})
        query_count = self._query_open_count.get((query_id, query_ordinal), 0)
        dossier_count = self._dossier_open_count.get(query_id, 0)
        if result.preopen_family not in ADMISSIBLE_FAMILIES or domain is None:
            decision = "SKIP_INADMISSIBLE"
        elif domain in opened:
            decision = "SKIP_DUPLICATE_DOMAIN"
            first_query, first_rank = opened[domain]
        elif query_count >= 2:
            decision = "SKIP_QUERY_QUOTA"
        elif dossier_count >= 6:
            decision = "SKIP_DOSSIER_QUOTA"
        else:
            decision = "OPEN_ATTEMPT"
            query_slot, dossier_slot = query_count + 1, dossier_count + 1
            page_id = runtime.page_attempt_id(
                query_id, query_ordinal, result_rank, result.resolved_url,
                query_slot, dossier_slot,
            )
            opened[domain] = (query_ordinal, result_rank)
            self._query_open_count[(query_id, query_ordinal)] = query_slot
            self._dossier_open_count[query_id] = dossier_slot
            intent = self._journal.intent(
                phase="IDENTITY_DISCOVERY", operation="PAGE_REQUEST", target_kind="URL",
                target_canonical=result.resolved_url, query_id=query_id,
                query_ordinal=query_ordinal, result_rank=result_rank,
            )
            try:
                self._audit_hook("PAGE_REQUEST", result.resolved_url)
                exchange = self._transport.consume("PAGE", page_id, result.normalized_hostname or "")
                dns = self._dns(
                    page_id, "PAGE", self._plan_for(query_id, query_ordinal),
                    result.normalized_hostname or "", exchange.addresses,
                )
            except Exception:
                self._journal.result(intent, outcome="STOP_INTEGRITY", error_type="IO_INTEGRITY")
                self._poisoned = True
                raise
            request = PageRequest(
                "GET", result.resolved_url, result.normalized_hostname or "", 443,
                result.normalized_hostname or "", dns.chosen_ip,
                (("Host", result.normalized_hostname or ""), *PAGE_HEADER_TAIL), False, 0,
            )
            if not dns.all_addresses_permitted:
                error = BrokerError("DNS", dns.error_type or "DNS", dns.error_type or "DNS")
                status = "NETWORK_ERROR"
            elif exchange.terminal != "RESPONSE":
                error = _transport_error(exchange.terminal, "PAGE_OPEN")
                status = "TIMEOUT" if error.error_type in {"CONNECT_TIMEOUT", "READ_TIMEOUT"} else (
                    "PARSE_ERROR" if error.error_type == "PARSE" else "NETWORK_ERROR"
                )
            else:
                http_status = exchange.http_status
                headers = _header_map(exchange.headers)
                mime = _mime_type(headers)
                encoding = headers.get("content-encoding", "identity").casefold()
                try:
                    streamed = _consume_bounded_body(self._transport, exchange, 10 * 1024 * 1024)
                except Exception:
                    self._journal.result(intent, outcome="STOP_INTEGRITY", error_type="IO_INTEGRITY")
                    self._poisoned = True
                    raise
                body = streamed.body or b""
                if 300 <= http_status <= 399:
                    status, error = "HTTP_ERROR", BrokerError("HTTP", "REDIRECT_FORBIDDEN", "REDIRECT_FORBIDDEN", http_status=http_status)
                elif not 200 <= http_status <= 299:
                    status, error = "HTTP_ERROR", BrokerError("HTTP", "HTTP_STATUS", "HTTP_STATUS", http_status=http_status)
                elif streamed.too_large:
                    status, error = "HTTP_ERROR", BrokerError("ARCHIVE", "TOO_LARGE", "TOO_LARGE")
                elif encoding != "identity":
                    status, error = "HTTP_ERROR", BrokerError("HTTP", "CONTENT_ENCODING", "CONTENT_ENCODING")
                elif mime is None or mime not in {"text/html", "text/plain", "application/pdf"}:
                    status, error = "HTTP_ERROR", BrokerError("HTTP", "UNSUPPORTED_MIME", "UNSUPPORTED_MIME")
                else:
                    try:
                        archive = self._archive_store.archive("pages", page_id, body)
                        if mime == "text/html":
                            decoded_source = body.decode(
                                "iso-8859-1" if (_charset(headers) or "").casefold() in {"iso-8859-1", "latin1", "latin-1"} else "utf-8",
                                errors="replace",
                            )
                            decoded = " ".join(html.fromstring(decoded_source).text_content().split())
                        elif mime == "text/plain":
                            decoded = " ".join(body.decode(
                                "iso-8859-1" if (_charset(headers) or "").casefold() in {"iso-8859-1", "latin1", "latin-1"} else "utf-8",
                                errors="replace",
                            ).split())
                        else:
                            import pypdfium2
                            document = pypdfium2.PdfDocument(body)
                            if len(document) > 50:
                                raise BrokerIntegrityStop("PDF exceeds the 50-page extraction ceiling")
                            page_texts: list[str] = []
                            for page_index in range(len(document)):
                                page = document[page_index]
                                text_page = page.get_textpage()
                                page_texts.append(text_page.get_text_range())
                            decoded = " ".join(" ".join(page_texts).split())
                            if len(decoded.encode("utf-8")) > 2 * 1024 * 1024:
                                raise BrokerIntegrityStop("PDF extracted text exceeds 2 MiB")
                        plan = self._plan_for(query_id, query_ordinal)
                        domain_policy = self._policy["domain_policy"]
                        facts, _ = replay.reconstruct_facts(
                            decoded, plan.crm_name, plan.crm_address, plan.crm_postcode,
                            stopwords=frozenset(domain_policy["name_stopwords"]),
                            minimum_name_token_length=domain_policy["significant_token_minimum_length"],
                        )
                        vector = {
                            "preopen_family": result.preopen_family,
                            "extracted_text": decoded,
                            "crm_name": plan.crm_name,
                            "crm_address": plan.crm_address,
                            "crm_postcode": plan.crm_postcode,
                            "normalized_hostname": result.normalized_hostname or "",
                        }
                        final_family, facts_eligible = replay.postopen_family(vector, self._policy)
                        qualified = tuple(dict.fromkeys(str(row[0]) for row in facts)) if facts_eligible else ()
                        self._qualified.update((query_id, siret) for siret in qualified)
                    except Exception:
                        self._journal.result(intent, outcome="STOP_INTEGRITY", error_type="IO_INTEGRITY")
                        self._poisoned = True
                        raise
                    status, error = "SUCCESS", None
            self._journal_terminal(intent, status, error, http_status, archive)
        response = PageResponse(
            BROKER_SCHEMA, "PAGE_RESPONSE", query_id, query_ordinal, result_rank,
            decision, first_query, first_rank, query_slot, dossier_slot, page_id,
            request, dns, status, http_status, mime, encoding, archive, final_family, facts_eligible,
            qualified, error,
        )
        self._page_responses[key] = response
        self._next_result_rank[(query_id, query_ordinal)] = result_rank + 1
        return response

    def _plan_for(self, query_id: str, query_ordinal: int) -> PlannedQuery:
        for item in self._plan:
            if (item.query_id, item.query_ordinal) == (query_id, query_ordinal):
                return item
        raise BrokerIntegrityStop("query is absent from sealed plan")

    def lookup_sirene(self, query_id: str, query_ordinal: int, siret: str) -> SireneLookupResponse:
        self._ensure_open("SIRENE_LOOKUP", query_id, query_ordinal, siret)
        if not _siret(siret) or (query_id, siret) not in self._qualified:
            self._fail("SIRENE lookup is not backed by a qualified discovered SIRET")
        key = (query_id, siret)
        if key in self._looked_up:
            self._fail("SIRENE lookup retry is forbidden")
        self._plan_for(query_id, query_ordinal)
        intent = self._journal.intent(
            phase="IDENTITY_DISCOVERY", operation="SIRENE_LOOKUP", target_kind="SIRET",
            target_canonical=siret, query_id=query_id, query_ordinal=query_ordinal,
        )
        served_from_cache = siret in self._sirene_cache
        if served_from_cache:
            records, record_hash = self._sirene_cache[siret]
        else:
            self._audit_hook("SIRENE_LOOKUP", siret)
            records = self._transport.records_for(siret)
            record_hash = _sha256(runtime.canonical_json([row.seal_projection() for row in records]))
            self._sirene_cache[siret] = (records, record_hash)
        record = records[0] if len(records) == 1 else None
        response = SireneLookupResponse(
            BROKER_SCHEMA, "SIRENE_LOOKUP_RESPONSE", query_id, siret,
            len(records) == 1, record, self._transport.sirene_snapshot_sha256,
            record_hash, served_from_cache,
        )
        self._journal.result(intent, outcome="SUCCESS")
        self._looked_up.add(key)
        return response

    def revoke(self) -> None:
        if self.state != "IDENTITY_NETWORK_OPEN":
            raise BrokerIntegrityStop("broker revocation is irreversible")
        if self._next_search != 90:
            raise BrokerIntegrityStop("cannot revoke before all 90 searches are attempted")
        expected_decisions = {
            (response.query_id, response.query_ordinal, result.result_rank)
            for response in self._responses.values()
            for result in response.results
        }
        if set(self._page_responses) != expected_decisions:
            raise BrokerIntegrityStop("cannot revoke before every top-five result has a page decision")
        if self._qualified != self._looked_up:
            raise BrokerIntegrityStop("cannot revoke before every qualified SIRET lookup")
        self._transport.assert_fully_consumed()
        self._gate.revoke()
