#!/usr/bin/env python3
"""Pure text/plain M3b primitives and post-seal candidate comparison.

Broker DTOs carry archives and byte spans, never a business qualification.
This module recomputes identity facts from those immutable inputs.  It has no
filesystem, network, child-process or dataframe dependency.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import ipaddress
import idna
import json
import re
import unicodedata
from typing import Any, Final, Mapping, Sequence
from urllib.parse import urlsplit


IDENTITY_BARRIER: Final = "IDENTITY_SEALED_NETWORK_REVOKED"
EXTRACTOR_RULE: Final = "ASCII_DIGIT_WITH_OPTIONAL_SPACE_DOT_HYPHEN_LUHN_V1"
FACT_RULE: Final = "FACT_RECONSTRUCTION_DIRECT_TRIPLE_V1"
EVIDENCE_RULE: Final = "TWO_FAMILIES_INCLUDING_SIRENE_V1"
DECISION_RULE: Final = "CONDITIONAL_SUPPORT_TOP1_COMPARISON_V1"
WEB_GROUPS: Final = frozenset({"PUBLIC_ADMINISTRATION", "ENTITY_OFFICIAL_SITE", "OFFICIAL_SECTOR_DIRECTORY", "DATED_PUBLIC_DOCUMENT"})
SIRET_RE: Final = re.compile(r"^[0-9]{14}$")
SHA_RE: Final = re.compile(r"^[0-9a-f]{64}$")
ROAD_STOPWORDS: Final = frozenset({"rue", "avenue", "av", "boulevard", "bd", "route", "chemin", "allee", "de", "du", "la", "le", "des"})
MAX_TRIPLE_SPAN: Final = 768
MAX_COMPONENT_GAP: Final = 192
CLAIM: Final = "M3B_TEXT_PLAIN_IDNA311_CONDITIONAL_SUPPORT_NOT_LABEL"
IDNA_VERSION: Final = "3.11"
PINNED_DEPENDENCIES: Final = ("idna==3.11",)
SEARCH_DOMAIN: Final = b"SIRETO-V412-R30-SEARCH\0"
SEARCH_RESULT_DOMAIN: Final = b"SIRETO-V412-R30-SEARCH-RESULT\0"
PAGE_DOMAIN: Final = b"SIRETO-V412-R30-PAGE\0"
TEXT_DECODER_RULE: Final = "TEXT_PLAIN_PINNED_DECODER_V1"
PUBLIC_SUFFIXES: Final = ("ac", "ai", "be", "biz", "ca", "ch", "co", "co.uk", "com", "de", "edu", "es", "eu", "fr", "gov", "gov.uk", "info", "io", "it", "me", "net", "nl", "org", "org.uk", "paris", "pro", "uk")
PUBLIC_SUFFIXES_SHA256: Final = "10fe038631c2a3dd619370e368be3dbd9b6cb8daf2bd4203ced236cf6226c823"
PREOPEN_FAMILIES: Final = frozenset({"PUBLIC_ADMINISTRATION", "ENTITY_OFFICIAL_SITE_CANDIDATE", "OFFICIAL_SECTOR_DIRECTORY", "DATED_PUBLIC_DOCUMENT_CANDIDATE", "INADMISSIBLE"})
INADMISSIBLE_REASONS: Final = frozenset({"UNSAFE_URL", "SEARCH_ENGINE", "COMMERCIAL_AGGREGATOR", "SIRENE_COPY", "AUTH_REQUIRED", "PAID_OR_VARIABLE_COST", "UNRECOGNIZED_PUBLIC_SUFFIX", "DOMAIN_NOT_ALLOWLISTED", "NONE"})


class BusinessIntegrityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CollectionQuery:
    query_id: str
    selection_ordinal: int
    query_ordinal: int
    search_query: str
    crm_name: str
    crm_address: str


@dataclass(frozen=True, slots=True)
class SearchAttempt:
    query_id: str
    query_ordinal: int
    search_attempt_id: str
    search_query: str
    status: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    query_id: str
    query_ordinal: int
    search_attempt_id: str
    result_rank: int
    title: str
    snippet: str
    resolved_url: str
    result_payload_sha256: str
    preopen_family: str
    inadmissible_reason: str
    normalized_hostname: str | None
    registrable_domain: str | None


@dataclass(frozen=True, slots=True)
class PageDecision:
    query_id: str
    query_ordinal: int
    result_rank: int
    decision: str
    normalized_domain: str | None
    query_open_slot: int | None
    dossier_open_ordinal: int | None
    page_attempt_id: str | None


@dataclass(frozen=True, slots=True)
class DnsIdentityRow:
    dns_attempt_id: str
    parent_attempt_id: str
    request_kind: str
    query_id: str
    query_ordinal: int
    result_rank: int | None
    status: str
    addresses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PageArchive:
    page_attempt_id: str
    dns_attempt_id: str
    query_id: str
    query_ordinal: int
    result_rank: int
    query_open_slot: int
    dossier_open_ordinal: int
    search_attempt_id: str
    result_payload_sha256: str
    requested_url: str
    independence_group: str
    raw_content_sha256: str
    extracted_text_sha256: str
    raw_content_b64: str
    extracted_text: str
    mime_type: str
    charset: str
    text_decoder: str
    decoder_rule_id: str
    crm_name: str
    crm_address: str


@dataclass(frozen=True, slots=True)
class IdentityOccurrence:
    occurrence_id: str
    page_attempt_id: str
    query_id: str
    siret: str
    siret_span: tuple[int, int]
    name_span: tuple[int, int]
    address_span: tuple[int, int]
    relation_span: tuple[int, int]
    source_excerpt_sha256: str
    extractor_rule_id: str


@dataclass(frozen=True, slots=True)
class OccurrenceProvenance:
    query_id: str
    proof_id: str
    related_siret: str
    related_siren: str
    raw_content_sha256: str
    extracted_text_sha256: str
    source_excerpt_sha256: str
    siret_span: tuple[int, int]
    name_span: tuple[int, int]
    address_span: tuple[int, int]
    relation_span: tuple[int, int]
    extractor_rule_id: str


@dataclass(frozen=True, slots=True)
class FactProvenance:
    query_id: str
    proof_id: str
    fact_type: str
    fact_value_normalized: str
    related_siret: str
    related_siren: str
    source_kind: str
    source_content_sha256: str
    source_excerpt_sha256: str
    byte_start: int | None
    byte_end: int | None
    page_attempt_id: str | None
    lookup_siret: str | None
    snapshot_sha256: str | None
    reconstruction_rule_id: str = FACT_RULE


@dataclass(frozen=True, slots=True)
class LookupPlanRow:
    siret: str
    lookup_ordinal: int
    query_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SireneRecord:
    siret: str
    lookup_ordinal: int
    found_count: int
    state: str | None
    names: tuple[str, ...]
    address: str | None
    snapshot_ref: str
    snapshot_sha256: str
    record_payload_sha256: str


@dataclass(frozen=True, slots=True)
class Fact:
    query_id: str
    proof_id: str
    independence_group: str
    related_siret: str
    related_siren: str
    fact_type: str
    fact_value_normalized: str
    site_specific: bool
    source_excerpt_sha256: str
    source_kind: str
    source_content_sha256: str
    byte_start: int | None
    byte_end: int | None
    reconstruction_rule_id: str = FACT_RULE
    provenance_verified: bool = False

    def __post_init__(self) -> None:
        if self.provenance_verified is not False:
            raise BusinessIntegrityError("broker/store provenance cannot be verified in this core")


@dataclass(frozen=True, slots=True)
class Evidence:
    query_id: str
    related_siret: str
    independence_group: str
    proof_id: str
    evidence_ref_id: str
    archive_sha256: str
    has_siret_identifier: bool
    name_rule_pass: bool
    address_rule_pass: bool
    site_specific_pass: bool
    within_group_contradiction: bool
    conditional_group_supports: bool
    evidence_rule_id: str = EVIDENCE_RULE
    provenance_verified: bool = False

    def __post_init__(self) -> None:
        if self.provenance_verified is not False:
            raise BusinessIntegrityError("broker/store provenance cannot be verified in this core")


@dataclass(frozen=True, slots=True)
class IdentitySeal:
    claim: str
    barrier: str
    query_ids: tuple[str, ...]
    lookup_plan: tuple[LookupPlanRow, ...]
    sirene_records: tuple[SireneRecord, ...]
    facts: tuple[Fact, ...]
    provenance: tuple[FactProvenance, ...]
    evidence: tuple[Evidence, ...]
    conditional_support_by_query: tuple[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]], ...]
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class CandidateComparison:
    query_id: str
    rank: int
    candidate_siret: str
    is_top1: bool
    is_conditionally_supported: bool
    conditional_support_groups: tuple[str, ...]
    evidence_contradiction: bool
    supported_same_siren_elsewhere: bool
    supported_cross_siren_elsewhere: bool
    sirene_unique: bool
    sirene_active: bool


@dataclass(frozen=True, slots=True)
class ConditionalSupportSummary:
    query_id: str
    conditional_outcome: str
    reliable: bool
    top1_siret: str
    conditionally_supported_sirets: tuple[str, ...]
    independent_group_count: int
    evidence_ref_ids: tuple[str, ...]
    conditional_alternative_siret: str | None
    conditional_alternative_in_top100: bool
    collision_kind: str | None
    decision_rule_id: str = DECISION_RULE

    def __post_init__(self) -> None:
        if self.reliable is not False:
            raise BusinessIntegrityError("conditional support cannot be promoted to a reliable label")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def search_attempt_id(query_id: str, query_ordinal: int, search_query: str) -> str:
    return hashlib.sha256(SEARCH_DOMAIN + canonical_bytes([query_id, query_ordinal, search_query])).hexdigest()


def search_result_id(query_id: str, query_ordinal: int, rank: int, title: str, snippet: str, resolved_url: str) -> str:
    return hashlib.sha256(SEARCH_RESULT_DOMAIN + canonical_bytes([query_id, query_ordinal, rank, title, snippet, resolved_url])).hexdigest()


def page_attempt_id(query_id: str, query_ordinal: int, rank: int, resolved_url: str, query_open_slot: int, dossier_open_ordinal: int) -> str:
    return hashlib.sha256(PAGE_DOMAIN + canonical_bytes([query_id, query_ordinal, rank, resolved_url, query_open_slot, dossier_open_ordinal])).hexdigest()


def _assert_dependencies() -> None:
    if getattr(idna, "__version__", None) != IDNA_VERSION:
        raise BusinessIntegrityError("pinned idna version mismatch")


def normalize_hostname(hostname: str) -> str:
    _assert_dependencies()
    if type(hostname) is not str or not hostname or "%" in hostname:
        raise BusinessIntegrityError("hostname is not an admissible IDNA input")
    value = unicodedata.normalize("NFC", hostname)
    if value.endswith("."):
        value = value[:-1]
    try:
        normalized = idna.encode(value, uts46=True, std3_rules=True).decode("ascii").lower()
    except (idna.IDNAError, UnicodeError) as exc:
        raise BusinessIntegrityError("hostname IDNA normalization failed") from exc
    if not normalized or any(not label or len(label.encode("ascii")) > 63 for label in normalized.split(".")) or len(normalized.encode("ascii")) > 253:
        raise BusinessIntegrityError("hostname label/total length is invalid")
    return normalized


def _registrable_domain(host: str) -> str | None:
    matching = [suffix for suffix in PUBLIC_SUFFIXES if host == suffix or host.endswith("." + suffix)]
    if not matching:
        return None
    suffix = max(matching, key=lambda value: len(value.split(".")))
    host_labels, suffix_labels = host.split("."), suffix.split(".")
    if len(host_labels) <= len(suffix_labels):
        return None
    return ".".join(host_labels[-len(suffix_labels) - 1:])


def evaluate_domain_hostname(input_hostname: str) -> dict[str, Any]:
    _assert_dependencies()
    try:
        host = normalize_hostname(input_hostname)
    except BusinessIntegrityError:
        return {"normalized_hostname": None, "registrable_domain": None, "matches_public_administration": False, "matches_sirene_copy": False, "safe": False}
    domain = _registrable_domain(host)
    try:
        ipaddress.ip_address(host)
    except ValueError:
        is_ip = False
    else:
        is_ip = True
    suffix_match = lambda suffix: host == suffix or host.endswith("." + suffix)
    return {
        "normalized_hostname": host,
        "registrable_domain": domain,
        "matches_public_administration": suffix_match("gouv.fr"),
        "matches_sirene_copy": suffix_match("annuaire-entreprises.data.gouv.fr") or suffix_match("sirene.fr"),
        "safe": bool(domain) and not is_ip,
    }


def _url_identity(url: str) -> tuple[bool, str, str | None, str | None]:
    _assert_dependencies()
    suffix_bytes = ("\n".join(PUBLIC_SUFFIXES) + "\n").encode("ascii")
    if len(PUBLIC_SUFFIXES) != 27 or hashlib.sha256(suffix_bytes).hexdigest() != PUBLIC_SUFFIXES_SHA256:
        raise BusinessIntegrityError("pinned public suffix tuple drifted")
    if type(url) is not str or not url or url != url.strip() or "\\" in url or any(unicodedata.category(char) in {"Cc", "Cf"} for char in url):
        return False, "UNSAFE_URL", None, None
    try:
        parsed = urlsplit(url)
        port = parsed.port
        host_input = parsed.hostname
        if parsed.scheme != "https" or not parsed.netloc or "%" in parsed.netloc or host_input is None or parsed.username is not None or parsed.password is not None or parsed.fragment or port not in {None, 443}:
            return False, "UNSAFE_URL", None, None
        host = normalize_hostname(host_input)
        raw_authority_host = parsed.netloc[:-4] if parsed.netloc.endswith(":443") else parsed.netloc
        if normalize_hostname(raw_authority_host) != host:
            return False, "UNSAFE_URL", None, None
        if host == "localhost" or host.endswith(".local"):
            return False, "UNSAFE_URL", host, None
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return False, "UNSAFE_URL", host, None
    except (BusinessIntegrityError, UnicodeError, ValueError):
        return False, "UNSAFE_URL", None, None
    domain = _registrable_domain(host)
    if domain is None:
        return True, "UNRECOGNIZED_PUBLIC_SUFFIX", host, None
    return True, "NONE", host, domain


def derive_page_decisions(search_results: Sequence[SearchResult]) -> tuple[PageDecision, ...]:
    """Apply deterministic domain de-duplication and 2/query, 6/dossier quotas."""

    keys = [(r.query_id, r.query_ordinal, r.result_rank) for r in search_results]
    if len(keys) != len(set(keys)):
        raise BusinessIntegrityError("duplicate SEARCH result primary key")
    opened_domains: dict[str, set[str]] = {}
    query_counts: dict[tuple[str, int], int] = {}
    dossier_counts: dict[str, int] = {}
    output: list[PageDecision] = []
    for result in sorted(search_results, key=lambda r: (r.query_id, r.query_ordinal, r.result_rank)):
        safe, url_reason, hostname, domain = _url_identity(result.resolved_url)
        if result.preopen_family not in PREOPEN_FAMILIES or result.inadmissible_reason not in INADMISSIBLE_REASONS or (result.normalized_hostname, result.registrable_domain) != (hostname, domain):
            raise BusinessIntegrityError("SEARCH pre-open classification mismatch")
        query_key = (result.query_id, result.query_ordinal)
        if not safe or domain is None:
            if result.preopen_family != "INADMISSIBLE" or result.inadmissible_reason != url_reason:
                raise BusinessIntegrityError("unsafe URL was not classified inadmissible")
            decision, query_slot, dossier_slot, attempt_id = "SKIP_INADMISSIBLE", None, None, None
        elif result.preopen_family == "INADMISSIBLE":
            if result.inadmissible_reason == "NONE":
                raise BusinessIntegrityError("inadmissible pre-open result lacks reason")
            decision, query_slot, dossier_slot, attempt_id = "SKIP_INADMISSIBLE", None, None, None
        elif result.inadmissible_reason != "NONE":
            raise BusinessIntegrityError("admissible pre-open result carries rejection reason")
        elif domain in opened_domains.setdefault(result.query_id, set()):
            decision, query_slot, dossier_slot, attempt_id = "SKIP_DUPLICATE_DOMAIN", None, None, None
        elif query_counts.get(query_key, 0) >= 2:
            decision, query_slot, dossier_slot, attempt_id = "SKIP_QUERY_QUOTA", None, None, None
        elif dossier_counts.get(result.query_id, 0) >= 6:
            decision, query_slot, dossier_slot, attempt_id = "SKIP_DOSSIER_QUOTA", None, None, None
        else:
            query_slot = query_counts.get(query_key, 0) + 1
            dossier_slot = dossier_counts.get(result.query_id, 0) + 1
            attempt_id = page_attempt_id(result.query_id, result.query_ordinal, result.result_rank, result.resolved_url, query_slot, dossier_slot)
            decision = "OPEN_ATTEMPT"
            query_counts[query_key] = query_slot
            dossier_counts[result.query_id] = dossier_slot
            opened_domains[result.query_id].add(domain)
        output.append(PageDecision(result.query_id, result.query_ordinal, result.result_rank, decision, domain, query_slot, dossier_slot, attempt_id))
    return tuple(output)


def normalize_value(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if unicodedata.category(c) != "Mn").casefold()
    return " ".join(re.sub(r"[^\w]+", " ", value, flags=re.UNICODE).split())


def _closed(raw: Mapping[str, Any], fields: set[str], label: str) -> None:
    if type(raw) is not dict or set(raw) != fields:
        raise BusinessIntegrityError(f"{label} closed schema mismatch")


def _text(value: Any, label: str, empty: bool = False) -> str:
    if type(value) is not str or (not empty and not value):
        raise BusinessIntegrityError(f"{label} must be a string")
    return value


def _int(value: Any, label: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise BusinessIntegrityError(f"{label} outside range")
    return value


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or SHA_RE.fullmatch(value) is None:
        raise BusinessIntegrityError(f"{label} must be SHA-256")
    return value


def _siret(value: Any, *, luhn: bool = True) -> str:
    if type(value) is not str or SIRET_RE.fullmatch(value) is None:
        raise BusinessIntegrityError("invalid SIRET syntax")
    if luhn and not _luhn(value):
        raise BusinessIntegrityError("invalid SIRET Luhn checksum")
    return value


def _luhn(value: str) -> bool:
    total = 0
    parity = len(value) % 2
    for index, char in enumerate(value):
        digit = ord(char) - 48
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _span(raw: Any, label: str, size: int) -> tuple[int, int]:
    if type(raw) is not list or len(raw) != 2:
        raise BusinessIntegrityError(f"{label} span schema mismatch")
    start, end = raw
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= size:
        raise BusinessIntegrityError(f"{label} span outside archived text")
    return start, end


def _slice(encoded: bytes, span: tuple[int, int], label: str) -> str:
    try:
        return encoded[span[0]:span[1]].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BusinessIntegrityError(f"{label} span splits UTF-8") from exc


def validate_collection_plan(rows: Sequence[Mapping[str, Any]]) -> tuple[CollectionQuery, ...]:
    if len(rows) != 90:
        raise BusinessIntegrityError("plan must contain exactly 30x3 rows")
    output, seen, grouped = [], set(), {}
    for raw in rows:
        _closed(raw, {"query_id", "selection_ordinal", "query_ordinal", "search_query", "crm_name", "crm_address"}, "plan")
        row = CollectionQuery(_text(raw["query_id"], "query_id"), _int(raw["selection_ordinal"], "selection_ordinal", 1, 30), _int(raw["query_ordinal"], "query_ordinal", 1, 3), _text(raw["search_query"], "search_query"), _text(raw["crm_name"], "crm_name"), _text(raw["crm_address"], "crm_address"))
        key = row.query_id, row.query_ordinal
        if key in seen:
            raise BusinessIntegrityError("duplicate plan primary key")
        seen.add(key); grouped.setdefault(row.query_id, []).append(row); output.append(row)
    if len(grouped) != 30 or {next(iter({r.selection_ordinal for r in rows_})) for rows_ in grouped.values()} != set(range(1, 31)):
        raise BusinessIntegrityError("plan dossier/selection mismatch")
    if any(sorted(r.query_ordinal for r in rows_) != [1, 2, 3] or len({r.selection_ordinal for r in rows_}) != 1 or len({(r.crm_name, r.crm_address) for r in rows_}) != 1 for rows_ in grouped.values()):
        raise BusinessIntegrityError("plan query ordinals mismatch")
    return tuple(sorted(output, key=lambda r: (r.selection_ordinal, r.query_ordinal)))


def convert_search_responses(plan: Sequence[CollectionQuery], responses: Sequence[Mapping[str, Any]]) -> tuple[tuple[SearchAttempt, ...], tuple[SearchResult, ...]]:
    expected = {(r.query_id, r.query_ordinal) for r in plan}
    if len(responses) != len(expected):
        raise BusinessIntegrityError("one SEARCH response required per plan row")
    attempts_out, results_out, seen_keys = [], [], set()
    plan_by_key = {(r.query_id, r.query_ordinal): r for r in plan}
    for raw in responses:
        _closed(raw, {"query_id", "query_ordinal", "search_attempt_id", "status", "results"}, "SEARCH")
        key = _text(raw["query_id"], "query_id"), _int(raw["query_ordinal"], "query_ordinal", 1, 3)
        attempt = _sha(raw["search_attempt_id"], "search_attempt_id")
        planned = plan_by_key.get(key)
        if planned is None or key in seen_keys or attempt != search_attempt_id(key[0], key[1], planned.search_query):
            raise BusinessIntegrityError("SEARCH PK/FK mismatch")
        seen_keys.add(key)
        status, results = raw["status"], raw["results"]
        if status not in {"SUCCESS", "ERROR"} or type(results) is not list or len(results) > 5 or (status == "ERROR" and results):
            raise BusinessIntegrityError("SEARCH response invalid")
        attempts_out.append(SearchAttempt(key[0], key[1], attempt, planned.search_query, status))
        for position, result in enumerate(results, 1):
            _closed(result, {"rank", "title", "snippet", "resolved_url", "result_payload_sha256", "preopen_family", "inadmissible_reason", "normalized_hostname", "registrable_domain"}, "SEARCH result")
            if _int(result["rank"], "rank", 1, 5) != position:
                raise BusinessIntegrityError("SEARCH ranks not contiguous")
            title, snippet, url = _text(result["title"], "title", True), _text(result["snippet"], "snippet", True), _text(result["resolved_url"], "resolved_url")
            payload_hash = _sha(result["result_payload_sha256"], "result_payload_sha256")
            if payload_hash != search_result_id(key[0], key[1], position, title, snippet, url):
                raise BusinessIntegrityError("SEARCH result payload identity mismatch")
            family, reason = _text(result["preopen_family"], "preopen_family"), _text(result["inadmissible_reason"], "inadmissible_reason")
            safe, url_reason, hostname, domain = _url_identity(url)
            supplied_host, supplied_domain = result["normalized_hostname"], result["registrable_domain"]
            if supplied_host is not None and type(supplied_host) is not str or supplied_domain is not None and type(supplied_domain) is not str:
                raise BusinessIntegrityError("SEARCH domain fields must be strings or null")
            if family not in PREOPEN_FAMILIES or reason not in INADMISSIBLE_REASONS or (supplied_host, supplied_domain) != (hostname, domain) or (not safe or domain is None) and (family, reason) != ("INADMISSIBLE", url_reason):
                raise BusinessIntegrityError("SEARCH pre-open URL classification mismatch")
            results_out.append(SearchResult(key[0], key[1], attempt, position, title, snippet, url, payload_hash, family, reason, supplied_host, supplied_domain))
    if seen_keys != expected:
        raise BusinessIntegrityError("SEARCH attempts are not bijective with plan")
    return tuple(sorted(attempts_out, key=lambda r: (r.query_id, r.query_ordinal))), tuple(sorted(results_out, key=lambda r: (r.query_id, r.query_ordinal, r.result_rank)))


def convert_dns_responses(plan: Sequence[CollectionQuery], responses: Sequence[Mapping[str, Any]]) -> tuple[DnsIdentityRow, ...]:
    planned = {(r.query_id, r.query_ordinal) for r in plan}
    output, seen = [], set()
    for raw in responses:
        _closed(raw, {"dns_attempt_id", "parent_attempt_id", "request_kind", "query_id", "query_ordinal", "result_rank", "status", "addresses"}, "DNS")
        dns_id, parent = _sha(raw["dns_attempt_id"], "dns_attempt_id"), _sha(raw["parent_attempt_id"], "parent_attempt_id")
        key = _text(raw["query_id"], "query_id"), _int(raw["query_ordinal"], "query_ordinal", 1, 3)
        kind, rank, addresses = raw["request_kind"], raw["result_rank"], raw["addresses"]
        if dns_id in seen or key not in planned or kind not in {"SEARCH", "PAGE"}:
            raise BusinessIntegrityError("DNS PK/FK mismatch")
        seen.add(dns_id)
        if kind == "SEARCH" and rank is not None or kind == "PAGE" and (type(rank) is not int or not 1 <= rank <= 5):
            raise BusinessIntegrityError("DNS rank/kind mismatch")
        if raw["status"] not in {"SUCCESS", "ERROR"} or type(addresses) is not list or any(type(v) is not str or not v for v in addresses) or addresses != sorted(set(addresses)) or (raw["status"] == "ERROR" and addresses):
            raise BusinessIntegrityError("DNS response invalid")
        output.append(DnsIdentityRow(dns_id, parent, kind, key[0], key[1], rank, raw["status"], tuple(addresses)))
    return tuple(sorted(output, key=lambda r: r.dns_attempt_id))


def convert_page_responses(plan: Sequence[CollectionQuery], search_attempts: Sequence[SearchAttempt], search_results: Sequence[SearchResult], page_decisions: Sequence[PageDecision], dns: Sequence[DnsIdentityRow], responses: Sequence[Mapping[str, Any]]) -> tuple[tuple[PageArchive, ...], tuple[IdentityOccurrence, ...], tuple[OccurrenceProvenance, ...]]:
    attempt_by_key = {(r.query_id, r.query_ordinal): r for r in search_attempts}
    result_by_key = {(r.query_id, r.query_ordinal, r.result_rank): r for r in search_results}
    expected_decisions = derive_page_decisions(search_results)
    if tuple(page_decisions) != expected_decisions:
        raise BusinessIntegrityError("PAGE decisions do not replay exactly")
    open_by_key = {(r.query_id, r.query_ordinal, r.result_rank): r for r in page_decisions if r.decision == "OPEN_ATTEMPT"}
    if len(responses) != len(open_by_key):
        raise BusinessIntegrityError("PAGE responses are not bijective with OPEN_ATTEMPT")
    frozen_crm = {r.query_id: (r.crm_name, r.crm_address) for r in plan}
    dns_by_id = {r.dns_attempt_id: r for r in dns}
    archives, occurrences, provenance, page_ids, page_keys, open_slots, dossier_slots, crm_by_query = [], [], [], set(), set(), set(), set(), {}
    for raw in responses:
        _closed(raw, {"page_attempt_id", "dns_attempt_id", "query_id", "query_ordinal", "result_rank", "query_open_slot", "dossier_open_ordinal", "search_attempt_id", "result_payload_sha256", "requested_url", "status", "independence_group", "raw_content_b64", "raw_content_sha256", "mime_type", "charset", "text_decoder", "decoder_rule_id", "crm", "occurrences"}, "PAGE")
        page_id, dns_id = _sha(raw["page_attempt_id"], "page_attempt_id"), _sha(raw["dns_attempt_id"], "dns_attempt_id")
        query_id, qo, rank = _text(raw["query_id"], "query_id"), _int(raw["query_ordinal"], "query_ordinal", 1, 3), _int(raw["result_rank"], "result_rank", 1, 5)
        query_slot, dossier_slot = _int(raw["query_open_slot"], "query_open_slot", 1, 2), _int(raw["dossier_open_ordinal"], "dossier_open_ordinal", 1, 6)
        attempt, result, decision = attempt_by_key.get((query_id, qo)), result_by_key.get((query_id, qo, rank)), open_by_key.get((query_id, qo, rank))
        if page_id in page_ids or (query_id, qo, rank) in page_keys or (query_id, qo, query_slot) in open_slots or (query_id, dossier_slot) in dossier_slots or attempt is None or result is None or decision is None:
            raise BusinessIntegrityError("orphan or duplicate PAGE")
        claimed_attempt, claimed_result, requested_url = _sha(raw["search_attempt_id"], "search_attempt_id"), _sha(raw["result_payload_sha256"], "result_payload_sha256"), _text(raw["requested_url"], "requested_url")
        if claimed_attempt != attempt.search_attempt_id or claimed_result != result.result_payload_sha256 or requested_url != result.resolved_url or (query_slot, dossier_slot, page_id) != (decision.query_open_slot, decision.dossier_open_ordinal, decision.page_attempt_id):
            raise BusinessIntegrityError("PAGE attempt/URL/result identity mismatch")
        page_ids.add(page_id); page_keys.add((query_id, qo, rank)); open_slots.add((query_id, qo, query_slot)); dossier_slots.add((query_id, dossier_slot))
        dns_row = dns_by_id.get(dns_id)
        if dns_row is None or dns_row.parent_attempt_id != page_id or dns_row.request_kind != "PAGE" or (dns_row.query_id, dns_row.query_ordinal, dns_row.result_rank) != (query_id, qo, rank) or dns_row.status != "SUCCESS":
            raise BusinessIntegrityError("PAGE to DNS FK mismatch")
        if raw["status"] not in {"SUCCESS", "ERROR"} or raw["independence_group"] not in WEB_GROUPS or type(raw["occurrences"]) is not list or (raw["status"] == "ERROR" and raw["occurrences"]):
            raise BusinessIntegrityError("PAGE status/group mismatch")
        _closed(raw["crm"], {"name", "address"}, "frozen CRM")
        crm = (_text(raw["crm"]["name"], "CRM name"), _text(raw["crm"]["address"], "CRM address"))
        if frozen_crm.get(query_id) != crm:
            raise BusinessIntegrityError("PAGE CRM differs from frozen plan")
        if query_id in crm_by_query and crm_by_query[query_id] != crm:
            raise BusinessIntegrityError("frozen CRM changed within dossier")
        crm_by_query[query_id] = crm
        try:
            raw_bytes = base64.b64decode(_text(raw["raw_content_b64"], "raw_content_b64"), validate=True)
        except Exception as exc:
            raise BusinessIntegrityError("raw archive is not strict base64") from exc
        raw_hash = _sha(raw["raw_content_sha256"], "raw_content_sha256")
        if hashlib.sha256(raw_bytes).hexdigest() != raw_hash:
            raise BusinessIntegrityError("PAGE archive hash mismatch")
        mime, charset, decoder, rule = raw["mime_type"], raw["charset"], raw["text_decoder"], raw["decoder_rule_id"]
        if mime != "text/plain" or rule != TEXT_DECODER_RULE or (decoder, charset) not in {("UTF8_STRICT", "UTF-8"), ("UTF8_REPLACE", "UTF-8"), ("ISO_8859_1", "ISO-8859-1")}:
            raise BusinessIntegrityError("unsupported PAGE text decoder contract")
        try:
            if decoder == "UTF8_STRICT":
                text = raw_bytes.decode("utf-8", errors="strict")
            elif decoder == "UTF8_REPLACE":
                text = raw_bytes.decode("utf-8", errors="replace")
            else:
                text = raw_bytes.decode("iso-8859-1", errors="strict")
        except UnicodeDecodeError as exc:
            raise BusinessIntegrityError("PAGE decoder rejected raw bytes") from exc
        encoded = text.encode("utf-8")
        text_hash = hashlib.sha256(encoded).hexdigest()
        archive = PageArchive(page_id, dns_id, query_id, qo, rank, query_slot, dossier_slot, claimed_attempt, claimed_result, requested_url, raw["independence_group"], raw_hash, text_hash, raw["raw_content_b64"], text, mime, charset, decoder, rule, crm[0], crm[1])
        archives.append(archive)
        for item in raw["occurrences"]:
            _closed(item, {"occurrence_id", "siret_span", "name_span", "address_span", "relation_span", "source_excerpt_sha256", "extractor_rule_id"}, "occurrence")
            occurrence_id = _sha(item["occurrence_id"], "occurrence_id")
            spans = tuple(_span(item[name], name, len(encoded)) for name in ("siret_span", "name_span", "address_span", "relation_span"))
            siret_span, name_span, address_span, relation_span = spans
            if item["extractor_rule_id"] != EXTRACTOR_RULE:
                raise BusinessIntegrityError("occurrence extractor rule mismatch")
            relation = _slice(encoded, relation_span, "relation")
            if _sha(item["source_excerpt_sha256"], "source_excerpt_sha256") != hashlib.sha256(relation.encode("utf-8")).hexdigest():
                raise BusinessIntegrityError("occurrence excerpt hash mismatch")
            if not all(relation_span[0] <= span[0] < span[1] <= relation_span[1] for span in (siret_span, name_span, address_span)) or relation_span[1] - relation_span[0] > MAX_TRIPLE_SPAN:
                raise BusinessIntegrityError("identity triple is not locally bounded")
            ordered = sorted((siret_span, name_span, address_span))
            if max((ordered[i + 1][0] - ordered[i][1] for i in range(2)), default=0) > MAX_COMPONENT_GAP:
                raise BusinessIntegrityError("SIRET is too distant from name/address")
            digits = re.sub(r"[ .-]", "", _slice(encoded, siret_span, "SIRET"))
            siret = _siret(digits)
            name, address = _slice(encoded, name_span, "name"), _slice(encoded, address_span, "address")
            if normalize_value(name) != normalize_value(crm[0]) or not _address_matches(address, crm[1]):
                raise BusinessIntegrityError("archived name/address do not reproduce frozen CRM")
            occurrence = IdentityOccurrence(occurrence_id, page_id, query_id, siret, siret_span, name_span, address_span, relation_span, item["source_excerpt_sha256"], EXTRACTOR_RULE)
            occurrences.append(occurrence)
            provenance.append(OccurrenceProvenance(query_id, page_id, siret, siret[:9], raw_hash, text_hash, item["source_excerpt_sha256"], siret_span, name_span, address_span, relation_span, EXTRACTOR_RULE))
    occ_keys = [(r.query_id, r.occurrence_id) for r in occurrences]
    if len(occ_keys) != len(set(occ_keys)):
        raise BusinessIntegrityError("duplicate occurrence primary key")
    return tuple(sorted(archives, key=lambda r: r.page_attempt_id)), tuple(sorted(occurrences, key=lambda r: (r.query_id, r.occurrence_id))), tuple(sorted(provenance, key=lambda r: (r.query_id, r.proof_id, r.related_siret)))


def derive_lookup_plan(occurrences: Sequence[IdentityOccurrence]) -> tuple[LookupPlanRow, ...]:
    queries: dict[str, set[str]] = {}
    for row in occurrences:
        queries.setdefault(row.siret, set()).add(row.query_id)
    return tuple(LookupPlanRow(siret, ordinal, tuple(sorted(queries[siret]))) for ordinal, siret in enumerate(sorted(queries), 1))


def convert_lookup_responses(plan: Sequence[LookupPlanRow], responses: Sequence[Mapping[str, Any]], snapshot_ref: str, snapshot_sha256: str) -> tuple[SireneRecord, ...]:
    expected = {r.siret: r for r in plan}
    if len(responses) != len(expected):
        raise BusinessIntegrityError("one global lookup response required per SIRET")
    output, seen = [], set()
    for raw in responses:
        _closed(raw, {"siret", "lookup_ordinal", "snapshot_ref", "snapshot_sha256", "records", "record_payload_sha256"}, "SIRENE lookup")
        siret = _siret(raw["siret"])
        if siret not in expected or siret in seen or raw["lookup_ordinal"] != expected[siret].lookup_ordinal:
            raise BusinessIntegrityError("SIRENE global cache/ordinal mismatch")
        seen.add(siret)
        if raw["snapshot_ref"] != snapshot_ref or raw["snapshot_sha256"] != snapshot_sha256:
            raise BusinessIntegrityError("SIRENE snapshot pin mismatch")
        records = raw["records"]
        if type(records) is not list:
            raise BusinessIntegrityError("SIRENE records must be a list")
        payload_hash = _sha(raw["record_payload_sha256"], "record_payload_sha256")
        if hashlib.sha256(canonical_bytes(records)).hexdigest() != payload_hash:
            raise BusinessIntegrityError("SIRENE record payload hash mismatch")
        state = address = None; names: tuple[str, ...] = ()
        for record in records:
            _closed(record, {"siret", "state", "names", "address"}, "SIRENE record")
            if _siret(record["siret"]) != siret or record["state"] not in {"A", "F"} or type(record["names"]) is not list or not record["names"] or record["names"] != sorted(set(record["names"])) or any(type(v) is not str or not v for v in record["names"]) or type(record["address"]) is not str or not record["address"]:
                raise BusinessIntegrityError("SIRENE record schema/value mismatch")
        if len(records) == 1:
            state, names, address = records[0]["state"], tuple(records[0]["names"]), records[0]["address"]
        output.append(SireneRecord(siret, expected[siret].lookup_ordinal, len(records), state, names, address, snapshot_ref, snapshot_sha256, payload_hash))
    return tuple(sorted(output, key=lambda r: r.lookup_ordinal))


def _address_matches(left: str, right: str) -> bool:
    a, b = normalize_value(left).split(), normalize_value(right).split()
    cp_a = [x for x in a if x.isascii() and x.isdigit() and len(x) == 5]; cp_b = [x for x in b if x.isascii() and x.isdigit() and len(x) == 5]
    no_a = [x for x in a if x.isascii() and x.isdigit() and len(x) <= 4 and x not in cp_a]; no_b = [x for x in b if x.isascii() and x.isdigit() and len(x) <= 4 and x not in cp_b]
    roads_a = {x for x in a if not x.isdigit() and x not in ROAD_STOPWORDS and len(x) >= 3}; roads_b = {x for x in b if not x.isdigit() and x not in ROAD_STOPWORDS and len(x) >= 3}
    return bool(cp_a and cp_a == cp_b and no_a[:1] == no_b[:1] and roads_a & roads_b)


def _eref(query_id: str, siret: str, group: str, proof: str) -> str:
    return hashlib.sha256(b"SIRETO-V412-R30-EVIDENCE\0" + canonical_bytes([query_id, siret, group, proof])).hexdigest()


def _derive_facts(archives: Sequence[PageArchive], occurrences: Sequence[IdentityOccurrence], provenance: Sequence[OccurrenceProvenance]) -> tuple[tuple[Fact, ...], tuple[FactProvenance, ...]]:
    archive_by_id = {r.page_attempt_id: r for r in archives}; prov_by_key = {(r.query_id, r.proof_id, r.related_siret): r for r in provenance}
    output, fact_provenance = [], []
    for occ in occurrences:
        archive, prov = archive_by_id.get(occ.page_attempt_id), prov_by_key.get((occ.query_id, occ.page_attempt_id, occ.siret))
        if archive is None or prov is None or prov.source_excerpt_sha256 != occ.source_excerpt_sha256:
            raise BusinessIntegrityError("occurrence/provenance/archive FK mismatch")
        encoded = archive.extracted_text.encode("utf-8")
        raw_siret = _slice(encoded, occ.siret_span, "SIRET")
        digit_count = 0; siren_end = occ.siret_span[0]
        for byte in encoded[occ.siret_span[0]:occ.siret_span[1]]:
            siren_end += 1
            if 48 <= byte <= 57: digit_count += 1
            if digit_count == 9: break
        if digit_count != 9: raise BusinessIntegrityError("SIREN span cannot be reconstructed")
        pieces = (
            ("SIRET_IDENTIFIER", occ.siret, True, occ.siret_span), ("SIREN_IDENTIFIER", occ.siret[:9], False, (occ.siret_span[0], siren_end)),
            ("SITE_NAME", normalize_value(_slice(encoded, occ.name_span, "name")), False, occ.name_span), ("SITE_ADDRESS", normalize_value(_slice(encoded, occ.address_span, "address")), True, occ.address_span),
            ("ENTITY_SITE_RELATION", normalize_value(_slice(encoded, occ.relation_span, "relation")), True, occ.relation_span),
        )
        for kind, value, specific, span in pieces:
            excerpt_hash = hashlib.sha256(_slice(encoded, span, kind).encode()).hexdigest()
            output.append(Fact(occ.query_id, occ.page_attempt_id, archive.independence_group, occ.siret, occ.siret[:9], kind, value, specific, excerpt_hash, "PAGE_ARCHIVE", archive.raw_content_sha256, span[0], span[1]))
            fact_provenance.append(FactProvenance(occ.query_id, occ.page_attempt_id, kind, value, occ.siret, occ.siret[:9], "PAGE_ARCHIVE", archive.raw_content_sha256, excerpt_hash, span[0], span[1], occ.page_attempt_id, None, None))
    keys = [(r.query_id, r.proof_id, r.fact_type, r.fact_value_normalized, r.related_siret) for r in output]
    if len(keys) != len(set(keys)):
        raise BusinessIntegrityError("duplicate facts")
    return tuple(sorted(output, key=lambda r: (r.query_id, r.proof_id, r.related_siret, r.fact_type, r.fact_value_normalized))), tuple(sorted(fact_provenance, key=lambda r: (r.query_id, r.proof_id, r.related_siret, r.fact_type, r.fact_value_normalized)))


def _sirene_facts(lookup_plan: Sequence[LookupPlanRow], records: Sequence[SireneRecord]) -> tuple[tuple[Fact, ...], tuple[FactProvenance, ...]]:
    facts, provenance = [], []
    by_siret = {r.siret: r for r in records}
    for planned in lookup_plan:
        record = by_siret[planned.siret]
        if record.found_count != 1:
            continue
        relation_object = {"siret": record.siret, "names": list(record.names), "address": record.address}
        values = [("SIRET_IDENTIFIER", record.siret, True, record.siret)]
        values += [("SIREN_IDENTIFIER", record.siret[:9], True, record.siret[:9])]
        values += [("SITE_NAME", normalize_value(name), False, name) for name in record.names]
        values += [("SITE_ADDRESS", normalize_value(record.address or ""), True, record.address or "")]
        values += [("ENTITY_SITE_RELATION", normalize_value(canonical_bytes(relation_object).decode()), True, relation_object)]
        for query_id in planned.query_ids:
            proof = hashlib.sha256(canonical_bytes([query_id, record.siret, "SIRENE_RECORD", record.snapshot_sha256])).hexdigest()
            for kind, value, specific, excerpt in values:
                excerpt_raw = canonical_bytes(excerpt)
                excerpt_hash = hashlib.sha256(excerpt_raw).hexdigest()
                facts.append(Fact(query_id, proof, "SIRENE_REGISTRY", record.siret, record.siret[:9], kind, value, specific, excerpt_hash, "SIRENE_RECORD", record.record_payload_sha256, None, None))
                provenance.append(FactProvenance(query_id, proof, kind, value, record.siret, record.siret[:9], "SIRENE_RECORD", record.record_payload_sha256, excerpt_hash, None, None, None, record.siret, record.snapshot_sha256))
    return tuple(sorted(facts, key=lambda r: (r.query_id, r.proof_id, r.related_siret, r.fact_type, r.fact_value_normalized))), tuple(sorted(provenance, key=lambda r: (r.query_id, r.proof_id, r.related_siret, r.fact_type, r.fact_value_normalized)))


def _derive_evidence(archives: Sequence[PageArchive], occurrences: Sequence[IdentityOccurrence], facts: Sequence[Fact], lookup_plan: Sequence[LookupPlanRow], records: Sequence[SireneRecord]) -> tuple[Evidence, ...]:
    archive_by_id = {r.page_attempt_id: r for r in archives}; types = {}; group_sirets = {}; occ_by = {}
    for fact in facts: types.setdefault((fact.query_id, fact.proof_id, fact.related_siret), set()).add(fact.fact_type)
    for occ in occurrences:
        archive = archive_by_id[occ.page_attempt_id]; group_sirets.setdefault((occ.query_id, archive.independence_group), set()).add(occ.siret); occ_by.setdefault((occ.query_id, occ.siret), []).append((occ, archive))
    required = {"SIRET_IDENTIFIER", "SIREN_IDENTIFIER", "SITE_NAME", "SITE_ADDRESS", "ENTITY_SITE_RELATION"}; output = []
    for occ in occurrences:
        archive = archive_by_id[occ.page_attempt_id]; complete = types.get((occ.query_id, occ.page_attempt_id, occ.siret)) == required; contradiction = len(group_sirets[(occ.query_id, archive.independence_group)]) > 1
        output.append(Evidence(occ.query_id, occ.siret, archive.independence_group, occ.page_attempt_id, _eref(occ.query_id, occ.siret, archive.independence_group, occ.page_attempt_id), archive.raw_content_sha256, complete, complete, complete, complete, contradiction, complete and not contradiction))
    record_by = {r.siret: r for r in records}
    for planned in lookup_plan:
        record = record_by[planned.siret]
        for query_id in planned.query_ids:
            matching = False
            if record.found_count == 1 and record.state == "A":
                names = {normalize_value(v) for v in record.names}
                matching = any(normalize_value(a.crm_name) in names and _address_matches(a.crm_address, record.address or "") for _, a in occ_by.get((query_id, planned.siret), ()))
            proof = hashlib.sha256(canonical_bytes([query_id, planned.siret, "SIRENE_REGISTRY", record.snapshot_sha256])).hexdigest(); contradiction = record.found_count != 1
            output.append(Evidence(query_id, planned.siret, "SIRENE_REGISTRY", proof, _eref(query_id, planned.siret, "SIRENE_REGISTRY", proof), record.record_payload_sha256, matching, matching, matching, matching, contradiction, matching and not contradiction))
    keys = [(r.query_id, r.related_siret, r.independence_group, r.proof_id) for r in output]
    if len(keys) != len(set(keys)): raise BusinessIntegrityError("duplicate evidence")
    return tuple(sorted(output, key=lambda r: (r.query_id, r.related_siret, r.independence_group, r.proof_id)))


def _supported(evidence: Sequence[Evidence]):
    groups = {}
    for row in evidence:
        if row.conditional_group_supports: groups.setdefault((row.query_id, row.related_siret), set()).add(row.independence_group)
    by_query = {}
    for (qid, siret), found in groups.items():
        if "SIRENE_REGISTRY" in found and len(found) >= 2: by_query.setdefault(qid, []).append((siret, tuple(sorted(found))))
    return tuple((qid, tuple(sorted(values))) for qid, values in sorted(by_query.items()))


def _payload(seal: IdentitySeal | None, query_ids, lookup_plan, records, facts, provenance, evidence, supported):
    return {"claim": CLAIM, "barrier": IDENTITY_BARRIER, "query_ids": query_ids, "lookup_plan": [asdict(v) for v in lookup_plan], "sirene_records": [asdict(v) for v in records], "facts": [asdict(v) for v in facts], "provenance": [asdict(v) for v in provenance], "evidence": [asdict(v) for v in evidence], "conditional_support_by_query": supported}


def seal_identity(plan: Sequence[CollectionQuery], search_attempts: Sequence[SearchAttempt], search_results: Sequence[SearchResult], page_decisions: Sequence[PageDecision], dns: Sequence[DnsIdentityRow], archives: Sequence[PageArchive], occurrences: Sequence[IdentityOccurrence], provenance: Sequence[OccurrenceProvenance], lookup_responses: Sequence[Mapping[str, Any]], *, snapshot_ref: str, snapshot_sha256: str) -> IdentitySeal:
    query_ids = tuple(sorted({r.query_id for r in plan})); _sha(snapshot_sha256, "snapshot_sha256"); _text(snapshot_ref, "snapshot_ref")
    if len(query_ids) != 30: raise BusinessIntegrityError("seal requires 30 dossiers")
    expected_attempts = {(r.query_id, r.query_ordinal): search_attempt_id(r.query_id, r.query_ordinal, r.search_query) for r in plan}
    attempt_by_key = {(r.query_id, r.query_ordinal): r for r in search_attempts}; result_by_key = {(r.query_id, r.query_ordinal, r.result_rank): r for r in search_results}
    dns_by = {r.dns_attempt_id: r for r in dns}; archive_by = {r.page_attempt_id: r for r in archives}; frozen_crm = {r.query_id: (r.crm_name, r.crm_address) for r in plan}
    if len(search_attempts) != 90 or set(attempt_by_key) != set(expected_attempts) or any(r.search_attempt_id != expected_attempts[key] or r.search_query != next(p.search_query for p in plan if (p.query_id, p.query_ordinal) == key) or r.status not in {"SUCCESS", "ERROR"} for key, r in attempt_by_key.items()) or len(result_by_key) != len(search_results) or len(dns_by) != len(dns) or len(archive_by) != len(archives): raise BusinessIntegrityError("seal duplicate/bijective SEARCH PK")
    for key, result in result_by_key.items():
        attempt = attempt_by_key.get((result.query_id, result.query_ordinal))
        if attempt is None or attempt.status != "SUCCESS" or result.search_attempt_id != attempt.search_attempt_id or result.result_payload_sha256 != search_result_id(result.query_id, result.query_ordinal, result.result_rank, result.title, result.snippet, result.resolved_url): raise BusinessIntegrityError("seal SEARCH result identity mismatch")
    expected_decisions = derive_page_decisions(search_results)
    if tuple(page_decisions) != expected_decisions or {d.page_attempt_id for d in page_decisions if d.decision == "OPEN_ATTEMPT"} != set(archive_by): raise BusinessIntegrityError("seal PAGE decision/open bijection mismatch")
    for archive in archives:
        dns_row = dns_by.get(archive.dns_attempt_id)
        result = result_by_key.get((archive.query_id, archive.query_ordinal, archive.result_rank)); attempt = attempt_by_key.get((archive.query_id, archive.query_ordinal))
        if result is None or attempt is None or not 1 <= archive.query_open_slot <= 2 or not 1 <= archive.dossier_open_ordinal <= 6 or result.search_attempt_id != attempt.search_attempt_id or archive.search_attempt_id != attempt.search_attempt_id or archive.result_payload_sha256 != result.result_payload_sha256 or archive.requested_url != result.resolved_url or archive.page_attempt_id != page_attempt_id(archive.query_id, archive.query_ordinal, archive.result_rank, result.resolved_url, archive.query_open_slot, archive.dossier_open_ordinal) or dns_row is None or dns_row.status != "SUCCESS" or dns_row.parent_attempt_id != archive.page_attempt_id or dns_row.request_kind != "PAGE" or (dns_row.query_id, dns_row.query_ordinal, dns_row.result_rank) != (archive.query_id, archive.query_ordinal, archive.result_rank) or frozen_crm.get(archive.query_id) != (archive.crm_name, archive.crm_address): raise BusinessIntegrityError("seal input FK mismatch")
        try: raw_content = base64.b64decode(archive.raw_content_b64, validate=True)
        except Exception as exc: raise BusinessIntegrityError("seal archive base64 mismatch") from exc
        if archive.mime_type != "text/plain" or archive.decoder_rule_id != TEXT_DECODER_RULE: raise BusinessIntegrityError("seal decoder contract mismatch")
        if archive.text_decoder == "UTF8_STRICT" and archive.charset == "UTF-8": rebuilt = raw_content.decode("utf-8", errors="strict")
        elif archive.text_decoder == "UTF8_REPLACE" and archive.charset == "UTF-8": rebuilt = raw_content.decode("utf-8", errors="replace")
        elif archive.text_decoder == "ISO_8859_1" and archive.charset == "ISO-8859-1": rebuilt = raw_content.decode("iso-8859-1", errors="strict")
        else: raise BusinessIntegrityError("seal decoder tuple mismatch")
        if hashlib.sha256(raw_content).hexdigest() != archive.raw_content_sha256 or rebuilt != archive.extracted_text or hashlib.sha256(rebuilt.encode()).hexdigest() != archive.extracted_text_sha256: raise BusinessIntegrityError("seal archive hash mismatch")
    if len({(a.query_id, a.query_ordinal, a.query_open_slot) for a in archives}) != len(archives) or len({(a.query_id, a.dossier_open_ordinal) for a in archives}) != len(archives): raise BusinessIntegrityError("seal page slot/quota mismatch")
    for occ in occurrences:
        archive = archive_by.get(occ.page_attempt_id)
        if archive is None or occ.query_id != archive.query_id or occ.extractor_rule_id != EXTRACTOR_RULE: raise BusinessIntegrityError("seal occurrence FK mismatch")
        encoded = archive.extracted_text.encode("utf-8")
        for span in (occ.siret_span, occ.name_span, occ.address_span, occ.relation_span):
            if type(span) is not tuple or len(span) != 2 or type(span[0]) is not int or type(span[1]) is not int or not 0 <= span[0] < span[1] <= len(encoded): raise BusinessIntegrityError("seal occurrence span mismatch")
        relation = _slice(encoded, occ.relation_span, "relation")
        if hashlib.sha256(relation.encode()).hexdigest() != occ.source_excerpt_sha256 or not all(occ.relation_span[0] <= span[0] < span[1] <= occ.relation_span[1] for span in (occ.siret_span, occ.name_span, occ.address_span)) or occ.relation_span[1] - occ.relation_span[0] > MAX_TRIPLE_SPAN: raise BusinessIntegrityError("seal occurrence excerpt/triple mismatch")
        ordered = sorted((occ.siret_span, occ.name_span, occ.address_span))
        if max((ordered[i + 1][0] - ordered[i][1] for i in range(2)), default=0) > MAX_COMPONENT_GAP: raise BusinessIntegrityError("seal occurrence distance mismatch")
        digits = re.sub(r"[ .-]", "", _slice(encoded, occ.siret_span, "SIRET"))
        if _siret(digits) != occ.siret or normalize_value(_slice(encoded, occ.name_span, "name")) != normalize_value(archive.crm_name) or not _address_matches(_slice(encoded, occ.address_span, "address"), archive.crm_address): raise BusinessIntegrityError("seal occurrence identity mismatch")
    expected_prov = {(o.query_id, o.page_attempt_id, o.siret): o for o in occurrences}
    if len(provenance) != len(expected_prov): raise BusinessIntegrityError("seal provenance cardinality mismatch")
    for prov in provenance:
        occ = expected_prov.get((prov.query_id, prov.proof_id, prov.related_siret)); archive = archive_by.get(prov.proof_id)
        if occ is None or archive is None or prov.related_siren != prov.related_siret[:9] or prov.raw_content_sha256 != archive.raw_content_sha256 or prov.extracted_text_sha256 != archive.extracted_text_sha256 or prov.source_excerpt_sha256 != occ.source_excerpt_sha256 or (prov.siret_span, prov.name_span, prov.address_span, prov.relation_span, prov.extractor_rule_id) != (occ.siret_span, occ.name_span, occ.address_span, occ.relation_span, occ.extractor_rule_id): raise BusinessIntegrityError("seal provenance mismatch")
    plan_lookup = derive_lookup_plan(occurrences); records = convert_lookup_responses(plan_lookup, lookup_responses, snapshot_ref, snapshot_sha256)
    web_facts, web_prov = _derive_facts(archives, occurrences, provenance); registry_facts, registry_prov = _sirene_facts(plan_lookup, records)
    facts = tuple(sorted(web_facts + registry_facts, key=lambda r: (r.query_id, r.proof_id, r.related_siret, r.fact_type, r.fact_value_normalized)))
    fact_provenance = tuple(sorted(web_prov + registry_prov, key=lambda r: (r.query_id, r.proof_id, r.related_siret, r.fact_type, r.fact_value_normalized)))
    evidence = _derive_evidence(archives, occurrences, web_facts, plan_lookup, records); supported = _supported(evidence)
    raw = canonical_bytes(_payload(None, query_ids, plan_lookup, records, facts, fact_provenance, evidence, supported)); digest = hashlib.sha256(raw).hexdigest()
    return IdentitySeal(CLAIM, IDENTITY_BARRIER, query_ids, plan_lookup, records, facts, fact_provenance, evidence, supported, digest)


def identity_bytes(seal: IdentitySeal) -> bytes:
    raw = canonical_bytes(_payload(seal, seal.query_ids, seal.lookup_plan, seal.sirene_records, seal.facts, seal.provenance, seal.evidence, seal.conditional_support_by_query))
    if seal.claim != CLAIM or seal.barrier != IDENTITY_BARRIER or hashlib.sha256(raw).hexdigest() != seal.identity_sha256: raise BusinessIntegrityError("identity barrier/hash mismatch")
    return raw


def compare_after_barrier(seal: IdentitySeal, candidate_context: Sequence[Mapping[str, Any]]) -> tuple[tuple[CandidateComparison, ...], tuple[ConditionalSupportSummary, ...]]:
    identity_bytes(seal)
    if len(candidate_context) != 3000: raise BusinessIntegrityError("candidate context must be exactly 30x100")
    candidates, seen, seen_sirets = {}, set(), set()
    for raw in candidate_context:
        _closed(raw, {"query_id", "rank", "candidate_siret"}, "candidate")
        qid, rank, siret = _text(raw["query_id"], "query_id"), _int(raw["rank"], "rank", 1, 100), _siret(raw["candidate_siret"], luhn=False)
        if qid not in seal.query_ids or (qid, rank) in seen or (qid, siret) in seen_sirets: raise BusinessIntegrityError("candidate PK mismatch")
        seen.add((qid, rank)); seen_sirets.add((qid, siret)); candidates.setdefault(qid, []).append((rank, siret))
    if set(candidates) != set(seal.query_ids) or any(sorted(r for r, _ in rows) != list(range(1, 101)) for rows in candidates.values()): raise BusinessIntegrityError("candidate ranks mismatch")
    support = {qid: dict(values) for qid, values in seal.conditional_support_by_query}; ev = {}; records = {r.siret: r for r in seal.sirene_records}
    for row in seal.evidence: ev.setdefault((row.query_id, row.related_siret), []).append(row)
    matrix, decisions = [], []
    for qid in seal.query_ids:
        pool = sorted(candidates[qid]); top1 = pool[0][1]; supported = support.get(qid, {}); sirets = tuple(sorted(supported))
        for rank, siret in pool:
            record = records.get(siret); matrix.append(CandidateComparison(qid, rank, siret, rank == 1, siret in supported, tuple(supported.get(siret, ())), any(x.within_group_contradiction for x in ev.get((qid, siret), ())), any(x != siret and x[:9] == siret[:9] for x in sirets), any(x[:9] != siret[:9] for x in sirets), bool(record and record.found_count == 1), bool(record and record.found_count == 1 and record.state == "A")))
        refs = tuple(sorted({x.evidence_ref_id for x in seal.evidence if x.query_id == qid and x.related_siret in supported and x.conditional_group_supports}))
        if not sirets: outcome, count, alt, in_pool = "NO_CONDITIONAL_SUPPORT", 0, None, False
        elif len(sirets) > 1: outcome, count, alt, in_pool = "MULTIPLE_CONDITIONAL_SUPPORT", min(len(supported[x]) for x in sirets), None, False
        elif sirets[0] == top1: outcome, count, alt, in_pool = "TOP1_IN_CONDITIONAL_SUPPORT", len(supported[top1]), None, False
        else: alt = sirets[0]; outcome, count, in_pool = "TOP1_OUTSIDE_CONDITIONAL_SUPPORT", len(supported[alt]), any(x == alt for _, x in pool)
        collision = None
        if outcome == "TOP1_OUTSIDE_CONDITIONAL_SUPPORT": collision = "SAME_SIREN_MULTISITE" if alt and alt[:9] == top1[:9] else "CROSS_SIREN_COLLISION"
        elif outcome == "MULTIPLE_CONDITIONAL_SUPPORT": collision = "SAME_SIREN_MULTISITE" if len({x[:9] for x in sirets}) == 1 else "CROSS_SIREN_COLLISION"
        decisions.append(ConditionalSupportSummary(qid, outcome, False, top1, sirets, count, refs, alt, in_pool, collision))
    return tuple(matrix), tuple(decisions)
