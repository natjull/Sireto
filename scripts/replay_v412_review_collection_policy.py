#!/usr/bin/env python3
"""Offline replay of the frozen V4.12-R30 collection policy goldens.

This module is deliberately pure with respect to external state: it reads only
the explicitly supplied policy/vector files and never opens a socket.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import unicodedata
from urllib.parse import parse_qsl, urlparse

import idna
from lxml import html


EVIDENCE_DOMAIN = b"SIRETO-V412-R30-EVIDENCE\0"
POLICY_SHA256 = "1238eb957f84c811ac64375c66a0d62e1bef977a139c0a685e669a5d18c63b88"
ALLOWED_RELATIVE_PINS = frozenset(
    {
        "config/v4_12_review_ddg_parser_fixture.html",
        "config/v4_12_review_ddg_parser_expected.json",
        "config/v4_12_review_ddg_charset_vectors.json",
        "config/v4_12_review_public_suffixes.txt",
        "config/v4_12_review_domain_vectors.json",
        "config/v4_12_review_postopen_validation_vectors.json",
        "config/v4_12_review_fact_reconstruction_vectors.json",
        "config/v4_12_review_evidence_vectors.json",
        "config/v4_12_review_adjudication_vectors.json",
        "config/v4_12_review_identifier_vectors.json",
        "config/v4_12_review_dns_security_vectors.json",
    }
)
ALLOWED_ABSOLUTE_PINS = frozenset(
    {
        "/opt/homebrew/etc/ca-certificates/cert.pem",
        "/private/etc/hosts",
        "/private/var/run/resolv.conf",
    }
)
NAME_STOPWORDS_DEFAULT = frozenset()
ROAD_STOPWORDS = frozenset(
    {"rue", "avenue", "av", "boulevard", "bd", "chemin", "route", "place", "allee", "impasse"}
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_same_fd(path: Path) -> str:
    """Hash one regular mono-link file without reopening or following its leaf."""

    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"absolute pin is not a mono-link regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ValueError(f"absolute pin mutated during same-FD read: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def normalize_value(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    ).casefold()
    collapsed = "".join(character if character.isalnum() else " " for character in without_marks)
    return " ".join(collapsed.split())


def name_tokens(value: str, stopwords: frozenset[str], minimum: int = 4) -> list[str]:
    return [
        token
        for token in normalize_value(value).split()
        if len(token) >= minimum and token not in stopwords
    ]


def _contains_contiguous(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return False
    width = len(needle)
    return any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1))


def luhn_valid(identifier: str) -> bool:
    if not identifier.isascii() or not identifier.isdigit():
        return False
    total = 0
    parity = len(identifier) % 2
    for index, character in enumerate(identifier):
        value = int(character)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


_SIRET_PATTERN = re.compile(rb"(?<![0-9])(?:[0-9][ .-]?){13}[0-9](?![0-9])")
_SIREN_PATTERN = re.compile(rb"(?<![0-9])[0-9]{9}(?![0-9])")


def extract_identifiers(text: str) -> tuple[list[str], list[str]]:
    payload = text.encode("utf-8")
    sirets: list[str] = []
    sirens: list[str] = []
    siret_spans: list[tuple[int, int]] = []
    for match in _SIRET_PATTERN.finditer(payload):
        digits = bytes(value for value in match.group() if 48 <= value <= 57).decode()
        if luhn_valid(digits):
            sirets.append(digits)
            sirens.append(digits[:9])
            siret_spans.append(match.span())
    for match in _SIREN_PATTERN.finditer(payload):
        if any(start <= match.start() and match.end() <= end for start, end in siret_spans):
            continue
        digits = match.group().decode()
        if luhn_valid(digits):
            sirens.append(digits)
    return list(dict.fromkeys(sirets)), list(dict.fromkeys(sirens))


def normalize_hostname(hostname: str) -> str:
    value = unicodedata.normalize("NFC", hostname)
    if value.endswith("."):
        value = value[:-1]
    return idna.encode(value, uts46=True, std3_rules=True).decode("ascii").lower()


def suffix_matches(host: str, suffix: str) -> bool:
    return host == suffix or host.endswith("." + suffix)


def registrable_domain(host: str, public_suffixes: list[str]) -> str | None:
    matching = [suffix for suffix in public_suffixes if suffix_matches(host, suffix)]
    if not matching:
        return None
    suffix = max(matching, key=lambda value: len(value.split(".")))
    host_labels = host.split(".")
    suffix_labels = suffix.split(".")
    if len(host_labels) <= len(suffix_labels):
        return None
    return ".".join(host_labels[-len(suffix_labels) - 1 :])


def evaluate_domain_vector(vector: dict, policy: dict, public_suffixes: list[str]) -> dict:
    try:
        host = normalize_hostname(vector["input_hostname"])
    except (idna.IDNAError, UnicodeError):
        host = vector["input_hostname"].casefold().rstrip(".")
        domain = None
        safe = False
    else:
        forbidden_hosts = set(policy["network_security"]["forbidden_hostnames"])
        forbidden_suffixes = policy["network_security"]["forbidden_hostname_suffixes"]
        domain = registrable_domain(host, public_suffixes)
        safe = bool(domain) and host not in forbidden_hosts and not any(
            suffix_matches(host, suffix) for suffix in forbidden_suffixes
        )
    domain_policy = policy["domain_policy"]
    return {
        "normalized_hostname": host,
        "registrable_domain": domain,
        "matches_public_administration": any(
            suffix_matches(host, suffix)
            for suffix in domain_policy["public_administration_suffixes"]
        ),
        "matches_sirene_copy": any(
            suffix_matches(host, suffix)
            for suffix in domain_policy["sirene_copy_suffixes"]
        ),
        "safe": safe,
    }


def classify_preopen(
    url: str,
    title: str,
    snippet: str,
    crm_name: str,
    policy: dict,
    public_suffixes: list[str],
) -> dict:
    """Apply the closed URL checks and pinned pre-open precedence."""

    unsafe = False
    try:
        parsed = urlparse(url)
        port = parsed.port
        host_input = parsed.hostname
        if (
            parsed.scheme.casefold() != "https"
            or not parsed.netloc
            or host_input is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or port not in {None, 443}
        ):
            unsafe = True
        if not unsafe:
            try:
                ipaddress.ip_address(host_input)
            except ValueError:
                pass
            else:
                unsafe = True
        host = normalize_hostname(host_input or "") if not unsafe else None
    except (ValueError, idna.IDNAError, UnicodeError):
        unsafe, host = True, None
    domain = registrable_domain(host, public_suffixes) if host is not None else None
    if unsafe:
        return {"family": "INADMISSIBLE", "reason": "UNSAFE_URL", "normalized_hostname": host, "registrable_domain": domain}
    domain_policy = policy["domain_policy"]
    network_policy = policy["network_security"]
    if host in network_policy["forbidden_hostnames"] or any(
        suffix_matches(host, suffix) for suffix in network_policy["forbidden_hostname_suffixes"]
    ):
        return {"family": "INADMISSIBLE", "reason": "UNSAFE_URL", "normalized_hostname": host, "registrable_domain": domain}
    if domain is None:
        return {"family": "INADMISSIBLE", "reason": "UNRECOGNIZED_PUBLIC_SUFFIX", "normalized_hostname": host, "registrable_domain": None}
    if suffix_matches(host, "duckduckgo.com"):
        return {"family": "INADMISSIBLE", "reason": "SEARCH_ENGINE", "normalized_hostname": host, "registrable_domain": domain}
    precedence = (
        ("sirene_copy_suffixes", "SIRENE_COPY"),
        ("always_inadmissible_suffixes", "COMMERCIAL_AGGREGATOR"),
    )
    for key, reason in precedence:
        if any(suffix_matches(host, suffix) for suffix in domain_policy[key]):
            return {"family": "INADMISSIBLE", "reason": reason, "normalized_hostname": host, "registrable_domain": domain}
    if any(suffix_matches(host, suffix) for suffix in domain_policy["public_administration_suffixes"]):
        family = "PUBLIC_ADMINISTRATION"
    elif any(suffix_matches(host, suffix) for suffix in domain_policy["official_sector_directory_suffixes"]):
        family = "OFFICIAL_SECTOR_DIRECTORY"
    else:
        crm_tokens = set(name_tokens(
            crm_name,
            frozenset(domain_policy["name_stopwords"]),
            domain_policy["significant_token_minimum_length"],
        ))
        observed_tokens = set(normalize_value(" ".join((host, title, snippet))).split())
        intersects = bool(crm_tokens & observed_tokens)
        if parsed.path.casefold().endswith(".pdf") and intersects:
            family = "DATED_PUBLIC_DOCUMENT_CANDIDATE"
        elif intersects:
            family = "ENTITY_OFFICIAL_SITE_CANDIDATE"
        else:
            return {"family": "INADMISSIBLE", "reason": "DOMAIN_NOT_ALLOWLISTED", "normalized_hostname": host, "registrable_domain": domain}
    return {"family": family, "reason": "NONE", "normalized_hostname": host, "registrable_domain": domain}


def forbidden_address(address: str, networks: list[ipaddress._BaseNetwork]) -> bool:
    parsed = ipaddress.ip_address(address)
    if any(parsed in network for network in networks):
        return True
    return bool(
        parsed.version == 6
        and parsed.ipv4_mapped is not None
        and any(parsed.ipv4_mapped in network for network in networks)
    )


def evaluate_resolution(addresses: list[str], networks: list[ipaddress._BaseNetwork]) -> dict:
    parsed = sorted({ipaddress.ip_address(value) for value in addresses}, key=lambda value: (value.version, value.packed))
    ordered = [str(value) for value in parsed]
    if not parsed:
        return {"expected_addresses": ordered, "permitted": False, "chosen_ip": None, "error_type": "DNS"}
    if any(forbidden_address(str(value), networks) for value in parsed):
        return {"expected_addresses": ordered, "permitted": False, "chosen_ip": None, "error_type": "PRIVATE_ADDRESS"}
    return {"expected_addresses": ordered, "permitted": True, "chosen_ip": ordered[0], "error_type": None}


def _decode_ddg_body(payload: bytes, charset: str | None) -> str:
    normalized = (charset or "").strip().strip('"\'').casefold()
    if normalized in {"utf-8", "utf8"}:
        return payload.decode("utf-8", errors="strict")
    if normalized in {"iso-8859-1", "latin1", "latin-1"}:
        return payload.decode("iso-8859-1", errors="strict")
    return payload.decode("utf-8", errors="replace")


def _resolve_ddg_href(observed: str) -> str | None:
    parsed = urlparse(observed)
    wrapper = False
    if observed.startswith("/l/"):
        parsed = urlparse("https://duckduckgo.com" + observed)
        wrapper = True
    elif observed.startswith("//"):
        parsed = urlparse("https:" + observed)
        wrapper = parsed.hostname == "duckduckgo.com" and parsed.path.startswith("/l/")
    elif parsed.scheme and parsed.netloc:
        wrapper = parsed.hostname == "duckduckgo.com" and parsed.path.startswith("/l/")
        if not wrapper:
            return observed
    else:
        return None
    if not wrapper:
        return None
    if any(
        character == "%" and (index + 2 >= len(parsed.query) or not re.fullmatch(r"[0-9A-Fa-f]{2}", parsed.query[index + 1 : index + 3]))
        for index, character in enumerate(parsed.query)
    ):
        return None
    try:
        pairs = parse_qsl(
            parsed.query,
            keep_blank_values=True,
            strict_parsing=False,
            encoding="utf-8",
            errors="strict",
        )
    except UnicodeDecodeError:
        return None
    values = [value for key, value in pairs if key == "uddg"]
    return values[0] if len(values) == 1 else None


def parse_ddg_results(payload: bytes, charset: str | None) -> list[dict]:
    document = html.fromstring(_decode_ddg_body(payload, charset))
    results: list[dict] = []
    seen: set[str] = set()
    for element in document.xpath("//div"):
        class_tokens = set((element.get("class") or "").split())
        if "result" not in class_tokens or "result--ad" in class_tokens or element.get("data-testid") == "ad":
            continue
        anchors = element.xpath('.//a[contains(concat(" ", normalize-space(@class), " "), " result__a ")][@href]')
        snippets = element.xpath('.//*[contains(concat(" ", normalize-space(@class), " "), " result__snippet ")]')
        if not anchors:
            continue
        observed = anchors[0].get("href")
        resolved = _resolve_ddg_href(observed)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        results.append(
            {
                "observed_href": observed,
                "resolved_url": resolved,
                "result_rank": len(results) + 1,
                "snippet": " ".join(snippets[0].text_content().split()) if snippets else "",
                "title": " ".join(anchors[0].text_content().split()),
            }
        )
        if len(results) == 5:
            break
    return results


def _tokens_with_byte_spans(value: str) -> list[tuple[str, int, int]]:
    """Apply the pinned normalizer while retaining original UTF-8 spans."""

    units: list[tuple[str | None, int, int]] = []
    byte_offset = 0
    for character in value:
        encoded = character.encode("utf-8")
        start, end = byte_offset, byte_offset + len(encoded)
        byte_offset = end
        normalized = "".join(
            item
            for item in unicodedata.normalize("NFKD", character)
            if unicodedata.category(item) != "Mn"
        ).casefold()
        for item in normalized:
            units.append((item if item.isalnum() else None, start, end))
    tokens: list[tuple[str, int, int]] = []
    current: list[str] = []
    token_start = token_end = 0
    for item, start, end in units + [(None, byte_offset, byte_offset)]:
        if item is not None:
            if not current:
                token_start = start
            current.append(item)
            token_end = end
        elif current:
            tokens.append(("".join(current), token_start, token_end))
            current = []
    return tokens


def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] < right[0]:
        return right[0] - left[1]
    if right[1] < left[0]:
        return left[0] - right[1]
    return 0


def _name_spans(
    tokens: list[tuple[str, int, int]], crm_name: str, stopwords: frozenset[str], minimum: int
) -> list[tuple[int, int]]:
    full = normalize_value(crm_name).split()
    significant = [item for item in full if len(item) >= minimum and item not in stopwords]
    if not significant:
        return []
    spans: list[tuple[int, int]] = []
    # Prefer the complete CRM name; it is the fact excerpt pinned by the vectors.
    for needle in (full, significant):
        width = len(needle)
        for index in range(len(tokens) - width + 1):
            if [item[0] for item in tokens[index : index + width]] == needle:
                spans.append((tokens[index][1], tokens[index + width - 1][2]))
        if spans:
            break
    return spans


def _address_spans(
    tokens: list[tuple[str, int, int]],
    crm_address: str,
    crm_postcode: str,
    local_start: int,
    local_end: int,
) -> list[tuple[int, int]]:
    address_tokens = normalize_value(crm_address).split()
    number = next((item for item in address_tokens if re.fullmatch(r"[0-9]{1,5}", item)), None)
    roads = [
        item
        for item in address_tokens
        if item not in ROAD_STOPWORDS and not item.isdigit() and len(item) >= 3
    ]
    required_road_count = 1 if number is not None else min(2, len(roads))
    local = [item for item in tokens if item[1] >= local_start and item[2] <= local_end]
    candidates: list[tuple[int, int]] = []
    for postcode in (item for item in local if item[0] == crm_postcode):
        window = [
            item
            for item in local
            if item[1] >= postcode[1] - 192 and item[2] <= postcode[2] + 192
        ]
        numbers = [item for item in window if number is not None and item[0] == number]
        if number is None:
            numbers = [None]
        road_occurrences = [item for item in window if item[0] in roads]
        distinct_roads = {item[0] for item in road_occurrences}
        if len(distinct_roads) < required_road_count:
            continue
        for number_occurrence in numbers:
            # Enumerating all subsets is unnecessary: each pair of boundary road
            # tokens defines the minimal span and the count can be checked inside.
            boundaries = road_occurrences or [postcode]
            for left_road in boundaries:
                for right_road in boundaries:
                    selected = [postcode, left_road, right_road]
                    if number_occurrence is not None:
                        selected.append(number_occurrence)
                    start = min(item[1] for item in selected)
                    end = max(item[2] for item in selected)
                    present = {item[0] for item in window if item[1] >= start and item[2] <= end and item[0] in roads}
                    if len(present) >= required_road_count:
                        candidates.append((start, end))
    return sorted(set(candidates), key=lambda span: (span[1] - span[0], span[0], span[1]))


def reconstruct_facts(
    text: str,
    crm_name: str,
    crm_address: str,
    crm_postcode: str,
    *,
    stopwords: frozenset[str],
    minimum_name_token_length: int,
) -> tuple[list[list], list[str]]:
    """Reconstruct the five pinned web facts for every qualified SIRET."""

    payload = text.encode("utf-8")
    tokens = _tokens_with_byte_spans(text)
    names = _name_spans(tokens, crm_name, stopwords, minimum_name_token_length)
    facts: list[list] = []
    unqualified_candidates: list[str] = []
    unqualified_set: set[str] = set()
    qualified_sirets: set[str] = set()
    for match in _SIRET_PATTERN.finditer(payload):
        digits = bytes(item for item in match.group() if 48 <= item <= 57).decode()
        if not luhn_valid(digits) or digits in qualified_sirets:
            continue
        local_start, local_end = max(0, match.start() - 512), min(len(payload), match.end() + 512)
        local_names = [span for span in names if span[0] >= local_start and span[1] <= local_end]
        addresses = [
            span
            for span in _address_spans(tokens, crm_address, crm_postcode, local_start, local_end)
            if span[1] <= match.start() or span[0] >= match.end()
        ]
        if not local_names or not addresses:
            if digits not in unqualified_set:
                unqualified_candidates.append(digits)
                unqualified_set.add(digits)
            continue
        name_span = min(local_names, key=lambda span: (_distance(span, match.span()), span[0], span[1]))
        address_span = min(addresses, key=lambda span: (_distance(span, match.span()), span[1] - span[0], span[0], span[1]))
        siret_span = match.span()
        digit_count = 0
        siren_end = match.start()
        for offset, byte in enumerate(match.group(), start=match.start()):
            if 48 <= byte <= 57:
                digit_count += 1
                if digit_count == 9:
                    siren_end = offset + 1
                    break
        spans = {
            "SIRET_IDENTIFIER": siret_span,
            "SIREN_IDENTIFIER": (match.start(), siren_end),
            "SITE_NAME": name_span,
            "SITE_ADDRESS": address_span,
        }
        spans["ENTITY_SITE_RELATION"] = (
            min(span[0] for span in spans.values()), max(span[1] for span in spans.values())
        )
        for fact_type in (
            "SIRET_IDENTIFIER", "SIREN_IDENTIFIER", "SITE_NAME", "SITE_ADDRESS", "ENTITY_SITE_RELATION"
        ):
            start, end = spans[fact_type]
            excerpt = payload[start:end]
            normalized = (
                digits if fact_type == "SIRET_IDENTIFIER" else
                digits[:9] if fact_type == "SIREN_IDENTIFIER" else
                normalize_value(excerpt.decode("utf-8", errors="strict"))
            )
            facts.append([digits, fact_type, normalized, start, end, sha256_bytes(excerpt)])
        qualified_sirets.add(digits)
    unqualified = [
        digits for digits in unqualified_candidates if digits not in qualified_sirets
    ]
    return facts, unqualified


def postopen_family(vector: dict, policy: dict) -> tuple[str, bool]:
    preopen = vector["preopen_family"]
    if preopen == "INADMISSIBLE":
        return "INADMISSIBLE_AFTER_OPEN", False
    domain_policy = policy["domain_policy"]
    facts, _ = reconstruct_facts(
        vector["extracted_text"], vector["crm_name"], vector["crm_address"], vector["crm_postcode"],
        stopwords=frozenset(domain_policy["name_stopwords"]),
        minimum_name_token_length=domain_policy["significant_token_minimum_length"],
    )
    qualifying = bool(facts)
    if not qualifying:
        return "INADMISSIBLE_AFTER_OPEN", False
    try:
        host = normalize_hostname(vector["normalized_hostname"])
    except (idna.IDNAError, UnicodeError):
        return "INADMISSIBLE_AFTER_OPEN", False
    if preopen == "PUBLIC_ADMINISTRATION" and not any(
        suffix_matches(host, suffix) for suffix in domain_policy["public_administration_suffixes"]
    ):
        return "INADMISSIBLE_AFTER_OPEN", False
    if preopen == "OFFICIAL_SECTOR_DIRECTORY" and not any(
        suffix_matches(host, suffix) for suffix in domain_policy["official_sector_directory_suffixes"]
    ):
        return "INADMISSIBLE_AFTER_OPEN", False
    if preopen == "DATED_PUBLIC_DOCUMENT_CANDIDATE":
        valid_date = False
        text = vector["extracted_text"]
        for year, month, day in re.findall(r"(?<![0-9])([0-9]{4})-([0-9]{2})-([0-9]{2})(?![0-9])", text):
            try:
                __import__("datetime").date(int(year), int(month), int(day)); valid_date = True
            except ValueError:
                pass
        for day, month, year in re.findall(r"(?<![0-9])([0-9]{2})[/-]([0-9]{2})[/-]([0-9]{4})(?![0-9])", text):
            try:
                __import__("datetime").date(int(year), int(month), int(day)); valid_date = True
            except ValueError:
                pass
        return ("DATED_PUBLIC_DOCUMENT", True) if valid_date else ("INADMISSIBLE_AFTER_OPEN", False)
    mapping = {
        "PUBLIC_ADMINISTRATION": "PUBLIC_ADMINISTRATION",
        "OFFICIAL_SECTOR_DIRECTORY": "OFFICIAL_SECTOR_DIRECTORY",
        "ENTITY_OFFICIAL_SITE_CANDIDATE": "ENTITY_OFFICIAL_SITE",
    }
    return mapping.get(preopen, "INADMISSIBLE_AFTER_OPEN"), preopen in mapping


def evidence_ref_id(query_id: str, siret: str, group: str, proof_id: str) -> str:
    return sha256_bytes(EVIDENCE_DOMAIN + canonical_json_bytes([query_id, siret, group, proof_id]))


def _address_matches(left: str, right: str) -> bool:
    left_tokens, right_tokens = normalize_value(left).split(), normalize_value(right).split()
    left_postcodes = [value for value in left_tokens if value.isdigit() and len(value) == 5]
    right_postcodes = [value for value in right_tokens if value.isdigit() and len(value) == 5]
    left_numbers = [value for value in left_tokens if value.isdigit() and len(value) <= 5 and value not in left_postcodes]
    right_numbers = [value for value in right_tokens if value.isdigit() and len(value) <= 5 and value not in right_postcodes]
    road_left = {value for value in left_tokens if not value.isdigit() and value not in ROAD_STOPWORDS and len(value) >= 3}
    road_right = {value for value in right_tokens if not value.isdigit() and value not in ROAD_STOPWORDS and len(value) >= 3}
    if not left_postcodes or left_postcodes != right_postcodes or left_numbers[:1] != right_numbers[:1]:
        return False
    common_roads = road_left & road_right
    required = 1 if left_numbers else min(2, len(road_left))
    return len(common_roads) >= required


def replay_evidence_case(case: dict) -> list[list]:
    web = case["web_proofs"]
    sirets_by_group: dict[str, set[str]] = {}
    for _, group, _, siret, name, address, site_specific in web:
        if site_specific and name and address:
            sirets_by_group.setdefault(group, set()).add(siret)
    rows: list[list] = []
    for proof, group, archive, siret, _name, _address, site_specific in web:
        contradiction = len(sirets_by_group.get(group, set())) > 1
        base_passes = bool(_name and _address)
        passes = bool(site_specific and base_passes)
        rows.append([siret, group, proof, archive, base_passes, base_passes, base_passes, passes, contradiction, passes and not contradiction, evidence_ref_id(case["query_id"], siret, group, proof)])
    by_siret: dict[str, list[list]] = {}
    for row in web:
        by_siret.setdefault(row[3], []).append(row)
    for proof, archive, siret, unique, state, names, address in case["sirene_records"]:
        matching = False
        if unique and state == "A":
            normalized_names = {normalize_value(value) for value in names}
            matching = any(
                bool(row[6])
                and normalize_value(row[4]) in normalized_names
                and _address_matches(row[5], address or "")
                for row in by_siret.get(siret, [])
            )
        contradiction = not unique
        rows.append([siret, "SIRENE_REGISTRY", proof, archive, matching, matching, matching, matching, contradiction, matching and not contradiction, evidence_ref_id(case["query_id"], siret, "SIRENE_REGISTRY", proof)])
    return rows


def adjudicate_vector(vector: dict) -> dict:
    support = {
        siret: sorted(set(groups))
        for siret, groups in vector.get("supporting_groups_by_siret", {}).items()
        if "SIRENE_REGISTRY" in groups and len(set(groups)) >= 2
    }
    sirets = sorted(support)
    if not sirets:
        status, reliable, count, alternative, in_pool = "UNRESOLVED", False, 0, None, False
    elif len(sirets) >= 2:
        status, reliable, count, alternative, in_pool = "AMBIGUOUS", True, min(map(len, support.values())), None, False
    elif sirets[0] == vector["top1_siret"]:
        status, reliable, count, alternative, in_pool = "TOP1_CORRECT", True, len(support[sirets[0]]), None, False
    else:
        alternative = sirets[0]
        status, reliable, count, in_pool = "TOP1_WRONG", True, len(support[alternative]), alternative in vector["top100_sirets"]
    return {"status": status, "reliable": reliable, "group_count": count, "alternative_siret": alternative, "alternative_in_top100": in_pool}


def validate_policy_goldens(repo_root: Path) -> dict[str, int]:
    repo_root = Path(repo_root).resolve()
    policy_path = repo_root / "config/v4_12_review_collection_policy.json"
    if (repo_root / "config").is_symlink() or policy_path.is_symlink() or not policy_path.is_file():
        raise ValueError("collection policy path is unsafe")
    policy_raw = policy_path.read_bytes()
    if sha256_bytes(policy_raw) != POLICY_SHA256:
        raise ValueError("collection policy trust-anchor mismatch")
    policy = json.loads(policy_raw)
    if policy.get("schema_version") != "sireto-v4.12-r30-collection-policy-1":
        raise ValueError("collection policy schema mismatch")
    expected_versions = policy["parser_runtime"]
    observed_versions = {
        "lxml_version": importlib.metadata.version("lxml"),
        "idna_version": importlib.metadata.version("idna"),
        "pypdfium2_version": importlib.metadata.version("pypdfium2"),
    }
    for key, observed in observed_versions.items():
        if observed != expected_versions[key]:
            raise ValueError(f"runtime version mismatch: {key}")
    pinned_paths: list[tuple[str, str]] = []
    def collect_pins(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("_path") and isinstance(item, str):
                    hash_key = key[:-5] + "_sha256"
                    if hash_key in value:
                        pinned_paths.append((item, value[hash_key]))
                collect_pins(item)
        elif isinstance(value, list):
            for item in value:
                collect_pins(item)
    collect_pins(policy)
    relative_pins = [(item, digest) for item, digest in pinned_paths if not Path(item).is_absolute()]
    absolute_pins = [(item, digest) for item, digest in pinned_paths if Path(item).is_absolute()]
    if {item for item, _ in relative_pins} != ALLOWED_RELATIVE_PINS or any(
        ".." in Path(item).parts for item, _ in relative_pins
    ):
        raise ValueError("relative policy path is not allowlisted")
    if (
        {item for item, _ in absolute_pins} != ALLOWED_ABSOLUTE_PINS
        or len(absolute_pins) != len(ALLOWED_ABSOLUTE_PINS)
    ):
        raise ValueError("absolute policy path is not allowlisted")
    checked_pins = 0
    for item, digest in relative_pins:
        raw_path = repo_root / item
        current = repo_root
        unsafe_link = False
        for component in Path(item).parts:
            current = current / component
            if current.is_symlink():
                unsafe_link = True
                break
        path = raw_path.resolve()
        if unsafe_link or repo_root not in path.parents or not path.is_file():
            raise ValueError(f"unsafe pinned policy path: {item}")
        if sha256_bytes(path.read_bytes()) != digest:
            raise ValueError(f"hash mismatch for {item}")
        checked_pins += 1
    for item, digest in absolute_pins:
        try:
            observed_digest = _sha256_same_fd(Path(item))
        except OSError as exc:
            raise ValueError(f"unsafe or missing absolute policy pin: {item}") from exc
        if observed_digest != digest:
            raise ValueError(f"hash mismatch for {item}")
        checked_pins += 1
    if checked_pins != len(ALLOWED_RELATIVE_PINS) + len(ALLOWED_ABSOLUTE_PINS):
        raise ValueError("policy pin set is incomplete")

    dns = json.loads((repo_root / policy["network_security"]["dns_security_vectors_path"]).read_text())
    networks = [ipaddress.ip_network(value, strict=True) for value in dns["forbidden_cidrs"]]
    for vector in dns["address_cases"]:
        if forbidden_address(vector["address"], networks) != vector["forbidden"]:
            raise ValueError(f"DNS address vector failed: {vector}")
    for vector in dns["resolution_cases"]:
        observed = evaluate_resolution(vector["addresses"], networks)
        expected = {key: vector[key] for key in observed}
        if observed != expected:
            raise ValueError(f"DNS resolution vector failed: {vector['case_id']}")

    public_suffixes = [line.strip() for line in (repo_root / policy["domain_policy"]["public_suffixes_path"]).read_text().splitlines() if line.strip()]
    domains = json.loads((repo_root / policy["domain_policy"]["domain_vectors_path"]).read_text())
    for vector in domains:
        observed = evaluate_domain_vector(vector, policy, public_suffixes)
        expected = {key: vector[key] for key in observed}
        if observed != expected:
            raise ValueError(f"domain vector failed: {vector['input_hostname']}")

    identifiers = json.loads((repo_root / policy["domain_policy"]["identifier_vectors_path"]).read_text())
    for vector in identifiers:
        sirets, sirens = extract_identifiers(vector["text"])
        if sirets != vector["expected_sirets"] or sirens != vector["expected_sirens"]:
            raise ValueError(f"identifier vector failed: {vector['case_id']}")

    parser_policy = policy["search_engine"]
    fixture = (repo_root / parser_policy["parser_fixture_path"]).read_bytes()
    observed_results = parse_ddg_results(fixture, parser_policy["parser_fixture_http_charset"])
    expected_results = json.loads((repo_root / parser_policy["parser_expected_path"]).read_text())
    if observed_results != expected_results:
        raise ValueError("DDG parser fixture failed")
    charsets = json.loads((repo_root / parser_policy["charset_vectors_path"]).read_text())
    for vector in charsets:
        results = parse_ddg_results(base64.b64decode(vector["body_base64"]), vector["http_charset"])
        if not results or results[0]["title"] != vector["expected_title"]:
            raise ValueError(f"DDG charset vector failed: {vector['case_id']}")

    postopen = json.loads((repo_root / policy["domain_policy"]["postopen_validation_vectors_path"]).read_text())
    for vector in postopen:
        family, eligible = postopen_family(vector, policy)
        if family != vector["expected_family"] or eligible != vector["facts_eligible"]:
            raise ValueError(f"postopen vector failed: {vector['case_id']}")

    facts = json.loads((repo_root / policy["domain_policy"]["fact_reconstruction_vectors_path"]).read_text())
    for vector in facts:
        text = "".join(segment.get("literal", segment.get("repeat", "") * segment.get("count", 0)) for segment in vector["text_segments"])
        payload = text.encode("utf-8")
        if sha256_bytes(payload) != vector["expanded_text_sha256"]:
            raise ValueError(f"fact text hash failed: {vector['case_id']}")
        observed_facts, observed_unqualified = reconstruct_facts(
            text,
            vector["crm_name"],
            vector["crm_address"],
            vector["crm_postcode"],
            stopwords=frozenset(policy["domain_policy"]["name_stopwords"]),
            minimum_name_token_length=policy["domain_policy"]["significant_token_minimum_length"],
        )
        if observed_facts != vector["expected_facts"]:
            raise ValueError(f"fact reconstruction failed: {vector['case_id']}")
        if observed_unqualified != vector["expected_unqualified_sirets"]:
            raise ValueError(f"unqualified SIRETs failed: {vector['case_id']}")

    evidence = json.loads((repo_root / policy["domain_policy"]["evidence_vectors_path"]).read_text())
    for case in evidence["cases"]:
        if replay_evidence_case(case) != case["expected_evidence"]:
            raise ValueError(f"evidence vector failed: {case['case_id']}")

    adjudications = json.loads((repo_root / policy["domain_policy"]["adjudication_vectors_path"]).read_text())
    for vector in adjudications:
        observed = adjudicate_vector(vector)
        expected = {
            "status": vector["expected_status"],
            "reliable": vector["expected_reliable"],
            "group_count": vector["expected_group_count"],
            "alternative_siret": vector["expected_alternative_siret"],
            "alternative_in_top100": vector["expected_alternative_in_top100"],
        }
        if observed != expected:
            raise ValueError(f"adjudication vector failed: {vector['case_id']}")

    return {
        "pins": checked_pins,
        "dns_addresses": len(dns["address_cases"]),
        "dns_resolutions": len(dns["resolution_cases"]),
        "domains": len(domains),
        "identifiers": len(identifiers),
        "ddg_results": len(expected_results),
        "ddg_charsets": len(charsets),
        "postopen": len(postopen),
        "fact_cases": len(facts),
        "evidence_cases": len(evidence["cases"]),
        "adjudication_cases": len(adjudications),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    arguments = parser.parse_args()
    print(json.dumps(validate_policy_goldens(arguments.repo_root), sort_keys=True))


if __name__ == "__main__":
    main()
