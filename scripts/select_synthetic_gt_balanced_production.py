#!/usr/bin/env python3
"""Select a counted, balanced synthetic-GT production batch.

The selector never writes CRM text.  It chooses official SIRENE targets and
train-observed field operators, solves the requested easy/medium/hard mix, and
assigns train-OOF failure-directed strata.  Luna remains the sole CRM writer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402
from scripts import manage_synthetic_gt_balanced_registry as registry_lib  # noqa: E402
from scripts import select_synthetic_gt_fragment_pilot as fragments  # noqa: E402


SAFE_NAME_RELATIONS = (
    "LEGAL_FORM_REMOVE", "TOKEN_ORDER", "TOKEN_SUBSET", "PUNCTUATION_REMOVED",
    "OFFICIAL_NAME_ALIAS",
)
SAFE_LOCATION_RELATIONS = (
    ("address", "ADDRESS_ABBREVIATE"),
    ("address", "ADDRESS_ALIAS_EXPAND"),
    ("address", "ADDRESS_TOKEN_SUBSET"),
    ("address", "PUNCTUATION_REMOVED"),
    ("city", "PUNCTUATION_REMOVED"),
)
DIFFICULTIES = ("EASY", "MEDIUM", "HARD")
STRATA = (
    "FAIL_BOTH_MODELS", "FAIL_XGB_ONLY", "FAIL_BGE_ONLY",
    "TRAIN_DISTRIBUTION", "NEAR_CLEAN_CONTROL",
)
STRATUM_TO_OOF_CELL = {
    "FAIL_BOTH_MODELS": "BOTH_WRONG",
    # Direction names describe the failing model, not the model that is right.
    "FAIL_XGB_ONLY": "BGE_ONLY_CORRECT",
    "FAIL_BGE_ONLY": "XGB_ONLY_CORRECT",
    "TRAIN_DISTRIBUTION": "BOTH_CORRECT",
}
TOKEN_SUBSET_FUNCTION_WORDS = {
    "a", "au", "aux", "d", "de", "des", "du", "en", "et", "l", "la", "le",
    "les", "par", "pour", "sous", "sur", "avec", "sans", "chez", "que", "qu",
    "qui", "y",
}
TOKEN_SUBSET_GENERIC_WORDS = {
    "service", "services", "conseil", "consulting", "management", "holding",
    "commerce", "distribution", "industrie", "batiment", "btp", "travaux",
    "transport", "transports", "location", "immobilier", "hotel", "garage",
    "restaurant", "auto", "autos", "electricite", "soldes",
}
TOKEN_SUBSET_ORG_DESCRIPTORS = {
    "association", "entreprise", "societe", "groupe", "hotel", "garage", "centre",
    "cabinet", "clinique", "ecole", "college", "lycee", "restaurant", "transports",
}
CONTEXT_INDEX_SCHEMA = "sireto-balanced-selector-context-index-1"


def sha256(path: Path) -> str:
    return fragments.sha256(path)


def rows(path: Path) -> list[dict[str, Any]]:
    return fragments.rows(path)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def exact_counts(total: int, shares: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in shares.items()}
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(shares, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def identity_share_coefficients(
    bounds: Sequence[float],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return exact integer coefficients for active-identity share bounds."""
    if len(bounds) != 2:
        raise ValueError("target identity share bounds must contain two values")
    lower, upper = (Fraction(str(value)) for value in bounds)
    if not 0 < lower <= upper < 1:
        raise ValueError("invalid target identity active-share bounds")

    def coefficients(value: Fraction) -> tuple[int, int]:
        return value.denominator - value.numerator, -value.numerator

    return coefficients(lower), coefficients(upper)


def remaining_quota_counts(
    new_count: int,
    final_total: int,
    shares: dict[str, float],
    prior_counts: Counter[str] | dict[str, int],
) -> dict[str, int]:
    """Allocate this batch against the remaining *global* corpus quotas."""
    if new_count < 0 or final_total <= 0:
        raise ValueError("invalid quota totals")
    final_counts = exact_counts(final_total, shares)
    remaining = {
        key: max(0, final_counts[key] - int(prior_counts.get(key, 0)))
        for key in shares
    }
    if new_count > sum(remaining.values()):
        raise ValueError(
            f"batch requests {new_count} variants but only {sum(remaining.values())} "
            "global quota slots remain"
        )
    if not new_count:
        return {key: 0 for key in shares}
    remaining_total = sum(remaining.values())
    allocation = exact_counts(
        new_count,
        {key: value / remaining_total for key, value in remaining.items()},
    )
    # Largest-remainder rounding cannot exceed a positive remaining cell when
    # new_count <= remaining_total, except for a zero-share cell (which exact_counts
    # keeps at zero).  Retain an explicit invariant because this gates production.
    if any(allocation[key] > remaining[key] for key in allocation):
        raise RuntimeError("global quota allocation exceeded a remaining cell")
    return allocation


def adapt_strata_to_easy_capacity(
    ideal_counts: dict[str, int],
    maximum_additions: dict[str, int],
    easy_capacity: int,
) -> dict[str, int]:
    """Defer near-clean rows that cannot be hosted by this batch's EASY supply."""
    result = dict(ideal_counts)
    control = "NEAR_CLEAN_CONTROL"
    retained_control = min(result[control], easy_capacity)
    deficit = result[control] - retained_control
    result[control] = retained_control
    if not deficit:
        return result
    spare = {
        key: max(0, maximum_additions[key] - result[key])
        for key in result if key != control
    }
    if sum(spare.values()) < deficit:
        raise ValueError("final augmentation-stratum remainders cannot absorb control deficit")
    additions = exact_counts(
        deficit, {key: value / sum(spare.values()) for key, value in spare.items()}
    )
    if any(additions[key] > spare[key] for key in additions):
        raise RuntimeError("stratum redistribution exceeded a final remainder")
    for key, value in additions.items():
        result[key] += value
    return result


def maximum_batch_target_additions(
    prior_variants: int,
    prior_targets: int,
    batch_variants: int,
    final_variants: int,
    maximum_targets: int,
    maximum_variants_per_target: int,
) -> int:
    """Reserve enough unique-target slots for the exact corpus residual."""
    if (
        min(prior_variants, prior_targets, batch_variants) < 0
        or final_variants <= 0
        or maximum_targets <= 0
        or maximum_variants_per_target <= 0
        or prior_variants + batch_variants > final_variants
    ):
        raise ValueError("invalid unique-target budget inputs")
    future_minimum_targets = math.ceil(
        (final_variants - prior_variants - batch_variants)
        / maximum_variants_per_target
    )
    return maximum_targets - prior_targets - future_minimum_targets


def registry_usage(
    path: Path,
) -> tuple[
    int, Counter[str], Counter[str], Counter[str], Counter[str], Counter[str],
    Counter[str], int, set[str], set[str], str,
]:
    production_registry = registry_lib.load_registry(path)
    snapshot = registry_lib.snapshot(production_registry)
    if production_registry.get("summary") != snapshot:
        raise ValueError("production registry summary is not sealed")
    return (
        int(snapshot["promoted_variants"]),
        Counter(snapshot["inspiration_ref_counts"]),
        Counter(snapshot["exact_operator_counts"]),
        Counter(snapshot["relation_pair_counts"]),
        Counter(snapshot["difficulty_counts"]),
        Counter(snapshot["augmentation_stratum_counts"]),
        Counter(snapshot["name_token_subset_signature_counts"]),
        int(snapshot["distinct_target_sirets"]),
        set(snapshot["excluded_target_sirets"]),
        set(snapshot["excluded_target_sirens"]),
        sha256(path),
    )


def seed_target_states(path: Path) -> dict[str, str]:
    """Read the frozen official state once per target from a seed input."""
    result: dict[str, str] = {}
    for value in rows(path):
        siret = str(value.get("target_siret", ""))
        state = str(
            value.get("seed_card", {}).get("official_context", {})
            .get("target", {}).get("state", "")
        )
        if not loop.valid_siret(siret) or state not in {"A", "F"}:
            raise ValueError(f"invalid frozen target state in {path}: {siret}/{state}")
        if siret in result and result[siret] != state:
            raise ValueError(f"inconsistent frozen target state in {path}: {siret}")
        result[siret] = state
    return result


def seed_state_usage(paths: Sequence[Path]) -> tuple[Counter[str], Counter[str]]:
    variants: Counter[str] = Counter()
    target_states: dict[str, str] = {}
    for path in paths:
        states = seed_target_states(path)
        for value in rows(path):
            siret = str(value["target_siret"])
            variants[states[siret]] += len(
                value.get("seed_card", {}).get("composite_contracts", [])
            )
        for siret, state in states.items():
            if siret in target_states and target_states[siret] != state:
                raise ValueError(f"inconsistent prior target state: {siret}")
            target_states[siret] = state
    return variants, Counter(target_states.values())


def registry_state_usage(path: Path) -> tuple[Counter[str], Counter[str]]:
    """Count only exact promoted states without changing the sealed snapshot."""
    registry = registry_lib.load_registry(path)
    variants: Counter[str] = Counter()
    target_states: dict[str, str] = {}
    for batch in registry["batches"]:
        seed = Path(batch["seed_input"]["path"])
        promoted = Path(batch["promoted"]["path"])
        manifest = Path(batch["promotion_manifest"]["path"])
        states = seed_target_states(seed)
        validated = registry_lib.validate_promoted_batch(seed, promoted, manifest)
        for value in validated["records"]:
            siret = value["target_siret"]
            state = states[siret]
            variants[state] += 1
            if siret in target_states and target_states[siret] != state:
                raise ValueError(f"inconsistent registered target state: {siret}")
            target_states[siret] = state
    return variants, Counter(target_states.values())


def pair_signature(pair: tuple[str, tuple[str, str]]) -> str:
    name_relation, (field, location_relation) = pair
    return f"name:{name_relation}+{field}:{location_relation}"


def production_usage(
    seed_inputs: Sequence[Path],
) -> tuple[int, Counter[str], Counter[str], Counter[str]]:
    """Reserve cumulative corpus caps from already-counted production inputs."""
    variant_count = 0
    ref_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    for path in seed_inputs:
        for row in rows(path):
            for contract in row.get("seed_card", {}).get("composite_contracts", []):
                relations = contract.get("field_relations", {})
                location = [
                    (field, relation) for field, relation in relations.items()
                    if field != "name"
                ]
                if "name" not in relations or len(location) != 1:
                    raise ValueError(f"invalid counted production contract: {path}")
                variant_count += 1
                pair_counts[pair_signature((relations["name"], location[0]))] += 1
                for fragment in contract.get("field_inspirations", {}).values():
                    ref_counts[fragment["inspiration_ref"]] += 1
                    operator_counts[fragment_operator(fragment)] += 1
    return variant_count, ref_counts, operator_counts, pair_counts


def context_flags(context: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    if context["target"]["state"] == "F":
        flags.add("CLOSED_TARGET")
    internal = context.get("internal_context", [])
    if any("SAME_SIREN" in value.get("relation_tags", []) for value in internal):
        flags.add("SAME_SIREN_COMPETITION")
    if any("SAME_OFFICIAL_ADDRESS" in value.get("relation_tags", []) for value in internal):
        flags.add("SAME_ADDRESS_COMPETITION")
    if len(internal) >= 20:
        flags.add("DENSE_CANDIDATE_SCENE")
    return flags


def variant_archetypes(
    context: dict[str, Any], pair: tuple[str, tuple[str, str]]
) -> set[str]:
    flags = context_flags(context)
    name_relation, (_field, location_relation) = pair
    name_weak = name_relation in {
        "TOKEN_ORDER", "TOKEN_SUBSET", "PUNCTUATION_REMOVED",
    }
    address_weak = location_relation in {"ADDRESS_TOKEN_SUBSET", "PUNCTUATION_REMOVED"}
    if name_weak and address_weak:
        flags.add("WEAK_NAME_AND_ADDRESS")
    elif name_weak:
        flags.add("WEAK_NAME_STRONG_ADDRESS")
    elif address_weak:
        flags.add("STRONG_NAME_WEAK_ADDRESS")
    return flags


def difficulty(
    context: dict[str, Any], pair: tuple[str, tuple[str, str]]
) -> str:
    name_relation, (_field, location_relation) = pair
    flags = context_flags(context)
    score = 0
    if "CLOSED_TARGET" in flags:
        score += 1
    if "SAME_SIREN_COMPETITION" in flags:
        score += 1
    if "SAME_ADDRESS_COMPETITION" in flags:
        score += 1
    if name_relation in {"TOKEN_ORDER", "TOKEN_SUBSET"}:
        score += 1
    if location_relation == "ADDRESS_TOKEN_SUBSET":
        score += 1
    if score == 0:
        return "EASY"
    if score >= 3:
        return "HARD"
    return "MEDIUM"


def runtime_stable_punctuation_fragment(
    source: str,
    fragment: dict[str, Any],
) -> bool:
    """Reject no-space joins that the frozen runtime cannot re-identify.

    ``PUNCTUATION_REMOVED`` records the original lexical boundary.  Removing
    an internal mark without a space also removes one target-token boundary.
    When more than one source token follows that join, the runtime's target
    boundary index necessarily shifts: an exact surface such as
    ``JEAN-MARC MERMET -> JEANMARC MERMET`` is reconstructed as a space edit
    at boundary zero and can never satisfy the frozen parameters.  A join of
    the final lexical pair remains stable because the shifted boundary is the
    target suffix.  Leading/trailing mark removal and space-preserving edits
    are stable as well.

    This is selector-only and fail-closed.  It neither writes CRM text nor
    relaxes runtime validation; explicit joined groups would require a future
    contract version.
    """
    if fragment.get("relation") != "PUNCTUATION_REMOVED":
        return True
    word_count = len(loop.normalized_words(source))
    edits = fragment.get("operation_parameters", {}).get("edits", [])
    if not isinstance(edits, list):
        return False
    for edit in edits:
        try:
            boundary = int(edit.get("after_token_index", -99))
        except (AttributeError, TypeError, ValueError):
            return False
        if edit.get("replacement") == "" and 0 <= boundary < word_count - 2:
            return False
    return True


def safe_capabilities(
    context: dict[str, Any],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    document_frequencies: Counter[str],
    allowed_name_relations: Sequence[str] = SAFE_NAME_RELATIONS,
    support_cache: dict[tuple[Any, ...], list[dict[str, Any]]] | None = None,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, str], list[dict[str, Any]]],
]:
    values = fragments.baseline(context)
    anchors = fragments.distinctive_name_tokens(context, document_frequencies)
    cache = support_cache if support_cache is not None else {}

    def supported(field: str, relation: str) -> list[dict[str, Any]]:
        source = values[field]
        key = (
            field, relation, source,
            tuple(anchors) if field == "name" else (),
        )
        if key in cache:
            return cache[key]
        if relation == "PUNCTUATION_REMOVED" and not any(
            character in loop.PUNCTUATION for character in source
        ):
            cache[key] = []
            return []
        source_count = len(loop.normalized_words(source))
        possible = [
            value for value in grouped.get((field, relation), [])
            if value.get("operation_parameters", {}).get("source_token_count", source_count)
            == source_count
            and runtime_stable_punctuation_fragment(source, value)
        ]
        result = [
            value for value in possible
            if fragments.fragment_supports(field, source, value, anchors)
        ]
        cache[key] = result
        return result

    names = {}
    for relation in allowed_name_relations:
        if relation == "OFFICIAL_NAME_ALIAS":
            continue
        candidates = supported("name", relation)
        if relation == "TOKEN_SUBSET":
            candidates = [
                value for value in candidates
                if strict_token_subset_anchor(context, value, document_frequencies)
            ]
        names[relation] = distinct_exact_operators(candidates)
    if "OFFICIAL_NAME_ALIAS" in allowed_name_relations:
        names["OFFICIAL_NAME_ALIAS"] = official_name_alias_fragments(context)
    locations = {
        key: distinct_exact_operators(supported(*key))
        for key in SAFE_LOCATION_RELATIONS
    }
    return names, locations


def materializable_relation_pair_count(
    names: dict[str, list[dict[str, Any]]],
    locations: dict[tuple[str, str], list[dict[str, Any]]],
) -> int:
    return sum(bool(value) for value in names.values()) * sum(
        bool(value) for value in locations.values()
    )


def distinct_exact_operators(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one deterministic proof per materialized exact operator."""
    by_operator: dict[str, dict[str, Any]] = {}
    for value in sorted(values, key=lambda item: item["inspiration_ref"]):
        by_operator.setdefault(fragment_operator(value), value)
    return list(by_operator.values())


def official_enseigne_tokens(context: dict[str, Any]) -> set[str]:
    return {
        token
        for option in context.get("target", {}).get("names", [])
        if option.get("kind") == "ENSEIGNE"
        for token in loop.normalized_words(option.get("value", ""))
    }


def strict_token_subset_anchor(
    context: dict[str, Any],
    fragment: dict[str, Any],
    document_frequencies: Counter[str],
) -> str | None:
    """Return the audited protected anchor, or reject this TOKEN_SUBSET."""
    source = fragments.baseline(context)["name"]
    words = loop.normalized_words(source)
    if not (4 <= len(words) <= 10) or any(
        character in loop.PUNCTUATION for character in source
    ):
        return None
    legal = {value.casefold() for value in loop.LEGAL_FORM_TOKENS}
    enseigne_tokens = official_enseigne_tokens(context)
    has_distinct_enseigne = any(
        loop.normalized_surface(option.get("value", ""))
        != loop.normalized_surface(source)
        for option in context.get("target", {}).get("names", [])
        if option.get("kind") == "ENSEIGNE" and str(option.get("value", "")).strip()
    )
    person_like = (
        not any(word in legal for word in words)
        and not has_distinct_enseigne
        and not any(word in TOKEN_SUBSET_ORG_DESCRIPTORS for word in words)
    )
    if person_like:
        return None
    retained = fragment.get("operation_parameters", {}).get("retained_positions")
    if not isinstance(retained, list) or retained != sorted(set(retained)):
        return None
    retained_set = set(retained)
    removed = [index for index in range(len(words)) if index not in retained_set]
    if (
        not (1 <= len(removed) <= 2)
        or removed != list(range(removed[0], removed[-1] + 1))
        or len(retained) < max(3, math.ceil(0.60 * len(words)))
        or all(words[index] in legal for index in removed)
    ):
        return None
    projected = [words[index] for index in retained]
    if not projected:
        return None
    if projected[0] in TOKEN_SUBSET_FUNCTION_WORDS and projected[0] not in legal:
        return None
    if projected[-1] in TOKEN_SUBSET_FUNCTION_WORDS and projected[-1] not in legal:
        return None
    left, right = removed[0] - 1, removed[-1] + 1
    if left in retained_set and right in retained_set and (
        words[left] in TOKEN_SUBSET_FUNCTION_WORDS
        or words[right] in TOKEN_SUBSET_FUNCTION_WORDS
    ):
        return None
    address_city_tokens = set(loop.normalized_words(
        f"{fragments.baseline(context)['address']} {fragments.baseline(context)['city']}"
    ))
    candidates = [
        word for index, word in enumerate(words)
        if index in retained_set
        and len(word) >= 3
        and not word.isdigit()
        and fragments.ROMAN_NUMERAL.fullmatch(word) is None
        and word not in legal
        and word.upper() not in fragments.STOP_ANCHORS
        and word not in TOKEN_SUBSET_GENERIC_WORDS
        and word not in TOKEN_SUBSET_FUNCTION_WORDS
        and word not in address_city_tokens
        and (document_frequencies.get(word, 0) <= 32 or word in enseigne_tokens)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda word: (document_frequencies.get(word, 0), -len(word), words.index(word)),
    )


def eligible_for_official_alias(context: dict[str, Any]) -> bool:
    qualification = context.get("qualification", {})
    address = context.get("target", {}).get("address", {})
    complete_context = all(
        qualification.get(key)
        for key in (
            "pre_generation_exact_eligible", "siblings_complete",
            "same_address_complete", "same_name_geography_complete",
        )
    )
    address_complete = all(
        str(address.get(key) or "").strip()
        for key in ("number", "street_type", "street", "postcode", "insee", "city")
    )
    return bool(
        complete_context and address_complete and official_name_alias_fragments(context)
    )


def official_name_alias_fragments(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Build target-specific evidence from distinct official SIRENE name options.

    This is not synthetic text and is not borrowed from a protected fold.  Luna
    must still emit the selected official alias byte-for-byte; the fragment only
    freezes the authoritative relationship and its provenance.
    """
    source = fragments.baseline(context)["name"]
    baseline_values = fragments.baseline(context)
    source_fingerprint = loop.comparison_fingerprint(baseline_values)
    seen_fingerprints = {source_fingerprint}
    result: list[dict[str, Any]] = []
    for option in context.get("target", {}).get("names", []):
        alias = str(option.get("value", "")).strip()
        if not alias:
            continue
        alias_values = {**baseline_values, "name": alias}
        alias_fingerprint = loop.comparison_fingerprint(alias_values)
        if alias_fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(alias_fingerprint)
        alias_sha = hashlib.sha256(alias.encode("utf-8")).hexdigest()
        parameters = loop.official_name_alias_parameters(alias)
        inspiration_ref = loop.digest_json({
            "evidence_source_type": "SIRENE_OFFICIAL_NAME_OPTION",
            "target_siret": context["target_siret"],
            "official_value": source,
            "observed_crm_value": alias,
            "operation_parameters": parameters,
        })
        result.append({
            "schema_version": "sireto-synthetic-field-inspiration-1",
            "field": "name",
            "relation": "OFFICIAL_NAME_ALIAS",
            "inspiration_ref": inspiration_ref,
            "official_value": source,
            "observed_crm_value": alias,
            "operation_parameters": parameters,
            "evidence_source_type": "SIRENE_OFFICIAL_NAME_OPTION",
            "source_fold": -1,
            "source_legacy_split": "sirene_official",
            "source_state": context["target"]["state"],
            "provenance_digest": loop.digest_json({
                "context_sha256": context["context_sha256"],
                "target_siret": context["target_siret"],
                "name_kind": option.get("kind"),
                "alias_sha256": alias_sha,
            }),
        })
    return sorted(result, key=lambda value: value["inspiration_ref"])


def context_index_cache_key(source_sha256: str, source_size: int) -> str:
    """Address an index by both its official input and eligibility semantics."""
    semantic_sources = [
        Path(__file__),
        Path(fragments.__file__),
        Path(fragments.legacy.__file__),
        Path(loop.__file__),
    ]
    return canonical_digest({
        "schema_version": CONTEXT_INDEX_SCHEMA,
        "source_sha256": source_sha256,
        "source_size": source_size,
        "semantic_source_hashes": {
            path.name: sha256(path) for path in semantic_sources
        },
    })


def context_index_content_digest(
    entries: Sequence[dict[str, Any]], document_frequencies: Counter[str],
) -> str:
    return canonical_digest({
        "entries": entries,
        "document_frequencies": dict(sorted(document_frequencies.items())),
    })


def build_context_index(
    source: Path, source_sha256: str, cache_directory: Path,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Build a sealed byte-offset index without retaining the 2.4 GiB source.

    Eligibility and global token document frequencies are evaluated by the
    exact production functions.  Only immutable selection metadata is cached;
    full contexts are re-read byte-for-byte for the deterministic capability
    pool.
    """
    source_size = source.stat().st_size
    cache_key = context_index_cache_key(source_sha256, source_size)
    cache_path = cache_directory / f"{cache_key}.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            entries = payload["entries"]
            document_frequencies = Counter(payload["document_frequencies"])
            if (
                payload.get("schema_version") == CONTEXT_INDEX_SCHEMA
                and payload.get("cache_key") == cache_key
                and payload.get("source_sha256") == source_sha256
                and payload.get("source_size") == source_size
                and payload.get("content_sha256") == context_index_content_digest(
                    entries, document_frequencies,
                )
            ):
                return entries, document_frequencies
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            # A partial/stale derived artifact is never authoritative. Rebuild
            # it from the already hash-checked official source.
            pass

    entries: list[dict[str, Any]] = []
    document_frequencies: Counter[str] = Counter()
    with source.open("rb") as stream:
        source_index = 0
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            context = json.loads(raw)
            alias_available = bool(official_name_alias_fragments(context))
            document_frequencies.update(set(loop.normalized_words(
                fragments.legacy.primary_name(context)
            )))
            entries.append({
                "source_index": source_index,
                "offset": offset,
                "length": len(raw),
                "target_siret": context["target_siret"],
                "target_siren": context["target_siren"],
                "state": context["target"]["state"],
                "regular_eligible": bool(fragments.eligible(context)),
                "alias_eligible": bool(
                    alias_available and eligible_for_official_alias(context)
                ),
                "alias_available": alias_available,
                "structural_easy_priority": int(
                    context["target"]["state"] == "A" and not context_flags(context)
                ) * 3,
            })
            source_index += 1
    payload = {
        "schema_version": CONTEXT_INDEX_SCHEMA,
        "cache_key": cache_key,
        "source_sha256": source_sha256,
        "source_size": source_size,
        "entries": entries,
        "document_frequencies": dict(sorted(document_frequencies.items())),
        "content_sha256": context_index_content_digest(
            entries, document_frequencies,
        ),
    }
    cache_directory.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, cache_path)
    return entries, document_frequencies


def indexed_context_stub(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_siret": entry["target_siret"],
        "target_siren": entry["target_siren"],
        "target": {"state": entry["state"]},
        "_context_source_index": entry["source_index"],
        "_context_offset": entry["offset"],
        "_context_length": entry["length"],
    }


def read_indexed_contexts(
    source: Path, selected_stubs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read selected contexts in source order, then restore pool order."""
    requested = {
        int(value["_context_source_index"]): value for value in selected_stubs
    }
    loaded: dict[int, dict[str, Any]] = {}
    with source.open("rb") as stream:
        for source_index, value in sorted(requested.items()):
            stream.seek(int(value["_context_offset"]))
            raw = stream.read(int(value["_context_length"]))
            context = json.loads(raw)
            if (
                context.get("target_siret") != value["target_siret"]
                or context.get("target_siren") != value["target_siren"]
            ):
                raise ValueError("official-context index identity mismatch")
            loaded[source_index] = context
    return [loaded[int(value["_context_source_index"])] for value in selected_stubs]


def candidate_bundles(
    context: dict[str, Any],
    names: dict[str, list[dict[str, Any]]],
    locations: dict[tuple[str, str], list[dict[str, Any]]],
    selection_seed: str,
    limit: int = 64,
    pair_remaining: dict[tuple[str, tuple[str, str]], int] | None = None,
    official_name_alias_remaining: int | None = None,
) -> list[tuple[tuple[str, tuple[str, str]], ...]]:
    pairs = [
        (name_relation, location_key)
        for name_relation, name_values in names.items() if name_values
        for location_key, location_values in locations.items() if location_values
        if pair_remaining is None
        or pair_remaining.get((name_relation, location_key), 0) > 0
    ]
    if not pairs:
        return []
    # The runtime contract is one to three variants per target.  Keeping only
    # triples strands clean controls late in production when a target has just
    # one or two distinct safe location operators.  Model all legal sizes and
    # let the corpus-level MILP preserve the exact number of variants.
    values = [
        bundle
        for size in (1, 2, 3)
        for bundle in itertools.combinations_with_replacement(pairs, size)
    ]
    # Validate the complete local bundle space before deterministic thinning.
    # The corpus MILP constrains more than difficulty: alias-family and exact
    # relation-pair caps are cumulative.  Thinning only by E/M/H can erase all
    # non-alias or all residual support for a pair and create false global
    # infeasibility.  The feature round-robin below preserves every local
    # (difficulty, alias-count, exact-pair, multiplicity) marginal.
    valid: list[tuple[tuple[str, tuple[str, str]], ...]] = []
    for bundle in values:
        if (
            official_name_alias_remaining is not None
            and sum(pair[0] == "OFFICIAL_NAME_ALIAS" for pair in bundle)
            > official_name_alias_remaining
        ):
            continue
        if pair_remaining is not None and any(
            count > pair_remaining.get(pair, 0)
            for pair, count in Counter(bundle).items()
        ):
            continue
        name_counts = Counter(pair[0] for pair in bundle)
        location_counts = Counter(pair[1] for pair in bundle)
        if name_counts["TOKEN_SUBSET"] > 1:
            continue
        if any(
            count > (
                3 * len({value["inspiration_ref"] for value in names[relation]})
                if relation == "OFFICIAL_NAME_ALIAS"
                else len({value["inspiration_ref"] for value in names[relation]})
            )
            for relation, count in name_counts.items()
        ) or any(
            count > len({value["inspiration_ref"] for value in locations[key]})
            for key, count in location_counts.items()
        ):
            continue
        counts = Counter(difficulty(context, pair) for pair in bundle)
        valid.append(bundle)
    feature_groups: dict[tuple[Any, ...], list[Any]] = defaultdict(list)
    for bundle in valid:
        counts = Counter(difficulty(context, pair) for pair in bundle)
        profile = tuple(counts[value] for value in DIFFICULTIES)
        alias_count = sum(pair[0] == "OFFICIAL_NAME_ALIAS" for pair in bundle)
        feature_groups[("PROFILE_ALIAS", profile, alias_count)].append(bundle)
        for pair, multiplicity in Counter(bundle).items():
            feature_groups[(
                "PAIR_SUPPORT", profile, alias_count,
                pair_signature(pair), multiplicity,
            )].append(bundle)
    ordered_groups: dict[tuple[Any, ...], list[Any]] = {}
    for feature in sorted(feature_groups):
        ordered_groups[feature] = sorted(
            feature_groups[feature],
            key=lambda bundle: hashlib.sha256(
                f"{selection_seed}|bundle|{context['target_siret']}|"
                f"{'|'.join(pair_signature(value) for value in bundle)}".encode()
            ).hexdigest(),
        )
    selected: list[Any] = []
    selected_keys: set[Any] = set()
    for depth in range(4):
        for feature in sorted(ordered_groups):
            candidates = ordered_groups[feature]
            if depth < len(candidates):
                bundle = candidates[depth]
                if bundle in selected_keys:
                    continue
                selected.append(bundle)
                selected_keys.add(bundle)
                if len(selected) >= limit:
                    return selected
    return selected


def stratified_context_pool(
    contexts: Sequence[dict[str, Any]],
    limit: int,
    selection_seed: str,
    alias_available: dict[str, bool],
    easy_capacity: dict[str, int] | None = None,
    active_share: float = 0.5,
) -> list[dict[str, Any]]:
    """Preserve state and alias/non-alias coverage before deterministic thinning.

    ``active_share`` only controls the candidate-computation pool.  The MILP
    keeps the output state mix exact.  A larger active share is useful there
    because EASY is structurally impossible for closed/same-site challenge
    scenes, while a large residual closed reserve remains sufficient.
    """
    if limit <= 0:
        return []
    if not 0.0 <= active_share <= 1.0:
        raise ValueError("active_share must be between zero and one")
    easy_capacity = easy_capacity or {}

    def order(value: dict[str, Any]) -> tuple[int, str]:
        siret = value["target_siret"]
        return (
            -int(easy_capacity.get(siret, 0)),
            hashlib.sha256(
                f"{selection_seed}|stratified-pool|{siret}".encode()
            ).hexdigest(),
        )

    result: list[dict[str, Any]] = []
    active_limit = int(round(limit * active_share))
    per_state = {"A": active_limit, "F": limit - active_limit}
    for state in ("A", "F"):
        state_values = [
            value for value in contexts if value["target"]["state"] == state
        ]
        state_limit = min(per_state[state], len(state_values))
        groups = {
            flag: sorted(
                [
                    value for value in state_values
                    if bool(alias_available.get(value["target_siret"], False)) == flag
                ],
                key=order,
            )
            for flag in (True, False)
        }
        quotas = {True: state_limit // 2, False: state_limit - state_limit // 2}
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        for flag in (True, False):
            for value in groups[flag][:quotas[flag]]:
                selected.append(value)
                selected_ids.add(value["target_siret"])
        if len(selected) < state_limit:
            remainder = sorted(
                [
                    value for value in state_values
                    if value["target_siret"] not in selected_ids
                ],
                key=order,
            )
            selected.extend(remainder[:state_limit - len(selected)])
        result.extend(selected)
    if len(result) < min(limit, len(contexts)):
        selected_ids = {value["target_siret"] for value in result}
        remainder = sorted(
            [value for value in contexts if value["target_siret"] not in selected_ids],
            key=order,
        )
        result.extend(remainder[:limit - len(result)])
    return result[:limit]


def choose_targets_and_bundles(
    contexts: list[dict[str, Any]],
    capabilities: dict[str, tuple[dict[str, Any], dict[tuple[str, str], Any]]],
    variant_count: int,
    maximum_target_count: int,
    target_active_share_bounds: tuple[float, float],
    prior_state_variant_counts: Counter[str],
    prior_state_target_counts: Counter[str],
    state_variant_additions: dict[str, int],
    future_state_variant_counts: dict[str, int],
    difficulty_counts: dict[str, int],
    difficulty_minimum_additions: dict[str, int],
    difficulty_maximum_additions: dict[str, int],
    relation_pair_cap: int,
    prior_pair_counts: Counter[str],
    relation_capacities: dict[tuple[str, str], int],
    official_name_alias_remaining: int,
    name_token_subset_remaining: int,
    ref_cap: int,
    operator_cap: int,
    prior_ref_counts: Counter[str],
    prior_operator_counts: Counter[str],
    effective_ref_caps: dict[str, int],
    token_subset_signature_cap: int,
    prior_token_subset_signature_counts: Counter[str],
    selection_seed: str,
    diagnose_infeasible: bool = False,
    feasibility_only: bool = False,
    solver_pool_limit: int | None = None,
) -> list[tuple[dict[str, Any], tuple[tuple[str, tuple[str, str]], ...]]]:
    options: list[tuple[int, tuple[tuple[str, tuple[str, str]], ...]]] = []
    context_by_index: list[dict[str, Any]] = []
    pair_remaining = {
        (name_relation, location_key): max(
            0,
            relation_pair_cap
            - prior_pair_counts[pair_signature((name_relation, location_key))],
        )
        for name_relation in SAFE_NAME_RELATIONS
        for location_key in SAFE_LOCATION_RELATIONS
    }
    # Authoritative aliases are comparatively rare and are the safe source of
    # easy capacity.  Once a composite pair reaches its final corpus cap it is
    # mathematically unusable: exclude it before pool ranking and bundle
    # thinning so saturated alias pairs cannot crowd out residual safe support.
    pool_limit = (
        solver_pool_limit
        if solver_pool_limit is not None and solver_pool_limit > 0
        else max(3000, variant_count)
    )
    alias_available = {
        value["target_siret"]: (
            official_name_alias_remaining > 0
            and
            bool(capabilities[value["target_siret"]][0].get(
                "OFFICIAL_NAME_ALIAS"
            ))
            and any(
                pair_remaining.get(("OFFICIAL_NAME_ALIAS", location_key), 0) > 0
                for location_key, location_values
                in capabilities[value["target_siret"]][1].items()
                if location_values
            )
        )
        for value in contexts
    }
    easy_capacity = {}
    for value in contexts:
        names, locations = capabilities[value["target_siret"]]
        easy_pairs = sum(
            pair_remaining.get((name_relation, location_key), 0) > 0
            and difficulty(value, (name_relation, location_key)) == "EASY"
            for name_relation, name_values in names.items() if name_values
            for location_key, location_values in locations.items() if location_values
        )
        easy_capacity[value["target_siret"]] = min(3, easy_pairs)
    eligible = stratified_context_pool(
        contexts, pool_limit, selection_seed, alias_available, easy_capacity,
        active_share=target_active_share_bounds[1],
    )
    for context in eligible:
        siret = context["target_siret"]
        names, locations = capabilities[siret]
        bundles = candidate_bundles(
            context, names, locations, selection_seed,
            pair_remaining=pair_remaining,
            official_name_alias_remaining=official_name_alias_remaining,
        )
        if not bundles:
            continue
        context_index = len(context_by_index)
        context_by_index.append(context)
        options.extend((context_index, bundle) for bundle in bundles)
    if not options:
        raise ValueError("no safe production bundles are feasible")

    variable_count = len(options)
    constraint_rows: list[dict[int, float]] = []
    minima: list[float] = []
    maxima: list[float] = []
    constraint_labels: list[str] = []

    def add(
        coefficients: dict[int, float], minimum: float, maximum: float, label: str,
    ) -> int:
        constraint_rows.append(coefficients)
        minima.append(minimum)
        maxima.append(maximum)
        constraint_labels.append(label)
        return len(constraint_rows) - 1

    add({
        index: len(bundle)
        for index, (_context_index, bundle) in enumerate(options)
    }, variant_count, variant_count, "TOTAL_VARIANTS")
    option_by_context: dict[int, list[int]] = defaultdict(list)
    for option_index, (context_index, _bundle) in enumerate(options):
        option_by_context[context_index].append(option_index)
    for indexes in option_by_context.values():
        add({index: 1 for index in indexes}, 0, 1, "TARGET_UNIQUENESS")
    add(
        {index: 1 for index in range(variable_count)},
        math.ceil(variant_count / 3), maximum_target_count, "TARGET_COUNT",
    )
    # State balance is corpus-final, not batch-local.  Compensate the exact
    # promoted prefix so the terminal corpus reaches its preregistered totals.
    for state in ("A", "F"):
        addition = state_variant_additions[state]
        add({
            index: len(bundle)
            for index, (context_index, bundle) in enumerate(options)
            if context_by_index[context_index]["target"]["state"] == state
        }, addition, addition, "STATE_VARIANTS")

    # Identity balance is also cumulative.  For an intermediate batch, retain
    # only prefixes from which the final 55--59% active envelope is reachable,
    # using the exact remaining state-variant deficits and max three variants
    # per future identity.  At the terminal residual these reduce to the exact
    # final linear bounds including all already promoted identities.
    active_minimum, active_maximum = target_active_share_bounds
    future_active_max = future_state_variant_counts["A"]
    future_closed_min = math.ceil(future_state_variant_counts["F"] / 3)
    # Represent the preregistered decimal bounds as exact integer inequalities,
    # avoiding floating MILP edges.  The 54.5% lower bound is the certified
    # residual-capacity avenant after the promoted prefix made 55% unreachable;
    # the 59% ceiling is unchanged.
    (lower_active, lower_closed), (upper_active, upper_closed) = (
        identity_share_coefficients((active_minimum, active_maximum))
    )
    lower_constant = (
        lower_active * (prior_state_target_counts["A"] + future_active_max)
        + lower_closed * (prior_state_target_counts["F"] + future_closed_min)
    )
    add({
        index: (
            lower_active
            if context_by_index[context_index]["target"]["state"] == "A"
            else lower_closed
        )
        for index, (context_index, _bundle) in enumerate(options)
    }, -lower_constant, np.inf, "STATE_TARGET_SHARE_REACHABLE")
    future_active_min = math.ceil(future_state_variant_counts["A"] / 3)
    future_closed_max = future_state_variant_counts["F"]
    upper_constant = (
        upper_active * (prior_state_target_counts["A"] + future_active_min)
        + upper_closed * (prior_state_target_counts["F"] + future_closed_max)
    )
    add({
        index: (
            upper_active
            if context_by_index[context_index]["target"]["state"] == "A"
            else upper_closed
        )
        for index, (context_index, _bundle) in enumerate(options)
    }, -np.inf, -upper_constant, "STATE_TARGET_SHARE_REACHABLE")
    difficulty_rows: dict[str, int] = {}
    for level in DIFFICULTIES:
        difficulty_rows[level] = add({
            index: sum(
                difficulty(context_by_index[context_index], pair) == level
                for pair in bundle
            )
            for index, (context_index, bundle) in enumerate(options)
        }, difficulty_minimum_additions[level],
           difficulty_maximum_additions[level], "DIFFICULTY")
    add({
        index: sum(pair[0] == "TOKEN_SUBSET" for pair in bundle)
        for index, (_context_index, bundle) in enumerate(options)
    }, 0, name_token_subset_remaining, "TOKEN_SUBSET_CAP")
    add({
        index: sum(pair[0] == "OFFICIAL_NAME_ALIAS" for pair in bundle)
        for index, (_context_index, bundle) in enumerate(options)
    }, 0, official_name_alias_remaining, "OFFICIAL_NAME_ALIAS_CAP")
    all_pairs = sorted({pair for _context_index, bundle in options for pair in bundle})
    for pair in all_pairs:
        remaining = max(0, relation_pair_cap - prior_pair_counts[pair_signature(pair)])
        add({
            index: sum(value == pair for value in bundle)
            for index, (_context_index, bundle) in enumerate(options)
        }, 0, remaining, "RELATION_PAIR_CAP")
    for (field, relation), capacity in relation_capacities.items():
        add({
            index: sum(
                ("name" if pair[0] == relation and field == "name" else None) == field
                or (pair[1] == (field, relation))
                for pair in bundle
            )
            for index, (_context_index, bundle) in enumerate(options)
        }, 0, capacity, "RELATION_CAPACITY")

    # Hall-style resource constraints couple target selection to the fragments
    # that are actually compatible with each surface.  Relation-level totals are
    # insufficient: thousands of targets can otherwise all depend on the same
    # almost-saturated abbreviation proof.
    for field, relation in (
        [("name", value) for value in SAFE_NAME_RELATIONS]
        + list(SAFE_LOCATION_RELATIONS)
    ):
        if (field, relation) == ("name", "OFFICIAL_NAME_ALIAS"):
            # Its evidence is target-specific, has local capacity three, and is
            # already bounded by candidate_bundles; cross-target Hall sets are
            # disjoint and add no information.
            continue
        support_by_context: dict[int, list[dict[str, Any]]] = {}
        for context_index, context in enumerate(context_by_index):
            names, locations = capabilities[context["target_siret"]]
            values = (
                names.get(relation, []) if field == "name"
                else locations.get((field, relation), [])
            )
            if values:
                support_by_context[context_index] = values
        if not support_by_context:
            continue
        resource_specs = [
            (
                "REF",
                lambda value: str(value["inspiration_ref"]),
                lambda resource: effective_ref_caps.get(resource, 0),
            ),
            (
                "OPERATOR",
                fragment_operator,
                lambda resource: max(
                    0, operator_cap - prior_operator_counts[resource]
                ),
            ),
        ]
        if (field, relation) == ("name", "TOKEN_SUBSET"):
            resource_specs.append((
                "TOKEN_SUBSET_SIGNATURE",
                lambda value: str(registry_lib.token_subset_signature(value)),
                lambda resource: max(
                    0, token_subset_signature_cap
                    - prior_token_subset_signature_counts[resource]
                ),
            ))
        for _resource_kind, resource_key, remaining_capacity in resource_specs:
            support_sets = {
                context_index: frozenset(resource_key(value) for value in values)
                for context_index, values in support_by_context.items()
            }
            unique_support_sets = sorted(
                set(support_sets.values()), key=lambda value: (len(value), sorted(value))
            )
            hall_sets = set(unique_support_sets)
            resources = set().union(*unique_support_sets)
            for resource in resources:
                touching = [value for value in unique_support_sets if resource in value]
                if len(touching) > 1:
                    hall_sets.add(frozenset().union(*touching))
            # The union of an overlap component is another necessary Hall cut;
            # disjoint components need no combined cut because capacities add.
            pending = set(unique_support_sets)
            while pending:
                component = {pending.pop()}
                component_union = set(next(iter(component)))
                changed = True
                while changed:
                    changed = False
                    touching = {
                        value for value in pending if component_union.intersection(value)
                    }
                    if touching:
                        pending.difference_update(touching)
                        component.update(touching)
                        component_union.update(*touching)
                        changed = True
                if len(component) > 1:
                    hall_sets.add(frozenset(component_union))
            for resource_set in sorted(
                hall_sets, key=lambda value: (len(value), sorted(value))
            ):
                capacity = sum(remaining_capacity(value) for value in resource_set)
                coefficients: dict[int, float] = {}
                for context_index, support in support_sets.items():
                    if not support.issubset(resource_set):
                        continue
                    for option_index in option_by_context[context_index]:
                        bundle = options[option_index][1]
                        demand = sum(
                            pair[0] == relation if field == "name"
                            else pair[1] == (field, relation)
                            for pair in bundle
                        )
                        if demand:
                            coefficients[option_index] = demand
                if coefficients:
                    add(coefficients, 0, capacity, f"HALL_{_resource_kind}")

    # Minimize the batch difficulty L-infinity deviation in the same MILP.
    # The former exact solve + relaxed solve + binary search could consume one
    # 180-second and several 60-second HiGHS time limits even though all those
    # solves used the same variables and guards.  A single bounded integer
    # deviation variable is mathematically equivalent and leaves every final
    # quota remainder, quality guard, Hall cut and cumulative cap unchanged.
    tolerance_index = variable_count
    for level, difficulty_row in difficulty_rows.items():
        coefficients = dict(constraint_rows[difficulty_row])
        coefficients[tolerance_index] = -1
        add(coefficients, -np.inf, difficulty_counts[level], "DIFFICULTY")
        coefficients = dict(constraint_rows[difficulty_row])
        coefficients[tolerance_index] = 1
        add(coefficients, difficulty_counts[level], np.inf, "DIFFICULTY")

    decision_variable_count = variable_count + 1
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    for row_index, coefficients in enumerate(constraint_rows):
        for column_index, coefficient in coefficients.items():
            if coefficient:
                matrix_rows.append(row_index)
                matrix_columns.append(column_index)
                matrix_values.append(coefficient)
    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(constraint_rows), decision_variable_count),
    ).tocsr()
    # One unit of deviation dominates the largest possible target-count change.
    # Among minimum-deviation solutions, minimize the number of selected targets.
    # HiGHS is deterministic for this stably ordered matrix; an arbitrary random
    # tie objective previously forced it to prove a scientifically meaningless
    # optimum among thousands of otherwise equivalent targets.
    if diagnose_infeasible:
        objective = np.zeros(decision_variable_count)
    elif feasibility_only:
        # The quota rows remain exact; minimize the redundant diagnostic
        # tolerance so a pure-feasibility certificate cannot report an
        # arbitrary upper-bound value such as ``variant_count``.
        objective = np.concatenate((np.zeros(variable_count), [1.0]))
    else:
        objective = np.concatenate((
            np.ones(variable_count),
            np.asarray([float(maximum_target_count + 1)]),
        ))
    solution = milp(
        objective,
        integrality=np.ones(decision_variable_count),
        bounds=Bounds(
            np.zeros(decision_variable_count),
            np.concatenate((np.ones(variable_count), [float(variant_count)])),
        ),
        constraints=LinearConstraint(matrix, minima, maxima),
        options={"time_limit": 180},
    )
    difficulty_tolerance = (
        int(round(solution.x[tolerance_index]))
        if solution.success and solution.x is not None else 0
    )
    if not solution.success or solution.x is None:
        diagnostic: dict[str, Any] = {
            "solver_status": int(solution.status),
            "solver_message": str(solution.message),
            "solver_mip_gap": (
                float(solution.mip_gap)
                if getattr(solution, "mip_gap", None) is not None else None
            ),
            "solver_mip_node_count": (
                int(solution.mip_node_count)
                if getattr(solution, "mip_node_count", None) is not None else None
            ),
            "contexts_by_state": dict(Counter(
                value["target"]["state"] for value in context_by_index
            )),
            "options": variable_count,
            "constraint_counts": dict(Counter(constraint_labels)),
            "difficulty_target": difficulty_counts,
            "difficulty_final_remainders": difficulty_maximum_additions,
            "bundle_profile_counts": dict(Counter(
                "/".join(str(sum(
                    difficulty(context_by_index[context_index], pair) == level
                    for pair in bundle
                )) for level in DIFFICULTIES)
                for context_index, bundle in options
            )),
            "easy_capability_by_state_and_alias": {
                f"{state}:{alias}": {
                    "contexts": sum(
                        value["target"]["state"] == state
                        and alias_available[value["target_siret"]] == alias
                        for value in eligible
                    ),
                    "easy_contexts": sum(
                        value["target"]["state"] == state
                        and alias_available[value["target_siret"]] == alias
                        and easy_capacity[value["target_siret"]] > 0
                        for value in eligible
                    ),
                    "easy_variant_upper_bound": sum(
                        easy_capacity[value["target_siret"]]
                        for value in eligible
                        if value["target"]["state"] == state
                        and alias_available[value["target_siret"]] == alias
                    ),
                }
                for state in ("A", "F") for alias in (True, False)
            },
            "easy_bundle_blocker_samples": [
                {
                    "target_siret": value["target_siret"],
                    "name_exact_operator_counts": {
                        relation: len(fragments)
                        for relation, fragments in capabilities[
                            value["target_siret"]
                        ][0].items() if fragments
                    },
                    "location_exact_operator_counts": {
                        f"{field}:{relation}": len(fragments)
                        for (field, relation), fragments in capabilities[
                            value["target_siret"]
                        ][1].items() if fragments
                    },
                }
                for value in eligible
                if easy_capacity[value["target_siret"]] > 0
                and not any(
                    any(
                        difficulty(value, pair) == "EASY" for pair in bundle
                    )
                    for bundle in candidate_bundles(
                        value, *capabilities[value["target_siret"]], selection_seed,
                        pair_remaining=pair_remaining,
                        official_name_alias_remaining=official_name_alias_remaining,
                    )
                )
            ][:5],
        }
        if diagnose_infeasible:
            # Diagnose with LP relaxations first.  LP infeasibility is a proof;
            # unlike a short secondary integer solve it cannot be confused with
            # a timeout on this 50k+ binary model.  Preserve every solver status
            # and message rather than reducing unknown outcomes to ``False``.
            strategic_relaxations = {
                "NONE": set(),
                "STATE_SHARE_AND_TARGET_COUNT": {
                    "STATE_TARGET_SHARE_REACHABLE", "TARGET_COUNT",
                },
                "ALL_STATE_AND_TARGET_COUNT": {
                    "STATE_VARIANTS", "STATE_TARGET_SHARE_REACHABLE", "TARGET_COUNT",
                },
                "ALL_MIX": {
                    "DIFFICULTY", "STATE_VARIANTS", "STATE_TARGET_SHARE_REACHABLE",
                },
                "ALL_FRAGMENT_CAPACITY": {
                    "RELATION_CAPACITY", "HALL_REF", "HALL_OPERATOR",
                    "HALL_TOKEN_SUBSET_SIGNATURE",
                },
                "PAIR_AND_FRAGMENT_CAPACITY": {
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE",
                },
                "ALL_GLOBAL_CAPS": {
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE",
                },
                "ALL_GLOBAL_CAPS_AND_TARGET_COUNT": {
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE", "TARGET_COUNT",
                },
                "ALL_MIX_AND_GLOBAL_CAPS": {
                    "DIFFICULTY", "STATE_VARIANTS", "STATE_TARGET_SHARE_REACHABLE",
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE",
                },
                "ALL_MIX_GLOBAL_CAPS_AND_TARGET_COUNT": {
                    "DIFFICULTY", "STATE_VARIANTS", "STATE_TARGET_SHARE_REACHABLE",
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE", "TARGET_COUNT",
                },
                "GLOBAL_CAPS_AND_STATE_SHARE": {
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE",
                    "STATE_TARGET_SHARE_REACHABLE",
                },
                "GLOBAL_CAPS_AND_STATE_VARIANTS": {
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE",
                    "STATE_VARIANTS",
                },
                "GLOBAL_CAPS_AND_DIFFICULTY": {
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE", "DIFFICULTY",
                },
                "GLOBAL_CAPS_STATE_SHARE_AND_VARIANTS": {
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE",
                    "STATE_TARGET_SHARE_REACHABLE", "STATE_VARIANTS",
                },
                "GLOBAL_CAPS_DIFFICULTY_AND_STATE_SHARE": {
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE", "DIFFICULTY",
                    "STATE_TARGET_SHARE_REACHABLE",
                },
                "GLOBAL_CAPS_DIFFICULTY_AND_STATE_VARIANTS": {
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE", "DIFFICULTY",
                    "STATE_VARIANTS",
                },
                "ALL_MIX_AND_FAMILY_CAPS": {
                    "DIFFICULTY", "STATE_VARIANTS", "STATE_TARGET_SHARE_REACHABLE",
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                },
                "ALL_MIX_AND_PAIR_CAPS": {
                    "DIFFICULTY", "STATE_VARIANTS", "STATE_TARGET_SHARE_REACHABLE",
                    "RELATION_PAIR_CAP",
                },
                "ALL_MIX_AND_FRAGMENT_CAPS": {
                    "DIFFICULTY", "STATE_VARIANTS", "STATE_TARGET_SHARE_REACHABLE",
                    "RELATION_CAPACITY", "HALL_REF", "HALL_OPERATOR",
                    "HALL_TOKEN_SUBSET_SIGNATURE",
                },
                "ALL_MIX_FAMILY_AND_PAIR_CAPS": {
                    "DIFFICULTY", "STATE_VARIANTS", "STATE_TARGET_SHARE_REACHABLE",
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_PAIR_CAP",
                },
                "ALL_MIX_FAMILY_AND_FRAGMENT_CAPS": {
                    "DIFFICULTY", "STATE_VARIANTS", "STATE_TARGET_SHARE_REACHABLE",
                    "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                    "RELATION_CAPACITY", "HALL_REF", "HALL_OPERATOR",
                    "HALL_TOKEN_SUBSET_SIGNATURE",
                },
                "ALL_MIX_PAIR_AND_FRAGMENT_CAPS": {
                    "DIFFICULTY", "STATE_VARIANTS", "STATE_TARGET_SHARE_REACHABLE",
                    "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                    "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE",
                },
            }
            relaxation_diagnostics = {}
            for name, labels in strategic_relaxations.items():
                keep = [
                    index for index, value in enumerate(constraint_labels)
                    if value not in labels
                ]
                relaxed = milp(
                    np.zeros(decision_variable_count),
                    integrality=np.zeros(decision_variable_count),
                    bounds=Bounds(
                        np.zeros(decision_variable_count),
                        np.concatenate((
                            np.ones(variable_count), [float(variant_count)]
                        )),
                    ),
                    constraints=LinearConstraint(
                        matrix[keep], np.asarray(minima)[keep], np.asarray(maxima)[keep],
                    ),
                    options={"time_limit": 30},
                )
                relaxation_diagnostics[name] = {
                    "success": bool(relaxed.success),
                    "status": int(relaxed.status),
                    "message": str(relaxed.message),
                    "mip_gap": (
                        float(relaxed.mip_gap)
                        if getattr(relaxed, "mip_gap", None) is not None else None
                    ),
                }
            diagnostic["lp_relaxation_diagnostics"] = relaxation_diagnostics
            lp_bounds = Bounds(
                np.zeros(decision_variable_count),
                np.concatenate((
                    np.ones(variable_count), [float(variant_count)]
                )),
            )

            def diagnostic_lp(
                dropped: set[str], objective_values: np.ndarray,
            ) -> tuple[Any, list[int]]:
                kept = [
                    index for index, value in enumerate(constraint_labels)
                    if value not in dropped
                ]
                result = milp(
                    objective_values,
                    integrality=np.zeros(decision_variable_count),
                    bounds=lp_bounds,
                    constraints=LinearConstraint(
                        matrix[kept], np.asarray(minima)[kept],
                        np.asarray(maxima)[kept],
                    ),
                    options={"time_limit": 30},
                )
                return result, kept

            global_cap_labels = {
                "TOKEN_SUBSET_CAP", "OFFICIAL_NAME_ALIAS_CAP",
                "RELATION_PAIR_CAP", "RELATION_CAPACITY", "HALL_REF",
                "HALL_OPERATOR", "HALL_TOKEN_SUBSET_SIGNATURE",
            }
            mix_labels = {
                "DIFFICULTY", "STATE_VARIANTS", "STATE_TARGET_SHARE_REACHABLE",
            }
            difficulty_ranges = {}
            for level in DIFFICULTIES:
                coefficients = np.asarray([
                    sum(
                        difficulty(context_by_index[context_index], pair) == level
                        for pair in bundle
                    )
                    for context_index, bundle in options
                ] + [0], dtype=float)
                endpoints = {}
                for endpoint_name, direction in (("minimum", 1.0), ("maximum", -1.0)):
                    endpoint, _kept = diagnostic_lp(
                        global_cap_labels | {"DIFFICULTY"},
                        direction * coefficients,
                    )
                    endpoints[endpoint_name] = {
                        "status": int(endpoint.status),
                        "message": str(endpoint.message),
                        "value": (
                            float(coefficients @ endpoint.x)
                            if endpoint.success and endpoint.x is not None else None
                        ),
                    }
                difficulty_ranges[level] = endpoints
            diagnostic["lp_difficulty_ranges_without_global_caps"] = difficulty_ranges

            total_coefficients = np.asarray([
                len(bundle) for _context_index, bundle in options
            ] + [0], dtype=float)
            fragment_labels = {
                "RELATION_CAPACITY", "HALL_REF", "HALL_OPERATOR",
                "HALL_TOKEN_SUBSET_SIGNATURE",
            }
            capacity_scenarios = {
                "FAMILY_CAPS_ONLY": (
                    mix_labels | {"TOTAL_VARIANTS", "RELATION_PAIR_CAP"}
                    | fragment_labels
                ),
                "ALIAS_CAP_ONLY": (
                    mix_labels | {"TOTAL_VARIANTS", "TOKEN_SUBSET_CAP",
                                  "RELATION_PAIR_CAP"} | fragment_labels
                ),
                "TOKEN_SUBSET_CAP_ONLY": (
                    mix_labels | {"TOTAL_VARIANTS", "OFFICIAL_NAME_ALIAS_CAP",
                                  "RELATION_PAIR_CAP"} | fragment_labels
                ),
                "PAIR_CAPS_ONLY": (
                    mix_labels | {"TOTAL_VARIANTS", "TOKEN_SUBSET_CAP",
                                  "OFFICIAL_NAME_ALIAS_CAP"} | fragment_labels
                ),
                "FAMILY_AND_PAIR_CAPS": (
                    mix_labels | {"TOTAL_VARIANTS"} | fragment_labels
                ),
                "ALL_GLOBAL_CAPS": mix_labels | {"TOTAL_VARIANTS"},
            }
            capacity_bounds = {}
            for name, dropped in capacity_scenarios.items():
                endpoint, _kept = diagnostic_lp(dropped, -total_coefficients)
                maximum = (
                    float(total_coefficients @ endpoint.x)
                    if endpoint.success and endpoint.x is not None else None
                )
                capacity_bounds[name] = {
                    "status": int(endpoint.status),
                    "message": str(endpoint.message),
                    "maximum_variants_lp": maximum,
                    "minimum_variant_extension_if_below_target": (
                        max(0.0, variant_count - maximum)
                        if maximum is not None else None
                    ),
                }
            diagnostic["lp_total_capacity_by_cap_scenario"] = capacity_bounds
        raise ValueError(
            f"balanced production MILP is infeasible: {solution.message}; "
            f"diagnostic={json.dumps(diagnostic, sort_keys=True)}"
        )
    result = [
        (context_by_index[context_index], bundle)
        for index, (context_index, bundle) in enumerate(options)
        if solution.x[index] > 0.5
    ]
    if sum(len(bundle) for _context, bundle in result) != variant_count:
        raise RuntimeError("MILP returned an incomplete variant selection")
    if not math.ceil(variant_count / 3) <= len(result) <= variant_count:
        raise RuntimeError("MILP returned an invalid target count")
    if difficulty_tolerance:
        print(json.dumps({
            "difficulty_batch_relaxation": {
                "ideal": difficulty_counts,
                "minimum_linf_tolerance": difficulty_tolerance,
                "final_remainders": difficulty_maximum_additions,
            }
        }, sort_keys=True), file=sys.stderr)
    return sorted(result, key=lambda value: value[0]["target_siret"])


def fragment_operator(value: dict[str, Any]) -> str:
    return fragments.fragment_operator(value)


def effective_ref_capacities(
    values: Sequence[dict[str, Any]],
    ref_cap: int,
    operator_cap: int,
    prior_ref_counts: Counter[str],
    prior_operator_counts: Counter[str],
) -> dict[str, int]:
    """Deterministically partition each residual operator cap across its refs.

    This turns the coupled ref/operator assignment into one capacitated matching:
    any flow respecting these effective ref caps respects both original caps.
    """
    refs_by_operator: dict[str, set[str]] = defaultdict(set)
    for value in values:
        refs_by_operator[fragment_operator(value)].add(value["inspiration_ref"])
    result: dict[str, int] = {}
    for operator, refs in sorted(refs_by_operator.items()):
        remaining_operator = max(0, operator_cap - prior_operator_counts[operator])
        residual = {
            ref: max(0, ref_cap - prior_ref_counts[ref]) for ref in sorted(refs)
        }
        # Round-robin waterfill is deterministic and avoids stranding capacity on
        # one proof when another compatible surface only admits a sibling proof.
        active = [ref for ref, capacity in residual.items() if capacity > 0]
        allocated: Counter[str] = Counter()
        while remaining_operator > 0 and active:
            next_active = []
            for ref in active:
                if remaining_operator <= 0:
                    break
                if allocated[ref] < residual[ref]:
                    allocated[ref] += 1
                    remaining_operator -= 1
                if allocated[ref] < residual[ref]:
                    next_active.append(ref)
            active = next_active
        result.update(allocated)
    return result


def assign_fragments(
    selected: list[tuple[dict[str, Any], tuple[tuple[str, tuple[str, str]], ...]]],
    capabilities: dict[str, tuple[dict[str, Any], dict[tuple[str, str], Any]]],
    ref_cap: int,
    operator_cap: int,
    prior_ref_counts: Counter[str],
    prior_operator_counts: Counter[str],
    effective_ref_caps: dict[str, int],
    token_subset_signature_cap: int,
    prior_token_subset_signature_counts: Counter[str],
    selection_seed: str,
) -> tuple[dict[str, list[dict[str, dict[str, Any]]]], Counter[str], Counter[str]]:
    graph = nx.DiGraph()
    source, sink = "SOURCE", "SINK"
    request_specs: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    by_ref: dict[str, dict[str, Any]] = {}
    for context, bundle in selected:
        siret = context["target_siret"]
        names, locations = capabilities[siret]
        for slot, (name_relation, location_key) in enumerate(bundle):
            for field, values in (
                ("name", names[name_relation]),
                (location_key[0], locations[location_key]),
            ):
                request = ("REQUEST", siret, slot, field)
                request_specs[request] = values
                graph.add_edge(source, request, capacity=1, weight=0)
                for fragment in values:
                    ref = fragment["inspiration_ref"]
                    operator = fragment_operator(fragment)
                    subset_signature = registry_lib.token_subset_signature(fragment)
                    ref_remaining = (
                        min(
                            ref_cap - prior_ref_counts[ref],
                            operator_cap - prior_operator_counts[operator],
                        )
                        if fragment.get("evidence_source_type")
                        == "SIRENE_OFFICIAL_NAME_OPTION"
                        else effective_ref_caps.get(ref, 0)
                    )
                    operator_remaining = operator_cap - prior_operator_counts[operator]
                    signature_remaining = (
                        token_subset_signature_cap
                        - prior_token_subset_signature_counts[subset_signature]
                        if subset_signature is not None else operator_remaining
                    )
                    if (
                        ref_remaining <= 0 or operator_remaining <= 0
                        or signature_remaining <= 0
                    ):
                        continue
                    by_ref[ref] = fragment
                    target_ref = ("TARGET_REF", siret, ref)
                    ref_node = ("REF", ref)
                    operator_node = ("OPERATOR", operator)
                    digest = hashlib.sha256(
                        f"{selection_seed}|flow|{siret}|{slot}|{field}|{ref}".encode()
                    ).hexdigest()
                    graph.add_edge(
                        request, target_ref, capacity=1,
                        weight=int(digest[:8], 16),
                    )
                    # A train-observed fragment stays unique within a target.  An
                    # authoritative alias is target-specific and may be paired with
                    # up to three distinct location operators for that same target.
                    target_ref_capacity = (
                        3 if fragment.get("evidence_source_type")
                        == "SIRENE_OFFICIAL_NAME_OPTION" else 1
                    )
                    graph.add_edge(
                        target_ref, ref_node, capacity=target_ref_capacity, weight=0
                    )
                    graph.add_edge(
                        ref_node, operator_node, capacity=ref_remaining, weight=0
                    )
                    if subset_signature is None:
                        graph.add_edge(
                            operator_node, sink, capacity=operator_remaining, weight=0
                        )
                    else:
                        signature_node = ("TOKEN_SUBSET_SIGNATURE", subset_signature)
                        graph.add_edge(
                            operator_node, signature_node,
                            capacity=operator_remaining, weight=0,
                        )
                        graph.add_edge(
                            signature_node, sink,
                            capacity=signature_remaining, weight=0,
                        )
    flow = nx.max_flow_min_cost(graph, source, sink)
    flow_value = sum(flow[source].values())
    if flow_value != len(request_specs):
        missing = [request for request in request_specs if not flow[source].get(request, 0)]
        missing_relations = Counter(
            (values[0]["field"], values[0]["relation"])
            for request, values in request_specs.items() if request in missing and values
        )
        saturated_refs = Counter()
        for request in missing:
            for fragment in request_specs[request]:
                ref = fragment["inspiration_ref"]
                ref_node = ("REF", ref)
                used = sum(flow.get(ref_node, {}).values())
                if used >= max(0, ref_cap - prior_ref_counts[ref]):
                    saturated_refs[ref] = used
        raise ValueError(
            f"global fragment flow is infeasible: {flow_value}/{len(request_specs)} assignments; "
            f"missing_relations={dict(missing_relations)}; "
            f"saturated_missing_refs={saturated_refs.most_common(10)}"
        )
    result: dict[str, list[dict[str, dict[str, Any]]]] = {
        context["target_siret"]: [dict() for _value in bundle]
        for context, bundle in selected
    }
    ref_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    for request in request_specs:
        selected_target_refs = [
            node for node, amount in flow[request].items()
            if amount and isinstance(node, tuple) and node[0] == "TARGET_REF"
        ]
        if len(selected_target_refs) != 1:
            raise RuntimeError(f"fragment flow has no unique choice for {request}")
        ref = selected_target_refs[0][2]
        fragment = by_ref[ref]
        _kind, siret, slot, field = request
        result[siret][slot][field] = fragment
        ref_counts[ref] += 1
        operator_counts[fragment_operator(fragment)] += 1
    return result, ref_counts, operator_counts


def assign_strata(
    variants: list[dict[str, Any]],
    stratum_counts: dict[str, int],
    catalog: dict[str, Any],
) -> dict[str, str]:
    graph = nx.DiGraph()
    source, sink = "SOURCE", "SINK"
    graph.add_node(source, demand=-len(variants))
    graph.add_node(sink, demand=len(variants))
    profile_lookup = {
        cell: {
            value["archetype"]: float(value["lift_vs_all_train_oof"])
            for value in profile["archetypes"]
        }
        for cell, profile in catalog["cell_profiles"].items()
    }
    for stratum, count in stratum_counts.items():
        node = ("STRATUM", stratum)
        graph.add_node(node, demand=0)
        graph.add_edge(source, node, capacity=count, weight=0)
        for variant in variants:
            if stratum == "NEAR_CLEAN_CONTROL" and variant["difficulty"] != "EASY":
                continue
            flags = variant["archetypes"]
            if stratum == "NEAR_CLEAN_CONTROL":
                score = 20.0 - len(flags)
            else:
                cell = STRATUM_TO_OOF_CELL[stratum]
                score = sum(profile_lookup.get(cell, {}).get(flag, 0.0) for flag in flags)
            graph.add_edge(
                node, ("VARIANT", variant["key"]), capacity=1,
                weight=int(round(-1000 * score)),
            )
    for variant in variants:
        node = ("VARIANT", variant["key"])
        graph.add_node(node, demand=0)
        graph.add_edge(node, sink, capacity=1, weight=0)
    _cost, flow = nx.network_simplex(graph)
    assignment: dict[str, str] = {}
    for stratum in stratum_counts:
        node = ("STRATUM", stratum)
        for target, amount in flow[node].items():
            if amount and isinstance(target, tuple) and target[0] == "VARIANT":
                assignment[target[1]] = stratum
    if len(assignment) != len(variants):
        raise RuntimeError("stratum flow did not assign every production variant")
    return assignment


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_count <= 0:
        raise ValueError("target-count must be positive")
    if args.variant_count < 0:
        raise ValueError("variant-count cannot be negative")
    variant_count = args.variant_count or args.target_count * 3
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    source_paths = {
        key: Path(value["path"]) for key, value in plan["sources"].items()
    }
    for key, path in source_paths.items():
        if sha256(path) != plan["sources"][key]["sha256"]:
            raise ValueError(f"source hash mismatch: {key}")
    catalog = json.loads(source_paths["train_oof_failure_catalog"].read_text(encoding="utf-8"))
    if (
        catalog.get("scope") != "TRAIN_OOF_AGGREGATES_ONLY"
        or catalog.get("contains_query_ids_or_entity_ids") is not False
        or set(catalog.get("allowed_folds", [])) != {2, 3, 4}
    ):
        raise ValueError("failure catalog does not prove train-only aggregate provenance")
    allowed_name_relations = tuple(plan["relations"]["name_allowed"])
    unsupported_name_relations = set(allowed_name_relations) - set(SAFE_NAME_RELATIONS)
    if unsupported_name_relations:
        raise ValueError(
            f"unsupported safe name relations in plan: {sorted(unsupported_name_relations)}"
        )

    prior_variant_count, prior_ref_counts, prior_operator_counts, prior_pair_counts = (
        production_usage(args.prior_counted_seed_input)
    )
    prior_difficulty_counts: Counter[str] = Counter()
    prior_stratum_counts: Counter[str] = Counter()
    prior_token_subset_signature_counts: Counter[str] = Counter()
    prior_state_variant_counts, prior_state_target_counts = seed_state_usage(
        args.prior_counted_seed_input
    )
    registry_sirets: set[str] = set()
    registry_sirens: set[str] = set()
    prior_distinct_target_count = 0
    registry_hash: str | None = None
    if args.production_registry is not None:
        if args.prior_counted_seed_input:
            raise ValueError(
                "--production-registry and --prior-counted-seed-input are mutually exclusive"
            )
        (
            prior_variant_count, prior_ref_counts, prior_operator_counts,
            prior_pair_counts, prior_difficulty_counts, prior_stratum_counts,
            prior_token_subset_signature_counts, prior_distinct_target_count,
            registry_sirets, registry_sirens, registry_hash,
        ) = registry_usage(args.production_registry)
        prior_state_variant_counts, prior_state_target_counts = (
            registry_state_usage(args.production_registry)
        )
    all_excluded_inputs = [
        *args.exclude_seed_input, *args.prior_counted_seed_input,
    ]
    excluded_sirets, excluded_sirens = fragments.excluded_target_ids(all_excluded_inputs)
    excluded_sirets.update(registry_sirets)
    excluded_sirens.update(registry_sirens)
    indexed_contexts = not args.disable_context_index_cache
    if indexed_contexts:
        cache_directory = (
            args.context_index_cache_directory
            if args.context_index_cache_directory is not None
            else source_paths["official_context"].parent
            / ".balanced_selector_context_index"
        )
        context_entries, document_frequencies = build_context_index(
            source_paths["official_context"],
            plan["sources"]["official_context"]["sha256"],
            cache_directory,
        )
        eligible_entries = [
            value for value in context_entries
            if value["target_siret"] not in excluded_sirets
            and value["target_siren"] not in excluded_sirens
            and (
                value["regular_eligible"]
                or (
                    "OFFICIAL_NAME_ALIAS" in allowed_name_relations
                    and value["alias_eligible"]
                )
            )
        ]
        contexts = [indexed_context_stub(value) for value in eligible_entries]
        official_alias_available = {
            value["target_siret"]: bool(value["alias_available"])
            for value in eligible_entries
        }
        structural_easy_priority = {
            value["target_siret"]: int(value["structural_easy_priority"])
            for value in eligible_entries
        }
    else:
        all_contexts = rows(source_paths["official_context"])
        document_frequencies = fragments.name_token_document_frequencies(all_contexts)
        contexts = [
            value for value in all_contexts
            if value["target_siret"] not in excluded_sirets
            and value["target_siren"] not in excluded_sirens
            and (
                fragments.eligible(value)
                or (
                    "OFFICIAL_NAME_ALIAS" in allowed_name_relations
                    and eligible_for_official_alias(value)
                )
            )
        ]
        official_alias_available = {
            value["target_siret"]: bool(official_name_alias_fragments(value))
            for value in contexts
        }
        structural_easy_priority = {
            value["target_siret"]: (
                3 if value["target"]["state"] == "A" and not context_flags(value) else 0
            )
            for value in contexts
        }
    cumulative_variant_count = prior_variant_count + variant_count
    final_variant_target = int(plan["objective"]["promoted_variant_target"])
    maximum_unique_targets = int(plan["objective"]["maximum_unique_targets"])
    target_active_share_bounds = tuple(
        float(value) for value in
        plan["corpus_balance"]["target_identity_active_share_bounds"]
    )
    if (
        len(target_active_share_bounds) != 2
        or not 0 <= target_active_share_bounds[0]
        <= target_active_share_bounds[1] <= 1
    ):
        raise ValueError("invalid target identity active-share bounds")
    final_state_variant_counts = exact_counts(
        final_variant_target, plan["corpus_balance"]["state_variants"]
    )
    state_variant_additions = remaining_quota_counts(
        variant_count, final_variant_target,
        plan["corpus_balance"]["state_variants"], prior_state_variant_counts,
    )
    future_state_variant_counts = {
        state: (
            final_state_variant_counts[state]
            - prior_state_variant_counts[state]
            - state_variant_additions[state]
        )
        for state in ("A", "F")
    }
    maximum_target_count = maximum_batch_target_additions(
        prior_variant_count, prior_distinct_target_count, variant_count,
        final_variant_target, maximum_unique_targets,
        int(plan["objective"]["maximum_variants_per_target"]),
    )
    if maximum_target_count < math.ceil(variant_count / 3):
        raise ValueError(
            "remaining unique-target budget cannot host this batch while reserving "
            "three variants per target for the final residual"
        )
    # These are corpus-final caps, not prefix quotas.  Enforcing 5%/10% on
    # every small early batch strands common but legitimate address operators;
    # the registry still debits every use and makes the final ceilings hard.
    relation_pair_cap = max(
        1, int(math.ceil(
            final_variant_target * plan["global_caps"]["relation_pair_share"]
        ))
    )
    official_name_alias_cap = int(math.floor(
        final_variant_target
        * plan["global_caps"]["official_name_alias_share"]
    ))
    prior_official_name_alias_count = sum(
        count for signature, count in prior_pair_counts.items()
        if signature.startswith("name:OFFICIAL_NAME_ALIAS+")
    )
    official_name_alias_remaining = max(
        0, official_name_alias_cap - prior_official_name_alias_count
    )
    operator_cap = max(
        1, int(math.ceil(
            final_variant_target * 2
            * plan["global_caps"]["exact_operator_share"]
        ))
    )
    if args.audit_exact_operator_cap:
        if not (args.feasibility_only or args.diagnose_infeasible):
            raise ValueError(
                "--audit-exact-operator-cap requires a read-only feasibility mode"
            )
        operator_cap = int(args.audit_exact_operator_cap)
    first_batch = plan["production"].get("first_batch", {})
    if args.batch_id == first_batch.get("batch_id") and not prior_variant_count:
        operator_cap = max(
            operator_cap, int(first_batch.get("bootstrap_exact_operator_cap", 0))
        )
    name_token_subset_cap = int(math.floor(
        final_variant_target * plan["global_caps"]["name_token_subset_share"]
    ))
    token_subset_signature_cap = min(
        int(math.floor(
            final_variant_target
            * plan["global_caps"]["name_token_subset_signature_share_global"]
        )),
        int(math.floor(
            name_token_subset_cap
            * plan["global_caps"]["name_token_subset_signature_share_within_family"]
        )),
    )
    ref_cap = int(plan["global_caps"]["inspiration_ref_uses"])
    if args.audit_inspiration_ref_cap:
        if not (args.feasibility_only or args.diagnose_infeasible):
            raise ValueError(
                "--audit-inspiration-ref-cap requires a read-only feasibility mode"
            )
        ref_cap = int(args.audit_inspiration_ref_cap)
    grouped_all = fragments.group_fragments(rows(source_paths["field_inspiration_bank"]))
    grouped = {
        key: [
            value for value in values
            if prior_ref_counts[value["inspiration_ref"]] < ref_cap
            and prior_operator_counts[fragment_operator(value)] < operator_cap
            and (
                registry_lib.token_subset_signature(value) is None
                or prior_token_subset_signature_counts[
                    registry_lib.token_subset_signature(value)
                ] < token_subset_signature_cap
            )
        ]
        for key, values in grouped_all.items()
    }
    capabilities = {}
    feasible_contexts = []
    support_cache: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    capability_pool_limit = (
        args.selection_pool_limit
        if args.selection_pool_limit > 0 else max(3000, variant_count)
    )
    capability_pool = stratified_context_pool(
        contexts, capability_pool_limit,
        f"{args.selection_seed}|capability", official_alias_available,
        structural_easy_priority, active_share=2 / 3,
    )
    capability_contexts = (
        read_indexed_contexts(source_paths["official_context"], capability_pool)
        if indexed_contexts else capability_pool
    )
    for context in capability_contexts:
        names, locations = safe_capabilities(
            context, grouped, document_frequencies, allowed_name_relations,
            support_cache,
        )
        if materializable_relation_pair_count(names, locations):
            # Runtime contracts and counted promotion are independently valid
            # per variant.  Since bundles may contain one to three variants,
            # one exact name/location pair is sufficient capability.
            capabilities[context["target_siret"]] = (names, locations)
            feasible_contexts.append(context)

    compatible_train_values = {
        value["inspiration_ref"]: value
        for names, locations in capabilities.values()
        for values in [*names.values(), *locations.values()]
        for value in values
        if value.get("evidence_source_type") != "SIRENE_OFFICIAL_NAME_OPTION"
    }
    effective_ref_caps = effective_ref_capacities(
        list(compatible_train_values.values()), ref_cap, operator_cap,
        prior_ref_counts, prior_operator_counts,
    )
    pruned_capabilities = {}
    feasible_contexts = []
    for context in capability_contexts:
        if context["target_siret"] not in capabilities:
            continue
        names, locations = capabilities[context["target_siret"]]
        names = {
            relation: [
                value for value in values
                if value.get("evidence_source_type")
                == "SIRENE_OFFICIAL_NAME_OPTION"
                or effective_ref_caps.get(value["inspiration_ref"], 0) > 0
            ]
            for relation, values in names.items()
        }
        locations = {
            key: [
                value for value in values
                if effective_ref_caps.get(value["inspiration_ref"], 0) > 0
            ]
            for key, values in locations.items()
        }
        if materializable_relation_pair_count(names, locations):
            pruned_capabilities[context["target_siret"]] = (names, locations)
            feasible_contexts.append(context)
    capabilities = pruned_capabilities
    capability_diagnostics: dict[str, dict[str, int]] = {}
    for context in feasible_contexts:
        siret = context["target_siret"]
        names, locations = capabilities[siret]
        source_kind = (
            "ALIAS" if names.get("OFFICIAL_NAME_ALIAS") else "NON_ALIAS"
        )
        key = f"{context['target']['state']}:{source_kind}"
        record = capability_diagnostics.setdefault(
            key, {"contexts": 0, "easy_contexts": 0, "easy_variant_upper_bound": 0}
        )
        record["contexts"] += 1
        easy_pairs = sum(
            difficulty(context, (name_relation, location_key)) == "EASY"
            for name_relation, name_values in names.items() if name_values
            for location_key, location_values in locations.items() if location_values
        )
        if easy_pairs:
            record["easy_contexts"] += 1
            record["easy_variant_upper_bound"] += min(3, easy_pairs)

    difficulty_counts = remaining_quota_counts(
        variant_count, final_variant_target,
        plan["corpus_balance"]["difficulty"], prior_difficulty_counts,
    )
    final_difficulty_counts = exact_counts(
        final_variant_target, plan["corpus_balance"]["difficulty"]
    )
    difficulty_maximum_additions = {
        level: max(
            0, final_difficulty_counts[level] - prior_difficulty_counts[level]
        )
        for level in DIFFICULTIES
    }
    hard_prefix_cap = int(math.floor(
        cumulative_variant_count
        * plan["global_caps"]["difficulty_hard_prefix_share"]
    ))
    difficulty_maximum_additions["HARD"] = min(
        difficulty_maximum_additions["HARD"],
        max(0, hard_prefix_cap - prior_difficulty_counts["HARD"]),
    )
    stratum_counts = remaining_quota_counts(
        variant_count, final_variant_target,
        plan["corpus_balance"]["augmentation_strata"], prior_stratum_counts,
    )
    ideal_stratum_counts = dict(stratum_counts)
    final_stratum_counts = exact_counts(
        final_variant_target, plan["corpus_balance"]["augmentation_strata"]
    )
    stratum_maximum_additions = {
        key: max(0, final_stratum_counts[key] - prior_stratum_counts[key])
        for key in final_stratum_counts
    }
    # Near-clean controls and EASY share the same scarce active-clean support.
    # Do not make the ideal *batch* control count a hard lower bound here: late
    # corpus batches can be one operator short even though the final remainder
    # is ample.  The L-infinity objective still maximizes attainable EASY toward
    # its ideal, and adapt_strata_to_easy_capacity defers only the exact shortfall
    # while keeping every final stratum upper bound inviolate.
    difficulty_minimum_additions = {level: 0 for level in DIFFICULTIES}
    relation_capacities: dict[tuple[str, str], int] = {}
    prior_name_token_subsets = sum(
        count for signature, count in prior_pair_counts.items()
        if signature.startswith("name:TOKEN_SUBSET+")
    )
    name_token_subset_remaining = max(
        0, name_token_subset_cap - prior_name_token_subsets
    )
    for field, relation in (
        [("name", value) for value in allowed_name_relations]
        + list(SAFE_LOCATION_RELATIONS)
    ):
        if (field, relation) == ("name", "OFFICIAL_NAME_ALIAS"):
            # Each target-specific official alias can support the three variants
            # when the location relation differs.  Per-ref and per-operator caps
            # are still enforced by the assignment flow.
            available = [
                value
                for names, _locations in capabilities.values()
                for value in names.get("OFFICIAL_NAME_ALIAS", [])
            ]
            relation_capacities[(field, relation)] = sum(
                min(3, ref_cap - prior_ref_counts[value["inspiration_ref"]])
                for value in available
                if prior_ref_counts[value["inspiration_ref"]] < ref_cap
                and prior_operator_counts[fragment_operator(value)] < operator_cap
            )
        else:
            available_by_ref = {
                value["inspiration_ref"]: value
                for names, locations in capabilities.values()
                for value in (
                    names.get(relation, []) if field == "name"
                    else locations.get((field, relation), [])
                )
            }
            available = list(available_by_ref.values())
            relation_capacities[(field, relation)] = min(
                sum(
                    effective_ref_caps.get(ref, 0)
                    for ref in {value["inspiration_ref"] for value in available}
                ),
                sum(
                    operator_cap - prior_operator_counts[operator]
                    for operator in {fragment_operator(value) for value in available}
                ),
            )
    if args.batch_id == first_batch.get("batch_id") and not prior_variant_count:
        for key, capacity in first_batch.get("bootstrap_relation_caps", {}).items():
            field, relation = key.split(":", 1)
            relation_capacities[(field, relation)] = min(
                relation_capacities.get((field, relation), int(capacity)), int(capacity)
            )
    # Bundle choice and fragment assignment are coupled by target-specific
    # support sets.  The MILP uses conservative Hall cuts, while the following
    # exact flow is the authority.  Retry only the deterministic selection
    # salt when a conservative choice strands a handful of compatible proofs;
    # no CRM text is generated and no cap is relaxed.
    selection_errors: list[str] = []
    selected = None
    assigned_fragments = None
    ref_counts = None
    operator_counts = None
    effective_selection_seed = args.selection_seed
    selection_attempt_limit = (
        1 if args.diagnose_infeasible or args.feasibility_only else 64
    )
    for selection_attempt in range(selection_attempt_limit):
        effective_selection_seed = (
            args.selection_seed if selection_attempt == 0
            else f"{args.selection_seed}|FLOW_RETRY_{selection_attempt}"
        )
        try:
            candidate_selection = choose_targets_and_bundles(
                feasible_contexts, capabilities, variant_count,
                maximum_target_count, target_active_share_bounds,
                prior_state_variant_counts, prior_state_target_counts,
                state_variant_additions, future_state_variant_counts,
                difficulty_counts,
                difficulty_minimum_additions, difficulty_maximum_additions,
                relation_pair_cap, prior_pair_counts, relation_capacities,
                official_name_alias_remaining,
                name_token_subset_remaining, ref_cap, operator_cap,
                prior_ref_counts, prior_operator_counts, effective_ref_caps,
                token_subset_signature_cap,
                prior_token_subset_signature_counts, effective_selection_seed,
                args.diagnose_infeasible, args.feasibility_only,
                capability_pool_limit,
            )
            candidate_assignment, candidate_ref_counts, candidate_operator_counts = (
                assign_fragments(
                    candidate_selection, capabilities, ref_cap,
                    operator_cap, prior_ref_counts, prior_operator_counts,
                    effective_ref_caps, token_subset_signature_cap,
                    prior_token_subset_signature_counts, effective_selection_seed,
                )
            )
        except ValueError as error:
            message = str(error)
            if not (
                message.startswith("balanced production MILP is infeasible")
                or message.startswith("global fragment flow is infeasible")
            ):
                raise
            selection_errors.append(message)
            continue
        selected = candidate_selection
        assigned_fragments = candidate_assignment
        ref_counts = candidate_ref_counts
        operator_counts = candidate_operator_counts
        break
    if selected is None or assigned_fragments is None:
        raise ValueError(
            f"no exact-capacity production selection after {selection_attempt_limit} "
            "deterministic salts; "
            f"last_errors={selection_errors[-3:]}"
        )

    variant_records = []
    for context, bundle in selected:
        for slot, pair in enumerate(bundle):
            variant_records.append({
                "key": f"{context['target_siret']}:v{slot + 1}",
                "context": context,
                "pair": pair,
                "difficulty": difficulty(context, pair),
                "archetypes": variant_archetypes(context, pair),
            })
    actual_difficulty_counts = Counter(
        value["difficulty"] for value in variant_records
    )
    stratum_counts = adapt_strata_to_easy_capacity(
        ideal_stratum_counts, stratum_maximum_additions,
        actual_difficulty_counts["EASY"],
    )
    stratum_assignment = assign_strata(variant_records, stratum_counts, catalog)

    output = []
    for context, bundle in selected:
        siret = context["target_siret"]
        target_values = fragments.baseline(context)
        contracts = []
        for slot, pair in enumerate(bundle):
            variant_id = f"v{slot + 1}"
            name_relation, (location_field, location_relation) = pair
            field_fragments = assigned_fragments[siret][slot]
            protected_name_anchor = (
                strict_token_subset_anchor(
                    context, field_fragments["name"], document_frequencies
                )
                if name_relation == "TOKEN_SUBSET" else None
            )
            if name_relation == "TOKEN_SUBSET" and not protected_name_anchor:
                raise RuntimeError(f"selected TOKEN_SUBSET lost its protected anchor: {siret}")
            key = f"{siret}:{variant_id}"
            variant_record = next(value for value in variant_records if value["key"] == key)
            stratum = stratum_assignment[key]
            cell = STRATUM_TO_OOF_CELL.get(stratum)
            contracts.append({
                "variant_id": variant_id,
                "requested_family": loop.COMPOSITE_FAMILY,
                "target_fields": ["name", location_field],
                "field_relations": {
                    "name": name_relation,
                    location_field: location_relation,
                },
                "field_inspirations": field_fragments,
                "protected_target_tokens": {
                    "name": [protected_name_anchor] if protected_name_anchor else []
                },
                "operator_guidance": {
                    field: {
                        "source_tokens_zero_indexed": list(enumerate(
                            loop.normalized_words(target_values[field])
                        )),
                        "expected_retained_tokens": [
                            loop.normalized_words(target_values[field])[index]
                            for index in fragment.get("operation_parameters", {}).get(
                                "retained_positions", []
                            )
                        ],
                    }
                    for field, fragment in field_fragments.items()
                },
                "difficulty": variant_record["difficulty"],
                "augmentation_stratum": stratum,
                "targeting_evidence": {
                    "catalog_sha256": plan["sources"]["train_oof_failure_catalog"]["sha256"],
                    "oof_failure_cell": cell,
                    "matched_archetypes": sorted(variant_record["archetypes"]),
                    "identity_free_aggregate": True,
                },
                "rules": {
                    "copy_non_target_fields_byte_for_byte": True,
                    "no_new_lexical_or_numeric_tokens": (
                        name_relation != "OFFICIAL_NAME_ALIAS"
                    ),
                    "official_alias_exact_value_authorized": (
                        name_relation == "OFFICIAL_NAME_ALIAS"
                    ),
                    "added_marks_forbidden": True,
                    "preserve_house_number": True,
                },
            })
        official = context["target"]
        seed_card = {
            "generation_mode": "OBSERVED_COMPOSITE_ANALOGY_V2",
            "name_options": [item["value"] for item in official["names"]],
            "enseigne_options": [
                item["value"] for item in official["names"] if item.get("kind") == "ENSEIGNE"
            ],
            "address": target_values["address"],
            "postcode": target_values["postcode"],
            "city": target_values["city"],
            "insee": target_values["insee"],
            "street_number": str(official["address"]["number"]),
            "street_type": str(official["address"]["street_type"]),
            "composite_contracts": contracts,
            "official_context": context["llm_view"],
            "qualification": context["qualification"],
            "internal_context": context["internal_context"],
            "context_sha256": context["context_sha256"],
        }
        output.append({
            "seed_id": f"BALANCED_V1_{args.batch_id}:{siret}",
            "target_siret": siret,
            "target_siren": context["target_siren"],
            "source_kind": "SIRENE_ONLY_TRAIN",
            "oof_fold": -1,
            "legacy_split": "train_synthetic",
            "seed_card": seed_card,
            "observed_train_profile": {
                "schema_version": "sireto-synthetic-field-evidence-1",
                "rows": sum(len(value) for value in grouped.values()),
                "source_sha256": sha256(source_paths["field_inspiration_bank"]),
                "source_folds": [2, 3, 4],
                "supported_families": [loop.COMPOSITE_FAMILY],
            },
            "risk_flags": [
                "BALANCED_PRODUCTION_COUNTED",
                "TRAIN_OOF_FAILURE_DIRECTED",
                "COMPOSITE_LLM_WRITTEN",
                "TARGET_CLOSED" if official["state"] == "F" else "TARGET_ACTIVE",
            ],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    loop.write_jsonl_atomic(args.output, output)
    manifest = {
        "schema_version": "sireto-synthetic-gt-balanced-selection-manifest-1",
        "batch_id": args.batch_id,
        "counts_toward_20000_target": True,
        "targets": len(output),
        "planned_variants": variant_count,
        "ideal_batch_counts": {
            "difficulty": difficulty_counts,
            "augmentation_strata": ideal_stratum_counts,
        },
        "difficulty_counts": dict(Counter(
            contract["difficulty"] for row in output
            for contract in row["seed_card"]["composite_contracts"]
        )),
        "augmentation_stratum_counts": dict(Counter(
            contract["augmentation_stratum"] for row in output
            for contract in row["seed_card"]["composite_contracts"]
        )),
        "state_counts": dict(Counter(
            row["seed_card"]["official_context"]["target"]["state"] for row in output
        )),
        "target_identity_active_share_bounds": list(target_active_share_bounds),
        "state_variant_counts": dict(Counter(
            row["seed_card"]["official_context"]["target"]["state"]
            for row in output
            for _contract in row["seed_card"]["composite_contracts"]
        )),
        "prior_promoted_state_variant_counts": dict(prior_state_variant_counts),
        "prior_promoted_state_target_counts": dict(prior_state_target_counts),
        "final_state_variant_targets": final_state_variant_counts,
        "contracts_per_target_counts": dict(Counter(
            len(row["seed_card"]["composite_contracts"]) for row in output
        )),
        "selection_capability_counts": capability_diagnostics,
        "relation_pair_counts": dict(Counter(
            "+".join(f"{field}:{relation}" for field, relation in sorted(
                contract["field_relations"].items()
            ))
            for row in output for contract in row["seed_card"]["composite_contracts"]
        )),
        "distinct_inspiration_refs": len(ref_counts),
        "maximum_inspiration_ref_uses": max(ref_counts.values(), default=0),
        "maximum_cumulative_inspiration_ref_uses": max(
            (prior_ref_counts + ref_counts).values(), default=0
        ),
        "distinct_exact_operators": len(operator_counts),
        "maximum_exact_operator_uses": max(operator_counts.values(), default=0),
        "maximum_cumulative_exact_operator_uses": max(
            (prior_operator_counts + operator_counts).values(), default=0
        ),
        "prior_counted_variants_reserved": prior_variant_count,
        "prior_counting_basis": (
            "FULL_SIRENE_EXACT_PROMOTIONS" if args.production_registry is not None
            else "LEGACY_PLANNED_SEED_INPUTS"
        ),
        "cumulative_planned_variants": cumulative_variant_count,
        "global_quota_targets": {
            "difficulty": exact_counts(
                final_variant_target, plan["corpus_balance"]["difficulty"]
            ),
            "augmentation_strata": exact_counts(
                final_variant_target, plan["corpus_balance"]["augmentation_strata"]
            ),
        },
        "prior_promoted_counts": {
            "difficulty": dict(prior_difficulty_counts),
            "augmentation_strata": dict(prior_stratum_counts),
        },
        "caps": {
            "inspiration_ref": int(plan["global_caps"]["inspiration_ref_uses"]),
            "exact_operator": operator_cap,
            "relation_pair": relation_pair_cap,
            "official_name_alias": official_name_alias_cap,
            "name_token_subset": name_token_subset_cap,
            "name_token_subset_signature": token_subset_signature_cap,
            "difficulty_hard_prefix": hard_prefix_cap,
            "unique_targets": maximum_unique_targets,
        },
        "maximum_batch_target_count": maximum_target_count,
        "cumulative_planned_unique_targets_upper_bound": (
            prior_distinct_target_count + len(output)
        ),
        "excluded_prior_sirets": len(excluded_sirets),
        "excluded_prior_sirens": len(excluded_sirens),
        "selection_seed": effective_selection_seed,
        "selection_flow_attempts": len(selection_errors) + 1,
        "source_hashes": {
            "plan": sha256(args.plan),
            **{key: sha256(path) for key, path in source_paths.items()},
            **{f"excluded:{path}": sha256(path) for path in args.exclude_seed_input},
            **{
                f"prior_counted:{path}": sha256(path)
                for path in args.prior_counted_seed_input
            },
            **({"production_registry": registry_hash} if registry_hash else {}),
        },
        "output_sha256": sha256(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--plan", type=Path,
        default=ROOT / "config/synthetic_gt_balanced_v1_plan.json",
    )
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--batch-id", default="P000")
    result.add_argument("--target-count", type=int, default=150)
    result.add_argument(
        "--variant-count", type=int, default=0,
        help=(
            "Exact number of contracts to select; zero keeps the historical "
            "three-per-target-count behavior. Used by residual final batches."
        ),
    )
    result.add_argument(
        "--selection-pool-limit", type=int, default=0,
        help="Bound costly capability materialization (0 = max(3000, 3*target-count)).",
    )
    result.add_argument(
        "--context-index-cache-directory", type=Path,
        help=(
            "Content-addressed official-context index directory (default: a "
            "derived cache next to the official-context artifact)."
        ),
    )
    result.add_argument(
        "--disable-context-index-cache", action="store_true",
        help="Compatibility/benchmark path: load the full official context in memory.",
    )
    result.add_argument(
        "--diagnose-infeasible", action="store_true",
        help="On one selection attempt, identify which constraint group causes infeasibility.",
    )
    result.add_argument(
        "--feasibility-only", action="store_true",
        help=(
            "Run one pure-feasibility MILP/flow attempt without the expensive "
            "infeasibility relaxation cascade."
        ),
    )
    result.add_argument(
        "--audit-exact-operator-cap", type=int, default=0,
        help=(
            "Read-only capacity audit override for the absolute exact-operator "
            "cap; requires --feasibility-only or --diagnose-infeasible."
        ),
    )
    result.add_argument(
        "--audit-inspiration-ref-cap", type=int, default=0,
        help=(
            "Read-only capacity audit override for the absolute inspiration-ref "
            "cap; requires --feasibility-only or --diagnose-infeasible."
        ),
    )
    result.add_argument("--selection-seed", default="SIRETO-BALANCED-P000-V1")
    result.add_argument("--exclude-seed-input", type=Path, action="append", default=[])
    result.add_argument(
        "--prior-counted-seed-input", type=Path, action="append", default=[],
        help=(
            "Prior counted production input: excludes its targets and reserves its "
            "reference/operator/relation capacities from global corpus caps."
        ),
    )
    result.add_argument(
        "--production-registry", type=Path,
        help=(
            "Sealed balanced-production registry. Only its full-SIRENE exact "
            "promotions reserve cumulative quotas and caps; every registered target "
            "SIRET/SIREN is excluded from later batches."
        ),
    )
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
