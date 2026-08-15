#!/usr/bin/env python3
"""Select a balanced field-fragment composite pilot without writing CRM text."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence
import unicodedata

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402
from scripts import select_synthetic_gt_composite_pilot as legacy  # noqa: E402


NAME_QUOTAS = {
    "TOKEN_SUBSET": 6,
    "TOKEN_ORDER": 6,
    "LEGAL_FORM_REMOVE": 8,
    "PUNCTUATION_REMOVED": 10,
}
# Frozen before pilot30_v2 generation from the expanded official-only context.
# The quotas keep every name relation represented without letting the easy
# subset transformation dominate the pilot.
PILOT30_NAME_QUOTAS = {
    "TOKEN_SUBSET": 24,
    "TOKEN_ORDER": 12,
    "LEGAL_FORM_REMOVE": 24,
    "PUNCTUATION_REMOVED": 30,
}
LOCATION_QUOTAS = {
    ("address", "ADDRESS_ABBREVIATE"): 5,
    ("address", "ADDRESS_ALIAS_EXPAND"): 3,
    ("address", "ADDRESS_TOKEN_SUBSET"): 8,
    ("address", "PUNCTUATION_REMOVED"): 7,
    ("city", "PUNCTUATION_REMOVED"): 7,
}
PILOT30_LOCATION_QUOTAS = {
    ("address", "ADDRESS_ABBREVIATE"): 18,
    ("address", "ADDRESS_ALIAS_EXPAND"): 6,
    ("address", "ADDRESS_TOKEN_SUBSET"): 36,
    ("address", "PUNCTUATION_REMOVED"): 2,
    ("city", "PUNCTUATION_REMOVED"): 28,
}
QUOTA_UNIT_TARGETS = 10
STOP_ANCHORS = {
    "ASS", "ASSOCIATION", "CABINET", "CENTRE", "DE", "DES", "DU", "ET", "LA", "LE",
    "LES", "MAISON", "SAS", "SARL", "SCI", "SOCIETE",
}
ADDRESS_FUNCTION_WORDS = {"d", "de", "des", "du", "l", "la", "le", "les"}
NAME_FUNCTION_TERMINALS = {
    "a", "au", "aux", "d", "da", "das", "de", "del", "della", "des", "do", "dos",
    "du", "en", "et", "l", "la", "le", "les", "sous", "sur", "van", "von",
}
ROMAN_NUMERAL = re.compile(r"[ivxlcdm]+", re.IGNORECASE)


def sha256(path: Path) -> str:
    return legacy.sha256(path)


def rows(path: Path) -> list[dict[str, Any]]:
    return legacy.rows(path)


def excluded_target_ids(paths: Sequence[Path]) -> tuple[set[str], set[str]]:
    sirets: set[str] = set()
    sirens: set[str] = set()
    for path in paths:
        for value in rows(path):
            siret = str(value.get("target_siret") or "").strip()
            siren = str(value.get("target_siren") or "").strip()
            if siret:
                sirets.add(siret)
            if siren:
                sirens.add(siren)
    return sirets, sirens


def baseline(context: dict[str, Any]) -> dict[str, str]:
    target = context["target"]
    return {
        "name": legacy.primary_name(context),
        "address": legacy.address_line(context),
        "postcode": str(target["address"]["postcode"]),
        "city": str(target["address"]["city"]),
        "insee": str(target["address"]["insee"]),
    }


def candidate_names(candidate: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(value).strip()
        for value in (
            list(candidate.get("name_values", []))
            + [candidate.get(key) for key in (
                "usual_name", "enseigne1", "enseigne2", "enseigne3", "legal_name"
            )]
        )
        if str(value or "").strip()
    ))


def name_token_document_frequencies(
    contexts: Sequence[dict[str, Any]],
) -> Counter[str]:
    """Count official-name tokens across the frozen target population."""
    return Counter(
        word
        for context in contexts
        for word in set(loop.normalized_words(legacy.primary_name(context)))
    )


def distinctive_name_tokens(
    context: dict[str, Any],
    document_frequencies: Counter[str] | None = None,
) -> list[str]:
    target_words = loop.normalized_words(legacy.primary_name(context))
    # These are identity anchors, not local-discrimination features.  A rare
    # official token remains part of the target identity even when a neighbour
    # shares it; exact local ambiguity is handled independently by full G_N_A.
    candidates = list(dict.fromkeys(
        word for word in target_words
        if (
            word.upper() not in STOP_ANCHORS
            and word.upper() not in loop.LEGAL_FORM_TOKENS
            and len(word) >= 2
            and not word.isdigit()
            and ROMAN_NUMERAL.fullmatch(word) is None
        )
    ))
    if document_frequencies is None:
        return candidates
    source_order = {word: index for index, word in enumerate(candidates)}
    return sorted(
        candidates,
        key=lambda word: (
            document_frequencies.get(word, 0), -len(word), source_order[word]
        ),
    )


def punctuation_boundaries(value: str) -> set[tuple[int, str]]:
    marks: set[tuple[int, str]] = set()
    token_index = -1
    in_token = False
    for character in value:
        if character.isalnum():
            if not in_token:
                token_index += 1
                in_token = True
        else:
            if character in loop.PUNCTUATION:
                marks.add((token_index, character))
            if character.isspace() or character in loop.PUNCTUATION:
                in_token = False
    return marks


def connected_token_pairs(value: str) -> set[tuple[int, int]]:
    """Return token positions joined by punctuation without whitespace."""
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    plain = "".join(
        character for character in decomposed
        if not unicodedata.combining(character)
    )
    matches = list(re.finditer(r"[a-z0-9]+", plain))
    result: set[tuple[int, int]] = set()
    for index, (left, right) in enumerate(zip(matches, matches[1:])):
        separator = plain[left.end():right.start()]
        if separator and not any(character.isspace() for character in separator) and any(
            character in loop.PUNCTUATION for character in separator
        ):
            result.add((index, index + 1))
    return result


def fragment_supports(
    field: str,
    source_value: str,
    fragment: dict[str, Any],
    anchors: list[str],
) -> bool:
    if fragment.get("field") != field or fragment.get("source_fold") not in {2, 3, 4}:
        return False
    relation = fragment.get("relation")
    if relation in {"DIACRITIC_ADDED", "PUNCTUATION_ADDED"}:
        return False
    parameters = fragment.get("operation_parameters", {})
    words = loop.normalized_words(source_value)
    source_count = parameters.get("source_token_count")
    if source_count is not None and source_count != len(words):
        return False
    if relation in {"TOKEN_SUBSET", "ADDRESS_TOKEN_SUBSET", "LEGAL_FORM_REMOVE"}:
        retained = parameters.get("retained_positions")
        if (
            not isinstance(retained, list)
            or not retained
            or max(retained, default=-1) >= len(words)
        ):
            return False
        if field == "name" and relation == "TOKEN_SUBSET":
            retained_set = set(retained)
            removed_words = [
                word for index, word in enumerate(words) if index not in retained_set
            ]
            return (
                len(retained) >= 2
                and len(words) >= 4
                and len(retained) * 2 >= len(words)
                and (len(words) < 4 or len(retained) >= 3)
                and words[retained[-1]] not in NAME_FUNCTION_TERMINALS
                and not all(word.upper() in loop.LEGAL_FORM_TOKENS for word in removed_words)
                # Luna reasons about apostrophised/hyphenated compounds as visual
                # units, whereas the deterministic runtime indexes their lexical
                # components separately.  A positional subset on such a source is
                # therefore not safely transferable, even if it retains the whole
                # connected group (for example J'ENTENDS).
                and not connected_token_pairs(source_value)
                and not any(character in loop.PUNCTUATION for character in source_value)
                and bool(anchors)
                and anchors[0] in {words[index] for index in retained}
            )
        if relation == "LEGAL_FORM_REMOVE":
            removed = [words[index] for index in range(len(words)) if index not in retained]
            return (
                bool(removed)
                and all(value.upper() in loop.LEGAL_FORM_TOKENS for value in removed)
                and [value.casefold() for value in removed]
                == [str(value).casefold() for value in parameters.get("removed_legal_forms", [])]
            )
        if relation == "ADDRESS_TOKEN_SUBSET":
            retained_words = [words[index] for index in retained]
            removed_indices = [index for index in range(len(words)) if index not in retained]
            first_non_digit = next(
                (index for index, value in enumerate(words) if not value.isdigit()), None
            )
            return (
                [value for value in retained_words if value.isdigit()]
                == [value for value in words if value.isdigit()]
                and first_non_digit in retained
                and len(removed_indices) == 1
                and words[removed_indices[0]] in ADDRESS_FUNCTION_WORDS
                and any(
                    not value.isdigit()
                    and index != first_non_digit
                    and value not in ADDRESS_FUNCTION_WORDS
                    for index, value in enumerate(words)
                    if index in retained
                )
            )
        return True
    if relation in {"TOKEN_ORDER", "ADDRESS_TYPE_ORDER"}:
        permutation = parameters.get("permutation")
        if not isinstance(permutation, list) or sorted(permutation) != list(range(len(words))):
            return False
        if relation == "ADDRESS_TYPE_ORDER":
            return True
        # Arbitrary long permutations transferred poorly in the first canary.
        # Admit only a local adjacent swap or moving an explicit legal form
        # between the two ends.  This remains selector-only: Luna still writes
        # the resulting CRM text and the runtime validates the exact operator.
        moved = [index for index, source_index in enumerate(permutation) if index != source_index]
        target_words = [words[index] for index in permutation]
        legal_form_end_move = bool(words) and (
            (words[0].upper() in loop.LEGAL_FORM_TOKENS and target_words[-1] == words[0])
            or (words[-1].upper() in loop.LEGAL_FORM_TOKENS and target_words[0] == words[-1])
        )
        return legal_form_end_move
    if relation in {"ADDRESS_ABBREVIATE", "ADDRESS_ALIAS_EXPAND"}:
        pairs = parameters.get("pairs", [])
        source_words = loop.normalized_words(source_value)
        pair_source = str(pairs[0].get("source", "")).casefold() if len(pairs) == 1 else ""
        pair_target = str(pairs[0].get("target", "")).casefold() if len(pairs) == 1 else ""
        matching_positions = [
            index for index, token in enumerate(source_words) if token == pair_source
        ]
        projected_words = list(source_words)
        if len(matching_positions) == 1:
            projected_words[matching_positions[0]] = pair_target
        return (
            len(pairs) == 1
            and len(matching_positions) == 1
            and loop.address_alias_relation(
                source_value, " ".join(projected_words)
            ) == relation
        )
    if relation == "PUNCTUATION_REMOVED":
        edits = parameters.get("edits", [])
        available = punctuation_boundaries(source_value)
        requested_marks = Counter(str(value.get("mark", "")) for value in edits)
        source_marks = Counter(
            character for character in source_value if character in loop.PUNCTUATION
        )
        return bool(edits) and all(
            (int(value.get("after_token_index", -99)), str(value.get("mark", ""))) in available
            for value in edits
        ) and all(
            value.get("replacement") in {"", " "} for value in edits
        ) and all(
            source_marks[mark] == count for mark, count in requested_marks.items()
        )
    if relation == "JOIN_SPLIT":
        groups = parameters.get("groups")
        return (
            isinstance(groups, list)
            and [index for group in groups for index in group] == list(range(len(words)))
            and any(len(group) > 1 for group in groups)
        )
    return False


def group_fragments(values: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for value in values:
        result[(str(value.get("field")), str(value.get("relation")))].append(value)
    for key in result:
        result[key].sort(key=lambda value: value["inspiration_ref"])
    return result


def scaled_quotas(values: dict[Any, int], target_count: int) -> dict[Any, int]:
    """Scale the preregistered canary mix to whole ten-target pilot blocks."""
    if target_count < QUOTA_UNIT_TARGETS or target_count % QUOTA_UNIT_TARGETS:
        raise ValueError("target_count must be a positive multiple of 10")
    multiplier = target_count // QUOTA_UNIT_TARGETS
    return {key: value * multiplier for key, value in values.items()}


def name_quotas_for_target_count(target_count: int) -> dict[str, int]:
    if target_count == 10:
        return dict(NAME_QUOTAS)
    if target_count == 30:
        return dict(PILOT30_NAME_QUOTAS)
    raise ValueError("only the preregistered canary10 and pilot30 sizes are supported")


def location_quotas_for_target_count(
    target_count: int,
) -> dict[tuple[str, str], int]:
    if target_count == 10:
        return dict(LOCATION_QUOTAS)
    if target_count == 30:
        return dict(PILOT30_LOCATION_QUOTAS)
    raise ValueError("only the preregistered canary10 and pilot30 sizes are supported")


def target_capabilities(
    context: dict[str, Any],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    document_frequencies: Counter[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    values = baseline(context)
    anchors = distinctive_name_tokens(context, document_frequencies)
    names = {
        relation: [
            fragment for fragment in grouped[("name", relation)]
            if fragment_supports("name", values["name"], fragment, anchors)
        ]
        for relation in NAME_QUOTAS
    }
    locations = {
        key: [
            fragment for fragment in grouped[key]
            if fragment_supports(key[0], values[key[0]], fragment, anchors)
        ]
        for key in LOCATION_QUOTAS
    }
    return names, locations


def relation_assignment(
    targets: list[dict[str, Any]],
    quotas: dict[Any, int],
    capabilities: dict[str, dict[Any, list[dict[str, Any]]]],
    *,
    distinct_per_target: bool,
) -> dict[str, list[Any]] | None:
    graph = nx.DiGraph()
    source_node, sink_node = "SOURCE", "SINK"
    total = sum(quotas.values())
    for relation, quota in quotas.items():
        relation_node = ("REL", relation)
        graph.add_edge(source_node, relation_node, capacity=quota)
        for target in targets:
            siret = target["target_siret"]
            if capabilities[siret].get(relation):
                node = ("TARGET_REL", siret, relation)
                graph.add_edge(relation_node, node, capacity=1 if distinct_per_target else 2)
                graph.add_edge(node, ("TARGET", siret), capacity=1 if distinct_per_target else 2)
    for target in targets:
        graph.add_edge(("TARGET", target["target_siret"]), sink_node, capacity=3)
    flow_value, flow = nx.maximum_flow(
        graph, source_node, sink_node, flow_func=nx.algorithms.flow.edmonds_karp
    )
    if flow_value != total:
        return None
    result: dict[str, list[Any]] = defaultdict(list)
    for relation in quotas:
        relation_node = ("REL", relation)
        for node, amount in flow.get(relation_node, {}).items():
            if amount:
                result[node[1]].extend([relation] * amount)
    return result if all(len(result[target["target_siret"]]) == 3 for target in targets) else None


def assign_unique_fragments(
    requests: list[tuple[str, int, str, str]],
    target_values: dict[str, dict[str, str]],
    target_anchors: dict[str, list[str]],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[tuple[str, int, str], dict[str, Any]] | None:
    graph = nx.DiGraph()
    source_node, sink_node = "SOURCE", "SINK"
    left_nodes = []
    by_ref = {
        value["inspiration_ref"]: value for values in grouped.values() for value in values
    }
    for siret, slot, field, relation in requests:
        node = ("REQUEST", siret, slot, field)
        left_nodes.append(node)
        graph.add_edge(source_node, node, capacity=1)
        fragments = grouped[(field, relation)]
        for fragment in fragments:
            if fragment_supports(
                field, target_values[siret][field], fragment, target_anchors[siret]
            ):
                ref_node = ("FRAGMENT", fragment["inspiration_ref"])
                target_ref_node = ("TARGET_FRAGMENT", siret, fragment["inspiration_ref"])
                operator = loop.digest_json({
                    "field": field, "relation": relation,
                    "parameters": fragment["operation_parameters"],
                })
                operator_node = ("OPERATOR", operator)
                graph.add_edge(node, target_ref_node, capacity=1)
                graph.add_edge(target_ref_node, ref_node, capacity=1)
                graph.add_edge(ref_node, operator_node, capacity=3)
                graph.add_edge(operator_node, sink_node, capacity=4)
    flow_value, flow = nx.maximum_flow(
        graph, source_node, sink_node, flow_func=nx.algorithms.flow.edmonds_karp
    )
    if flow_value != len(left_nodes):
        return None
    result = {}
    for node in left_nodes:
        target_refs = [
            target for target, amount in flow[node].items()
            if amount and isinstance(target, tuple) and target[0] == "TARGET_FRAGMENT"
        ]
        selected = [
            target for target, amount in flow[target_refs[0]].items()
            if amount and isinstance(target, tuple) and target[0] == "FRAGMENT"
        ] if len(target_refs) == 1 else []
        if len(selected) != 1:
            return None
        result[(node[1], node[2], node[3])] = by_ref[selected[0][1]]
    return result


def assign_fragment_plan(
    targets: list[dict[str, Any]],
    quotas: dict[Any, int],
    target_values: dict[str, dict[str, str]],
    target_anchors: dict[str, list[str]],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    distinct_per_target: bool,
    exact_operator_cap: int = 4,
    inspiration_ref_cap: int = 3,
    prior_operator_counts: Counter[str] | None = None,
    prior_ref_counts: Counter[str] | None = None,
) -> dict[str, list[dict[str, Any]]] | None:
    """Jointly assign relations and evidence fragments under all reuse caps."""
    graph = nx.DiGraph()
    source_node, sink_node = "SOURCE", "SINK"
    by_ref = {
        value["inspiration_ref"]: value for values in grouped.values() for value in values
    }
    prior_operator_counts = prior_operator_counts or Counter()
    prior_ref_counts = prior_ref_counts or Counter()
    target_ids = sorted(target["target_siret"] for target in targets)
    for relation_key, quota in quotas.items():
        field, relation = (
            relation_key if isinstance(relation_key, tuple) else ("name", relation_key)
        )
        relation_node = ("RELATION", field, relation)
        graph.add_edge(source_node, relation_node, capacity=quota)
        for fragment in grouped[(field, relation)]:
            operator = loop.digest_json({
                "field": field,
                "relation": relation,
                "parameters": fragment["operation_parameters"],
            })
            operator_node = ("OPERATOR", operator)
            ref_node = ("FRAGMENT", fragment["inspiration_ref"])
            operator_capacity = exact_operator_cap - prior_operator_counts[operator]
            ref_capacity = inspiration_ref_cap - prior_ref_counts[fragment["inspiration_ref"]]
            if operator_capacity <= 0 or ref_capacity <= 0:
                continue
            graph.add_edge(relation_node, operator_node, capacity=operator_capacity)
            graph.add_edge(operator_node, ref_node, capacity=ref_capacity)
            for target in targets:
                siret = target["target_siret"]
                if not fragment_supports(
                    field, target_values[siret][field], fragment, target_anchors[siret]
                ):
                    continue
                target_ref = ("TARGET_FRAGMENT", siret, fragment["inspiration_ref"])
                target_relation = ("TARGET_RELATION", siret, field, relation)
                graph.add_edge(ref_node, target_ref, capacity=1)
                graph.add_edge(target_ref, target_relation, capacity=1)
                graph.add_edge(
                    target_relation,
                    ("TARGET", siret),
                    capacity=1 if distinct_per_target else 2,
                )
    for siret in target_ids:
        graph.add_edge(("TARGET", siret), sink_node, capacity=3)
    flow_value, flow = nx.maximum_flow(
        graph, source_node, sink_node, flow_func=nx.algorithms.flow.edmonds_karp
    )
    if flow_value != sum(quotas.values()):
        return None
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node, outgoing in flow.items():
        if not isinstance(node, tuple) or node[0] != "FRAGMENT":
            continue
        for target_node, amount in outgoing.items():
            if amount and isinstance(target_node, tuple) and target_node[0] == "TARGET_FRAGMENT":
                result[target_node[1]].append(by_ref[node[1]])
    if any(len(result[siret]) != 3 for siret in target_ids):
        return None
    for siret in result:
        result[siret].sort(key=lambda value: (value["field"], value["relation"], value["inspiration_ref"]))
    return dict(result)


def fragment_operator(fragment: dict[str, Any]) -> str:
    return loop.digest_json({
        "field": fragment["field"],
        "relation": fragment["relation"],
        "parameters": fragment["operation_parameters"],
    })


def update_plan_counts(
    plan: dict[str, list[dict[str, Any]]],
    operator_counts: Counter[str],
    ref_counts: Counter[str],
) -> None:
    for fragments in plan.values():
        for fragment in fragments:
            operator_counts[fragment_operator(fragment)] += 1
            ref_counts[fragment["inspiration_ref"]] += 1


def pair_fragment_plans(
    names: list[dict[str, Any]], locations: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """Pair independently assigned fields without repeating a composite operator."""
    if len(names) != 3 or len(locations) != 3:
        return None
    name_operators = [fragment_operator(value) for value in names]
    for candidate in itertools.permutations(locations):
        operator_pairs = {
            (name_operators[index], fragment_operator(candidate[index]))
            for index in range(3)
        }
        if len(operator_pairs) == 3:
            return list(candidate)
    return None


def pair_fragment_plans_globally(
    name_plan: dict[str, list[dict[str, Any]]],
    location_plan: dict[str, list[dict[str, Any]]],
    *,
    composite_signature_cap: int,
) -> dict[str, list[dict[str, Any]]] | None:
    candidates: dict[str, list[tuple[dict[str, Any], ...]]] = {}
    for siret, names in name_plan.items():
        name_operators = [fragment_operator(value) for value in names]
        valid = []
        for permutation in itertools.permutations(location_plan[siret]):
            operator_pairs = [
                (name_operators[index], fragment_operator(permutation[index]))
                for index in range(3)
            ]
            if len(set(operator_pairs)) == 3:
                valid.append(permutation)
        if not valid:
            return None
        candidates[siret] = valid

    options: list[tuple[str, tuple[dict[str, Any], ...], tuple[tuple[str, str], ...]]] = []
    for siret in sorted(candidates):
        names = name_plan[siret]
        for permutation in candidates[siret]:
            signatures = tuple(
                (fragment_operator(names[index]), fragment_operator(permutation[index]))
                for index in range(3)
            )
            options.append((siret, permutation, signatures))
    variable_count = len(options)
    constraint_rows: list[np.ndarray] = []
    minima: list[float] = []
    maxima: list[float] = []
    for siret in sorted(candidates):
        coefficients = np.zeros(variable_count)
        for index, option in enumerate(options):
            coefficients[index] = option[0] == siret
        constraint_rows.append(coefficients)
        minima.append(1)
        maxima.append(1)
    all_signatures = sorted({value for _, _, signatures in options for value in signatures})
    for signature in all_signatures:
        coefficients = np.zeros(variable_count)
        for index, option in enumerate(options):
            coefficients[index] = signature in option[2]
        constraint_rows.append(coefficients)
        minima.append(0)
        maxima.append(composite_signature_cap)
    objective = np.zeros(variable_count)
    for index, (siret, permutation, _signatures) in enumerate(options):
        refs = "|".join(value["inspiration_ref"] for value in permutation)
        digest = hashlib.sha256(f"{siret}|{refs}".encode()).hexdigest()
        objective[index] = int(digest[:16], 16) / float(2**64)
    solution = milp(
        objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(np.asarray(constraint_rows), minima, maxima),
    )
    if not solution.success or solution.x is None:
        return None
    return {
        siret: list(permutation)
        for index, (siret, permutation, _signatures) in enumerate(options)
        if solution.x[index] > 0.5
    }


def eligible(context: dict[str, Any]) -> bool:
    value = baseline(context)
    address = context["target"]["address"]
    return bool(
        context.get("qualification", {}).get("pre_generation_exact_eligible")
        and len(loop.normalized_words(value["name"])) >= 3
        and distinctive_name_tokens(context)
        and address.get("number") and address.get("street_type") and address.get("street")
        and address.get("postcode") and address.get("insee") and address.get("city")
    )


def is_multi(context: dict[str, Any]) -> bool:
    return any("SAME_SIREN" in value.get("relation_tags", []) for value in context["internal_context"])


def is_multi_active(context: dict[str, Any]) -> bool:
    return any(
        "SAME_SIREN" in value.get("relation_tags", []) and value.get("state") == "A"
        for value in context["internal_context"]
    )


def select_integrated_fragment_targets(
    contexts: list[dict[str, Any]],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    document_frequencies: Counter[str],
    selection_seed: str,
    target_count: int,
    name_quotas: dict[str, int],
    location_quotas: dict[tuple[str, str], int],
    exact_operator_cap: int,
    inspiration_ref_cap: int,
    minimum_multi_active: int,
    minimum_multi: int,
    name_quota_bounds: dict[str, tuple[int, int]] | None = None,
    location_quota_bounds: dict[tuple[str, str], tuple[int, int]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[Any, int]],
]:
    """Solve target, relation and exact-operator capacities in one MILP."""
    candidates: list[dict[str, Any]] = []
    capabilities: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}
    for context in contexts:
        if not eligible(context):
            continue
        names, locations = target_capabilities(context, grouped, document_frequencies)
        if sum(bool(value) for value in names.values()) < 2 or not any(locations.values()):
            continue
        siret = context["target_siret"]
        capabilities[siret] = {
            **{("name", relation): values for relation, values in names.items()},
            **locations,
        }
        candidates.append(context)
    candidates.sort(key=lambda value: value["target_siret"])

    target_variables = len(candidates)
    assignment_specs: list[dict[str, Any]] = []
    operator_refs: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for target_index, context in enumerate(candidates):
        siret = context["target_siret"]
        for (field, relation), fragments in capabilities[siret].items():
            by_operator: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
            for fragment in fragments:
                operator = fragment_operator(fragment)
                by_operator[operator][fragment["inspiration_ref"]] = fragment
                operator_refs[(field, relation, operator)][fragment["inspiration_ref"]] = fragment
            for operator, by_ref in by_operator.items():
                assignment_specs.append({
                    "target_index": target_index,
                    "siret": siret,
                    "field": field,
                    "relation": relation,
                    "operator": operator,
                    "refs": tuple(sorted(by_ref)),
                })

    variable_count = target_variables + len(assignment_specs)
    lower = np.zeros(variable_count)
    upper = np.ones(variable_count)
    integrality = np.ones(variable_count)
    for offset, spec in enumerate(assignment_specs, start=target_variables):
        # One target may reuse a broad relation, but never the same exact
        # operator twice.  Otherwise two variants can collapse to the same CRM
        # transformation even when they cite distinct inspiration rows.
        upper[offset] = min(1, len(spec["refs"]))

    constraint_rows: list[dict[int, float]] = []
    minima: list[float] = []
    maxima: list[float] = []

    def add(coefficients: dict[int, float], minimum: float, maximum: float) -> None:
        constraint_rows.append(coefficients)
        minima.append(minimum)
        maxima.append(maximum)

    add({index: 1 for index in range(target_variables)}, target_count, target_count)
    for state in ("A", "F"):
        add({
            index: 1 for index, context in enumerate(candidates)
            if context["target"]["state"] == state
        }, target_count // 2, target_count // 2)
    add({
        index: 1 for index, context in enumerate(candidates) if is_multi(context)
    }, minimum_multi, np.inf)
    add({
        index: 1 for index, context in enumerate(candidates) if is_multi_active(context)
    }, minimum_multi_active, np.inf)

    assignment_indexes: dict[tuple[int, str], list[int]] = defaultdict(list)
    relation_indexes: dict[tuple[str, str], list[int]] = defaultdict(list)
    target_relation_indexes: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    operator_indexes: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for offset, spec in enumerate(assignment_specs, start=target_variables):
        target_index = spec["target_index"]
        field_group = "name" if spec["field"] == "name" else "location"
        assignment_indexes[(target_index, field_group)].append(offset)
        relation_indexes[(spec["field"], spec["relation"])].append(offset)
        target_relation_indexes[(target_index, spec["field"], spec["relation"])].append(offset)
        operator_indexes[(spec["field"], spec["relation"], spec["operator"])].append(offset)

    for target_index in range(target_variables):
        for field_group in ("name", "location"):
            coefficients = {
                index: 1 for index in assignment_indexes[(target_index, field_group)]
            }
            coefficients[target_index] = -3
            add(coefficients, 0, 0)
        for field, relation in list(relation_indexes):
            indexes = target_relation_indexes[(target_index, field, relation)]
            if indexes:
                coefficients = {index: 1 for index in indexes}
                coefficients[target_index] = -2
                add(coefficients, -np.inf, 0)

    for relation, quota in name_quotas.items():
        minimum, maximum = (
            name_quota_bounds[relation] if name_quota_bounds else (quota, quota)
        )
        add(
            {index: 1 for index in relation_indexes[("name", relation)]},
            minimum, maximum,
        )
    for (field, relation), quota in location_quotas.items():
        minimum, maximum = (
            location_quota_bounds[(field, relation)]
            if location_quota_bounds else (quota, quota)
        )
        add(
            {index: 1 for index in relation_indexes[(field, relation)]},
            minimum, maximum,
        )
    for operator_key, indexes in operator_indexes.items():
        ref_capacity = len(operator_refs[operator_key]) * inspiration_ref_cap
        add({index: 1 for index in indexes}, 0, min(exact_operator_cap, ref_capacity))

    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    for row_index, coefficients in enumerate(constraint_rows):
        for column_index, coefficient in coefficients.items():
            matrix_rows.append(row_index)
            matrix_columns.append(column_index)
            matrix_values.append(coefficient)
    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(constraint_rows), variable_count),
    ).tocsr()
    objective = np.zeros(variable_count)
    for index, context in enumerate(candidates):
        digest = hashlib.sha256(
            f"{selection_seed}|integrated-target|{context['target_siret']}".encode()
        ).hexdigest()
        objective[index] = int(digest[:16], 16) / float(2**64)
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, minima, maxima),
        options={"time_limit": 120},
    )
    if not result.success or result.x is None:
        raise ValueError("integrated fragment MILP is infeasible")

    selected = [
        context for index, context in enumerate(candidates) if result.x[index] > 0.5
    ]
    resolved_name_quotas = {
        relation: int(round(sum(
            result.x[index] for index in relation_indexes[("name", relation)]
        )))
        for relation in name_quotas
    }
    resolved_location_quotas = {
        (field, relation): int(round(sum(
            result.x[index] for index in relation_indexes[(field, relation)]
        )))
        for field, relation in location_quotas
    }
    selected_sirets = {value["target_siret"] for value in selected}
    refs_by_target_operator = {
        (spec["siret"], spec["field"], spec["relation"], spec["operator"]): spec["refs"]
        for spec in assignment_specs if spec["siret"] in selected_sirets
    }
    plans: dict[str, list[dict[str, Any]]] = defaultdict(list)
    quota_groups = (
        ("name", {
            ("name", relation): quota
            for relation, quota in resolved_name_quotas.items()
        }),
        ("location", dict(resolved_location_quotas)),
    )
    for field_group, quotas in quota_groups:
        graph = nx.DiGraph()
        source_node, sink_node = "SOURCE", "SINK"
        for (field, relation), quota in quotas.items():
            relation_node = ("RELATION", field, relation)
            graph.add_edge(source_node, relation_node, capacity=quota, weight=0)
            for operator_key, by_ref in sorted(operator_refs.items()):
                if operator_key[:2] != (field, relation):
                    continue
                operator_node = ("OPERATOR", *operator_key)
                graph.add_edge(
                    relation_node, operator_node, capacity=exact_operator_cap, weight=0
                )
                for ref in sorted(by_ref):
                    ref_node = ("REF", *operator_key, ref)
                    graph.add_edge(
                        operator_node, ref_node, capacity=inspiration_ref_cap, weight=0
                    )
                    for siret in sorted(selected_sirets):
                        compatible_refs = refs_by_target_operator.get(
                            (siret, field, relation, operator_key[2]), ()
                        )
                        if ref not in compatible_refs:
                            continue
                        target_relation_node = ("TARGET_RELATION", siret, field, relation)
                        digest = hashlib.sha256(
                            f"{selection_seed}|flow|{field}|{relation}|{operator_key[2]}|{ref}|{siret}".encode()
                        ).hexdigest()
                        graph.add_edge(
                            ref_node, target_relation_node, capacity=1,
                            weight=int(digest[:8], 16),
                        )
        for siret in sorted(selected_sirets):
            target_group_node = ("TARGET_GROUP", field_group, siret)
            for field, relation in quotas:
                target_relation_node = ("TARGET_RELATION", siret, field, relation)
                if target_relation_node in graph:
                    graph.add_edge(
                        target_relation_node, target_group_node, capacity=2, weight=0
                    )
            graph.add_edge(target_group_node, sink_node, capacity=3, weight=0)
        flow = nx.max_flow_min_cost(graph, source_node, sink_node)
        flow_value = sum(flow[source_node].values())
        if flow_value != sum(quotas.values()):
            raise ValueError(f"integrated {field_group} ref flow is infeasible")
        for operator_key, by_ref in operator_refs.items():
            if (operator_key[0] == "name") != (field_group == "name"):
                continue
            for ref, fragment in by_ref.items():
                ref_node = ("REF", *operator_key, ref)
                for target_node, amount in flow.get(ref_node, {}).items():
                    if amount and isinstance(target_node, tuple) and target_node[0] == "TARGET_RELATION":
                        plans[target_node[1]].append(fragment)

    name_plan = {
        siret: sorted(
            [value for value in plans[siret] if value["field"] == "name"],
            key=lambda value: (value["relation"], value["inspiration_ref"]),
        )
        for siret in plans
    }
    location_plan = {
        siret: sorted(
            [value for value in plans[siret] if value["field"] != "name"],
            key=lambda value: (value["field"], value["relation"], value["inspiration_ref"]),
        )
        for siret in plans
    }
    if any(len(name_plan.get(value["target_siret"], [])) != 3 for value in selected):
        raise RuntimeError("integrated name plan is incomplete")
    if any(len(location_plan.get(value["target_siret"], [])) != 3 for value in selected):
        raise RuntimeError("integrated location plan is incomplete")
    paired_locations = pair_fragment_plans_globally(
        name_plan, location_plan, composite_signature_cap=3
    )
    if paired_locations is None:
        raise ValueError("integrated fragment pairing failed")
    location_plan = paired_locations
    selected.sort(key=lambda value: hashlib.sha256(
        f"{selection_seed}|integrated-final|{value['target_siret']}".encode()
    ).hexdigest())
    return selected, name_plan, location_plan, {
        "name": resolved_name_quotas,
        "location": resolved_location_quotas,
    }


def select_feasible_targets(
    contexts: list[dict[str, Any]],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    document_frequencies: Counter[str],
    selection_seed: str,
    requested_target_count: int,
    name_quotas: dict[str, int],
    location_quotas: dict[tuple[str, str], int],
    exact_operator_cap: int,
    inspiration_ref_cap: int,
    prior_operator_counts: Counter[str] | None = None,
    prior_ref_counts: Counter[str] | None = None,
    minimum_multi_active: int | None = None,
    minimum_multi: int | None = None,
    use_derived_capability_minima: bool = False,
) -> tuple[
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    candidates: list[dict[str, Any]] = []
    all_caps: dict[str, Any] = {}
    for context in contexts:
        if not eligible(context):
            continue
        name_caps, location_caps = target_capabilities(
            context, grouped, document_frequencies
        )
        if (
            sum(bool(value) for value in name_caps.values()) < 2
            or not any(location_caps.values())
        ):
            continue
        siret = context["target_siret"]
        all_caps[siret] = {"name": name_caps, "location": location_caps}
        candidates.append(context)
    candidates.sort(key=lambda value: value["target_siret"])

    name_relations = list(name_quotas)
    location_relations = list(location_quotas)
    candidate_count = len(candidates)
    name_offset = candidate_count
    location_offset = name_offset + candidate_count * len(name_relations)
    variable_count = location_offset + candidate_count * len(location_relations)
    lower = np.zeros(variable_count)
    upper = np.full(variable_count, 2.0)
    upper[:candidate_count] = 1.0
    integrality = np.ones(variable_count)
    base_rows: list[np.ndarray] = []
    base_lower: list[float] = []
    base_upper: list[float] = []

    def constraint(coefficients: np.ndarray, minimum: float, maximum: float) -> None:
        base_rows.append(coefficients)
        base_lower.append(minimum)
        base_upper.append(maximum)

    prior_operator_counts = prior_operator_counts or Counter()
    prior_ref_counts = prior_ref_counts or Counter()

    def per_target_relation_capacity(fragments: list[dict[str, Any]]) -> int:
        available_refs = {
            fragment["inspiration_ref"]
            for fragment in fragments
            if prior_ref_counts[fragment["inspiration_ref"]] < inspiration_ref_cap
            and prior_operator_counts[fragment_operator(fragment)] < exact_operator_cap
        }
        return min(2, len(available_refs))

    for index, context in enumerate(candidates):
        siret = context["target_siret"]
        name_slice = slice(
            name_offset + index * len(name_relations),
            name_offset + (index + 1) * len(name_relations),
        )
        location_slice = slice(
            location_offset + index * len(location_relations),
            location_offset + (index + 1) * len(location_relations),
        )
        for current_slice in (name_slice, location_slice):
            coefficients = np.zeros(variable_count)
            coefficients[current_slice] = 1
            coefficients[index] = -3
            constraint(coefficients, 0, 0)
        for relation_index, relation in enumerate(name_relations):
            variable = name_offset + index * len(name_relations) + relation_index
            upper[variable] = per_target_relation_capacity(
                all_caps[siret]["name"][relation]
            )
            coefficients = np.zeros(variable_count)
            coefficients[variable] = 1
            coefficients[index] = -2
            constraint(coefficients, -np.inf, 0)
        for relation_index, relation in enumerate(location_relations):
            variable = location_offset + index * len(location_relations) + relation_index
            upper[variable] = per_target_relation_capacity(
                all_caps[siret]["location"][relation]
            )
            coefficients = np.zeros(variable_count)
            coefficients[variable] = 1
            coefficients[index] = -2
            constraint(coefficients, -np.inf, 0)

    coefficients = np.zeros(variable_count)
    coefficients[:candidate_count] = 1
    constraint(coefficients, requested_target_count, requested_target_count)
    for state in ("A", "F"):
        coefficients = np.zeros(variable_count)
        for index, context in enumerate(candidates):
            coefficients[index] = context["target"]["state"] == state
        constraint(coefficients, requested_target_count // 2, requested_target_count // 2)
    for relation_index, relation in enumerate(name_relations):
        coefficients = np.zeros(variable_count)
        for index in range(candidate_count):
            coefficients[name_offset + index * len(name_relations) + relation_index] = 1
        constraint(coefficients, name_quotas[relation], name_quotas[relation])
    for relation_index, relation in enumerate(location_relations):
        coefficients = np.zeros(variable_count)
        for index in range(candidate_count):
            coefficients[
                location_offset + index * len(location_relations) + relation_index
            ] = 1
        constraint(coefficients, location_quotas[relation], location_quotas[relation])

    scale = requested_target_count // QUOTA_UNIT_TARGETS
    minimum_multi_active = minimum_multi_active if minimum_multi_active is not None else 1 * scale
    minimum_multi = minimum_multi if minimum_multi is not None else 4 * scale
    if requested_target_count == 10 and not use_derived_capability_minima:
        capability_minima = {
            "legal": 6, "name_punctuation": 7,
            "address_punctuation": 3, "city_punctuation": 3,
        }
    else:
        # The assignment variables allow at most two uses of one relation on a
        # target.  These are the exact mathematical minima implied by the
        # frozen pilot quotas; higher scaled canary minima made the 30-target
        # programme infeasible without adding any new diversity guarantee.
        capability_minima = {
            "legal": (name_quotas["LEGAL_FORM_REMOVE"] + 1) // 2,
            "name_punctuation": (name_quotas["PUNCTUATION_REMOVED"] + 1) // 2,
            "address_punctuation": (
                location_quotas[("address", "PUNCTUATION_REMOVED")] + 1
            ) // 2,
            "city_punctuation": (
                location_quotas[("city", "PUNCTUATION_REMOVED")] + 1
            ) // 2,
        }
    requirements = (
        (minimum_multi_active, is_multi_active),
        (minimum_multi, is_multi),
        (capability_minima["legal"], lambda value: bool(all_caps[value["target_siret"]]["name"]["LEGAL_FORM_REMOVE"])),
        (capability_minima["name_punctuation"], lambda value: bool(all_caps[value["target_siret"]]["name"]["PUNCTUATION_REMOVED"])),
        (capability_minima["address_punctuation"], lambda value: bool(all_caps[value["target_siret"]]["location"][("address", "PUNCTUATION_REMOVED")])),
        (capability_minima["city_punctuation"], lambda value: bool(all_caps[value["target_siret"]]["location"][("city", "PUNCTUATION_REMOVED")])),
    )
    for minimum, predicate in requirements:
        coefficients = np.zeros(variable_count)
        for index, context in enumerate(candidates):
            coefficients[index] = predicate(context)
        constraint(coefficients, minimum, np.inf)

    objective = np.zeros(variable_count)
    for index, context in enumerate(candidates):
        digest = hashlib.sha256(
            f"{selection_seed}|target|{context['target_siret']}".encode()
        ).hexdigest()
        objective[index] = int(digest[:16], 16) / float(2**64)
    rejected_target_sets: list[list[int]] = []
    for _trial in range(128):
        rows = list(base_rows)
        minima = list(base_lower)
        maxima = list(base_upper)
        for rejected in rejected_target_sets:
            coefficients = np.zeros(variable_count)
            coefficients[rejected] = 1
            rows.append(coefficients)
            minima.append(-np.inf)
            maxima.append(requested_target_count - 1)
        result = milp(
            objective,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(np.asarray(rows), minima, maxima),
            options={"time_limit": 30},
        )
        if not result.success or result.x is None:
            break
        selected_indices = [
            index for index in range(candidate_count) if result.x[index] > 0.5
        ]
        rejected_target_sets.append(selected_indices)
        selected = [candidates[index] for index in selected_indices]
        target_values = {value["target_siret"]: baseline(value) for value in selected}
        target_anchors = {
            value["target_siret"]: distinctive_name_tokens(value, document_frequencies)
            for value in selected
        }
        name_plan = assign_fragment_plan(
            selected, name_quotas, target_values, target_anchors, grouped,
            distinct_per_target=False, exact_operator_cap=exact_operator_cap,
            inspiration_ref_cap=inspiration_ref_cap,
            prior_operator_counts=prior_operator_counts,
            prior_ref_counts=prior_ref_counts,
        )
        if name_plan is None:
            continue
        combined_operator_counts = Counter(prior_operator_counts or {})
        combined_ref_counts = Counter(prior_ref_counts or {})
        update_plan_counts(name_plan, combined_operator_counts, combined_ref_counts)
        location_plan = assign_fragment_plan(
            selected, location_quotas, target_values, target_anchors, grouped,
            distinct_per_target=False, exact_operator_cap=exact_operator_cap,
            inspiration_ref_cap=inspiration_ref_cap,
            prior_operator_counts=combined_operator_counts,
            prior_ref_counts=combined_ref_counts,
        )
        if name_plan is not None and location_plan is not None:
            paired_locations = {
                siret: pair_fragment_plans(name_plan[siret], location_plan[siret])
                for siret in target_values
            }
            if any(value is None for value in paired_locations.values()):
                continue
            location_plan = {
                siret: value for siret, value in paired_locations.items() if value is not None
            }
            selected.sort(key=lambda value: hashlib.sha256(
                f"{selection_seed}|final|{value['target_siret']}".encode()
            ).hexdigest())
            return selected, name_plan, location_plan, all_caps
    raise ValueError(
        f"could not solve a feasible balanced {requested_target_count}-target fragment pilot"
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.target_count % 2:
        raise ValueError("target_count must be even for the frozen A/F balance")
    name_quotas = name_quotas_for_target_count(args.target_count)
    location_quotas = location_quotas_for_target_count(args.target_count)
    exact_operator_cap = 4 if args.target_count == 10 else 6
    inspiration_ref_cap = 3
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    context_path = Path(plan["sources"]["official_context"]["path"])
    fragment_path = Path(plan["sources"]["field_inspiration_bank"]["path"])
    for key, path in (("official_context", context_path), ("field_inspiration_bank", fragment_path)):
        if sha256(path) != plan["sources"][key]["sha256"]:
            raise ValueError(f"source hash mismatch: {key}")
    grouped = group_fragments(rows(fragment_path))
    contexts = rows(context_path)
    document_frequencies = name_token_document_frequencies(contexts)
    excluded_sirets, excluded_sirens = excluded_target_ids(args.exclude_seed_input)
    contexts = [
        value for value in contexts
        if value["target_siret"] not in excluded_sirets
        and value["target_siren"] not in excluded_sirens
    ]
    resolved_quotas: dict[str, dict[Any, int]] | None = None
    if args.target_count == 10:
        targets, name_plan, location_plan, _caps = select_feasible_targets(
            contexts, grouped, document_frequencies, args.selection_seed,
            args.target_count, name_quotas, location_quotas,
            exact_operator_cap, inspiration_ref_cap,
        )
    else:
        targets, name_plan, location_plan, resolved_quotas = select_integrated_fragment_targets(
            contexts, grouped, document_frequencies, args.selection_seed,
            args.target_count, name_quotas, location_quotas,
            exact_operator_cap, inspiration_ref_cap,
            minimum_multi_active=4, minimum_multi=12,
        )
    target_values = {value["target_siret"]: baseline(value) for value in targets}
    target_anchors = {
        value["target_siret"]: distinctive_name_tokens(value, document_frequencies)
        for value in targets
    }
    output = []
    used_refs: Counter[str] = Counter()
    for target in targets:
        siret = target["target_siret"]
        name_fragments = name_plan[siret]
        location_fragments = location_plan[siret]
        contracts = []
        for slot, variant_id in enumerate(("v1", "v2", "v3")):
            name_fragment = name_fragments[slot]
            location_fragment = location_fragments[slot]
            location_field = location_fragment["field"]
            location_relation = location_fragment["relation"]
            field_fragments = {
                "name": name_fragment,
                location_field: location_fragment,
            }
            refs = {value["inspiration_ref"] for value in field_fragments.values()}
            if any(used_refs[ref] >= inspiration_ref_cap for ref in refs):
                raise RuntimeError("fragment reference exceeds pilot reuse cap")
            used_refs.update(refs)
            anchors = target_anchors[siret]
            protected_anchor: list[str] = []
            if name_fragment["relation"] == "TOKEN_SUBSET":
                retained = name_fragment["operation_parameters"]["retained_positions"]
                retained_words = {
                    loop.normalized_words(target_values[siret]["name"])[index]
                    for index in retained
                }
                protected_anchor = [
                    value for value in anchors if value in retained_words
                ][:1]
                if not protected_anchor:
                    raise RuntimeError("TOKEN_SUBSET lacks a retained distinctive anchor")
            contracts.append({
                "variant_id": variant_id,
                "requested_family": loop.COMPOSITE_FAMILY,
                "target_fields": ["name", location_field],
                "field_relations": {
                    "name": name_fragment["relation"], location_field: location_relation,
                },
                "field_inspirations": field_fragments,
                "protected_target_tokens": {"name": protected_anchor},
                "operator_guidance": {
                    field: {
                        "source_tokens_zero_indexed": list(enumerate(
                            loop.normalized_words(target_values[siret][field])
                        )),
                        "expected_retained_tokens": [
                            loop.normalized_words(target_values[siret][field])[index]
                            for index in fragment.get("operation_parameters", {}).get(
                                "retained_positions", []
                            )
                        ],
                    }
                    for field, fragment in field_fragments.items()
                },
                "rules": {
                    "copy_non_target_fields_byte_for_byte": True,
                    "no_new_lexical_or_numeric_tokens": True,
                    "added_marks_forbidden": True,
                    "preserve_house_number": True,
                },
            })
        value = target_values[siret]
        official = target["target"]
        seed_card = {
            "generation_mode": "OBSERVED_COMPOSITE_ANALOGY_V2",
            "name_options": [item["value"] for item in official["names"]],
            "enseigne_options": [
                item["value"] for item in official["names"] if item.get("kind") == "ENSEIGNE"
            ],
            "address": value["address"], "postcode": value["postcode"],
            "city": value["city"], "insee": value["insee"],
            "street_number": str(official["address"]["number"]),
            "street_type": str(official["address"]["street_type"]),
            "composite_contracts": contracts,
            "official_context": target["llm_view"],
            "qualification": target["qualification"],
            "internal_context": target["internal_context"],
            "context_sha256": target["context_sha256"],
        }
        output.append({
            "seed_id": f"COMPOSITE_V3:{siret}",
            "target_siret": siret, "target_siren": target["target_siren"],
            "source_kind": "SIRENE_ONLY_TRAIN", "oof_fold": -1,
            "legacy_split": "train_synthetic", "seed_card": seed_card,
            "observed_train_profile": {
                "schema_version": "sireto-synthetic-field-evidence-1",
                "rows": sum(len(value) for value in grouped.values()),
                "source_sha256": sha256(fragment_path), "source_folds": [2, 3, 4],
                "supported_families": [loop.COMPOSITE_FAMILY],
            },
            "risk_flags": [
                "COMPOSITE_LLM_WRITTEN", "FIELD_FRAGMENT_RECOMBINATION",
                "MULTI_SITE_SIREN" if is_multi(target) else "SINGLE_SITE_SIREN",
                "TARGET_CLOSED" if official["state"] == "F" else "TARGET_ACTIVE",
            ],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    loop.write_jsonl_atomic(args.output, output)
    relation_counts = Counter(
        f"{field}:{relation}"
        for row in output for contract in row["seed_card"]["composite_contracts"]
        for field, relation in contract["field_relations"].items()
    )
    operation_counts = Counter(
        loop.digest_json({
            "field": field, "relation": fragment["relation"],
            "parameters": fragment["operation_parameters"],
        })
        for row in output for contract in row["seed_card"]["composite_contracts"]
        for field, fragment in contract["field_inspirations"].items()
    )
    if max(operation_counts.values(), default=0) > exact_operator_cap:
        raise RuntimeError("exact operator exceeds pilot-wide reuse cap")
    manifest = {
        "schema_version": "sireto-synthetic-fragment-pilot-manifest-1",
        "rows": len(output), "planned_pairs": len(output) * 3,
        "requested_target_count": args.target_count,
        "state_counts": dict(Counter(row["seed_card"]["official_context"]["target"]["state"] for row in output)),
        "multi_site_targets": sum("MULTI_SITE_SIREN" in row["risk_flags"] for row in output),
        "multi_active_targets": sum(is_multi_active(target) for target in targets),
        "distinct_inspiration_refs": len(used_refs),
        "maximum_inspiration_ref_uses": max(used_refs.values(), default=0),
        "relation_counts": dict(sorted(relation_counts.items())),
        "distinct_exact_operators": len(operation_counts),
        "top_exact_operator_count": max(operation_counts.values(), default=0),
        "exact_operator_cap": exact_operator_cap,
        "inspiration_ref_cap": inspiration_ref_cap,
        "quota_mode": "bounded_pre_generation_milp" if resolved_quotas else "exact",
        "resolved_name_quotas": (
            resolved_quotas["name"] if resolved_quotas else name_quotas
        ),
        "resolved_location_quotas": {
            f"{field}:{relation}": quota
            for (field, relation), quota in (
                resolved_quotas["location"] if resolved_quotas else location_quotas
            ).items()
        },
        "forbidden_added_mark_contracts": sum(
            "ADDED" in relation for relation in relation_counts
        ),
        "text_generation": "none_selector_only", "selection_seed": args.selection_seed,
        "excluded_prior_sirets": len(excluded_sirets),
        "excluded_prior_sirens": len(excluded_sirens),
        "exclusion_source_hashes": {
            str(path): sha256(path) for path in args.exclude_seed_input
        },
        "source_hashes": {
            "plan": sha256(args.plan), "official_context": sha256(context_path),
            "field_inspiration_bank": sha256(fragment_path),
        },
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=ROOT / "config/synthetic_gt_composite_v3_plan.json")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--target-count", type=int, default=10)
    result.add_argument(
        "--exclude-seed-input", type=Path, action="append", default=[],
        help="Prior seed JSONL whose target SIRET and SIREN must be excluded",
    )
    result.add_argument("--selection-seed", default="SIRETO-COMPOSITE-FRAGMENT-PILOT-V3")
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
