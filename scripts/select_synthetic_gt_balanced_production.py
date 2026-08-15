#!/usr/bin/env python3
"""Select a counted, balanced synthetic-GT production batch.

The selector never writes CRM text.  It chooses official SIRENE targets and
train-observed field operators, solves the requested easy/medium/hard mix, and
assigns train-OOF failure-directed strata.  Luna remains the sole CRM writer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
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
from scripts import select_synthetic_gt_fragment_pilot as fragments  # noqa: E402


SAFE_NAME_RELATIONS = ("LEGAL_FORM_REMOVE", "TOKEN_ORDER", "PUNCTUATION_REMOVED")
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


def sha256(path: Path) -> str:
    return fragments.sha256(path)


def rows(path: Path) -> list[dict[str, Any]]:
    return fragments.rows(path)


def exact_counts(total: int, shares: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in shares.items()}
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(shares, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def pair_signature(pair: tuple[str, tuple[str, str]]) -> str:
    name_relation, (field, location_relation) = pair
    return f"name:{name_relation}+{field}:{location_relation}"


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
    name_weak = name_relation in {"TOKEN_ORDER", "PUNCTUATION_REMOVED"}
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
    if name_relation == "TOKEN_ORDER":
        score += 1
    if location_relation == "ADDRESS_TOKEN_SUBSET":
        score += 1
    if score == 0:
        return "EASY"
    if score >= 3:
        return "HARD"
    return "MEDIUM"


def safe_capabilities(
    context: dict[str, Any],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    document_frequencies: Counter[str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, str], list[dict[str, Any]]],
]:
    values = fragments.baseline(context)
    anchors = fragments.distinctive_name_tokens(context, document_frequencies)
    names = {
        relation: [
            value for value in grouped.get(("name", relation), [])
            if fragments.fragment_supports("name", values["name"], value, anchors)
        ]
        for relation in SAFE_NAME_RELATIONS
    }
    locations = {
        key: [
            value for value in grouped.get(key, [])
            if fragments.fragment_supports(key[0], values[key[0]], value, anchors)
        ]
        for key in SAFE_LOCATION_RELATIONS
    }
    return names, locations


def candidate_bundles(
    context: dict[str, Any],
    names: dict[str, list[dict[str, Any]]],
    locations: dict[tuple[str, str], list[dict[str, Any]]],
    selection_seed: str,
    limit: int = 24,
) -> list[tuple[tuple[str, tuple[str, str]], ...]]:
    pairs = [
        (name_relation, location_key)
        for name_relation, name_values in names.items() if name_values
        for location_key, location_values in locations.items() if location_values
    ]
    if len(pairs) < 3:
        return []
    values = list(itertools.combinations(pairs, 3))
    # Keep several difficulty profiles before deterministic thinning.
    by_profile: dict[tuple[int, int, int], list[Any]] = defaultdict(list)
    for bundle in values:
        name_counts = Counter(pair[0] for pair in bundle)
        location_counts = Counter(pair[1] for pair in bundle)
        if any(
            count > len({value["inspiration_ref"] for value in names[relation]})
            for relation, count in name_counts.items()
        ) or any(
            count > len({value["inspiration_ref"] for value in locations[key]})
            for key, count in location_counts.items()
        ):
            continue
        counts = Counter(difficulty(context, pair) for pair in bundle)
        profile = tuple(counts[value] for value in DIFFICULTIES)
        by_profile[profile].append(bundle)
    selected = []
    for profile in sorted(by_profile):
        candidates = sorted(
            by_profile[profile],
            key=lambda bundle: hashlib.sha256(
                f"{selection_seed}|bundle|{context['target_siret']}|"
                f"{'|'.join(pair_signature(value) for value in bundle)}".encode()
            ).hexdigest(),
        )
        selected.extend(candidates[:4])
    return sorted(
        selected,
        key=lambda bundle: hashlib.sha256(
            f"{selection_seed}|thin|{context['target_siret']}|"
            f"{'|'.join(pair_signature(value) for value in bundle)}".encode()
        ).hexdigest(),
    )[:limit]


def choose_targets_and_bundles(
    contexts: list[dict[str, Any]],
    capabilities: dict[str, tuple[dict[str, Any], dict[tuple[str, str], Any]]],
    target_count: int,
    difficulty_counts: dict[str, int],
    relation_pair_cap: int,
    relation_capacities: dict[tuple[str, str], int],
    selection_seed: str,
) -> list[tuple[dict[str, Any], tuple[tuple[str, tuple[str, str]], ...]]]:
    options: list[tuple[int, tuple[tuple[str, tuple[str, str]], ...]]] = []
    context_by_index: list[dict[str, Any]] = []
    # A bounded deterministic pool keeps the MILP quick while leaving ample choice.
    eligible = sorted(
        contexts,
        key=lambda value: hashlib.sha256(
            f"{selection_seed}|pool|{value['target_siret']}".encode()
        ).hexdigest(),
    )[:3000]
    for context in eligible:
        siret = context["target_siret"]
        names, locations = capabilities[siret]
        bundles = candidate_bundles(context, names, locations, selection_seed)
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

    def add(coefficients: dict[int, float], minimum: float, maximum: float) -> None:
        constraint_rows.append(coefficients)
        minima.append(minimum)
        maxima.append(maximum)

    add({index: 1 for index in range(variable_count)}, target_count, target_count)
    option_by_context: dict[int, list[int]] = defaultdict(list)
    for option_index, (context_index, _bundle) in enumerate(options):
        option_by_context[context_index].append(option_index)
    for indexes in option_by_context.values():
        add({index: 1 for index in indexes}, 0, 1)
    for state in ("A", "F"):
        add({
            index: 1 for index, (context_index, _bundle) in enumerate(options)
            if context_by_index[context_index]["target"]["state"] == state
        }, target_count // 2, target_count // 2)
    for level in DIFFICULTIES:
        add({
            index: sum(
                difficulty(context_by_index[context_index], pair) == level
                for pair in bundle
            )
            for index, (context_index, bundle) in enumerate(options)
        }, difficulty_counts[level], difficulty_counts[level])
    all_pairs = sorted({pair for _context_index, bundle in options for pair in bundle})
    for pair in all_pairs:
        add({
            index: int(pair in bundle)
            for index, (_context_index, bundle) in enumerate(options)
        }, 0, relation_pair_cap)
    for (field, relation), capacity in relation_capacities.items():
        add({
            index: sum(
                ("name" if pair[0] == relation and field == "name" else None) == field
                or (pair[1] == (field, relation))
                for pair in bundle
            )
            for index, (_context_index, bundle) in enumerate(options)
        }, 0, capacity)

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
        shape=(len(constraint_rows), variable_count),
    ).tocsr()
    objective = np.asarray([
        int(hashlib.sha256(
            f"{selection_seed}|option|{context_by_index[context_index]['target_siret']}|"
            f"{'|'.join(pair_signature(value) for value in bundle)}".encode()
        ).hexdigest()[:16], 16) / float(2**64)
        for context_index, bundle in options
    ])
    solution = milp(
        objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix, minima, maxima),
        options={"time_limit": 180},
    )
    if not solution.success or solution.x is None:
        raise ValueError(f"balanced production MILP is infeasible: {solution.message}")
    result = [
        (context_by_index[context_index], bundle)
        for index, (context_index, bundle) in enumerate(options)
        if solution.x[index] > 0.5
    ]
    if len(result) != target_count:
        raise RuntimeError("MILP returned an incomplete target selection")
    return sorted(result, key=lambda value: value[0]["target_siret"])


def fragment_operator(value: dict[str, Any]) -> str:
    return fragments.fragment_operator(value)


def assign_fragments(
    selected: list[tuple[dict[str, Any], tuple[tuple[str, tuple[str, str]], ...]]],
    capabilities: dict[str, tuple[dict[str, Any], dict[tuple[str, str], Any]]],
    ref_cap: int,
    operator_cap: int,
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
                    graph.add_edge(target_ref, ref_node, capacity=1, weight=0)
                    graph.add_edge(ref_node, operator_node, capacity=ref_cap, weight=0)
                    graph.add_edge(operator_node, sink, capacity=operator_cap, weight=0)
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
                if used >= ref_cap:
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
    if args.target_count <= 0 or args.target_count % 2:
        raise ValueError("target-count must be a positive even integer")
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

    all_contexts = rows(source_paths["official_context"])
    document_frequencies = fragments.name_token_document_frequencies(all_contexts)
    excluded_sirets, excluded_sirens = fragments.excluded_target_ids(args.exclude_seed_input)
    contexts = [
        value for value in all_contexts
        if value["target_siret"] not in excluded_sirets
        and value["target_siren"] not in excluded_sirens
        and fragments.eligible(value)
    ]
    grouped = fragments.group_fragments(rows(source_paths["field_inspiration_bank"]))
    capabilities = {}
    feasible_contexts = []
    for context in contexts:
        names, locations = safe_capabilities(context, grouped, document_frequencies)
        if sum(bool(value) for value in names.values()) and sum(
            bool(value) for value in locations.values()
        ):
            pairs = sum(bool(value) for value in names.values()) * sum(
                bool(value) for value in locations.values()
            )
            if pairs >= 3:
                capabilities[context["target_siret"]] = (names, locations)
                feasible_contexts.append(context)

    variant_count = args.target_count * 3
    difficulty_counts = exact_counts(
        variant_count, plan["corpus_balance"]["difficulty"]
    )
    stratum_counts = exact_counts(
        variant_count, plan["corpus_balance"]["augmentation_strata"]
    )
    relation_pair_cap = max(
        1, int(math.ceil(variant_count * plan["global_caps"]["relation_pair_share"]))
    )
    operator_cap = max(
        1, int(math.ceil(variant_count * 2 * plan["global_caps"]["exact_operator_share"]))
    )
    first_batch = plan["production"].get("first_batch", {})
    if args.batch_id == first_batch.get("batch_id"):
        operator_cap = max(
            operator_cap, int(first_batch.get("bootstrap_exact_operator_cap", 0))
        )
    ref_cap = int(plan["global_caps"]["inspiration_ref_uses"])
    relation_capacities: dict[tuple[str, str], int] = {}
    for field, relation in (
        [("name", value) for value in SAFE_NAME_RELATIONS]
        + list(SAFE_LOCATION_RELATIONS)
    ):
        available = grouped.get((field, relation), [])
        relation_capacities[(field, relation)] = min(
            len({value["inspiration_ref"] for value in available}) * ref_cap,
            len({fragment_operator(value) for value in available}) * operator_cap,
        )
    if args.batch_id == first_batch.get("batch_id"):
        for key, capacity in first_batch.get("bootstrap_relation_caps", {}).items():
            field, relation = key.split(":", 1)
            relation_capacities[(field, relation)] = min(
                relation_capacities.get((field, relation), int(capacity)), int(capacity)
            )
    selected = choose_targets_and_bundles(
        feasible_contexts, capabilities, args.target_count, difficulty_counts,
        relation_pair_cap, relation_capacities, args.selection_seed,
    )
    assigned_fragments, ref_counts, operator_counts = assign_fragments(
        selected, capabilities, ref_cap,
        operator_cap, args.selection_seed,
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
                "protected_target_tokens": {"name": []},
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
                    "no_new_lexical_or_numeric_tokens": True,
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
        "relation_pair_counts": dict(Counter(
            "+".join(f"{field}:{relation}" for field, relation in sorted(
                contract["field_relations"].items()
            ))
            for row in output for contract in row["seed_card"]["composite_contracts"]
        )),
        "distinct_inspiration_refs": len(ref_counts),
        "maximum_inspiration_ref_uses": max(ref_counts.values(), default=0),
        "distinct_exact_operators": len(operator_counts),
        "maximum_exact_operator_uses": max(operator_counts.values(), default=0),
        "caps": {
            "inspiration_ref": int(plan["global_caps"]["inspiration_ref_uses"]),
            "exact_operator": operator_cap,
            "relation_pair": relation_pair_cap,
        },
        "excluded_prior_sirets": len(excluded_sirets),
        "excluded_prior_sirens": len(excluded_sirens),
        "selection_seed": args.selection_seed,
        "source_hashes": {
            "plan": sha256(args.plan),
            **{key: sha256(path) for key, path in source_paths.items()},
            **{f"excluded:{path}": sha256(path) for path in args.exclude_seed_input},
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
    result.add_argument("--selection-seed", default="SIRETO-BALANCED-P000-V1")
    result.add_argument("--exclude-seed-input", type=Path, action="append", default=[])
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
