#!/usr/bin/env python3
"""Select a 30-target field-fragment composite pilot without writing CRM text."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402
from scripts import select_synthetic_gt_composite_pilot as legacy  # noqa: E402


NAME_QUOTAS = {
    "TOKEN_SUBSET": 8,
    "TOKEN_ORDER": 2,
    "LEGAL_FORM_REMOVE": 6,
    "PUNCTUATION_REMOVED": 7,
    "JOIN_SPLIT": 7,
}
LOCATION_QUOTAS = {
    ("address", "ADDRESS_ABBREVIATE"): 8,
    ("address", "ADDRESS_TOKEN_SUBSET"): 8,
    ("address", "PUNCTUATION_REMOVED"): 7,
    ("city", "PUNCTUATION_REMOVED"): 7,
}
STOP_ANCHORS = {
    "ASS", "ASSOCIATION", "CABINET", "CENTRE", "DE", "DES", "DU", "ET", "LA", "LE",
    "LES", "MAISON", "SAS", "SARL", "SCI", "SOCIETE",
}


def sha256(path: Path) -> str:
    return legacy.sha256(path)


def rows(path: Path) -> list[dict[str, Any]]:
    return legacy.rows(path)


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


def distinctive_name_tokens(context: dict[str, Any]) -> list[str]:
    target_words = loop.normalized_words(legacy.primary_name(context))
    competitors = {
        word
        for candidate in context.get("internal_context", [])
        for name in candidate_names(candidate)
        for word in loop.normalized_words(name)
    }
    return list(dict.fromkeys(
        word for word in target_words
        if word.upper() not in STOP_ANCHORS and len(word) >= 3 and word not in competitors
    ))


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
            return bool(set(anchors) & {words[index] for index in retained})
        if relation == "LEGAL_FORM_REMOVE":
            removed = [words[index] for index in range(len(words)) if index not in retained]
            return bool(removed) and all(value.upper() in loop.LEGAL_FORM_TOKENS for value in removed)
        if relation == "ADDRESS_TOKEN_SUBSET":
            retained_words = [words[index] for index in retained]
            return (
                [value for value in retained_words if value.isdigit()]
                == [value for value in words if value.isdigit()]
            )
        return True
    if relation in {"TOKEN_ORDER", "ADDRESS_TYPE_ORDER"}:
        permutation = parameters.get("permutation")
        return isinstance(permutation, list) and sorted(permutation) == list(range(len(words)))
    if relation == "ADDRESS_ABBREVIATE":
        pairs = parameters.get("pairs", [])
        return (
            len(pairs) == 1
            and str(pairs[0].get("source", "")).upper()
            == loop.normalized_surface(source_value.split()[1] if len(source_value.split()) > 1 else "").upper()
        )
    if relation == "PUNCTUATION_REMOVED":
        edits = parameters.get("edits", [])
        available = punctuation_boundaries(source_value)
        return bool(edits) and all(
            (int(value.get("after_token_index", -99)), str(value.get("mark", ""))) in available
            for value in edits
        )
    if relation == "JOIN_SPLIT":
        return parameters.get("target_token_count", len(words)) < len(words)
    return False


def group_fragments(values: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for value in values:
        result[(str(value.get("field")), str(value.get("relation")))].append(value)
    for key in result:
        result[key].sort(key=lambda value: value["inspiration_ref"])
    return result


def target_capabilities(
    context: dict[str, Any], grouped: dict[tuple[str, str], list[dict[str, Any]]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]]:
    values = baseline(context)
    anchors = distinctive_name_tokens(context)
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
                graph.add_edge(relation_node, node, capacity=1 if distinct_per_target else 3)
                graph.add_edge(node, ("TARGET", siret), capacity=1 if distinct_per_target else 3)
    for target in targets:
        graph.add_edge(("TARGET", target["target_siret"]), sink_node, capacity=3)
    flow_value, flow = nx.maximum_flow(graph, source_node, sink_node)
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
    flow_value, flow = nx.maximum_flow(graph, source_node, sink_node)
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


def select_feasible_targets(
    contexts: list[dict[str, Any]],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    selection_seed: str,
) -> tuple[list[dict[str, Any]], dict[str, list[Any]], dict[str, list[Any]], dict[str, Any]]:
    candidates = []
    all_caps: dict[str, Any] = {}
    for context in contexts:
        if not eligible(context):
            continue
        name_caps, location_caps = target_capabilities(context, grouped)
        if (
            sum(bool(value) for value in name_caps.values()) < 3
            or not any(location_caps.values())
        ):
            continue
        siret = context["target_siret"]
        all_caps[siret] = {"name": name_caps, "location": location_caps}
        order = hashlib.sha256(f"{selection_seed}|{siret}".encode()).hexdigest()
        candidates.append((order, context))
    candidates.sort(key=lambda value: value[0])
    for trial in range(512):
        selected: list[dict[str, Any]] = []
        state_counts = {"A": 0, "F": 0}

        def feature_counts() -> dict[str, int]:
            return {
                "multi_active": sum(is_multi_active(value) for value in selected),
                "multi": sum(is_multi(value) for value in selected),
                "legal": sum(bool(all_caps[value["target_siret"]]["name"]["LEGAL_FORM_REMOVE"]) for value in selected),
                "name_punctuation": sum(bool(all_caps[value["target_siret"]]["name"]["PUNCTUATION_REMOVED"]) for value in selected),
                "address_punctuation": sum(bool(all_caps[value["target_siret"]]["location"][("address", "PUNCTUATION_REMOVED")]) for value in selected),
                "city": sum(bool(all_caps[value["target_siret"]]["location"][("city", "PUNCTUATION_REMOVED")]) for value in selected),
            }

        requirements = {
            "multi_active": 1, "multi": 4, "legal": 6,
            "name_punctuation": 7, "address_punctuation": 3, "city": 3,
        }
        weights = {
            "multi_active": 5, "multi": 1, "legal": 4,
            "name_punctuation": 4, "address_punctuation": 5, "city": 1,
        }
        while len(selected) < 10:
            current = feature_counts()
            choices = []
            for _order, context in candidates:
                state = context["target"]["state"]
                if context in selected or state_counts[state] >= 5:
                    continue
                caps = all_caps[context["target_siret"]]
                features = {
                    "multi_active": is_multi_active(context),
                    "multi": is_multi(context),
                    "legal": bool(caps["name"]["LEGAL_FORM_REMOVE"]),
                    "name_punctuation": bool(caps["name"]["PUNCTUATION_REMOVED"]),
                    "address_punctuation": bool(caps["location"][("address", "PUNCTUATION_REMOVED")]),
                    "city": bool(caps["location"][("city", "PUNCTUATION_REMOVED")]),
                }
                score = sum(
                    weights[key] for key, enabled in features.items()
                    if enabled and current[key] < requirements[key]
                )
                tie = hashlib.sha256(
                    f"{selection_seed}|{trial}|{context['target_siret']}".encode()
                ).hexdigest()
                choices.append((-score, tie, context))
            if not choices:
                break
            choices.sort(key=lambda value: value[:2])
            chosen = choices[0][2]
            selected.append(chosen)
            state_counts[chosen["target"]["state"]] += 1
        if len(selected) != 10 or any(
            feature_counts()[key] < minimum for key, minimum in requirements.items()
        ):
            continue
        name_assignment = relation_assignment(
            selected, NAME_QUOTAS,
            {siret: value["name"] for siret, value in all_caps.items()},
            distinct_per_target=True,
        )
        location_assignment = relation_assignment(
            selected, LOCATION_QUOTAS,
            {siret: value["location"] for siret, value in all_caps.items()},
            distinct_per_target=False,
        )
        if name_assignment is not None and location_assignment is not None:
            selected.sort(key=lambda value: hashlib.sha256(
                f"{selection_seed}|final|{value['target_siret']}".encode()
            ).hexdigest())
            target_values = {value["target_siret"]: baseline(value) for value in selected}
            target_anchors = {
                value["target_siret"]: distinctive_name_tokens(value) for value in selected
            }
            name_requests: list[tuple[str, int, str, str]] = []
            location_requests: list[tuple[str, int, str, str]] = []
            for value in selected:
                siret = value["target_siret"]
                for slot, relation in enumerate(sorted(name_assignment[siret])):
                    name_requests.append((siret, slot, "name", relation))
                for slot, (field, relation) in enumerate(sorted(location_assignment[siret], key=str)):
                    location_requests.append((siret, slot, field, relation))
            if (
                assign_unique_fragments(
                    name_requests, target_values, target_anchors, grouped
                ) is None
                or assign_unique_fragments(
                    location_requests, target_values, target_anchors, grouped
                ) is None
            ):
                continue
            return selected, name_assignment, location_assignment, all_caps
    raise ValueError("could not find a feasible balanced 30-target fragment pilot")


def build(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    context_path = Path(plan["sources"]["official_context"]["path"])
    fragment_path = Path(plan["sources"]["field_inspiration_bank"]["path"])
    for key, path in (("official_context", context_path), ("field_inspiration_bank", fragment_path)):
        if sha256(path) != plan["sources"][key]["sha256"]:
            raise ValueError(f"source hash mismatch: {key}")
    grouped = group_fragments(rows(fragment_path))
    targets, name_assignment, location_assignment, _caps = select_feasible_targets(
        rows(context_path), grouped, args.selection_seed
    )
    target_values = {value["target_siret"]: baseline(value) for value in targets}
    target_anchors = {value["target_siret"]: distinctive_name_tokens(value) for value in targets}
    name_requests: list[tuple[str, int, str, str]] = []
    location_requests: list[tuple[str, int, str, str]] = []
    for target in targets:
        siret = target["target_siret"]
        name_relations = sorted(name_assignment[siret])
        location_relations = sorted(location_assignment[siret], key=str)
        for slot, relation in enumerate(name_relations):
            name_requests.append((siret, slot, "name", relation))
        for slot, (field, relation) in enumerate(location_relations):
            location_requests.append((siret, slot, field, relation))
    name_fragments = assign_unique_fragments(
        name_requests, target_values, target_anchors, grouped
    )
    location_fragments = assign_unique_fragments(
        location_requests, target_values, target_anchors, grouped
    )
    if name_fragments is None or location_fragments is None:
        raise ValueError("relation-feasible targets lack a globally unique fragment matching")

    output = []
    used_refs: Counter[str] = Counter()
    for target in targets:
        siret = target["target_siret"]
        name_relations = sorted(name_assignment[siret])
        location_relations = sorted(location_assignment[siret], key=str)
        contracts = []
        for slot, variant_id in enumerate(("v1", "v2", "v3")):
            location_field, location_relation = location_relations[slot]
            field_fragments = {
                "name": name_fragments[(siret, slot, "name")],
                location_field: location_fragments[(siret, slot, location_field)],
            }
            refs = {value["inspiration_ref"] for value in field_fragments.values()}
            if any(used_refs[ref] >= 3 for ref in refs):
                raise RuntimeError("fragment reference exceeds pilot reuse cap")
            used_refs.update(refs)
            anchors = target_anchors[siret]
            contracts.append({
                "variant_id": variant_id,
                "requested_family": loop.COMPOSITE_FAMILY,
                "target_fields": ["name", location_field],
                "field_relations": {
                    "name": name_relations[slot], location_field: location_relation,
                },
                "field_inspirations": field_fragments,
                "protected_target_tokens": {"name": anchors[:3]},
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
    manifest = {
        "schema_version": "sireto-synthetic-fragment-pilot-manifest-1",
        "rows": len(output), "planned_pairs": len(output) * 3,
        "state_counts": dict(Counter(row["seed_card"]["official_context"]["target"]["state"] for row in output)),
        "multi_site_targets": sum("MULTI_SITE_SIREN" in row["risk_flags"] for row in output),
        "multi_active_targets": sum(is_multi_active(target) for target in targets),
        "distinct_inspiration_refs": len(used_refs),
        "maximum_inspiration_ref_uses": max(used_refs.values(), default=0),
        "relation_counts": dict(sorted(relation_counts.items())),
        "distinct_exact_operators": len(operation_counts),
        "top_exact_operator_count": max(operation_counts.values(), default=0),
        "forbidden_added_mark_contracts": sum(
            "ADDED" in relation for relation in relation_counts
        ),
        "text_generation": "none_selector_only", "selection_seed": args.selection_seed,
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
    result.add_argument("--selection-seed", default="SIRETO-COMPOSITE-FRAGMENT-PILOT-V3")
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
