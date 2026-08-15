#!/usr/bin/env python3
"""Select a preregistered composite pilot without generating CRM text.

The selector only joins immutable official contexts with existing opaque
official-to-CRM examples from train folds 2/3/4.  Luna remains the sole writer
of every new CRM field in the downstream agentic loop.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402


CONTRACT_MASKS = {
    "v1": ["name", "address"],
    "v2": ["name", "city"],
    "v3": ["name", "address", "city"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [value for _raw, value in loop.iter_jsonl_raw(path)]


def primary_name(context: dict[str, Any]) -> str:
    names = context["target"]["names"]
    official = [item["value"] for item in names if item.get("kind") == "OFFICIAL_NAME"]
    return str((official or [item["value"] for item in names])[0]).strip()


def address_line(context: dict[str, Any]) -> str:
    value = context["target"]["address"]
    return " ".join(
        str(value.get(key) or "").strip()
        for key in ("number", "repetition_index", "street_type", "street")
        if str(value.get(key) or "").strip()
    )


def count_relations(context: dict[str, Any], tag: str) -> int:
    return sum(tag in row.get("relation_tags", []) for row in context["internal_context"])


def eligible(context: dict[str, Any]) -> bool:
    target = context["target"]
    address = target["address"]
    name = primary_name(context)
    return bool(
        context["qualification"].get("pre_generation_exact_eligible")
        and len(loop.normalized_words(name)) >= 3
        and address.get("street_type")
        and address.get("number")
        and address.get("street")
        and address.get("postcode")
        and address.get("insee")
        and address.get("city")
        and (loop.has_diacritic(address["city"]) or loop.punctuation_marks(address["city"]))
    )


def inspiration_groups(values: Iterable[dict[str, Any]]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    result: dict[tuple[str, ...], list[dict[str, Any]]] = {
        tuple(mask): [] for mask in CONTRACT_MASKS.values()
    }
    for value in values:
        if value.get("source_fold") not in {2, 3, 4}:
            continue
        signature = value.get("structural_signature", {})
        mask = tuple(signature.get("changed_fields", []))
        if mask not in result or signature.get("missing_fields"):
            continue
        if not value.get("analogy_safety", {}).get("lexical_tokens_subset_of_official"):
            continue
        if not value.get("analogy_safety", {}).get("numeric_tokens_subset_of_official"):
            continue
        result[mask].append(value)
    return result


def stable_order(values: Iterable[dict[str, Any]], seed: str, identity) -> list[dict[str, Any]]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{seed}|{identity(value)}".encode("utf-8")
        ).hexdigest(),
    )


def select_targets(values: list[dict[str, Any]], plan: dict[str, Any], selection_seed: str) -> list[dict[str, Any]]:
    strata = plan["pilot_strata"]
    wanted = {"A": int(strata["active_targets"]), "F": int(strata["closed_targets"])}
    candidates = stable_order(
        (value for value in values if eligible(value)), selection_seed,
        lambda value: value["target_siret"],
    )
    selected: list[dict[str, Any]] = []
    for state in ("A", "F"):
        options = [value for value in candidates if value["target"]["state"] == state]
        selected.extend(options[:wanted[state]])
        if len(options) < wanted[state]:
            raise ValueError(f"not enough eligible targets in state {state}: {len(options)}")

    minimum_multi = int(strata["minimum_multi_site_siren"])
    minimum_active = int(strata["minimum_multi_active_siren"])

    def is_multi(value: dict[str, Any]) -> bool:
        return count_relations(value, "SAME_SIREN") > 0

    def is_multi_active(value: dict[str, Any]) -> bool:
        return sum(
            "SAME_SIREN" in row.get("relation_tags", []) and row.get("state") == "A"
            for row in value["internal_context"]
        ) > 0

    def satisfy(predicate, minimum: int) -> None:
        while sum(predicate(value) for value in selected) < minimum:
            replacement = next(
                (value for value in candidates if value not in selected and predicate(value)), None
            )
            if replacement is None:
                raise ValueError(f"cannot satisfy pilot stratum minimum {minimum}")
            replace_index = next(
                (
                    index for index in range(len(selected) - 1, -1, -1)
                    if selected[index]["target"]["state"] == replacement["target"]["state"]
                    and not predicate(selected[index])
                ),
                None,
            )
            if replace_index is None:
                raise ValueError(f"cannot preserve state balance for stratum minimum {minimum}")
            selected[replace_index] = replacement

    satisfy(is_multi_active, minimum_active)
    satisfy(is_multi, minimum_multi)
    return stable_order(selected, selection_seed + "|final", lambda value: value["target_siret"])


def build(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    official_path = Path(plan["sources"]["official_context"]["path"])
    inspiration_path = Path(plan["sources"]["train_inspiration_bank"]["path"])
    for key, path in (("official_context", official_path), ("train_inspiration_bank", inspiration_path)):
        expected = plan["sources"][key]["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"source hash mismatch for {key}: {actual}")

    targets = select_targets(rows(official_path), plan, args.selection_seed)
    groups = inspiration_groups(rows(inspiration_path))
    ordered_groups = {
        mask: stable_order(values, args.selection_seed + "|" + "+".join(mask), lambda value: value["inspiration_ref"])
        for mask, values in groups.items()
    }
    needed = len(targets)
    for mask, values in ordered_groups.items():
        if len(values) < needed:
            raise ValueError(f"not enough inspirations for {mask}: {len(values)} < {needed}")

    output: list[dict[str, Any]] = []
    used_refs: set[str] = set()
    for target_index, context in enumerate(targets):
        official = context["target"]
        baseline = {
            "name": primary_name(context),
            "address": address_line(context),
            "postcode": str(official["address"]["postcode"]),
            "city": str(official["address"]["city"]),
            "insee": str(official["address"]["insee"]),
        }
        contracts: list[dict[str, Any]] = []
        for variant_id, mask_list in CONTRACT_MASKS.items():
            mask = tuple(mask_list)
            inspiration = ordered_groups[mask][target_index]
            ref = inspiration["inspiration_ref"]
            if ref in used_refs:
                raise RuntimeError(f"inspiration reused in pilot: {ref}")
            used_refs.add(ref)
            contracts.append({
                "variant_id": variant_id,
                "requested_family": "OBSERVED_COMPOSITE_ANALOGY",
                "target_fields": mask_list,
                "inspiration_ref": ref,
                "inspiration": inspiration,
                "rules": {
                    "copy_non_target_fields_byte_for_byte": True,
                    "no_new_lexical_or_numeric_tokens": True,
                    "no_added_punctuation_or_diacritic": True,
                    "preserve_house_number": True,
                },
            })
        names = [item["value"] for item in official["names"]]
        seed_card = {
            "generation_mode": "OBSERVED_COMPOSITE_ANALOGY_V2",
            "name_options": names,
            "enseigne_options": [
                item["value"] for item in official["names"] if item.get("kind") == "ENSEIGNE"
            ],
            "address": baseline["address"],
            "postcode": baseline["postcode"],
            "city": baseline["city"],
            "insee": baseline["insee"],
            "street_number": str(official["address"]["number"]),
            "street_type": str(official["address"]["street_type"]),
            "composite_contracts": contracts,
            "official_context": context["llm_view"],
            "qualification": context["qualification"],
            "internal_context": context["internal_context"],
            "context_sha256": context["context_sha256"],
        }
        output.append({
            "seed_id": f"COMPOSITE_V2:{context['target_siret']}",
            "target_siret": context["target_siret"],
            "target_siren": context["target_siren"],
            "source_kind": "SIRENE_ONLY_TRAIN",
            "oof_fold": -1,
            "legacy_split": "train_synthetic",
            "seed_card": seed_card,
            "observed_train_profile": {
                "schema_version": "sireto-synthetic-composite-evidence-2",
                "rows": len(rows(inspiration_path)),
                "source_sha256": sha256(inspiration_path),
                "source_folds": [2, 3, 4],
                "supported_families": ["OBSERVED_COMPOSITE_ANALOGY"],
            },
            "risk_flags": [
                "COMPOSITE_LLM_WRITTEN",
                "MULTI_SITE_SIREN" if count_relations(context, "SAME_SIREN") else "SINGLE_SITE_SIREN",
                "TARGET_CLOSED" if official["state"] == "F" else "TARGET_ACTIVE",
            ],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    loop.write_jsonl_atomic(args.output, output)
    states = Counter(row["seed_card"]["official_context"]["target"]["state"] for row in output)
    manifest = {
        "schema_version": "sireto-synthetic-gt-composite-pilot-manifest-2",
        "rows": len(output),
        "planned_pairs": len(output) * 3,
        "state_counts": dict(sorted(states.items())),
        "multi_site_targets": sum("MULTI_SITE_SIREN" in row["risk_flags"] for row in output),
        "multi_active_targets": sum(
            any(
                "SAME_SIREN" in candidate.get("relation_tags", []) and candidate.get("state") == "A"
                for candidate in row["seed_card"]["internal_context"]
            )
            for row in output
        ),
        "distinct_inspiration_refs": len(used_refs),
        "text_generation": "none_selector_only",
        "selection_seed": args.selection_seed,
        "source_hashes": {
            "plan": sha256(args.plan),
            "official_context": sha256(official_path),
            "train_inspiration_bank": sha256(inspiration_path),
        },
        "output_sha256": sha256(args.output),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=ROOT / "config/synthetic_gt_composite_v2_plan.json")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--selection-seed", default="SIRETO-COMPOSITE-PILOT-V2")
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
