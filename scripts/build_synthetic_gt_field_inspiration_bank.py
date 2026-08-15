#!/usr/bin/env python3
"""Build train-only, field-level CRM edit inspirations without generating text.

Each output row is one real official-to-CRM field delta from folds 2/3/4.
Identifiers are replaced by opaque hashes.  Only deletion/reordering/surface
operators with an exact, transferable parameterization are retained; additions
of punctuation or diacritics are deliberately excluded.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts import agentic_synthetic_gt_orchestrator as source  # noqa: E402
from scripts import build_synthetic_gt_compound_evidence as evidence  # noqa: E402
from scripts import build_synthetic_gt_inspiration_bank as inspiration  # noqa: E402
from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402


FIELDS = ("name", "address", "city")
SAFE_RELATIONS = {
    "name": {
        "TOKEN_ORDER", "TOKEN_SUBSET", "LEGAL_FORM_REMOVE", "JOIN_SPLIT",
        "PUNCTUATION_REMOVED", "DIACRITIC_REMOVED",
    },
    "address": {
        "ADDRESS_ABBREVIATE", "ADDRESS_TYPE_ORDER", "ADDRESS_TOKEN_SUBSET", "JOIN_SPLIT",
        "PUNCTUATION_REMOVED", "DIACRITIC_REMOVED",
    },
    "city": {"JOIN_SPLIT", "PUNCTUATION_REMOVED", "DIACRITIC_REMOVED"},
}


def sha256(path: Path) -> str:
    return evidence.sha256(path)


def clean(value: Any) -> str:
    return evidence.clean(value)


def stable_positions(source_words: list[str], target_words: list[str]) -> list[int] | None:
    """Return the unique left-to-right retained positions, or no operator."""
    positions: list[int] = []
    cursor = 0
    for target in target_words:
        try:
            index = source_words.index(target, cursor)
        except ValueError:
            return None
        positions.append(index)
        cursor = index + 1
    return positions


def permutation(source_words: list[str], target_words: list[str]) -> list[int] | None:
    if Counter(source_words) != Counter(target_words):
        return None
    available: dict[str, list[int]] = {}
    for index, word in enumerate(source_words):
        available.setdefault(word, []).append(index)
    result: list[int] = []
    for word in target_words:
        result.append(available[word].pop(0))
    return result if result != list(range(len(source_words))) else None


def removed_punctuation(source_value: str, target_value: str) -> list[str] | None:
    left = Counter(character for character in source_value if character in loop.PUNCTUATION)
    right = Counter(character for character in target_value if character in loop.PUNCTUATION)
    if right - left:
        return None
    removed = sorted((left - right).elements())
    return removed or None


def diacritic_projection(value: str) -> list[tuple[str, tuple[str, ...]]]:
    result: list[tuple[str, tuple[str, ...]]] = []
    for character in value:
        decomposed = unicodedata.normalize("NFD", character)
        base = "".join(item.casefold() for item in decomposed if not unicodedata.combining(item))
        marks = tuple(sorted(unicodedata.name(item, "") for item in decomposed if unicodedata.combining(item)))
        if base.isalnum():
            result.append((base, marks))
    return result


def removed_diacritics(source_value: str, target_value: str) -> list[dict[str, Any]] | None:
    left = diacritic_projection(source_value)
    right = diacritic_projection(target_value)
    if len(left) != len(right) or [value[0] for value in left] != [value[0] for value in right]:
        return None
    edits = []
    for index, ((base, source_marks), (_, target_marks)) in enumerate(zip(left, right, strict=True)):
        if set(target_marks) - set(source_marks):
            return None
        removed = sorted(set(source_marks) - set(target_marks))
        if removed:
            edits.append({"alnum_index": index, "base": base, "removed_marks": removed})
    return edits or None


def abbreviation_pairs(source_value: str, target_value: str) -> list[dict[str, str]] | None:
    left = loop.normalized_words(source_value)
    right = loop.normalized_words(target_value)
    if len(left) != len(right) or loop.expanded_street_words(source_value) != loop.expanded_street_words(target_value):
        return None
    pairs = [
        {"source": source_word.upper(), "target": target_word.upper()}
        for source_word, target_word in zip(left, right, strict=True)
        if source_word.upper() != target_word.upper()
    ]
    return pairs or None


def operation_parameters(field: str, relation: str, source_value: str, target_value: str) -> dict[str, Any] | None:
    return loop.composite_operation_parameters(field, relation, source_value, target_value)


def field_fragments(
    crm_row: Any,
    record: dict[str, Any],
    target_siret: str,
    salt: str,
) -> list[dict[str, Any]]:
    official = {
        "name": inspiration.best_official_name(record, clean(crm_row["crm_name"])),
        "address": evidence.official_address(record),
        "city": clean(record["city"]),
    }
    observed = {
        "name": clean(crm_row["crm_name"]),
        "address": clean(crm_row["crm_adresse"]),
        "city": clean(crm_row["crm_commune"]),
    }
    result: list[dict[str, Any]] = []
    for field in FIELDS:
        source_value = official[field]
        target_value = observed[field]
        if not source_value or not target_value:
            continue
        relation = loop.composite_relation_class(field, source_value, target_value)
        if relation is None and field == "address":
            left, right = loop.normalized_words(source_value), loop.normalized_words(target_value)
            positions = loop.composite_stable_positions(left, right)
            if (
                positions is not None and right != left and len(right) >= 2
                and len(right) * 2 >= len(left)
                and [value for value in left if value.isdigit()]
                == [value for value in right if value.isdigit()]
            ):
                relation = "ADDRESS_TOKEN_SUBSET"
        if relation not in SAFE_RELATIONS[field]:
            continue
        parameters = operation_parameters(field, relation, source_value, target_value)
        if parameters is None:
            continue
        if field == "address":
            source_digits = [value for value in loop.normalized_words(source_value) if value.isdigit()]
            target_digits = [value for value in loop.normalized_words(target_value) if value.isdigit()]
            if source_digits != target_digits:
                continue
        ref = hashlib.sha256(f"{salt}|{target_siret}|{field}".encode("utf-8")).hexdigest()
        payload = {
            "schema_version": "sireto-synthetic-field-inspiration-1",
            "inspiration_ref": ref,
            "field": field,
            "relation": relation,
            "operation_parameters": parameters,
            "official_value": source_value,
            "observed_crm_value": target_value,
            "source_fold": int(crm_row["oof_fold"]),
            "source_legacy_split": str(crm_row["legacy_split"]),
            "source_state": clean(record["state"]),
        }
        payload["provenance_digest"] = loop.digest_json(payload)
        result.append(payload)
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.resolve()
    plan = source.load_plan(plan_path)
    verified = source.verify_sources(plan, plan_path)
    population, _components, _sirens, _all = source.load_population(plan, verified)
    targets = list(population["gt_siret"].map(clean))
    frame = source._query_frame(Path(verified["sirene_establishments"]["path"]), targets)
    units = source._query_units(
        Path(verified["sirene_legal_units"]["path"]), [value[:9] for value in targets]
    )
    records = {
        clean(row["siret"]): source.record_from_row(row, units.get(clean(row["siren"])))
        for _, row in frame.iterrows()
    }
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for (_, crm_row), target_siret in zip(population.iterrows(), targets, strict=True):
        for value in field_fragments(crm_row, records[target_siret], target_siret, args.salt):
            order = hashlib.sha256(
                f"{args.selection_seed}|{value['inspiration_ref']}".encode("ascii")
            ).hexdigest()
            candidates.append((order, target_siret[:9], value))
    candidates.sort(key=lambda value: value[0])
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _order, siren, value in candidates:
        key = (siren, value["field"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(value)
    selected.sort(key=lambda value: value["inspiration_ref"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    loop.write_jsonl_atomic(args.output, selected)
    relation_counts = Counter(f"{value['field']}:{value['relation']}" for value in selected)
    manifest = {
        "schema_version": "sireto-synthetic-field-inspiration-bank-manifest-1",
        "rows": len(selected),
        "distinct_refs": len({value["inspiration_ref"] for value in selected}),
        "relation_counts": dict(sorted(relation_counts.items())),
        "forbidden_relations": ["DIACRITIC_ADDED", "PUNCTUATION_ADDED"],
        "source_folds": sorted({value["source_fold"] for value in selected}),
        "source_hashes": {key: value["sha256"] for key, value in verified.items()},
        "plan_sha256": sha256(plan_path),
        "identity_fields_published": [],
        "text_generation": "none",
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
    result.add_argument("--plan", type=Path, default=ROOT / "config/synthetic_gt_corpus_plan.json")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--selection-seed", type=int, default=43)
    result.add_argument("--salt", default="SIRETO-FIELD-INSPIRATION-V1")
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
