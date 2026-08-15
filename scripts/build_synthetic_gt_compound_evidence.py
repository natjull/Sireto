#!/usr/bin/env python3
"""Measure train-only compound CRM deltas against official SIRENE records.

This module is evidence collection only.  It never creates or edits synthetic
CRM text.  It classifies observed train fields with finite, deterministic
rules and publishes counts, distinct-SIREN support, and real train examples
for later direct generation by Luna.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts import agentic_synthetic_gt_orchestrator as source  # noqa: E402
from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402


NON_OPERATIONAL = {"EXACT", "CASE_STYLE"}


def clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.upper() in {"", "[ND]", "NAN", "NONE"} else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def delta_marks(official: str, observed: str) -> str:
    left_punctuation = Counter(character for character in official if character in loop.PUNCTUATION)
    right_punctuation = Counter(character for character in observed if character in loop.PUNCTUATION)
    removed = sorted((left_punctuation - right_punctuation).elements())
    added = sorted((right_punctuation - left_punctuation).elements())
    left_diacritics = sum(bool(loop.has_diacritic(character)) for character in official)
    right_diacritics = sum(bool(loop.has_diacritic(character)) for character in observed)
    parts: list[str] = []
    if removed:
        parts.append("punctuation_removed:" + "".join(removed))
    if added:
        parts.append("punctuation_added:" + "".join(added))
    if right_diacritics < left_diacritics:
        parts.append("diacritic_removed")
    elif right_diacritics > left_diacritics:
        parts.append("diacritic_added")
    return "+".join(parts) or "spacing_or_separator"


def strip_legal(words: Iterable[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for word in words:
        if word.upper() in loop.LEGAL_FORM_TOKENS:
            removed.append(word.upper())
        else:
            kept.append(word)
    return kept, removed


def classify_name(
    official_values: Iterable[str], enseignes: Iterable[str], observed: Any
) -> dict[str, str]:
    target = clean(observed)
    sources = list(dict.fromkeys(clean(value) for value in official_values if clean(value)))
    if not target:
        return {"family": "FIELD_MISSING", "delta_class": "name_missing", "official": sources[0] if sources else ""}
    if not sources:
        return {"family": "UNCLASSIFIED", "delta_class": "no_official_name", "official": ""}
    for value in sources:
        if target == value:
            return {"family": "EXACT", "delta_class": "byte_exact", "official": value}
    for value in sources:
        if loop.normalized_surface(target) == loop.normalized_surface(value):
            return {"family": "CASE_STYLE", "delta_class": "case_or_space", "official": value}
    enseigne_surfaces = {
        loop.normalized_surface(value): value for value in enseignes if clean(value)
    }
    if loop.normalized_surface(target) in enseigne_surfaces:
        return {
            "family": "ENSEIGNE_VS_DENOMINATION",
            "delta_class": "official_enseigne",
            "official": sources[0],
        }
    for value in sources:
        if loop.normalized_alnum(target) == loop.normalized_alnum(value):
            return {
                "family": "ACCENT_PUNCTUATION",
                "delta_class": delta_marks(value, target),
                "official": value,
            }
    for value in sources:
        left = loop.normalized_words(value)
        right = loop.normalized_words(target)
        if left and Counter(left) == Counter(right) and left != right:
            delta = "two_token_reverse" if len(left) == 2 and right == list(reversed(left)) else f"permutation_{len(left)}"
            return {"family": "TOKEN_ORDER", "delta_class": delta, "official": value}
    for value in sources:
        left = loop.normalized_words(value)
        right = loop.normalized_words(target)
        left_kept, left_legal = strip_legal(left)
        right_kept, right_legal = strip_legal(right)
        if left_kept == right_kept and left_legal != right_legal and (left_legal or right_legal):
            removed = sorted(set(left_legal) - set(right_legal))
            added = sorted(set(right_legal) - set(left_legal))
            delta = "remove:" + "+".join(removed) if removed else "add:" + "+".join(added)
            return {"family": "LEGAL_FORM", "delta_class": delta, "official": value}
    return {"family": "UNCLASSIFIED", "delta_class": "name_other", "official": sources[0]}


def official_address(record: dict[str, Any]) -> str:
    return " ".join(
        value for value in (
            clean(record.get("number")), clean(record.get("street_type")), clean(record.get("street"))
        ) if value
    )


def abbreviation_delta(official: str, observed: str) -> str | None:
    left = loop.normalized_words(official)
    right = loop.normalized_words(observed)
    expand = lambda words: [
        loop.STREET_TYPE_ABBREVIATIONS.get(word.upper(), word.upper()) for word in words
    ]
    if expand(left) != expand(right):
        return None
    pairs = [
        f"{source_word.upper()}->{target_word.upper()}"
        for source_word, target_word in zip(left, right)
        if source_word.upper() != target_word.upper()
    ]
    return "+".join(pairs) if pairs else None


def classify_address(official: Any, observed: Any) -> dict[str, str]:
    source_value = clean(official)
    target = clean(observed)
    if not target:
        return {"family": "FIELD_MISSING", "delta_class": "address_missing", "official": source_value}
    if target == source_value:
        return {"family": "EXACT", "delta_class": "byte_exact", "official": source_value}
    if loop.normalized_surface(target) == loop.normalized_surface(source_value):
        return {"family": "CASE_STYLE", "delta_class": "case_or_space", "official": source_value}
    abbreviation = abbreviation_delta(source_value, target)
    if abbreviation:
        return {"family": "ADDRESS_ABBREVIATION", "delta_class": abbreviation, "official": source_value}
    if loop.normalized_alnum(target) == loop.normalized_alnum(source_value):
        return {
            "family": "ACCENT_PUNCTUATION",
            "delta_class": delta_marks(source_value, target),
            "official": source_value,
        }
    left = loop.normalized_words(source_value)
    right = loop.normalized_words(target)
    if left and Counter(left) == Counter(right) and left != right:
        delta = "two_token_reverse" if len(left) == 2 and right == list(reversed(left)) else f"permutation_{len(left)}"
        return {"family": "ADDRESS_TOKEN_ORDER", "delta_class": delta, "official": source_value}
    return {"family": "UNCLASSIFIED", "delta_class": "address_other", "official": source_value}


def classify_geo(field: str, official: Any, observed: Any) -> dict[str, str]:
    source_value = clean(official)
    target = clean(observed)
    if not target:
        return {"family": "FIELD_MISSING", "delta_class": f"{field}_missing", "official": source_value}
    if target == source_value:
        return {"family": "EXACT", "delta_class": "byte_exact", "official": source_value}
    if loop.normalized_surface(target) == loop.normalized_surface(source_value):
        return {"family": "CASE_STYLE", "delta_class": "case_or_space", "official": source_value}
    if field == "city" and loop.normalized_alnum(target) == loop.normalized_alnum(source_value):
        return {
            "family": "ACCENT_PUNCTUATION",
            "delta_class": delta_marks(source_value, target),
            "official": source_value,
        }
    return {"family": "UNCLASSIFIED", "delta_class": f"{field}_other", "official": source_value}


def operation_signature(field: str, classification: dict[str, str]) -> str:
    return f"{field}:{classification['family']}:{classification['delta_class']}"


def build(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.resolve()
    plan = source.load_plan(plan_path)
    verified = source.verify_sources(plan, plan_path)
    population, _components, _sirens, _all = source.load_population(plan, verified)
    crm = pd.read_csv(
        verified["crm_ok_gt"]["path"], sep=";", dtype=str, keep_default_na=False
    ).reset_index(names="query_id")
    crm["query_id"] = crm["query_id"].astype(str)
    train = crm.iloc[population["query_id"].astype(int).tolist()].copy()
    targets = list(population["gt_siret"].map(clean))
    frame = source._query_frame(Path(verified["sirene_establishments"]["path"]), targets)
    units = source._query_units(
        Path(verified["sirene_legal_units"]["path"]), [value[:9] for value in targets]
    )
    records = {
        clean(row["siret"]): source.record_from_row(row, units.get(clean(row["siren"])))
        for _, row in frame.iterrows()
    }
    if set(targets) != set(records):
        raise RuntimeError("train target missing from official SIRENE evidence")

    signature_rows: Counter[str] = Counter()
    signature_sirens: dict[str, set[str]] = defaultdict(set)
    family_rows: Counter[str] = Counter()
    family_sirens: dict[str, set[str]] = defaultdict(set)
    pattern_rows: Counter[str] = Counter()
    pattern_sirens: dict[str, set[str]] = defaultdict(set)
    changed_field_sets: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    classified_rows = 0
    composite_rows = 0

    for (_, crm_row), target_siret in zip(train.iterrows(), targets):
        record = records[target_siret]
        siren = target_siret[:9]
        classifications = {
            "name": classify_name(record["names"], record["enseigne"], crm_row["crm_name"]),
            "address": classify_address(official_address(record), crm_row["crm_adresse"]),
            "postcode": classify_geo("postcode", record["postcode"], crm_row["crm_cp"]),
            "city": classify_geo("city", record["city"], crm_row["crm_commune"]),
            "insee": classify_geo("insee", record["insee"], crm_row["crm_insee"]),
        }
        operations = [
            operation_signature(field, value)
            for field, value in classifications.items()
            if value["family"] not in NON_OPERATIONAL
        ]
        recognized = [value for value in operations if ":UNCLASSIFIED:" not in value]
        changed_fields = sorted(
            field for field, value in classifications.items() if value["family"] not in NON_OPERATIONAL
        )
        changed_field_sets["+".join(changed_fields) or "NONE"] += 1
        if not any(value["family"] == "UNCLASSIFIED" for value in classifications.values()):
            classified_rows += 1
        if len(recognized) >= 2:
            composite_rows += 1
        pattern = "|".join(sorted(recognized)) or "NO_RECOGNIZED_OPERATION"
        pattern_rows[pattern] += 1
        pattern_sirens[pattern].add(siren)
        for field, value in classifications.items():
            signature = operation_signature(field, value)
            signature_rows[signature] += 1
            signature_sirens[signature].add(siren)
            family_rows[value["family"]] += 1
            family_sirens[value["family"]].add(siren)
        if len(examples[pattern]) < args.examples_per_pattern:
            examples[pattern].append({
                "example_ref": hashlib.sha256(f"{args.example_salt}|{target_siret}".encode()).hexdigest(),
                "official": {
                    "name": classifications["name"]["official"],
                    "address": classifications["address"]["official"],
                    "postcode": clean(record["postcode"]),
                    "city": clean(record["city"]),
                    "insee": clean(record["insee"]),
                },
                "observed_crm": {
                    "name": clean(crm_row["crm_name"]),
                    "address": clean(crm_row["crm_adresse"]),
                    "postcode": clean(crm_row["crm_cp"]),
                    "city": clean(crm_row["crm_commune"]),
                    "insee": clean(crm_row["crm_insee"]),
                },
                "classifications": classifications,
            })

    signature_evidence = {
        signature: {"rows": count, "distinct_sirens": len(signature_sirens[signature])}
        for signature, count in sorted(signature_rows.items())
    }
    family_evidence = {
        family: {"rows": count, "distinct_sirens": len(family_sirens[family])}
        for family, count in sorted(family_rows.items())
    }
    compound_patterns = [
        {
            "pattern": pattern,
            "rows": count,
            "distinct_sirens": len(pattern_sirens[pattern]),
            "examples": examples[pattern],
        }
        for pattern, count in sorted(pattern_rows.items(), key=lambda item: (-item[1], item[0]))
    ]
    result = {
        "schema_version": "sireto-synthetic-gt-compound-evidence-1",
        "rows": len(train),
        "fully_classified_rows": classified_rows,
        "recognized_composite_rows": composite_rows,
        "changed_field_sets": dict(sorted(changed_field_sets.items())),
        "family_evidence": family_evidence,
        "signature_evidence": signature_evidence,
        "compound_patterns": compound_patterns,
        "classification_policy": {
            "operations_are_observed_baseline_to_crm_deltas": True,
            "case_style_is_not_an_operation": True,
            "unclassified_is_not_generation_evidence": True,
            "minimum_sirens_for_generation": args.minimum_sirens,
        },
        "eligible_signatures": sorted(
            signature for signature, evidence in signature_evidence.items()
            if evidence["distinct_sirens"] >= args.minimum_sirens
            and ":UNCLASSIFIED:" not in signature
            and not any(f":{family}:" in signature for family in NON_OPERATIONAL)
        ),
        "provenance": {
            "plan_sha256": sha256(plan_path),
            "source_sha256": {key: value["sha256"] for key, value in verified.items()},
            "allowed_folds": plan["generator"]["allowed_oof_folds"],
            "allowed_legacy_split": plan["generator"]["allowed_legacy_split"],
            "examples_are_real_train_rows": True,
            "text_generation": "none",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "rows": len(train),
        "fully_classified_rows": classified_rows,
        "recognized_composite_rows": composite_rows,
        "eligible_signatures": result["eligible_signatures"],
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=ROOT / "config/synthetic_gt_corpus_plan.json")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--minimum-sirens", type=int, default=20)
    result.add_argument("--examples-per-pattern", type=int, default=3)
    result.add_argument("--example-salt", default="SIRETO-COMPOUND-EVIDENCE-V1")
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
