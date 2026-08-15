#!/usr/bin/env python3
"""Build an opaque train-only bank of real official-to-CRM compound deltas.

The bank contains existing CRM text from allowed train folds and its official
SIRENE baseline.  It never creates a corruption.  SIRET, SIREN, CRM IDs and
query IDs are removed; Luna later uses the real pairs only as style examples
when it directly writes new SIRENE-only CRM variants.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts import agentic_synthetic_gt_orchestrator as source  # noqa: E402
from scripts import build_synthetic_gt_compound_evidence as evidence  # noqa: E402
from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402


FIELDS = ("name", "address", "postcode", "city", "insee")


def clean(value: Any) -> str:
    return evidence.clean(value)


def sha256(path: Path) -> str:
    return evidence.sha256(path)


def best_official_name(record: dict[str, Any], observed: str) -> str:
    values = list(dict.fromkeys(clean(value) for value in record.get("names", []) if clean(value)))
    if not values:
        return ""
    target = loop.normalized_alnum(observed)
    return min(
        values,
        key=lambda value: (
            loop.edit_distance(loop.normalized_alnum(value), target),
            abs(len(loop.normalized_alnum(value)) - len(target)),
            loop.normalized_surface(value),
        ),
    )


def bin_count(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3_4"
    if value <= 8:
        return "5_8"
    return "9_PLUS"


def ratio_bin(source_value: str, target_value: str) -> str:
    if not source_value:
        return "NO_SOURCE"
    ratio = len(target_value) / len(source_value)
    if ratio < 0.5:
        return "LT_050"
    if ratio < 0.8:
        return "050_079"
    if ratio <= 1.2:
        return "080_120"
    if ratio <= 1.5:
        return "121_150"
    return "GT_150"


def surface_changed(left: str, right: str) -> bool:
    return loop.normalized_surface(left) != loop.normalized_surface(right)


def multiset_subset(target: list[str], source_words: list[str]) -> bool:
    return not (Counter(target) - Counter(source_words))


def expanded_address_words(value: str) -> list[str]:
    return [
        loop.STREET_TYPE_ABBREVIATIONS.get(word.upper(), word.upper())
        for word in loop.normalized_words(value)
    ]


def added_marks(official: str, observed: str) -> list[str]:
    source_punctuation = Counter(
        character for character in official if character in loop.PUNCTUATION
    )
    target_punctuation = Counter(
        character for character in observed if character in loop.PUNCTUATION
    )
    result = list((target_punctuation - source_punctuation).elements())
    if sum(loop.has_diacritic(character) for character in observed) > sum(
        loop.has_diacritic(character) for character in official
    ):
        result.append("DIACRITIC")
    return sorted(result)


def inspiration_row(
    crm_row: pd.Series,
    record: dict[str, Any],
    target_siret: str,
    salt: str,
) -> dict[str, Any] | None:
    official = {
        "name": best_official_name(record, clean(crm_row["crm_name"])),
        "address": evidence.official_address(record),
        "postcode": clean(record["postcode"]),
        "city": clean(record["city"]),
        "insee": clean(record["insee"]),
    }
    observed = {
        "name": clean(crm_row["crm_name"]),
        "address": clean(crm_row["crm_adresse"]),
        "postcode": clean(crm_row["crm_cp"]),
        "city": clean(crm_row["crm_commune"]),
        "insee": clean(crm_row["crm_insee"]),
    }
    if not official["name"] or not official["address"] or not observed["name"]:
        return None
    if loop.leaked_identifier(observed, target_siret, target_siret[:9]):
        return None
    changed = [field for field in FIELDS if surface_changed(official[field], observed[field])]
    if "name" not in changed or not ({"address", "city"} & set(changed)):
        return None
    if observed["postcode"] not in {"", official["postcode"]}:
        return None
    if observed["insee"] not in {"", official["insee"]}:
        return None
    if not multiset_subset(
        loop.normalized_words(observed["name"]),
        loop.normalized_words(official["name"]) + loop.normalized_words(official["city"]),
    ):
        return None
    if not multiset_subset(
        expanded_address_words(observed["address"]),
        expanded_address_words(official["address"])
        + loop.normalized_words(official["city"])
        + loop.normalized_words(official["postcode"]),
    ):
        return None
    if not multiset_subset(
        loop.normalized_words(observed["city"]), loop.normalized_words(official["city"])
    ):
        return None
    source_numbers = {
        word for value in official.values() for word in loop.normalized_words(value) if word.isdigit()
    }
    target_numbers = {
        word for value in observed.values() for word in loop.normalized_words(value) if word.isdigit()
    }
    if not target_numbers.issubset(source_numbers):
        return None
    official_house_numbers = {
        word for word in loop.normalized_words(clean(record.get("number"))) if word.isdigit()
    }
    observed_address_numbers = {
        word for word in loop.normalized_words(observed["address"]) if word.isdigit()
    }
    if official_house_numbers and not official_house_numbers.issubset(observed_address_numbers):
        return None
    name_class = evidence.classify_name(record["names"], record["enseigne"], observed["name"])
    address_class = evidence.classify_address(official["address"], observed["address"])
    city_class = evidence.classify_geo("city", official["city"], observed["city"])
    missing = sorted(field for field in FIELDS if not observed[field])
    signature = {
        "changed_fields": changed,
        "missing_fields": missing,
        "name_relation": f"{name_class['family']}:{name_class['delta_class']}",
        "address_relation": f"{address_class['family']}:{address_class['delta_class']}",
        "city_relation": f"{city_class['family']}:{city_class['delta_class']}",
        "name_source_tokens": bin_count(len(loop.normalized_words(official["name"]))),
        "name_target_tokens": bin_count(len(loop.normalized_words(observed["name"]))),
        "address_source_tokens": bin_count(len(loop.normalized_words(official["address"]))),
        "address_target_tokens": bin_count(len(loop.normalized_words(observed["address"]))),
        "name_length_ratio": ratio_bin(official["name"], observed["name"]),
        "address_length_ratio": ratio_bin(official["address"], observed["address"]),
    }
    ref = hashlib.sha256(f"{salt}|{target_siret}".encode("utf-8")).hexdigest()
    return {
        "schema_version": "sireto-synthetic-gt-inspiration-1",
        "inspiration_ref": ref,
        "source_fold": int(crm_row["oof_fold"]),
        "source_legacy_split": str(crm_row["legacy_split"]),
        "source_state": clean(record["state"]),
        "official": official,
        "observed_crm": observed,
        "structural_signature": signature,
        "analogy_safety": {
            "lexical_tokens_subset_of_official": True,
            "numeric_tokens_subset_of_official": True,
            "added_marks": {
                field: added_marks(official[field], observed[field])
                for field in ("name", "address", "city")
            },
        },
        "provenance_digest": loop.digest_json({
            "ref": ref,
            "fold": int(crm_row["oof_fold"]),
            "official": official,
            "observed_crm": observed,
            "signature": signature,
        }),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.resolve()
    plan = source.load_plan(plan_path)
    verified = source.verify_sources(plan, plan_path)
    population, _components, _sirens, _all = source.load_population(plan, verified)
    train = population.copy()
    targets = list(train["gt_siret"].map(clean))
    frame = source._query_frame(Path(verified["sirene_establishments"]["path"]), targets)
    units = source._query_units(
        Path(verified["sirene_legal_units"]["path"]), [value[:9] for value in targets]
    )
    records = {
        clean(row["siret"]): source.record_from_row(row, units.get(clean(row["siren"])))
        for _, row in frame.iterrows()
    }
    candidates: list[tuple[str, dict[str, Any]]] = []
    for (_, crm_row), target_siret in zip(train.iterrows(), targets):
        value = inspiration_row(crm_row, records[target_siret], target_siret, args.salt)
        if value is not None:
            stable = hashlib.sha256(
                f"{args.selection_seed}|{value['inspiration_ref']}".encode("ascii")
            ).hexdigest()
            candidates.append((stable, value))
    candidates.sort(key=lambda item: item[0])
    selected: list[dict[str, Any]] = []
    seen_sirens: set[str] = set()
    # The SIREN is used only transiently to enforce independence and is never
    # written to the inspiration bank.
    ref_to_siren = {
        hashlib.sha256(f"{args.salt}|{target}".encode("utf-8")).hexdigest(): target[:9]
        for target in targets
    }
    for _, value in candidates:
        siren = ref_to_siren[value["inspiration_ref"]]
        if siren in seen_sirens:
            continue
        seen_sirens.add(siren)
        selected.append(value)
    signature_counts = Counter(
        loop.digest_json(value["structural_signature"]) for value in selected
    )
    fold_counts = Counter(str(value["source_fold"]) for value in selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    loop.write_jsonl_atomic(args.output, selected)
    manifest = {
        "schema_version": "sireto-synthetic-gt-inspiration-bank-manifest-1",
        "rows": len(selected),
        "distinct_source_sirens": len(seen_sirens),
        "fold_counts": dict(sorted(fold_counts.items())),
        "distinct_structural_signatures": len(signature_counts),
        "top_signature_count": max(signature_counts.values(), default=0),
        "selection_seed": args.selection_seed,
        "source_hashes": {key: value["sha256"] for key, value in verified.items()},
        "plan_sha256": sha256(plan_path),
        "output_sha256": sha256(args.output),
        "identity_fields_published": [],
        "examples_are_existing_train_text": True,
        "text_generation": "none",
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
    result.add_argument("--plan", type=Path, default=ROOT / "config/synthetic_gt_corpus_plan.json")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--selection-seed", type=int, default=42)
    result.add_argument("--salt", default="SIRETO-INSPIRATION-BANK-V1")
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
