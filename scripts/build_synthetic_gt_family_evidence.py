#!/usr/bin/env python3
"""Build train-only corruption evidence by comparing CRM with official SIRENE.

This command only measures observed differences.  It never creates or edits a
CRM value and therefore cannot generate synthetic training text.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


def clean(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.upper() in {"", "[ND]", "NAN", "NONE"} else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_substitutions(official: Any, observed: Any) -> list[tuple[str, str]]:
    left = loop.normalized_alnum(official)
    right = loop.normalized_alnum(observed)
    if not left or len(left) != len(right) or left == right:
        return []
    maximum = min(3, max(1, len(left) // 10))
    if loop.edit_distance(left, right) > maximum:
        return []
    return [(a, b) for a, b in zip(left, right) if a != b]


def same_tokens_different_order(official: Any, observed: Any) -> bool:
    left = loop.normalized_words(official)
    right = loop.normalized_words(observed)
    return bool(left and Counter(left) == Counter(right) and left != right)


def first_matching_pair(
    official_values: Iterable[str],
    observed: str,
) -> list[tuple[str, str]]:
    candidates = [
        bounded_substitutions(value, observed)
        for value in official_values
        if clean(value)
    ]
    candidates = [value for value in candidates if value]
    return min(candidates, key=lambda value: (len(value), value)) if candidates else []


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

    base = json.loads(args.base_profile.read_text(encoding="utf-8"))
    phenomena = Counter({key: int(value) for key, value in base["phenomena"].items()})
    name_substitutions: Counter[tuple[str, str]] = Counter()
    address_substitutions: Counter[tuple[str, str]] = Counter()
    phenomena["ADDRESS_OCR"] = 0
    for (_, crm_row), target_siret in zip(train.iterrows(), targets):
        record = records[target_siret]
        crm_name = clean(crm_row["crm_name"])
        crm_address = clean(crm_row["crm_adresse"])
        official_names = [clean(value) for value in record["names"] if clean(value)]
        if any(same_tokens_different_order(value, crm_name) for value in official_names):
            phenomena["TOKEN_ORDER"] += 1
        name_pairs = first_matching_pair(official_names, crm_name)
        if name_pairs:
            phenomena["OCR_LIMITED"] += 1
            name_substitutions.update(name_pairs)

        official_address = " ".join(
            value for value in (
                clean(record["number"]), clean(record["street_type"]), clean(record["street"])
            ) if value
        )
        if same_tokens_different_order(official_address, crm_address):
            phenomena["ADDRESS_TOKEN_ORDER"] += 1
        official_digits = "".join(character for character in official_address if character.isdigit())
        observed_digits = "".join(character for character in crm_address if character.isdigit())
        address_pairs = (
            bounded_substitutions(official_address, crm_address)
            if official_digits == observed_digits
            else []
        )
        if address_pairs:
            phenomena["ADDRESS_OCR"] += 1
            address_substitutions.update(address_pairs)

        enseignes = {loop.normalized_surface(value) for value in record["enseigne"] if clean(value)}
        official_primary = loop.normalized_surface(official_names[0]) if official_names else ""
        if (
            loop.normalized_surface(crm_name) in enseignes
            and loop.normalized_surface(crm_name) != official_primary
        ):
            phenomena["ENSEIGNE_VS_DENOMINATION"] += 1

    result = {
        **base,
        "schema_version": "sireto-synthetic-gt-observed-profile-2",
        "phenomena": dict(sorted(phenomena.items())),
        "supported_families": sorted(key for key, value in phenomena.items() if value > 0),
        "ocr_substitution_pairs": [
            {"source": left, "target": right, "count": count}
            for (left, right), count in sorted(
                name_substitutions.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "address_ocr_substitution_pairs": [
            {"source": left, "target": right, "count": count}
            for (left, right), count in sorted(
                address_substitutions.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "evidence": {
            "comparison": "CRM_OK_GT_STRICT_TRAIN_VS_OFFICIAL_SIRENE",
            "allowed_rows": len(train),
            "plan_sha256": sha256(plan_path),
            "base_profile_sha256": sha256(args.base_profile),
            "source_sha256": {key: value["sha256"] for key, value in verified.items()},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "rows": len(train),
        "phenomena": result["phenomena"],
        "ocr_pair_count": len(result["ocr_substitution_pairs"]),
        "address_ocr_pair_count": len(result["address_ocr_substitution_pairs"]),
        "sha256": sha256(args.output),
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, default=ROOT / "config/synthetic_gt_corpus_plan.json")
    result.add_argument("--base-profile", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
