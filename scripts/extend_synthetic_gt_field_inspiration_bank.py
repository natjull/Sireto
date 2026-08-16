#!/usr/bin/env python3
"""Extend the train-only field bank with novel exact operators only.

The extension is deliberately narrow: it revisits the already-authorized OOF
train folds 2/3/4 and retains one opaque proof for each exact structural
operator that is absent from the sealed base bank.  It never invents a surface,
uses a model, or publishes a source identity.
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts import agentic_synthetic_gt_orchestrator as source  # noqa: E402
from scripts import build_synthetic_gt_field_inspiration_bank as bank  # noqa: E402
from scripts import run_synthetic_gt_agentic_loop as loop  # noqa: E402
from scripts.manage_synthetic_gt_balanced_registry import fragment_operator  # noqa: E402


BINDING_RELATIONS = frozenset({
    ("name", "LEGAL_FORM_REMOVE"),
    ("name", "TOKEN_ORDER"),
    ("name", "TOKEN_SUBSET"),
    ("address", "ADDRESS_ABBREVIATE"),
    ("address", "ADDRESS_ALIAS_EXPAND"),
    ("address", "ADDRESS_TOKEN_SUBSET"),
})
ALLOWED_FOLDS = frozenset({2, 3, 4})
EXTENSION_SALT = "SIRETO-FIELD-BINDING-EXTENSION-V1"


def relation_key(value: dict[str, Any]) -> tuple[str, str]:
    return str(value.get("field")), str(value.get("relation"))


def opaque_extension_ref(value: dict[str, Any]) -> str:
    payload = {
        "prior_opaque_ref": str(value["inspiration_ref"]),
        "operator": fragment_operator(value),
        "source_fold": int(value["source_fold"]),
    }
    return hashlib.sha256(
        (EXTENSION_SALT + "|" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )).encode("utf-8")
    ).hexdigest()


def extend_rows(
    base_rows: Sequence[dict[str, Any]],
    candidates: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Append one deterministic train proof per previously unseen operator."""
    if any(int(value.get("source_fold", -1)) not in ALLOWED_FOLDS for value in base_rows):
        raise ValueError("base inspiration bank is not folds-2/3/4-only")
    base_operators = {fragment_operator(value) for value in base_rows}
    ordered = sorted(
        (
            value for value in candidates
            if relation_key(value) in BINDING_RELATIONS
            and int(value.get("source_fold", -1)) in ALLOWED_FOLDS
            and str(value.get("source_legacy_split")) == "train"
        ),
        key=lambda value: (
            relation_key(value), fragment_operator(value),
            str(value.get("inspiration_ref")),
        ),
    )
    additions: list[dict[str, Any]] = []
    seen = set(base_operators)
    for value in ordered:
        operator = fragment_operator(value)
        if operator in seen:
            continue
        seen.add(operator)
        addition = dict(value)
        addition["inspiration_ref"] = opaque_extension_ref(value)
        addition["provenance_digest"] = loop.digest_json({
            key: item for key, item in addition.items() if key != "provenance_digest"
        })
        additions.append(addition)
    additions.sort(key=lambda value: value["inspiration_ref"])
    combined = sorted(
        [*base_rows, *additions], key=lambda value: value["inspiration_ref"]
    )
    refs = [value["inspiration_ref"] for value in combined]
    if len(refs) != len(set(refs)):
        raise ValueError("opaque inspiration refs are not unique")
    return combined, additions


def build(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.resolve()
    plan = source.load_plan(plan_path)
    verified = source.verify_sources(plan, plan_path)
    population, _components, _sirens, _all = source.load_population(plan, verified)
    targets = list(population["gt_siret"].map(bank.clean))
    frame = source._query_frame(
        Path(verified["sirene_establishments"]["path"]), targets
    )
    units = source._query_units(
        Path(verified["sirene_legal_units"]["path"]),
        [value[:9] for value in targets],
    )
    records = {
        bank.clean(row["siret"]): source.record_from_row(
            row, units.get(bank.clean(row["siren"]))
        )
        for _, row in frame.iterrows()
    }
    candidates = [
        fragment
        for (_, crm_row), target_siret in zip(
            population.iterrows(), targets, strict=True
        )
        for fragment in bank.field_fragments(
            crm_row, records[target_siret], target_siret, EXTENSION_SALT
        )
    ]
    base_rows = [
        json.loads(line) for line in args.base_bank.open(encoding="utf-8")
        if line.strip()
    ]
    combined, additions = extend_rows(base_rows, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    loop.write_jsonl_atomic(args.output, combined)

    def counts(values: Sequence[dict[str, Any]]) -> dict[str, int]:
        return dict(sorted(Counter(
            f"{value['field']}:{value['relation']}" for value in values
        ).items()))

    def operator_counts(values: Sequence[dict[str, Any]]) -> dict[str, int]:
        grouped: dict[str, set[str]] = {}
        for value in values:
            grouped.setdefault(
                f"{value['field']}:{value['relation']}", set()
            ).add(fragment_operator(value))
        return {key: len(grouped[key]) for key in sorted(grouped)}

    manifest = {
        "schema_version": "sireto-synthetic-field-inspiration-bank-extension-1",
        "rows": len(combined),
        "extension_rows": len(additions),
        "distinct_refs": len({value["inspiration_ref"] for value in combined}),
        "relation_counts": counts(combined),
        "exact_operator_counts": operator_counts(combined),
        "added_relation_counts": counts(additions),
        "added_exact_operator_counts": operator_counts(additions),
        "targeted_binding_relations": [
            f"{field}:{relation}" for field, relation in sorted(BINDING_RELATIONS)
        ],
        "selection_rule": "ONE_TRAIN_PROOF_PER_NOVEL_EXACT_OPERATOR",
        "source_folds": sorted({int(value["source_fold"]) for value in combined}),
        "source_legacy_splits": sorted({
            str(value["source_legacy_split"]) for value in combined
        }),
        "base_bank": {
            "path": str(args.base_bank.resolve()),
            "sha256": bank.sha256(args.base_bank),
        },
        "source_hashes": {key: value["sha256"] for key, value in verified.items()},
        "plan_sha256": bank.sha256(plan_path),
        "identity_fields_published": [],
        "text_generation": "none",
        "model_or_retrieval_used": False,
        "output_sha256": bank.sha256(args.output),
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
        "--plan", type=Path, default=ROOT / "config/synthetic_gt_corpus_plan.json"
    )
    result.add_argument("--base-bank", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.set_defaults(func=build)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
