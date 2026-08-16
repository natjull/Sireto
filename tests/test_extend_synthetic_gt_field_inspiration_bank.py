from __future__ import annotations

from scripts import extend_synthetic_gt_field_inspiration_bank as extension
from scripts.manage_synthetic_gt_balanced_registry import fragment_operator


def fragment(ref: str, relation: str, parameters: dict, fold: int = 2) -> dict:
    return {
        "inspiration_ref": ref,
        "field": "address",
        "relation": relation,
        "operation_parameters": parameters,
        "official_value": "12 RUE TEST",
        "observed_crm_value": "12 R TEST",
        "source_fold": fold,
        "source_legacy_split": "train",
        "source_state": "A",
        "provenance_digest": ref,
    }


def test_extension_adds_only_novel_binding_operators() -> None:
    base = [fragment("base", "ADDRESS_ABBREVIATE", {
        "pairs": [{"source": "RUE", "target": "R"}],
    })]
    duplicate = fragment("duplicate", "ADDRESS_ABBREVIATE", {
        "pairs": [{"source": "RUE", "target": "R"}],
    })
    novel = fragment("novel", "ADDRESS_ABBREVIATE", {
        "pairs": [{"source": "RUE", "target": "R."}],
    })
    ignored = fragment("ignored", "PUNCTUATION_REMOVED", {
        "edits": [{"after_token_index": 1, "mark": "-", "replacement": " "}],
    })
    combined, additions = extension.extend_rows(
        base, [duplicate, novel, ignored]
    )
    assert len(combined) == 2
    assert len(additions) == 1
    assert fragment_operator(additions[0]) == fragment_operator(novel)
    assert additions[0]["inspiration_ref"] not in {"base", "novel"}


def test_extension_rejects_non_train_base_fold() -> None:
    bad = fragment("bad", "ADDRESS_ABBREVIATE", {
        "pairs": [{"source": "RUE", "target": "R"}],
    }, fold=1)
    try:
        extension.extend_rows([bad], [])
    except ValueError as error:
        assert "folds-2/3/4-only" in str(error)
    else:
        raise AssertionError("protected fold must fail closed")
