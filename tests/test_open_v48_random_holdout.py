from __future__ import annotations

import pandas as pd
import pytest

from scripts.open_v48_random_holdout import (
    create_exclusive_ledger,
    parse_args,
    random_gate,
)


def _predictions(
    *,
    variant: str,
    correct_auto: int,
    negative_auto: int,
) -> pd.DataFrame:
    rows = []
    for index in range(46):
        rows.append(
            {
                "query_id": f"correct-{index}",
                "adjudication_label": "TOP1_CORRECT",
                "acceptor_target": 1,
                "auto": index < correct_auto,
                "variant": variant,
            }
        )
    for index in range(5):
        rows.append(
            {
                "query_id": f"wrong-{index}",
                "adjudication_label": "TOP1_WRONG",
                "acceptor_target": 0,
                "auto": index < negative_auto,
                "variant": variant,
            }
        )
    rows.append(
        {
            "query_id": "ambiguous",
            "adjudication_label": "AMBIGUOUS",
            "acceptor_target": 0,
            "auto": negative_auto > 5,
            "variant": variant,
        }
    )
    return pd.DataFrame(rows)


def test_random_gate_passes_exact_contract() -> None:
    winner = _predictions(variant="HARD_W1", correct_auto=30, negative_auto=0)
    frozen = _predictions(variant="BASE_FROZEN", correct_auto=31, negative_auto=2)
    gate = random_gate(
        winner_predictions=winner,
        frozen_predictions=frozen,
    )
    assert gate["passed"]
    assert gate["correct_auto_delta"] == -1


def test_random_gate_rejects_one_negative_auto() -> None:
    winner = _predictions(variant="HARD_W1", correct_auto=30, negative_auto=1)
    frozen = _predictions(variant="BASE_FROZEN", correct_auto=30, negative_auto=0)
    gate = random_gate(
        winner_predictions=winner,
        frozen_predictions=frozen,
    )
    assert not gate["passed"]
    assert not gate["checks"]["zero_negative_auto"]


def test_random_gate_rejects_coverage_loss() -> None:
    winner = _predictions(variant="HARD_W1", correct_auto=28, negative_auto=0)
    frozen = _predictions(variant="BASE_FROZEN", correct_auto=30, negative_auto=0)
    gate = random_gate(
        winner_predictions=winner,
        frozen_predictions=frozen,
    )
    assert not gate["passed"]
    assert not gate["checks"]["at_most_one_correct_auto_below_frozen"]


def test_random_gate_rejects_unpaired_ids() -> None:
    winner = _predictions(variant="HARD_W1", correct_auto=30, negative_auto=0)
    frozen = _predictions(variant="BASE_FROZEN", correct_auto=30, negative_auto=0)
    frozen.loc[0, "query_id"] = "different-query"
    with pytest.raises(ValueError, match="paired random IDs"):
        random_gate(
            winner_predictions=winner,
            frozen_predictions=frozen,
        )


def test_global_ledger_is_exclusive(tmp_path) -> None:
    ledger = tmp_path / "OPENING_LEDGER.json"
    create_exclusive_ledger(ledger, {"opening_status": "OPENED_ONCE_GLOBAL"})
    with pytest.raises(FileExistsError):
        create_exclusive_ledger(ledger, {"opening_status": "SECOND_OPEN"})


@pytest.mark.parametrize(
    "forbidden_argument",
    ["--no-canonical-checks", "--output-root", "--development-runner"],
)
def test_cli_has_no_noncanonical_bypass(monkeypatch, forbidden_argument) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["open_v48_random_holdout.py", forbidden_argument, "/tmp/bypass"],
    )
    with pytest.raises(SystemExit):
        parse_args()
