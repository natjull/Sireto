import json

import pandas as pd
import pytest

from scripts.prepare_v4_ambiguous_retrieval import prepare_role


def test_prepare_role_uses_probe_only_for_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "query_id": [f"q{index}" for index in range(142)],
            "label_kind": ["AMBIGUOUS"] * 142,
            "direct_active_sirets_json": [
                json.dumps(["11111111100001", "22222222200002"])
            ]
            * 142,
        }
    )
    output = prepare_role(frame, "fit_ambiguous")
    assert output.loc[0, "diagnostic_probe_siret"] == "11111111100001"
    assert output.loc[0, "retrieval_uses_diagnostic_probe"] == False  # noqa: E712
    assert output.loc[0, "split"] == "fit_ambiguous"


def test_prepare_role_rejects_non_ambiguous_evidence() -> None:
    frame = pd.DataFrame(
        {
            "query_id": ["q1"] * 142,
            "label_kind": ["AMBIGUOUS"] * 142,
            "direct_active_sirets_json": [json.dumps(["one"])] * 142,
        }
    )
    with pytest.raises(ValueError, match="at least two"):
        prepare_role(frame, "fit_ambiguous")
