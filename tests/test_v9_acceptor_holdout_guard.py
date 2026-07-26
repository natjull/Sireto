import json

import pytest

from scripts.train_v9_acceptor import validate_final_holdout_authorization


def test_final_holdout_guard_refuses_consumed_selective_test(tmp_path):
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "purpose": "downstream_final_holdout",
                "dataset_manifest_id": "dataset",
                "current_selective_test": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="consumed selective test"):
        validate_final_holdout_authorization(
            authorization,
            dataset_manifest_id="dataset",
        )


def test_final_holdout_guard_accepts_new_bound_dataset(tmp_path):
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "purpose": "downstream_final_holdout",
                "dataset_manifest_id": "dataset",
                "current_selective_test": False,
            }
        ),
        encoding="utf-8",
    )

    payload = validate_final_holdout_authorization(
        authorization,
        dataset_manifest_id="dataset",
    )

    assert payload["current_selective_test"] is False
