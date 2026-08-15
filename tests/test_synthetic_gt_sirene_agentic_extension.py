import pytest

from scripts.run_synthetic_gt_agentic_loop import validate_seed


def seed(source="SIRENE_ONLY_TRAIN", fold=-1):
    return {
        "seed_id": "SIRENE_ONLY_TRAIN:12345678900011",
        "target_siret": "12345678900011",
        "target_siren": "123456789",
        "source_kind": source,
        "oof_fold": fold,
        "legacy_split": "train_synthetic",
        "seed_card": {},
        "observed_train_profile": {},
        "risk_flags": [],
    }


def test_sirene_only_uses_explicit_non_fold_marker():
    assert validate_seed(seed())["oof_fold"] == -1


def test_sirene_only_cannot_claim_an_oof_fold():
    with pytest.raises(ValueError, match="oof_fold=-1"):
        validate_seed(seed(fold=2))


def test_crm_synthetic_still_requires_allowed_oof_fold():
    value = seed(source="SIRENE_SYNTHETIC", fold=2)
    value["seed_id"] = "SIRENE_SYNTHETIC:12345678900011"
    assert validate_seed(value)["oof_fold"] == 2
