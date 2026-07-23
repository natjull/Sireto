from src.xgb_matcher.features import (
    FEATURE_NAMES,
    make_feature_rows_from_preprocessed,
    preprocess_crm_row,
)


def test_train_serve_contract_has_same_54_ordered_values():
    crm = preprocess_crm_row(
        {
            "crm_name": "École Saint-Joseph",
            "crm_address": "12 rue de l'Église",
            "crm_city": "Lyon",
            "postcode": "69001",
            "insee": "69123",
        }
    )
    candidates = [
        {
            "siret": "12345678900011",
            "siren": "123456789",
            "denomination": "OGEC SAINT JOSEPH",
            "enseigne1": "ECOLE SAINT JOSEPH",
            "address": "12 RUE DE L EGLISE",
            "postcode": "69001",
            "city": "LYON",
            "insee": "69123",
        }
    ]
    training_values = make_feature_rows_from_preprocessed(
        crm, candidates, include_semantic=False
    )[0]
    serving_values = make_feature_rows_from_preprocessed(
        crm, candidates, include_semantic=False
    )[0]
    assert len(FEATURE_NAMES) == 54
    assert list(training_values) == FEATURE_NAMES
    assert training_values == serving_values
