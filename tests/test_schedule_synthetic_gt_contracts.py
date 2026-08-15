from scripts import schedule_synthetic_gt_contracts as schedule


def test_feasible_contracts_use_observed_ocr_pairs_and_distinct_dimensions():
    card = {
        "name_options": ["ALPHA BETA"], "enseigne_options": [],
        "address": "12 RUE DES LILAS", "street_type": "RUE",
        "city": "PARIS", "postcode": "75001", "insee": "75056",
        "ocr_substitution_pairs": [{"source": "a", "target": "e", "count": 1}],
        "address_ocr_substitution_pairs": [
            {"source": "i", "target": "l", "count": 1}
        ],
    }
    values = schedule.feasible_contracts(card)
    assert any(
        value["name"] == "TOKEN_ORDER"
        and value["address"] == "ADDRESS_ABBREVIATION"
        and value["orthographic"] == "OCR_LIMITED"
        for value in values
    )
    assert all(
        not (
            value["name"] == value["orthographic"] == "OCR_LIMITED"
        )
        for value in values
    )
