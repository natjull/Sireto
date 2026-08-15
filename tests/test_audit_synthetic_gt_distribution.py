from scripts.audit_synthetic_gt_distribution import summary


def test_summary_reports_families_and_seed_counts():
    rows = [
        {"target_siret": "12345678900011", "crm": {"name": "A", "address": "1 RUE A", "postcode": "75001", "city": "PARIS", "insee": "75056"}, "corruption_families_observed": ["FIELD_MISSING"]},
        {"target_siret": "12345678900011", "crm": {"name": "A", "address": "1 R A", "postcode": "75001", "city": "PARIS", "insee": ""}, "corruption_families_observed": ["ADDRESS_ABBREVIATION"]},
    ]
    result = summary(rows)
    assert result["rows"] == 2
    assert result["seeds"] == 1
    assert result["variants_per_seed"] == {2: 1}
    assert result["families"]["FIELD_MISSING"] == 1
