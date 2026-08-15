from scripts.build_sirene_candidate_cards import value


def test_nd_is_absent_not_an_identity_value():
    assert value("[ND]") == ""
    assert value("  Paris ") == "Paris"

