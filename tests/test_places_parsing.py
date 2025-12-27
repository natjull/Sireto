from pipe_v6.serper_places_client import _parse_french_address


def test_parse_french_address_range():
    address = "15-17 Rue Emile Zola, 69150 Decines-Charpieu"
    parsed = _parse_french_address(address)
    assert parsed["street_number"] == "15-17"
    assert parsed["street_name"] == "Rue Emile Zola"
    assert parsed["postcode"] == "69150"
    assert parsed["city"] == "Decines-Charpieu"


def test_parse_french_address_no_number():
    address = "Lieu-dit Le Carreau, 69960 Corbas"
    parsed = _parse_french_address(address)
    assert parsed["street_number"] is None
    assert parsed["street_name"] == "Lieu-dit Le Carreau"
    assert parsed["postcode"] == "69960"
    assert parsed["city"] == "Corbas"


def test_parse_french_address_bis():
    address = "1 BIS Rue Emile Zola, 69150 Decines-Charpieu"
    parsed = _parse_french_address(address)
    assert parsed["street_number"] == "1 BIS"
    assert parsed["street_name"] == "Rue Emile Zola"
