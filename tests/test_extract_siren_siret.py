import pytest

from pipe_v6.external_sources import extract_siren_siret_from_url


def test_extract_siret_14_digits():
    url = "https://www.pappers.fr/entreprise/acme-12345678900012"
    siren, siret = extract_siren_siret_from_url(url)
    assert siren == "123456789"
    assert siret == "12345678900012"


def test_extract_siren_9_digits():
    url = "https://www.societe.com/societe/acme-123456789.html"
    siren, siret = extract_siren_siret_from_url(url)
    assert siren == "123456789"
    assert siret is None


def test_extract_multiple_patterns_keeps_last():
    url = "https://example.com/old/111111111/new/999999999"
    siren, siret = extract_siren_siret_from_url(url)
    assert siren == "999999999"
    assert siret is None


def test_extract_no_match():
    url = "https://example.com/no-digits-here"
    siren, siret = extract_siren_siret_from_url(url)
    assert siren is None
    assert siret is None


def test_extract_from_query_string():
    url = "https://example.com/search?siren=123456789"
    siren, siret = extract_siren_siret_from_url(url)
    assert siren == "123456789"
    assert siret is None

