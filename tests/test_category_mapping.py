import logging

import pytest

from pipe_v6.category_mapping import map_legal_to_category


def test_public_codes():
    assert map_legal_to_category("7111") == "PUBLIC"
    assert map_legal_to_category("7210") == "PUBLIC"
    assert map_legal_to_category("5532") == "PUBLIC"


def test_private_codes():
    assert map_legal_to_category("5710") == "PRIVE"
    assert map_legal_to_category("9220") == "PRIVE"
    assert map_legal_to_category("9230") == "PRIVE"


def test_none_returns_inconnu():
    assert map_legal_to_category(None) == "INCONNU"


def test_empty_returns_inconnu():
    assert map_legal_to_category("") == "INCONNU"


def test_unknown_code_returns_inconnu_and_logs_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert map_legal_to_category("9999", logger=logging.getLogger(__name__)) == "INCONNU"
    assert any("9999" in record.message for record in caplog.records)
