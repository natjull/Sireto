import pytest

from scripts.evaluate_retrieval_v76_history_parent import reference_visits


def test_reference_visits_count_duplicates_not_unique_global_sites():
    row=dict(primary_trace=dict(sources=dict(current=dict(decoded=100),historical_legal_exact=dict(decoded=3))))
    assert reference_visits(row)==103


@pytest.mark.parametrize('bad',[-1,1.5,'3'])
def test_invalid_diagnostic_count_refused(bad):
    with pytest.raises(ValueError,match='diagnostic'):
        reference_visits(dict(primary_trace=dict(sources=dict(current=dict(decoded=bad)))))
