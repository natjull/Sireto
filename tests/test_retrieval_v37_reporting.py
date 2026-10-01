import pytest

from scripts.audit_retrieval_v37_reporting import SEGMENTS, segmented


def fixture():
    label = {c: '' for c in SEGMENTS}
    label.update(fold=2, unseen_siren=False, new_site_known_siren=False, mega_base_pool=None)
    row = dict(query_id='q', fold=2, exact_rank=101, operational_rank=99,
               baseline_exact_rank=None, label_kind='', qualification_v2='', qualification_v3='')
    return row, {'q': label}


def test_unknown_is_published_without_filtering_or_conflating_false():
    row, labels = fixture()
    result = segmented([row], labels)
    assert result['qualification_v2']['UNKNOWN']['queries'] == 1
    assert result['mega_base_pool']['UNKNOWN']['100']['exact_hits'] == 0
    assert result['mega_base_pool']['UNKNOWN']['100']['operational_hits'] == 1
    assert result['unseen_siren']['False']['queries'] == 1


def test_absent_truth_stays_in_denominator():
    row, labels = fixture()
    row.update(exact_rank=None, operational_rank=None)
    assert segmented([row], labels)['fold']['2']['queries'] == 1


def test_duplicates_and_metadata_drift_fail_closed():
    row, labels = fixture()
    with pytest.raises(ValueError, match='population'):
        segmented([row, row], labels)
    row['qualification_v3'] = 'invented'
    with pytest.raises(ValueError, match='metadata'):
        segmented([row], labels)


@pytest.mark.parametrize('fold', [0, 1, 5])
def test_forbidden_folds_fail_closed(fold):
    row, labels = fixture()
    row['fold'] = labels['q']['fold'] = fold
    with pytest.raises(ValueError, match='fold'):
        segmented([row], labels)
