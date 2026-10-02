import itertools
import numpy as np
import pytest

from src.xgb_matcher.retrieval_impact_v68 import ImpactIndex, official_initialisms, document_impacts, query_impacts
from src.xgb_matcher.retrieval_initialisms_v67 import membership


@pytest.mark.parametrize('alias', [
    'ALPHA BETA', 'SAS ALPHA DE BETA', 'SOCIETE DES ALPHA ET BETA GAMMA',
    'ALPHA EN BETA', 'ALPHA 123456789 BETA', 'SARL LE DELTA', 'A D A D A D B',
    'ALPHA BETA GAMMA DELTA EPSILON ZETA ETA', '',
])
def test_index_time_initialisms_match_single_alias_regex(alias):
    codes = official_initialisms(alias)
    alphabet = sorted({t[0] for t in alias.split() if t and t[0].isalpha()})
    for length in range(2, 5):
        for letters in itertools.product(alphabet, repeat=length):
            code = ''.join(letters)
            assert (code in codes) == bool(membership(code, [alias]))
    assert all(membership(code, [alias]) for code in codes)


def test_never_pool_initials_across_aliases_or_truncate_complexity():
    features = document_impacts({'legal_names': ['ALPHA', 'BETA']})
    assert 'I:AB' not in features
    with pytest.raises(ValueError, match='NOT_TRUNCATION'):
        official_initialisms('ALPHA BETA', max_variants=1)


@pytest.mark.parametrize('block_size', [1, 7, 64, 256])
def test_exact_top100_including_ties_and_late_blocks(block_size):
    rng = np.random.default_rng(68)
    documents = {f'{i:05}': {f't{j}': float(rng.integers(1, 10)) / 7
                 for j in rng.choice(31, size=7, replace=False)} for i in range(1103)}
    index = ImpactIndex.build(documents, block_size)
    for _ in range(15):
        query = {f't{j}': float(rng.integers(1, 10)) / 11 for j in rng.choice(34, size=5, replace=False)}
        result, trace = index.search(query)
        assert result == index.exhaustive(query)
        assert trace['heap_capacity'] == 100 and trace['fixed_candidate_reservoir'] is None
    ties = ImpactIndex.build({f'{i:05}': {'x': 1.} for i in range(333)}, block_size)
    assert ties.search({'x': 1.})[0] == ties.exhaustive({'x': 1.})


def test_skips_provably_inferior_blocks_without_changing_top100():
    documents = {f'{i:05}': {'x': 10. if i >= 900 else 1.} for i in range(1000)}
    index = ImpactIndex.build(documents, 100)
    hits, trace = index.search({'x': 1.})
    assert hits == index.exhaustive({'x': 1.})
    assert trace['scored_documents'] == 100 and trace['visited_blocks'] == 1


def test_query_identifiers_not_encoded_and_absent_words_do_not_remove_denominator():
    row = dict(crm_name='AB027 123456789 12345678900001', crm_insee='75056', crm_postcode='75001')
    terms = query_impacts(row)
    assert 'I:AB' in terms and not any('123456789' in t for t in terms)
    index = ImpactIndex.build({'one': {'x': 1.}})
    assert index.search({'missing': 1.})[0] == []


def test_empty_index_zero_impacts_and_safety_contract():
    index = ImpactIndex.build({'empty': {'x': 0.}})
    assert index.search({})[0] == index.exhaustive({}) == []
    assert ImpactIndex.build({}).search({'x': 1.})[0] == []
    with pytest.raises(ValueError): index.search({'x': -1.})
    with pytest.raises(ValueError): index.search({'x': np.nan})
    with pytest.raises(ValueError): index.search({'x': 1.}, 101)
    with pytest.raises(ValueError): ImpactIndex.build({'one': {'x': -1.}})
