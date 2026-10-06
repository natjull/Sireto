import random
import re
from types import SimpleNamespace

import numpy as np
import pytest
import tantivy

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_compound_initialisms_v222 import partitions
from src.xgb_matcher.retrieval_nominal_abbreviation_v256 import patterns, component
from src.xgb_matcher.retrieval_proof_v56 import components
from src.xgb_matcher.retrieval_regex_compiler_v353 import (
    Backend, SIZE_ERROR, equivalent_pattern, suffix_factor, search)
from src.xgb_matcher.retrieval_regional_initial_v286 import search as old_search
from tests.test_retrieval_corroborated_initial_v217 import tiny_native


def schema():
    b = tantivy.SchemaBuilder()
    b.add_text_field('names', tokenizer_name='raw')
    return b.build()


@pytest.mark.parametrize('token', ['SESSAD', 'SAASAS', 'ABCDEF', 'SSSSSS', 'AAAAAA', 'EEEEEE'])
def test_all_31_partitions_compile_without_discarding(token):
    b = Backend(SimpleNamespace(_tantivy=tantivy))
    ps = patterns(token)
    assert len(ps) == 31
    for p in ps:
        b.queries.regex_query(schema(), 'names', p)
    if token in ('SESSAD', 'SAASAS'):
        assert b.queries.repairs


def test_reproduces_original_failure_and_repairs_only_that_pattern():
    ps = patterns('SESSAD')
    with pytest.raises(ValueError, match='1000 states'):
        tantivy.Query.regex_query(schema(), 'names', ps[-1])
    b = Backend(SimpleNamespace(_tantivy=tantivy))
    for p in ps:
        b.queries.regex_query(schema(), 'names', p)
    assert len(b.queries.repairs) == 1
    assert b.queries.repairs[0]['original'] == ps[-1]


def test_suffix_trie_exact_literal_union_including_empty_prefixes():
    words = ['A ', 'AU ', 'AUX ', 'SA ', 'SAS ', 'SASU ', 'SARL ', 'SOCIETE ', 'STE ']
    factored = suffix_factor(words)
    original = '(' + '|'.join(words) + ')'
    for word in words + ['', 'SAS', 'S AS ', 'SASU  ', 'SOCIETES ', 'ETS ']:
        assert bool(re.fullmatch(original, word)) == bool(re.fullmatch(factored, word))
    with pytest.raises(ValueError):
        suffix_factor(['.*'])


def test_full_language_oracle_all_partitions_legal_particles_and_poison():
    count = 0
    for token in ('SESSAD', 'SAASAS', 'ABCDEF'):
        for p, chunks in zip(patterns(token), partitions(token)):
            repaired = equivalent_pattern(p)
            assert repaired != p
            for suffix in ('', 'TOKEN', '3'):
                words = [chunk + suffix for chunk in chunks]
                for between in (' ', ' DE ', ' AUX DE ET '):
                    body = between.join(words)
                    for prefix in ('', 'SAS ', 'SA SAS SOCIETE DE '):
                        for tail in ('', ' SARL', ' SAS SARL'):
                            alias = prefix + body + tail
                            for variant in (alias, alias + ' EXTRA', 'OTHER ' + alias,
                                            alias + ' ', alias.lower(), alias.replace(' ', '  ')):
                                assert bool(re.fullmatch(p, variant)) == bool(re.fullmatch(repaired, variant))
                                count += 1
    assert count > 45000


def test_no_rewrite_when_native_compiles_and_other_errors_remain_fatal():
    calls = []
    sentinel = object()
    native = SimpleNamespace(regex_query=lambda s, f, p: calls.append((s, f, p)) or sentinel)
    b = Backend(SimpleNamespace(_tantivy=SimpleNamespace(Query=native)))
    assert b.queries.regex_query('schema', 'names', '(A|SA)*') is sentinel
    assert calls == [('schema', 'names', '(A|SA)*')] and not b.queries.repairs

    def fails(*args):
        raise ValueError('unknown error')
    b = Backend(SimpleNamespace(_tantivy=SimpleNamespace(Query=SimpleNamespace(regex_query=fails))))
    with pytest.raises(ValueError, match='unknown error'):
        b.queries.regex_query(None, 'names', '(A|SA)*')

    def unrepairable(*args):
        raise ValueError(SIZE_ERROR)
    b = Backend(SimpleNamespace(_tantivy=SimpleNamespace(Query=SimpleNamespace(regex_query=unrepairable))))
    with pytest.raises(ValueError, match='1000 states'):
        b.queries.regex_query(None, 'names', '(A|SA)*')
    assert not b.queries.repairs


def test_native_match_and_constant_score_no_alias_borrowing():
    b = tiny_native()
    aliases = [
        ['SERVICE EDUCATION SOINS SPECIALISES ACCOMPAGNEMENT DOMICILE'],
        ['SAS SERVICE DE EDUCATION SOINS AUX SPECIALISES ACCOMPAGNEMENT DOMICILE SARL'],
        ['SERVICE EDUCATION SOINS', 'SPECIALISES ACCOMPAGNEMENT DOMICILE'],
        ['SERVICE EDUCATION SOINS SPECIALISES ACCOMPAGNEMENT DOMICILE EXTRA'],
        ['SERVICE EDUCATION SOINS SPECIALISES ACCOMPAGNEMENT DOMICILE'],
    ]
    w = b.index.writer()
    for i, names in enumerate(aliases, 10):
        w.add_document(tantivy.Document.from_dict(dict(document_type='siret',
            siret=f'100000000{i:05d}', siren='100000000', legal_names_exact=names,
            insee='22222' if i == 14 else '11111', postcode='11111'), b.index.schema))
    w.commit(); b.index.reload(); b.searcher = b.index.searcher()
    fixed = Backend(b)
    q = RetrievalQuery(name='SESSAD', insee='11111')
    hits = b.searcher.search(component(fixed, q, components(fixed, q)), limit=100, count=False).hits
    assert {b._first(b.searcher.doc(loc).to_dict(), 'siret') for _, loc in hits} == {
        '10000000000010', '10000000000011'}
    assert all(score == 1. for score, _ in hits)
    assert len(fixed.queries.repairs) == 3  # Same complete predicate in all three exact fields.


def evidence():
    return SimpleNamespace(files=SimpleNamespace(guard=lambda: None),
        name_owners=lambda name: (set(), set(), {}), related=lambda p, s: (set(), {}),
        radial_proofs=lambda g: ([], {}))


def test_historical_successes_same_native_scores_order_and_one_collector():
    b = tiny_native()
    for name in ('ALPHA', 'COMMUNE DE ALPHA ECOLE BETA', 'ABCDEF', 'AZB'):
        q = RetrievalQuery(name=name, insee='11111', address='ROAD')
        args = (evidence(), q, {}, np.ones(11), [], 0., [])
        before, old_trace = old_search(b, *args)
        after, trace = search(b, *args)
        assert after == before
        assert trace['regex_patterns_repaired'] == 0
        assert trace['regex_patterns_discarded'] == 0
        assert trace['initial_cap'] == old_trace['initial_cap'] == 100
    real = b.searcher
    calls = []
    def called(parsed, **kwargs):
        calls.append(kwargs)
        return real.search(parsed, **kwargs)
    b.searcher = SimpleNamespace(search=called, doc=real.doc)
    search(b, evidence(), RetrievalQuery(name='SESSAD', insee='11111'), {}, np.ones(11), [], 0., [])
    assert calls == [dict(limit=100, count=False)]


def test_factoring_operator_and_regional_qualifier_semantics():
    p = '(SA |SAS )*(ALPHA REGION|BETA REGION|GAMMA REGION) S[A-Z0-9]*( (SA|SAS))*'
    fixed = equivalent_pattern(p)
    rng = random.Random(353)
    for _ in range(1000):
        prefix = ''.join(rng.choices(['SA ', 'SAS '], k=rng.randrange(4)))
        value = prefix + rng.choice(['ALPHA REGION', 'BETA REGION', 'GAMMA REGION', 'DELTA REGION'])
        value += rng.choice([' SERVICE', ' SERVICE SA', ' SERVICE SAS SA', ' SERVICE EXTRA'])
        assert bool(re.fullmatch(p, value)) == bool(re.fullmatch(fixed, value))
