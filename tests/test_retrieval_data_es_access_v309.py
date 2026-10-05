"""Source access fixtures only; not a retrieval/recall/actual producer test."""
from types import SimpleNamespace
from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_data_es_access_v309 import Facilities
from scripts import retrieval_v308_data_es_source as source
from test_retrieval_data_es_source_v308 import write_source


def access(tmp_path, rows):
    csvpath, db = tmp_path/'source.csv', tmp_path/'source.sqlite'
    write_source(csvpath, rows)
    stats = source.compile_db(db, csvpath)
    f = Facilities.__new__(Facilities)
    f.c, f.stats = source.readonly(db), stats
    f.files = SimpleNamespace(guard=lambda: None, check_epochs=lambda: None)
    return f


def test_exact_whole_alias_fragment_all_localities_and_missing_ids(tmp_path):
    f = access(tmp_path, [
        {'SIRET Installation': '10000000100001', "Nom de l'équipement sportif": 'ALPHA BETA', 'Commune INSEE': '75056'},
        {'SIRET Installation': '10000000200001', "Nom de l'équipement sportif": 'ALPHA BETA', 'Commune INSEE': '99999'},
        {'SIRET Installation': '10000000300001', "Nom de l'équipement sportif": 'ALPHA BETA'},
        {"Nom de l'équipement sportif": 'ALPHA BETA', 'Nom du propriétaire': 'COMMUNE ALPHA'},
        {'SIRET Installation': '10000000400001', "Nom de l'équipement sportif": 'ALPHA GAMMA BETA', 'Commune INSEE': '75056'},
    ])
    groups, t = f.facts(RetrievalQuery(name='PREFIX ALPHA BETA', insee='75056'))
    assert groups == {'local': {'10000000100001'}, 'nonlocal': {'10000000200001'}, 'unknown': {'10000000300001'}}
    assert t['source_records_missing_explicit_siret'] == 1 and t['distinct_explicit_endpoints'] == 3
    assert not t['candidates_emitted'] and not t['owner_inferred_from_municipality']
    assert not t['official_physical_site_identity_certified']
    f.close()


def test_all_owner_facts_without_top100_cap_or_rank(tmp_path):
    f = access(tmp_path, [{'SIRET Installation': f'{100000000+i:09d}00001',
        "Nom de l'installation sportive": 'ALPHA BETA', 'Code Postal': '11111'} for i in range(151)])
    groups, t = f.facts(RetrievalQuery(name='ALPHA BETA', postcode='11111'))
    assert len(groups['local']) == 151 and t['source_fact_or_site_cap'] is None
    assert not t['candidate_score_computed']
    f.close()


def test_source_postal_or_insee_membership_and_no_numeric_similarity(tmp_path):
    f = access(tmp_path, [
        {'SIRET Installation': '10000000100001', "Nom de l'équipement sportif": 'ALPHA BETA', 'Code Postal': '11111', 'Commune INSEE': '22222'},
        {'SIRET Installation': '10000000200001', "Nom de l'équipement sportif": 'ALPHA BETA', 'Code Postal': '11112'},
    ])
    groups, _ = f.facts(RetrievalQuery(name='ALPHA BETA', postcode='11111', insee='22223'))
    assert groups['local'] == {'10000000100001'} and groups['nonlocal'] == {'10000000200001'}
    missing, _ = f.facts(RetrievalQuery(name='ALPHA BETA'))
    assert missing['unknown'] == {'10000000100001', '10000000200001'} and not missing['nonlocal']
    assert f.facts(RetrievalQuery(name='ALPHA', postcode='11111'))[0] == {'local': set(), 'unknown': set(), 'nonlocal': set()}
    f.close()
