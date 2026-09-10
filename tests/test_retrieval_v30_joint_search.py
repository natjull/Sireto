import ast
import json
from pathlib import Path

import duckdb
import pytest
import tantivy

from scripts.build_hierarchical_retrieval_index import _schema
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend,RetrievalQuery,character_ngrams
from scripts.retrieval_v30_joint_search import search,terms,config_at


@pytest.fixture
def backend(tmp_path):
    (tmp_path/'tantivy').mkdir()
    index=tantivy.Index(_schema(tantivy),path=str(tmp_path/'tantivy'))
    w=index.writer(heap_size=30_000_000,num_threads=1)
    rows=[('00000000000001','ALPHA LOGISTICS','ZZZ','11111'),
          ('00000000000002','ZZZ','99 RUE BLEUE','11111'),
          ('00000000000003','ALPHX LOGISTIQ','1 AVENUE BLEUE','11111'),
          ('00000000000004','ALPHA LOGISTICS','99 RUE BLEUE','22222')]
    for siret,name,address,insee in rows:
        doc=tantivy.Document()
        for key,value in dict(document_type='siret',siret=siret,siren='999999999',number='1',insee=insee,postcode='11100',
              current_name_ngrams=' '.join(character_ngrams(name)),address_ngrams=' '.join(character_ngrams(address))).items():doc.add_text(key,value)
        w.add_document(doc)
    w.commit();w.wait_merging_threads()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(typed_name_portfolio=True)))
    return TantivyBackend(tmp_path)


def test_partial_joint_match_without_number_and_without_cross_site_proof(backend):
    q=RetrievalQuery('ALPHA LOGISTICS','99 RUE BLEUE','99','11100','11111')
    rows=search(backend,q)
    assert [r['siret'] for r in rows]==['00000000000003']
    assert rows[0]['siren']=='999999999'  # stored mapping, not a derived prefix
    assert rows[0]['evidence_rank']==rows[0]['candidate_rank']==1
    assert search(backend,RetrievalQuery(q.name,q.address,'',q.postcode,q.insee))==rows


def test_joint_query_can_recover_candidate_missed_by_separate_heads(backend):
    q=RetrievalQuery('ALPHA LOGISTICS','99 RUE BLEUE','99','11100','11111')
    found=[]
    for field,text in [('current_name_ngrams',q.name),('address_ngrams',q.address)]:
        p=backend._combined_query(q,backend._character_query(field,text),document_type='siret')
        _,loc=backend.searcher.search(p,limit=1).hits[0]
        found.append(backend._first(backend.searcher.doc(loc).to_dict(),'siret'))
    assert set(found)=={'00000000000001','00000000000002'}
    assert search(backend,q,1)[0]['siret']=='00000000000003'


def test_missing_fields_keep_query_but_return_empty_channel(backend):
    for q in [RetrievalQuery('', '99 RUE BLEUE',insee='11111'),RetrievalQuery('ALPHA','',insee='11111'),
              RetrievalQuery('ALPHA','99 RUE BLEUE'),RetrievalQuery('A','B',insee='11111')]:assert search(backend,q)==[]


def test_shared_ngram_markers_never_become_search_terms():
    assert not {'g3','g4','g5'} & set(terms('ALPHA'))
    assert all(all(c in '0123456789abcdef' for c in t) for t in terms('ALPHA'))


@pytest.mark.parametrize('key,value',[('channel_cap',513),('master_cap',2001),('workers',8),('folds',[0,2,3]),
                                     ('require_exact_number',True),('retrain',True),('national_run',True)])
def test_frozen_config(tmp_path,key,value):
    c=json.loads(Path('config/retrieval_v30_joint_search.json').read_text());c[key]=value
    p=tmp_path/'bad.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError):config_at(p)


def test_refuses_wider_channel(backend):
    with pytest.raises(ValueError):search(backend,RetrievalQuery('ALPHA','BLEUE'),513)


@pytest.mark.parametrize('filename',['scripts/retrieval_v30_joint_search.py','scripts/audit_retrieval_v30_results.py'])
def test_literal_sql_and_no_fit(filename):
    tree=ast.parse(Path(filename).read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
            assert n.func.attr!='fit'
            if n.func.attr=='execute' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str):duckdb.extract_statements(n.args[0].value)


def test_discovery_and_generation_never_open_truth_fields():
    text=Path('scripts/retrieval_v30_joint_search.py').read_text()
    for n in ast.parse(text).body:
        if isinstance(n,ast.FunctionDef) and n.name in ('discover','generate'):
            part=ast.get_source_segment(text,n)
            for forbidden in ('gt_siret','gt_siren','cohort_kind','qualification_v2',"cfg['baseline']",'outcomes.parquet'):
                assert forbidden not in part
