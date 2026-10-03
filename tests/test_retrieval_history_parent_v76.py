import duckdb
import numpy as np
import pytest
import tantivy

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_history_parent_v76 import alias_pairs, matching_parents, parent_query


def backend(tmp_path, rows):
    # A minimal main-index-compatible schema for scalar proof parity.
    b=tantivy.SchemaBuilder()
    for f in ('siret','siren','document_type','insee','postcode','number','is_siege',
              'legal_names_exact','trade_names_exact','site_names_exact','addresses_exact'):
        b.add_text_field(f,stored=True,tokenizer_name='raw',index_option='basic')
    for f in ('legal_names','trade_names','site_names','historical_names','supporting_names',
              'addresses','historical_addresses','supporting_addresses'):
        b.add_text_field(f,stored=True,index_option='basic')
    for f in ('current_name_ngrams','address_ngrams'):
        b.add_text_field(f,tokenizer_name='raw',index_option='basic')
    i=tantivy.Index(b.build(),path=str(tmp_path)); w=i.writer(heap_size=15000000,num_threads=1)
    for r in rows:
        d=tantivy.Document()
        for f,value in r.items():
            for v in value if isinstance(value,list) else [value]:
                d.add_text(f,v)
        w.add_document(d)
    w.commit(); w.wait_merging_threads(); i.reload()
    class Backend:
        index=i; searcher=i.searcher(); _tantivy=tantivy
    return Backend()


def site(j,**updates):
    parent=f'{123000000+j:09d}'
    return dict(siret=parent+'00001',siren=parent,document_type='siret',
                addresses='10 RUE BETA',addresses_exact='10 RUE BETA',number='10',
                insee='75056',postcode='75001',is_siege='1',**updates)


def test_alias_predicate_exhaustive_no_parent_limit_geography_or_label():
    c=duckdb.connect(); c.execute('CREATE TABLE aliases(name VARCHAR,siren VARCHAR)')
    c.executemany('INSERT INTO aliases VALUES (?,?)',[('ALPHA',f'{123000000+j:09d}') for j in range(250)])
    assert len(matching_parents(c,'Alpha 12345678900001'))==250
    assert matching_parents(c,'12345678900001')==[]
    assert matching_parents(c,'missing')==[]
    c.close()


def test_alias_normalization_and_membership_preserves_official_links():
    assert alias_pairs([dict(siren='123456789',normalized_name='Alpha'),
                        dict(siren='123456789',normalized_name='ALPHA 12345678900001')])==[('ALPHA','123456789')]
    with pytest.raises(ValueError,match='membership'):
        alias_pairs([dict(siren='bad',normalized_name='ALPHA')])


def test_all_sites_joined_not_filtered_by_geo_state_or_current_name(tmp_path):
    a=site(0); b=site(1); b['insee']='99999'; b['postcode']='99999'
    c=site(2); b['document_type']='siren'
    base=backend(tmp_path,[a,b,c]); weights=[0.]*11; weights[0]=2.; weights[2]=1.
    q=RetrievalQuery(name='ANCIEN NOM',insee='99999')
    hits=base.searcher.search(parent_query(base,q,[a['siren'],b['siren']],weights),limit=100).hits
    assert len(hits)==1 and hits[0][0]==3.
    assert base.searcher.doc(hits[0][1]).to_dict()['siret']==[a['siret']]


def test_native_conditional_score_matches_independent_scalar_formula(tmp_path):
    records=[]; features=[]
    for j in range(8):
        r=site(j); a=(j&1)!=0; p=(j&2)!=0; siege=(j&4)!=0
        if not a:
            r['addresses']='UNRELATED'; r['addresses_exact']='UNRELATED'
        if not p:
            r['insee']='99999'; r['postcode']='99999'
        r['is_siege']='1' if siege else '0'; records.append(r)
        features.append([1.,float(a),1.,float(a),float(p),float(p),float(p),
                         float(a and p),2.*float(a and p),float(siege and p),float(a and p)])
    base=backend(tmp_path,records); w=np.arange(1.,12.)/10
    q=RetrievalQuery(name='ANCIEN NOM',address='10 RUE BETA',number='10',insee='75056',postcode='75001')
    hits=base.searcher.search(parent_query(base,q,[r['siren'] for r in records],w),limit=100).hits
    actual={base.searcher.doc(loc).to_dict()['siret'][0]:score for score,loc in hits}
    for r,f in zip(records,features):
        assert actual[r['siret']]==pytest.approx(float(np.dot(f,w)),abs=2e-6)


def test_more_than100_parents_native_top100_no_fixed_parent_pool(tmp_path):
    records=[site(j) for j in range(150)]
    for j,r in enumerate(records):
        if j<50:
            r['insee']='99999'; r['postcode']='99999'
    base=backend(tmp_path,records); w=[0.]*11; w[0]=1.; w[4]=2.
    hits=base.searcher.search(parent_query(base,RetrievalQuery(name='ALPHA',insee='75056'),
        [r['siren'] for r in records],w),limit=100).hits
    assert len(hits)==100 and all(score==3. for score,_ in hits)
    assert len({base.searcher.doc(loc).to_dict()['siret'][0] for _,loc in hits})==100


def test_no_address_match_no_fictitious_interaction(tmp_path):
    r=site(0); base=backend(tmp_path,[r]); w=[0.]*11; w[8]=1.
    assert not base.searcher.search(parent_query(base,RetrievalQuery(name='ALPHA',insee='75056',address='OTHER'),
        [r['siren']],w),limit=100).hits


@pytest.mark.parametrize('w',[[0.]*10,[-1.]+[0.]*10,[float('nan')]+[0.]*10])
def test_bad_weights_refused(tmp_path,w):
    base=backend(tmp_path,[site(0)])
    with pytest.raises(ValueError,match='metric'):
        parent_query(base,RetrievalQuery(name='ALPHA'),['123000000'],w)
