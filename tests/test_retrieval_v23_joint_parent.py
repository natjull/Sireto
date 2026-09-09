from copy import deepcopy

import duckdb
import pandas as pd
import pyarrow as pa
import pytest

from src.xgb_matcher.retrieval_v23_joint_parent import FEATURES,NAME_SELECTION_SQL,build_features,make_master
from scripts.retrieval_v23_joint_parent import training_sample
from src.xgb_matcher.retrieval_parent_fusion_v19 import validate_fold_isolation


def fixture():
    return {
      'q':[dict(query_id='q',fold=2,crm_name='ALPHA',crm_street='RUE BLEUE',crm_number='1',crm_suffix='',crm_insee='11111',crm_postcode='11100')],
      'pool':[dict(query_id='q',fold=2,siren='100000001',siret=f'1000000010000{i}',base_rank=i,parent_rrf=.1,direct_rrf=.01) for i in (1,2)],
      'names':[dict(siren='100000001',siret='10000000100001',subject_kind='SIRET',name_role='SITE_CURRENT',name='ALPHA'),
               dict(siren='100000001',siret='10000000100002',subject_kind='SIRET',name_role='SITE_CURRENT',name='OMEGA')],
      'sites':[dict(siren='100000001',siret=f'1000000010000{i}',street_normalized=street,street_number='1',street_number_suffix='',insee='11111',postcode='11100',administrative_state='A') for i,street in [(1,'AUTRE VOIE'),(2,'RUE BLEUE')]]}


def db(rows):
    c=duckdb.connect()
    for name,values in rows.items():c.register(name,pa.Table.from_pylist(values))
    return c


def test_official_name_projection_normalizes_and_excludes_supporting_names():
    r=fixture();r.pop('names')
    r['name_source']=[dict(siren='100000001',siret='',subject_kind='SIREN',name_role=role,
                          normalized_value='École-Test',selected_for_subject=True) for role in ('LEGAL_CURRENT','SUPPORTING')]
    c=db(r);c.execute(NAME_SELECTION_SQL)
    assert c.execute('SELECT name_role,"name" FROM names').fetchall()==[('LEGAL_CURRENT','ECOLE TEST')]
    c.close()


def test_joint_proof_cannot_mix_site_name_and_sibling_address():
    r=fixture();r['sites'][0]['street_number']='99'
    c=db(r);build_features(c)
    assert c.execute('SELECT best_site_name_tokens,best_street,best_joint_tokens FROM parent_features').fetchone()==(1.,1.,0.)
    c.close()


def test_legal_name_can_support_each_own_site_and_exact_joint_wins():
    r=fixture();r['names'].append(dict(siren='100000001',siret='',subject_kind='SIREN',name_role='LEGAL_CURRENT',name='ALPHA'))
    c=db(r);build_features(c)
    assert c.execute('SELECT best_joint_tokens FROM parent_features').fetchone()==(1.,)
    assert c.execute("SELECT joint_site_rank FROM site_features WHERE siret='10000000100002'").fetchone()==(1,)
    c.close()


def test_name_missing_is_not_fake_perfect_match():
    r=fixture();r['q'][0]['crm_name']=''
    c=db(r);build_features(c)
    assert c.execute('SELECT best_joint_jw,best_joint_tokens,crm_name_missing FROM parent_features').fetchone()==(0.,0.,1)
    c.close()


def test_scores_do_not_use_identifiers_or_labels():
    a=fixture();b=deepcopy(a)
    for values in b.values():
        values.reverse()
        for r in values:
            r['siren']='900000001' if 'siren' in r else r.get('siren')
            if 'siret' in r:r['siret']=r['siret'].replace('100000001','900000001')
            if 'query_id' in r:r['query_id']='renamed'
            r.update(gt_siret='FORBIDDEN',data_origin='OTHER',label_kind='MATCH_EXACT')
    snapshots=[]
    for rows in (a,b):
        c=db(rows);build_features(c)
        snapshots.append(c.execute('SELECT '+','.join(FEATURES)+' FROM parent_features').fetchall());c.close()
    assert snapshots[0]==snapshots[1]


def test_oof_sample_uses_only_train_folds_and_natural_positives():
    rows=[]
    for f in (2,3,4):
        for i in range(800):
            rows.append(dict(query_id=f'q{f}',fold=f,siren=str(i),gt_siren='777',**{k:float(f) for k in FEATURES}))
    frame=pd.DataFrame(rows)
    X,y,g,ids=training_sample(frame,[2,3])
    assert set(X[:,0])=={2.,3.} and list(g)==[513,513] and y.sum()==2
    assert ids==['q2','q3']
    frame.loc[frame.fold.eq(3),'gt_siren']='NOT_PRESENT'
    X,y,g,ids=training_sample(frame,[2,3]);assert ids==['q2'] and y.sum()==1
    with pytest.raises(ValueError,match='Forbidden'):training_sample(frame,[1])


def test_shared_truth_parent_between_folds_fails():
    truth=pd.DataFrame([dict(query_id=f'q{f}',fold=f,gt_siren='same' if f in (2,3) else 'different') for f in (2,3,4)])
    with pytest.raises(ValueError,match='share'):validate_fold_isolation(truth)


def test_oof_master_cap_and_direct_witness_outside_geography():
    c=duckdb.connect()
    c.register('site_features',pa.Table.from_pylist([dict(query_id='q',fold=2,siren='s',siret=f'{i:014d}',
        direct_rrf=float(i),local_site=False,joint_site_rank=1) for i in range(2100)]))
    c.register('parent_scores',pa.Table.from_pylist([dict(query_id='q',fold=2,siren='s',parent_score=1.)]))
    make_master(c)
    assert c.execute('SELECT count(*),count(DISTINCT siret),max(candidate_rank),min(rescue_score) FROM master').fetchone()==(2000,2000,2000,100.)
    c.close()
