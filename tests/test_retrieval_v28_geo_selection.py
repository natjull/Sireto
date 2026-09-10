import ast
import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pytest

from scripts.retrieval_v28_geo_selection import FEATURES,UL_FIELDS,config_at,training_sample,build_features
from src.xgb_matcher.retrieval_parent_fusion_v19 import assert_identifier_free_features


def test_identifier_and_person_fields_excluded():
    assert_identifier_free_features(FEATURES)
    assert not {'nomUniteLegale','nomUsageUniteLegale'} & set(UL_FIELDS)


def test_config_refuses_budget_drift(tmp_path):
    cfg=json.loads(Path('config/retrieval_v28_geo_selection.json').read_text());cfg['parent_cap']=1024
    p=tmp_path/'bad.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):config_at(p)


def test_sampler_only_uses_natural_positive_and_training_folds():
    rows=[]
    for query,fold,parents,truth in [('a',2,['11','12'],'11'),('b',2,['21'],'29'),('c',3,['31','32'],'31')]:
        for rank,parent in enumerate(parents):
            rows.append(dict(query_id=query,fold=fold,siren=parent,gt_siren=truth,legacy_parent_rank=rank+1,**{f:float(fold) for f in FEATURES}))
    X,y,g,q=training_sample(pd.DataFrame(rows),[2])
    assert q==['a'] and g.tolist()==[2] and y.tolist()==[1,0]
    assert (X==2).all()
    with pytest.raises(ValueError):training_sample(pd.DataFrame(rows),[1])


def test_joint_site_proof_does_not_migrate_between_siblings():
    c=duckdb.connect()
    c.execute("CREATE TABLE q AS SELECT 'q' query_id,2 fold,'ALPHA' crm_name,'1' crm_number")
    rows=[]
    for site,street,rank in [('a',0.,1),('b',1.,2)]:
        rows.append(dict(query_id='q',fold=2,siren='p',siret=site,candidate_street_similarity=street,
            candidate_number_exact=0,candidate_same_postcode=1,candidate_state='A',geo_distance_m=rank*10.,
            geo_neighbor_rank=rank,ban_street_similarity=1.,ban_certification_commune=True,ban_uncertainty_m=10.,ban_declared_weak=False))
    c.register('pool',pa.Table.from_pylist(rows))
    c.execute("CREATE TABLE names AS SELECT 'p' siren,'a' siret,'SIRET' subject_kind,'ALPHA' AS name")
    build_features(c)
    r=c.execute('SELECT site_name_jw,street,joint_name_street,number_exact FROM parent_features').fetchone()
    assert r==(1.,1.,0.,0)
    c.close()


def test_literal_sql_parses():
    tree=ast.parse(Path('scripts/retrieval_v28_geo_selection.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='execute' and node.args:
            arg=node.args[0]
            if isinstance(arg,ast.Constant) and isinstance(arg.value,str):duckdb.extract_statements(arg.value)
