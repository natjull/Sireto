import json
import ast
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from scripts.audit_retrieval_v26_evidence import geo_summary,names_summary,load,FIELDS


def test_literal_sql_parses_before_index_reads():
    tree=ast.parse(Path('scripts/audit_retrieval_v26_evidence.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='execute' and node.args:
            arg=node.args[0]
            if isinstance(arg,ast.Constant) and isinstance(arg.value,str):
                duckdb.extract_statements(arg.value)


def test_geo_counts_do_not_treat_missing_as_far_or_close_and_keep_weak_separate():
    df=pd.DataFrame(dict(exact_hit=[False]*3,operational_hit=[False]*3,
        ban_resolved=[True,True,False],ban_declared_weak=[False,True,True],
        nearest_parent_m=[20.,20.,np.nan],exact_distance_m=[400.,20.,np.nan],
        nearest_active_m=[20.,np.nan,np.nan],nearest_different_street_m=[20.,20.,np.nan],
        parent_site_count=[2,1,1],valid_coordinate_sites=[2,1,0],previous_geo_parent_present=[False]*3))
    r=geo_summary(df)
    assert r['queries']==3 and r['distance_available']==2
    assert r['nearest_parent_m_le_50']==2 and r['parent_le_50_ban_not_weak']==1
    assert r['exact_distance_m_le_50']==1 and r['other_site_strictly_closer']==1
    assert r['baseline_exact']==r['baseline_operational']==0  # proximity never relabels


def test_name_loss_is_not_a_candidate_hit_or_operational_hit():
    d=dict(exact_hit=[False,False],operational_hit=[False,False],unrepresented_names=[1,0])
    for scope in ('portfolio','global_docs','local_docs'):
        for k in ('name_count','exact_name','any_token','jw','tokens'):
            d[f'{scope}_{k}']=[1,1] if scope=='portfolio' else [0,1]
    r=names_summary(pd.DataFrame(d))
    assert r['exact_name_lost_global']==1 and r['exact_name_lost_local']==1
    assert r['token_overlap_lost_local']==1 and r['baseline_exact']==0


def test_supporting_is_not_authorized_by_name_audit():
    assert set(FIELDS)=={'LEGAL_CURRENT','TRADE_CURRENT','SITE_CURRENT','HISTORICAL'}


def test_config_refuses_persisting_coordinates_before_source_read(tmp_path):
    c=json.loads(open('config/retrieval_v26_evidence_audit.json').read());c['persist_raw_coordinates']=True
    p=tmp_path/'bad.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='drift'):load(p)
