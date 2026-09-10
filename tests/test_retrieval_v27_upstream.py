import ast
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from scripts.audit_retrieval_v27_upstream import geo_reason,load,UL_FIELDS,SITE_FIELDS


def test_exhaustive_geo_attribution_and_unknown_not_far():
    assert geo_reason(False,0,0,np.nan,np.nan)=='BAN_UNRESOLVED'
    assert geo_reason(True,0,0,np.nan,np.nan)=='NO_VALID_PARENT_COORDINATE'
    assert geo_reason(True,3,0,np.nan,np.nan)=='COMMUNE_BOUNDARY'
    assert geo_reason(True,3,1,np.nan,np.nan)=='SITE_CAP_4096'
    assert geo_reason(True,3,1,600,513)=='PARENT_CAP_512'
    assert geo_reason(True,3,1,600,512)=='PRESENT'
    with pytest.raises(ValueError): geo_reason(True,3,1,600,np.nan)


def test_names_allowlist_excludes_person_names():
    assert len(UL_FIELDS)==5 and len(SITE_FIELDS)==4
    assert not {'nomUniteLegale','nomUsageUniteLegale','prenom1UniteLegale'} & set(UL_FIELDS)


def test_arrow_registration_handles_pandas_string_dtype():
    c=duckdb.connect();d=pd.DataFrame({'geo_reason':['PRESENT','COMMUNE_BOUNDARY']})
    c.register('final',pa.Table.from_pandas(d,preserve_index=False))
    assert c.execute('SELECT count(*) FROM final').fetchone()[0]==2
    c.close()


def test_config_refuses_new_candidates(tmp_path):
    cfg=json.loads(Path('config/retrieval_v27_upstream_audit.json').read_text());cfg['new_candidates']=True
    p=tmp_path/'bad.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):load(p)


def test_literal_sql_parses():
    tree=ast.parse(Path('scripts/audit_retrieval_v27_upstream.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='execute' and node.args:
            arg=node.args[0]
            if isinstance(arg,ast.Constant) and isinstance(arg.value,str):duckdb.extract_statements(arg.value)
