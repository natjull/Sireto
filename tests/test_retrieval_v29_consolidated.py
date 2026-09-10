import ast
import json
from pathlib import Path

import duckdb
import pytest

from scripts.retrieval_v29_consolidated import config_at, replacement_union, equal_sql, check_q


@pytest.mark.parametrize('key,value', [('master_cap',2001), ('geo_parent_cap',513), ('folds',[1,2,3]),
                                     ('retrain',True), ('production_activation',True), ('national_run',True)])
def test_frozen_recipe(tmp_path, key, value):
    cfg=json.loads(Path('config/retrieval_v29_consolidated.json').read_text());cfg[key]=value
    path=tmp_path/'config.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):config_at(path)


def test_replacement_removes_geo_only_retains_independent_support():
    c=duckdb.connect()
    c.execute('''CREATE TABLE presence(query_id VARCHAR,fold INTEGER,siren VARCHAR,from_text BOOLEAN,from_address BOOLEAN,from_geo BOOLEAN)''')
    c.executemany('INSERT INTO presence VALUES (?,?,?,?,?,?)', [('q',2,'oldgeo',False,False,True),
        ('q',2,'textgeo',True,False,True),('q',2,'address',False,True,False)])
    c.execute("CREATE TABLE selected AS SELECT 'q' query_id,2 fold,'newgeo' siren,1 selected_rank")
    c.execute('''CREATE TABLE signals(query_id VARCHAR,fold INTEGER,siren VARCHAR,source_siret VARCHAR,channel VARCHAR,channel_rank INTEGER)''')
    c.executemany('INSERT INTO signals VALUES (?,?,?,?,?,?)', [('q',2,'oldgeo','','v17_union',1),
        ('q',2,'textgeo','','v17_union',2),('q',2,'address','','v17_union',3),
        ('q',2,'oldgeo','direct','siret_address_word',7)])
    replacement_union(c)
    assert c.execute("SELECT siren FROM signals WHERE channel='v17_union' ORDER BY siren").fetchall()==[('address',),('newgeo',),('textgeo',)]
    assert c.execute("SELECT siren,source_siret,channel_rank FROM signals WHERE channel<>'v17_union'").fetchall()==[('oldgeo','direct',7)]
    c.close()


def test_equality_detects_duplicates_not_just_set_difference():
    c=duckdb.connect()
    with pytest.raises(ValueError):equal_sql(c, 'SELECT 1 UNION ALL SELECT 1', 'SELECT 1')
    equal_sql(c, 'SELECT 1 UNION ALL SELECT 2', 'SELECT 2 UNION ALL SELECT 1');c.close()


@pytest.mark.parametrize('fold', [0,1,5])
def test_query_fold_lock(fold):
    c=duckdb.connect();c.execute("CREATE TABLE q AS SELECT 'q' query_id,? fold", [fold])
    with pytest.raises(ValueError):check_q(c,1)
    c.close()


@pytest.mark.parametrize('filename',['scripts/retrieval_v29_consolidated.py','scripts/audit_retrieval_v29_results.py'])
def test_literal_sql_parses_and_no_fit_call(filename):
    tree=ast.parse(Path(filename).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr != 'fit'
            if node.func.attr=='execute' and node.args:
                arg=node.args[0]
                if isinstance(arg,ast.Constant) and isinstance(arg.value,str):duckdb.extract_statements(arg.value)


def test_generator_does_not_read_truth_columns_or_baseline_outcomes():
    text=Path('scripts/retrieval_v29_consolidated.py').read_text()
    tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='generate')
    source=ast.get_source_segment(text,node)
    for forbidden in ('gt_siret','gt_siren','cohort_kind','qualification_v2','outcomes.parquet',"cfg['baseline']"):
        assert forbidden not in source


@pytest.mark.parametrize('tamper',[False,True])
def test_independent_metrics_keeps_catalog_absence_and_detects_false_gain(tamper):
    from scripts.audit_retrieval_v29_results import verify_metrics
    c=duckdb.connect()
    c.execute('''CREATE TABLE t AS SELECT 'q' query_id,'a' gt_siret,'p' gt_siren,true exact_hit,
      true operational_hit,true previous_exact_hit,true site_in_reservoir,true old_reservoir_hit,'HIT' loss_stage
      UNION ALL SELECT 'missing','b','r',false,false,false,false,false,'OFFICIAL_CATALOG_ABSENT' ''')
    c.execute("CREATE TABLE master AS SELECT 'q' query_id,'a' siret")
    for table in ('oldmaster','pool','oldpool'):c.execute(f'CREATE TABLE {table} AS SELECT * FROM master')
    c.execute("CREATE TABLE parents AS SELECT 'q' query_id,'p' siren")
    c.execute('CREATE TABLE acceptable(query_id VARCHAR,siret VARCHAR)')
    c.execute("CREATE TABLE catalog AS SELECT 'q' query_id,true official_catalog_present UNION ALL SELECT 'missing',false")
    primary=dict(queries=2,exact_hits=2 if tamper else 1,operational_hits=1,gains_vs_v23=0,losses_vs_v23=0)
    em=dict(primary=primary,historical_all_human=primary,segments={})
    if tamper:
        with pytest.raises(AssertionError):verify_metrics(c,em)
    else:verify_metrics(c,em)
    c.close()
