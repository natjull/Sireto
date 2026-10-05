"""Selection/geometry fixtures only, no actual CRM/targets/search opened."""
from copy import deepcopy
import inspect

import pytest

from scripts import retrieval_v322_large_panel as work
from scripts import retrieval_v323_corroborated_population as generation
from src.xgb_matcher import retrieval_evaluation_authority_v325 as authority
from scripts import evaluate_retrieval_v326_postseal as evaluation
from test_retrieval_large_population_v323 import queries


def all_queries():
    return [dict(query_id='Q'+str(i),fold=2+i%3,crm_name='ACME',crm_number='1',crm_suffix='',
                 crm_street='ROAD',crm_postcode='75001',crm_insee='75056') for i in range(23571)]


def test_selection_all2000_label_blind_disjoint_and_input_order_independent():
    rows=all_queries(); pilot=rows[:128]; train=[q['query_id'] for q in rows[128:640]]
    selected=work.choose(rows,pilot,train)
    assert len(selected)==2000 and len({q['query_id'] for q in selected})==2000
    assert not {q['query_id'] for q in selected} & {q['query_id'] for q in rows[:640]}
    assert work.choose(list(reversed(rows)),list(reversed(pilot)),list(reversed(train)))==selected
    changed=deepcopy(rows)
    for q in changed:
        q['crm_name']='';q['crm_street']='';q['crm_postcode']='';q['crm_insee']=''
    assert [q['query_id'] for q in work.choose(changed,changed[:128],train)]==[q['query_id'] for q in selected]
    assert rows==all_queries()


@pytest.mark.parametrize('bad',['short','duplicate','fold0','fold1','label','train_short','train_overlap','train_foreign','other_size'])
def test_no_filtered_population_or_closed_fold_or_train_overlap(bad):
    rows=all_queries();pilot=deepcopy(rows[:128]);train=[q['query_id'] for q in rows[128:640]];size=2000
    if bad=='short':rows.pop()
    elif bad=='duplicate':rows[-1]=deepcopy(rows[0])
    elif bad=='fold0':rows[-1]['fold']=0
    elif bad=='fold1':rows[-1]['fold']=1
    elif bad=='label':rows[-1]['human_ground_truth_siret']='10000000000001'
    elif bad=='train_short':train.pop()
    elif bad=='train_overlap':train[0]=pilot[0]['query_id']
    elif bad=='train_foreign':train[0]='OUTSIDE'
    else:size=1999
    with pytest.raises(ValueError):work.choose(rows,pilot,train,size)


def test_geometry_only_adds_same_compatibility_flag_no_label_site_repair():
    q=queries()[0];g=dict(query_id=q['query_id'],fold=q['fold'],ban_resolved=True,ban_x_l93=700000.,ban_y_l93=6600000.)
    v=work.geometry(q,g);assert v==dict(g,spatial_crs_compatible=True) and 'spatial_crs_compatible' not in g
    q['crm_insee']='97101';assert not work.geometry(q,g)['spatial_crs_compatible']
    g['query_id']='OTHER'
    with pytest.raises(ValueError):work.geometry(q,g)


def test_frozen_native_business_recipe_is_exactly_the_retained_engine():
    from scripts import retrieval_v287_corroborated_population as old
    assert generation.search is old.search and generation.Evidence is old.Evidence and generation.Regions is old.Regions
    assert generation.checked_update.__code__.co_code==old.checked_update.__code__.co_code
    assert generation.config()['resources']==old.config()['resources']
    assert generation.config()['query_seconds']==old.config()['query_seconds']
    assert evaluation.config()['generation_config']['path']==str(generation.CONFIG)
    assert 'human_ground_truth' not in inspect.getsource(work)
    assert 'read_table' not in inspect.getsource(generation)


@pytest.mark.parametrize('kind',['missing','stop'])
def test_no_target_lookup_before_complete_generation(tmp_path,monkeypatch,kind):
    monkeypatch.setattr(generation,'config',lambda:dict(output_root=str(tmp_path)))
    def forbidden():raise AssertionError('Official/target lookup before actual complete2000 seal')
    monkeypatch.setattr(authority,'source_scope',forbidden)
    if kind=='stop':(tmp_path/'status.json').write_text('{"status":"STOP"}')
    with pytest.raises((ValueError,FileNotFoundError)):authority.authorize(tmp_path)


def test_target_arrow_filters_both_fold_and_selected_id(monkeypatch):
    controls=[dict(position=i,query_id=q['query_id'],fold=q['fold']) for i,q in enumerate(queries())]
    monkeypatch.setattr(evaluation,'checked',lambda ref:'SIMULATED')
    monkeypatch.setattr(evaluation.pq,'read_schema',lambda p:type('S',(),{'names':['query_id','fold','x']})())
    captured=[]
    def read_table(p,**kw):
        captured.append(kw);return type('T',(),{'to_pylist':lambda self:[]})()
    monkeypatch.setattr(evaluation.pq,'read_table',read_table)
    assert evaluation.selected({},controls,('query_id','fold','x'))==[]
    assert captured[0]['filters']==[('fold','in',[2,3,4]),('query_id','in',[q['query_id'] for q in controls])]
    assert captured[0]['use_threads'] is False
    controls[-1]['fold']=1
    with pytest.raises(ValueError):evaluation.selected({},controls,('query_id',))
    assert len(captured)==1
