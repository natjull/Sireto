import copy

import pytest

from scripts.audit_retrieval_v91_scoped_evidence import branch_reason,check_point,validate_population
from src.xgb_matcher.retrieval_v20_workbench import record


@pytest.mark.parametrize('outside,score,geo,xy,expected',[
    (True,0,True,True,'OUTSIDE_V84_ENDPOINT_SCOPE'),
    (False,0,False,False,'NO_MATCHING_WORD_PROOF_IN_SCOPED_SOURCE'),
    (False,.5,False,True,'QUERY_BAN_UNAVAILABLE_OR_INCOMPATIBLE'),
    (False,.5,True,False,'TARGET_XY_ABSENT'),
    (False,.5,True,True,'POSITIVE_SCOPED_WORD_SITE_PROOF'),
])
def test_branch_reasons_are_scoped_not_global_absence(outside,score,geo,xy,expected):
    assert branch_reason(dict(outside_graph_scope=outside,score=score),geo,xy)==expected


def test_point_lookup_detects_missing_above_boundary_and_wrong_emitted_metric():
    rows=[dict(siret=str(i),score=2.) for i in range(100)]
    assert check_point(rows,'999',2.)==(False,2.) # Global ties may exclude a target.
    assert check_point(rows,'999',1.)==(False,2.)
    assert check_point(rows,'0',2.)==(True,2.)
    with pytest.raises(ValueError,match='CERTIFICATE_POINT'): check_point(rows,'999',2.01)
    with pytest.raises(ValueError,match='EMITTED_POINT'): check_point(rows,'0',1.99)
    with pytest.raises(ValueError,match='Nonfinite'): check_point(rows,'999',float('nan'))
    with pytest.raises(ValueError,match='CERTIFICATE_POINT'): check_point(rows[:99],'999',.01)
    assert check_point([], '999',0.)==(False,0.)


def test_all128_prelabel_gate_rejects_duplicate_cap_order_fold_and_truth_drift(tmp_path,monkeypatch):
    from scripts.audit_retrieval_v91_scoped_evidence import v46
    # Population schema is separately tested by v46; spy confirms it is called.
    seen=[]
    monkeypatch.setattr(v46,'validate_queries',lambda q:seen.append(q))
    (tmp_path/'manifest.json').write_text('sealed')
    manifest=dict(status='SEALED_INITIAL_TOP100',queries=128,truth_opened=False)
    evaluation=dict(generation=record(tmp_path/'manifest.json'))
    queries=[dict(query_id=str(i),fold=2+i%3) for i in range(128)]
    rows=[dict(**q,primary_trace=dict(certified=True),results=dict(primary=[dict(siret='1',score=1.)])) for q in queries]
    validate_population(manifest,evaluation,tmp_path,queries,rows)
    assert seen==[queries]
    for key,value in [('truth_opened',True),('queries',127),('status','STOP')]:
        bad={**manifest,key:value}
        with pytest.raises(ValueError,match='blind generation'): validate_population(bad,evaluation,tmp_path,queries,rows)
    for mutation in ['duplicate','cap','order','fold','uncertified','duplicate_output','nonfinite','zero']:
        bad=copy.deepcopy(rows)
        if mutation=='duplicate': bad.append(copy.deepcopy(bad[-1]))
        elif mutation=='cap': bad[0]['results']['primary']=[dict(siret=str(i),score=1.) for i in range(101)]
        elif mutation=='order': bad.reverse()
        elif mutation=='fold': bad[0]['fold']=1
        elif mutation=='uncertified': bad[0]['primary_trace']['certified']=False
        elif mutation=='duplicate_output': bad[0]['results']['primary']*=2
        elif mutation=='nonfinite': bad[0]['results']['primary'][0]['score']=float('inf')
        else: bad[0]['results']['primary'][0]['score']=0.
        with pytest.raises(ValueError): validate_population(manifest,evaluation,tmp_path,queries,bad)
