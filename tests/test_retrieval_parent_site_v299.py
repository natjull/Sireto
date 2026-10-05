from copy import deepcopy
import pytest
from scripts import retrieval_v299_parent_site_diagnostic as m

def fixture():
 controls=[dict(query_id=i,fold=2+i%3,position=i) for i in range(128)]
 labels=[dict(**{k:c[k] for k in ('query_id','fold')},human_ground_truth_siret='10000000000001',human_ground_truth_siren='100000000') for c in controls]
 outputs=[dict(**c,results=dict(primary=[dict(siret='10000000000001',siren='100000000')])) for c in controls]
 return outputs,controls,labels

def test_exact_parent_metrics_separate_no_cheating_site():
 outputs,controls,labels=fixture();outputs[0]['results']['primary']=[dict(siret='10000000000002',siren='100000000')]
 outputs[1]['results']['primary']=[dict(siret='20000000000001',siren='200000000')]
 r=m.score(outputs,controls,labels)
 assert r['exact_hits']==126 and r['parent_hits']==127 and r['queries']==128
 assert r['counts']==dict(HIT=126,PARENT_RETURNED_EXACT_SITE_MISSING=1,PARENT_MISSING=1)
 assert 'human_ground_truth_siret' not in str(r) and 'human_ground_truth_siren' not in str(r)

@pytest.mark.parametrize('bad',['partial','fold1','duplicate','wrongcontrol','cap101','parentconflict'])
def test_allhuman_control_failures_not_filtered(bad):
 outputs,controls,labels=fixture()
 if bad=='partial':labels.pop()
 elif bad=='fold1':controls[0]['fold']=1
 elif bad=='duplicate':labels[-1]=labels[0]
 elif bad=='wrongcontrol':outputs[0]['position']=1
 elif bad=='cap101':outputs[0]['results']['primary']*=101
 else:outputs[0]['results']['primary'][0]['siren']='200000000'
 with pytest.raises(ValueError):m.score(outputs,controls,labels)

def test_only_new_complete_evaluators_and_no_unsafe_old_folds():
 assert m.VERSIONS==('288','291','294','298')
 with pytest.raises(ValueError):m.module('0')

def test_stop_refused_before_any_target_parse(tmp_path,monkeypatch):
 monkeypatch.setattr(m,'OUTPUT',tmp_path);root=tmp_path/'stop';root.mkdir();(root/'status.json').touch()
 with pytest.raises(ValueError):m.load(root)
