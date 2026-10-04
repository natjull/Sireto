"""Synthetic training metadata fixture ONLY, not national evidence."""
from copy import deepcopy
import pytest
from scripts.audit_retrieval_v226_training_geography import summarize


def fixture():
    registry={str(f):[f'{f:09d}'] for f in (2,3,4)}
    def vector(p): return [0.,0.,0.,0.,float(p)]+[0.]*14
    scenes=[dict(query_id=f'TRAIN{i}',fold=2+i%3,positive_siren=f'{2+i%3:09d}',
        positive_features=vector(1),negatives=[dict(siren='000000010',features=vector(1)),
                                             dict(siren='000000011',features=vector(0))]) for i in range(512)]
    models={str(f):dict(heldout_fold=f,query_ids=[s['query_id'] for s in scenes if s['fold']!=f],
        weights=[0.]*11) for f in (2,3,4)}
    return scenes,registry,models,[dict(query_id='FOREIGN_PILOT')]


def test_place_contrast_is_measured_not_assumed_local():
    r=summarize(*fixture())
    for f in r.values():
        assert f['queries_with_no_place_contrast'] == 0
        assert f['positive_place_negative_place_counts']['1_0'] == f['training_queries']
        assert f['negative_pairs'] == 2*f['training_queries']


def test_protected_entities_are_removed_before_pair_counts():
    args=fixture();scenes=args[0]
    scenes[0]['negatives'][1]['siren']='000000003'
    r=summarize(*args)
    assert r['3']['queries_with_no_place_contrast'] == 1
    assert r['4']['queries_with_no_place_contrast'] == 0


@pytest.mark.parametrize('bad',['partial','fold0','overlap','heldquery','nonfinite'])
def test_unsafe_training_metadata_is_not_an_admitted_audit(bad):
    scenes,registry,models,controls=fixture()
    if bad=='partial':scenes.pop()
    elif bad=='fold0':scenes[0]['fold']=0
    elif bad=='overlap':controls[0]['query_id']=scenes[0]['query_id']
    elif bad=='heldquery':models['2']['query_ids'].append(scenes[0]['query_id'])
    else:scenes[0]['positive_features'][0]=float('nan')
    with pytest.raises(ValueError):summarize(scenes,registry,models,controls)
