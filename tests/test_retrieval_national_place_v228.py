"""Synthetic numerical/control fixtures; not a retrieval quality result."""
from copy import deepcopy
import pytest
from src.xgb_matcher.retrieval_national_place_v228 import fit_place


def fixture():
    def v(place,name=0.):return [name,0.,0.,0.,float(place)]+[0.]*6
    registry={str(f):[str(f)] for f in (2,3,4)}
    scenes=[dict(query_id=f'T{i}',fold=2+i%3,positive_siren=str(2+i%3),positive_features=v(1),
        negatives=[dict(siren='LOCAL',features=v(1))]) for i in range(512)]
    mined=[dict(query_id=s['query_id'],fold=s['fold'],channels=dict(name=[dict(siret='NATIONAL',siren='N')],joint=[]),
        features=dict(NATIONAL=v(0,1.))) for s in scenes[:64]]
    original={str(f):dict(weights=[1.,0.,0.,0.,0.]+[0.]*6,query_ids=[s['query_id'] for s in scenes if s['fold']!=f]) for f in (2,3,4)}
    return scenes,mined,registry,original,{s['query_id']:set() for s in scenes[:64]},2


def test_only_place_changes_on_genuine_national_contrasts():
    args=fixture();m=fit_place(*args)
    assert m['weights'][4]>0
    assert [w for i,w in enumerate(m['weights']) if i!=4]==[w for i,w in enumerate(args[3]['2']['weights']) if i!=4]
    assert m['place_training']['objective_after']<m['place_training']['objective_before']
    assert len(m['place_training']['query_ids'])>=16


def test_acceptable_national_positive_never_becomes_negative():
    args=fixture()
    for a in args[4].values():a.add('NATIONAL')
    with pytest.raises(ValueError,match='contrast'):fit_place(*args)


def test_held_negative_entities_are_excluded():
    args=fixture()
    for row in args[1]:row['channels']['name'].append(dict(siret='HELD',siren='2'));row['features']['HELD']=[0.]*11
    m=fit_place(*args)
    assert m['place_training']['exclusions']['held_national_references']>0


@pytest.mark.parametrize('bad',['partial','query_membership','nonfinite','fold0'])
def test_partial_or_contaminated_training_stops(bad):
    args=fixture()
    if bad=='partial':args[1].pop()
    elif bad=='query_membership':args[3]['2']['query_ids']=[]
    elif bad=='nonfinite':args[0][1]['positive_features'][0]=float('nan')
    else:args[0][1]['fold']=0
    with pytest.raises(ValueError):fit_place(*args)
