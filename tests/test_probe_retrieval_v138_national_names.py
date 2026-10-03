import pytest

from scripts.probe_retrieval_v138_national_names import query_rows


def source():
    return dict(truth_opened=False,old_candidates_opened=False,queries=[dict(query_id=f'q{i}',fold=2+i%3,
        crm_name='TEST',crm_number='',crm_suffix='',crm_street='',crm_postcode='',crm_insee='') for i in range(128)])


def test_fixed_query_prefix_no_labels_models_or_old_candidates_required():
    old=source();rows=query_rows(old,4)
    assert rows==old['queries'][:4] and [r['query_id'] for r in rows]==['q0','q1','q2','q3']


@pytest.mark.parametrize('kind',['truth','hits','fold0','fold1','extra_field','duplicate','count'])
def test_query_side_closed_folds_fields_and_denominator_guards(kind):
    old=source()
    if kind=='truth':old['truth_opened']=True
    elif kind=='hits':old['old_candidates_opened']=True
    elif kind=='fold0':old['queries'][-1]['fold']=0
    elif kind=='fold1':old['queries'][-1]['fold']=1
    elif kind=='extra_field':old['queries'][0]['human_siret']='00000000100001'
    elif kind=='duplicate':old['queries'][-1]['query_id']='q0'
    elif kind=='count':old['queries'].pop()
    with pytest.raises(ValueError,match='query-only'):query_rows(old,4)
