from copy import deepcopy

import duckdb
import pyarrow as pa
import pytest

from scripts.retrieval_v22_global_budget import select_master, outcomes, summary


def rows(n=2300):
    return [dict(query_id='q',fold=2,siret=f'{i:014d}',siren=f'{i//10:09d}',base_rank=i+1 if i<1750 else None,
                 rescue_score=float(i),direct_rrf=.1,parent_rrf=.1) for i in range(n)]


def selected(data):
    c=duckdb.connect();c.register('candidates',pa.Table.from_pylist(data));select_master(c)
    result=c.execute('SELECT siret,rescue_score,candidate_rank FROM master ORDER BY candidate_rank').fetchall()
    c.close();return result


def test_former_rank_one_can_lose_to_complementary_candidates():
    data=rows();result=selected(data)
    assert len(result)==len({r[0] for r in result})==2000
    assert data[0]['siret'] not in {r[0] for r in result}
    assert result[0]==(data[-1]['siret'],2299.,1)
    assert result[-1][2]==2000


def test_baseline_audit_rank_and_unused_labels_do_not_affect_admission():
    data=rows();changed=deepcopy(data)
    for r in changed:
        r.update(base_rank=1,human_ground_truth_siret='DO_NOT_READ',qualification='MATCH_EXACT',data_origin='other')
    changed.reverse()
    assert selected(data)==selected(changed)


def test_small_pool_kept_without_padding_or_positive_injection():
    assert len(selected(rows(3)))==3


@pytest.mark.parametrize('fold',[0,1])
def test_forbidden_folds_rejected(fold):
    data=rows(2);data[0]['fold']=fold
    with pytest.raises(ValueError,match='folds'):selected(data)


def test_duplicate_siret_and_nonfinite_score_fail_closed():
    data=rows(2)
    with pytest.raises(ValueError,match='Duplicate'):selected(data+[data[0]])
    data[0]['rescue_score']=float('nan')
    with pytest.raises(ValueError,match='scores'):selected(data)


def test_evaluation_publishes_control_regression_and_gain_without_filtering():
    data={
      'q':[dict(query_id='a',fold=2),dict(query_id='b',fold=3)],
      't':[dict(query_id='a',fold=2,gt_siret='A',previous_exact_hit=True,site_in_reservoir=True,label_kind='UNRESOLVED',cohort_kind='BASE_TOP1750_CONTROL'),
           dict(query_id='b',fold=3,gt_siret='B',previous_exact_hit=False,site_in_reservoir=True,label_kind='MATCH_EXACT',cohort_kind='BASE_TOP1750_MISS')],
      'master':[dict(query_id='a',siret='OTHER',candidate_rank=1),dict(query_id='b',siret='B',candidate_rank=1)],
      'official_loss':[dict(query_id=q,official_catalog_present=True,parent_in_evidence=True,exact_state='A') for q in ('a','b')],
      'acceptable':[dict(query_id=q,siret=s,administrative_state='A',exact_truth=True) for q,s in [('a','A'),('b','B')]]}
    c=duckdb.connect()
    for name,r in data.items():c.register(name,pa.Table.from_pylist(r))
    df=outcomes(c);result=summary(df)
    assert result['queries']==2 and result['exact_hits']==1
    assert result['gains_vs_v21']==result['losses_vs_v21']==1
    assert df.loc[df.query_id=='a','loss_stage'].iloc[0]=='EVICTED_RESERVOIR_TO_MASTER2000'
    c.close()
