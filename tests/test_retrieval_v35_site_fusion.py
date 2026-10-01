"""Leakage/candidate invariants for the isolated retrieval admission pilot."""
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from scripts.retrieval_v35_site_fusion import (
    FEATURES, PARAMS, check_model_split, context_features, negative_indices,
    sample_training, summarize, train_context,
)
from src.xgb_matcher.retrieval_v23_joint_parent import build_features, make_master


@pytest.mark.parametrize('outer,query,expected', [(2,3,4),(2,4,3),(3,2,4),(3,4,2),(4,2,3),(4,3,2)])
def test_nested_context_excludes_both(outer,query,expected):
    assert train_context(outer,query) == expected
    with pytest.raises(ValueError):
        train_context(outer,outer)


def test_reject_upstream_target_training_identity():
    split = dict(train_folds=[4],target_folds=[2,3],trained_query_ids=['leak'])
    with pytest.raises(ValueError,match='leaked'):
        check_model_split(split,'inner',4,[('leak',2)])
    check_model_split(split,'inner',4,[('other',2),('leak',4)])


def test_sampling_keeps_only_natural_truth_and_neutralizes_equivalents():
    frame = pd.DataFrame([dict(query_id=q,siret=str(i),candidate_rank=i+1)
                          for q in ('present','absent') for i in range(200)])
    sampled, stats = sample_training(frame,dict(present='190',absent='missing'),dict(present={'0','1'}))
    assert stats['queries']==2 and stats['trainable']==1 and stats['absent_positive']==1
    assert stats['neutralized']==2 and len(sampled)==129
    assert set(sampled.query_id)=={'present'} and sampled.target.sum()==1
    assert not {'0','1','missing'} & set(sampled.siret)
    assert '190' in set(sampled.siret)


def test_negative_sampler_covers_tail_without_duplicates():
    for n in (0,1,128,129,1999):
        ix=negative_indices(n)
        assert len(ix)==min(128,n)==len(set(ix))
        assert all(0<=i<n for i in ix)
    assert negative_indices(1999)[-1]>1950


def test_missing_truth_counts_in_denominator_and_loss():
    rows=[dict(exact_rank=1,operational_rank=1,baseline_exact_rank=110),
          dict(exact_rank=None,operational_rank=None,baseline_exact_rank=None)]
    s=summarize(rows)
    assert s['queries']==2 and s['100']['exact_recall']==.5
    assert s['100']['gains']==1 and s['100']['losses']==0
    assert s['2000']['exact_hits']==1


def test_no_identifier_features_or_external_compute():
    assert not {'query_id','siren','siret','crm_number','crm_name','crm_postcode'} & set(FEATURES)
    assert PARAMS['n_jobs']==2 and PARAMS['tree_method']=='hist'
    assert not any(k in PARAMS for k in ('early_stopping_rounds','eval_set'))


def test_separate_name_and_street_evidence_survives_wrong_number():
    c=duckdb.connect()
    tables={
      'q':[dict(query_id='q',fold=2,crm_name='ACME',crm_number='10',crm_suffix='',crm_street='RUE PARIS',crm_insee='A',crm_postcode='P')],
      'pool':[dict(query_id='q',fold=2,siren='parent',siret='site',base_rank=1,parent_rrf=.1,direct_rrf=.1)],
      'names':[dict(siren='parent',siret='site',subject_kind='SIRET',name_role='SITE_CURRENT',name='ACME')],
      'sites':[dict(siren='parent',siret='site',street_normalized='RUE PARIS',street_number='12',street_number_suffix='',insee='A',postcode='P',administrative_state='A')],
    }
    for k,v in tables.items(): c.register(k,pa.Table.from_pylist(v))
    build_features(c)
    row=c.execute('SELECT site_name_jw,street,number_exact,joint_jw FROM site_evidence').fetchone()
    assert row==(1.,1.,False,0.)
    # The input to the new ranker exposes the un-gated evidence, unlike joint_jw.
    assert {'site_name_jw','street','number_exact','soft_name_street'} <= set(FEATURES)
    c.close()
