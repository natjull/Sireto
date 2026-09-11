import duckdb
import pyarrow as pa
import pytest

from scripts.evaluate_retrieval_v33_full_replay import score_batch,summary,loss_stage,evaluate


@pytest.mark.parametrize('args,result',[
 ((True,True,True,True),'HIT'),((False,False,False,False),'OFFICIAL_CATALOG_ABSENT'),
 ((False,True,False,False),'PARENT_NOT_RETRIEVED'),((False,True,True,False),'SITE_NOT_EXPANDED'),
 ((False,True,True,True),'EVICTED_RESERVOIR_TO_MASTER2000')])
def test_exhaustive_loss_stage(args,result):assert loss_stage(*args)==result


def test_scoring_preserves_missing_queries_and_separates_operational():
    c=duckdb.connect()
    c.execute('''CREATE TABLE truth AS SELECT 'exact' query_id,2 fold,'a' gt_siret,'p' gt_siren,'A' exact_state,true official_catalog_present,false multiple_active_equivalents
      UNION ALL SELECT 'op',2,'b','p','F',true,false
      UNION ALL SELECT 'empty',3,'c','x','A',true,false
      UNION ALL SELECT 'catalog',4,'d','y',NULL,false,false''')
    c.execute('CREATE TABLE q AS SELECT query_id,fold FROM truth')
    c.execute("CREATE TABLE master AS SELECT 'exact' query_id,2 fold,'p' siren,'a' siret,5 candidate_rank UNION ALL SELECT 'op',2,'p','e',7")
    c.execute('CREATE TABLE pool AS SELECT query_id,fold,siren,siret FROM master')
    c.execute('CREATE TABLE parents AS SELECT DISTINCT query_id,fold,siren FROM pool')
    c.execute('''CREATE TABLE acceptable AS SELECT query_id,fold,gt_siret siret,true exact_truth,exact_state administrative_state FROM truth
      UNION ALL SELECT 'op',2,'e',false,'A' ''')
    rows=score_batch(c);byid={r['query_id']:r for r in rows}
    assert len(rows)==4 and byid['empty']['loss_stage']=='PARENT_NOT_RETRIEVED'
    assert byid['catalog']['loss_stage']=='OFFICIAL_CATALOG_ABSENT'
    assert not byid['op']['exact_hit'] and byid['op']['operational_hit']
    assert byid['op']['active_equivalent_when_exact_closed']
    s=summary(rows)
    assert s['queries']==4 and s['exact_hits']==1 and s['operational_hits']==2
    assert s['exact_recall']==.25 and s['operational_recall']==.5


def test_partial_generation_refused_before_labels(tmp_path):
    (tmp_path/'manifest.json').write_text('{"status":"SEALED_PAIRED_BATCH"}')
    with pytest.raises(ValueError,match='Complete sealed generation'):evaluate(tmp_path,tmp_path/'out')


def test_operational_requires_number_suffix_street_and_strict_geography():
    from scripts.evaluate_retrieval_v33_full_replay import operational_same_site
    q=dict(crm_number='12',crm_suffix='BIS',crm_street='RUE TEST',crm_insee='75101',crm_postcode='75001')
    s=dict(street_number='12',street_number_suffix='B',street_normalized='RUE TEST',insee='75101',postcode='75001')
    assert operational_same_site(q,s,set())
    assert not operational_same_site(q,dict(s,street_number_suffix=''),set())
    assert not operational_same_site(q,dict(s,street_normalized='RUE ADJACENTE'),set())
    assert not operational_same_site(q,dict(s,postcode='75002'),set())
    assert operational_same_site(q,dict(s,postcode='75002'),{'75101'})
    assert not operational_same_site(q,None,set())
