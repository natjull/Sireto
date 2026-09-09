from copy import deepcopy

import duckdb
import pyarrow as pa
import pytest

from src.xgb_matcher.retrieval_v21_membership import build_tables, open_workbench_db
from scripts.evaluate_retrieval_v21_membership import evaluate_tables, summarize


def fixture():
    q = [dict(query_id='q',fold=2,crm_insee='11111',crm_postcode='11100',crm_number='1',crm_suffix='',crm_street='RUE A')]
    raw = [dict(query_id='q',fold=2,siren='100000001',siret='10000000100001',base_rank=1)]
    signals = [dict(query_id='q',fold=2,siren='100000001',source_siret='10000000100001',channel='legacy_site_pool',channel_rank=1),
               dict(query_id='q',fold=2,siren='200000001',source_siret='',channel='v17_union',channel_rank=17),
               dict(query_id='q',fold=2,siren='300000001',source_siret='30000000100001',channel='siret_site_name_word',channel_rank=1)]
    sites = [dict(siren=siren,siret=siret,insee=insee,postcode=cp,street_number='1',street_number_suffix='',street_normalized='RUE A',administrative_state='A')
             for siren,siret,insee,cp in [('100000001','10000000100001','11111','11100'),
              ('200000001','20000000100001','11111','11100'),('200000001','20000000100002','11111','11100'),
              ('300000001','30000000100001','99999','99900')]]
    return dict(q=q,raw=raw,signals=signals,sites=sites)


def db(rows):
    c=duckdb.connect()
    for k,v in rows.items(): c.register(k,pa.Table.from_pylist(v))
    return c


def snapshot(rows):
    c=db(rows);build_tables(c)
    r=c.execute('SELECT siret,parent_rrf,parent_rank,site_address_rank,direct_rrf,rescue_score FROM candidates ORDER BY siret').fetchall()
    c.close();return r


def test_membership_without_rank_and_equal_site_evidence():
    c=db(fixture());build_tables(c)
    assert c.execute("SELECT parent_rrf FROM parents WHERE siren='200000001'").fetchone()==(0.0,)
    assert c.execute("SELECT count(*),count(DISTINCT site_address_rank),count(DISTINCT rescue_score),max(direct_rrf) FROM candidates WHERE siren='200000001'").fetchone()==(2,1,1,0.0)
    assert c.execute("SELECT count(*) FROM master WHERE siret='30000000100001'").fetchone()==(1,)
    c.close()


def test_artificial_rank_is_not_relevance():
    a=fixture();b=deepcopy(a);b['signals'][1]['channel_rank']=999999
    assert snapshot(a)==snapshot(b)


def test_scores_invariant_to_identity_permutation_even_when_evidence_ties():
    a=fixture();b=deepcopy(a)
    replacements={'q':'opaque-other-query','100000001':'900000009','200000001':'800000009','300000001':'700000009',
      '10000000100001':'90000000900009','20000000100001':'80000000900009',
      '20000000100002':'80000000900001','30000000100001':'70000000900009'}
    for rows in b.values():
        for row in rows:
            for key in ('query_id','siren','siret','source_siret'):
                if key in row:row[key]=replacements.get(row[key],row[key])
    inverse={v:k for k,v in replacements.items()}
    assert snapshot(a)==sorted((inverse[r[0]],*r[1:]) for r in snapshot(b))


def test_row_order_duplicate_evidence_and_unused_labels_do_not_change_scores():
    a=fixture();b=deepcopy(a)
    b['signals']+=deepcopy(b['signals'])*2
    for rows in b.values():
        rows.reverse()
        for row in rows:
            row.update(human_ground_truth_siret='FORBIDDEN',data_origin='ANOTHER_CRM',qualification_v3='UNRESOLVED',crm_alias='DO_NOT_READ')
    assert snapshot(a)==snapshot(b)


def test_queries_do_not_influence_each_other():
    a=fixture();b=deepcopy(a)
    for name in ('q','raw','signals'):
        b[name]+=[{**r,'query_id':'other','fold':4} for r in a[name]]
    c=db(b);build_tables(c)
    rows=c.execute("SELECT siret,parent_rrf,parent_rank,site_address_rank,direct_rrf,rescue_score FROM candidates WHERE query_id='q' ORDER BY siret").fetchall()
    assert rows==snapshot(a);c.close()


@pytest.mark.parametrize('missing_field',['crm_street','crm_number'])
def test_missing_address_does_not_create_joint_evidence(missing_field):
    rows=fixture();rows['q'][0][missing_field]=''
    c=db(rows);build_tables(c)
    assert c.execute('SELECT count(*) FROM candidates WHERE joint_address').fetchone()==(0,)
    c.close()


def test_cap_and_protected_prefix_with_more_than_2000_sites():
    rows=fixture();site=rows['sites'][0]
    rows['raw']=[dict(query_id='q',fold=2,siren='100000001',siret=f'{10000000100000+i:014d}',base_rank=i) for i in range(1,2001)]
    rows['sites']=[{**site,'siret':f'{10000000100000+i:014d}'} for i in range(1,2301)]
    rows['signals']=rows['signals'][:1]
    c=db(rows);build_tables(c)
    assert c.execute('SELECT count(*),count(DISTINCT siret),min(candidate_rank),max(candidate_rank) FROM master').fetchone()==(2000,2000,1,2000)
    assert c.execute('SELECT count(*) FROM raw ANTI JOIN master USING(query_id,siret) WHERE base_rank<=1750').fetchone()==(0,)
    c.close()


def test_forbidden_fold_fails_closed():
    rows=fixture();rows['q'][0]['fold']=1
    c=db(rows)
    with pytest.raises(ValueError,match='Forbidden fold'):build_tables(c)
    c.close()


def test_spill_is_on_the_explicit_output_volume_and_bounded(tmp_path):
    c=open_workbench_db(tmp_path)
    assert c.execute("SELECT current_setting('temp_directory')").fetchone()[0]==str(tmp_path/'_duckdb_spill')
    assert c.execute("SELECT current_setting('max_temp_directory_size')").fetchone()[0]=='12.0 GiB'
    c.close()


def test_evaluator_keeps_absent_unresolved_truth_in_denominator():
    rows={
      'q':[dict(query_id='control',fold=2),dict(query_id='missing',fold=4)],
      't':[dict(query_id='control',fold=2,gt_siret='10000000100001',cohort_kind='BASE_TOP1750_CONTROL',previous_exact_hit=True,label_kind='MATCH_EXACT'),
           dict(query_id='missing',fold=4,gt_siret='90000000100001',cohort_kind='BASE_TOP1750_MISS',previous_exact_hit=False,label_kind='UNRESOLVED')],
      'master':[dict(query_id='control',siret='10000000100001',candidate_rank=1)],
      'candidates':[dict(query_id='control',siret='10000000100001',base_rank=1)],
      'official_loss':[dict(query_id='control',official_catalog_present=True,parent_in_evidence=True,exact_state='A'),
                       dict(query_id='missing',official_catalog_present=False,parent_in_evidence=False,exact_state='')],
      'acceptable':[dict(query_id='control',siret='10000000100001',administrative_state='A',exact_truth=True)]}
    c=db(rows);result=evaluate_tables(c);s=summarize(result)
    assert s['queries']==2 and s['exact_hits']==1 and s['exact_recall']==.5
    assert result.loc[result.query_id=='missing','loss_stage'].iloc[0]=='OFFICIAL_CATALOG_ABSENT'
    c.close()
