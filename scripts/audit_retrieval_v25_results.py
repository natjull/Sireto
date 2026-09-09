#!/usr/bin/env python3
"""Independent source validation and ledger recomputation; diagnostic only."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.xgb_matcher.retrieval_v20_workbench import Budget,record,verify,write_json,sql_path
from src.xgb_matcher.retrieval_v21_membership import manifest_files,open_workbench_db
from src.xgb_matcher.retrieval_v25_conjunctive import code_records


def run(generation,evaluation,output):
    gm,gf=manifest_files(record(Path(generation)/'manifest.json'))
    em,ef=manifest_files(record(Path(evaluation)/'manifest.json'))
    if gm['implementation']!=code_records() or em['generation_manifest']!=record(Path(generation)/'manifest.json'):
        raise ValueError('Recipe lineage')
    _,bf=manifest_files(em['baseline_generation'])
    _,af=manifest_files(dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v25_discovery_audit/arrowfix/manifest.json',
                            sha256='e6b7b842b6e77cffa90e7e7f2b2b0190c119158424167405e181c046bf2adb83'))
    _,pf=manifest_files(record(Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v25_conjunctive_preflight/fixed512/manifest.json')))
    source=verify(dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/siren_dossier_retrieval_v3_full/retrieval_siret_documents.parquet',
                       sha256='33d9112bdf7c41640e65023b597754c4d4495ff510324ca6c7aff54e01bd30ef'))
    ledger=verify(dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet',
                       sha256='24f5928cbf2669b0a79ac00a284d14e5fe2288abb5a4ddcbd77a46677c315720'))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with Budget(dict(max_seconds=1800,max_rss_bytes=15032385536,max_swapout_pages_delta=0)) as budget:
        c=open_workbench_db(output)
        for name,path in [('q',gf['queries.parquet']),('s',gf['signals.parquet']),('e',ef['outcomes.parquet']),
                          ('base',bf['candidates.parquet']),('d',af['label_side_diagnoses.parquet']),
                          ('official',verify(em['official_loss_sidecar'])),('acceptable',verify(em['acceptable_operational_sidecar']))]:
            c.execute(f'CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)',[str(path)])
        c.execute('''CREATE TEMP TABLE labels AS SELECT l.query_id,l.fold,l.human_ground_truth_siret gt_siret,
          l.human_ground_truth_siren gt_siren FROM read_parquet(?) l SEMI JOIN q USING(query_id,fold) WHERE l.fold IN (2,3,4)''',[str(ledger)])
        assert c.execute('SELECT count(*),count(DISTINCT query_id) FROM labels').fetchone()==(1034,1034)
        c.execute('''CREATE TEMP TABLE docs AS SELECT document_id siret,siren,insee,postcode,number
          FROM read_parquet(?) WHERE document_id IN (SELECT siret FROM s) OR siren IN (SELECT gt_siren FROM labels)''',[str(source)])
        assert c.execute('SELECT count(*) FROM s ANTI JOIN docs USING(siren,siret)').fetchone()[0]==0
        assert c.execute('SELECT count(*) FROM s JOIN docs x USING(siren,siret) JOIN q USING(query_id,fold) WHERE upper(trim(x.number))<>q.number OR q.number=\'\' OR q.address=\'\'').fetchone()[0]==0
        assert c.execute('''SELECT count(*) FROM s JOIN docs x USING(siren,siret) JOIN q USING(query_id,fold)
          JOIN d USING(query_id,fold) WHERE CASE WHEN d.effective_geo_field='insee' THEN x.insee<>q.insee ELSE x.postcode<>q.postcode END''').fetchone()[0]==0
        for filename,order in [('queries.parquet',['query_id']),('signals.parquet',['query_id','candidate_rank'])]:
            small=pq.read_table(pf[filename]).to_pandas().sort_values(order).reset_index(drop=True)
            large=pq.read_table(gf[filename]).to_pandas()
            large=large.loc[large.query_id.isin(small.query_id)].sort_values(order).reset_index(drop=True)
            pd.testing.assert_frame_equal(small,large)
        c.execute('''CREATE TEMP TABLE natural_union AS SELECT query_id,fold,siren,siret FROM base
          UNION SELECT query_id,fold,siren,siret FROM s''')
        c.execute('''CREATE TEMP TABLE audit AS SELECT l.*,o.official_catalog_present,
          EXISTS(SELECT 1 FROM base b WHERE b.query_id=l.query_id AND b.siret=l.gt_siret) old_exact,
          EXISTS(SELECT 1 FROM natural_union u WHERE u.query_id=l.query_id AND u.siret=l.gt_siret) exact,
          EXISTS(SELECT 1 FROM natural_union u WHERE u.query_id=l.query_id AND u.siren=l.gt_siren) parent,
          EXISTS(SELECT 1 FROM natural_union u JOIN acceptable a USING(query_id,siret) WHERE u.query_id=l.query_id) op,
          EXISTS(SELECT 1 FROM s WHERE s.query_id=l.query_id AND s.siret=l.gt_siret) channel_exact,
          EXISTS(SELECT 1 FROM s WHERE s.query_id=l.query_id AND s.siren=l.gt_siren) channel_parent
          FROM labels l JOIN official o USING(query_id,fold,gt_siret,gt_siren)''')
        assert c.execute('''SELECT count(*) FROM audit a JOIN e USING(query_id,fold)
          WHERE a.exact<>e.union_exact OR a.parent<>e.union_parent OR (a.exact OR a.op)<>e.union_operational
          OR a.channel_exact<>e.channel_exact OR a.channel_parent<>e.channel_parent''').fetchone()[0]==0
        losses=c.execute('''SELECT CASE WHEN exact THEN 'NATURAL_UNION_EXACT_HIT'
          WHEN NOT official_catalog_present THEN 'OFFICIAL_CATALOG_ABSENT'
          WHEN NOT parent THEN 'PARENT_ABSENT_FROM_RESERVOIR'
          ELSE 'SITE_ABSENT_FROM_RESERVOIR' END stage,count(*) n FROM audit GROUP BY 1''').fetchall()
        c.execute('''CREATE TEMP TABLE number_diagnostic AS SELECT l.query_id,l.fold,d.loss_stage,d.discovery_diagnosis,
          q.number<>'' AND q.address<>'' query_eligible,
          coalesce(bool_or(q.number<>'' AND upper(trim(x.number))=q.number AND
            CASE WHEN d.effective_geo_field='insee' THEN x.insee=q.insee ELSE x.postcode=q.postcode END),false) any_local_parent_number_match,
          coalesce(bool_or(q.number<>'' AND x.siret=l.gt_siret AND upper(trim(x.number))=q.number),false) exact_site_number_match
          FROM labels l JOIN q USING(query_id,fold) JOIN d USING(query_id,fold)
          LEFT JOIN docs x ON x.siren=l.gt_siren GROUP BY ALL''')
        nums=c.execute('''SELECT discovery_diagnosis,count(*) n,sum(CAST(query_eligible AS INT)) eligible,
          sum(CAST(any_local_parent_number_match AS INT)) local_parent_number_match,
          sum(CAST(exact_site_number_match AS INT)) exact_number_match FROM number_diagnostic
          WHERE loss_stage='PARENT_NOT_RETRIEVED' GROUP BY 1''').df().to_dict('records')
        c.execute(f'COPY number_diagnostic TO {sql_path(output/"number_diagnostic.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        counts=c.execute('SELECT count(*),sum(CAST(exact AS INT)),sum(CAST(parent AS INT)),sum(CAST(exact OR op AS INT)),sum(CAST(channel_exact AS INT)),sum(CAST(channel_parent AS INT)) FROM audit').fetchone()
        c.close()
    result=dict(schema='sireto-v25-result-audit-1',passed=True,counts=dict(zip(['queries','union_exact','union_parent','union_operational','channel_exact','channel_parent'],counts)),
        union_loss_counts=dict(losses),number_diagnostic=nums,generation_manifest=record(Path(generation)/'manifest.json'),
        evaluation_manifest=record(Path(evaluation)/'manifest.json'),sources=[record(source),record(ledger)],
        implementation=record(Path(__file__)),preflight_full_equal=True,all_signal_identities_numbers_geography_valid=True,
        master_measured=False,fold0_opened=False,fold1_opened=False,resources=budget.report(),
        files={p.name:record(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'manifest.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('generation','evaluation','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.generation,a.evaluation,a.output)
