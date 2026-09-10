"""Preflight for production-like replay; no change to frozen v23/v30 recipes.

Population/isolation is label-side. Query and legacy replay project only input
fields and never use the ledger to obtain a candidate. Failure blocks the full run.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.xgb_matcher.hierarchical_retrieval import RetrievalQuery
from src.xgb_matcher.retrieval_v20_workbench import ROOT, Budget, record, verify, write_json

DATA = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100')
INPUT = DATA/'datasets/retrieval_all_human_v11_benchmark/ec2a91a849e46cba/development.parquet'
LEDGER = DATA/'datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet'
RAW = DATA/'experiments/retrieval_all_human_v11/raw_union_train234.parquet'
PANEL = DATA/'audits/retrieval_master_v15_feasibility_cohort/72896b6f70d7c9eb/queries.parquet'
MODEL = DATA/'audits/retrieval_v23_oof_master/bb12058773f80d6f'
SAFE = ['query_id','oof_fold','crm_name','crm_address','postcode','insee']
RESOURCE = dict(max_seconds=1800, max_rss_bytes=14*1024**3,max_swapout_pages_delta=0)


def input_rows(path=INPUT):
    # Predicate and projection at the scanner boundary, never load forbidden folds.
    return pq.read_table(path, columns=SAFE, filters=[('oof_fold','in',[2,3,4])]).to_pylist()


def normalize_input(row):
    return RetrievalQuery.from_mapping({k:row.get(k) for k in SAFE if k not in ('query_id','oof_fold')})


def normalized_rows(rows):
    return [dict(query_id=r['query_id'],fold=int(r['oof_fold']),**asdict(normalize_input(r))) for r in rows]


def db():
    c=duckdb.connect(); c.execute("SET memory_limit='2GB'");c.execute('SET threads=2');return c


def audit_population(output):
    sources=[record(verify(dict(path=INPUT,sha256='fb6d29ee1bc44d88162266485bb916d5cc178b46809ef9222b5d45fc80915705'))),
             record(verify(dict(path=LEDGER,sha256='24f5928cbf2669b0a79ac00a284d14e5fe2288abb5a4ddcbd77a46677c315720'))),
             record(verify(dict(path=PANEL,sha256='f9125a7ed7b7c45c18035f675f13c3a75f5261737d86e47f57b44a766997ab48')))]
    c=db(); c.register('input',pa.Table.from_pylist(input_rows()))
    c.execute('CREATE TABLE labels AS SELECT * FROM read_parquet(?) WHERE fold IN (2,3,4)',[str(LEDGER)])
    counts=c.execute('SELECT fold,count(*) n,count(DISTINCT query_id) unique_queries FROM labels GROUP BY fold ORDER BY fold').fetchall()
    assert c.execute('SELECT count(*) FROM (SELECT query_id FROM labels GROUP BY 1 HAVING count(*)<>1)').fetchone()[0]==0
    assert c.execute("SELECT count(*) FROM labels WHERE NOT regexp_full_match(human_ground_truth_siret,'[0-9]{14}')").fetchone()[0]==0
    assert c.execute('SELECT count(*) FROM (SELECT human_ground_truth_siren FROM labels GROUP BY 1 HAVING count(DISTINCT fold)>1)').fetchone()[0]==0
    assert c.execute('SELECT count(*) FROM input i FULL JOIN labels l ON i.query_id=l.query_id AND i.oof_fold=l.fold WHERE i.query_id IS NULL OR l.query_id IS NULL').fetchone()[0]==0
    split_records=[]
    for fold in (2,3,4):
        p=MODEL/f'outer{fold}_split.json';s=json.loads(p.read_text());sources.extend([record(p),record(MODEL/f'outer{fold}.json')])
        assert s['target_folds']==[fold] and s['train_folds']==[f for f in (2,3,4) if f!=fold]
        ids=s['trained_query_ids'];c.register('trained',pa.table({'query_id':ids}))
        assert c.execute('SELECT count(*) FROM trained ANTI JOIN labels USING(query_id)').fetchone()[0]==0
        assert c.execute('SELECT count(*) FROM trained JOIN labels USING(query_id) WHERE fold=?',[fold]).fetchone()[0]==0
        overlap=c.execute('''SELECT count(*) FROM labels v SEMI JOIN
          (SELECT DISTINCT human_ground_truth_siren FROM labels JOIN trained USING(query_id)) t
          USING(human_ground_truth_siren) WHERE v.fold=?''',[fold]).fetchone()[0]
        assert overlap==0
        split_records.append(dict(fold=fold,trained_queries=len(ids),target_siren_overlap=overlap))
    excluded=LEDGER.parent/'excluded_missing_human_labels.parquet'
    sources.append(record(verify(dict(path=excluded,sha256='12d991f2dd5129ac2230a2caa420f3de899f948588954c6ce1efec38a9cb8baa'))))
    exclusion_counts=c.execute('SELECT fold,count(*) FROM read_parquet(?) WHERE fold IN (2,3,4) GROUP BY fold ORDER BY fold',[str(excluded)]).fetchall()
    inputs=normalized_rows(input_rows());pq.write_table(pa.Table.from_pylist(inputs),output/'queries.parquet')
    # Existing query surface used only as a parity oracle, never as a generation input.
    c.execute('''CREATE TABLE old AS SELECT query_id,fold,arg_min(crm_name,union_rank) crm_name,
      arg_min(crm_address,union_rank) crm_address,arg_min(crm_number,union_rank) crm_number,
      arg_min(crm_postcode,union_rank) crm_postcode,arg_min(crm_insee,union_rank) crm_insee,
      arg_min(postcode,union_rank) postcode,arg_min(insee,union_rank) insee
      FROM read_parquet(?) WHERE fold IN (2,3,4) GROUP BY query_id,fold''',[str(RAW)])
    by_id={r['query_id']:r for r in inputs};mismatches=[]
    for r in c.execute('SELECT * FROM old ORDER BY query_id').df().to_dict('records'):
        expected=by_id[r['query_id']];actual=asdict(RetrievalQuery.from_mapping(r))
        fields=[k for k in actual if actual[k]!=expected[k]]
        if fields:mismatches.append(dict(query_id=r['query_id'],fold=r['fold'],fields=fields))
    assert c.execute('SELECT count(*) FROM old').fetchone()[0]==len(inputs)
    write_json(output/'query_mismatches.json',mismatches)
    return dict(status='PASS_POPULATION_AND_INPUT_PARITY' if not mismatches else 'STOP_QUERY_RECONSTRUCTION_MISMATCH',
       eligible_queries=len(inputs),counts_by_fold=counts,excluded_counts_by_fold=exclusion_counts,
       excluded_sidecar=str(excluded),model_isolation=split_records,query_mismatches=len(mismatches),sources=sources,
       fold0_opened=False,fold1_opened=False,quality_measured=False)


def replay_legacy(output,count=16):
    from src.xgb_matcher.official_evidence_retrieval import OfficialEvidenceRetriever,OfficialEvidenceRetrievalConfig
    from src.xgb_matcher.retrieval_all_human_v11 import AllHumanV11Retriever,RetrievalV11Config
    from src.xgb_matcher.retrieval_sidecars_v11 import RetrievalSidecarReader
    if not 1<=count<=16:raise ValueError('Bounded preflight only')
    v15=json.loads((ROOT/'config/retrieval_all_human_v15_master_feasibility.json').read_text())['authorities']
    backends=json.loads((ROOT/'config/retrieval_all_human_v16_parent_workbench.json').read_text())['authorities']['backends']
    for a in [v15['base_index'],v15['sidecars'],*v15['overlay_indices'],backends['temporal'],backends['roles'],backends['sector']]:
        verify(dict(path=Path(a['path'])/'manifest.json',sha256=a['manifest_sha256']))
    panel=pq.read_table(PANEL,columns=['query_id','fold']).to_pylist()
    chosen={r['query_id'] for r in sorted(panel,key=lambda r:r['query_id'])[:count]}
    rows=sorted([r for r in input_rows() if r['query_id'] in chosen],key=lambda r:r['query_id'])
    assert len(rows)==count
    c=db();c.register('selected',pa.table({'query_id':sorted(chosen)}))
    old=c.execute('''SELECT r.query_id,r.siret,r.siren,r.union_rank FROM read_parquet(?) r
      SEMI JOIN selected USING(query_id) WHERE fold IN (2,3,4) ORDER BY r.query_id,r.union_rank''',[str(RAW)]).fetchall()
    expected={r['query_id']:[] for r in rows}
    for qid,siret,siren,rank in old:expected[qid].append((siret,siren,rank))
    vc=RetrievalV11Config.load(ROOT/'config/retrieval_all_human_v11.json')
    oc=replace(OfficialEvidenceRetrievalConfig.load(ROOT/'config/retrieval_ltr_admission_dossier_v2.json'),
        enable_direct_parent_expansion=False,enable_consensus_parent_expansion=False,relation_seed_limit=vc.relation_seed_limit,search_workers=4)
    report=[]
    with RetrievalSidecarReader(Path(v15['sidecars']['path'])) as sidecars:
        sidecars._connection.execute("SET memory_limit='2GB'");sidecars._connection.execute('SET threads=2')
        official=OfficialEvidenceRetriever.from_index_paths(v15['base_index']['path'],
          [a['path'] for a in v15['overlay_indices']],oc,temporal_overlay_index_path=backends['temporal']['path'],
          legal_entity_role_index_path=backends['roles']['path'],sector_official_index_path=backends['sector']['path'])
        retriever=AllHumanV11Retriever(official,sidecars,vc)
        try:
            for i,row in enumerate(rows):
                start=time.monotonic();result=retriever.retrieve(normalize_input(row))
                actual=[(r.siret,r.siren,r.union_rank) for r in result.candidates]
                base=expected[row['query_id']]
                report.append(dict(query_id=row['query_id'],fold=row['oof_fold'],exact_order=actual==base,
                  missing=len(set(base)-set(actual)),added=len(set(actual)-set(base)),
                  candidate_count=len(actual),seconds=time.monotonic()-start))
                print(json.dumps(dict(processed=i+1,total=count,parity=actual==base,seconds=report[-1]['seconds'])),flush=True)
        finally:official._search_pool.shutdown(wait=True)
    write_json(output/'legacy_rows.json',report)
    return dict(status='PASS_LEGACY_RECONSTRUCTION' if all(r['exact_order'] for r in report) else 'STOP_LEGACY_RECONSTRUCTION_MISMATCH',
      queries=count,matching_queries=sum(r['exact_order'] for r in report),quality_measured=False,
      note='This validates legacy membership/ranks only, not the remaining v16-v30 channels or product latency.')


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['population','legacy']);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=16);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    with Budget(RESOURCE) as budget:
        result=audit_population(a.output) if a.phase=='population' else replay_legacy(a.output,a.count)
    result.update(resources=budget.report(),implementation=record(Path(__file__)),phase=a.phase)
    write_json(a.output/'manifest.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
