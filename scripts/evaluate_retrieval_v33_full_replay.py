"""Post-seal, streaming all-human evaluation. Never called by retrieval."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.retrieval_v33_full_replay import authority, code, manifests, files_for, DATA, ROOT, RESOURCE
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json, sql_path
from src.xgb_matcher.retrieval_site_inventory_v17 import same_site_evidence

LEDGER=dict(path=str(DATA/'datasets/retrieval_all_human_v12_label_ledger/1a99b7cb1db5a1692e39/label_ledger.parquet'),
 sha256='24f5928cbf2669b0a79ac00a284d14e5fe2288abb5a4ddcbd77a46677c315720')
SEGMENTS=('fold','qualification_v2','qualification_v3','label_kind','data_origin','ground_truth_state',
          'unseen_siren','new_site_known_siren','mega_base_pool')


def summary(rows):
    n=len(rows)
    return dict(queries=n,exact_hits=sum(r['exact_hit'] for r in rows),operational_hits=sum(r['operational_hit'] for r in rows),
      exact_recall=sum(r['exact_hit'] for r in rows)/n if n else None,
      operational_recall=sum(r['operational_hit'] for r in rows)/n if n else None,
      operational_only=sum(r['operational_hit'] and not r['exact_hit'] for r in rows),
      active_equivalent_when_exact_closed=sum(r['active_equivalent_when_exact_closed'] for r in rows),
      multiple_active_equivalents=sum(r['multiple_active_equivalents'] for r in rows),
      natural_pool_hits=sum(r['site_in_reservoir'] for r in rows),
      curves={str(k):dict(exact=sum(r['exact_rank'] is not None and r['exact_rank']<=k for r in rows),
        operational=sum(r['operational_rank'] is not None and r['operational_rank']<=k for r in rows)) for k in (100,200,500,1000,2000)})


def loss_stage(hit,catalog,parent,site):
    if hit:return 'HIT'
    if not catalog:return 'OFFICIAL_CATALOG_ABSENT'
    if not parent:return 'PARENT_NOT_RETRIEVED'
    if not site:return 'SITE_NOT_EXPANDED'
    return 'EVICTED_RESERVOIR_TO_MASTER2000'


def score_batch(c):
    rows=c.execute('''SELECT t.*,
      EXISTS(SELECT 1 FROM parents p WHERE p.query_id=t.query_id AND p.siren=t.gt_siren) parent_in_evidence,
      EXISTS(SELECT 1 FROM pool p WHERE p.query_id=t.query_id AND p.siret=t.gt_siret) site_in_reservoir,
      (SELECT min(candidate_rank) FROM master x WHERE x.query_id=t.query_id AND x.siret=t.gt_siret) exact_rank,
      (SELECT min(candidate_rank) FROM master x JOIN acceptable a USING(query_id,fold,siret) WHERE x.query_id=t.query_id) operational_rank,
      EXISTS(SELECT 1 FROM master x JOIN acceptable a USING(query_id,fold,siret)
        WHERE x.query_id=t.query_id AND NOT a.exact_truth AND a.administrative_state='A' AND t.exact_state='F') active_equivalent_when_exact_closed
      FROM truth t SEMI JOIN q USING(query_id,fold)''').fetch_arrow_table().to_pylist()
    for r in rows:
        r['exact_hit']=r['exact_rank'] is not None;r['operational_hit']=r['operational_rank'] is not None
        r['loss_stage']=loss_stage(r['exact_hit'],r['official_catalog_present'],r['parent_in_evidence'],r['site_in_reservoir'])
    return rows


def evaluate(generation,output):
    gm=json.loads((generation/'manifest.json').read_text())
    if gm['status']!='SEALED_FULL_PAIRED_GENERATION' or gm['queries']!=23571 or gm['quality_measured'] or gm['truth_columns_opened'] or gm['code']!=code():
        raise ValueError('Complete sealed generation required before labels')
    prepared=verify(gm['prepared']).parent;m=authority(prepared)
    batch_records=[];seen=set()
    # Complete denominator reconciliation before opening any labels.
    for r in gm['batches']:
        p=verify(r);bm,shards=manifests(p.parent)
        if bm['parity'] or bm['prepared']!=gm['prepared']:raise ValueError('Foreign batch')
        for s in shards:
            q=pq.read_table(verify(s['files']['queries.parquet']),columns=['query_id','fold']).to_pylist()
            ids={(r['query_id'],r['fold']) for r in q}
            if len(ids)!=len(q) or seen & ids:raise ValueError('Duplicate query/shard')
            seen |= ids
        batch_records.append((bm,shards))
    scope={(r['query_id'],r['fold']) for r in pq.read_table(verify(m['scope'])).to_pylist()}
    if seen!=scope or len(seen)!=23571:raise ValueError('Incomplete generation population')
    output.mkdir(parents=True,exist_ok=False)
    with Budget(RESOURCE) as budget:
        labels=pq.read_table(verify(LEDGER),columns=['query_id','fold','human_ground_truth_siret','human_ground_truth_siren',*SEGMENTS[1:]],
          filters=[('fold','in',[2,3,4])]).to_pylist()
        if len(labels)!=23571 or {(r['query_id'],r['fold']) for r in labels}!=scope:raise ValueError('Label denominator mismatch')
        c=duckdb.connect();c.execute("SET memory_limit='3GB'");c.execute('SET threads=2')
        c.register('labels',pa.Table.from_pylist(labels));sites=verify(m['sources']['sites'])
        official=c.execute('''SELECT s.* FROM read_parquet(?) s WHERE
          s.siren IN (SELECT human_ground_truth_siren FROM labels)
          OR s.siret IN (SELECT human_ground_truth_siret FROM labels)''',[str(sites)]).fetch_arrow_table().to_pylist()
        by_siren=defaultdict(list);by_site={}
        for s in official:
            if s['siret'] in by_site:raise ValueError('Duplicate official site')
            by_site[s['siret']]=s;by_siren[s['siren']].append(s)
        # Same-INSEE with a conflicting CP is only admissible for a multi-CP
        # commune (or missing CRM CP), as required by the business policy.
        multi={r[0] for r in c.execute('SELECT insee FROM read_parquet(?) WHERE postcode IS NOT NULL AND postcode<>\'\' GROUP BY insee HAVING count(DISTINCT postcode)>1',[str(sites)]).fetchall()}
        qrows=[]
        for _,ss in batch_records:
            for s in ss:qrows.extend(pq.read_table(verify(s['files']['queries.parquet'])).to_pylist())
        qmap={r['query_id']:r for r in qrows};cat=[];acceptable=[]
        for t in labels:
            q=qmap[t['query_id']];exact=by_site.get(t['human_ground_truth_siret'])
            if exact is not None and exact['siren']!=t['human_ground_truth_siren']:raise ValueError('Label/official parent conflict')
            cat.append(dict(query_id=t['query_id'],fold=t['fold'],gt_siret=t['human_ground_truth_siret'],gt_siren=t['human_ground_truth_siren'],
              official_catalog_present=exact is not None,exact_state=exact['administrative_state'] if exact else None))
            acceptable.append(dict(query_id=t['query_id'],fold=t['fold'],siret=t['human_ground_truth_siret'],exact_truth=True,
              administrative_state=exact['administrative_state'] if exact else None,same_site_evidence=False))
            for s in by_siren[t['human_ground_truth_siren']]:
                if s['siret']==t['human_ground_truth_siret']:continue
                geography=(q['crm_postcode'] and q['crm_postcode']==s['postcode']) or (
                  q['crm_insee'] and q['crm_insee']==s['insee'] and (not q['crm_postcode'] or s['insee'] in multi))
                if not geography:continue
                if same_site_evidence(query_number=q['crm_number']+' '+q['crm_suffix'],query_street=q['crm_street'],
                  query_insee=q['crm_insee'],query_postcode=q['crm_postcode'],site_number=s['street_number'],site_suffix=s['street_number_suffix'],
                  site_street=s['street_normalized'],site_insee=s['insee'],site_postcode=s['postcode']):
                    acceptable.append(dict(query_id=t['query_id'],fold=t['fold'],siret=s['siret'],exact_truth=False,
                      administrative_state=s['administrative_state'],same_site_evidence=True))
        # These are label-side targets ONLY, written and sealed before scoring hits.
        pq.write_table(pa.Table.from_pylist(cat),output/'official_catalog_sidecar.parquet',compression='zstd')
        pq.write_table(pa.Table.from_pylist(acceptable),output/'acceptable_operational.parquet',compression='zstd')
        write_json(output/'label_side_manifest.json',dict(ledger=LEDGER,official_sites=m['sources']['sites'],generation=record(generation/'manifest.json'),
          implementation=record(Path(__file__)),policy=record(ROOT/'docs/siret_operational_equivalence_policy.md'),
          files={p.name:record(p) for p in output.glob('*.parquet')},queries=23571,never_used_in_retrieval=True))
        c.register('catalog',pa.Table.from_pylist(cat));c.register('acceptable',pa.Table.from_pylist(acceptable))
        c.execute('''CREATE TABLE truth AS SELECT l.*,cat.gt_siret,cat.gt_siren,cat.exact_state,cat.official_catalog_present,
          (SELECT count(*) FROM acceptable a WHERE a.query_id=l.query_id AND a.administrative_state='A')>1 multiple_active_equivalents
          FROM labels l JOIN catalog cat USING(query_id,fold)''')
        outcomes=defaultdict(list);caps={v:0 for v in ('v23_stable','v30_stable')}
        for bm,ss in batch_records:
            for v in outcomes.keys() | caps.keys():
                masters=files_for(ss,v+'/master.parquet');pools=files_for(ss,v+'/pool.parquet');parents=files_for(ss,v+'/parents.parquet')
                qs=files_for(ss,'queries.parquet')
                c.execute('CREATE OR REPLACE TABLE q AS SELECT query_id,fold FROM read_parquet(?)',[qs])
                c.execute('CREATE OR REPLACE TABLE master AS SELECT query_id,fold,siren,siret,candidate_rank FROM read_parquet(?)',[masters])
                c.execute('CREATE OR REPLACE TABLE pool AS SELECT * FROM read_parquet(?)',[pools])
                c.execute('CREATE OR REPLACE TABLE parents AS SELECT * FROM read_parquet(?)',[parents])
                if c.execute('SELECT count(*) FROM master ANTI JOIN pool USING(query_id,fold,siren,siret)').fetchone()[0]:raise ValueError('Non-natural master')
                if c.execute('SELECT count(*) FROM master ANTI JOIN q USING(query_id,fold)').fetchone()[0]:raise ValueError('Foreign master')
                if c.execute('SELECT count(*) FROM (SELECT query_id,count(*) n,count(DISTINCT siret) nd,min(candidate_rank) lo,max(candidate_rank) hi FROM master GROUP BY query_id HAVING n<>nd OR n>2000 OR lo<>1 OR hi<>n)').fetchone()[0]:raise ValueError('Master integrity')
                caps[v]=max(caps[v],c.execute('SELECT coalesce(max(n),0) FROM (SELECT count(*) n FROM master GROUP BY query_id)').fetchone()[0])
                outcomes[v].extend(score_batch(c))
        result={}
        for v,rows in outcomes.items():
            if len(rows)!=23571 or len({r['query_id'] for r in rows})!=23571:raise ValueError('Outcome denominator')
            segments={}
            for dim in SEGMENTS:
                groups=defaultdict(list)
                for r in rows:groups[str(r[dim]) if r[dim] is not None else 'UNKNOWN'].append(r)
                segments[dim]={k:summary(rs) for k,rs in groups.items()}
            losses=defaultdict(int)
            for r in rows:losses[r['loss_stage']]+=1
            result[v]=dict(primary=summary(rows),historical_all_human=summary(rows),segments=segments,loss_counts=dict(losses),max_master=caps[v])
            pq.write_table(pa.Table.from_pylist(sorted(rows,key=lambda r:r['query_id'])),output/(v+'_outcomes.parquet'),compression='zstd')
        a={r['query_id']:r for r in outcomes['v23_stable']};b={r['query_id']:r for r in outcomes['v30_stable']}
        paired={k:dict(gains=sum(b[q][k] and not a[q][k] for q in a),losses=sum(a[q][k] and not b[q][k] for q in a)) for k in ('exact_hit','operational_hit')}
    write_json(output/'manifest.json',dict(status='MEASURED_FULL_CRM_MASTER_ONLY',generation=record(generation/'manifest.json'),results=result,paired=paired,
      label_side=record(output/'label_side_manifest.json'),resources=budget.report(),implementation=record(Path(__file__)),
      decision='STOP',decision_reason='MASTER_ONLY_NO_PRERANK500_RECALL100_OR_PRODUCT_SLO_GATE',ready_for_fold1=False,
      fold0_opened=False,fold1_opened=False,prerank500_measured=False,recall100_product_measured=False,
      warning='Curves are ranks of Master only, not the learned downstream 500/100. Reused CRM, no cross-CRM claim.',
      files={p.name:record(p) for p in output.glob('*.parquet')}))
    print(json.dumps(dict(status='MEASURED_FULL_CRM_MASTER_ONLY',primary={v:r['primary'] for v,r in result.items()},paired=paired)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();evaluate(a.generation,a.output)
