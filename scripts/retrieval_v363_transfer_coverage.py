"""Post-seal coverage of official ASSET_TRANSFER links, no identity merge."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import duckdb
import pyarrow as pa

from scripts import embeddinggemma2_night as io
from scripts.retrieval_v320_bodacc_relations import INPUT
from src.xgb_matcher.retrieval_v20_workbench import record,verify

ROOT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits')


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    evidence=ROOT/'retrieval_v360_miss_evidence/run_20261007_a'
    prior=io.read(evidence/'manifest.json'); rows=io.read(verify(prior['artifacts']['rows']))
    misses=[r for r in rows if not r['exact_hit']]
    report=io.read(verify(prior['source_report']))
    generation=io.read(verify(prior['generation']))
    outputs=io.read(verify(generation['outputs']))['rows']
    parents={r['query_id']:{c['siren'] for c in r['results']['primary']} for r in outputs}
    c=duckdb.connect(); c.execute('SET threads=1'); c.execute("SET memory_limit='64MB'")
    c.register('wanted',pa.table({'siren':sorted({r['official_siren'] for r in misses})}))
    # Original sealed parquet, NEVER rejected/partial V320 payload.
    relations=c.execute('''SELECT relation_id,from_identifier,to_identifier,effective_date
        FROM read_parquet(?) WHERE source='BODACC' AND relation_type='ASSET_TRANSFER'
        AND from_kind='SIREN' AND to_kind='SIREN'
        AND (from_identifier IN (SELECT siren FROM wanted) OR to_identifier IN (SELECT siren FROM wanted))''',
        [str(verify(INPUT))]).fetchall(); c.close()
    adjacency=defaultdict(list)
    for rid,source,target,date in relations:
        if source==target: continue
        adjacency[source].append((target,rid,'FROM',str(date)))
        adjacency[target].append((source,rid,'TO',str(date)))
    findings=[]
    for row in misses:
        edges=adjacency[row['official_siren']]
        seeds=[dict(other_siren=s,relation_id=rid,truth_role=role,effective_date=date)
            for s,rid,role,date in edges if s in parents[row['query_id']]]
        findings.append(dict(query_id=row['query_id'],fold=row['fold'],official_siren=row['official_siren'],
            linked_official_sirens=sorted({e[0] for e in edges}),already_retrieved_related_seed=seeds,
            parent_already_found=bool(row['same_parent_output_ranks'])))
    summary=dict(status='POSTSEAL_TRANSFER_COVERAGE_NOT_RETRIEVAL_RECALL',misses=len(misses),
        misses_with_transfer_link=sum(bool(r['linked_official_sirens']) for r in findings),
        misses_with_naturally_retrieved_related_seed=sum(bool(r['already_retrieved_related_seed']) for r in findings),
        missing_parent_with_retrieved_related_seed=sum(bool(r['already_retrieved_related_seed']) and not r['parent_already_found'] for r in findings),
        source=INPUT,prior_diagnostic=record(evidence/'manifest.json'),
        source_only_no_crm_aliases=True,asset_transfer_is_not_entity_equivalence=True,
        no_claim_of_expansion_admission_gain=True,fold0_opened=False,fold1_opened=False,serving_changed=False)
    io.write(out/'rows.json',findings); io.write(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k.startswith('miss')})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'retrieval_v363_transfer_coverage'): raise ValueError('Dedicated namespace')
    run(a.output)
