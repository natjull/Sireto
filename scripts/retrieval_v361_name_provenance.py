"""Compare official pre-native name facts with the indexed names, postseal only."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import time

import duckdb
import pyarrow as pa

from scripts import embeddinggemma2_night as io
from scripts.retrieval_v360_miss_evidence import name_evidence
from src.xgb_matcher.retrieval_v20_workbench import record,verify

SOURCE=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v360_miss_evidence/run_20261007_a')
DICTIONARY=dict(path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v130_variant_dictionary/compile_20261003_a/manifest.json',
    sha256='a69477e35c53f9f6d702b9e870580e377a765fb1e92e990782c909fed29241f4')
OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v361_name_provenance')


def run(out):
    out.mkdir(parents=True,exist_ok=False); start=time.monotonic()
    prior=io.read(SOURCE/'manifest.json'); rows=io.read(verify(prior['artifacts']['rows']))
    if prior['population']!=2000 or prior['misses']!=177: raise ValueError('Complete sealed diagnostic required')
    dictionary=io.read(verify(DICTIONARY))
    if dictionary['status']!='SEALED_V130_COMPLETE_ALL_OFFICIAL_NAME_OWNER_DICTIONARY': raise ValueError('Official names authority')
    sites={int(r['official_siret']) for r in rows}; parents={int(r['official_siren']) for r in rows}
    c=duckdb.connect(); c.execute('SET threads=1'); c.execute("SET memory_limit='128MB'")
    c.register('wanted_site',pa.table({'owner_key':pa.array(sorted(sites),type=pa.uint64())}))
    c.register('wanted_parent',pa.table({'parent_key':pa.array(sorted(parents),type=pa.uint32())}))
    by_site=defaultdict(set); by_parent=defaultdict(set); verified=[]; epochs={}
    try:
        for i,b in enumerate(dictionary['buckets']):
            if time.monotonic()-start>240: raise RuntimeError('STOP_RAW_NAME_AUDIT_TIME_BUDGET')
            free,level=io.pressure()
            if free<50 or level!=1: raise RuntimeError('STOP_RAW_NAME_AUDIT_PRESSURE')
            fact_ref=b['artifacts']['facts.parquet']; path=verify(fact_ref)
            epochs[str(path)]=(path.stat().st_size,path.stat().st_mtime_ns)
            facts=c.execute('''SELECT name_id,parent_key,owner_key,subject_code,scope_code
                FROM read_parquet(?) WHERE (subject_code=1 AND parent_key IN (SELECT parent_key FROM wanted_parent))
                OR (subject_code=2 AND owner_key IN (SELECT owner_key FROM wanted_site))''',[str(path)]).fetchall()
            verified.append(fact_ref)
            if facts:
                name_ref=b['artifacts']['names.parquet']; names_path=verify(name_ref)
                epochs[str(names_path)]=(names_path.stat().st_size,names_path.stat().st_mtime_ns)
                c.register('wanted_name',pa.table({'name_id':pa.array(sorted({r[0] for r in facts}),type=pa.uint32())}))
                names=dict(c.execute('SELECT name_id,name_text FROM read_parquet(?) WHERE name_id IN (SELECT name_id FROM wanted_name)',
                    [str(names_path)]).fetchall())
                verified.append(name_ref)
                for nid,parent,owner,subject,scope in facts:
                    if subject==1 and scope in (1,2): by_parent[parent].add(names[nid])
                    elif subject==2 and scope==3: by_site[owner].add(names[nid])
                    else: raise RuntimeError('Official name scope drift')
            io.write(out/'state.json',dict(status='SCANNING_OFFICIAL_NAMES',buckets=i+1,of=64,
                elapsed_seconds=time.monotonic()-start))
    finally: c.close()
    # Each consumed file was fully hashed before reading; verify epochs afterwards.
    for path,epoch in epochs.items():
        p=Path(path)
        if (p.stat().st_size,p.stat().st_mtime_ns)!=epoch: raise RuntimeError('Official source changed')
    result=[]
    for r in rows:
        parent_names=sorted(by_parent[int(r['official_siren'])]); site_names=sorted(by_site[int(r['official_siret'])])
        names=sorted(set(parent_names)|set(site_names))
        indexed={a for v in r['official_names'].values() for a in v['aliases']}
        new=sorted(set(names)-indexed)
        similarities=name_evidence(r['crm']['crm_name'],{'legal_names':names})['legal_names']
        raw_similarity=similarities['max_character_similarity']; raw_coverage=similarities['max_query_token_coverage']
        flags=[]
        if raw_similarity>=.8 and r['name_character_similarity']<.5:
            flags.append('STRONG_PRE_NATIVE_NAME_WEAK_INDEXED_NAME')
        if raw_similarity<.5 and raw_coverage<.5:
            flags.append('WEAK_NAME_EVEN_IN_OFFICIAL_DICTIONARY_HEURISTIC')
        result.append(dict(query_id=r['query_id'],exact_hit=r['exact_hit'],crm_name=r['crm']['crm_name'],
            parent_names=parent_names,site_names=site_names,pre_native_names_not_in_native_stored_fields=new,
            indexed_name_similarity=r['name_character_similarity'],pre_native_name_similarity=raw_similarity,
            pre_native_query_token_coverage=raw_coverage,diagnostic_flags=flags))
    miss=[r for r in result if not r['exact_hit']]
    summary=dict(status='COMPLETE_OFFICIAL_PRE_NATIVE_NAME_DIAGNOSTIC_NOT_CAUSAL_ATTRIBUTION',
        misses=len(miss),controls=len(result)-len(miss),whole_dictionary_names=dictionary['names'],
        whole_dictionary_facts=dictionary['facts'],buckets_scanned=64,
        miss_flags=Counter(f for r in miss for f in r['diagnostic_flags']),
        misses_with_extra_pre_native_names=sum(bool(r['pre_native_names_not_in_native_stored_fields']) for r in miss),
        consumed_sources=verified,dictionary=DICTIONARY,prior_diagnostic=record(SOURCE/'manifest.json'),
        official_projection_scope='ALL_LEXICALLY_ELIGIBLE_V129_NAMES_IN_V130_NOT_MASKED_NONLEXICAL_ROWS',
        raw_original_source_observation_rows_not_audited=True,serving_changed=False,recall_changed=False,
        fold0_opened=False,fold1_opened=False,downstream_models_changed=False,
        elapsed_seconds=time.monotonic()-start)
    io.write(out/'rows.json',result); io.write(out/'summary.json',summary)
    io.write(out/'manifest.json',dict(summary=record(out/'summary.json'),rows=record(out/'rows.json')))
    print(json.dumps({k:summary[k] for k in ('misses','miss_flags','misses_with_extra_pre_native_names','elapsed_seconds')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if not a.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated postseal namespace')
    run(a.output)
