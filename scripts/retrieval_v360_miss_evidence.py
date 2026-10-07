"""Post-seal diagnostic of official evidence, never a serving candidate source."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

from scripts import evaluate_retrieval_v358_postseal as ev
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.hierarchical_retrieval import TantivyBackend,normalize_text
from src.xgb_matcher.retrieval_v20_workbench import record,verify

EVALUATION=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v358_postseal_evaluation/run_20261006_a')
OUTPUT=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v360_miss_evidence')


def name_evidence(query, doc):
    query=normalize_text(query); tokens=set(query.split())
    result={}
    for field in ('legal_names','trade_names','site_names','historical_names','supporting_names'):
        aliases=[normalize_text(x) for x in doc.get(field,[]) if str(x).strip()]
        result[field]=dict(aliases=aliases,
            max_character_similarity=max((SequenceMatcher(None,query,a).ratio() for a in aliases),default=0.),
            max_query_token_coverage=max((len(tokens&set(a.split()))/len(tokens) for a in aliases),default=0.) if tokens else 0.)
    return result


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    # Revalidate ALL500 own producers and their sealed source/exit/PASS first.
    permission,permission_ref=ev.permitted(EVALUATION)
    config=ev.config()
    labels,_,_,target_refs,_,_=ev.targets(config,permission)
    final=io.read(EVALUATION/'manifest.json'); report=io.read(verify(final['report']))
    if report['primary']['queries']!=2000 or report['primary']['exact_hits']!=1823:
        raise ValueError('Original complete measured population required')
    label_map={x['query_id']:x for x in labels}
    generation=io.read(verify(final['generation']))
    emitted=io.read(verify(generation['outputs']))['rows']
    candidates={r['query_id']:r for r in emitted}
    panel=io.read(io.PANEL)
    queries={r['query_id']:r for r in panel['queries']}; geo={r['query_id']:r for r in panel['geo_queries']}
    misses=[r for r in report['outcomes'] if not r['exact_hit']]
    controls=sorted((r for r in report['outcomes'] if r['exact_hit']),
        key=lambda r:__import__('hashlib').sha256(r['query_id'].encode()).digest())[:64]
    scope=misses+controls
    ids=sorted({label_map[r['query_id']]['human_ground_truth_siret'] for r in scope})
    backend=TantivyBackend(io.INDEX); tv=backend._tantivy; docs={}
    for start in range(0,len(ids),64):
        part=ids[start:start+64]
        query=tv.Query.term_set_query(backend.index.schema,'siret',part)
        for _,loc in backend.searcher.search(query,limit=len(part)+1,count=False).hits:
            doc=backend.searcher.doc(loc).to_dict(); docs[backend._first(doc,'siret')]=doc
    if set(docs)!=set(ids): raise ValueError('Do not silently drop missing official documents')
    # Point lookups in the entire official arrays, not nearest-GT candidate generation.
    cfg=io.read(io.ROOT/'config/retrieval_v187_staged_initial.json')
    structured=io.read(verify(cfg['structured']))
    catalog=io.read(verify(structured['sources']['catalog']))
    keys_ref=catalog['input_arrays']['endpoint_keys']; xy_ref=structured['reused_arrays']['site_xy']
    keys=np.load(keys_ref['path'],mmap_mode='r',allow_pickle=False)
    xy=np.load(xy_ref['path'],mmap_mode='r',allow_pickle=False)
    if keys.dtype!=np.uint64 or xy.shape!=(len(keys),2): raise ValueError('Official XY namespace drift')
    results=[]
    try:
        for outcome in scope:
            qid=outcome['query_id']; label=label_map[qid]; q=queries[qid]; g=geo[qid]
            siret=label['human_ground_truth_siret']; siren=label['human_ground_truth_siren']; doc=docs[siret]
            returned=candidates[qid]['results']['primary']; trace=candidates[qid]['primary_trace']
            names=name_evidence(q['crm_name'],doc)
            p=int(np.searchsorted(keys,np.uint64(int(siret)))); distance=None
            compatible=g.get('spatial_crs_compatible',False)
            if p<len(keys) and int(keys[p])==int(siret) and compatible and np.isfinite(xy[p]).all():
                distance=float(np.linalg.norm(xy[p]-np.array([g['ban_x_l93'],g['ban_y_l93']])))
            insee=backend._first(doc,'insee'); postcode=backend._first(doc,'postcode')
            name_similarity=max(v['max_character_similarity'] for v in names.values())
            name_coverage=max(v['max_query_token_coverage'] for v in names.values())
            same_parent=[r for r in returned if r['siren']==siren]
            addresses=[str(a) for f in ('addresses','historical_addresses','supporting_addresses') for a in doc.get(f,[])]
            address=' '.join(q[k] for k in ('crm_number','crm_suffix','crm_street'))
            similarity=max((SequenceMatcher(None,normalize_text(address),normalize_text(a)).ratio()
                for a in addresses),default=0.)
            flags=[]
            if same_parent and not outcome['exact_hit']: flags.append('PARENT_FOUND_SITE_MISSED')
            if not same_parent and not outcome['exact_hit']: flags.append('PARENT_NOT_IN_OUTPUT')
            if name_similarity<.5 and name_coverage<.5: flags.append('WEAK_INDEXED_NAME_SIMILARITY_HEURISTIC')
            if name_similarity>=.8: flags.append('STRONG_INDEXED_NAME_SIMILARITY_HEURISTIC')
            if insee!=q['crm_insee'] and postcode!=q['crm_postcode']: flags.append('ADMINISTRATIVE_LOCATION_DIFFERS')
            if not trace['geometry'].get('compatible'): flags.append('GEO_CHANNEL_UNAVAILABLE')
            if distance is not None and distance<=300: flags.append('OFFICIAL_GT_WITHIN_300M')
            if distance is not None and distance>2000: flags.append('OFFICIAL_GT_BEYOND_2KM')
            results.append(dict(query_id=qid,fold=q['fold'],exact_hit=outcome['exact_hit'],
                diagnostic_flags=flags,crm=q,official_siret=siret,official_siren=siren,
                official_names=names,official_addresses=addresses,official_insee=insee,official_postcode=postcode,
                name_character_similarity=name_similarity,name_token_coverage=name_coverage,
                address_character_similarity=similarity,ban_gt_distance_m=distance,
                ban_resolved=g['ban_resolved'],geometry=trace['geometry'],name_owners=trace['names'],
                same_parent_output_ranks=[r['retrieval_rank'] for r in same_parent],
                top100_min_score=returned[-1]['source_score'] if returned else None,
                emitted_same_parent_sirets=[r['siret'] for r in same_parent]))
    finally: keys._mmap.close(); xy._mmap.close()
    missing=[r for r in results if not r['exact_hit']]; hit=[r for r in results if r['exact_hit']]
    summary=dict(status='POSTSEAL_OFFICIAL_EVIDENCE_DIAGNOSTIC_NOT_CAUSAL_ATTRIBUTION',
        population=2000,misses=len(missing),hit_diagnostic_controls=len(hit),
        miss_flags=Counter(f for r in missing for f in r['diagnostic_flags']),
        hit_control_flags=Counter(f for r in hit for f in r['diagnostic_flags']),
        exact_recall=report['primary']['exact_recall'],metric_changed=False,
        source_report=final['report'],generation=final['generation'],permission=permission_ref,
        target_refs=target_refs,coordinate_refs=[keys_ref,xy_ref],
        raw_source_proof_before_indexing_not_audited=True,gt_native_score_not_measured=True,
        heuristic_flags_overlap=True,fold0_opened=False,fold1_opened=False,downstream_models_changed=False)
    io.write(out/'rows.json',results); io.write(out/'summary.json',summary)
    summary['artifacts']={'rows':record(out/'rows.json'),'summary':record(out/'summary.json')}
    io.write(out/'manifest.json',summary)
    print(json.dumps({k:summary[k] for k in ('population','misses','miss_flags','hit_control_flags')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if not a.output.resolve().is_relative_to(OUTPUT): raise ValueError('Dedicated postseal namespace')
    run(a.output)
