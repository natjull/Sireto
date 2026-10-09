"""Post-seal diagnostic, not a retrieval variant or evaluation improvement.

Reads labels only for attribution after the COMPLETE 526-query V15 pilot.
Never edits its inputs, changes candidates, or removes any denominator row.
Optional dense attribution re-encodes only original query batches and reads
existing corpus vectors; no document encoding and no gradient steps.
"""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_compare as old
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = io.HOME / 'v14/representative_20261008_b'
OUTPUT = io.HOME / 'v16/failure_audit_20261009_a'
NAME_FIELDS = ('legal_names', 'trade_names', 'site_names', 'historical_names', 'supporting_names')
ADDRESS_FIELDS = ('addresses', 'historical_addresses', 'supporting_addresses')
GENERIC = {'DE', 'DU', 'DES', 'LA', 'LE', 'LES', 'ET', 'A', 'AU', 'AUX',
           'SA', 'SAS', 'SARL', 'SASU', 'EURL', 'SCI', 'SOCIETE'}


def tokens(value):
    value = ''.join(c for c in unicodedata.normalize('NFKD', value.upper())
                    if not unicodedata.combining(c))
    return set(re.findall(r'[A-Z0-9]+', value)) - GENERIC


def overlap(query, values):
    q = tokens(query)
    return max((len(q & tokens(v)) / len(q) for v in values), default=0.) if q else None


def fusion_trace(outputs, q, siret, weight):
    ranks = {}
    totals = Counter()
    for channel, factor in (('gemma2', 1.), ('lexical', 1.), ('geo', weight)):
        for rank, candidate in enumerate(outputs[channel][q], 1):
            key = candidate['siret']
            totals[key] += factor / (60 + rank)
            if key == siret:
                ranks[channel] = rank
    ordered = sorted(totals, key=lambda k: (-totals[k], k))
    return dict(channel_ranks=ranks, fusion_rank=ordered.index(siret)+1 if siret in totals else None,
                fusion_score=totals.get(siret), fusion_cutoff=totals[ordered[99]] if len(ordered)>=100 else None,
                diagnostic_union_size=len(totals), not_a_product_pool=True)


def coordinate_status(geometry, target_xy, radius, cutoff):
    import math
    if not geometry['spatial_crs_compatible']:
        return dict(status='QUERY_BAN_UNRESOLVED_OR_INCOMPATIBLE')
    if target_xy is None or not all(math.isfinite(v) for v in target_xy):
        return dict(status='TARGET_XY_MISSING')
    x, y = target_xy
    if not (0 <= x <= 1300000 and 6000000 <= y <= 7300000):
        return dict(status='TARGET_XY_OUTSIDE_CRS_GUARD')
    distance = math.hypot(x-geometry['ban_x_l93'], y-geometry['ban_y_l93'])
    status = 'OUTSIDE_RADIUS' if distance > radius else 'WITHIN_RADIUS_BEYOND_NEAREST100' if cutoff is not None and distance > cutoff+1e-6 else 'WITHIN_RADIUS_AT_OR_BEFORE_CUTOFF'
    return dict(status=status, target_distance_m=distance, radius_m=radius, nearest100_cutoff_m=cutoff)


def prepare():
    import ijson
    import numpy as np
    import pyarrow.parquet as pq
    from src.xgb_matcher.hierarchical_retrieval import TantivyBackend
    if OUTPUT.exists():
        raise ValueError('Preserve existing audit; use explicit separate namespace for another audit')
    state = io.read(SOURCE/'state.json')
    if state['status'] != 'COMPLETE_PROBABILITY_PILOT':
        raise ValueError('No labels before complete pilot')
    launch = engine.authority(SOURCE)
    report = io.read(verify(state['report']))
    generation = io.read(SOURCE/'generation.json')
    outputs = io.read(verify(generation['outputs']))
    panel = {q['query_id']: q for q in io.read(verify(launch['panel']))}
    outcomes = io.read(SOURCE/'outcomes.json')
    census = io.read(verify(launch['census']))
    for z in census['zones']:
        for tag in ('prepare', 'v3_official512'):
            ex = io.read(engine.zone_path(SOURCE,z['zone'])/(tag+'_exit.json'))
            if ex['exit_code'] != 0 or not ex['own_child_exit_confirmed']:
                raise ValueError('Every original owned exit0 required')
    if len(panel) != 526 or any(q['fold'] not in (2,3,4) for q in panel.values()):
        raise ValueError('Exact completed pilot scope, no closed folds')
    missed = {r['query_id']: r for r in outcomes['gemma2_lexical_geo'] if not r['exact']}
    cfg = io.read(io.ROOT/'config/retrieval_v358_postseal_evaluation.json')
    ev = io.read(verify(cfg['evaluation']))
    side = io.read(verify(ev['label_side']))
    if side['never_used_in_retrieval'] is not True: raise ValueError('Official label-side authority')
    catalog = pq.read_table(verify(side['files']['official_catalog_sidecar.parquet']),
        filters=[('query_id','in',list(missed))],use_threads=False).to_pylist()
    labels = {r['query_id']:r for r in catalog}
    if set(labels) != set(missed): raise ValueError('All missed rows retained')
    sites = pq.read_table(side['official_sites']['path'],
        filters=[('siret','in',[r['gt_siret'] for r in catalog])],use_threads=False).to_pylist()
    locations = {r['siret']:r for r in sites}
    targets = defaultdict(dict)
    for q in missed: targets[panel[q]['crm_insee']][labels[q]['gt_siret']] = True
    target_docs = {}
    corpus_refs = {}
    for zone, keys in sorted(targets.items()):
        ref = io.read(engine.zone_path(SOURCE,zone)/'documents.seal.json')
        corpus_refs[zone] = ref
        with verify(ref).open('rb') as f:
            for doc in ijson.items(f,'item'):
                if doc['key'] in keys: target_docs[doc['key']] = doc
        print(json.dumps(dict(stage='CORPUS_MEMBERSHIP_AUDIT',zone=zone,found=len(target_docs))),flush=True)
    geo_rows = {r['query_id']:r for r in io.read(verify(launch['projected_geo']))}
    geographic = launch['geographic']
    # Already verified by the original complete national generation; metadata
    # pinned here plus precise random-access samples, no new full XY scan.
    for ref in (geographic['keys'],geographic['xy']):
        if Path(ref['path']).stat().st_size != ref['size_bytes']: raise ValueError('XY size drift')
    keys = np.load(geographic['keys']['path'],mmap_mode='r',allow_pickle=False)
    xy = np.load(geographic['xy']['path'],mmap_mode='r',allow_pickle=False)
    backend = TantivyBackend(io.INDEX)
    rows = []
    for q in sorted(missed):
        crm = panel[q]; label = labels[q]; key = label['gt_siret']; doc = target_docs.get(key)
        hits = backend.searcher.search(backend._tantivy.Query.term_query(backend.index.schema,'siret',key,'basic'),limit=2)
        if len(hits.hits) != 1: raise ValueError('One official target record required')
        raw = backend.searcher.doc(hits.hits[0][1]).to_dict()
        names = list(dict.fromkeys(v for field in NAME_FIELDS for v in raw.get(field,[])))
        addresses = list(dict.fromkeys(v for field in ADDRESS_FIELDS for v in raw.get(field,[])))
        extra = doc.get('extra',{}) if doc else {}
        if doc and not extra:
            con = sqlite3.connect(f'file:{SOURCE}/names/owners.sqlite?mode=ro&immutable=1',uri=True)
            try:
                extra = dict(legal=sorted(engine.lookup(con,1,[label['gt_siren']])[int(label['gt_siren'])]),
                    own_site=sorted(engine.lookup(con,2,[key])[int(key)]))
            finally: con.close()
        all_names = list(dict.fromkeys(names+extra.get('legal',[])+extra.get('own_site',[])))
        geo = geo_rows[q]; pos=int(np.searchsorted(keys,np.uint64(key)))
        coord = [float(v) for v in xy[pos]] if pos<len(keys) and int(keys[pos])==int(key) else None
        trace = fusion_trace(outputs,q,key,geo['rrf_geo_weight'])
        actual = [r['siret'] for r in outputs['gemma2_lexical_geo'][q]]
        if (trace['fusion_rank'] is not None and trace['fusion_rank']<=100) != (key in actual):
            raise ValueError('Diagnostic fusion replay disagrees with sealed output')
        row = dict(query_id=q,crm=crm,label=label,
            attribution='FUSION_EVICTION' if trace['channel_ranks'] else 'ALL_CHANNELS_MISS_AT100',
            target_in_encoded_commune=doc is not None,official_index=raw,
            encoded_target_document=doc,official_extra_names=extra,official_location=locations.get(key),
            trace=trace,geometry=geo['geometry'],geo_radius_sites=geo['official_sites_in_radius'],
            geo=coordinate_status(geo['geometry'],coord,geo['radius_m'],
                geo['candidates'][-1]['distance_m'] if len(geo['candidates'])==100 else None),
            evidence=dict(name_token_coverage=overlap(crm['crm_name'],all_names),
                address_token_coverage=overlap(crm['crm_street'],addresses),
                official_names=all_names,official_addresses=addresses,
                names_lost_from_index_to_doc=sorted(set(names)-set(doc['own_names'])) if doc else names,
                addresses_lost_from_index_to_doc=sorted(set(addresses)-set(doc['all_addresses'])) if doc else addresses),
            label_error_claimed=False)
        rows.append(row)
    keys._mmap.close(); xy._mmap.close()
    maps = {k:{r['query_id']:r for r in v} for k,v in outcomes.items()}
    union = {q:any(maps[k][q]['exact'] for k in ('gemma2','lexical','geo')) for q in panel}
    summary = dict(status='POSTSEAL_DIAGNOSTIC_NOT_NEW_RECALL',queries=526,misses=len(rows),
        original_report=state['report'],attribution=dict(Counter(r['attribution'] for r in rows)),
        targets_in_encoded_communes=sum(r['target_in_encoded_commune'] for r in rows),
        indexed_name_loss_cases=sum(bool(r['evidence']['names_lost_from_index_to_doc']) for r in rows),
        indexed_address_loss_cases=sum(bool(r['evidence']['addresses_lost_from_index_to_doc']) for r in rows),
        geo_causes=dict(Counter(r['geo']['status'] for r in rows)),
        union_diagnostic_hits=sum(union.values()),
        union_diagnostic_weighted_ceiling=sum(maps['gemma2'][q]['calibrated_query_weight']*union[q] for q in panel)/sum(r['calibrated_query_weight'] for r in maps['gemma2'].values()),
        label_errors_proven=0,no_new_candidate_generation=True,no_document_reencoding=True,
        no_gradient_steps=True,test_opened=False,fold0_opened=False,fold1_opened=False)
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT/'cases.json',rows); io.write(OUTPUT/'summary.json',summary)
    io.write(OUTPUT/'launch.json',dict(source_launch=record(SOURCE/'launch.json'),source_report=state['report'],
        source_generation=record(SOURCE/'generation.json'),source_outputs=generation['outputs'],
        label_side=ev['label_side'],cases=record(OUTPUT/'cases.json'),corpora=corpus_refs,
        geographic=geographic,xy_full_rehash_not_repeated=True,implementation=record(Path(__file__)),
        postseal_only=True,never_used_for_retrieval=True))
    print(json.dumps(summary),flush=True)


def dense_audit():
    import numpy as np
    from scripts import embeddinggemma2_v3_fit as fit
    from scripts import embeddinggemma2_v2_data as data
    launch = engine.authority(SOURCE)
    audit = io.read(OUTPUT/'launch.json'); verify(audit['implementation'])
    rows = io.read(verify(audit['cases'])); byzone=defaultdict(list)
    for r in rows: byzone[r['crm']['crm_insee']].append(r)
    panel = io.read(verify(launch['panel']))
    cfg = io.read(verify(launch['night_config']))
    engine.guard()
    model,params,encode,inference,tokenizer = fit.load_model(cfg,verify(launch['new_adapter']))
    results=[]
    for zone, cases in sorted(byzone.items()):
        engine.guard()
        directory=engine.zone_path(SOURCE,zone); cache=directory/'v3_official512'
        # Some completed zones were reused from the original full run.
        if not cache.exists():
            cache=Path(io.read(directory/'v3_official512.seal.json')['path']).parent/'v3_official512'
        docs=io.read(verify(audit['corpora'][zone])); views=[]
        for d in docs:
            if 'site_views' not in d:
                anchor=io.clean_text(d['own_names'][:1],d['all_addresses'][:1],d['postcode'])
                d['site_views']=data.pack_atoms(data.site_atoms(d,d['extra']),tokenizer,512,anchor)[0]
            views.append([io.DOC_PROMPT+v for v in d['site_views']])
        texts=list(dict.fromkeys(v for vs in views for v in vs)); positions={v:i for i,v in enumerate(texts)}
        binding=io.read(cache/'texts_binding.json')
        if engine.text_hash(texts)!=binding['order_sha256'] or len(texts)!=binding['texts']:
            raise ValueError('Exact original full-corpus text order required')
        qs=[q for q in panel if q['crm_insee']==zone]
        vectors=inference([io.QUERY_PROMPT+old.query_text(q) for q in qs])
        matrices=[]
        for lo in range(0,len(texts),1024):
            path=engine.sealed(cache/f'vectors_{lo:08d}.npy')
            if not path: raise ValueError('All original vector shards required')
            matrices.append(np.load(path,mmap_mode='r',allow_pickle=False))
        def vector(value):
            idx=positions[value]; return matrices[idx//1024][idx%1024]
        indices={q['query_id']:i for i,q in enumerate(qs)}
        doc_indices={d['key']:i for i,d in enumerate(docs)}
        target_scores={r['query_id']:float(engine.score_views(vectors, [views[doc_indices[r['label']['gt_siret']]]],vector)[indices[r['query_id']],0]) for r in cases if r['target_in_encoded_commune']}
        ranks=Counter(); nearest={r['query_id']:[] for r in cases}
        for lo in range(0,len(docs),128):
            scores=engine.score_views(vectors,views[lo:lo+128],vector)
            for r in cases:
                q=r['query_id']
                if q not in target_scores: continue
                s=target_scores[q]; row=scores[indices[q]]
                ranks[q]+=sum(float(v)>s or (float(v)==s and d['key']<r['label']['gt_siret']) for v,d in zip(row,docs[lo:lo+128]))
                nearest[q]=old.top100(nearest[q],row,[dict(siret=d['key'],siren=d['siren']) for d in docs[lo:lo+128]])
        original=io.read(engine.sealed(directory/'v3_official512.json'))
        original={r['query_id']:r['candidates'] for r in original}
        for r in cases:
            q=r['query_id']; saved=original[q]
            agreement=[c['siret'] for c in saved]==[x[1] for x in nearest[q]]
            max_error=max((abs(c['score']-x[0]) for c,x in zip(saved,nearest[q])),default=0.)
            if not agreement or max_error>1e-6:
                raise ValueError(f'Original top100 numeric replay drift for {q}: {max_error}')
            results.append(dict(query_id=q,dense_rank=ranks[q]+1 if q in target_scores else None,
                target_score=target_scores.get(q),top100_cutoff=saved[-1]['score'] if saved else None,
                original_top100_reproduced=agreement,max_score_error=max_error,
                winning_target_view=r['encoded_target_document'].get('site_views',[]) if q in target_scores else [],
                dense_top3=[dict(siret=x[1],score=x[0],names=docs[doc_indices[x[1]]]['own_names'],
                    addresses=docs[doc_indices[x[1]]]['all_addresses']) for x in nearest[q][:3]]))
        for matrix in matrices: matrix._mmap.close()
        io.write(OUTPUT/'dense_cases.json',results)
        print(json.dumps(dict(stage='DENSE_ATTRIBUTION_ORIGINAL_TOP100_REPRODUCED',zone=zone,completed_cases=len(results))),flush=True)
        del matrices,docs,views,texts,positions,doc_indices,nearest; gc.collect()
    io.write(OUTPUT/'dense_completion.json',dict(status='COMPLETE_POSTSEAL_DENSE_ATTRIBUTION',cases=len(results),
        source_vectors_only=True,query_batches_original=True,document_reencodings=0,gradient_steps=0,
        dense_cases=record(OUTPUT/'dense_cases.json'),implementation=record(Path(__file__))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','dense'])
    args=parser.parse_args()
    (prepare if args.action=='prepare' else dense_audit)()
