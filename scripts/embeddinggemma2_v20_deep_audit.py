"""Post-seal attribution ONLY: no retrieval recipe, fit or candidate generation.

Full-score diagnostics use labels intentionally, hence are NEVER a product
recall or a dataset suitable for tuning on these same queries. Closed folds
and reserved test are not opened. Historical artifacts remain immutable.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sqlite3
import time

import numpy as np

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from scripts import embeddinggemma2_v15_probability_pilot as pilot
from scripts import embeddinggemma2_v17_structural_retrieval as structural
from scripts import embeddinggemma2_v18_identity_retrieval as previous
from scripts import embeddinggemma2_v19_proof_retrieval as proof
from scripts import embeddinggemma2_v19_official_audit as official
from scripts import embeddinggemma2_v2_data as data
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = io.HOME / 'v19/proof_engine_20261011_b'
OUTPUT = io.HOME / 'v20/deep_audit_20261011_a'
FIELDS = ('site', 'identity', 'name', 'address', 'street', 'geo')


def rank(scores, keys, loc):
    value = scores[loc]
    return int(1 + np.count_nonzero(scores > value) +
               np.count_nonzero((scores == value) & (keys < keys[loc])))


def native_branches(features):
    return np.stack((features['site'], .75*features['site']+.25*features['identity'],
                     features['address'], .85*features['name']+.15*features['street']))


def normalized(values):
    return {data.canonical(v) for v in values if data.safe(v)}


def source_address(row):
    # Preserve original fields AND compare structured number/street evidence.
    n = row['street_number']; raw = row['raw_value']
    if n and not proof.address(raw)[0]:
        raw = ' '.join(x for x in (n, row['street_number_suffix'], raw) if x)
    return raw


def weighted(rows, field):
    denominator = sum(r['weight'] for r in rows)
    return sum(r['weight']*r[field] for r in rows)/denominator


def run(out):
    import duckdb
    import ijson
    import pyarrow as pa
    import pyarrow.parquet as pq
    if out.exists():
        raise ValueError('Fresh immutable audit namespace only')
    started = time.monotonic()
    state = io.read(SOURCE/'state.json')
    verify(state['report'])
    ex = io.read(SOURCE/'generate_exit.json')
    gen = io.read(SOURCE/'generation.json')
    if (state['status'] != 'COMPLETE_NATIVE_PROOF_COMPARISON' or
        ex['exit_code'] != 0 or not ex['own_child_exit_confirmed'] or
        gen['status'] != 'COMPLETE_SEALED_ALL526'):
        raise ValueError('Complete sealed generation and owned exit0 required')
    launch = io.read(SOURCE/'launch.json')
    source = io.read(verify(launch['source_launch']))
    manifest = io.read(verify(source['training_manifest']))
    membership = pilot.population.SOURCE/'validation_membership.json'
    if io.digest(membership) != manifest['split_hashes']['validation']:
        raise ValueError('Original validation label authority')
    panel = io.read(verify(source['panel']))
    scope = {r['query_id'] for r in panel}
    labels = {r['query_id']: r for r in io.read(membership) if r['query_id'] in scope}
    if len(scope) != 526 or set(labels) != scope or any(r['fold'] not in (2,3,4) for r in labels.values()):
        raise ValueError('All526, safe folds only, no reserved test')
    outputs = io.read(verify(gen['outputs']))
    old = io.read(verify(launch['baseline_outputs']))
    v18 = io.read(verify(launch['v18_outputs']))
    geo_gen = io.read(previous.GEO_SOURCE/'generation.json')
    geo = io.read(verify(geo_gen['outputs']))
    channels = dict(site=old['gemma2'], joint=v18['joint'], lexical=old['lexical'],
                    oldgeo=old['geo'], newgeo=geo['geo'], family=geo['family'],
                    native=outputs['evidence_native'])
    outcomes = io.read(SOURCE/'outcomes.json')
    meta = {r['query_id']: r for r in outcomes['v18_reference']}
    native_outcomes = {r['query_id']: r for r in outcomes['evidence_native']}
    focus = {q for q in scope if not meta[q]['exact'] or not native_outcomes[q]['exact']}
    out.mkdir(parents=True)
    rows = {}; refs = [state['report'], gen['outputs'], record(membership)]
    geometries = {r['query_id']: r['geometry'] for r in io.read(verify(launch['geo']))}
    for zone, corpus_ref in sorted(launch['corpora'].items()):
        dest = SOURCE/'zones'/(zone or 'UNKNOWN')
        b = io.read(engine.sealed(dest/'features_binding.json')) if (dest/'features_binding.json').exists() else None
        qs = [q for q in panel if q['crm_insee'] == zone]
        if not b:
            if io.read(verify(corpus_ref)) or any(outputs['evidence_native'][q['query_id']] for q in qs):
                raise ValueError('Missing complete feature binding')
            for q in qs:
                rows[q['query_id']] = dict(query_id=q['query_id'], crm=q, label=labels[q['query_id']],
                    weight=meta[q['query_id']]['calibrated_query_weight'], corpus_absent=True,
                    baseline_hit=meta[q['query_id']]['exact'],native_hit=native_outcomes[q['query_id']]['exact'],
                    six_channel_oracle=any(any(c['siret']==labels[q['query_id']]['siret'] for c in v[q['query_id']]) for k,v in channels.items() if k!='native'),
                    seven_channel_oracle=any(any(c['siret']==labels[q['query_id']]['siret'] for c in v[q['query_id']]) for v in channels.values()))
            continue
        if b['corpus'] != corpus_ref or b['recipe'] != launch['recipe'] or b['queries'] != qs or not b['full_unpruned_scores']:
            raise ValueError('Exact complete query/corpus/recipe feature binding')
        keys = np.array(b['sirets'], dtype=np.uint64)
        locs = {k: i for i,k in enumerate(b['sirets'])}
        features = {k: np.load(engine.sealed(dest/(k+'.npy')), mmap_mode='r', allow_pickle=False) for k in FIELDS}
        refs.append(record(dest/'features_binding.json'))
        wanted_docs = set()
        for i,q in enumerate(qs):
            qid = q['query_id']; label = labels[qid]; loc = locs.get(label['siret'])
            r = dict(query_id=qid, crm=q, label=label, weight=meta[qid]['calibrated_query_weight'],
                     stratum=meta[qid]['stratum'], corpus_sites=len(keys), corpus_absent=loc is None,
                     baseline_hit=meta[qid]['exact'], native_hit=native_outcomes[qid]['exact'])
            r['channel_ranks'] = {k: next((j for j,c in enumerate(values[qid],1) if c['siret']==label['siret']),None)
                                  for k,values in channels.items()}
            r['six_channel_oracle'] = any(r['channel_ranks'][k] is not None for k in channels if k!='native')
            r['seven_channel_oracle'] = any(v is not None for v in r['channel_ranks'].values())
            if loc is not None:
                f = {k: a[i] for k,a in features.items()}; branches = native_branches(f)
                native = branches.max(axis=0)+.05*f['geo']
                r['feature_ranks'] = {k: rank(v,keys,loc) for k,v in f.items()}
                r['full_ranks'] = dict(site=rank(f['site'],keys,loc),
                    joint=rank(branches[1],keys,loc), native=rank(native,keys,loc))
                for k,values in (('site',old['gemma2']),('joint',v18['joint']),('native',outputs['evidence_native'])):
                    if (r['full_ranks'][k]<=100) != any(c['siret']==label['siret'] for c in values[qid]):
                        raise ValueError('Full diagnostic rank must replay sealed membership')
                r['target_features'] = {k: float(v[loc]) for k,v in f.items()}
                r['target_dominant_branch'] = int(branches[:,loc].argmax())
                selected = [locs[c['siret']] for c in outputs['evidence_native'][qid]]
                r['native_top100_branch_counts'] = dict(Counter(str(int(v)) for v in branches[:,selected].argmax(axis=0)))
                r['native_top100_address_branch_low_name'] = int(sum(branches[:,j].argmax()==2 and f['name'][j]<.5 for j in selected))
                r['native_cutoff_score'] = float(native[selected[-1]]) if selected else None
                r['target_native_score'] = float(native[loc])
                if qid in focus:
                    wanted_docs.add(label['siret'])
                    r['competitors'] = []
                    for c in outputs['evidence_native'][qid][:5]:
                        j = locs[c['siret']]; wanted_docs.add(c['siret'])
                        r['competitors'].append(dict(siret=c['siret'],siren=c['siren'],score=c['score'],
                            dominant_branch=int(branches[:,j].argmax()),features={k:float(v[j]) for k,v in f.items()}))
            rows[qid] = r
        if wanted_docs:
            with verify(corpus_ref).open('rb') as handle:
                docs = {d['key']:d for d in ijson.items(handle,'item') if d['key'] in wanted_docs}
            for q in qs:
                r=rows[q['query_id']]
                if q['query_id'] in focus:
                    r['target_document']=docs.get(r['label']['siret'])
                    for c in r.get('competitors',[]): c['document']=docs[c['siret']]
        for a in features.values(): a._mmap.close()
        io.write(out/'state.json',dict(status='POSTSEAL_FULL_SCORE_ATTRIBUTION',queries=len(rows),elapsed_s=time.monotonic()-started))
    if set(rows)!=scope: raise ValueError('No denominator exclusions')
    # Expanded official source inspection covers ALL64 baseline/native misses.
    parents=sorted({labels[q]['siren'] for q in focus})
    con=duckdb.connect();con.execute('SET threads=1');con.execute("SET memory_limit='512MB'")
    tmp=out/'duck_tmp';tmp.mkdir();con.execute('SET temp_directory=?',[str(tmp)])
    con.register('wanted',pa.table({'siren':parents}))
    dossier=io.read(verify(official.DOSSIER));root=Path(official.DOSSIER['path']).parent
    facts={}
    for name in ('name_evidence.parquet','address_evidence.parquet'):
        ref=dict(path=str(root/name),**dossier['files'][name]);p=verify(ref);refs.append(ref)
        facts[name]=official.serial(con.execute('SELECT * FROM read_parquet(?) WHERE siren IN (SELECT siren FROM wanted)',[str(p)]).fetch_arrow_table().to_pylist())
    relation_ref=dict(path=str(root/'relations.parquet'),**dossier['files']['relations.parquet']);refs.append(relation_ref)
    relations=official.serial(con.execute('''SELECT * FROM read_parquet(?) WHERE
        (from_kind='SIREN' AND from_identifier IN (SELECT siren FROM wanted)) OR
        (to_kind='SIREN' AND to_identifier IN (SELECT siren FROM wanted))''',[str(verify(relation_ref))]).fetch_arrow_table().to_pylist())
    succ_ref=io.read(official.SUCCESSION)['source'];refs.append(succ_ref)
    successions=official.serial(con.execute('''SELECT * FROM read_parquet(?) WHERE
        substr(siretEtablissementPredecesseur,1,9) IN (SELECT siren FROM wanted) OR
        substr(siretEtablissementSuccesseur,1,9) IN (SELECT siren FROM wanted)''',[str(verify(succ_ref))]).fetch_arrow_table().to_pylist());con.close()
    io.write(out/'original_source_records.json',dict(names=facts['name_evidence.parquet'],addresses=facts['address_evidence.parquet'],relations=relations,successions=successions))
    names_by=defaultdict(list);addr_by=defaultdict(list);rel_by=defaultdict(list)
    for r in facts['name_evidence.parquet']: names_by[r['siren']].append(r)
    for r in facts['address_evidence.parquet']: addr_by[r['siren']].append(r)
    for r in relations:
        for k in ('from_identifier','to_identifier'):
            if r[k] in parents: rel_by[r[k]].append(r)
    names_db=sqlite3.connect(f"file:{verify(launch['names'])}?mode=ro&immutable=1",uri=True)
    for qid in sorted(focus):
        r=rows[qid];p=r['label']['siren'];target=r['label']['siret'];d=r.get('target_document')
        own=[n for n in names_by[p] if n['subject_kind']=='SIREN' or n['siret']==target]
        sibling=[n for n in names_by[p] if n['subject_kind']=='SIRET' and n['siret']!=target]
        name_set=normalized(n['raw_value'] for n in own)
        views=normalized((d['own_names']+d.get('extra',{}).get('legal',[])+d.get('extra',{}).get('own_site',[])) if d else [])
        missing_names=sorted(name_set-views)
        aliases=normalized(n['name'] if isinstance(n,dict) else n for n in engine.lookup(names_db,1,[p])[int(p)]|engine.lookup(names_db,2,[target])[int(target)])
        qa=proof.query_address(r['crm']);raw_addr=[a for a in addr_by[p] if a['subject_kind']=='SIRET' and a['siret']==target]
        raw_addresses=[source_address(a) for a in raw_addr]
        indexed_keys={proof.address(a) for a in d['all_addresses']} if d else set()
        direct=[s for s in successions if target in (s['siretEtablissementPredecesseur'],s['siretEtablissementSuccesseur'])]
        peers={s['siretEtablissementSuccesseur'] if s['siretEtablissementPredecesseur']==target else s['siretEtablissementPredecesseur'] for s in direct}
        natural_sites={c['siret'] for ch in channels.values() for c in ch[qid]};natural_parents={c['siren'] for ch in channels.values() for c in ch[qid]}
        neighbors={n for rel in rel_by[p] for n in (rel['from_identifier'],rel['to_identifier']) if n!=p}
        r['official_audit']=dict(own_and_parent_names=sorted(name_set),missing_from_site_view_names=missing_names,
            missing_names_present_in_owner_dictionary=sorted(set(missing_names)&aliases),
            missing_name_query_similarity=structural.similarity(structural.normalize(r['crm']['crm_name']),tuple(missing_names)),
            own_name_query_similarity=structural.similarity(structural.normalize(r['crm']['crm_name']),tuple(name_set)),
            sibling_context_query_similarity=structural.similarity(structural.normalize(r['crm']['crm_name']),tuple(normalized(n['raw_value'] for n in sibling))),
            own_addresses=raw_addr,structured_own_addresses_missing_from_index=[a for a in raw_addresses if proof.address(a) not in indexed_keys],
            query_exact_structured_address_in_raw=qa in {proof.address(a) for a in raw_addresses},
            query_exact_structured_address_in_index=qa in indexed_keys,
            direct_target_successions=direct,naturally_retrieved_direct_succession_sites=sorted(peers&natural_sites),
            naturally_retrieved_direct_succession_parents=sorted({v[:9] for v in peers}&natural_parents),
            parent_transfer_relations=rel_by[p],naturally_retrieved_transfer_parents=sorted(neighbors&natural_parents),
            geometry=geometries[qid],label_error_proven=False)
    names_db.close()
    allrows=[rows[q] for q in sorted(scope)]
    lost=[r for r in allrows if r.get('baseline_hit') and not r.get('native_hit')]
    gained=[r for r in allrows if r.get('native_hit') and not r.get('baseline_hit')]
    unresolved=[r for r in allrows if not r.get('seven_channel_oracle',False)]
    summary=dict(status='COMPLETE_POSTSEAL_DEEP_ATTRIBUTION_NOT_RETRIEVAL',all_human_queries=526,focus_queries=len(focus),
        old_hits=sum(r.get('baseline_hit',False) for r in allrows),new_hits=sum(r.get('native_hit',False) for r in allrows),
        gains=[r['query_id'] for r in gained],losses=[r['query_id'] for r in lost],
        diagnostic_only_six_channel_oracle_hits=sum(r.get('six_channel_oracle',False) for r in allrows),
        diagnostic_only_seven_channel_oracle_hits=sum(r.get('seven_channel_oracle',False) for r in allrows),
        diagnostic_only_seven_channel_oracle_weighted=weighted(allrows,'seven_channel_oracle'),
        unresolved_all_seven_channels=[r['query_id'] for r in unresolved],
        corpus_absences=[r['query_id'] for r in allrows if r['corpus_absent']],
        native_top100_dominant_branch_counts=dict(sum((Counter(r.get('native_top100_branch_counts',{})) for r in allrows),Counter())),
        lost_address_branch_low_name_top100=sum(r.get('native_top100_address_branch_low_name',0) for r in lost),
        raw_source_counts={k:len(v) for k,v in facts.items()},source_refs=refs,
        label_errors_proven=0,labels_corrected=0,denominator_changed=False,
        no_fit=True,no_retrieval_generation=True,closed_folds_opened=False,test_opened=False,
        source_coverage_bounded_to_pinned_dossier=True,oracle_is_not_top100_or_a_validation=True,
        implementation=record(Path(__file__)),elapsed_s=time.monotonic()-started)
    io.write(out/'rows.json',allrows);summary['rows']=record(out/'rows.json')
    io.write(out/'summary.json',summary);io.write(out/'state.json',dict(status=summary['status'],summary=record(out/'summary.json')))
    print({k:v for k,v in summary.items() if k in ('focus_queries','old_hits','new_hits','diagnostic_only_six_channel_oracle_hits','diagnostic_only_seven_channel_oracle_hits','diagnostic_only_seven_channel_oracle_weighted','native_top100_dominant_branch_counts','elapsed_s')},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
