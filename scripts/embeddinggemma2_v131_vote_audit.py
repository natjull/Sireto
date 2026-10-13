"""Read-only exact source-vote decomposition on already sealed output100.

Not a reranker, candidate generator or recall experiment. All526 outputs are
audited, no target reference is used in the national stock lookup. Multiple
source votes are not automatically errors: this only measures their overlap.
"""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import math
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
from scripts.embeddinggemma2_v131_mechanism_audit import HOME, BASE, read, write, record, verify

OUTPUT=HOME/'v131/source_vote_overlap_20261013_night'
STOP=set('DE DU DES LA LE LES ET EN AU AUX A D L SA SAS SASU SARL SCI EURL STE SOCIETE ETS ETABLISSEMENTS GROUPE GROUP SERVICES SERVICE CENTRE COMMUNE COMMUNAUTE MAIRIE INSTITUT'.split())


def tokens(value):
    text=''.join(c for c in unicodedata.normalize('NFKD',str(value or '')) if not unicodedata.combining(c))
    return {t for t in re.sub('[^A-Za-z0-9]+',' ',text).upper().split()
            if len(t)>=3 and t not in STOP and not t.isdigit()}


def main(out):
    import duckdb
    import ijson
    import pyarrow as pa
    if out.exists():raise ValueError('Fresh source diagnostic namespace required')
    gen,ex=read(BASE/'generation.json'),read(BASE/'generate_exit.json')
    if gen['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:
        raise ValueError('Complete owned all526 generation first')
    outputs=read(verify(gen['outputs']))['learned_native_ooc']
    launch=read(BASE/'launch.json');panel=read(verify(launch['panel']))
    if len(panel)!=526 or set(outputs)!={q['query_id'] for q in panel}:raise ValueError('Whole panel')
    if any(q['fold'] not in (2,3,4) for q in panel):raise ValueError('Closed human fold')
    fg=read(BASE/'features_generation.json');cal=read(BASE/'calibration/generation.json')
    assignment=read(verify(cal['assignment']));models={k:read(verify(v)) for k,v in cal['models'].items()}
    out.mkdir(parents=True);(out/'spill').mkdir()
    wanted=sorted({r['siret'] for rows in outputs.values() for r in rows})
    # Only sealed-output refs, never target-seeded source extraction. Public
    # diffusion checked BEFORE administrative address fields are returned.
    con=duckdb.connect();con.execute('SET threads=1');con.execute("SET memory_limit='512MB'")
    con.execute('SET temp_directory=?',[str(out/'spill')])
    con.register('selected',pa.table({'siret':wanted}))
    owner=Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/sources/sirene/2026-08-01/StockUniteLegale.parquet')
    admin_rows=con.execute("""SELECT e.siret,e.codeCommuneEtablissement insee,e.codePostalEtablissement postcode
       FROM read_parquet(?) e JOIN selected s USING(siret)
       JOIN read_parquet(?) u USING(siren)
       WHERE e.statutDiffusionEtablissement='O' AND u.statutDiffusionUniteLegale='O'""",
       [launch['stock_source']['path'],str(owner)]).fetch_arrow_table().to_pylist();con.close()
    admin={r['siret']:(r['insee'],r['postcode']) for r in admin_rows}
    write(out/'public_output_administrative.json',admin_rows)
    votes={qid:{r['siret']:{'actor':0.,'typed_context':0.,'initials':0.,'activity':0.}
                for r in rows if r['siret'] in admin} for qid,rows in outputs.items()}
    owners_by_query={qid:{s[:9] for s in rows} for qid,rows in votes.items()}
    def gated(qid,s):
        q=queries[qid];i,p=admin[s]
        return bool((q['crm_insee'] and q['crm_insee']==i) or (q['crm_postcode'] and q['crm_postcode']==p))
    queries={q['query_id']:q for q in panel}
    # Query source channels are streamed one object at a time, never 3.5GiB
    # activity JSON. Lossless activity shards total only45MiB on disk.
    with verify(fg['context_expansion']).open('rb') as stream:
        for qid,entries in ijson.kvitems(stream,'',use_float=True):
            scores={e['siren']:e['confidence'] for e in entries if e['siren'] in owners_by_query[qid]}
            for s,v in votes[qid].items():
                if gated(qid,s):v['typed_context']=float(scores.get(s[:9],0.))
    with verify(fg['initial_alignment']).open('rb') as stream:
        for qid,entry in ijson.kvitems(stream,'',use_float=True):
            parent={e['siren_ref']:e['confidence'] for e in entry['parent_initials'] if e['siren_ref'] in owners_by_query[qid]}
            sites={s:e['confidence'] for e in entry['own_city_initials'] for s in e['siret_refs'] if s in votes[qid]}
            for s,v in votes[qid].items():
                if gated(qid,s):v['initials']=max(float(parent.get(s[:9],0.)),float(sites.get(s,0.)))
    for qid in votes:
        with np.load(verify(fg['activity_shards'][qid]),allow_pickle=False) as f:
            scores={s.decode():float(v) for s,v in zip(f['refs'],f['confidence']) if s.decode() in owners_by_query[qid]}
        for s,v in votes[qid].items():
            if gated(qid,s):v['activity']=scores.get(s[:9],0.)
    # Exact original actor IDF uses distinct aliases, not owner-document DF.
    query_tokens={qid:tokens(q['crm_name']) for qid,q in queries.items()}
    useful=set().union(*query_tokens.values());aliases=set();relevant=defaultdict(set)
    selected_owners=set().union(*owners_by_query.values())
    with verify(fg['corporate_names']).open('rb') as stream:
        for parent,groups in ijson.kvitems(stream,'',use_float=True):
            names={n for ns in groups.values() for n in ns};aliases.update(names)
            if parent in selected_owners:relevant[parent].update(names)
    dfs=Counter()
    for name in aliases:dfs.update(tokens(name)&useful)
    n=len(aliases);del aliases;gc.collect()
    for qid,rows in votes.items():
        weights={t:math.log1p((n+.5)/(dfs[t]+.5)) for t in query_tokens[qid]};norm=sum(weights.values())
        for s,v in rows.items():
            v['actor']=max((sum(weights.get(t,0.) for t in tokens(name))/norm
                           for name in relevant.get(s[:9],())) if norm else (),default=0.)
    maximum=(1<<20)-1
    details=[]
    for q in panel:
        qid=q['query_id'];model=models[str(assignment[qid])]
        if qid not in model['heldout_queries'] or qid in model['train_queries']:raise ValueError('Heldout calibrator')
        coefficient=float(model['coefficients'][6]);rows=[]
        for rank,c in enumerate(outputs[qid],1):
            s=c['siret']
            if s not in votes[qid]:continue
            values=votes[qid][s];quantized={k:math.floor(v*maximum)/maximum for k,v in values.items()}
            total=sum(quantized.values());best=max(quantized.values(),default=0.)
            rows.append(dict(rank=rank,siret=s,native_score=c['score'],sources=values,
                frozen_context_coefficient=coefficient,context_vote_sum=coefficient*total,
                context_vote_max=coefficient*best,additive_minus_max=coefficient*(total-best),
                channels_positive=sum(v>0 for v in values.values())))
        details.append(dict(query_id=qid,query=q,rows=rows))
    flattened=[r for q in details for r in q['rows']]
    summary=dict(public_output_rows=len(flattened),unique_public_refs=len(admin),
        rows_at_least2_positive_channels=sum(r['channels_positive']>=2 for r in flattened),
        rows_additive_surplus_above_point25=sum(r['additive_minus_max']>.25 for r in flattened),
        queries_with_multi_vote=sum(any(r['channels_positive']>=2 for r in q['rows']) for q in details),
        context_surplus_quantiles=np.quantile([r['additive_minus_max'] for r in flattened],[0,.5,.9,.99,1]).tolist())
    write(out/'source_votes_all526.json',details)
    write(out/'report.json',dict(status='COMPLETE_READONLY_ALL526_SOURCE_VOTE_OVERLAP',implementation=record(Path(__file__)),
        baseline_generation=record(BASE/'generation.json'),baseline_exit=record(BASE/'generate_exit.json'),
        votes=record(out/'source_votes_all526.json'),administrative=record(out/'public_output_administrative.json'),
        source_bindings={k:fg[k] for k in ('context_expansion','initial_alignment','corporate_names')},
        summary=summary,actual_administrative_gate_applied=True,exact_frozen_coefficient=True,
        distinct_sources_not_necessarily_same_evidence=True,duplicate_error_not_proven=True,
        no_retrieval_or_fitting=True,no_recall_or_counterfactual_result=True,
        no_target_source_selection=True,protected_addresses_not_reconstructed=True,observed_unix=time.time()))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    main(p.parse_args().output)
