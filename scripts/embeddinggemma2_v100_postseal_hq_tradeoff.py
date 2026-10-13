"""Post-seal attribution of HQ gains/losses, not retrieval/training inputs."""
from pathlib import Path
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v32_learned_query as labels_source
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v100/postseal_hq_tradeoff_20261013_a'
ROOTS={'old':io.HOME/'v81/indirect_lexical_query_20261013_a',
    'hq':io.HOME/'v95/parent_headquarters_context_20261013_a'}


def main():
    import duckdb
    import pyarrow as pa
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh postseal audit')
    source={}
    for name,p in ROOTS.items():
        g=io.read(p/'generation.json');ex=io.read(p/'generate_exit.json')
        if g['status']!='COMPLETE_SEALED_ALL526' or ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned whole generation before labels')
        cal=io.read(p/'calibration/generation.json')
        source[name]=dict(generation=record(p/'generation.json'),owned_exit=record(p/'generate_exit.json'),
            outputs=io.read(verify(g['outputs']))['learned_native_ooc'],
            outcomes={r['query_id']:r for r in io.read(p/'outcomes.json')['learned_native_ooc']},
            features=io.read(p/'features_generation.json'),assignment=io.read(verify(cal['assignment'])),
            models={k:io.read(verify(v)) for k,v in cal['models'].items()})
    labels,controls,label_refs=labels_source.calibration_sources();labels={r['query_id']:r for r in labels}
    scope=set(source['old']['outcomes'])
    if len(scope)!=526 or scope!=set(source['hq']['outcomes']):raise ValueError('Same526')
    panel={q['query_id']:q for q in io.read(verify(io.read(ROOTS['old']/'launch.json')['panel']))}
    refs=sorted({c['siret'] for s in source.values() for cs in s['outputs'].values() for c in cs})
    c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='1GB'")
    c.register('wanted',pa.table({'siret':refs}))
    rows=c.execute('SELECT e.siret,e.etablissementSiege FROM read_parquet(?) e JOIN wanted w USING(siret)',[str(mandate.STOCK)]).fetchall();c.close()
    flags=dict(rows);result=[];concentration={}
    for name,s in source.items():
        proportions=[sum(bool(flags.get(v['siret'])) for v in cs)/max(1,len(cs)) for cs in s['outputs'].values()]
        concentration[name]=dict(mean_headquarters_share=float(np.mean(proportions)),
            median_headquarters_share=float(np.median(proportions)))
    for qid in sorted(scope):
        a=source['old']['outcomes'][qid];b=source['hq']['outcomes'][qid]
        if a['exact']==b['exact']:continue
        mandate.guard(mandate.DEADLINE);gt=int(labels[qid]['siret']);detail={}
        if labels[qid]['fold'] not in (2,3,4):raise ValueError('Closed human fold')
        for name,s in source.items():
            with np.load(verify(s['features']['features'][qid]),allow_pickle=False) as f:
                at=int(np.searchsorted(f['keys'],gt))
                if at>=len(f['keys']) or f['keys'][at]!=gt:raise ValueError('Diagnostic GT full scope')
                x=f['features'][at].copy()
            m=s['models'][str(s['assignment'][qid])];num=np.array(m['coefficients'][-len(x):]);cs=s['outputs'][qid]
            at=next((i+1 for i,v in enumerate(cs) if int(v['siret'])==gt),None)
            detail[name]=dict(exact=s['outcomes'][qid]['exact'],gt_numerical_score=float(x@num),
                gt_headquarters_bonus=float(x[-5:]@num[-5:]) if name=='hq' else 0.,
                gt_headquarters_flag=float(x[-5]) if name=='hq' else None,
                gt_rank_if_selected=at,output100_cut_score=cs[-1]['score'],
                output100_headquarters=sum(bool(flags.get(v['siret'])) for v in cs),
                actor_coefficient=m['coefficients'][6],
                gt_source_name=float(x[2]),gt_own_geo=float(x[5]),gt_parent_geo=float(x[16]))
        result.append(dict(query_id=qid,query=panel[qid],label_siret=str(gt).zfill(14),
            kind='GAIN' if b['exact'] else 'LOSS',variants=detail,
            exact_headquarters_doc_not_implied_by_entity_match=True,label_side_only=True))
    OUTPUT.mkdir(parents=True);io.write(OUTPUT/'cases.json',result)
    io.write(OUTPUT/'report.json',dict(status='COMPLETE_HQ_TRADEOFF_ATTRIBUTION_NOT_RETRIEVAL',
        implementation=record(Path(__file__)),cases=record(OUTPUT/'cases.json'),label_sources=label_refs,
        sources={k:dict(generation=s['generation'],owned_exit=s['owned_exit']) for k,s in source.items()},
        all526_denominator_retained=True,headquarters_concentration=concentration,
        no_active_v99_outcomes_read=True,no_model_or_query_changed=True,no_new_recall=True,
        numeric_bonus_not_whole_lexical_score=True,observed_unix=time.time()))
    print(concentration,flush=True)


if __name__=='__main__':main()
