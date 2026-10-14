"""Postseal ALL526 source coverage ONLY, not retrieval or source generation.

Human owner refs are read AFTER full direct source generation/ownedexit0.
No target enters source confidence, no alias/label/score/pipeline changes.
"""
from collections import Counter, defaultdict
from pathlib import Path
import math
import time

import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as words
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE=io.HOME/'v154/direct_corporate_activity_20261014_night_d'
BASE=io.HOME/'v115/streamed_activity_ablation_20261013_a'
NAMES=io.HOME/'v101/postseal_name_roles_20261013_a'
STATUS=io.HOME/'v103/raw_alias_coverage_20261013_a'
STRUCT=io.HOME/'v140/structural_native_disk_20261013_night'
OUT=io.HOME/'v154/source_coverage_diagnostic_20261014_night_b'
SOURCE_SHA='db060d006a81a5a733088c6930f0c755bf8eb7d10211e0a5ca83315bf95fd4c9'


def guard():
    import psutil
    free,level=io.pressure()
    if time.time()>=1791954368 or psutil.Process().memory_info().rss>1024**3 or free<25 or level not in (0,1,2):
        raise RuntimeError('Light postseal audit yields; source and recipes preserved')


def confidence(ref,owner):
    with np.load(verify(ref),allow_pickle=False) as f:
        needle=np.bytes_(owner);at=int(np.searchsorted(f['refs'],needle))
        return float(f['confidence'][at]) if at<len(f['refs']) and f['refs'][at]==needle else 0.


def main():
    import duckdb
    import pyarrow as pa
    started=time.monotonic();guard()
    if OUT.exists():raise ValueError('Fresh label-side namespace')
    if record(SOURCE/'generation.json')['sha256']!=SOURCE_SHA:
        raise ValueError('Exact whole sealed source before label read')
    g=io.read(SOURCE/'generation.json');sx=io.read(SOURCE/'owned_exit.json');tx=io.read(SOURCE/'owned_tool_exit.json')
    if g['status']!='COMPLETE_SEALED_WHOLE_SOURCE_DIRECT_CORPORATE_ACTIVITY_ALL526' or sx['exit_code'] or tx['exit_code'] or not sx['own_child_exit_confirmed'] or not tx['own_tool_exit_confirmed']:
        raise ValueError('Actual child and tool exit0, no partial source diagnosis')
    launch=io.read(verify(g['launch']));panel=io.read(verify(g['query_panel']))
    if len(panel)!=526 or any(q['fold'] not in (2,3,4) for q in panel):raise ValueError('Only same526 safe folds')
    fg=io.read(BASE/'features_generation.json');bl=io.read(BASE/'launch.json')
    if bl['panel']!=g['query_panel']:raise ValueError('Exact query binding')
    labels={r['query_id']:r for r in io.read(verify(io.read(NAMES/'report.json')['cases']))}
    statuses={r['qid']:r for r in io.read(verify(io.read(STATUS/'report.json')['statuses']))}
    outcomes={r['query_id']:r for r in io.read(STRUCT/'structural_sources_max/outcomes.json')['learned_native_ooc']}
    stx=io.read(STRUCT/'owned_tool_exit.json')
    if stx['exit_code'] or not stx['own_tool_exit_confirmed'] or set(labels)!=set(statuses) or set(labels)!=set(outcomes) or set(labels)!={q['query_id'] for q in panel}:
        raise ValueError('Whole existing consumed structural outcomes, no closed populations')
    cal=io.read(BASE/'calibration/generation.json')
    controls=set(io.read(verify(cal['encoder_control_queries'])))
    public=[dict(siret=labels[q['query_id']]['label_siret'],siren=labels[q['query_id']]['label_siret'][:9]) for q in panel
            if statuses[q['query_id']]['site_diffusion']=='O' and statuses[q['query_id']]['owner_diffusion']=='O']
    # Only codes/administrative fields on both-public source subjects/sites;
    # no name/address profile hydration or protected reconstruction.
    con=duckdb.connect(str(SOURCE/'aggregate.duckdb'),read_only=True)
    con.execute('SET threads=1');con.execute("SET memory_limit='128MB'")
    con.register('public_targets',pa.Table.from_pylist(public))
    stock=verify(launch['public_units']);site=verify(bl['stock_source'])
    rows=con.execute("""SELECT DISTINCT p.siret,p.siren,u.categorieJuridiqueUniteLegale legal_category,
        e.codeCommuneEtablissement insee,e.codePostalEtablissement postcode
        FROM public_targets p JOIN read_parquet(?) u USING(siren) JOIN read_parquet(?) e USING(siret)
        WHERE u.statutDiffusionUniteLegale='O' AND e.statutDiffusionEtablissement='O'""",[str(stock),str(site)]).fetchall()
    admin={s:dict(siren=o,legal_category=str(c) if c is not None else None,insee=i,postcode=p) for s,o,c,i,p in rows}
    eligible={r['siren'] for r in admin.values() if r['legal_category'] is not None and r['legal_category']!='1000'}
    con.register('selected_owners',pa.table({'subject':sorted(eligible)}))
    available={s for s, in con.execute('SELECT DISTINCT a.subject FROM all_owners a JOIN selected_owners w USING(subject)').fetchall()}
    token_owner=defaultdict(set)
    for token,subject in con.execute('SELECT t.token,t.subject FROM token_owner t JOIN selected_owners w USING(subject)').fetchall():
        token_owner[subject].add(token)
    pubs=defaultdict(dict)
    provenance=verify(g['provenance'])
    for token,subject,pid,date,edition,nt,nf in con.execute('SELECT p.* FROM read_parquet(?) p JOIN selected_owners w USING(subject)',[str(provenance)]).fetchall():
        item=pubs[subject].setdefault((pid,str(date)),dict(publication_id=pid,publication_date=str(date),publication_edition=edition,
            notice_type=nt,notice_family=nf,tokens=set()))
        item['tokens'].add(token)
    con.close();guard()
    keys=np.load(verify(bl['keys']),mmap_mode='r');xy=np.load(verify(bl['xy']),mmap_mode='r')
    geometry={r['query_id']:r['geometry'] for r in io.read(verify(bl['geometry']))}
    cases=[]
    for q in panel:
        qid=q['query_id'];status=statuses[qid];old_label=labels[qid]
        if old_label['query']!=q:raise ValueError('Unchanged original query/label-side alignment')
        protected=status['site_diffusion']!='O' or status['owner_diffusion']!='O'
        row=dict(query_id=qid,fold=q['fold'],zone=outcomes[qid]['zone'],role=outcomes[qid]['role'],
            prior_v140_exact=outcomes[qid]['exact'],prior_v140_siren=outcomes[qid]['siren'],prior_v140_operational=outcomes[qid]['operational'],
            encoder_checkpoint_control_query=qid in controls,protected_target_skipped=protected,
            new_confidence=None,old_confidence=None,public_corporate_eligible=False,
            label_and_denominator_unchanged=True,source_diagnostic_not_generation_input=True)
        if protected:
            cases.append(row);continue
        target=old_label['label_siret'];owner=target[:9];a=admin.get(target)
        if a is None:raise ValueError('Both-public administrative source unexpectedly absent')
        if a['legal_category'] is None:
            row.update(public_unknown_category_excluded_by_frozen_corporate_source=True)
            cases.append(row);continue
        if a['legal_category']=='1000':
            row.update(public_EI_excluded_by_frozen_corporate_source=True)
            cases.append(row);continue
        old=confidence(fg['activity_shards'][qid],owner);new=confidence(g['activity_shards'][qid],owner)
        ts=words.tokens(q['crm_name']);shared=ts & token_owner[owner]
        if bool(new>0)!=bool(shared):raise ValueError('Whole-source confidence matches same-owner informative tokens')
        support=[]
        for item in pubs[owner].values():
            overlap=sorted(ts & item['tokens'])
            if overlap:
                support.append(dict({k:v for k,v in item.items() if k!='tokens'},shared_informative_query_tokens=overlap,
                    complete_informative_query_in_same_publication=bool(ts and ts<=item['tokens']),
                    activity_field='listepersonnes.personne.activite',subject_role='SAME_EXPLICIT_PUBLIC_CORPORATE_PERSONNE_OBJECT',
                    historical_owner_context_not_name_or_site_identity=True))
        at=int(np.searchsorted(keys,np.uint64(target)));present=at<len(keys) and keys[at]==np.uint64(target)
        point=xy[at] if present else np.array([np.nan,np.nan])
        usable=bool(np.isfinite(point).all() and 0<=point[0]<=1300000 and 6000000<=point[1]<=7300000)
        gate=bool((q['crm_insee'] and q['crm_insee']==a['insee']) or (q['crm_postcode'] and q['crm_postcode']==a['postcode']))
        category=('NEW_DIRECT_POSITIVE_OLD_CONTEXT_ABSENT' if new>0 and old==0 else
            'BOTH_POSITIVE_NO_INDEPENDENCE_GUARANTEE' if new>0 and old>0 else
            'OLD_ONLY_POSITIVE' if old>0 else 'NEITHER_QUERY_TOKEN_CONTEXT')
        row.update(query=q,label_siret=target,public_corporate_eligible=True,
            direct_activity_owner_available_in_whole_corpus=owner in available,
            new_confidence=new,old_confidence=old,old_weighted_token_coverage=2*old,
            old_exact_shared_tokens_not_exported_by_original_shards=True,
            old_weighted_coverage_full=math.isclose(old,.5,rel_tol=0,abs_tol=1e-12),
            shared_informative_query_tokens=sorted(shared),complete_informative_tokens_union=bool(ts and ts<=shared),
            direct_complete_in_same_notice=any(p['complete_informative_query_in_same_publication'] for p in support),
            supporting_publication_observations=support,source_relation_category=category,
            new_confidence_above_old=new>old+1e-12,original_own_xy_usable=usable,
            frozen_query_crm_geometry_compatible=geometry[qid]['spatial_crs_compatible'],
            original_admin_gate=gate,activity_has_no_xy_gate=True,
            source_presence_or_confidence_not_target_score_or_recall_gain=True)
        cases.append(row)
    del keys,xy;guard()
    eligible_rows=[r for r in cases if r['public_corporate_eligible']]
    remaining=[r for r in cases if not r['prior_v140_exact']]
    def counts(rows):
        e=[r for r in rows if r['public_corporate_eligible']]
        return dict(queries=len(rows),eligible_public_corporate=len(e),protected_skipped=sum(r['protected_target_skipped'] for r in rows),
            public_EI_skipped=sum(r.get('public_EI_excluded_by_frozen_corporate_source',False) for r in rows),
            direct_owner_available=sum(r['direct_activity_owner_available_in_whole_corpus'] for r in e),
            direct_positive=sum(r['new_confidence']>0 for r in e),old_positive=sum(r['old_confidence']>0 for r in e),
            direct_positive_old_zero=sum(r['new_confidence']>0 and r['old_confidence']==0 for r in e),
            both_positive=sum(r['new_confidence']>0 and r['old_confidence']>0 for r in e),
            direct_complete_token_union=sum(r['complete_informative_tokens_union'] for r in e),
            direct_complete_same_notice=sum(r['direct_complete_in_same_notice'] for r in e),
            direct_positive_no_own_xy=sum(r['new_confidence']>0 and not r['original_own_xy_usable'] for r in e),
            direct_positive_admin_false=sum(r['new_confidence']>0 and not r['original_admin_gate'] for r in e),
            direct_positive_encoder_checkpoint_control=sum(r['new_confidence']>0 and r['encoder_checkpoint_control_query'] for r in e),
            source_relation_categories=dict(Counter(r['source_relation_category'] for r in e)))
    segments={field:{str(value):counts([r for r in cases if r[field]==value]) for value in {r[field] for r in cases}}
              for field in ('fold','zone','role','encoder_checkpoint_control_query')}
    OUT.mkdir(parents=True);io.write(OUT/'all526_source_coverage.json',cases);io.write(OUT/'remaining20_source_coverage.json',remaining)
    io.write(OUT/'report.json',dict(status='COMPLETE_POSTSEAL_ALL526_DIRECT_ACTIVITY_SOURCE_COVERAGE_NOT_RECALL',
        implementation=record(Path(__file__)),direct_source_generation=record(SOURCE/'generation.json'),
        direct_source_actual_child_exit=record(SOURCE/'owned_exit.json'),direct_source_actual_tool_exit=record(SOURCE/'owned_tool_exit.json'),
        old_source_generation=fg['activity_generation'],old_shards_binding=record(BASE/'features_generation.json'),
        existing_label_side=record(NAMES/'report.json'),diffusion_statuses=record(STATUS/'report.json'),
        consumed_structural_owned_exit=record(STRUCT/'owned_tool_exit.json'),encoder_control_ids=cal['encoder_control_queries'],
        all526=record(OUT/'all526_source_coverage.json'),remaining20=record(OUT/'remaining20_source_coverage.json'),
        summary=counts(cases),remaining_summary=counts(remaining),segments=segments,
        human_owner_lookup_performed_only_after_source_sealed_and_exited=True,
        no_target_given_to_source_or_retrieval=True,no_alias_or_label_or_denominator_change=True,
        no_native_or_counterfactual_retrieval_or_model_fit=True,closed_sets_opened=False,
        full_encoder_oof=False,development_consumed_not_independent_validation=True,
        source_confidence_not_probability_of_identity=True,not_a_recall_gain=True,
        old_exact_token_share_unknown_only_weighted_coverage_reported=True,
        elapsed_seconds=time.monotonic()-started,observed_unix=time.time()))
    print(dict(all526=counts(cases),remaining20=counts(remaining)),flush=True)


if __name__=='__main__':main()
