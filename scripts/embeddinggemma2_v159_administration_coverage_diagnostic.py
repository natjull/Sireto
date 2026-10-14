"""Postseal same ALL526 corporate administration coverage, never retrieval.

Both source grammars and whole outputs/exits frozen before existing labels.
Public/EI/protected classification is reused unchanged, not reinvented here.
Corporate actor mentions are context, not names of or identity of the site.
"""
from collections import Counter
from pathlib import Path
import math
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as words
from scripts.embeddinggemma2_v154_direct_activity_coverage_diagnostic import confidence,guard
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_v20_workbench import record,verify

SOURCES={'V158B':io.HOME/'v158/corporate_administration_20261014_night_b',
         'V159':io.HOME/'v159/parenthetical_corporate_administration_20261014_night'}
SHAS={'V158B':'8265910a9970c12c78f515b059fbe31e9ce4b0130e2d3f7f28eb0ac6e0d818de',
      'V159':'48b4e796b76c03663338e5a52d04988062609575332cb8df5a63a78aea560e96'}
PRIOR=io.HOME/'v154/source_coverage_diagnostic_20261014_night_b'
BASE=io.HOME/'v115/streamed_activity_ablation_20261013_a'
CURRENT=io.HOME/'v142/bano_native_20261013_night'
RNE=io.HOME/'v57/corporate_site_nullsafe_20261013_a'
OUT=io.HOME/'v159/coverage_diagnostic_20261014_night'


def main():
    import duckdb
    import pyarrow as pa
    import ijson
    start=time.monotonic();guard()
    if OUT.exists():raise ValueError('Fresh paired source-only diagnosis')
    gens={};launches={};refs={}
    # ALL source/execution gates precede every human label/outcome lookup.
    for key,root in SOURCES.items():
        ref=record(root/'generation.json');g=io.read(root/'generation.json')
        ex=io.read(root/'owned_exit.json');tx=io.read(root/'owned_tool_exit.json')
        if ref['sha256']!=SHAS[key] or g['status']!='COMPLETE_WHOLE19YEAR_CORPORATE_ADMINISTRATION_SOURCE_NOT_RECALL' or g['query_count']!=526 or len(g['coverage'])!=19 or ex['exit_code'] or tx['exit_code'] or not ex['own_child_exit_confirmed'] or not tx['own_tool_exit_confirmed']:
            raise ValueError('Both complete source seals and actual owned exits0')
        gens[key]=g;launches[key]=io.read(verify(g['launch']));refs[key]=dict(generation=ref,child_exit=record(root/'owned_exit.json'),tool_exit=record(root/'owned_tool_exit.json'))
    if launches['V158B']['query_panel']!=launches['V159']['query_panel'] or launches['V158B']['public_units']!=launches['V159']['public_units']:
        raise ValueError('Identical source population and eligibility')
    panel=io.read(verify(gens['V159']['query_panel']))
    if len(panel)!=526 or any(q['fold'] not in (2,3,4) for q in panel):raise ValueError('Same safe ALL526')
    for variant in ('bano_geo_only','all_new_sources_max_bano'):
        g=io.read(CURRENT/variant/'generation.json');ex=io.read(CURRENT/variant/'generate_exit.json')
        if g['status']!='COMPLETE_SEALED_ALL526' or g['queries']!=526 or ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Both current national outputs already complete')
    cx=io.read(CURRENT/'owned_tool_exit.json')
    if cx['exit_code'] or not cx['own_tool_exit_confirmed']:raise ValueError('Current family actual owned tool exit0')
    previous_report=io.read(PRIOR/'report.json');px=io.read(PRIOR/'owned_tool_exit.json')
    if px['exit_code'] or not px['own_tool_exit_confirmed']:raise ValueError('Unchanged sealed public classification')
    prior=io.read(verify(previous_report['all526']));old={r['query_id']:r for r in prior}
    before={r['query_id']:r for r in io.read(BASE/'outcomes.json')['learned_native_ooc']}
    current={r['query_id']:r for r in io.read(CURRENT/'all_new_sources_max_bano/outcomes.json')['learned_native_ooc']}
    if set(old)!=set(before) or set(old)!=set(current) or set(old)!={q['query_id'] for q in panel}:
        raise ValueError('Identical ALL526 denominator without manual labels')
    eligible={r['label_siret'][:9] for r in prior if r['public_corporate_eligible']}
    # Stream admitted RNE corporate names, never read giant activity JSON.
    rneg=io.read(RNE/'features_generation.json');rneref=rneg['corporate_names'];rnenames={}
    with verify(rneref).open('rb') as stream:
        for subject,groups in ijson.kvitems(stream,''):
            if subject in eligible:rnenames[subject]=sorted({n for values in groups.values() for n in values})
    cal=io.read(BASE/'calibration/generation.json');assignment=io.read(verify(cal['assignment']))
    coef={int(f):float(io.read(verify(ref))['coefficients'][6]) for f,ref in cal['models'].items()}
    if any(v<0 or not math.isfinite(v) for v in coef.values()):raise ValueError('Frozen corporate coefficient')
    source_info={}
    for key,root in SOURCES.items():
        guard();g=gens[key];c=duckdb.connect(str(root/'source.duckdb'),read_only=True);c.execute('SET threads=1');c.execute("SET memory_limit='128MB'")
        c.register('selected',pa.table({'subject':sorted(eligible)}))
        cols='r.subject,r.actor,r.actor_name,r.role,'+('r.qualification_origin,' if key=='V159' else '')+'r.already_rne_pair,d.degree'
        groups={s:[] for s in eligible}
        for row in c.execute('SELECT '+cols+' FROM relations r JOIN selected s USING(subject) JOIN degrees d USING(actor) ORDER BY r.subject,r.actor,r.actor_name,r.role').fetchall():
            if key=='V159':subject,actor,name,role,origin,known,degree=row
            else:subject,actor,name,role,known,degree=row;origin='PREFIX_OR_SUFFIX_LEGAL_FORM_ONLY'
            groups[subject].append(dict(actor_siren_reference=actor,corporate_actor_name=name,published_role=role,qualification_origin=origin,
                already_rne_subject_actor_pair=bool(known),global_source_actor_subject_degree=degree,exact_unique_current_public_corporate_name_resolution=True))
        publications={s:{} for s in eligible}
        cols='p.subject,p.actor,p.actor_name,p.role,'+('p.qualification_origin,' if key=='V159' else '')+'p.publication_id,p.publication_date,p.edition,p.notice_type,p.notice_family'
        for row in c.execute('SELECT '+cols+' FROM read_parquet(?) p JOIN selected s USING(subject) ORDER BY p.subject,p.publication_date,p.publication_id',[str(verify(g['provenance']))]).fetchall():
            if key=='V159':subject,actor,name,role,origin,pid,date,edition,nt,nf=row
            else:subject,actor,name,role,pid,date,edition,nt,nf=row;origin='PREFIX_OR_SUFFIX_LEGAL_FORM_ONLY'
            publications[subject].setdefault((actor,name,role,origin),[]).append(dict(publication_id=pid,publication_date=str(date),edition=edition,
                notice_type=nt,notice_family=nf,source_field='listepersonnes.personne.administration',publication_not_effective_date=True,
                directly_registered_public_pm_subject=True,not_a_current_role_ownership_or_operator_claim=True))
        c.close();source_info[key]=(groups,publications)
    guard();cases=[];bits=(1<<20)-1
    for q in panel:
        qid=q['query_id'];fact=old[qid]
        row={k:fact[k] for k in ('query_id','fold','zone','role','prior_v140_exact','prior_v140_siren','prior_v140_operational','encoder_checkpoint_control_query','protected_target_skipped','public_corporate_eligible')}
        row.update(prior_v115_exact=before[qid]['exact'],current_v142_exact=current[qid]['exact'],current_v142_operational=current[qid]['operational'],
            source_scope_exclusion_not_pipeline_denominator_filter=True,source_only_no_native_or_label_change=True,source_versions={})
        if not fact['public_corporate_eligible']:
            row['public_EI_skipped']=fact.get('public_EI_excluded_by_frozen_corporate_source',False);cases.append(row);continue
        owner=fact['label_siret'][:9];qt=words.tokens(q['crm_name']);whole=normalized(q['crm_name']);known_names=rnenames.get(owner,[])
        row.update(label_siret_reference=fact['label_siret'],original_admin_gate=fact['original_admin_gate'],original_own_xy_usable=fact['original_own_xy_usable'],
            frozen_corporate_coefficient6=coef[assignment[qid]],internal_calibration_assignment_is_not_closed_human_fold=True,
            current_rne_admitted_actor_name_available=bool(known_names),current_rne_admitted_actor_any_query_overlap=any(qt&words.tokens(n) for n in known_names),
            current_rne_admitted_actor_complete_query=any(qt and qt<=words.tokens(n) for n in known_names),
            current_rne_names_not_exported_no_new_actor_protected_profile_lookup=True)
        for key in SOURCES:
            gen=gens[key];groups,pubs=source_info[key];entries=[]
            raw=confidence(gen['raw_shards'][qid],owner);novel=confidence(gen['novel_shards'][qid],owner)
            for item in groups[owner]:
                shared=sorted(qt&words.tokens(item['corporate_actor_name']))
                if not shared:continue
                relation=(item['actor_siren_reference'],item['corporate_actor_name'],item['published_role'],item['qualification_origin'])
                entries.append(dict(item,shared_informative_query_tokens=shared,complete_informative_query_in_same_actor_name=bool(qt and qt<=words.tokens(item['corporate_actor_name'])),
                    exact_whole_normalized_query_equals_actor_name=bool(whole and whole==item['corporate_actor_name']),publication_provenance=pubs[owner].get(relation,[])))
            if bool(raw>0)!=bool(entries) or bool(novel>0)!=any(not e['already_rne_subject_actor_pair'] for e in entries):
                raise ValueError('Sealed owner source confidence versus typed corporate evidence')
            row['source_versions'][key]=dict(owner_relations_available=bool(groups[owner]),raw_confidence=raw,novel_absent_rne_pair_confidence=novel,
                positive_corporate_actor_mentions=entries,complete_same_actor_query=any(e['complete_informative_query_in_same_actor_name'] for e in entries),
                novel_complete_same_actor_query=any(not e['already_rne_subject_actor_pair'] and e['complete_informative_query_in_same_actor_name'] for e in entries),
                exact_whole_same_actor_query=any(e['exact_whole_normalized_query_equals_actor_name'] for e in entries),
                novel_exact_whole_same_actor_query=any(not e['already_rne_subject_actor_pair'] and e['exact_whole_normalized_query_equals_actor_name'] for e in entries),
                potential_frozen_w6_admin_branch_bonus_if_added=coef[assignment[qid]]*(math.floor(novel*bits)/bits) if fact['original_admin_gate'] else 0.,
                diagnostic_unadded_bonus_not_current_pipeline_score_rank_or_probability=True)
        cases.append(row)
    def summary(rows):
        e=[r for r in rows if r['public_corporate_eligible']];result=dict(all_rows=len(rows),public_corporate=len(e),
            protected_skipped=sum(r['protected_target_skipped'] for r in rows),public_EI_skipped=sum(r.get('public_EI_skipped',False) for r in rows),
            existing_rne_actor_any_query_overlap=sum(r['current_rne_admitted_actor_any_query_overlap'] for r in e),sources={})
        for key in SOURCES:
            values=[r['source_versions'][key] for r in e]
            result['sources'][key]=dict(owner_relations_available=sum(v['owner_relations_available'] for v in values),raw_positive=sum(v['raw_confidence']>0 for v in values),
                novel_absent_rne_pair_positive=sum(v['novel_absent_rne_pair_confidence']>0 for v in values),raw_complete_same_actor_query=sum(v['complete_same_actor_query'] for v in values),
                novel_complete_same_actor_query=sum(v['novel_complete_same_actor_query'] for v in values),raw_exact_whole_same_actor_query=sum(v['exact_whole_same_actor_query'] for v in values),
                novel_exact_whole_same_actor_query=sum(v['novel_exact_whole_same_actor_query'] for v in values),novel_positive_admin_true=sum(r['source_versions'][key]['novel_absent_rne_pair_confidence']>0 and r['original_admin_gate'] for r in e),
                novel_positive_without_own_xy=sum(r['source_versions'][key]['novel_absent_rne_pair_confidence']>0 and not r['original_own_xy_usable'] for r in e),
                novel_positive_encoder_control=sum(r['source_versions'][key]['novel_absent_rne_pair_confidence']>0 and r['encoder_checkpoint_control_query'] for r in e),
                novel_positive_no_existing_rne_query_overlap=sum(r['source_versions'][key]['novel_absent_rne_pair_confidence']>0 and not r['current_rne_admitted_actor_any_query_overlap'] for r in e))
        return result
    misses={label:[r for r in cases if not r[field]] for label,field in [('original_v115_22','prior_v115_exact'),('v140_20','prior_v140_exact'),('current_v142_19','current_v142_exact')]}
    if {k:len(v) for k,v in misses.items()}!={'original_v115_22':22,'v140_20':20,'current_v142_19':19}:raise ValueError('Unchanged audited miss populations')
    OUT.mkdir(parents=True);io.write(OUT/'all526_source_coverage.json',cases)
    for key,rows in misses.items():io.write(OUT/(key+'_source_coverage.json'),rows)
    io.write(OUT/'report.json',dict(status='COMPLETE_PAIRED_ALL526_POSTSEAL_CORPORATE_ADMINISTRATION_SOURCE_COVERAGE_NOT_RECALL',implementation=record(Path(__file__)),
        source_authorities=refs,current_owned_exit=record(CURRENT/'owned_tool_exit.json'),current_outcomes=record(CURRENT/'all_new_sources_max_bano/outcomes.json'),
        public_classification=record(PRIOR/'report.json'),existing_rne_context_names=rneref,frozen_calibration=record(BASE/'calibration/generation.json'),
        all526=record(OUT/'all526_source_coverage.json'),all526_summary=summary(cases),miss_summaries={k:summary(v) for k,v in misses.items()},
        segments={f:{str(v):summary([r for r in cases if r[f]==v]) for v in {r[f] for r in cases}} for f in ('fold','zone','role','encoder_checkpoint_control_query')},
        sources_frozen_before_labels=True,no_parser_selection_or_label_correction=True,no_native_or_fit_or_counterfactual_rank=True,
        no_full_encoder_oof=True,consumed_development_only=True,source_confidence_not_identity_probability=True,
        source_role_text_not_structured_verified_ownership_or_operator_identity=True,public_alias_resolution_not_asof_certified=True,
        no_physical_director_names_addresses_or_protected_profiles_exported=True,closed_sets_opened=False,elapsed_seconds=time.monotonic()-start))
    print(dict(all526=summary(cases),misses={k:summary(v) for k,v in misses.items()}),flush=True)


if __name__=='__main__':main()
