"""Postseal public truth-owner coverage of actual SIGLE, not recall.

Reuses already sealed ALL526 public/EL/EI/control/admin/XY classification.
No source constructor receives labels; no native query, alias or label change.
"""
from collections import Counter
from pathlib import Path
import time

from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as words
from scripts.embeddinggemma2_v154_direct_activity_coverage_diagnostic import confidence, guard
from src.xgb_matcher.retrieval_variants_v118 import normalized
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE=io.HOME/'v156/bodacc_sigle_20261014_night'
PRIOR=io.HOME/'v154/source_coverage_diagnostic_20261014_night_b'
OUT=io.HOME/'v156/coverage_diagnostic_20261014_night'
SOURCE_SHA='a7c72a6dc1b5508067b37504e1441b64f65185bbe1d1801e108ae5daae5d3be8'


def main():
    import duckdb
    import pyarrow as pa
    started=time.monotonic();guard()
    if OUT.exists():raise ValueError('Fresh postseal source audit namespace')
    if record(SOURCE/'generation.json')['sha256']!=SOURCE_SHA:raise ValueError('Exact complete whole source seal')
    gen=io.read(SOURCE/'generation.json');ex=io.read(SOURCE/'owned_exit.json');tx=io.read(SOURCE/'owned_tool_exit.json')
    if gen['status']!='COMPLETE_SEALED_WHOLE19YEAR_PUBLIC_CORPORATE_SIGLE_SOURCE_ALL526' or ex['exit_code'] or tx['exit_code'] or not ex['own_child_exit_confirmed'] or not tx['own_tool_exit_confirmed']:
        raise ValueError('Actual whole-source child/tool exits0 before label audit')
    launch=io.read(verify(gen['launch']));panel=io.read(verify(gen['query_panel']))
    prior=io.read(PRIOR/'report.json');pex=io.read(PRIOR/'owned_tool_exit.json')
    if pex['exit_code'] or not pex['own_tool_exit_confirmed'] or prior['status']!='COMPLETE_POSTSEAL_ALL526_DIRECT_ACTIVITY_SOURCE_COVERAGE_NOT_RECALL':
        raise ValueError('Complete corrected existing public classification')
    previous=io.read(verify(prior['all526']));old_gen=io.read(verify(prior['direct_source_generation']));old_launch=io.read(verify(old_gen['launch']))
    if launch['public_units']!=old_launch['public_units'] or old_gen['query_panel']!=gen['query_panel'] or len(previous)!=526 or len(panel)!=526 or any(q['fold'] not in (2,3,4) for q in panel):
        raise ValueError('Exact public eligibility/input binding, no closed human group')
    old={r['query_id']:r for r in previous}
    eligible={r['label_siret'][:9] for r in previous if r['public_corporate_eligible']}
    con=duckdb.connect(str(SOURCE/'source.duckdb'),read_only=True);con.execute('SET threads=1');con.execute("SET memory_limit='128MB'")
    con.register('selected',pa.table({'subject':sorted(eligible)}))
    pairs={s:[] for s in eligible}
    for subject,name,known,current in con.execute('SELECT n.* FROM read_parquet(?) n JOIN selected s USING(subject)',[str(verify(gen['owner_sigles']))]).fetchall():
        pairs[subject].append(dict(name_text=name,already_admitted_dictionary=known,same_current_sirene_sigle=current,novel=not known and not current))
    pubs={s:{} for s in eligible}
    for subject,name,pid,date,edition,nt,nf in con.execute('SELECT p.* FROM read_parquet(?) p JOIN selected s USING(subject)',[str(verify(gen['provenance']))]).fetchall():
        pubs[subject].setdefault(name,[]).append(dict(publication_id=pid,publication_date=str(date),publication_edition=edition,
            notice_type=nt,notice_family=nf,source_field='listepersonnes.personne.sigle',
            same_public_corporate_personne_object_registration=True,publication_not_effective_date=True))
    con.close();guard();cases=[]
    for q in panel:
        qid=q['query_id'];fact=old[qid]
        row={k:fact[k] for k in ('query_id','fold','zone','role','prior_v140_exact','prior_v140_siren','prior_v140_operational',
            'encoder_checkpoint_control_query','protected_target_skipped','public_corporate_eligible')}
        row.update(no_label_or_denominator_change=True,no_target_given_to_source_or_native=True,
            raw_sigle_confidence=None,novel_sigle_confidence=None,raw_owner_sigles_available=False,
            source_parent_name_not_siret_site_or_recall=True)
        if not fact['public_corporate_eligible']:
            row.update(public_EI_skipped=fact.get('public_EI_excluded_by_frozen_corporate_source',False))
            cases.append(row);continue
        target=fact['label_siret'];owner=target[:9];ts=words.tokens(q['crm_name']);whole=normalized(q['crm_name'])
        raw=confidence(gen['raw_shards'][qid],owner);novel=confidence(gen['novel_shards'][qid],owner)
        aliases=[]
        for alias in pairs[owner]:
            overlap=sorted(ts & words.tokens(alias['name_text']))
            aliases.append(dict(alias,shared_informative_query_tokens=overlap,
                complete_informative_query_same_sigle=bool(ts and ts<=words.tokens(alias['name_text'])),
                exact_whole_normalized_query_name=bool(whole and whole==alias['name_text']),
                publication_observations=pubs[owner].get(alias['name_text'],[])))
        if bool(raw>0)!=any(r['shared_informative_query_tokens'] for r in aliases) or bool(novel>0)!=any(r['novel'] and r['shared_informative_query_tokens'] for r in aliases):
            raise ValueError('Exact presealed source confidence versus published same-owner aliases')
        row.update(query=q,label_siret=target,raw_sigle_confidence=raw,novel_sigle_confidence=novel,
            raw_owner_sigles_available=bool(aliases),novel_owner_sigles_available=any(r['novel'] for r in aliases),
            source_sigles=aliases,raw_complete_informative_query=any(r['complete_informative_query_same_sigle'] for r in aliases),
            novel_complete_informative_query=any(r['novel'] and r['complete_informative_query_same_sigle'] for r in aliases),
            raw_exact_whole_normalized_name=any(r['exact_whole_normalized_query_name'] for r in aliases),
            novel_exact_whole_normalized_name=any(r['novel'] and r['exact_whole_normalized_query_name'] for r in aliases),
            matched_only_already_admitted=raw>0 and novel==0,
            original_admin_gate=fact['original_admin_gate'],original_own_xy_usable=fact['original_own_xy_usable'],
            frozen_query_crm_geometry_compatible=fact['frozen_query_crm_geometry_compatible'],
            max_one_alias_not_synthetic_cross_notice_word_union=True,
            raw_new_confidence_not_a_score_or_probability=True,legacy_dictionary_dedup_not_asof_certification=True)
        cases.append(row)
    def summary(rows):
        public=[r for r in rows if r['public_corporate_eligible']]
        return dict(queries=len(rows),public_corporate=len(public),protected_skipped=sum(r['protected_target_skipped'] for r in rows),
            public_EI_skipped=sum(r.get('public_EI_skipped',False) for r in rows),
            raw_owner_has_sigle=sum(r['raw_owner_sigles_available'] for r in public),novel_owner_has_sigle=sum(r['novel_owner_sigles_available'] for r in public),
            raw_positive_informative_context=sum(r['raw_sigle_confidence']>0 for r in public),
            novel_positive_informative_context=sum(r['novel_sigle_confidence']>0 for r in public),
            matched_only_existing_names=sum(r['matched_only_already_admitted'] for r in public),
            raw_complete_same_sigle=sum(r['raw_complete_informative_query'] for r in public),
            novel_complete_same_sigle=sum(r['novel_complete_informative_query'] for r in public),
            raw_exact_whole_name=sum(r['raw_exact_whole_normalized_name'] for r in public),
            novel_exact_whole_name=sum(r['novel_exact_whole_normalized_name'] for r in public),
            novel_positive_prior_v140_hits=sum(r['novel_sigle_confidence']>0 and r['prior_v140_exact'] for r in public),
            novel_positive_encoder_controls=sum(r['novel_sigle_confidence']>0 and r['encoder_checkpoint_control_query'] for r in public),
            novel_positive_admin_false=sum(r['novel_sigle_confidence']>0 and not r['original_admin_gate'] for r in public),
            novel_positive_no_own_xy=sum(r['novel_sigle_confidence']>0 and not r['original_own_xy_usable'] for r in public))
    remaining=[r for r in cases if not r['prior_v140_exact']]
    OUT.mkdir(parents=True);io.write(OUT/'all526_source_coverage.json',cases);io.write(OUT/'remaining20_source_coverage.json',remaining)
    segments={field:{str(value):summary([r for r in cases if r[field]==value]) for value in {r[field] for r in cases}}
        for field in ('fold','zone','role','encoder_checkpoint_control_query')}
    io.write(OUT/'report.json',dict(status='COMPLETE_POSTSEAL_ALL526_ACTUAL_SIGLE_NOVELTY_COVERAGE_NOT_RECALL',
        implementation=record(Path(__file__)),source_generation=record(SOURCE/'generation.json'),
        source_actual_child_exit=record(SOURCE/'owned_exit.json'),source_actual_tool_exit=record(SOURCE/'owned_tool_exit.json'),
        existing_corrected_label_public_classification=record(PRIOR/'report.json'),existing_classification_actual_exit=record(PRIOR/'owned_tool_exit.json'),
        all526=record(OUT/'all526_source_coverage.json'),remaining20=record(OUT/'remaining20_source_coverage.json'),
        summary=summary(cases),remaining_summary=summary(remaining),segments=segments,
        no_labels_given_to_source_before_whole19year_seal_and_actual_exit0=True,
        no_native_or_counterfactual_retrieval_or_fit_or_index_or_alias_or_label_change=True,
        all526_denominator_and_protected_EI_rows_preserved=True,closed_sets_opened=False,
        actual_published_sigle_not_generated_initialism=True,legacy_dedup_not_asof_guarantee=True,
        exact_whole_name_separate_from_informative_confidence=True,
        full_encoder_oof=False,consumed_development_not_independent_validation=True,
        source_new_alias_coverage_does_not_prove_recall_gain=True,elapsed_seconds=time.monotonic()-started,observed_unix=time.time()))
    print(dict(all526=summary(cases),remaining20=summary(remaining)),flush=True)


if __name__=='__main__':main()
