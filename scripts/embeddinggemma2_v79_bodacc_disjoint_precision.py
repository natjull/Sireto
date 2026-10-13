"""Source-only blind BODACC QA; does not tune any active retrieval recipe.

Disjoint on published parent NAME, subject ID, announcement ID and exact
normalized clause. Parent IDs/events remain unresolved, so this is NOT a
certified legal-entity/event-disjoint sample or 99% extraction certificate.
"""
import hashlib
from pathlib import Path
import time
import pyarrow.parquet as pq
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v56_corporate_site as mandate
from scripts import embeddinggemma2_v49_postseal_bodacc_roles as source
from src.xgb_matcher.retrieval_v20_workbench import record,verify

OUTPUT=io.HOME/'v79/bodacc_disjoint_precision_20261013_a'
SOURCE=io.HOME/'v65/bodacc_strict_common_parent_20261013_a'
PRIOR=[io.HOME/'v64/bodacc_parent_clauses_20261013_a/blind_precision_sample.json',
       SOURCE/'blind_precision_sample.json']


def main():
    mandate.guard(mandate.DEADLINE)
    if OUTPUT.exists():raise ValueError('Fresh source QA namespace')
    generation=io.read(SOURCE/'generation.json');ex=io.read(SOURCE/'compile_exit.json')
    if ex['exit_code'] or not ex['own_child_exit_confirmed']:raise ValueError('Owned complete source before independent QA')
    prior=[r for path in PRIOR for r in io.read(verify(record(path)))]
    parents={source.norm(r['context_parent_name']) for r in prior}
    subjects={r['subject_siren'] for r in prior};announcements={r['announcement_id'] for r in prior}
    clauses={source.norm(r['exact_clause']) for r in prior}
    facts=pq.read_table(verify(generation['candidate_facts']),use_threads=False).to_pylist()
    eligible=[r for r in facts if source.norm(r['context_parent_name']) not in parents
        and r['subject_siren'] not in subjects and r['announcement_id'] not in announcements
        and source.norm(r['exact_clause']) not in clauses]
    ordered=sorted(eligible,key=lambda r:hashlib.sha256(('bodacc-parent-subject-clause-blind-v1|'+r['announcement_id']+'|'+r['subject_siren']+'|'+r['context_parent_name']).encode()).digest())
    chosen=[];used_names=set();used_clauses=set();used_subjects=set()
    for r in ordered:
        parent=source.norm(r['context_parent_name']);clause=source.norm(r['exact_clause'])
        if parent in used_names or clause in used_clauses or r['subject_siren'] in used_subjects:continue
        chosen.append(r);used_names.add(parent);used_clauses.add(clause);used_subjects.add(r['subject_siren'])
        if len(chosen)==32:break
    if len(chosen)!=32:raise ValueError('Cannot silently reduce requested QA denominator')
    OUTPUT.mkdir(parents=True)
    io.write(OUTPUT/'sample.json',chosen)
    io.write(OUTPUT/'generation.json',dict(implementation=record(Path(__file__)),source_generation=record(SOURCE/'generation.json'),
        source_facts=generation['candidate_facts'],prior_samples=[record(p) for p in PRIOR],sample=record(OUTPUT/'sample.json'),
        source_facts_total=len(facts),eligible_facts=len(eligible),eligible_published_parent_names=len({source.norm(r['context_parent_name']) for r in eligible}),
        reviewed_sample_size=32,sample_not_yet_annotated=True,parent_name_subject_announcement_clause_disjoint=True,
        parent_ids_unresolved_not_certified_entity_disjoint=True,events_unresolved_not_certified_event_disjoint=True,
        labels_read=False,crm_read=False,no_active_recipe_or_candidate_changed=True,observed_unix=time.time()))


if __name__=='__main__':main()
