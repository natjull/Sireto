"""Post-seal label-side source coverage, never generation or scoring inputs."""
from pathlib import Path
import time
import numpy as np
from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = io.HOME/'v145/candidate_own_address_source_20261014_night'
NAMES = io.HOME/'v101/postseal_name_roles_20261013_a'
STATUS = io.HOME/'v103/raw_alias_coverage_20261013_a'
PIPELINE = io.HOME/'v140/structural_native_disk_20261013_night'
OUT = io.HOME/'v145/target_coverage_diagnostic_20261014_night'


def main():
    if OUT.exists():
        raise ValueError('Fresh label-side diagnostic namespace')
    manifest = io.read(SOURCE/'manifest.json')
    sx = io.read(verify(manifest['own_exit']))
    if manifest['status'] != 'READY_HASHED_SOURCE_ONLY_FOR_LATER_DIAGNOSTIC_PROJECTION_NOT_RECALL' or sx['exit_code'] or not sx['own_child_exit_confirmed']:
        raise ValueError('Actual complete query-only source before label audit')
    generation = io.read(verify(manifest['generation']))
    keys = np.load(verify(generation['artifacts']['keys.npy']))
    nr, sr = io.read(NAMES/'report.json'), io.read(STATUS/'report.json')
    labels = {r['query_id']: r for r in io.read(verify(nr['cases']))}
    statuses = {r['qid']: r for r in io.read(verify(sr['statuses']))}
    ex = io.read(PIPELINE/'owned_tool_exit.json')
    if ex['exit_code'] or not ex['own_tool_exit_confirmed']:
        raise ValueError('Actual complete ALL526 structural pipeline')
    launch = io.read(PIPELINE/'launch.json')
    panel = {q['query_id']: q for q in io.read(verify(launch['panel']))}
    if len(panel) != 526 or set(panel) != set(labels) or set(panel) != set(statuses) or any(q['fold'] not in (2, 3, 4) for q in panel.values()):
        raise ValueError('Only same current526 labels, no closed populations')
    outcomes = {r['query_id']: r for r in io.read(PIPELINE/'structural_sources_max/outcomes.json')['learned_native_ooc']}
    if set(outcomes) != set(panel):
        raise ValueError('Same complete outcome population')
    cases = []
    for qid, q in panel.items():
        s = statuses[qid]
        protected = s['site_diffusion'] != 'O' or s['owner_diffusion'] != 'O'
        row = dict(query_id=qid, fold=q['fold'], already_exact=outcomes[qid]['exact'],
            protected_target_not_looked_up=protected, source_key_member=None if protected else False)
        if not protected:
            label = labels[qid]
            if label['query'] != q:
                raise ValueError('Original immutable query/label-side authority')
            target = int(label['label_siret'])
            at = int(np.searchsorted(keys, target))
            row['source_key_member'] = bool(at < len(keys) and keys[at] == target)
        cases.append(row)
    remaining = [r for r in cases if not r['already_exact']]
    if len(remaining) != 20 or sum(r['protected_target_not_looked_up'] for r in remaining) != 2:
        raise ValueError('Exhaustive20 including protected2 unchanged')
    summary = dict(all526=len(cases), source_rows=len(keys),
        public_targets=sum(not r['protected_target_not_looked_up'] for r in cases),
        protected_targets_skipped=sum(r['protected_target_not_looked_up'] for r in cases),
        public_targets_in_source=sum(r['source_key_member'] is True for r in cases),
        public_current_hits_in_source=sum(r['source_key_member'] is True and r['already_exact'] for r in cases),
        remaining20=len(remaining), remaining_public18=sum(not r['protected_target_not_looked_up'] for r in remaining),
        remaining_public_targets_in_source=sum(r['source_key_member'] is True for r in remaining),
        remaining_protected_skipped=sum(r['protected_target_not_looked_up'] for r in remaining))
    OUT.mkdir(parents=True)
    io.write(OUT/'all526_coverage.json', cases)
    io.write(OUT/'remaining20_coverage.json', remaining)
    io.write(OUT/'report.json', dict(status='COMPLETE_POSTSEAL_SOURCE_TARGET_COVERAGE_NOT_RECALL',
        implementation=record(Path(__file__)), source_manifest=record(SOURCE/'manifest.json'), source_exit=manifest['own_exit'],
        existing_label_side_names=record(NAMES/'report.json'), public_diffusion_status=record(STATUS/'report.json'),
        structural_owned_exit=record(PIPELINE/'owned_tool_exit.json'), all526=record(OUT/'all526_coverage.json'),
        remaining20=record(OUT/'remaining20_coverage.json'), summary=summary,
        no_target_ids_or_aliases_given_to_source_or_generation=True, no_score_or_native_query=True,
        no_label_or_denominator_change=True, closed_sets_opened=False,
        membership_is_not_positive_score_or_recall_gain=True,
        no_generalization_claim_from20_or526_to_source_population=True, observed_unix=time.time()))
    print(summary, flush=True)


if __name__ == '__main__':
    main()
