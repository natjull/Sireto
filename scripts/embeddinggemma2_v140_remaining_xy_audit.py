"""Public-only eighteen exact-coordinate lookups; never spatial retrieval."""
from pathlib import Path
import time
import numpy as np

from scripts import embeddinggemma2_night as io
from scripts.embeddinggemma2_v140_remaining_structure_audit import guard
from src.xgb_matcher.retrieval_v20_workbench import record, verify

SOURCE = io.HOME/'v140/remaining_structure_20261014_night'
PIPELINE = io.HOME/'v140/structural_native_disk_20261013_night'
OUT = io.HOME/'v140/remaining_xy_20261014_night'


def main():
    guard()
    if OUT.exists():
        raise ValueError('Fresh coordinate diagnostic namespace')
    report = io.read(SOURCE/'report.json')
    ex = io.read(SOURCE/'owned_tool_exit.json')
    if report['status'] != 'COMPLETE_ALL20_POSTSEAL_INFORMATION_REPRESENTATION_SCORE_AUDIT' or ex['exit_code'] or not ex['own_tool_exit_confirmed']:
        raise ValueError('Complete observed protected-safe postseal audit')
    cases = io.read(verify(report['cases']))
    launch = io.read(PIPELINE/'launch.json')
    keys = np.load(verify(launch['keys']), mmap_mode='r')
    xy = np.load(verify(launch['xy']), mmap_mode='r')
    if xy.shape != (len(keys), 2):
        raise ValueError('Frozen original coordinate snapshot schema')
    rows = []
    for case in cases:
        guard()
        row = dict(query_id=case['query_id'], category=case['primary_category'],
                   protected_not_looked_up=case['protected_no_reconstruction'])
        if case['protected_no_reconstruction']:
            rows.append(row)
            continue
        target = int(case['label_siret'])
        at = int(np.searchsorted(keys, target))
        present = at < len(keys) and keys[at] == target
        usable = False
        if present:
            point = np.asarray(xy[at], dtype=float)
            usable = bool(np.isfinite(point).all() and 0 <= point[0] <= 1300000 and 6000000 <= point[1] <= 7300000)
        proofs = case['target_numeric_proofs']
        row.update(query=case['query'], label_siret=case['label_siret'], key_in_coordinate_snapshot=bool(present),
            original_xy_usable_metropolitan_l93=usable, own_geo_quality=proofs['own_geo_quality'],
            filled_own_site_geo=proofs['filled_own_site_geo'], added_source_score=case['source_target_scores']['total_added'],
            no_coordinates_or_address_copied_or_reconstructed=True)
        rows.append(row)
    OUT.mkdir(parents=True)
    io.write(OUT/'cases.json', rows)
    public = [r for r in rows if not r['protected_not_looked_up']]
    io.write(OUT/'report.json', dict(status='COMPLETE_PUBLIC_ONLY_18_EXACT_XY_LOOKUPS_NOT_RETRIEVAL',
        implementation=record(Path(__file__)), original_audit=record(SOURCE/'report.json'),
        original_owned_exit=record(SOURCE/'owned_tool_exit.json'), coordinate_keys=launch['keys'], coordinates=launch['xy'],
        cases=record(OUT/'cases.json'), public_lookups=len(public), protected_skipped=len(rows)-len(public),
        usable_original_xy=sum(r['original_xy_usable_metropolitan_l93'] for r in public),
        missing_or_incompatible_original_xy=sum(not r['original_xy_usable_metropolitan_l93'] for r in public),
        no_nearest_neighbor_search_or_new_source=True, no_data_or_model_or_label_change=True,
        closed_sets_opened=False, observed_unix=time.time()))
    print('PUBLIC_EXACT_XY_DIAGNOSIS', {r['query']['crm_name']+':'+r['query_id']: r['original_xy_usable_metropolitan_l93'] for r in public}, flush=True)


if __name__ == '__main__':
    main()
