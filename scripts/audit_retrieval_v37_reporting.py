"""Audit frozen v35-v37 outcomes; never train, select candidates or open fold 0/1."""
import argparse
from collections import Counter
import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts import retrieval_v35_site_fusion as v35
from src.xgb_matcher.retrieval_v20_workbench import record, verify, write_json


ROOT = v35.DATA / 'audits'
AUTHORITIES = {
    'v35': (ROOT / 'retrieval_v35_site_fusion/pilot_20261001_a/manifest.json',
            '8000f1f8851d28af788708007b14c7fc2e2e5735d7fa8bbd34f46c27ca37112a'),
    'v36': (ROOT / 'retrieval_v36_evidence_ranker/recovery_20261002_a/manifest.json',
            '97abf5e886068ccd878304f8f44ddec4efe470a5ddd7798949c9a9e602e471db'),
    'v37': (ROOT / 'retrieval_v37_parent_evidence/pilot_20261002_a/manifest.json',
            'f6386d054f7b68186f6e7f8550efe7052548b8f942efbda949e6427655e7e018'),
}
SEGMENTS = ('fold', 'label_kind', 'qualification_v2', 'qualification_v3',
            'data_origin', 'ground_truth_state', 'unseen_siren',
            'new_site_known_siren', 'mega_base_pool')


def segmented(rows, labels):
    ids = [r['query_id'] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(labels):
        raise ValueError('Outcome/label population mismatch')
    merged = []
    for row in rows:
        label = labels[row['query_id']]
        if row['fold'] not in (2, 3, 4) or row['fold'] != label['fold']:
            raise ValueError('Forbidden or inconsistent fold')
        for column in ('label_kind', 'qualification_v2', 'qualification_v3'):
            if row[column] != label[column]:
                raise ValueError('Audit metadata drift')
        merged.append({**row, **{c: label[c] for c in SEGMENTS}})
    result = {}
    for column in SEGMENTS:
        groups = {}
        for row in merged:
            value = row[column]
            key = 'UNKNOWN' if value is None or value == '' else str(value)
            groups.setdefault(key, []).append(row)
        result[column] = {key: v35.summarize(group) for key, group in sorted(groups.items())}
        if sum(group['queries'] for group in result[column].values()) != len(rows):
            raise ValueError('Segment denominator drift')
    return result


def run(output):
    if output.exists():
        raise ValueError('Refuse to overwrite a reporting audit')
    manifests = {name: v35.load_manifest(path, sha) for name, (path, sha) in AUTHORITIES.items()}
    refs = {'v35': manifests['v35']['outcomes'],
            'v36': manifests['v36']['primary']['outcomes'],
            'v36_training_ablation': manifests['v36']['ablation']['outcomes'],
            'v37': manifests['v37']['primary']['outcomes']}
    outcomes = {name: pq.read_table(verify(ref)).to_pylist() for name, ref in refs.items()}
    ids = {r['query_id'] for r in outcomes['v35']}
    if len(ids) != 1024:
        raise ValueError('Pilot scope drift')
    ledger = verify(dict(path=v35.LEDGER, sha256=v35.LEDGER_SHA))
    labels = {r['query_id']: r for r in pq.read_table(
        ledger, columns=['query_id', *SEGMENTS], filters=[('fold', 'in', [2, 3, 4])]
    ).to_pylist() if r['query_id'] in ids}
    reports = {}
    for name, rows in outcomes.items():
        reports[name] = dict(historical_all_human=v35.summarize(rows),
                             segments=segmented(rows, labels),
                             losses=dict(Counter(r['loss_stage'] for r in rows)))
    current = outcomes['v37']
    current_vs_heuristic = [dict(r, baseline_exact_rank=r['heuristic_exact_rank']) for r in current]
    write_json(output, dict(status='AUDITED_FROZEN_PILOT_REPORTING_ONLY', decision='PIVOT_GENERATION',
        authorities={name: record(path) for name, (path, _) in AUTHORITIES.items()},
        ledger=record(ledger), outcomes=refs, reports=reports,
        paired_v37_vs_heuristic=v35.summarize(current_vs_heuristic),
        unknown_metadata={col: sum(r[col] in (None, '') for r in labels.values()) for col in SEGMENTS},
        interpretation='V2/V3/mega_base_pool are missing in the source ledger. UNKNOWN is not a qualification.',
        fold0_opened=False, fold1_opened=False, model_fit_performed=False, production_gate=False,
        implementation=record(Path(__file__))))
    print(json.dumps(dict(output=record(output), queries=len(ids), status='PASS_REPORTING_AUDIT')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
