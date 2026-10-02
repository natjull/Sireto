"""Correct V73 alias COUNT bookkeeping; historic kernel/artifacts immutable."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts import retrieval_v38_initial_top100 as v38
from scripts.preflight_retrieval_v72_observation_cost import authority
from src.xgb_matcher.retrieval_observation_v72 import Observation
from src.xgb_matcher.retrieval_owner_observation_v73 import compose
from src.xgb_matcher.retrieval_v20_workbench import Budget, record, verify, write_json


def corrected_counts(before, after, bindings, reported):
    if len(before) != len(after):
        raise ValueError('Observation denominator drift')
    counters = dict(reported); counters['added_legal_aliases'] = 0; counters['sites_receiving_legal_alias'] = 0
    corrections = {}; removed_duplicates = 0
    for old, new in zip(before, after):
        if old.evidence_id != new.evidence_id or old.siret != new.siret:
            raise ValueError('Observation alignment drift')
        old_set, new_set = set(old.names), set(new.names)
        if not old_set <= new_set:
            raise ValueError('Original name proof removed')
        added = len(new_set-old_set)
        if old.subject_kind == 'SIRET':
            counters['added_legal_aliases'] += added
            counters['sites_receiving_legal_alias'] += bool(added)
            removed_duplicates += (len(old.names)-len(old_set))-(len(new.names)-len(new_set))
            corrections[(old.evidence_id, old.source_record_id)] = added
    fixed = [dict(b, legal_aliases_added=corrections[(b['site_evidence_id'], b['site_source_record_id'])]) for b in bindings]
    return counters, fixed, removed_duplicates


def run(output, source):
    output.mkdir(parents=True, exist_ok=False)
    with Budget(dict(max_seconds=80, max_rss_bytes=4294967296, max_swapout_pages_delta=0)) as budget:
        m = json.loads(verify(source).read_text()); launch = json.loads(verify(m['launch']).read_text())
        v38.verify_launch(launch); verify(m['metrics']); verify(m['bindings'])
        if m['truth_opened'] or m['recall_measured']:
            raise ValueError('Not the no-label structural probe')
        spec, fingerprint = launch['source_spec'], launch['source_authority']
        if authority(spec) != fingerprint:
            raise ValueError('Source changed')
        rows = []
        for batch in pq.ParquetFile(fingerprint['path']).iter_batches(batch_size=4096):
            rows.extend(batch.slice(0, min(len(batch), launch['config']['rows']-len(rows))).to_pylist())
            if len(rows) == launch['config']['rows']:
                break
        before = [Observation.from_row(r) for r in rows]
        after, bindings, reported = compose(rows)
        original = json.loads(verify(m['metrics']).read_text())
        if reported != original['counts'] or bindings != json.loads(verify(m['bindings']).read_text()):
            raise ValueError('Historic results not reproduced')
        counts, fixed, removed = corrected_counts(before, after, bindings, reported)
        if authority(spec) != fingerprint:
            raise ValueError('Source changed during correction audit')
    write_json(output/'corrected_bindings.json', fixed)
    write_json(output/'manifest.json', dict(status='SEALED_ALIAS_COUNT_ERRATUM_NO_KERNEL_CHANGE', source=source,
        implementation=record(Path(__file__)), corrected_bindings=record(output/'corrected_bindings.json'),
        reported_counts=reported, corrected_counts=counts, duplicate_alias_entries_collapsed=removed,
        reason='TUPLE_LENGTH_DIFFERENCE_CONFUSES_ADDITION_WITH_DEDUPLICATION_USE_SET_DIFFERENCE',
        resources=budget.report(), same_name_term_sets=True, scores_unchanged_by_counter_correction=True,
        recall_measured=False, truth_opened=False, historic_artifacts_modified=False))
    print(json.dumps(dict(corrected_counts=counts, duplicate_alias_entries_collapsed=removed, resources=budget.report())))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(); run(a.output, dict(
        path='/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v73_owner_observation/probe_20261003_a/manifest.json',
        sha256='3014558abd4c309549acf15d99d13b973ff562b120c86050de4582e55ad55b84'))
