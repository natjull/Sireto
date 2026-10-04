"""Post-seal V195/V196 prediction parity only; no GT and no generation input."""
import argparse
import json
from pathlib import Path

from scripts import probe_retrieval_v195_pair_initial as baseline
from scripts import probe_retrieval_v196_pair_initial as current
from src.xgb_matcher.retrieval_v20_workbench import record, verify, write_json


def compare(old_reference, new_reference):
    manifests, outputs, configs = [], [], []
    for reference, module in ((old_reference, baseline), (new_reference, current)):
        path = verify(reference)
        if path.name != 'manifest.json': raise ValueError('Exact already sealed generation manifest required')
        m = json.loads(path.read_text())
        if (m['status'] != module.STATUS or m['queries'] != 2 or m['population'] != 128
                or m['initial100_emitted'] is not True or m['complete_national_source'] is not True
                or m['truth_opened'] is not False or m['old_candidates_opened'] is not False
                or m['fold0_opened'] is not False or m['fold1_opened'] is not False
                or m['recall_measured'] is not False or m['full_query_population'] is not False
                or m['ready_for_fold1'] is not False or m['slo_validated'] is not False):
            raise ValueError('Actual complete closed-label national pair before independent parity required')
        module.admission(path.parent)  # Actual all producers/audits/ledger, NOT a self-reported status.
        c = json.loads(verify(m['launch']).read_text())['config']
        configs.append({k: v for k, v in c.items() if k != 'output_root'})
        p = json.loads(verify(m['outputs']).read_text())
        if (p['truth_opened'] is not False or p['recall_measured'] is not False
                or p['population'] != 128 or p['full_query_population'] is not False
                or [r['position'] for r in p['rows']] != [0, 1]):
            raise ValueError('Same exact own two-query outputs, never labels or partial denominator metric')
        outputs.append(p)
        manifests.append(m)
    if configs[0] != configs[1]: raise ValueError('Same sources/model/query/weights/physical plan required')
    for old, new in zip(outputs[0]['rows'], outputs[1]['rows']):
        if (any(old[k] != new[k] for k in ('position', 'query_id', 'fold', 'trace'))
                or old['candidates'] != new['candidates']
                or len(new['candidates']) > 100
                or len({r['siret'] for r in new['candidates']}) != len(new['candidates'])):
            raise ValueError('NON_PARITY: own-query candidates, scores, order, whole-world traces must be EXACT')
    return dict(status='VERIFIED_POSTSEAL_V195_V196_EXACT_PREDICTION_PARITY_ONLY',
        baseline=old_reference, current=new_reference, baseline_outputs=manifests[0]['outputs'],
        current_outputs=manifests[1]['outputs'], queries=2, population=128,
        exact_candidates_scores_order_and_traces_equal=True,
        counts=[len(r['candidates']) for r in outputs[1]['rows']],
        old_candidates_opened=True, old_candidates_role='POSTSEAL_COMPARISON_ONLY_NOT_GENERATION_INPUT',
        candidates_used_as_features=False, generation_influenced_by_parity=False,
        truth_opened=False, fold0_opened=False, fold1_opened=False, recall_measured=False,
        full_query_population=False, ready_for_fold1=False, slo_validated=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    a = parser.parse_args()
    scope = Path('/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_pair_semantic_parity')
    if a.report.exists() or not a.report.resolve().is_relative_to(scope.resolve()):
        raise ValueError('Fresh own parity report outside IMMUTABLE generation roots required')
    result = compare(record(a.baseline), record(a.current))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.report, result)
    print(json.dumps(dict(status=result['status'], report=record(a.report), counts=result['counts'],
                         truth_opened=False, recall_measured=False)))
