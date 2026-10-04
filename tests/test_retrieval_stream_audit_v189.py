"""Real tiny hashes, simulated supervised fixtures; no national claim."""
import json

import pytest

from scripts import probe_retrieval_v187_staged_initial as original
from src.xgb_matcher import retrieval_stream_audit_v189 as workbench
from src.xgb_matcher.retrieval_v20_workbench import write_json
from test_probe_retrieval_v187_staged_initial import stages_fixture, replace_artifact


def test_streaming_plan_and_receipts_are_exact_original187_dicts(tmp_path):
    output, cfg = stages_fixture(tmp_path)
    assert workbench.score_plan(output, cfg, original.stage) == original.audit_plan(output, cfg)
    assert workbench.audited_plan(output, cfg, original.stage) == original.audited_plan(output, cfg)


@pytest.mark.parametrize('change', ['missing_site', 'signature', 'epoch', 'truth', 'target', 'pool',
                                   'parent', 'receipt_group', 'receipt_target'])
def test_streaming_keeps_all_source_control_and_admission_guards(tmp_path, change):
    output, cfg = stages_fixture(tmp_path)
    if change == 'parent':
        path = output / 'audit-facts' / 'supervision.json'
        s = json.loads(path.read_text()); s['exit_code'] = 1
        write_json(path, s)
    elif change.startswith('receipt_'):
        replace_artifact(output, 'audit-facts', 'receipt', lambda r: r.update(
            {'group': 'postings'} if change == 'receipt_group' else {'exact_target_digest': 'f'*64}))
    elif change in ('truth', 'epoch', 'target'):
        def mutate(p):
            if change == 'truth':
                p['truth_opened'] = True
            elif change == 'epoch':
                p['source_epochs'][0]['fingerprint'][3] += 1
            else:
                p['groups']['facts'][0]['sha256'] = 'f'*64
        replace_artifact(output, 'score000', 'audit_packet', mutate)
    else:
        def mutate(p):
            if change == 'missing_site':
                p['stats']['sites_completed'] -= 1
            elif change == 'signature':
                p['signature'] = 'f'*64
            else:
                p['stats']['fixed_reservoir'] = 2000
        replace_artifact(output, 'score000', 'partition', mutate)
    with pytest.raises(ValueError):
        workbench.audited_plan(output, cfg, original.stage)
