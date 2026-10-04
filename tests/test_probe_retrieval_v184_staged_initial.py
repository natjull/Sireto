"""Supervisor/control fixtures only, clearly fake process/resource reports."""
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.probe_retrieval_v184_staged_initial as workbench
from src.xgb_matcher.retrieval_v20_workbench import record, write_json
from test_seal_retrieval_v175_address_parts import gate, supervision, resource_report


def phase_fixture(tmp_path):
    output = tmp_path / 'output'
    dest = output / 'names'
    dest.mkdir(parents=True)
    write_json(output / 'launch.json', {'fixture': 'CONTROL_ONLY_NOT_NATIONAL'})
    write_json(dest / 'preflight.json', gate())
    write_json(dest / 'supervision.json', supervision())
    proof = dest / 'proof.json'
    write_json(proof, {'fixture': True})
    source = tmp_path / 'source.json'
    write_json(source, {'fixture': True})
    m = dict(status='SEALED_V184_NAMES', launch=record(output / 'launch.json'),
        preflight=record(dest / 'preflight.json'), artifacts={'proof': record(proof)},
        source_epochs=workbench.epochs([record(source)]), resources=resource_report(),
        old_candidates_opened=False, **workbench.FLAGS)
    write_json(dest / 'manifest.json', m)
    return output, dest, m, source


def test_stage_requires_actual_exit_report_source_and_output_hash(tmp_path):
    out, dest, m, source = phase_fixture(tmp_path)
    actual, ref = workbench.stage(out, 'names')
    assert actual == m and ref == record(dest / 'manifest.json')


@pytest.mark.parametrize('change', ['no_parent', 'unconfirmed', 'nonzero', 'rss', 'pressure', 'truth',
                                   'wrong_launch', 'bad_gate', 'output_hash', 'source_epoch', 'stop'])
def test_partial_or_stop_stage_never_admitted_to_final_initial100(tmp_path, change):
    out, dest, m, source = phase_fixture(tmp_path)
    if change == 'no_parent':
        (dest / 'supervision.json').unlink()  # Own pytest fixture only.
    elif change in ('unconfirmed', 'nonzero'):
        parent = supervision()
        parent.update(own_child_exit_confirmed=False if change == 'unconfirmed' else True,
                      exit_code=1 if change == 'nonzero' else 0)
        write_json(dest / 'supervision.json', parent)
    elif change == 'rss':
        m['resources']['peak_rss_bytes'] = 2**30+1
    elif change == 'pressure':
        m['resources']['pressure_levels'] = [2]
    elif change == 'truth':
        m['truth_opened'] = True
    elif change == 'wrong_launch':
        m['launch']['sha256'] = '0'*64
    elif change == 'bad_gate':
        g = gate(); g['pass_system_quiet_policy'] = False
        write_json(dest / 'preflight.json', g)
        m['preflight'] = record(dest / 'preflight.json')
    elif change == 'output_hash':
        write_json(dest / 'proof.json', {'changed': True})
    elif change == 'source_epoch':
        write_json(source, {'changed': True})
    else:
        write_json(dest / 'supervisor_stop.json', {'status': 'STOP'})
    write_json(dest / 'manifest.json', m)
    with pytest.raises((ValueError, FileNotFoundError)):
        workbench.stage(out, 'names')


def test_all128_query_validation_rejects_closed_last_fold_before_model_or_proofs(tmp_path):
    p = tmp_path / 'query.json'
    rows = [dict(query_id=str(i), fold=2+i%3, crm_name='NOM', crm_number='', crm_suffix='',
                 crm_street='', crm_postcode='', crm_insee='') for i in range(128)]
    rows[-1]['fold'] = 1
    write_json(p, dict(queries=rows, truth_opened=False, old_candidates_opened=False,
                      activity_inputs={'proofs': {'path': 'LABEL_SIDE_DO_NOT_OPEN'}},
                      models={'path': 'MODEL_MUST_NOT_BE_READ_ON_BAD_QUERY'}))
    with pytest.raises(ValueError, match='Frozen query-only'):
        workbench.inputs({'query_only_launch': record(p), 'queries': 1})


def test_budget_policy_cannot_be_relaxed_by_config(tmp_path, monkeypatch):
    cfg = json.loads(workbench.CONFIG.read_text())
    cfg['resources']['max_rss_bytes'] += 1
    p = tmp_path / 'config.json'
    write_json(p, cfg)
    monkeypatch.setattr(workbench, 'CONFIG', p)
    with pytest.raises(ValueError, match='authorized'):
        workbench.config()


def test_scoring_query_proofs_reopened_readonly_not_modified_or_removed(tmp_path):
    n = 7
    a = tmp_path / 'coverage.npy'; e = tmp_path / 'exact.npy'
    np.save(a, np.array([0., .1, .2, .3, .4, .5, 1.]))
    np.save(e, np.array([False]*6+[True]))
    refs = {'coverage': record(a), 'exact': record(e)}
    restored = workbench.reopen_names({'artifacts': refs}, {'sites': n, 'chunk': 3})
    try:
        cov, exact = restored.gather(np.arange(n, dtype=np.uint32))
        np.testing.assert_array_equal(cov, np.load(a))
        np.testing.assert_array_equal(exact, np.load(e).astype(np.float64))
        restored.verify_after()
    finally:
        restored.close()
    assert record(a) == refs['coverage'] and record(e) == refs['exact']
