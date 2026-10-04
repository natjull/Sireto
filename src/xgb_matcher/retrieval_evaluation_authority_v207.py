"""Label-free full128 authority. Never open targets on a self-reported seal."""
import json
from pathlib import Path

from scripts import run_retrieval_v206_population_initial as generation
from . import retrieval_postscore_audit_v187 as hashes
from .retrieval_memory_policy_v153 import LIMITS
from .retrieval_v20_workbench import record, verify

STATUS = 'PERMITTED_V207_ALL128_POSTSEAL_EVALUATION_ONLY'


def header(root):
    root = Path(root)
    cfg = generation.config()
    if not root.resolve().is_relative_to(Path(cfg['output_root']).resolve()):
        raise ValueError('Only own complete V206 generation is admissible')
    if (root / 'status.json').exists(): raise ValueError('STOP generation cannot authorize target reads')
    manifest = json.loads((root / 'manifest.json').read_text())
    if (manifest.get('status') != generation.STATUS
            or any(manifest.get(k) is not True for k in
                   ('initial100_emitted', 'full_query_population', 'complete_national_source'))
            or any(manifest.get(k) is not False for k in
                   ('old_candidates_opened', 'ready_for_fold1', 'slo_validated', 'downstream_models_changed'))
            or any(manifest.get(k) is not v for k, v in generation.original.FLAGS.items())
            or any(type(manifest.get(k)) is not int or manifest[k] != 128 for k in ('queries', 'population'))
            or type(manifest.get('peak_worker_rss_bytes')) is not int
            or not 0 < manifest['peak_worker_rss_bytes'] <= LIMITS['max_rss_bytes']):
        raise ValueError('Actual full128 protected original initial100 seal required')
    launch = json.loads(verify(manifest['launch']).read_text())
    if launch != generation.load(root): raise ValueError('Exact frozen original launch required')
    return root, manifest


def authorize(root):
    """Caller supervises this potentially costly label-free validation worker.

    Permission alone is NOT an evaluation or READY. A consuming worker must
    verify the actual producer's exit/resource report and this artifact SHA.
    No ledger of human targets or official target-side file is touched here.
    """
    root, manifest = header(root)
    checked, plan = generation.admission(root)  # ALL384 producers, full audits,64merges,final93507 ledger.
    if (manifest['ledger'] != checked['_reference']
            or manifest['outputs'] != checked['artifacts']['draft']
            or manifest['plan'] != record(root / 'plan/manifest.json')
            or manifest['source_and_derived_files_full_hashed'] != plan['files']
            or manifest['full_bytes_hashed'] != plan['full_bytes']
            or type(manifest['phases']) is not int or manifest['phases'] != 450+len(plan['chunks'])):
        raise ValueError('Original ROOT output/ledger/complete full-audit count binding required')
    outputs = json.loads(verify(manifest['outputs']).read_text())
    receipt = json.loads(verify(checked['artifacts']['receipt']).read_text())
    controls = [dict(position=r['position'], query_id=r['query_id'], fold=r['fold'], signature=r['trace']['signature'])
                for r in outputs['rows']]
    generation.whole.controls(controls, list(range(128)))
    if (controls != plan['rows'] or len(outputs['rows']) != 128
            or outputs['truth_opened'] is not False or outputs['recall_measured'] is not False
            or outputs['full_query_population'] is not True or outputs['population'] != 128
            or outputs['initial100_emitted'] is not False):
        raise ValueError('Entire original128 still-unlabelled outputs required')
    return dict(status=STATUS, generation=record(root / 'manifest.json'), launch=manifest['launch'],
        outputs=manifest['outputs'], ledger=manifest['ledger'], ledger_receipt=checked['artifacts']['receipt'],
        plan=receipt['plan'], epochs=receipt['epochs'], controls=controls,
        queries=128, population=128, current_catalog=generation.original.config()['catalog'],
        actual_generation_admission_verified=True, all128_before_targets=True,
        truth_opened=False, recall_measured=False, ready_for_fold1=False,
        fold0_opened=False, fold1_opened=False, downstream_models_changed=False,
        evaluation_process_supervision_still_required=True)


def recheck(permission):
    """After actual permission-producer supervision, before/after target reads."""
    if (permission.get('status') != STATUS
            or any(type(permission.get(k)) is not int or permission[k] != 128 for k in ('queries', 'population'))
            or permission.get('evaluation_process_supervision_still_required') is not True
            or any(permission.get(k) is not True for k in ('actual_generation_admission_verified', 'all128_before_targets'))
            or any(permission.get(k) is not False for k in
                   ('truth_opened', 'recall_measured', 'ready_for_fold1', 'fold0_opened', 'fold1_opened', 'downstream_models_changed'))):
        raise ValueError('Exact closed-label whole128 permission required')
    root, manifest = header(verify(permission['generation']).parent)
    for field in ('launch', 'outputs', 'ledger'):
        if permission[field] != manifest[field]: raise ValueError('Original generation binding changed')
    generation.whole.controls(permission['controls'], list(range(128)))
    receipt = json.loads(verify(permission['ledger_receipt']).read_text())
    plan = json.loads(verify(permission['plan']).read_text())
    if (receipt['draft'] != permission['outputs'] or receipt['plan'] != permission['plan']
            or receipt['epochs'] != permission['epochs'] or receipt['queries'] != 128
            or plan['rows'] != permission['controls']
            or permission['current_catalog'] != generation.original.config()['catalog']):
        raise ValueError('Same entire original plan/epoch/control/source authority required')
    hashes.check_epochs(json.loads(verify(permission['epochs']).read_text()))
    return root, manifest
