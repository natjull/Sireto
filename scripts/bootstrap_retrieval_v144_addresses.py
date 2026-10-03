"""Canonical artifact references, unchanged V143 source-only execution engine."""
import argparse
import json
from pathlib import Path

import scripts.resume_retrieval_v143_addresses as backend
from scripts.probe_retrieval_v118_official_variants import fingerprint
from src.xgb_matcher.retrieval_append_catalog_v141 import AppendKeys
from src.xgb_matcher.retrieval_proof_bridge_v137 import PROTECTED
from src.xgb_matcher.retrieval_system_quiet_v143 import quiet
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v144_address_bootstrap.json'


def canonical(reference):
    value=record(verify(reference))
    if 'size_bytes' in reference and reference['size_bytes']!=value['size_bytes']:
        raise ValueError('Explicit artifact byte size drift')
    return value


def init(output):
    output.mkdir(parents=True,exist_ok=False);resume=json.loads(CONFIG.read_text())
    resume['original_launch']=canonical(resume['original_launch']);resume['original_stop']=canonical(resume['original_stop'])
    old=json.loads(verify(resume['original_launch']).read_text());old_root=Path(resume['original_launch']['path']).parent
    for ref in old['implementation'].values():verify(ref)
    if any(old.get(k) is not False for k in PROTECTED):raise ValueError('Protected original source-only launch required')
    stop=json.loads(verify(resume['original_stop']).read_text())
    if stop['status']!='STOP' or any(stop.get(k) is not False for k in PROTECTED) or stop['resources']['pass_resource_policy']:
        raise ValueError('Actual terminal failed original part required')
    original=json.loads(verify(old['plan']).read_text());cfg=dict(old['config']);cfg['batch_rows']=resume['batch_rows']
    if (original['status']!='SEALED_V142_ALL_OFFICIAL_ADDRESS_PHYSICAL_PLAN' or any(original.get(k) is not False for k in PROTECTED)
        or original['total_observations']!=cfg['expected_total_observations'] or original['role_order']!=['official_addresses','current_addresses']
        or not isinstance(cfg['batch_rows'],int) or isinstance(cfg['batch_rows'],bool)
        or not cfg['scalar_first_source_rows']<=cfg['batch_rows']<old['config']['batch_rows']
        or cfg['resources']['max_swapout_pages_delta']!=0):raise ValueError('Whole unchanged original projection and smaller batches required')
    print(json.dumps(dict(phase='initial_system_quiet_gate',required_seconds=resume['initial_quiet_seconds'])),flush=True)
    gate=quiet(resume['initial_quiet_seconds'],resume['min_system_free_percent'],resume['max_quiet_wait_seconds'])
    write_json(output/'preflight.json',gate);imports={};parts=0
    with Budget(cfg['resources']) as budget:
        if budget.before!=gate['final_snapshot']['swapouts']:raise RuntimeError('STOP_SYSTEM_CHANGED_SINCE_QUIET_GATE')
        catalog=AppendKeys(cfg['catalog'])
        if catalog.sites!=cfg['expected_namespace_sites']:raise ValueError('Whole unchanged address-admitting namespace required')
        for role in original['role_order']:
            source=original['sources'][role];verify(source['reference'])
            if fingerprint(source['reference']['path'])!=source['fingerprint']:raise ValueError('Original raw source changed')
            for spec in source['parts']:
                path=backend.directory(old_root,role,spec['part_id'])/'manifest.json'
                if not path.exists():continue
                ref=record(path);backend.validate(ref,role,spec,source,resume['original_launch'],cfg['catalog'],cfg['scalar_first_source_rows'])
                imports[f'{role}/{spec["part_id"]}']=ref;parts+=1
        if parts!=resume['expected_reused_parts'] or any(f'{role}/0' not in imports for role in original['role_order']):
            raise ValueError('Every successful original part and both first scalar oracles required')
    plan=dict(original,status='SEALED_V143_COMPLETE_ORIGINAL_ADDRESS_RESUMPTION_PLAN',original_plan=old['plan'],
        imported_parts=imports,imported_part_count=parts,resumption_audit_resources=budget.report(),bootstrap_revision=144)
    write_json(output/'plan.json',plan)
    paths=[Path(__file__),CONFIG,ROOT/'docs/retrieval_v144_address_bootstrap.md',
        ROOT/'scripts/resume_retrieval_v143_addresses.py',ROOT/'config/retrieval_v143_address_resume.json',
        ROOT/'docs/retrieval_v143_address_resume.md',ROOT/'src/xgb_matcher/retrieval_system_quiet_v143.py']
    implementation=dict(old['implementation']);implementation.update({str(p):record(p) for p in paths})
    write_json(output/'launch.json',dict(config=cfg,resumption=resume,plan=record(output/'plan.json'),implementation=implementation,
        preflight=record(output/'preflight.json'),bootstrap_revision=144,execution_backend='UNCHANGED_V143',**{k:False for k in PROTECTED}))
    print(json.dumps(dict(phase='plan',imported_parts=parts,total_parts=plan['total_parts'],batch_rows=cfg['batch_rows'],
        resources=budget.report(),quiet_seconds=gate['quiet_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['init','part','seal','remaining'])
    p.add_argument('--output',required=True,type=Path);p.add_argument('--role');p.add_argument('--part',type=int);a=p.parse_args()
    if a.mode=='init':init(a.output)
    elif a.mode=='part':
        if a.role is None or a.part is None:p.error('role/part required')
        backend.execute(a.output,a.role,a.part)
    else:{'seal':backend.seal,'remaining':backend.remaining}[a.mode](a.output)
