"""Complete all-site registry CSR, before scoring and without labels."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from src.xgb_matcher.retrieval_national_membership_v96 import build
from src.xgb_matcher.retrieval_v20_workbench import Budget,ROOT,record,verify,write_json

CONFIG=ROOT/'config/retrieval_v96_national_membership.json'


def source(cfg):
    manifest=json.loads(verify(cfg['source_manifest']).read_text())
    if (manifest['status']!='SEALED_COMPLETE_VIRTUAL_NATIONAL_SCOPED_NAMES_AND_MEMBERSHIP' or
            manifest['truth_opened'] or manifest['crm_opened'] or
            manifest['stats']['sites']!=cfg['expected_sites'] or manifest['stats']['graph_membership_required']):
        raise ValueError('Complete official national registry required')
    launch=json.loads(verify(manifest['launch']).read_text())
    for ref in launch['implementation'].values(): verify(ref)
    return verify(manifest['artifacts']['national_membership.parquet'])


def worker(output):
    launch=json.loads((output/'launch.json').read_text()); cfg=launch['config']; budget=None
    for ref in launch['implementation'].values(): verify(ref)
    try:
        with Budget(cfg['resources']) as budget:
            registry=source(cfg); stats=build(registry,output,lambda row:print(json.dumps(row),flush=True)); source(cfg)
            if stats['sites']!=cfg['expected_sites']: raise ValueError('Complete registry denominator drift')
            size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size>cfg['max_output_bytes']: raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values(): verify(ref)
        write_json(output/'manifest.json',dict(status='SEALED_COMPLETE_NATIONAL_SITE_AND_PARENT_MEMBERSHIP',
            launch=record(output/'launch.json'),source_manifest=cfg['source_manifest'],stats=stats,
            artifacts={p.name:record(p) for p in output.glob('*.npy')},resources=budget.report(),output_bytes=size,
            truth_opened=False,crm_opened=False,model_fit=False,recall_measured=False,fold0_opened=False,fold1_opened=False,
            identifiers_as_features=False,downstream_models_changed=False,ready_for_fold1=False))
        print(json.dumps(dict(stats=stats,resources=budget.report(),output_bytes=size)),flush=True)
    except BaseException as exc:
        write_json(output/'worker_stop.json',dict(status='STOP',reason=str(exc),truth_opened=False,
            resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None)); raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg=json.loads(CONFIG.read_text())
    paths=(Path(__file__),CONFIG,ROOT/'docs/retrieval_v96_national_membership.md',ROOT/'src/xgb_matcher/retrieval_national_membership_v96.py',
        ROOT/'src/xgb_matcher/retrieval_scoped_paths_v90.py',ROOT/'src/xgb_matcher/retrieval_scoped_paths_v87.py',
        ROOT/'src/xgb_matcher/retrieval_succession_csr_v84.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py')
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.compile_retrieval_v96_national_membership','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        print((output/'worker.log').read_text()[-4000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),truth_opened=False)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=['run','worker'])
    p.add_argument('--output',required=True,type=Path); a=p.parse_args(); (run if a.mode=='run' else worker)(a.output)
