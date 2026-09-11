"""Run the authorized complete comparison and then its separate evaluation.

No retries, no new recipe, no polling of external services. This is the local
job itself, not a scheduled automation. User-visible status is persisted.
"""
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys

from scripts.retrieval_v33_full_replay import ROOT, authority, code
from src.xgb_matcher.retrieval_v20_workbench import record, verify, write_json


def launch(prepared,gate,output):
    authority(prepared);g=json.loads((gate/'manifest.json').read_text())
    if g['status']!='PASS_READY_FOR_FULL_REPLAY' or g['prepared']!=record(prepared/'manifest.json') or g['code']!=code():
        raise ValueError('Technical and cost gate must pass')
    output.mkdir(parents=True,exist_ok=False)
    evaluator=record(ROOT/'scripts/evaluate_retrieval_v33_full_replay.py')
    policy=record(ROOT/'docs/siret_operational_equivalence_policy.md')
    write_json(output/'launch.json',dict(pid=os.getpid(),prepared=record(prepared/'manifest.json'),gate=record(gate/'manifest.json'),
      evaluator=evaluator,policy=policy,launcher=record(Path(__file__)),quality_measured=False))
    try:
        with (output/'generation.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/retrieval_v33_full_replay.py'),'full','--prepared',str(prepared),
              '--gate',str(gate),'--output',str(output/'generation')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        verify(evaluator);verify(policy)
        with (output/'evaluation.log').open('x') as log:
            subprocess.run([sys.executable,str(verify(evaluator)),'--generation',str(output/'generation'),
              '--output',str(output/'evaluation')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(output/'status.json',dict(status='COMPLETED_FULL_CRM_MASTER_COMPARISON',evaluation=record(output/'evaluation/manifest.json')))
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();launch(a.prepared,a.gate,a.output)
