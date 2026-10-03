"""Same national V150 snapshots, with query-scoped readonly lexical mappings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import scripts.probe_retrieval_v150_rebound_names as original
from src.xgb_matcher.retrieval_name_pages_v151 import ScopedNamePages
from src.xgb_matcher.retrieval_v20_workbench import ROOT,record,write_json

CONFIG=ROOT/'config/retrieval_v151_name_pages.json'
STATUS='SEALED_V151_ALL_OFFICIAL_REBOUND_NAMES_WITH_QUERY_SCOPED_PAGES_COST_ONLY'


def implementation():
    refs=original.implementation()
    for path in (Path(__file__),CONFIG,ROOT/'docs/retrieval_v151_name_pages.md',ROOT/'src/xgb_matcher/retrieval_name_pages_v151.py'):
        refs[str(path)]=record(path)
    return refs


def configure():
    # Runtime specialization only. No edits to the measured V150 source.
    original.NationalNames=ScopedNamePages;original.STATUS=STATUS


def run(output):
    cfg=json.loads(CONFIG.read_text())
    if not output.resolve().is_relative_to(Path(cfg['output_root']).resolve()):raise ValueError('CATNAT-only query probe root required')
    if output.exists():raise FileExistsError('Immutable query probe root exists')
    original.inputs(cfg)
    gate=original.quiet(cfg['quiet_seconds'],cfg['min_system_free_percent'],cfg['max_quiet_wait_seconds'])
    output.mkdir(parents=True,exist_ok=False);write_json(output/'preflight.json',gate)
    write_json(output/'launch.json',dict(config=cfg,preflight=record(output/'preflight.json'),implementation=implementation(),
        source_page_policy='V151_FULL_READONLY_MAP_ROTATION_AFTER_EVERY_COMPLETE_QUERY',old_candidates_opened=False,**original.FLAGS))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v151_name_pages','worker','--output',str(output)],
                check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-11000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),old_candidates_opened=False,**original.FLAGS));raise


def worker(output):
    launch=json.loads((output/'launch.json').read_text())
    if launch.get('source_page_policy')!='V151_FULL_READONLY_MAP_ROTATION_AFTER_EVERY_COMPLETE_QUERY':
        raise ValueError('Exact sealed query-scoped page policy required')
    configure();original.worker(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['run','worker']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    (run if a.mode=='run' else worker)(a.output)
