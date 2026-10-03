"""V118 exact factorization physical revision, immutable source sample only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import duckdb

from src.xgb_matcher.retrieval_variant_factor_v119 import compare_dictionaries, factorize
from src.xgb_matcher.retrieval_v20_workbench import Budget, ROOT, record, verify, write_json

CONFIG = ROOT / 'config/retrieval_v119_variant_factorization.json'


def inputs(cfg):
    manifest = json.loads(verify(cfg['input_manifest']).read_text())
    if (manifest['status'] != 'SEALED_SOURCE_VARIANT_COST_PROTOTYPE_ONLY' or
        manifest['truth_opened'] or manifest['crm_opened'] or manifest['recall_measured'] or
        manifest['stats']['national_publication'] or
        not manifest['stats']['factorization']['full_observation_roundtrip']):
        raise ValueError('Sealed official-source cost-only sample required')
    paths = {name:verify(manifest['artifacts'][name]) for name in
             ('raw_observations.parquet','names.parquet','facts.parquet','observations.parquet')}
    return manifest, paths


def worker(output):
    launch = json.loads((output / 'launch.json').read_text()); budget = None; c = None
    try:
        for ref in launch['implementation'].values():
            verify(ref)
        cfg = launch['config']
        with Budget(cfg['resources']) as budget:
            manifest, paths = inputs(cfg)
            c = duckdb.connect(); c.execute('SET threads=1'); c.execute('SET preserve_insertion_order=false')
            c.execute('SET memory_limit=?', [cfg['duckdb_memory']])
            c.execute('SET temp_directory=?', [str(output / 'spill')])
            c.execute('SET max_temp_directory_size=?', [cfg['spill_limit']])
            stats = factorize(c, paths['raw_observations.parquet'], output,
                lambda row:print(json.dumps(row),flush=True))
            parity = compare_dictionaries(c, paths['names.parquet'].parent, output)
            old_stats = manifest['stats']['factorization']
            for key in ('observations','lexical_eligible_observations','names','facts','max_observations_per_fact'):
                if stats[key] != old_stats[key]:
                    raise ValueError('V118/V119 statistic parity drift: ' + key)
            c.close(); c = None; inputs(cfg)
            output_bytes = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if output_bytes > cfg['max_output_bytes']:
                raise ValueError('STOP_OUTPUT_SIZE')
        for ref in launch['implementation'].values():
            verify(ref)
        write_json(output / 'manifest.json', dict(status='SEALED_SAME_VARIANT_FACTORIZATION_COST_ONLY',
            launch=record(output / 'launch.json'), input_manifest=cfg['input_manifest'], stats=stats,
            parity=parity, resources=budget.report(), output_bytes=output_bytes,
            artifacts={p.name:record(p) for p in output.glob('*.parquet')},
            truth_opened=False, crm_opened=False, fold0_opened=False, fold1_opened=False,
            model_fit=False, recall_measured=False, national_publication=False, index_built=False,
            identifiers_as_features=False, downstream_models_changed=False, ready_for_fold1=False))
        print(json.dumps(dict(stats=stats,parity=parity,resources=budget.report(),output_bytes=output_bytes)),flush=True)
    except BaseException as exc:
        if c is not None:
            c.close()
        write_json(output / 'worker_stop.json', dict(status='STOP', reason=str(exc), recall_measured=False,
            truth_opened=False, resources=budget.report() if budget is not None and hasattr(budget,'elapsed') else None))
        raise


def run(output):
    output.mkdir(parents=True,exist_ok=False); cfg = json.loads(CONFIG.read_text())
    paths = [Path(__file__),CONFIG,ROOT/'src/xgb_matcher/retrieval_variant_factor_v119.py',
        ROOT/'src/xgb_matcher/retrieval_variants_v118.py',ROOT/'src/xgb_matcher/retrieval_v20_workbench.py',
        ROOT/'src/xgb_matcher/retrieval_parent_discovery_v18.py',ROOT/'docs/retrieval_v119_variant_factorization.md']
    write_json(output/'launch.json',dict(config=cfg,implementation={str(p):record(p) for p in paths},truth_opened=False))
    try:
        with (output/'worker.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m','scripts.probe_retrieval_v119_variant_factorization','worker',
                '--output',str(output)],check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
        print((output/'worker.log').read_text()[-10000:],flush=True)
    except BaseException as exc:
        write_json(output/'status.json',dict(status='STOP',reason=str(exc),recall_measured=False,truth_opened=False));raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['run','worker'])
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    (run if args.mode=='run' else worker)(args.output)
