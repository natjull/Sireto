"""Pin BODACC sources with actual SHA256 before V61 source-only compilation."""
import argparse
from pathlib import Path
import signal
import sys
import time
from scripts import embeddinggemma2_v61_bodacc_parent_clauses as compiler
from scripts import embeddinggemma2_night as io
from scripts import embeddinggemma2_v14_full_pipeline as engine
from src.xgb_matcher.retrieval_v20_workbench import record

OUTPUT=io.HOME/'v63/bodacc_parent_clauses_bound_20261013_a'

def initialize(out):
    compiler.initialize(out)
    launch=io.read(out/'launch.json')
    sources=[]
    for item in launch['sources']:
        path=Path(item['path'])
        if path.stat().st_size!=item['size_bytes']:raise ValueError('Source size disagrees with canonical manifest')
        sources.append(record(path))
    launch['sources']=sources;launch['helpers'].append(record(Path(__file__)))
    io.write(out/'launch.json',launch)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','compile']);p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args()
    signal.signal(signal.SIGTERM,engine.interrupted);signal.signal(signal.SIGINT,engine.interrupted)
    try:
        if a.action=='initialize':initialize(a.output)
        else:compiler.compile(a.output)
    except engine.ResourceYield as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if a.output.exists():io.write(a.output/(a.action+'_error.json'),dict(error=repr(e),updated_unix=time.time()))
        raise
