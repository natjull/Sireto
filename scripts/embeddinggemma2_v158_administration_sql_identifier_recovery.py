"""Execution-only reserved SQL alias recovery; original source preserved.

The original worker failed before its first eligible source was staged.
Only the transient SQL column identifier admin -> admin_text is changed.
Parser, matching, dates, public eligibility, formula and resource caps unchanged.
"""
import argparse
import inspect
from pathlib import Path
import time
from scripts import embeddinggemma2_v158_corporate_administration_source as original
from scripts import embeddinggemma2_night as io

OLD=original.OUT
NEW=io.HOME/'v158/corporate_administration_20261014_night_b'


def bind():
    original.OUT=NEW
    body=inspect.getsource(original.worker)
    if body.count(") admin,CAST")!=1 or body.count("r['admin']")!=1:
        raise ValueError('Exact historical SQL identifier binding')
    body=body.replace(") admin,CAST",") admin_text,CAST").replace("r['admin']","r['admin_text']")
    exec(compile(body,str(Path(__file__)), 'exec'),original.__dict__)
    runner=inspect.getsource(original.run).replace('scripts.embeddinggemma2_v158_corporate_administration_source','scripts.embeddinggemma2_v158_administration_sql_identifier_recovery')
    exec(compile(runner,str(Path(__file__)), 'exec'),original.__dict__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','worker','run']);a=p.parse_args();bind()
    try:
        if a.action=='initialize':
            ex=io.read(OLD/'owned_exit.json')
            if ex['exit_code']!=1 or not ex['own_child_exit_confirmed']:raise ValueError('Preserved real failed exit')
            original.initialize();launch=io.read(NEW/'launch.json')
            launch['execution_recovery']=original.prior.record(Path(__file__));launch['prior_failed_exit']=original.prior.record(OLD/'owned_exit.json')
            io.write(NEW/'launch.json',launch)
        else:
            original.prior.verify(io.read(NEW/'launch.json')['execution_recovery']);getattr(original,a.action)()
    except BaseException as error:
        if NEW.exists():io.write(NEW/(a.action+'_error.json'),dict(error=repr(error),updated_unix=time.time()))
        raise
