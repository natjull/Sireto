"""Lossless retrieval-only view of a complete, observed OSM source.

Streams per-query entries and drops ONLY unused buildings/nonmatching roads.
Every POI/address field is retained; matched geometry is separately referenced.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from scripts import embeddinggemma2_night as io
from src.xgb_matcher.retrieval_v20_workbench import record,verify


def main(root):
    g=io.read(root/'generation.json');exit_record=io.read(root/'owned_tool_exit.json')
    if g['status']!='COMPLETE_ALL526_LOCAL_OSM_SOURCE' or exit_record['exit_code']!=0:
        raise ValueError('Complete source plus actually observed exit0 required')
    source=verify(g['files']['query_context']);shapes=verify(g['files']['shapes'])
    out=root/'query_context_for_retrieval.json'
    if out.exists():raise ValueError('Immutable projection already exists')
    sys.path.insert(0,str(root/'deps'));import ijson
    counts={'queries':0,'pois':0,'addresses':0};tmp=out.with_suffix('.json.tmp')
    with source.open('rb') as src,tmp.open('w') as dst:
        dst.write('{');first=True
        for qid,value in ijson.kvitems(src,'',use_float=True):
            projected={k:value[k] for k in ('spatial_crs_compatible','pois','addresses')}
            if not first:dst.write(',')
            dst.write(json.dumps(qid));dst.write(':');json.dump(projected,dst,ensure_ascii=False,separators=(',',':'))
            first=False;counts['queries']+=1
            counts['pois']+=len(projected['pois']);counts['addresses']+=len(projected['addresses'])
        dst.write('}')
    if counts['queries']!=526:raise ValueError('ALL526 projection required')
    os.replace(tmp,out)
    io.write(root/'projection_generation.json',dict(status='COMPLETE_LOSSLESS_RETRIEVAL_PROJECTION_ALL526',
        source_generation=record(root/'generation.json'),source_exit=record(root/'owned_tool_exit.json'),
        implementation=record(Path(__file__)),context=record(out),shapes=g['files']['shapes'],
        counts=counts,no_labels_read=True,no_field_rewriting=True,completed_unix=time.time()))
    print('COMPLETE_PROJECTION',counts,'bytes',out.stat().st_size,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
