#!/usr/bin/env python3
"""Build structured FINESS/UAI/DINUM evidence and its Tantivy overlay."""

from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.xgb_matcher.official_sector_evidence import (  # noqa:E402
    build_sector_index, iter_administrations, iter_education, iter_finess,
    sha256_file, write_sector_evidence,
)

def main()->None:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    e=sub.add_parser("extract"); e.add_argument("--finess",type=Path,required=True); e.add_argument("--education",type=Path,required=True); e.add_argument("--administrations",type=Path,required=True); e.add_argument("--output",type=Path,required=True)
    b=sub.add_parser("build-index"); b.add_argument("--evidence",type=Path,required=True); b.add_argument("--siret-documents",type=Path,required=True); b.add_argument("--siren-documents",type=Path,required=True); b.add_argument("--output-root",type=Path,required=True)
    a=p.parse_args()
    if a.command=="extract":
        count=write_sector_evidence(itertools.chain(iter_finess(a.finess),iter_education(a.education),iter_administrations(a.administrations)),a.output)
        manifest={"schema_version":"sireto-official-sector-extract-manifest-v1","rows":count,"inputs":{str(x):sha256_file(x) for x in (a.finess,a.education,a.administrations)},"output_sha256":sha256_file(a.output)}
        a.output.with_suffix(a.output.suffix+".manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
        print(a.output); return
    print(build_sector_index(evidence_path=a.evidence,siret_documents_path=a.siret_documents,siren_documents_path=a.siren_documents,output_root=a.output_root))

if __name__=="__main__": main()

