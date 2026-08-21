#!/usr/bin/env python3
"""Build structured FINESS/UAI/DINUM evidence and its Tantivy overlay."""

from __future__ import annotations
import argparse, itertools, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.xgb_matcher.official_sector_evidence import (  # noqa:E402
    build_sector_index, iter_administrations, iter_associations, iter_data_es,
    iter_education, iter_finess,
    sha256_file, write_sector_evidence,
)

def main()->None:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    e=sub.add_parser("extract"); e.add_argument("--finess",type=Path); e.add_argument("--education",type=Path); e.add_argument("--administrations",type=Path); e.add_argument("--data-es",type=Path); e.add_argument("--associations",type=Path); e.add_argument("--output",type=Path,required=True)
    b=sub.add_parser("build-index"); b.add_argument("--evidence",type=Path,required=True); b.add_argument("--siret-documents",type=Path,required=True); b.add_argument("--siren-documents",type=Path,required=True); b.add_argument("--output-root",type=Path,required=True)
    a=p.parse_args()
    if a.command=="extract":
        adapters=[]
        for path,adapter in ((a.finess,iter_finess),(a.education,iter_education),(a.administrations,iter_administrations),(a.data_es,iter_data_es),(a.associations,iter_associations)):
            if path is not None: adapters.append((path,adapter))
        if not adapters: raise SystemExit("extract requires at least one official source")
        count=write_sector_evidence(itertools.chain.from_iterable(adapter(path) for path,adapter in adapters),a.output)
        manifest={"schema_version":"sireto-official-sector-extract-manifest-v2","rows":count,"inputs":{path.name:{"sha256":sha256_file(path),"adapter":adapter.__name__} for path,adapter in adapters},"output_sha256":sha256_file(a.output),"structured_identifier_required":True,"free_text_excluded":["association_object","observations","web_content"]}
        a.output.with_suffix(a.output.suffix+".manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
        print(a.output); return
    print(build_sector_index(evidence_path=a.evidence,siret_documents_path=a.siret_documents,siren_documents_path=a.siren_documents,output_root=a.output_root))

if __name__=="__main__": main()
