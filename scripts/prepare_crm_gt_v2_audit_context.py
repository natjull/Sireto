#!/usr/bin/env python3
"""Enrich the preregistered CRM GT v2 sample with official identity evidence."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import sys
import duckdb, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_crm_gt_v2_population import sha256

SCHEMA_VERSION="sireto-crm-gt-v2-audit-context-1"

def main():
    p=argparse.ArgumentParser(); p.add_argument("--sample",type=Path,required=True); p.add_argument("--sirene",type=Path,required=True); p.add_argument("--legal-units",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
    sample=pd.read_csv(a.sample,dtype=str,keep_default_na=False)
    if len(sample)!=400 or sample.target_siren.nunique()!=400: raise ValueError("Expected 400 distinct SIRENs")
    wanted=sample[["gt_siret"]].rename(columns={"gt_siret":"siret"}); con=duckdb.connect(); con.register("wanted",wanted)
    official=con.execute("""SELECT CAST(e.siret AS VARCHAR) gt_siret, CAST(e.siren AS VARCHAR) target_siren,
      e.enseigne1Etablissement, e.enseigne2Etablissement, e.enseigne3Etablissement,
      e.denominationUsuelleEtablissement, e.numeroVoieEtablissement, e.indiceRepetitionEtablissement,
      e.typeVoieEtablissement, e.libelleVoieEtablissement, e.codePostalEtablissement,
      e.libelleCommuneEtablissement, e.codeCommuneEtablissement, e.etatAdministratifEtablissement,
      u.denominationUniteLegale, u.sigleUniteLegale, u.nomUniteLegale, u.nomUsageUniteLegale,
      u.prenom1UniteLegale, u.prenomUsuelUniteLegale, u.denominationUsuelle1UniteLegale,
      u.denominationUsuelle2UniteLegale, u.denominationUsuelle3UniteLegale
      FROM read_parquet(?) e JOIN wanted w ON CAST(e.siret AS VARCHAR)=w.siret
      LEFT JOIN read_parquet(?) u ON CAST(e.siren AS VARCHAR)=CAST(u.siren AS VARCHAR)""",[str(a.sirene),str(a.legal_units)]).fetchdf(); con.close()
    merged=sample.merge(official,on=["gt_siret","target_siren"],validate="one_to_one")
    if len(merged)!=400: raise ValueError("Official audit join is incomplete")
    a.output_dir.mkdir(parents=True,exist_ok=True); out=a.output_dir/"audit_context.jsonl"
    with out.open("w",encoding="utf-8") as f:
        for row in merged.to_dict("records"):
            clean={k:("" if pd.isna(v) else str(v)) for k,v in row.items()}
            f.write(json.dumps(clean,ensure_ascii=False,sort_keys=True)+"\n")
    identity={"schema_version":SCHEMA_VERSION,"sample_sha256":sha256(a.sample),"sirene_sha256":sha256(a.sirene),"legal_units_sha256":sha256(a.legal_units),"rows":400}
    manifest={**identity,"created_at":datetime.now(timezone.utc).isoformat(),"outputs":{"audit_context.jsonl":sha256(out)},"retrieval_inputs_used":False}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(a.output_dir)
if __name__=="__main__": main()
