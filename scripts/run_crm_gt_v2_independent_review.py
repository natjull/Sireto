#!/usr/bin/env python3
"""Run a bounded independent Luna review of the frozen CRM GT v2 sample."""
from __future__ import annotations
import argparse, concurrent.futures, json, subprocess
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_crm_gt_v2_population import sha256

KEEP=("query_id","crm_name","crm_adresse","crm_cp","crm_insee","crm_commune","gt_siret",
      "sirene_etat","loc_match_type","denominationUniteLegale","sigleUniteLegale","nomUniteLegale",
      "nomUsageUniteLegale","prenom1UniteLegale","prenomUsuelUniteLegale","denominationUsuelle1UniteLegale",
      "denominationUsuelle2UniteLegale","denominationUsuelle3UniteLegale","enseigne1Etablissement",
      "enseigne2Etablissement","enseigne3Etablissement","denominationUsuelleEtablissement",
      "numeroVoieEtablissement","indiceRepetitionEtablissement","typeVoieEtablissement",
      "libelleVoieEtablissement","codePostalEtablissement","libelleCommuneEtablissement",
      "codeCommuneEtablissement")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--context",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--concurrency",type=int,default=16); a=p.parse_args()
    rows=[json.loads(x) for x in a.context.read_text().splitlines() if x.strip()]
    if len(rows)!=400 or len({r["query_id"] for r in rows})!=400: raise ValueError("Expected 400 unique tasks")
    a.output_dir.mkdir(parents=True,exist_ok=True); schema=a.output_dir/"schema.json"
    schema.write_text(json.dumps({"type":"object","additionalProperties":False,"required":["reviews"],"properties":{"reviews":{"type":"array","minItems":5,"maxItems":5,"items":{"type":"object","additionalProperties":False,"required":["query_id","verdict","reason"],"properties":{"query_id":{"type":"string"},"verdict":{"type":"string","enum":["PASS","BORDERLINE","CERTAIN_FALSE_LABEL"]},"reason":{"type":"string"}}}}}}))
    batches=[rows[i:i+5] for i in range(0,len(rows),5)]
    def run(item):
        idx,batch=item; raw=a.output_dir/f"batch5_{idx:03d}.json"; log=a.output_dir/f"batch5_{idx:03d}.log"
        expected={r["query_id"] for r in batch}
        if raw.exists():
            prior=json.loads(raw.read_text()).get("reviews",[])
            if len(prior)==5 and {x["query_id"] for x in prior}==expected: return prior
        payload=[{k:r.get(k,"") for k in KEEP} for r in batch]
        prompt=("Tu es un auditeur indépendant de ground truth SIRET. Juge uniquement si le CRM soutient l'identité "
          "et le site SIRENE officiels fournis. N'utilise aucun retrieval, score ou rang. PASS si nom/enseigne et adresse "
          "sont cohérents ou si l'adresse officielle exacte constitue une preuve forte sans contradiction d'identité. "
          "CERTAIN_FALSE_LABEL seulement si les données désignent clairement une autre entité ou un autre site. "
          "BORDERLINE si la preuve est insuffisante. Retourne exactement un résultat par query_id.\n"+json.dumps(payload,ensure_ascii=False))
        cmd=["codex","exec","--ephemeral","--ignore-rules","--skip-git-repo-check","-m","gpt-5.6-luna","-c","model_reasoning_effort=\"low\"","-s","read-only","--output-schema",str(schema),"-o",str(raw),prompt]
        for attempt in range(3):
            proc=subprocess.run(cmd,cwd="/tmp",text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
            log.write_text(proc.stdout)
            if proc.returncode==0:
                reviews=json.loads(raw.read_text())["reviews"]
                if len(reviews)==5 and {x["query_id"] for x in reviews}==expected: return reviews
        raise ValueError(f"batch {idx} row mismatch after retries")
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        grouped=list(pool.map(run,enumerate(batches)))
    reviews=[x for group in grouped for x in group]
    reviews.sort(key=lambda x:x["query_id"]); out=a.output_dir/"reviews.jsonl"
    out.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in reviews))
    counts={v:sum(x["verdict"]==v for x in reviews) for v in ("PASS","BORDERLINE","CERTAIN_FALSE_LABEL")}
    manifest={"schema_version":"sireto-crm-gt-v2-independent-review-1","context_sha256":sha256(a.context),"rows":400,"counts":counts,"gate_pass":counts["CERTAIN_FALSE_LABEL"]==0,"outputs":{"reviews.jsonl":sha256(out)}}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
