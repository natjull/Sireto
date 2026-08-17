#!/usr/bin/env python3
"""Publish CRM GT v2 with component-safe prospective folds, without retrieval."""
from __future__ import annotations

import argparse, hashlib, json, os, tempfile, unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import duckdb
import pandas as pd

SCHEMA_VERSION = "sireto-crm-gt-v2-population-1"
TRAIN_FOLDS = (2, 3, 4)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def clean(v: object) -> str:
    if v is None or pd.isna(v): return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s

def norm(v: object) -> str:
    s = unicodedata.normalize("NFKD", clean(v)).encode("ascii", "ignore").decode().upper()
    return " ".join("".join(c if c.isalnum() else " " for c in s).split())

def stable(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")

def verified_base(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    out = []
    for name in ("queries.parquet", "labels.parquet", "fold_assignments.parquet"):
        path = root / name
        if manifest["outputs"][name] != sha256(path): raise ValueError(f"Manifest mismatch: {path}")
        out.append(pd.read_parquet(path))
    return (*out, manifest)

def existing_map(existing_crm: Path, labels: pd.DataFrame, folds: pd.DataFrame):
    crm = pd.read_csv(existing_crm, sep=";", dtype=str, keep_default_na=False).reset_index(names="query_id")
    crm["query_id"] = crm["query_id"].astype(str)
    crm["siren"] = crm["gt_siret"].str[:9]
    j = crm[["query_id", "siren"]].merge(folds, on="query_id", validate="one_to_one")
    result, conflicts = {}, set()
    for siren, g in j.groupby("siren"):
        pairs = {(str(r.siren_component_id), int(r.oof_fold)) for r in g.itertuples(index=False)}
        if len(pairs) != 1: conflicts.add(str(siren).zfill(9))
        else: result[str(siren).zfill(9)] = next(iter(pairs))
    labelled = labels[["query_id", "ground_truth_siren"]].merge(
        folds[["query_id", "siren_component_id", "oof_fold"]], on="query_id", validate="one_to_one"
    )
    for siren, g in labelled[labelled.ground_truth_siren.map(clean).ne("")].groupby("ground_truth_siren"):
        key = str(siren).zfill(9)
        pairs = {(str(r.siren_component_id), int(r.oof_fold)) for r in g.itertuples(index=False)}
        if len(pairs) == 1 and key not in conflicts:
            if key in result and result[key] != next(iter(pairs)): conflicts.add(key); result.pop(key, None)
            else: result[key] = next(iter(pairs))
    return result, conflicts

def assign_unseen(rows: pd.DataFrame, seed: int):
    unseen = rows[rows.existing_component_relation.eq("UNSEEN_SIREN_NEEDS_ASSIGNMENT")]
    groups = []
    for siren, g in unseen.groupby("target_siren"):
        state = "F" if g.sirene_etat.eq("F").any() else "A"
        groups.append((str(siren).zfill(9), len(g), state + "|" + "+".join(sorted(set(g.loc_match_type)))))
    result = {}
    for stratum in sorted({x[2] for x in groups}):
        members = [x for x in groups if x[2] == stratum]
        desired = {"TRAIN": sum(x[1] for x in members)*.70,
                   "PROSPECTIVE_DEV": sum(x[1] for x in members)*.15,
                   "PROSPECTIVE_TEST": sum(x[1] for x in members)*.15}
        allocated = defaultdict(int)
        for siren, count, _ in sorted(members, key=lambda x: (-x[1], stable(seed, x[0]))):
            role = min(desired, key=lambda r: ((allocated[r]+count-desired[r])**2-(allocated[r]-desired[r])**2,
                                                stable(seed, siren, r)))
            allocated[role] += count
            fold = TRAIN_FOLDS[stable(seed, "TRAIN", siren) % 3] if role == "TRAIN" else {"PROSPECTIVE_DEV":0,"PROSPECTIVE_TEST":1}[role]
            result[siren] = (role, fold)
    return result

def operational_sets(snapshot: Path, rows: pd.DataFrame):
    wanted = pd.DataFrame({"siren": sorted(set(rows.target_siren))})
    con = duckdb.connect(); con.register("wanted", wanted)
    official = con.execute("""SELECT CAST(siret AS VARCHAR) siret, CAST(s.siren AS VARCHAR) siren,
      numeroVoieEtablissement num, indiceRepetitionEtablissement rep, typeVoieEtablissement typ,
      libelleVoieEtablissement voie, codePostalEtablissement cp, codeCommuneEtablissement insee
      FROM read_parquet(?) s JOIN wanted w ON CAST(s.siren AS VARCHAR)=w.siren""", [str(snapshot)]).fetchdf(); con.close()
    for c in official.columns: official[c] = official[c].map(clean)
    official["site"] = official.apply(lambda r: "|".join(norm(r[c]) for c in ("num","rep","typ","voie","cp","insee")), axis=1)
    members = official.groupby(["siren","site"]).siret.apply(lambda x: sorted(set(str(v).zfill(14) for v in x))).to_dict()
    by = official.set_index("siret")
    out = {}
    for siret in set(rows.gt_siret):
        r = by.loc[siret]; key = str(r.site)
        out[siret] = members[(str(r.siren), key)] if all(key.split("|")) else [siret]
    return out

def audit_sample(rows: pd.DataFrame, seed: int, excluded_query_ids: set[str] | None = None):
    excluded_query_ids = excluded_query_ids or set()
    rows = rows[~rows.query_id.astype(str).isin(excluded_query_ids)]
    one = rows.assign(_key=rows.apply(lambda r: stable(seed,"AUDIT",r.target_siren,r.query_id),axis=1)).sort_values("_key").drop_duplicates("target_siren")
    targets={"TRAIN":200,"PROSPECTIVE_DEV":100,"PROSPECTIVE_TEST":100}; selected=[]
    for role,n in targets.items():
        pool=one[one.split_role.eq(role)].copy()
        pool["_stratum"] = pool.sirene_etat.astype(str) + "|" + pool.loc_match_type.astype(str)
        sizes = pool["_stratum"].value_counts().sort_index()
        if len(pool) < n: raise AssertionError(f"Insufficient audit population for {role}")
        raw = sizes * n / len(pool)
        allocation = raw.astype(int)
        if n >= len(sizes):
            allocation = allocation.clip(lower=1)
        while int(allocation.sum()) > n:
            candidates = [key for key in allocation.index if allocation[key] > 1]
            key = min(candidates, key=lambda value: (raw[value] - allocation[value], stable(seed, role, value)))
            allocation[key] -= 1
        while int(allocation.sum()) < n:
            candidates = [key for key in allocation.index if allocation[key] < sizes[key]]
            key = max(candidates, key=lambda value: (raw[value] - allocation[value], -stable(seed, role, value)))
            allocation[key] += 1
        for stratum, count in allocation.items():
            selected += pool[pool._stratum.eq(stratum)].sort_values("_key").iloc[:int(count)].index.tolist()
    sample=one.loc[selected].drop(columns="_key").copy()
    if len(sample)!=400 or sample.target_siren.duplicated().any(): raise AssertionError("Audit sample failed")
    sample["audit_verdict"]="PENDING_INDEPENDENT_REVIEW"; sample["audit_reason"]=""
    return sample.sort_values(["split_role","query_id"])

def build(args):
    bq,bl,bf,bm=verified_base(args.base_population)
    new=pd.read_csv(args.new_increment,sep=";",dtype=str,keep_default_na=False)
    if len(new)!=args.expected_new_rows: raise ValueError(f"Expected {args.expected_new_rows}, got {len(new)}")
    new.target_siren=new.target_siren.str.zfill(9); new.gt_siret=new.gt_siret.str.zfill(14)
    if new.crm_gt_fingerprint.duplicated().any(): raise ValueError("Duplicate CRM/GT fingerprints")
    new["query_id"]=new.crm_gt_fingerprint.map(lambda x:"NEWCRM:"+x[:32])
    em,conflicts=existing_map(args.existing_crm,bl,bf); ua=assign_unseen(new,args.seed)
    components=[]; folds=[]; roles=[]
    for r in new.itertuples():
        if r.target_siren in em:
            comp,fold=em[r.target_siren]; role="TRAIN" if fold in TRAIN_FOLDS else ("PROSPECTIVE_DEV" if fold==0 else "PROSPECTIVE_TEST")
        elif r.existing_component_relation=="UNSEEN_SIREN_NEEDS_ASSIGNMENT": role,fold=ua[r.target_siren]; comp=r.target_siren
        else:
            comp=f"CONFLICT:{r.target_siren}"; fold=-1; role="QUARANTINE_COMPONENT_CONFLICT"
        components.append(comp); folds.append(fold); roles.append(role)
    new["siren_component_id"]=components; new["oof_fold"]=folds; new["split_role"]=roles
    ops=operational_sets(args.sirene,new)
    new["acceptable_sirets_operational"]=new.gt_siret.map(lambda x:json.dumps(ops[x],separators=(",",":")))
    eligible=new[new.oof_fold.ge(0)].copy()
    nq=pd.DataFrame({"query_id":eligible.query_id,"crm_record_id":eligible.crm_id,"crm_name":eligible.crm_name,"crm_address":eligible.crm_adresse,
      "crm_postcode":eligible.crm_cp,"crm_city":eligible.crm_commune,"crm_insee":eligible.crm_insee,"crm_name_norm":eligible.crm_name.map(norm),
      "crm_address_norm":eligible.crm_adresse.map(norm),"crm_city_norm":eligible.crm_commune.map(norm),"reference_date":"",
      "oof_fold":eligible.oof_fold.astype("int8"),"legacy_split_status":"FRESH_CRM_PROSPECTIVE_20260817",
      "data_origin":"REAL_CRM_20260817","split_role":eligible.split_role})
    nl=pd.DataFrame({"query_id":eligible.query_id,"label_kind":"MATCH_EXACT","ground_truth_siret":eligible.gt_siret,"ground_truth_siren":eligible.target_siren,
      "historical_ground_truth_siret":eligible.gt_siret,"historical_ground_truth_siren":eligible.target_siren,"ground_truth_state":eligible.sirene_etat,
      "label_source":"REAL_CRM_20260817","validator":"SIRENE_INSEE_CP_STRICT_V1","reliability":"HIGH_AUTOMATED",
      "evidence_reference":eligible.source_reference,"label_is_human_validated":False,"exact_metric_eligible":True,
      "identity_training_eligible":True,"operational_training_eligible":True,"ranker_weight":eligible.sirene_etat.map({"A":1.,"F":.5}),
      "acceptor_weight":0.,"legacy_split":"fresh_crm_20260817","oof_fold":eligible.oof_fold.astype("int8"),
      "data_origin":"REAL_CRM_20260817","split_role":eligible.split_role,"ground_truth_siret_exact":eligible.gt_siret,
      "acceptable_sirets_operational":eligible.acceptable_sirets_operational,"label_audit_status":"PENDING_INDEPENDENT_REVIEW"})
    nf=pd.DataFrame({"query_id":eligible.query_id,"siren_component_id":eligible.siren_component_id,"oof_fold":eligible.oof_fold,
                     "legacy_split":"fresh_crm_20260817","split_role":eligible.split_role,"data_origin":"REAL_CRM_20260817"})
    bq=bq.assign(data_origin="REAL_CRM_HISTORICAL",split_role="LEGACY_OOF")
    bl=bl.assign(data_origin="REAL_CRM_HISTORICAL",split_role="LEGACY_OOF",ground_truth_siret_exact=bl.ground_truth_siret,
      acceptable_sirets_operational=bl.ground_truth_siret.map(lambda x:json.dumps([str(x)]) if clean(x) else "[]"),
      label_audit_status=bl.label_is_human_validated.map({True:"HUMAN_VALIDATED",False:"HISTORICAL_QUALIFIED"}))
    bf=bf.assign(data_origin="REAL_CRM_HISTORICAL",split_role="LEGACY_OOF")
    queries=pd.concat([bq,nq],ignore_index=True,sort=False); labels=pd.concat([bl,nl],ignore_index=True,sort=False); fold_frame=pd.concat([bf,nf],ignore_index=True,sort=False)
    if fold_frame.groupby("siren_component_id").oof_fold.nunique().max()!=1: raise AssertionError("Component leakage")
    old=pd.read_csv(args.existing_crm,sep=";",dtype=str,keep_default_na=False)
    core=["crm_name","crm_cp","crm_insee","crm_id","crm_commune","gt_siret","crm_adresse","SITE_CLI_COMMUNE","sirene_insee","sirene_cp","sirene_etat","loc_match_type"]
    excluded_query_ids: set[str] = set()
    if args.audit_exclude_sample:
        excluded_query_ids = set(pd.read_csv(args.audit_exclude_sample, dtype=str, keep_default_na=False).query_id)
    crm=pd.concat([old[core],new[core]],ignore_index=True); audit=audit_sample(eligible,args.audit_seed,excluded_query_ids)
    identity={"schema_version":SCHEMA_VERSION,"seed":args.seed,"audit_seed":args.audit_seed,"split_ratio":[70,15,15],"audit_size":400,"builder_sha256":sha256(Path(__file__)),
      "inputs":{"base":sha256(args.base_population/"manifest.json"),"new":sha256(args.new_increment),"crm":sha256(args.existing_crm),"sirene":sha256(args.sirene)}}
    if args.audit_exclude_sample:
        identity["inputs"]["audit_exclude_sample"] = sha256(args.audit_exclude_sample)
    bid=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]; dest=args.output_root/bid
    if dest.exists(): return dest
    args.output_root.mkdir(parents=True,exist_ok=True); tmp=Path(tempfile.mkdtemp(prefix=f".{bid}.",dir=args.output_root))
    queries.to_parquet(tmp/"queries.parquet",index=False); labels.to_parquet(tmp/"labels.parquet",index=False); fold_frame.to_parquet(tmp/"fold_assignments.parquet",index=False)
    crm.to_csv(tmp/"crm_ok_gt_v2.csv",sep=";",index=False); audit.to_csv(tmp/"independent_audit_sample_400.csv",index=False)
    counts={"crm_ok_gt_v2_rows":len(crm),"model_population_rows":len(labels),"new_rows":len(new),"new_distinct_sirens":new.target_siren.nunique(),
      "new_model_eligible_rows":len(eligible),"new_quarantined_component_conflict_rows":int(new.oof_fold.lt(0).sum()),
      "new_split_roles":{str(k):int(v) for k,v in new.split_role.value_counts().items()},"new_folds":{str(k):int(v) for k,v in new.oof_fold.value_counts().sort_index().items()},"audit_rows":400}
    (tmp/"report.md").write_text(f"# Population CRM GT v2\n\n- GT CRM : **{len(crm)}**\n- Population modèle : **{len(labels)}**\n- Audit : **PENDING (400)**\n")
    names=["queries.parquet","labels.parquet","fold_assignments.parquet","crm_ok_gt_v2.csv","independent_audit_sample_400.csv","report.md"]
    manifest={"schema_version":SCHEMA_VERSION,"build_id":bid,"created_at":datetime.now(timezone.utc).isoformat(),"build_identity":identity,"counts":counts,
      "qualification":{"retrieval_inputs_used":False,"insee_authoritative":True,"postcode_fallback_only_when_insee_missing":True},
      "audit_gate":{"required_rows":400,"required_certain_false_labels":0,"status":"PENDING_INDEPENDENT_REVIEW"},"outputs":{n:sha256(tmp/n) for n in names}}
    (tmp/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); os.replace(tmp,dest); return dest

def main():
    p=argparse.ArgumentParser()
    for name in ("base_population","new_increment","existing_crm","sirene","output_root"): p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--seed",type=int,default=42); p.add_argument("--audit-seed",type=int,default=42)
    p.add_argument("--audit-exclude-sample",type=Path)
    p.add_argument("--expected-new-rows",type=int,default=20209)
    print(build(p.parse_args()))
if __name__=="__main__": main()
