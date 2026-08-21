from __future__ import annotations
from pathlib import Path
import csv, gzip, json
import pyarrow.parquet as pq
from src.xgb_matcher.official_sector_evidence import iter_administrations,iter_education,iter_finess,write_sector_evidence

def test_sector_extractors_require_structured_identifiers(tmp_path:Path)->None:
    f=tmp_path/"finess.json.gz"
    with gzip.open(f,"wt") as out: json.dump({"pmej":[{"informationsGeneralesPMEJ":{"siren":"123456789","denominationPm":"HOPITAL TEST","numFinessPm":"01"},"adresse":[{"cogCommune":"75056","codePostal":"75001"}],"ege":[{"informationsGeneralesEGE":{"siret":"12345678900011","nomEgeLong":"CLINIQUE TEST","numFinessEge":"02"},"adresse":[{"cogCommune":"75056","codePostal":"75001","ligneQuatre":"10 RUE TEST"}],"etatObjet":"A"}]}]},out)
    u=tmp_path/"uai.csv"; u.write_text("Identifiant_de_l_etablissement;Nom_etablissement;SIREN_SIRET;Code_commune;Code_postal;etat\n001;ECOLE TEST;98765432100012;75056;75002;OUVERT\n002;NO ID;;75056;75002;OUVERT\n")
    a=tmp_path/"admin.csv"; a.write_text("siren,nom_complet\n111222333,MAIRIE TEST\n")
    rows=[*iter_finess(f),*iter_education(u),*iter_administrations(a)]
    assert {(r["source"],r["subject_kind"]) for r in rows}=={("FINESS","SIRET"),("FINESS","SIREN"),("EDUCATION_UAI","SIRET"),("DINUM_ADMINISTRATION","SIREN")}
    out=tmp_path/"sector.parquet"; assert write_sector_evidence(rows,out)==4
    assert pq.read_table(out).num_rows==4
