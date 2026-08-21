from __future__ import annotations
from pathlib import Path
import csv, gzip, json
import pyarrow.parquet as pq
from src.xgb_matcher.official_sector_evidence import iter_administrations,iter_associations,iter_data_es,iter_education,iter_finess,write_sector_evidence

def test_sector_extractors_require_structured_identifiers(tmp_path:Path)->None:
    f=tmp_path/"finess.json.gz"
    with gzip.open(f,"wt") as out: json.dump({"pmej":[{"informationsGeneralesPMEJ":{"siren":"123456789","denominationPm":"HOPITAL TEST","numFinessPm":"01"},"adresse":[{"cogCommune":"75056","codePostal":"75001"}],"ege":[{"informationsGeneralesEGE":{"siret":"12345678900011","nomEgeLong":"CLINIQUE TEST","numFinessEge":"02"},"adresse":[{"cogCommune":"75056","codePostal":"75001","ligneQuatre":"10 RUE TEST"}],"etatObjet":"A"}]}]},out)
    u=tmp_path/"uai.csv"; u.write_text("Identifiant_de_l_etablissement;Nom_etablissement;SIREN_SIRET;Code_commune;Code_postal;etat\n001;ECOLE TEST;98765432100012;75056;75002;OUVERT\n002;NO ID;;75056;75002;OUVERT\n")
    a=tmp_path/"admin.csv"; a.write_text("siren,nom_complet\n111222333,MAIRIE TEST\n")
    rows=[*iter_finess(f),*iter_education(u),*iter_administrations(a)]
    assert {(r["source"],r["subject_kind"]) for r in rows}=={("FINESS","SIRET"),("FINESS","SIREN"),("EDUCATION_UAI","SIRET"),("DINUM_ADMINISTRATION","SIREN")}
    out=tmp_path/"sector.parquet"; assert write_sector_evidence(rows,out)==4
    assert pq.read_table(out).num_rows==4


def test_data_es_and_association_require_structured_business_ids(tmp_path:Path)->None:
    sports=tmp_path/"data-es.csv"
    sports.write_text(
        "Numéro de l'équipement sportif;Nom de l'équipement sportif;Nom de l'installation sportive;SIRET Installation;Adresse;Commune INSEE;Code Postal\n"
        "E1;GYMNASE ANNEXE;COMPLEXE SPORTIF;12345678900011;10 RUE DES SPORTS;75056;75001\n"
        "E2;TERRAIN SANS ID;STADE;;20 RUE TEST;75056;75002\n",
        encoding="utf-8",
    )
    associations=tmp_path/"associations.jsonl"
    associations.write_text(
        json.dumps({
            "data": {
                "rna":"W751000001", "siren":"987654321",
                "siret_siege":"98765432100029", "titre":"CLUB DES TESTS",
                "objet":"CE TEXTE LIBRE NE DOIT JAMAIS ETRE INDEXE",
                "adresse_siege":{"adresse":"5 RUE DU CLUB","code_insee":"75056","code_postal":"75003"},
                "etablissements":[{"siret":"98765432100037","nom":"ANTENNE CLUB","adresse":{"adresse":"7 RUE DU CLUB","code_insee":"75056","code_postal":"75004"}}],
            }
        })+"\n"+json.dumps({"data":{"rna":"W751000002","titre":"SANS SIREN"}})+"\n",
        encoding="utf-8",
    )

    sport_rows=list(iter_data_es(sports))
    association_rows=list(iter_associations(associations))

    assert {(row["source"],row["siret"]) for row in sport_rows}=={
        ("DATA_ES","12345678900011")
    }
    assert {row["siret"] for row in association_rows}=={
        "98765432100029","98765432100037"
    }
    assert all("CE TEXTE LIBRE" not in row["name"] for row in association_rows)
