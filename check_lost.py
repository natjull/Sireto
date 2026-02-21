import duckdb
import os
import glob

con = duckdb.connect()

print("1. Checking if closed SIRET 18004301000545 is in candidates_v7_all/insee/insee=12202")
try:
    files = glob.glob('data/candidates_v7_all/insee/insee=12202/*.parquet')
    if not files:
        print("No parquet files found for this insee partition.")
    else:
        df = con.execute(f"SELECT siret, insee FROM read_parquet('{files[0]}') WHERE siret = '18004301000545'").df()
        if len(df) == 0:
            print("SIRET is NOT in the partition. It was excluded during partition build!")
        else:
            print("SIRET IS in the partition:", df.to_dict('records'))
except Exception as e:
    print("Error:", e)

print("\n2. Checking full names for PRUNED_BY_TFIDF cases in DB")
query = """
    SELECT e.siret, e.etatAdministratifEtablissement as etat, e.codeCommuneEtablissement as insee, 
           e.enseigne1Etablissement as enseigne, ul.denominationUniteLegale as denom_ul, ul.nomUniteLegale as nom_ul
    FROM read_parquet('data/StockEtablissement_utf8.parquet') e
    LEFT JOIN read_parquet('data/StockUniteLegale_utf8.parquet') ul ON e.siren = ul.siren
    WHERE e.siret IN ('90877009200017', '41014039600038', '32104275600015')
"""
df_names = con.execute(query).df()
print(df_names.to_string())
