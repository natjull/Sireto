# V174 — présence vectorisée sans listes par ligne

Contrat avant mesure,4octobre2026. V173 reproduit0,458/0,434/0,436s
sur le même lot49 contre oracle0,069s. Aucun gain chaud constaté ; ne pas
retenir son kernel list_transform pour accélérer l'exécution nationale.

V174 remplace les listes intermédiaires par des colonnes compact/padded
matérialisées une fois puis des sommes vectorisées de contains bool castés
float64. Groupes physiques de32expressions, TOUS les tokens distincts
consommés même au-delà32 ; compteurs additionnés AVANT division par le
nombre entier de tokens de la requête. Pas de quota métier, perte de mots,
annulation de petites fractions, saturation, float32 ou changement de score.
Mots avec frontières, char3/4 underscores et exact non vide identiques172.
Toutes les lignes/IDs vérifiées à chaque groupe ; temp table propre détruite
avant sortie, sources jamais écrites. Identifiants uniquement joins.

Reprendre toutes les protections/sourceauthorities/tests/protocole173 :
lot49 entier certifié, même première requête query-onlyV115 folds234, trois
timings séparés et égalité entre répétitions ET aux postings exhaustifs,
hashes before/after/epochs, oracle quatre lecteurs, vrai childexit0,
worker1Gio/80s/normal pressure/free50/préflight30s/swap global télémétrie,
DuckDB1thread/256Mio/spill3Gio, freshroot174 sans retry d'anciens résultats.
Fixtures supplémentaires >32tokens et trois oracles scalar/172/postings.
Ce coût d'un lot n'est ni national, ni recall, ni SLO ; cache disque inconnu.
GT/anciens candidats/folds0/1/aval fermés, aucun claim99,2%.
