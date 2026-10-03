# V81 streaming — plan physique seulement, avant relance

Tentative V81_a arrêtée au spill1GB pendant CTAS bindings. Les observations
éligibles et endpoints avaient été lus, pas de corpus final scellé ni labels
ouverts. Ancien code, configuration, tentative et logs restent intacts.

Intercepter UNIQUEMENT la matérialisation de la jointure large : même SELECT,
COPY Parquet streaming puis vue sur ce fichier. Libérer ensuite les tables
comme dans le plan original. Aucune modification de jointure, colonne,
classe de site/coverage, population, ordre logique ou choix d'état. Parité
des lignes finales et diagnostics avec CTAS sur fixtures, comparée sans
ordre de stockage. Le fichier intermédiaire contient aussi les endpoints
incomplets éligibles ; il est source-only, pas un index/corpus à évaluer.

Gardes EXACTEMENT identiques :80s/4Gio/zéro swap, DuckDB1GB/2threads,
spill1GB, TOUS fichiers résultat+intermédiaire+spill≤2Gio, supervision90s.
Même9730972 observations et43896818 sites, mêmes hashes complets avant/après,
aucun CRM/label, nom transféré, modèle/aval, fold0/1. Pas de relance élargie.
La compilation éventuelle ne vaut pas gain de rappel ni gate initial100.
