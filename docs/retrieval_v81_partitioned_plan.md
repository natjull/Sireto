# V81 — jointures partitionnées, avant mesure

V81_a CTAS et V81_b COPY ont atteint le même plafond spill1GB pendant la
double jointure large. Le sink streaming seul ne réduit pas assez les deux
tables de hash. Garder code/config/logs de ces tentatives inchangés.

Plan physique : partitionner les observations sur predecessor%16, entier
utilisé pour stockage/jointure uniquement.16 buckets disjoints couvrent TOUTE
la population éligible ; aucune sélection CRM/label/état/géo/nom. Dans chaque
bucket, prédécesseurs limités au même bucket et successeurs réduits par
semi-jointure exacte des IDs présents. Même SELECT et deux LEFT JOIN ; mêmes
NULL/absences/classes/provenances, pas de limite de lignes. Fichiers Parquet
intermédiaires concaténés logiquement ; couverture complète9730972 vérifiée.

Ce partitionnement est une optimisation I/O/hash, pas un pool candidat ni
un score de similarité par identifiant. Test de parité exhaustive avec CTAS
et COPY, y compris doublons source, sites fermés, absent, preuve incomplète,
changements d'adresse et aucun sibling ajouté. Conserver les16 fichiers
hashés avec le manifeste source, pas un sous-ensemble à évaluer.

Budgets inchangés80s/4Gio/zéro swapout, DuckDB1GB/2threads/spill1GB,
résultat+TOUS intermédiaires+spill≤2Gio, supervision90s. STOP sans élargir
si cela ne suffit pas. Aucun fit, nom transféré ou recall initial mesuré.
