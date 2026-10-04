# Comparaison indépendante après scellement V195/V196 — sans labels

Ce protocole est figé avant comparaison et indépendant des générateurs,
dont les fichiers et hashes restent immuables. Il exige deux racines SEALED
et leurs admissions réelles : producteurs, audits complets et ledger.
Sources, requêtes, modèles, preuves, coefficients et partitions doivent être
identiques ; seul `output_root` peut différer.

Le diagnostic lit les deux sorties de prédiction scellées, jamais les labels.
Il déclare `old_candidates_opened=True`, exclusivement pour une comparaison
APRÈS génération : aucune feature, recherche de secours ou modification des
sorties. Les générateurs restent `old_candidates_opened=False` et intacts.

L'égalité de tous les candidats, scores, ordres et traces nationales doit être
EXACTE pour les requêtes 0 et 1 ; chaque liste contient au plus 100 SIRET
uniques. Un écart produit NON_PARITY, jamais une sélection opportuniste.
Le rapport est créé dans une nouvelle racine CATNAT, hors des générations
immuables. Les tests utilisent de vrais hashes de petits fichiers et des
admissions de processus explicitement simulées. La comparaison réelle exige
le commit préalable du code, du contrat et des tests, puis SEALED V196.

Ce diagnostic ne mesure ni recall, précision, SLO ou OOF complet ; il ne lit
ni labels ni folds 0/1. Il vérifie seulement l'absence d'impact de cette
optimisation physique sur deux sorties. Les 128 sorties doivent être scellées
avant évaluation séparée ; l'objectif de 99,2 % exact initial reste à atteindre.
