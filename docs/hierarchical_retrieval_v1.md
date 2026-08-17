# Retrieval hiérarchique SIRET v1

Statut : code livré, index national et évaluation dev non exécutés.

Ce moteur est un chemin opt-in isolé du retrieval TF-IDF historique. Il ne
consomme aucun alias ni label CRM. Il indexe les noms/adresses SIRENE courants,
les valeurs historiques officielles lorsqu'elles sont fournies, les liens de
succession officiels dans les deux sens, et des documents agrégés
`(INSEE, SIREN)`.

Le runtime applique `INSEE`, puis `CP` seulement en fallback, six canaux BM25,
exact et q-grams, une expansion des cinq meilleurs SIREN limitée à 32 sites par
SIREN, puis une fusion déterministe. L'union interne est plafonnée à 1 000 et
la sortie à 100. L'adresse et le numéro précèdent le statut de siège.

Le backend de production est Tantivy 0.25.1. Le backend mémoire est réservé aux
tests. Le builder DuckDB joint et parcourt les deux snapshots courants en flux,
écrit par lots et produit un répertoire adressé par le contenu avec manifeste.
Un build sans les trois sources temporelles reste possible, mais le manifeste
porte alors `temporal_complete=false` et liste les rôles manquants.

`scripts/evaluate_hierarchical_retrieval.py` est l'unique entrée d'évaluation
bornée. Elle publie Recall@100 exact et opérationnel séparément, le maximum de
candidats, les latences p50/p95/p99 et un hash de scellement. Elle refuse un
index de production temporellement incomplet.

Les tests ciblés couvrent le fallback géographique, les codes INSEE corses,
l'historique et les successions, l'expansion, le classement des sites, le
plafond/déterminisme et la séparation stricte des métriques exacte et
opérationnelle. Aucun build 42 M ni aucune ouverture du fold test n'a été lancé.
