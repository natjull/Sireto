# Dossier SIREN officiel — architecture Parquet/DuckDB

Le dossier SIREN est la couche canonique commune à tous les outils SIRETO. Il
ne remplace pas les sources brutes : il les projette sous forme de preuves
officielles dédupliquées, datées et traçables.

## Grain et tables

- `legal_units.parquet` : une unité légale par SIREN, issue du stock SIRENE
  courant ;
- `establishments.parquet` : les SIRET rattachés à leur SIREN, leur état et
  leur site courant ;
- `name_evidence.parquet` : noms légaux, usuels, enseignes et noms historiques,
  avec source et intervalle de validité ;
- `address_evidence.parquet` : adresses courantes et historiques, avec source,
  géographie et intervalle de validité ;
- `address_site_resolution.parquet` : résolution prudente d'une adresse portée
  au niveau SIREN vers un SIRET uniquement si l'adresse et la géographie
  correspondent à un site unique ;
- `relations.parquet` : successions et cessions structurées ;
- `siren_summary.parquet` : agrégats descriptifs sans label ni score modèle ;
- `dossier.duckdb` : catalogue portable des Parquet immuables. À l'ouverture,
  le helper crée des vues temporaires vers les fichiers frères ; aucun chemin
  absolu de la machine de build n'est persisté.

Le répertoire est content-addressé par les SHA-256 de toutes ses entrées.
BODACC ne crée jamais automatiquement une identité SIRET : une adresse BODACC
portée par un SIREN reste une preuve SIREN tant qu'un site unique n'est pas
établi. Les dirigeants, bénéficiaires effectifs et textes libres sont exclus.

## Utilisation par les modèles

La projection commune est indexée par `(query_id, candidate_siret)` :

- **retrieval** : noms/adresses multi-sources, historique, relations et route
  hiérarchique SIREN vers ses sites ;
- **ranker XGBoost** : similarités maximales nom/adresse, accords INSEE/CP,
  état du site, nombre de sources, ancienneté et ambiguïté ;
- **decider** : force et diversité des preuves, marge top-1/top-2 calculée en
  aval, résolution de site et contradictions ;
- **risk** : taille/complexité du dossier, nombre de sites, désaccords entre
  sources, temporalité et ambiguïtés ;
- **BGE/CamemBERT/fusion** : vues textuelles séparées par champ et source avec
  masques de provenance. On ne concatène pas aveuglément tout le dossier.

Les anciens modèles et bundles restent gelés. Le nouveau chemin est opt-in
jusqu'au gate retrieval ; ranker, decider, risk et fusion ne sont pas réentraînés
avant Recall@100 exact >= 99 % sur le périmètre identifiable.

## Temporalité

À l'entraînement et à l'évaluation, seules les preuves dont `valid_from` est
antérieur à la date de référence CRM sont visibles. Une preuve sans date peut
être utilisée comme signal faible et doit être marquée comme telle. Aucun rang,
score ou label CRM n'entre dans le dossier.
