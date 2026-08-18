# Dossier SIREN officiel v3 — architecture Parquet/DuckDB

Le dossier SIREN est la couche canonique commune à tous les outils SIRETO. Il
ne remplace pas les sources brutes : il les projette sous forme de preuves
officielles dédupliquées, datées et traçables.

## Grain et tables

- `legal_units.parquet` : une unité légale par SIREN, issue du stock SIRENE
  courant ;
- `establishments.parquet` : les SIRET rattachés à leur SIREN, leur état et
  leur site courant ;
- `name_evidence.parquet` : noms légaux, usuels, enseignes et noms historiques,
  avec source, identifiant de preuve/record, priorité, date d'observation et
  intervalle de validité ;
- `address_evidence.parquet` : adresses courantes et historiques, avec source,
  géographie, provenance du record, date d'observation et intervalle de
  validité ;
- `entity_evidence.parquet` : état, rôle siège/principal, dates de validité et
  fraîcheur de chaque observation RNE/BODACC, sans dirigeant ni texte libre ;
- `address_site_resolution.parquet` : résolution prudente d'une adresse portée
  au niveau SIREN vers un SIRET uniquement si l'adresse et la géographie
  correspondent à un site unique ;
- `relations.parquet` : successions et cessions structurées, reliées au record
  officiel qui les porte ;
- `rne_account_deposits.parquet` : dépôts de comptes au grain SIREN, dates,
  type, devise, confidentialité et présence d'un bilan structuré. Les cellules
  détaillées de liasse ne sont pas intégrées à ce stade ;
- `siren_summary.parquet` : agrégats descriptifs sans label ni score modèle ;
- `dossier.duckdb` : catalogue portable des Parquet immuables. À l'ouverture,
  le helper crée des vues temporaires vers les fichiers frères ; aucun chemin
  absolu de la machine de build n'est persisté.

Le répertoire est content-addressé par les SHA-256, tailles et rôles de toutes
ses entrées, sans dépendre des chemins absolus de la machine.
BODACC ne crée jamais automatiquement une identité SIRET : une adresse BODACC
portée par un SIREN reste une preuve SIREN tant qu'un site unique n'est pas
établi. Les dirigeants, bénéficiaires effectifs et textes libres sont exclus.

Le stock RNE formalités est lu directement dans son ZIP, tableau JSON en
streaming. Une formalité produit une preuve SIREN pour l'entreprise puis des
preuves SIRET distinctes pour l'établissement principal, l'établissement
modifié et chaque autre établissement explicitement identifié. SIRENE reste
l'autorité d'existence et d'état courant; RNE ajoute des preuves datées.

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

Les comptes annuels sont `held_out_structured` dans le manifest v3 : ils sont
stockés et auditables, mais exclus du retrieval, des textes de fusion et des
features modèles tant qu'un addendum de politique et une ablation train/dev ne
les autorisent pas. Ils ne sont jamais attribués à un SIRET.

Les anciens modèles et bundles restent gelés. Le nouveau chemin est opt-in
jusqu'au gate retrieval ; ranker, decider, risk et fusion ne sont pas réentraînés
avant Recall@100 exact >= 99 % sur le périmètre identifiable.

## Temporalité

À l'entraînement et à l'évaluation, seules les preuves dont `valid_from` est
antérieur à la date de référence CRM sont visibles. Une preuve sans date peut
être utilisée comme signal faible et doit être marquée comme telle. Aucun rang,
score ou label CRM n'entre dans le dossier.

## Portefeuille de noms v3

Le dossier ne matérialise plus un `bag of names` unique. Les valeurs sont
dédupliquées avec leur provenance, classées par actualité, priorité officielle,
consensus et date, puis plafonnées par rôle :

- `LEGAL_CURRENT` : 4 noms par SIREN ;
- `TRADE_CURRENT` : 8 noms par SIREN ;
- `SITE_CURRENT` : 6 noms par SIRET et 12 par SIREN agrégé ;
- `HISTORICAL` : 6 par SIRET et 12 par SIREN ;
- `SUPPORTING` : 4 par SIRET et 6 par SIREN.

Seuls les trois rôles actuels alimentent les champs exacts. Les noms et
adresses historiques ainsi que les preuves BODACC sont des canaux de rescue.
Tantivy fournit une union brute bornée à 2 000 : il ne réserve plus de places
et ne fabrique plus de top100. L'admission LambdaMART protège ensuite les
exacts actuels et les consensus avant de compléter déterministiquement jusqu'à
100. Les relations BODACC structurées de succession/cession sont suivies sur
un saut seulement, puis filtrées par INSEE/CP et ordonnées par adresse avant le
statut de siège.

## Features officielles supplémentaires

La projection `(query_id,candidate_siret)` expose, sans identifiant modèle :

- similarités et exactitudes distinctes pour nom légal actuel, nom commercial,
  enseigne du site, historique, RNE et BODACC ;
- similarités distinctes pour adresse SIRENE actuelle, historique, RNE et
  BODACC ;
- nombre de sources concordantes, variantes actuelles et conflits d'état ;
- consensus sur le siège, fraîcheur de la preuve et résolution d'adresse vers
  un site unique ;
- nombres de successions et de transferts d'actifs structurés ;
- activité, forme juridique, âge et complexité multi-sites issus de SIRENE.

Ces colonnes sont disponibles pour l'admission retrieval. Le ranker, le
decider, le risk model et les modèles de fusion restent gelés tant que le gate
Recall@100 n'est pas franchi.

## Données effectivement disponibles

- BODACC : historique annuel 2008–2026, 54 099 311 preuves fusionnées et
  692 010 relations structurées dans le build local ;
- RNE : stock national formalités NIVEAU1 scellé (15,1 Go compressés), avec
  entreprise et établissements principal/modifié/autres ; l'ingestion v3 est
  streaming et multi-SIRET ;
- comptes RNE : métadonnées de dépôt conservées au grain SIREN mais exclues de
  tous les consommateurs modèles ;
- SIRENE : autorité de l'existence, de l'état et de l'adresse courante.

Le chemin v3 est donc complet pour noms, enseignes, adresses, temporalité,
consensus et relations. Il ne prétend pas exploiter les pièces RNE, dirigeants,
bénéficiaires effectifs, texte libre BODACC ou cellules de comptes.
