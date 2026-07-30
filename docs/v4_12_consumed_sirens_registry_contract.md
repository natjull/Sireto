# Contrat V4.12 — Registre immuable des SIREN consommés

Statut : préenregistré avant construction du registre et avant ouverture de
tout futur CRM.

Identifiant : `V412_CONSUMED_SIRENS_REGISTRY`.

## 1. Objet et frontière

Ce registre ferme l'ensemble des identités SIREN déjà consommées par les
labels, oracles, qualifications et lignées CRM historiques utilisés jusqu'aux
expériences V4.11 et V4.12. Il sert uniquement à interdire qu'une future preuve
de généralisation réutilise une entité déjà connue.

Le builder :

- n'ouvre aucun futur CRM ;
- n'importe et n'exécute aucun retrieval, ranker, accepteur ou autre modèle ;
- ne lit aucun rang, score, hit ou décision pour sélectionner un SIREN ;
- ne transforme jamais un candidat ou une prédiction en identité consommée ;
- lit chaque entrée par la projection physique minimale épinglée dans le plan.

Les manifests d'événement sans payload SIREN sont vérifiés et enregistrés
comme preuves de consommation, mais ne créent aucune ligne d'identité.

## 2. Catalogue fermé

Le catalogue canonique est
`config/v4_12_consumed_sirens_registry_plan.json`. Il épingle, pour chaque
source porteuse :

- chemin absolu, taille, SHA-256 et nombre de lignes ;
- schéma physique des seules colonnes projetées ;
- clé de ligne, champs SIRET/SIREN, rôle d'identité et scope de consommation ;
- filtre déterministe éventuel.

Il épingle séparément les manifests d'événement lus en octets complets.
Toute source, projection ou règle absente du catalogue est interdite. Toute
divergence de chemin, type, nullabilité, taille, nombre de lignes ou hash
produit `STOP_INPUT_DRIFT`.

Les sources porteuses couvrent :

1. les 23 384 lignes historiques du registre V4.11 ;
2. les 225 lignes `UNSEEN` ensuite consommées par le challenge V4.11 ;
3. le benchmark fermé historique ;
4. les labels V4-Fresh actuels, historiques et V3 ;
5. les labels V4.6 effectivement consommés par le développement V4.11/V4.12 ;
6. les labels mécaniques du challenge V4.11 ;
7. l'ancien holdout final ouvert et ses identités d'oracle ;
8. l'oracle unitaire V4.12.

Les manifests de preuve directe et d'évaluation de la garde V4.12 restent
épinglés comme événements de consommation. Leurs candidats, sondes et
preuves `sole_direct_*` ne fournissent aucune identité au registre.

## 3. Identités autorisées

Les rôles autorisés sont :

```text
INPUT_LINEAGE
GROUND_TRUTH_CURRENT
GROUND_TRUTH_HISTORICAL
GROUND_TRUTH_V3
EVALUATION_ORACLE
```

Une valeur SIREN directe est normalisée par Unicode NFKC puis `trim`. Elle
n'est admise que si elle correspond exactement à `^[0-9]{9}$`. Aucune
conversion numérique, notation scientifique, suppression de ponctuation ou
complétion par zéros n'est autorisée.

Une valeur SIRET est traitée de la même façon et n'est admise que si elle
correspond exactement à `^[0-9]{14}$`. Le SIREN dérivé est constitué des neuf
premiers chiffres.

Lorsqu'un rôle fournit à la fois SIRET et SIREN :

- deux valeurs valides doivent être cohérentes ;
- une SIREN absente peut être remplacée par le préfixe d'un SIRET valide ;
- un SIRET absent peut laisser une SIREN directe valide ;
- une incohérence produit `STOP_SIRET_SIREN_MISMATCH`.

Aucun contrôle Luhn n'est appliqué. Les valeurs nulles, vides ou invalides
sont exclues du registre final mais comptées exhaustivement par source, champ
et motif dans `rejected_values.parquet`.

## 4. Filtres par rôle

- Une vérité non nulle est consommée quel que soit le label qui la transporte.
  `MATCH_EXACT` exige néanmoins une identité valide et cohérente.
- Une SIREN autoritative connue pour un cas ambigu est consommée.
- Les champs typés Arrow `null` sont projetés et audités, mais ne peuvent
  produire aucune identité.

## 5. Exclusions obligatoires

Sont toujours exclus, même s'ils contiennent un SIRET ou un SIREN :

- `candidate_siret`, `candidate_siren` et listes de candidats ;
- `selected_active_siret`, `selected_active_siren`, dérivés d'un singleton
  de candidats du snapshot et non d'une vérité autoritative indépendante ;
- `sole_direct_siret`, `sole_direct_siren` et preuves candidates directes ;
- `diagnostic_probe_siret` et toute autre sonde technique ;
- `predicted_siret`, `predicted_siren`, top-1 et scènes ;
- sorties retrieval, rangs, scores, hits et décisions ;
- voisinages ou univers complets du snapshot SIRENE ;
- `direct_active_sirets_json`, qui décrit plusieurs preuves possibles ;
- `siren_component_id`, qui est un hash de composante et non un SIREN.

Une valeur ne peut entrer dans le registre que par un champ et un rôle
explicitement déclarés dans une source porteuse du plan.

## 6. Observations et déduplication

Le builder produit d'abord une observation pour chaque identité valide :

```text
siren
identity_role
consumption_scope
source_id
source_path
source_sha256
source_manifest_sha256
source_record_locator
source_field
label_kind
derivation
observation_key_sha256
```

`source_record_locator` est la représentation canonique de `query_id` ou de
`source_row_number`. `derivation` vaut `DIRECT_SIREN` ou `SIRET_PREFIX`.

`observation_key_sha256` est le SHA-256 du JSON canonique :

```text
[
  source_sha256,
  source_record_locator,
  source_field,
  identity_role,
  siren
]
```

Seules les observations possédant la même clé sont dédupliquées. Deux sources
différentes produisant le même SIREN conservent deux provenances.

`consumed_sirens.parquet` contient ensuite une ligne par SIREN distinct,
triée par ordre ASCII croissant, avec :

```text
siren
provenance_count
identity_roles_json
consumption_scopes_json
source_ids_json
provenance_payload_sha256
```

Les tableaux JSON sont triés et sans doublon.
`provenance_payload_sha256` ferme la liste ordonnée des
`observation_key_sha256` du SIREN.

## 7. Sorties et build ID

Le build est publié sous :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/registries/
v4_12_consumed_sirens/<build_id>/
```

Il contient exactement :

```text
sources.json
observations.parquet
consumed_sirens.parquet
rejected_values.parquet
manifest.json
```

Le `build_id` est constitué des 16 premiers caractères du SHA-256 du JSON
canonique comprenant :

- `schema_version` ;
- version des règles de normalisation ;
- chemin, SHA-256, projection et rôles de toutes les sources porteuses dans
  l'ordre du plan ;
- chemin et SHA-256 de tous les manifests d'événement dans l'ordre du plan ;
- SHA-256 du contrat, du plan et du futur builder ;
- commit Git du builder.

Le timestamp, le chemin temporaire et les métadonnées de runtime non
fonctionnelles sont exclus du calcul.

## 8. Scellement sans autoréférence

`manifest.json` ne contient pas son propre SHA-256. Il contient :

- l'identité complète du build ;
- tous les pins d'entrée ;
- les volumes et rejets par source/rôle ;
- le hash logique des observations ;
- le hash logique de la liste SIREN ;
- pour les quatre autres fichiers, chemin relatif, taille et SHA-256 ;
- `tree_payload_sha256`, calculé sur la map canonique
  `relative_path -> {size_bytes, sha256}` des quatre autres fichiers.

Le SHA-256 externe de `manifest.json` est calculé après fermeture puis publié
dans le receipt de promotion et dans le plan d'intake V4.12. Le tree hash
n'inclut jamais `manifest.json`, ce qui évite toute autoréférence.

## 9. Durabilité et publication

Tous les chemins sont résolus depuis `/` avec `openat`, `O_NOFOLLOW` et
`O_CLOEXEC`. Les fichiers d'entrée sont lus depuis le même descripteur entre
les hashes avant/après. Les sorties temporaires sont créées sur le même
filesystem cible avec `O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC`.
Le runner fixe `umask(0077)`, crée tous les répertoires privés en `0700` et
tous les fichiers en `0600`.

Le build byte-identique utilise exclusivement macOS arm64, Python `3.14.3` et
PyArrow `23.0.0`, sans pandas pour sérialiser. Avant écriture, chaque table est
castée vers le schéma Arrow exact et réassemblée avec un chunk par colonne.
Les trois fichiers Parquet utilisent tous : Parquet `2.6`, compression ZSTD
niveau `9`, dictionnaire désactivé, statistiques activées, data page `1.0`,
stockage du schéma Arrow et row groups de taille maximale `65 536`. L'ordre
des lignes et des colonnes est celui du plan ; le nombre physique de row
groups vaut `ceil(nombre_de_lignes / 65 536)`, avec un unique row group pour
une table vide. Les métadonnées de schéma Arrow sont nulles. `sources.json`
est le JSON canonique UTF-8 du plan, clés triées, séparateurs compacts,
`allow_nan=false`, sans échappement ASCII et avec un unique LF final.

Ordre de durabilité macOS :

1. écrire puis `fsync` et `F_FULLFSYNC` chaque fichier ;
2. `fsync` du répertoire temporaire ;
3. validation complète des schémas, hashes et invariants ;
4. promotion non destructive par `renameatx_np(..., RENAME_EXCL)` ;
5. `fsync` puis `F_FULLFSYNC` du répertoire parent.

Un build existant n'est jamais écrasé, supprimé ou complété. La récupération
après crash consiste uniquement à valider et promouvoir un arbre temporaire
déjà complet. Elle ne relit pas les sources pour reconstruire partiellement.

## 10. Invariants bloquants

Le build est valide uniquement si :

- les 23 384 lignes `consumed` et les 225 lignes `unseen` sont toutes lues,
  sans chevauchement, soit 23 609 lignées CRM consommées ;
- chaque source et chaque manifest correspond exactement à son pin ;
- toutes les projections sont exactes et aucune colonne non déclarée n'est
  chargée ;
- chaque SIREN final contient exactement neuf chiffres ;
- chaque observation possède une provenance et une clé uniques ;
- l'union des observations reproduit exactement `consumed_sirens.parquet` ;
- chaque SIREN final possède au moins une observation ;
- toutes les incohérences SIRET/SIREN et valeurs rejetées sont comptées ;
- aucune identité ne provient d'un candidat ou d'une prédiction ;
- le rebuild avec les mêmes entrées est logiquement et byte-for-byte
  identique pour les quatre fichiers hors manifeste, sous les pins exacts de
  runtime, schémas, writer, ordre et code du plan ;
- l'arbre publié contient exactement les cinq fichiers attendus.

Toute violation produit `STOP_CONSUMED_SIRENS_INTEGRITY`.

## 11. Usage par l'intake frais

Le processus de qualification du futur holdout ne reçoit que :

- `consumed_sirens.parquet`, projeté sur la seule colonne `siren` ;
- le manifeste du registre et son hash externe.

Il ne reçoit jamais `observations.parquet`, les labels historiques ou les
manifests d'événement. L'état `READY` du futur intake est interdit tant que ce
registre n'est pas construit, audité et épinglé. Une collision avec une
vérité `MATCH_EXACT` ou une SIREN autoritative connue d'un cas ambigu entraîne
l'exclusion scientifique prévue par le contrat d'intake.
