# V4.12 — Contrat de l'oracle séparé du moteur unitaire

## 1. Objet et frontière

Construire la seule table de vérité autorisée pour évaluer le futur moteur
V4.12 sur les 1 456 requêtes dev historiques.

L'oracle est physiquement séparé du paquet d'entrée
`ca0b22e79cd2e92a32c009266e6d967b4ea48654de8736bca2b0ea7fdc9f8d6e`.
Il n'est jamais remis au worker. Avant le run, seuls le builder dédié et les
auditeurs read-only peuvent ouvrir les sources sensibles et l'oracle publié.
Pendant le run, le worker ne peut pas l'ouvrir. Après scellement des sorties,
seul un évaluateur distinct rejoint ces sorties à l'oracle.

La séparation de répertoires sur un même Mac n'est pas présentée comme une
barrière de sécurité OS. Le futur contrat worker devra donc ajouter une
barrière réelle avec `/usr/bin/sandbox-exec`, interdire les deux racines
oracle/audit et prouver par un test sentinelle qu'une ouverture échoue. Le
worker recevra seulement le paquet runtime, ses modèles et les stores
explicitement autorisés. Un audit du code et des fichiers ouverts complètera
ce contrôle.

L'oracle contient une vérité métier minimale. Il ne contient aucune ancienne
prédiction, candidat, feature, rang, hit, score, décision, raison de review ou
preuve directe. L'ancien pipeline ne devient donc pas une cible de vérité.

## 2. Sources gelées et projections physiques

### IDs dev sûrs

Source :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/inputs/v4_12_unit_engine/
ca0b22e79cd2e92a32c009266e6d967b4ea48654de8736bca2b0ea7fdc9f8d6e/
queries_dev.parquet
```

- taille : 62 365 octets ;
- SHA-256 :
  `b1fef6ba72e4a557175a60c7e21e658f8fbb739b1d6607b77e5f70da873a031f` ;
- projection physique exacte : `query_id`.

Le manifeste runtime associé, SHA-256
`34e7a8d1c97c962b7f2295a99cd45b68c37afa9c866a5eb22e302330cb9010c4`,
est lui-même revalidé avec le file-set runtime exact, le build ID, les
déclarations anti-fuite et le hash de `queries_dev.parquet` avant ouverture
des sources sensibles. Son `GO` indépendant est déjà scellé dans
`reports/v9/v4_12_unit_input_results.md`.

### Labels historiques

Source :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_input_blind/
ec4326ec57e4411d/labels.parquet
```

- 7 003 lignes, 178 217 octets ;
- SHA-256 :
  `69032b745817959422ef26e4c0c1228686260c1daa272ca5d6aba1d7be087b04` ;
- projection physique exacte :
  `query_id,label_kind,ground_truth_siret,ground_truth_siren`.

### Split historique

Source :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_input_blind/
ec4326ec57e4411d/split_assignments.parquet
```

- 7 003 lignes, 170 201 octets ;
- SHA-256 :
  `33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193` ;
- projection physique exacte :
  `query_id,siren_component_id,split`.

Lire d'autres colonnes puis les supprimer est interdit.

## 3. Qualification indépendante du retrieval

La population est la semi-jointure exacte avec les 1 456 `query_id` du paquet
sûr. Chaque ligne doit avoir `split='dev'`.

`dev_partition` est calculé sans modèle :

- `threshold_dev` si le premier octet de
  `SHA-256("v411-threshold:" + siren_component_id)` est inférieur à 128 ;
- `comparison_dev` sinon.

Il est interdit d'utiliser pour qualifier, filtrer ou ordonner une ligne :

- `is_ground_truth` d'un pool candidat ;
- hit, rang, score, top-1 ou présence dans un pool ;
- `sole_direct_siret`, preuve V4.12 ou décision de garde ;
- prédiction ranker, scène, accepteur ou résultat historique.

## 4. Sortie exacte

`oracle_dev.parquet` possède ce schéma Arrow exact, sans metadata :

| Colonne | Type | Nullable |
|---|---|---|
| `query_id` | string | non |
| `dev_partition` | string | non |
| `label_kind` | string | non |
| `ground_truth_siret` | string | oui |
| `ground_truth_siren` | string | oui |

Contraintes :

- 1 456 IDs uniques, exactement égaux et dans le même ordre que
  `queries_dev.parquet` ;
- `label_kind` appartient à `MATCH_EXACT|AMBIGUOUS` ;
- `MATCH_EXACT` possède un SIRET à 14 chiffres et un SIREN à 9 chiffres égal
  aux neuf premiers chiffres du SIRET ;
- `AMBIGUOUS` possède deux vérités nulles ;
- aucune chaîne vide, espace périphérique, NUL ou saut de ligne.

Cardinalités gelées :

| Population | Total | MATCH_EXACT | AMBIGUOUS |
|---|---:|---:|---:|
| `threshold_dev` | 710 | 583 | 127 |
| `comparison_dev` | 746 | 634 | 112 |
| total | 1 456 | 1 217 | 239 |

Ordre, premiers IDs et derniers IDs :

```text
SHA-256("v412-unit-engine:" + query_id), query_id
premiers = 1394, 16308, 2454, 8914, 9265
derniers = 13579, 1174, 9760, 4986, 9307
```

Hash logique ligne par ligne :

```text
query_id
0x00
dev_partition
0x00
label_kind
0x00
ground_truth_siret ou les deux octets "\N"
0x00
ground_truth_siren ou les deux octets "\N"
0x0A
```

- taille du payload : 80 282 octets ;
- SHA-256 :
  `9b25bf45b7ff074a1db8e6ca166569eec9754e34b29f032aebae63925d71ff42`.

Le payload séparé `query_id + 0x0A` mesure 10 299 octets et possède le
SHA-256
`37016d9dc480194b52b545fffb3272847d485c5f209025d064d313b425c03bf4`.

## 5. Publication et preuve

Racine oracle :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/oracles/v4_12_unit_engine/<build_id>
```

Fichiers exacts :

- `oracle_dev.parquet` ;
- `integrity.json` ;
- `manifest.json`.

Racine de preuve séparée :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_unit_oracle/<build_id>
```

Fichiers exacts :

- `data_inputs.parquet`, ledger des quatre fichiers ouverts : IDs sûrs,
  manifeste sûr, labels et split ;
- `provenance.json` ;
- `manifest.json`.

Le `build_id` dépend du plan, du verrou, des sources Git, des quatre inputs,
du runtime et du hash logique de l'oracle. L'audit est publié d'abord,
l'oracle en dernier. Les deux racines sont immuables, sur le même filesystem,
et un artefact orphelin reste `STOP`.

Le ledger publie rôle, chemin absolu, projection, taille et hash avant/après.
Le manifeste oracle ne publie aucun chemin permettant au futur worker de
retrouver l'oracle historique. L'oracle et sa preuve ne sont jamais copiés
dans le paquet runtime.

Schéma Arrow exact du ledger, sept champs non nullables et sans metadata :

```text
role: string
absolute_path: string
projection: string
size_bytes_before: uint64
sha256_before: string
size_bytes_after: uint64
sha256_after: string
```

Les quatre rôles uniques sont `safe_queries_dev`, `safe_runtime_manifest`,
`labels` et `split`. Les lignes sont triées par octets UTF-8 de
`role,absolute_path`.

`integrity.json` possède exactement :

```text
schema_version
build_id
query_count
population_counts
query_id_payload_sha256
truth_logical_sha256
declarations
```

`manifest.json` de l'oracle possède exactement :

```text
schema_version
build_id
safe_input_build_id
files
population_counts
query_id_payload_sha256
truth_logical_sha256
runtime
declarations
historical_development_only
independent_truth
production_certified
```

`files` possède exactement les records `oracle_dev.parquet` et
`integrity.json`. Le record Parquet contient hash, taille, lignes, schéma
complet et metadata ; le record JSON contient hash et taille.

`provenance.json` possède exactement :

```text
schema_version
build_id
git_commit
sources
lock_sha256
runtime
data_input_count
declarations
oracle_manifest_sha256
```

Le manifeste d'audit possède exactement
`schema_version,build_id,files`; `files` scelle exactement
`data_inputs.parquet` et `provenance.json` par hash et taille.

Tous les JSON sont encodés UTF-8 sous forme canonique
`sort_keys=true,separators=(',',':'),allow_nan=false`, avec un unique LF
final et rejet des clés dupliquées.

Le manifeste publie obligatoirement :

```text
historical_development_only = true
independent_truth = false
production_certified = false
```

Déclarations obligatoires :

```text
retrieval_opened = false
candidate_results_opened = false
direct_evidence_opened = false
guard_decisions_opened = false
models_opened = false
challenge_or_final_opened = false
```

Le `build_id` est le SHA-256 hexadécimal complet des octets JSON canoniques
de l'objet suivant :

```text
schema_version = sireto-v4.12-unit-oracle-build-identity-1
plan_sha256
lock_sha256
source_hashes
input_hashes
safe_input_build_id
query_id_payload_sha256
truth_logical_sha256
runtime
```

Les keysets sont exacts, sans valeur implicite. Les deux manifests et
`integrity.json` doivent republier ce même identifiant.

## 6. Exécution sûre

- builder autonome, sans import de code de matching ;
- aucune API, réseau, LLM, GPU ou dépense ;
- aucun fichier challenge, holdout, random V4.8, V4.9 ou test final ;
- chemins et hashes de `candidates_sparse_top100.parquet`,
  `query_audit.parquet`, V4.12 direct-evidence, V4.12 guard-historical,
  ranker, scènes, accepteur et modèles explicitement refusés ;
- fichiers réguliers, aucun symlink, `lstat` de chaque composant ;
- sources et blobs Git liés au verrou ;
- projection, hash et snapshot du même input revalidés avant chaque
  promotion ;
- `python -B`, bytecode désactivé, staging/TMPDIR SSD privé ;
- RSS maximal 8 Gio, `fsync`, renommage atomique, postvalidation ;
- aucun bypass.

Le verrou externe possède exactement :

```text
schema_version
purpose
audit_verdict
git_commit
source_hashes
input_paths
input_hashes
safe_input_build_id
expected_population
expected_id_payload_sha256
expected_truth_logical_sha256
runtime
output_root
audit_output_root
temp_root
max_rss_bytes
```

Il exige le purpose `V4.12_UNIT_ORACLE`, le verdict
`GO_CODE_V412_UNIT_ORACLE`, les quatre inputs et les trois racines propres à
l'oracle.

## 7. Tests et verdicts

Tests minimaux :

- projections réellement limitées aux colonnes autorisées ;
- IDs exactement égaux au paquet sûr, ordre et payload exacts ;
- cardinalités 710/746 et 1 217/239 ;
- règles SIRET/SIREN/null strictes ;
- refus d'une colonne/valeur modifiée, vide ou réinjectée ;
- refus de toute source de candidat, preuve, décision, modèle ou challenge ;
- mutation/restauration pendant lecture détectée ;
- lock, Git, RSS, staging, atomicité et immutabilité ;
- ledger exhaustif de quatre entrées ;
- oracle absent du paquet runtime ;
- profil `sandbox-exec` futur refusant les racines oracle/audit, avec test
  sentinelle d'ouverture réellement refusée.

Verdicts :

- `GO_V412_UNIT_ORACLE` : oracle minimal construit et contre-audité ;
- `STOP_V412_UNIT_ORACLE` : toute divergence ou contamination.

Seul `GO_V412_UNIT_ORACLE` autorise le préenregistrement du store et du moteur
unitaire. Il n'autorise ni l'ouverture de l'oracle par le worker, ni un
benchmark, ni une décision produit. Les ouvertures préalables documentées par
le builder et les auditeurs oracle sont nécessaires à sa construction et ne
constituent pas une ouverture par le worker.
