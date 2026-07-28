# V4.12 — Contrat des entrées sûres du moteur unitaire

## 1. Objet

Construire un paquet d'entrée physiquement séparé de tout oracle, label,
score ou décision avant d'implémenter le moteur V4.12 requête par requête.

Ce paquet contient uniquement :

- les six champs CRM autorisés ;
- un inventaire cryptographique des partitions candidates ;
- un inventaire cryptographique du cache TF-IDF V4.11.

Il ne contient aucun candidat retrouvé, feature, score, modèle, preuve,
vérité, SIRET/SIREN CRM ou résultat historique.

## 2. Sources CRM et projection physique

Sources gelées :

- `queries.parquet`, 7 003 lignes, SHA-256
  `3a47aef768cee1436ad77a6e114defe50e685b7495f0e75137e9fd06dfe9fc68` ;
- `split_assignments.parquet`, SHA-256
  `33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193`.

Le builder lit physiquement, par projection Parquet :

```text
queries:
  query_id
  crm_name
  crm_address
  crm_postcode
  crm_city
  crm_insee

split_assignments:
  query_id
  split
```

Il est interdit de lire les autres colonnes puis de les supprimer. Sont
notamment interdits : `crm_record_id`, les trois champs `*_norm`,
`siren_component_id`, `oof_fold`, tout SIRET/SIREN source, label, vérité,
candidat, hit, rang, score, segment ou décision.

## 3. Populations publiées

### `queries_all.parquet`

- 7 003 `query_id` uniques ;
- schéma exact des six `VARCHAR` ci-dessus ;
- ordre
  `sha256('v412-unit-evidence:' || query_id), query_id` ;
- payload `query_id + byte 0x0A` : 50 396 octets ;
- SHA-256 du payload :
  `8dc263fb5da7d609b0ec43d238b6f28283b3aec6694cf79c774536103faef762`.

Premiers IDs : `7417, 788, 103, fresh:AC019060, 4526`.
Derniers IDs : `13075, 2196, 9089, 10240, 13883`.

### `queries_dev.parquet`

Semi-jointure sur les seuls IDs dont `split='dev'` :

- 1 456 `query_id` uniques ;
- même schéma physique ;
- ordre
  `sha256('v412-unit-engine:' || query_id), query_id` ;
- payload `query_id + byte 0x0A` : 10 299 octets ;
- SHA-256 du payload :
  `37016d9dc480194b52b545fffb3272847d485c5f209025d064d313b425c03bf4`.

Premiers IDs : `1394, 16308, 2454, 8914, 9265`.
Derniers IDs : `13579, 1174, 9760, 4986, 9307`.

Le builder vérifie aussi 5 547 lignes fit et 1 456 lignes dev, sans publier
le split.

Après réouverture, il compare chaque valeur des six colonnes, ligne par
ligne, à la projection physique des sources épinglées. Les deux sorties ne
contiennent aucun null, aucun metadata Arrow supplémentaire et utilisent un
schéma explicitement non nullable. Une colonne CRM vidée ou altérée arrête le
build même si les `query_id` restent exacts.

## 4. Inventaire des partitions

Racine gelée : `data/candidates_v7_all`.

Attendus :

- 4 119 fichiers réguliers, aucun symlink ;
- 1 969 745 065 octets ;
- signature historique
  `2f6668f60da8bc9fe52b683b32ef35641803679c01f8c8fd124e2e86a41e2b82` ;
- 27 594 915 lignes déclarées ;
- `manifest/insee_counts.parquet`, 175 506 octets, SHA-256
  `a07bf9cd084f2f8e4842c30b545a913a93da7096a5d9c9a81d3f48c8b866ab0a` ;
- `manifest/postcode_counts.parquet` est vide, SHA-256 du fichier vide, et
  ne doit pas être désérialisé.

`partition_inventory.parquet` publie, dans l'ordre du chemin relatif :

```text
relative_path: VARCHAR
size_bytes: UBIGINT
sha256: VARCHAR
```

Le hash logique de l'arbre est calculé sur chaque record canonique :

```text
relative_path UTF-8
byte 0x00
size décimal ASCII
byte 0x00
sha256 ASCII
byte 0x0A
```

Tous les fichiers sont rehashés après construction des sorties et avant
publication. Le hash logique attendu de cet inventaire est :

```text
680f1884879bfa5b8cf2c335a0658604010e3d4c546ed6eaeb2e2ef34c954463
```

Les 27 594 915 lignes sont la somme des footers des 4 117 Parquet situés
uniquement sous `insee/` et `cp/`. Les deux fichiers `manifest/` n'entrent
jamais dans ce total.

## 5. Inventaire du cache TF-IDF

Racine gelée :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/cache/v46_aligned_b/v411_verified/
296c7891107249a073c00d93c7310c55a652243de4bcfa7165d09dbfc3349a82
```

Attendus :

- 1 454 pickle + 1 454 sidecars, soit 2 908 fichiers ;
- 6 730 554 690 octets ;
- namespace
  `296c7891107249a073c00d93c7310c55a652243de4bcfa7165d09dbfc3349a82` ;
- hash logique TF-IDF
  `92b68d1f7aa386f181edbede280e58df72f8583d7663419d77da88300d241c61` ;
- hash config sparse
  `aeaa671959fc00dcec2e8a5393976d1e68da9dfa5ae48ef4d836e9dbdc3c564e`.

Pour chaque clé, le builder vérifie sans désérialiser le pickle :

- `safe_key = partition_key.replace('|','_').replace('/','_').replace('\\','_')` ;
- pickle exact `<safe_key>.pkl` et sidecar exact
  `<safe_key>.pkl.sha256.json` ;
- `safe_key` conforme à `^(?:[0-9]{5}_|_[0-9]{5})$` sur ce snapshot ;
- unicité de la clé, du `safe_key` et des deux chemins, sans collision ;
- sidecar `sireto-tfidf-cache-integrity-1` ;
- JSON lu avec rejet des clés dupliquées et ensemble exact
  `schema_version, config_hash, partition_key, size_bytes, sha256` ;
- `config_hash`, `partition_key`, taille et SHA-256 déclarés ;
- égalité avec le fichier pickle réel ;
- aucun fichier supplémentaire.

`tfidf_inventory.parquet` contient un record par clé :

```text
partition_key: VARCHAR
pickle_relative_path: VARCHAR
pickle_size_bytes: UBIGINT
pickle_sha256: VARCHAR
sidecar_relative_path: VARCHAR
sidecar_size_bytes: UBIGINT
sidecar_sha256: VARCHAR
```

Les 1 454 records sont triés par `partition_key`, puis
`pickle_relative_path`, puis `sidecar_relative_path`, selon l'ordre
lexicographique de leurs octets UTF-8. Le Parquet et le payload du hash
utilisent exactement ce même ordre après réouverture.

Le même encodage canonique que l'inventaire partitions, appliqué à ces sept
champs dans cet ordre, produit un hash logique d'inventaire. Une modification
cohérente pickle + sidecar change donc le paquet d'entrée et invalide le
futur verrou runtime.

Le hash logique attendu de l'inventaire cache est :

```text
589360b10fa65d190bae9a2521e05d8e60e71c2cbd1fc0c5843c044332a183ce
```

Les deux hashes d'inventaire sont des attentes, pas de simples identités de
sortie : un écart arrête le build avant publication.

Le hash `92b68d...` est nommé `tfidf_config_artifact_hash` : il décrit l'artefact
logique historique, pas le contenu des 2 908 fichiers. Seul le hash
d'inventaire `589360...` constitue l'ancre de contenu préenregistrée.

## 6. Artefact et publication

Racine, physiquement distincte de tout oracle :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/inputs/v4_12_unit_engine/<build_id>
```

Fichiers exacts :

- `queries_all.parquet` ;
- `queries_dev.parquet` ;
- `partition_inventory.parquet` ;
- `tfidf_inventory.parquet` ;
- `integrity.json` ;
- `runtime_manifest.json`.

Cardinalités et schémas Arrow exacts, tous non nullables :

| Fichier | Lignes | Colonnes |
|---|---:|---|
| `queries_all.parquet` | 7 003 | six `string` CRM |
| `queries_dev.parquet` | 1 456 | six `string` CRM |
| `partition_inventory.parquet` | 4 119 | `string, uint64, string` |
| `tfidf_inventory.parquet` | 1 454 | `string, string, uint64, string, string, uint64, string` |

Le `build_id` dépend du plan, du verrou, des sources, des deux Parquet CRM,
des deux inventaires complets et du runtime.

Le builder impose :

- verrou externe après commit du code ;
- fichiers/sources réguliers et non symboliques ;
- `lstat` de chaque composant de chemin, sans suivre un symlink ;
- vérification Git des sources ;
- TOCTOU avant et après les deux inventaires ;
- RSS maximal de 8 Gio ;
- staging SSD, même `st_dev` que la sortie, `fsync`, renommage atomique et
  postvalidation ;
- artefact immuable et absence de sortie préalable.

Le manifeste runtime publie pour chaque Parquet hash, taille, nombre de
lignes et schéma complet. Il contient aussi un record obligatoire
`integrity.json` avec hash et taille, refuse tout fichier runtime
supplémentaire, et publie les hashes
logiques d'arbres, runtime et déclarations, mais aucun chemin absolu source,
nom de sibling, plan de provenance ou chemin permettant de reconstruire le
dossier historique. La provenance complète reste dans le plan et le verrou
du builder, qui ne seront jamais fournis au worker.

```text
labels_opened = false
oracle_opened = false
models_opened = false
candidate_results_opened = false
```

### Preuve de provenance séparée

Le même build publie séparément, sous :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_unit_inputs/<build_id>
```

- `data_inputs.parquet` : ledger exact des 7 029 fichiers data ouverts
  (`queries`, `split`, 4 119 partitions, 2 908 caches), avec rôle, chemin
  absolu, projection éventuelle, taille et hash avant/après ;
- `provenance.json` : commit, sources, runtime, verrou, déclarations
  anti-fuite et hash du manifeste runtime ;
- `manifest.json` : hashes des deux sorties de preuve.

Cette racine sensible n'est jamais transmise au worker. L'audit est publié
d'abord, puis le paquet runtime en dernier. Un audit orphelin ou un paquet
sans audit concordant reste `STOP` et n'est jamais autorisé.

Si la promotion runtime échoue après la promotion audit, l'audit orphelin
reste immuable et explicitement refusé. Un nouveau build utilise une nouvelle
identité ; aucune reprise ne peut recycler l'audit sauf égalité bit à bit de
tous les outputs et revalidation complète avant promotion.

Avant toute remise ultérieure du paquet, le validateur de concordance exige :
même nom/build ID des deux racines, hash du `runtime_manifest.json` égal à la
référence de `provenance.json`, puis hashes du ledger et de la provenance
égaux au manifeste d'audit.

## 7. Anti-fuite

Sont interdits :

- `labels.parquet`, challenges, holdouts et tests finaux ;
- candidats historiques, ranker, scènes, décisions et preuves ;
- modèles, taxonomie et lookup ;
- réseau, API, LLM, GPU ou dépense externe ;
- toute importation des builders historiques de matching ;
- toute écriture dans les sources, partitions ou cache.

Le builder est autonome : bibliothèque standard, DuckDB/PyArrow pour les
projections et hashes de fichiers. Il ne calcule aucun match.

Le verrou externe possède exactement :

```text
schema_version
purpose
audit_verdict
git_commit
source_hashes
input_paths
input_hashes
partition_inventory_sha256
tfidf_inventory_sha256
runtime
output_root
audit_output_root
temp_root
max_rss_bytes
```

Il épingle le commit, contrat, plan, builder, tests, queries, split, les deux
hashes d'inventaire attendus, Python 3.14.3, DuckDB 1.4.3, PyArrow 23.0.1,
macOS arm64, le plafond RSS et les trois racines SSD. Il n'existe aucun flag
de bypass.

L'inventaire partitions certifie les fichiers disponibles, pas encore le
comportement du store runtime. Le futur contrat worker devra contrôler
séparément chaque méthode de lecture et distinguer une partition réellement
vide d'une erreur I/O. `GO_V412_UNIT_INPUTS` ne vaut pas `GO_STORE_RUNTIME`.

Le builder est lancé avec `python -B`, `PYTHONDONTWRITEBYTECODE=1` et un
`TMPDIR` sous son staging. Le ledger et les sources Git sont rehashés
immédiatement après création des sorties et juste avant chaque promotion.
L'audit indépendant est une recomputation read-only documentée dans le
rapport ; il ne modifie jamais les deux artefacts.

## 8. Tests et autorisation

Ordre :

1. commit isolé contrat + plan ;
2. builder + tests ;
3. audit indépendant et commit ;
4. verrou externe et contre-audit ;
5. une construction réelle ;
6. contre-audit des projections et inventaires ;
7. rapport et handover.

Tests minimaux :

- projection physique réellement limitée aux six/deux colonnes ;
- 7 003/1 456, unicité, ordres, bornes et payloads LF ;
- sortie sans aucune colonne interdite ;
- égalité complète des six valeurs CRM, nullabilité et metadata ;
- fichier partition/cache absent, extra, modifié, symlink ou non régulier ;
- sidecar faux, pickle modifié et pickle + sidecar rescellés ;
- JSON sidecar avec clé dupliquée, collision `safe_key` et nom non canonique ;
- fichier vide postcode accepté sans désérialisation ;
- hash logique sensible au chemin, ordre, taille et contenu ;
- source/query/split modifiés pendant le run ;
- RSS, staging, `fsync`, atomicité, postvalidation et cleanup ;
- mutation post-publication de `integrity.json` ;
- mini-publication complète ;
- ledger exhaustif de 7 029 entrées, paquet runtime sans chemin sensible et
  audit séparé concordant.

Verdicts :

- `GO_V412_UNIT_INPUTS` : paquet sûr et inventaires exacts ;
- `STOP_V412_UNIT_INPUTS` : projection, inventaire, provenance, ressource ou
  publication invalide.

Seul `GO_V412_UNIT_INPUTS` autorise le préenregistrement de l'oracle
physiquement séparé. Il n'autorise pas encore le worker ni le benchmark.
