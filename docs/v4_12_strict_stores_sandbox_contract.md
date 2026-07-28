# V4.12 — Contrat Gate A des stores stricts et de la sandbox

## 1. Objet

Certifier les seules lectures de données autorisées au futur moteur unitaire,
avant tout chargement de ranker ou d'accepteur :

```text
CRM sûr
→ partition géographique stricte
→ cache TF-IDF vérifié strictement read-only
→ lookup snapshot DuckDB strictement read-only
```

Le gate ne calcule aucun match, feature, score ou décision. Il ne lit aucun
label, oracle, candidat historique, preuve, scène ou modèle.

Verdicts :

- `GO_V412_STRICT_STORES_SANDBOX` ;
- `STOP_V412_STRICT_STORES`.

Seul le GO autorise le contrat du moteur et de la parité.

## 2. Entrées sûres gelées

Paquet runtime :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/inputs/v4_12_unit_engine/
ca0b22e79cd2e92a32c009266e6d967b4ea48654de8736bca2b0ea7fdc9f8d6e
```

Ancres :

- `runtime_manifest.json` :
  `34e7a8d1c97c962b7f2295a99cd45b68c37afa9c866a5eb22e302330cb9010c4` ;
- `queries_dev.parquet` :
  `b1fef6ba72e4a557175a60c7e21e658f8fbb739b1d6607b77e5f70da873a031f` ;
- `queries_all.parquet` :
  `2f84eea594ed209042eed7b758f1c2390089cbb32cbe5dd832eff7dd80272fb9` ;
- `integrity.json` :
  `195083c8fdc26a72db47b7deaebe106004a625afb904de4bcdebaffd15932aa1` ;
- `partition_inventory.parquet` :
  `d17621743fad9c9c18ab46d2ba137521e4f0cfb1ccb44de8d08fca243c925bfe` ;
- inventaire logique partitions :
  `680f1884879bfa5b8cf2c335a0658604010e3d4c546ed6eaeb2e2ef34c954463` ;
- `tfidf_inventory.parquet` :
  `845e332df405b3e9d631923ccf17661c5420be8b9518fe68583766b7b4f6881d` ;
- inventaire logique cache :
  `589360b10fa65d190bae9a2521e05d8e60e71c2cbd1fc0c5843c044332a183ce`.

La projection CRM est exactement :

```text
query_id,crm_name,crm_address,crm_postcode,crm_city,crm_insee
```

Aucun autre fichier du dataset historique n'est accessible.

## 3. Routage géographique figé

Le routage reproduit V4.11 `mega_insee_policy=full_insee` :

1. normaliser `crm_insee` ;
2. si la partition INSEE inventoriée existe, utiliser `<insee>_` ;
3. sinon normaliser `crm_postcode` et utiliser `_<postcode>` si inventorié ;
4. sinon `STOP`, jamais pool vide implicite.

Sur les 1 456 requêtes dev :

- 1 449 routes INSEE et 7 routes code postal ;
- 648 clés distinctes ;
- zéro clé absente ;
- payload ordonné `query_id + 0x00 + partition_key + 0x0A` :
  20 491 octets, SHA-256
  `41477bbcc9dea2cee8d49922679064995c46e49f60db1560e5d8a0033adc79bf`.

Sous-ensemble exact sollicité :

- 648 fichiers partitions, 449 454 881 octets ;
- hash logique canonique :
  `f772a7d75d92b0e0655d1815f924283f16c27b0d3761c80a9f8b3b6d7075d977` ;
- 648 paires pickle/sidecar, 4 042 655 632 octets ;
- hash logique canonique :
  `de14bc981ea4e4354d857b9a63a2b94fcf5c7fbcbd96fa07d7d0d843a897fcf8`.

Les algorithmes canoniques sont ceux du contrat des entrées sûres, appliqués
aux seuls records sélectionnés et triés par octets UTF-8.

La normalisation est exacte : `None` donne `None`; puis si
`str(value).strip()==""` ou `str(value).lower()=="nan"`, résultat `None`.
Sinon `s=str(value).strip()` est conservé, sauf si
`re.fullmatch(r"\d+\.0+",s)` réussit, auquel cas `s.split(".")[0]` est
retenu. Aucun padding, correction ou conversion numérique supplémentaire.

## 4. `StrictPartitionStore`

L'adaptateur :

- ne peut résoudre qu'un chemin présent dans l'inventaire gelé ;
- vérifie `lstat` de chaque composant, fichier régulier, absence de symlink,
  taille et SHA-256 avant lecture ;
- impose une unique partition correspondant à la clé ;
- vérifie le schéma Parquet, la valeur de partition et les types nécessaires
  au retrieval ;
- distingue `VALID_EMPTY` d'une absence, erreur I/O, footer ou schéma ;
- transforme toute erreur en `STOP_V412_STRICT_PARTITION`, jamais en `[]` ;
- cache mémoire borné à cinq partitions, sans modifier les rows restituées ;
- interdit toute écriture et toute lecture hors inventaire.

Les méthodes historiques qui attrapent une exception pour retourner zéro ou
un pool vide ne sont jamais utilisées directement par le worker.

Schéma physique autorisé, dans cet ordre :

```text
siret:string, siren:string, denomination:string,
enseigne1:string, enseigne2:string, enseigne3:string,
etablissementSiege:bool, is_siege:bool,
numeroVoie:string, typeVoie:string, libelleVoie:string,
complementAdresse:string,
[postcode:string, city:string pour INSEE |
 city:string, insee:string pour CP],
cj_ul:string, etat_admin:string, last_treatment_date:timestamp[us],
sigle_ul:string, denomination_ul:string,
denomination_usuelle_ul:string, nom_ul:string, prenom_usuel_ul:string,
pm_dirigeant_names:string
```

La valeur Hive absente du fichier est injectée depuis le chemin inventorié :
pour une clé `<insee>_`, `insee` est la composante avant `_` et doit égaler
la valeur du segment `insee=<insee>` ; pour une clé `_<postcode>`, `postcode`
est la composante après `_` et doit égaler le segment
`postcode=<postcode>`. Aucun autre champ ou cast n'est accepté. Les metadata
pandas historiques sont ignorées pour les valeurs, mais leur hash est déjà
couvert par le hash du fichier.

## 5. `StrictVerifiedTfidfCache`

L'adaptateur :

- accepte uniquement une des 648 clés routées et inventoriées ;
- applique le `safe_key` canonique ;
- vérifie pickle et sidecar par chemin, taille et SHA-256 ;
- rejette les clés JSON dupliquées et impose le keyset/version exact ;
- exige `config_hash` égal au namespace
  `296c7891107249a073c00d93c7310c55a652243de4bcfa7165d09dbfc3349a82` ;
- désérialise seulement après tous les contrôles ;
- exige un tuple de sept éléments et des dimensions cohérentes avec la
  partition ;
- cache mémoire borné à vingt clés ;
- sur miss, corruption ou incompatibilité : `STOP_V412_STRICT_TFIDF` ;
- `put`, `clear`, rebuild, fichier temporaire et écriture sont impossibles.

Le chemin historique `miss → _build_tfidf_artifacts → put` est interdit.

Tuple exact, dans l'ordre :

```text
0 name_vec: sklearn TfidfVectorizer | None
1 name_mat: scipy sparse matrix | None
2 names: list[str]
3 char_vec: sklearn TfidfVectorizer | None
4 char_mat: scipy sparse matrix | None
5 addr_vec: sklearn TfidfVectorizer | None
6 addr_mat: scipy sparse matrix | None
```

`len(names)` égale le nombre de rows du **pool sparse aligné** défini
ci-dessous, jamais le nombre de rows physiques brutes. Chaque matrice non
nulle possède ce même nombre de rows. Un vectorizer et sa matrice sont soit
tous deux nuls, soit tous deux présents ; le nombre de colonnes de la matrice
égale la taille du vocabulaire du vectorizer. Les noms sont des `str` exacts.
Le contexte read-only n'expose ni `put` ni `clear`.

Le pool sparse aligné est redérivé sans utiliser de résultat de retrieval,
dans cet ordre exact et stable :

1. partir des rows physiques dans leur ordre Parquet ;
2. supprimer une row si les neuf champs
   `denomination,denomination_usuelle_ul,enseigne1,enseigne2,enseigne3,
   denomination_ul,sigle_ul,nom_ul,prenom_usuel_ul` sont tous falsy ;
3. supprimer une row seulement si `etat_admin == "F"` exactement ;
4. poser la clé `siret=str(value or "").strip()`, supprimer la clé vide,
   dédupliquer par affectation dictionnaire « last value wins » tout en
   conservant l'ordre de première insertion des clés ;
5. convertir le dictionnaire en liste dans cet ordre.

Le cache est accepté seulement si `len(names)` et les rows des trois matrices
égalent la taille de ce pool aligné. En mode V4.11 figé
`tfidf_name_mode="bag"` et `siren_siblings=false`, `names` doit en outre être
exactement la liste
`normalize_text_for_tfidf(candidate_tfidf_text(row) or "")` redérivée sur le
pool aligné, dans le même ordre. Sur le sous-ensemble gelé attendu : 8 030 285
rows physiques deviennent 4 764 472 rows alignées ; les 648 clés doivent
toutes concorder, sinon `STOP`.

Ces trois opérations — filtrage/déduplication, construction du bag-of-names et
normalisation TF-IDF — sont réimplémentées en fonctions privées, pures et
figées dans `src/xgb_matcher/v412_strict_stores.py`. Ce module n'importe aucun
module historique du projet (`retrieval`, `blocking`, `naming`, `candidates`
ou leurs dépendances). Son blob Git et son SHA-256 entrent dans
`source_hashes`, le verrou et le build ID. La concordance exacte avec les
4 764 472 chaînes `names` hashées dans les caches constitue le test exhaustif
de parité ; un seul écart provoque `STOP`.

## 6. `StrictSnapshotLookup`

Artefact :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/
v4_12_snapshot_lookup/ff0f33ad10803cfb
```

Ancres :

- manifeste :
  `04e098952ee4cc7957155623599d3ba35b95f9126932e5a5d420ec02b110b15e` ;
- base DuckDB, 2 732 863 488 octets :
  `5da123bb0dde06d55886dfbc5c36e142c9d528ffec1b6899022e5c7c63bee894` ;
- intégrité :
  `048901e99bc6f6f33aeb34434d0662af8e94b6667e056f83f1689b72c8314b2a` ;
- timing :
  `c266db1075c2f777b8942cea7bc7e446f3a09707a07a3b9952b71a73d4cc6e9b`.

La base est rehashée avant ouverture, sans WAL, puis ouverte avec
`read_only=True`. Le schéma, la table et l'index unique sont exacts.
L'API accepte 0 à 100 SIRET canoniques, déduplique en ordre de première
apparition, utilise une requête paramétrée et refuse 101 entrées.

`ATTACH`, `COPY`, extensions, fichiers temporaires et toute écriture sont
interdits.

Le manifeste lookup historique contient une référence de parité à
`candidates_sparse_top100.parquet`. Il est donc **parent-only** : le
certificateur de confiance le rehash et le valide avant la sandbox, mais il
n'est ni monté, ni lisible, ni transmis au probe enfant ou au futur worker.
Il en va de même pour `integrity.json`, `timing.json`, le plan Gate A complet
et le ledger.

Le parent produit un `lookup_descriptor.json` assaini avec le keyset exact :

```text
schema_version,database_sha256,database_size_bytes,table_name,
columns,column_types,index_name,index_unique,row_count,max_sirets_per_call,
read_only
```

Valeurs exactes : table `candidate_details`; colonnes
`siret,candidate_state,enseigne1,enseigne2,enseigne3,denomination_usuelle,activity_code`;
sept types `VARCHAR`; index `candidate_details_siret_uidx`, unique ; 42 322 035
rows ; maximum 100 ; `read_only=true`.

## 7. Probe réel Gate A

Le probe traite les 648 clés une seule fois, dans l'ordre UTF-8 :

1. rehash et lecture stricte de la partition ;
2. rehash et désérialisation stricte du cache ;
3. validation des dimensions ;
4. sortie du contexte privé, suppression des références puis `gc.collect()`.

Le probe conserve au plus une clé TF-IDF à la fois. Le futur adapter pourra
utiliser un LRU de vingt clés seulement s'il impose aussi un plafond d'un Gio
estimé sur les buffers matrices/arrays ; toute insertion dépassant ce plafond
évince avant de poursuivre.

Le lookup est testé sur un échantillon déterministe label-free d'au plus
10 000 SIRET issus de ces partitions, ordonné par
`SHA-256("v412-store-lookup:" + siret), siret`, par lots de 100. Zéro SIRET
extra ou dupliqué est accepté ; les absences éventuelles sont publiées en
nombre brut et provoquent `STOP`.

Certification sans candidat ni vérité :

```text
store_probe.json
lookup_descriptor.json
run_spec.json
sandbox_profile_effective.sb
integrity.json
manifest.json
```

`store_probe.json` publie uniquement volumes, hashes logiques, compteurs,
RSS, durées agrégées, résultats sandbox et zéro miss/rebuild/write.

La certification est publiée sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/certifications/v4_12_strict_stores/<build_id>`.
La preuve sensible séparée est publiée sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_strict_stores/<build_id>`
avec exactement `open_ledger.parquet`, `provenance.json` et `manifest.json`.

Le ledger contient exactement les six fichiers runtime sûrs, les 648
partitions, 648 pickles, 648 sidecars et les quatre fichiers lookup, avec
chemin, rôle, taille et hash avant/après : 1 954 lignes.

Publication staging SSD, `fsync`, audit d'abord, certification en dernier,
artefacts immuables et concordance par même build ID. Aucun résultat par
requête ou candidat n'est publié.

`run_spec.json` possède exactement :

```text
schema_version,safe_input_build_id,query_count,routing_payload_sha256,
partition_records,cache_records,lookup_descriptor_sha256,
allowed_read_files,staging_dir,tmp_dir,max_rss_bytes,declarations
```

`partition_records` contient les 648 records à trois champs exacts
`relative_path,size_bytes,sha256`. `cache_records` contient les 648 records
à sept champs exacts
`partition_key,pickle_relative_path,pickle_size_bytes,pickle_sha256,
sidecar_relative_path,sidecar_size_bytes,sidecar_sha256`.
`allowed_read_files`
est la liste triée et unique des 1 945 fichiers data externes lisibles par
l'enfant : 648 partitions, 1 296 caches et la base lookup. Le descriptor et
le run-spec sont dans la racine privée `RUN_ROOT`; ils ne figurent donc pas
dans cette liste. Aucun parent de répertoire data n'est autorisé globalement.

Chaque élément de `allowed_read_files` possède exactement :

```text
role,partition_key,absolute_path,size_bytes,sha256
```

Les rôles et cardinalités sont `partition` (648), `cache_pickle` (648),
`cache_sidecar` (648) et `lookup_database` (1). `partition_key` est la clé
routée pour les trois premiers rôles et la chaîne vide pour le lookup. La
liste est triée par octets UTF-8 de `(role,absolute_path)`. Les chemins sont
absolus, déjà résolus sans symlink et bijectifs avec `partition_records`,
`cache_records` et le descriptor lookup. Un chemin est globalement unique et
une `partition_key` est unique à l'intérieur de chacun des trois rôles
partitionnés.

Les deux chemins de travail sont des valeurs canoniques, indépendantes du
répertoire aléatoire créé par le parent :

```text
staging_dir = output
tmp_dir = tmp
```

Le processus enfant a pour `cwd` la vraie racine privée et reçoit
`TMPDIR=<cwd>/tmp`. Aucun chemin absolu aléatoire n'entre dans le run-spec,
le profil effectif ou le build ID.
Toutes les valeurs fournies via `-D` sont les chemins `realpath` canoniques
(`Path.resolve(strict=True)`), jamais un alias tel que `/tmp`; chacun de leurs
composants est contrôlé par `lstat` et aucun symlink n'est accepté.

`lookup_descriptor.json` possède exactement :

```text
schema_version,database_sha256,database_size_bytes,table_name,
columns,column_types,index_name,index_unique,row_count,max_sirets_per_call,
read_only
```

Versions exactes :

```text
lookup descriptor   sireto-v4.12-strict-lookup-descriptor-1
run spec            sireto-v4.12-strict-stores-run-spec-1
store probe         sireto-v4.12-strict-stores-probe-1
integrity           sireto-v4.12-strict-stores-integrity-1
cert manifest       sireto-v4.12-strict-stores-certification-1
provenance          sireto-v4.12-strict-stores-provenance-1
audit manifest      sireto-v4.12-strict-stores-audit-manifest-1
```

Le `build_id` est le SHA-256 des octets JSON canoniques de l'objet exact :

```text
schema_version = sireto-v4.12-strict-stores-build-identity-1
plan_sha256
lock_sha256
source_hashes
safe_runtime_manifest_sha256
partition_inventory_sha256
tfidf_inventory_sha256
partition_subset_logical_sha256
cache_subset_logical_sha256
lookup_input_hashes
sandbox_profile_sha256
lookup_descriptor_sha256
run_spec_sha256
runtime
```

JSON UTF-8 trié, compact, sans NaN, avec LF final et rejet des clés
dupliquées.

`source_hashes` est un objet trié `chemin relatif Git → SHA-256`. Les
`lookup_input_hashes` ont exactement les clés
`database,manifest,integrity,timing`. `runtime` a exactement les clés
`python,numpy,pandas,pyarrow,scikit_learn,scipy,joblib,duckdb,machine,platform`.

`store_probe.json` possède exactement :

```text
schema_version,build_id,query_count,distinct_key_count,
partition_verified_count,partition_raw_row_count,cache_verified_count,
aligned_pool_row_count,cache_miss_count,
rebuild_count,write_count,lookup_sample_count,lookup_missing_count,
lookup_extra_count,sandbox_checks,peak_rss_bytes,durations_ns,declarations
```

`partition_raw_row_count=8_030_285` et
`aligned_pool_row_count=4_764_472` exactement.
Les compteurs miss/rebuild/write/lookup missing/extra valent zéro.
`sandbox_checks` possède exactement
`allowed_read,oracle_denied,oracle_audit_denied,network_denied,write_denied`.
`durations_ns` possède `partitions,cache,lookup,total`, entiers non négatifs.

Les `declarations` présentes dans le run-spec, le probe, l'intégrité, le
manifeste de certification et la provenance ont exactement ce keyset et ces
valeurs :

```text
labels_opened=false
oracle_opened=false
historical_candidates_opened=false
models_opened=false
network_used=false
writes_outside_staging=false
cache_rebuild_attempted=false
```

`integrity.json` possède exactement :

```text
schema_version,build_id,run_spec_sha256,lookup_descriptor_sha256,
sandbox_profile_effective_sha256,store_probe_sha256,data_input_count,
data_ledger_sha256,declarations
```

`data_input_count` vaut 1 954. `data_ledger_sha256` est le SHA-256 du fichier
Parquet final publié dans l'audit.

Le manifeste certification possède exactement :

```text
schema_version,build_id,files,runtime,declarations,verdict
```

`files` est une liste triée par `path` de cinq records exacts
`path,size_bytes,sha256`, pour `store_probe.json`, `lookup_descriptor.json`,
`run_spec.json`, `sandbox_profile_effective.sb` et `integrity.json`.
`verdict` est `GO_V412_STRICT_STORES_SANDBOX` ou
`STOP_V412_STRICT_STORES`.

`provenance.json` possède exactement :

```text
schema_version,build_id,git_commit,source_hashes,lock_sha256,plan_sha256,
sandbox_profile_effective_sha256,runtime,data_input_count,
certification_manifest_sha256,declarations
```

Le manifeste d'audit possède exactement :

```text
schema_version,build_id,files
```

`files` est une liste triée de deux records exacts `path,size_bytes,sha256`
pour `open_ledger.parquet` et `provenance.json`.

Le ledger data utilise les rôles, cardinalités et projections littérales
suivants :

| rôle | lignes | projection |
|---|---:|---|
| `safe_runtime_manifest` | 1 | `JSON_EXACT` |
| `safe_queries_all` | 1 | `HASH_ONLY` |
| `safe_queries_dev` | 1 | `query_id,crm_name,crm_address,crm_postcode,crm_city,crm_insee` |
| `safe_partition_inventory` | 1 | `relative_path,size_bytes,sha256` |
| `safe_tfidf_inventory` | 1 | `partition_key,pickle_relative_path,pickle_size_bytes,pickle_sha256,sidecar_relative_path,sidecar_size_bytes,sidecar_sha256` |
| `safe_input_integrity` | 1 | `HASH_ONLY` |
| `partition` | 648 | `STRICT_PARTITION_SCHEMA` |
| `cache_pickle` | 648 | `PICKLE_TUPLE_7` |
| `cache_sidecar` | 648 | `config_hash,partition_key,schema_version,sha256,size_bytes` |
| `lookup_database` | 1 | `siret,candidate_state,enseigne1,enseigne2,enseigne3,denomination_usuelle,activity_code` |
| `lookup_manifest` | 1 | `PARENT_VALIDATION_ONLY` |
| `lookup_integrity` | 1 | `PARENT_VALIDATION_ONLY` |
| `lookup_timing` | 1 | `PARENT_VALIDATION_ONLY` |

La somme est exactement 1 954. Toute autre combinaison rôle/projection ou
toute ligne supplémentaire provoque `STOP`.

Le verrou externe possède exactement :

```text
schema_version,purpose,audit_verdict,git_commit,source_hashes,
input_paths,input_hashes,expected_routing,expected_partition_subset_sha256,
expected_cache_subset_sha256,runtime,output_root,audit_output_root,temp_root,
max_rss_bytes
```

Valeurs exactes :

```text
schema_version = sireto-v4.12-strict-stores-execution-lock-1
purpose = V4.12_STRICT_STORES_SANDBOX
audit_verdict = GO_CODE_V412_STRICT_STORES
```

`source_hashes` possède exactement un record par chemin de `sources` dans le
plan, sans ajout. `input_paths` possède exactement :

```text
safe_input_root,safe_runtime_manifest,safe_queries_all,safe_queries_dev,
safe_partition_inventory,safe_tfidf_inventory,safe_input_integrity,
partition_root,cache_root,lookup_database,lookup_manifest,lookup_integrity,
lookup_timing,sandbox_executable,python_framework_bin,python_framework_app,
system_read_roots,device_read_literals
```

`input_hashes` possède exactement :

```text
safe_runtime_manifest,safe_queries_all,safe_queries_dev,
safe_partition_inventory,safe_tfidf_inventory,safe_input_integrity,
partition_full_inventory_logical,tfidf_full_inventory_logical,
lookup_database,lookup_manifest,lookup_integrity,lookup_timing,
sandbox_executable,python_framework_bin,python_framework_app
```

Chaque chemin est la copie canonique du plan ou, pour les six fichiers
runtime, le membre correspondant du paquet `safe_input`; chaque hash est
redérivé puis comparé au manifeste/plan avant scellement.
`expected_routing` est la copie canonique de l'objet `routing` du plan avec
exactement :

```text
cp_query_count,distinct_key_count,insee_query_count,missing_key_count,
payload_bytes,payload_sha256,query_count
```

`runtime` est la copie canonique de l'objet `runtime` du plan avec son keyset
exact déjà défini. Les trois racines de sortie et `max_rss_bytes` sont les
valeurs littérales du plan.

## 8. Sandbox macOS obligatoire

Le parent génère un profil **effectif déterministe** depuis les entrées
gelées. Le texte contient les 1 945 fichiers data externes comme règles
`literal` individuelles, les fichiers de code nécessaires comme règles
`literal`, et les seules racines système/runtime explicitement admises.
Il ne contient aucun chemin de vérité, de résultat historique ou de modèle.

Les exécutables sont rehashés avant le lancement :

```text
/usr/bin/sandbox-exec
  8290e4be7387a0df83cd1559e86afd880464f269450573d012795761fe298f16
/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/
Versions/3.14/bin/python3.14
  cbf84109626aa1013bbe408fbb9590bd0f1c1548f038b2221c6b8b87de26ca43
/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/
Versions/3.14/Resources/Python.app/Contents/MacOS/Python
  7ecc1ecbf9daa9303c4bf502ff62ffdd9010ed5c08729d470ae9380c10ce1211
```

Les seules racines runtime lues par `subpath` sont `/System`, `/usr` et
`/opt/homebrew`; les seuls devices ajoutés en `literal` sont `/dev/null` et
`/dev/urandom`. Elles permettent les bibliothèques du runtime, jamais les
données du projet situées dans `/Users` et `/Volumes`.

La racine privée aléatoire n'est pas sérialisée : le profil utilise
`(param "RUN_ROOT")`, `(param "RUN_SPEC")`,
`(param "LOOKUP_DESCRIPTOR")`, `(param "RUN_OUTPUT")` et
`(param "RUN_TMP")`. Le fichier identique
`sandbox_profile_effective.sb` est hashé avant exécution, inclus dans le
build ID et publié dans la certification. Il est exécuté exactement ainsi :

```text
/usr/bin/sandbox-exec \
  -D RUN_ROOT=<racine_privée_absolue> \
  -D RUN_SPEC=<racine_privée_absolue>/run_spec.json \
  -D LOOKUP_DESCRIPTOR=<racine_privée_absolue>/lookup_descriptor.json \
  -D RUN_OUTPUT=<racine_privée_absolue>/output \
  -D RUN_TMP=<racine_privée_absolue>/tmp \
  -f <racine_privée_absolue>/sandbox_profile_effective.sb \
  /opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/Versions/3.14/bin/python3.14 \
  -B /Users/nathanjullia/Documents/Projets/SIRETO/src/xgb_matcher/v412_strict_stores.py \
  --run-spec <racine_privée_absolue>/run_spec.json \
  --forbidden-oracle <manifeste_oracle_existant> \
  --forbidden-audit <manifeste_audit_oracle_existant>
```

`v412_strict_stores.py` est à la fois le module des trois adapters et le CLI
enfant du probe. Le binaire
`bin/python3.14` est le launcher explicite ; le binaire
`Resources/Python.app/Contents/MacOS/Python` est uniquement le second
`process-exec` interne réalisé par ce launcher sur macOS.

La commande réelle, le SHA-256 du profil lu par `sandbox-exec` et la valeur
de `RUN_ROOT` sont conservés parent-only dans la provenance d'exécution
interne ; seule la valeur non sensible du hash est publiée. Le profil :

- `deny default`, `deny network*` ;
- lecture seulement des runtimes système/Python nécessaires et de la liste
  blanche de code/données ;
- `file-read-metadata` seulement sur `RUN_ROOT`, lecture de contenu littérale
  seulement sur `RUN_SPEC` et `LOOKUP_DESCRIPTOR`, aucune lecture du reste de
  la racine privée ;
- `file-read-metadata` littéral sur chacun des ancêtres canoniques fixes de
  `RUN_ROOT` (`/`, `/Volumes`, le volume, puis chaque composant de
  `temp_root`) uniquement pour la traversée ; aucune règle `subpath` sur ces
  ancêtres ;
- `file-read-metadata` littéral, dédupliqué et trié, sur chaque ancêtre
  canonique de chacun des 1 945 fichiers data externes et de chaque source de
  code autorisée ; aucune lecture de contenu ni règle `subpath` n'en découle.
  Le probe exécute réellement `lstat` sur chaque composant et exige la
  concordance exacte avec cette fermeture d'ancêtres ;
- lecture/écriture seulement sous `RUN_OUTPUT` et `RUN_TMP` pour les deux
  espaces de travail privés ; aucune écriture sur le reste de `RUN_ROOT` ;
- `deny process-fork` et aucun sous-processus après démarrage ;
- `process-exec` seulement pour les deux exécutables Python Framework réels
  nécessaires au lancement (`bin/python3.14` et
  `Resources/Python.app/Contents/MacOS/Python`) ;
- interdiction explicite des racines `oracles`, `audits`, `final`,
  `challenges` et des jeux test/holdout ;
- `python -B`, `PYTHONDONTWRITEBYTECODE=1`.

La déclaration `writes_outside_staging=false` signifie exactement : aucune
écriture hors des deux racines privées autorisées `RUN_OUTPUT` et `RUN_TMP`.

Sous le même profil et dans le même processus de probe :

- lecture d'un input autorisé : succès ;
- ouverture du manifeste oracle
  `c4045da8...d2fa70/manifest.json` : `EPERM`, `errno=1` ;
- ouverture de son manifeste d'audit : `EPERM`, `errno=1` ;
- accès réseau loopback et externe : `EPERM`, `errno=1` ;
- création d'un fichier hors staging : `EPERM`, `errno=1`.

Le parent de publication peut lire la preuve ; le processus sandboxé ne le
peut pas.

Le processus enfant ne reçoit jamais le plan complet. Le plan, le verrou,
les manifests historiques, leur contenu et les preuves restent hors de sa
liste blanche. Les deux chemins absolus de sentinelle oracle/audit lui sont
passés comme arguments opaques uniquement pour tenter `open(...,"rb")` et
exiger `EPERM` (`errno=1`) ; ils ne figurent ni dans le run-spec, ni dans la certification,
ni dans le build ID. `lookup_descriptor.json` et `run_spec.json` sont conservés
dans la certification finale et scellés pour contre-audit.

Runtime épinglé : Python 3.14.3, NumPy 2.4.2, pandas 2.3.3, PyArrow 23.0.1,
scikit-learn 1.8.0, SciPy 1.16.3, joblib 1.5.3, DuckDB 1.4.3 et macOS arm64.

## 9. Anti-fuite et tests

Interdits par chemin, hash, projection et sandbox :

- oracle et audits ;
- `labels.parquet`, candidats/predictions historiques, `is_ground_truth` ;
- preuves directes, décisions de garde, scènes, modèles ;
- challenges, holdouts, random V4.8/V4.9 et test final ;
- réseau, API, LLM, GPU ou dépense.

Tests minimaux :

- fichier absent, extra, muté, symlink ou substitution ;
- erreur Arrow/Parquet jamais transformée en vide ;
- clé cache absente/collisionnée, JSON dupliqué, pickle/sidecar corrompu ;
- tentative `put`, `clear`, rebuild ou écriture ;
- dimensions TF-IDF incohérentes ;
- lookup muté, WAL, écriture, extension et 101 SIRET ;
- mutation/restauration TOCTOU ;
- profil sandbox réel : allow et deny sentinelles ;
- aucune source oracle/finale importée ou ouverte ;
- ledger exhaustif 1 954 lignes et ordre canonique ;
- RSS, atomicité, permissions et manifests redérivés.

## 10. Portée du verdict

`GO_V412_STRICT_STORES_SANDBOX` prouve seulement que les lectures nécessaires
sont strictes, complètes et isolées. Il n'autorise pas encore les modèles,
l'ouverture de l'oracle, la mesure de performance ou la latence.

Le ledger est explicitement un **ledger data**. Schéma non nullable :
`role:string,absolute_path:string,projection:string,size_before:uint64,
sha256_before:string,size_after:uint64,sha256_after:string`, trié par octets
UTF-8 de `(role,absolute_path)`. Code, profil, run-spec, descriptor et
bibliothèques sont exclus du compteur 1 954 et scellés séparément par
sources Git, manifests et build ID.
