# Contrat V4.12 S0 — Scanner/sealer synthétique de l’intake frais

## 1. Autorité fermée

S0 éprouve uniquement, sur un paquet synthétique pré-pinné, la stabilité,
le scellement, les receipts, le scan structurel, la quarantaine et la reprise
jusqu’à `INGESTED`.

S0 est `SYNTHETIC_ONLY`. Il n’autorise jamais l’ouverture ou l’énumération
d’un CRM réel, de `data/`, des racines `fresh_holdout_intake` réelles, des
registres, modèles, rapports, challenges, outputs ou ledgers historiques.
Keychain, réseau, fork, exec, subprocess, retrieval, qualification, oracle,
`QUALIFIED` et `READY` sont interdits.

Le plan S0 épingle :

- le contrat fresh
  `fb2f52ae2a656622e547975d054ef1089d18065813e83b129706ce6c54d2485c` ;
- le plan fresh
  `5bf1a327a46042ffbdcbb736c2e6d7ac0f93cb21bb5f0fb8a57cafbe524fcc38`.

Le présent contrat ne s’auto-pinne pas. Son hash brut est porté par le plan
S0 canonique.

## 2. Sandbox, racine et identités S0

L’autorité effective vient d’un profil sandbox et d’un launcher futurs,
épinglés dans le lock : deny-default, aucun réseau/fork/exec/subprocess ou
Keychain, et accès limité aux descripteurs déjà ouverts du code, plan, lock,
runtime, control manifest et de la racine :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake_synthetic/<synthetic_run_id>`

La denylist de chemins est une défense secondaire ; elle ne remplace jamais
l’allowlist de capacités. Des canaris réels prouvent qu’une tentative
d’ouverture des racines interdites échoue.

`synthetic_run_id` et `attempt_id` sont chacun un composant unique conforme à
`^[a-p]{64}$`, sans `/`, `.` ou `..`. Ils utilisent :

```text
alphabet = abcdefghijklmnop
mapping = hex nibble 0..f vers a..p
digest = SHA256(domain UTF-8 terminé par NUL || JSON canonique sans LF)
run domain = SIRETO-V412-FRESH-SYNTHETIC-RUN-ID\0
run projection = fixture_control_manifest_sha256, plan_sha256
attempt domain = SIRETO-V412-FRESH-SYNTHETIC-ATTEMPT-ID\0
attempt projection = synthetic_run_id, fixture_control_manifest_sha256,
                     logical_time_utc
```

Répertoires `0700`, fichiers `0600`, `umask 0077`. Les chemins sont traversés
depuis `/` par `openat`, avec `O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`; les
terminaux sont réguliers, mono-liens, du UID, device et UUID épinglés.

## 3. Control manifest hors paquet

`fixture_control_manifest.json` est un artefact de contrôle séparé, canonique,
pré-pinné dans le lock, lu par FD. Il n’entre ni dans l’inbox, ni dans les
cinq payloads, ni dans l’arbre scellé.

Champs exacts :

```text
schema_version
synthetic_fixture
fixture_spec_sha256
synthetic_run_id
logical_time_utc
batch_count
expected_source_row_count
producer_exclusions
collection_source_manifest_sha256
source_manifest_sha256
crm_safe_csv_sha256
evidence_source_manifest_sha256
evidence_source_parquet_sha256
```

Valeurs imposées : `synthetic_fixture=true`, `batch_count=1`,
`expected_source_row_count=6`, `producer_exclusions=[]`.
`fixture_spec_sha256` est le SHA-256 du JSON canonique compact UTF-8 sans LF de
l'objet `fixture` exact du plan. Il est donc indépendant de tout run.
`synthetic_run_id` est ensuite dérivé de
`H({fixture_spec_sha256,plan_sha256})`. Tous les hashes sont 64 hex minuscules
et concordent avec les octets.

`logical_time_utc` est RFC3339 UTC à la seconde, format exact
`YYYY-MM-DDTHH:MM:SSZ`. Il ferme tous les payloads reproductibles. Les temps
d’audit observés peuvent différer mais ne rentrent jamais dans les payloads
reproductibles.

## 4. Paquet synthétique exact

L’inbox autorisée contient exactement cinq fichiers réguliers mono-liens :

```text
collection_source_manifest.json
source_manifest.json
crm_safe.csv
evidence_source_manifest.json
evidence_source.parquet
```

Les trois manifests JSON sont canoniques, `synthetic_fixture=true`, portent le
même run, collection, export, batch, source, portfolio et `reference_date`.
Les dates sont RFC3339 UTC seconde ; `period_start_utc <= period_end_utc <=
export_cutoff_utc`, `reference_date` est `YYYY-MM-DD`.

Le manifest collection déclare exactement un batch, six lignes,
`ALL_SOURCE_RECORDS_IN_FRAME`, les listes de batch/portfolio exactes et
`producer_exclusions=[]`. Source et evidence reprennent exactement export,
batch, provenance, temps et noms de fichiers. Les tailles et hashes portent
sur les octets exacts.

Le CSV est UTF-8 strict **sans BOM**, sans NUL, avec header exact :

```text
source_batch_id,source_record_id,source_system,portfolio_id,crm_name_raw,
crm_address_raw,crm_postcode_raw,crm_city_raw,crm_insee_raw
```

Il contient exactement six lignes, virgule, RFC4180, et utilise soit LF, soit
CRLF sans mélange. Limites : 4 MiB fichier, 64 KiB ligne, 16 KiB cellule.

Le parquet evidence est physiquement validé sans interprétation :

- schéma Arrow exact du plan ;
- zéro ligne ;
- exactement un row group vide ;
- aucune métadonnée applicative ;
- taille/hash exacts ;
- manifest `evidence_row_count=0`.

Archive, répertoire imbriqué, extra, symlink, FIFO, socket, device et hardlink
multiple produisent `STOP`.

## 5. Stabilité : un processus vivant

Le CLI synthétique ouvre les cinq fichiers une fois, conserve tous leurs FDs
dans **le même processus vivant pendant au moins 60 secondes monotones**, puis
effectue la seconde observation. Il n’existe ni mode two-phase, ni checkpoint
de stabilité, ni argument pour réduire l’intervalle.

Pour chaque FD, `device`, inode, taille, `mtime_ns` et `ctime_ns` sont
identiques aux deux observations. Deux lectures complètes jusqu’à EOF sur ce
même FD donnent tailles/hashes identiques au `fstat` et aux manifests.

Seuls les tests unitaires injectent horloge et attente, toujours dans le même
processus et avec les mêmes FDs. Mutation, substitution, lecture courte,
bytes après EOF attendu, dérive de hash, UID/device/UUID ou horloge produit
`STOP`.

## 6. Arbre d’entrée scellé

Après stabilité, les cinq octets payload sont copiés byte-for-byte dans un
temp exclusif du même filesystem. L’arbre exact est :

```text
collection_source_manifest.json
source_manifest.json
crm_safe.csv
evidence_source_manifest.json
evidence_source.parquet
payload_manifest.json
seal.json
```

`payload_manifest.json` ferme les cinq chemins, tailles et hashes, sans
s’inclure. `seal.json` ferme les octets du manifest et le hash logique de
l’arbre. Écriture complète, `fsync`, `F_FULLFSYNC`, validation indépendante,
puis `renameatx_np(RENAME_EXCL)` et sync parent sont obligatoires.

## 7. Receipts et identités

Après scellement et avant parsing :

1. receipt collection `O_EXCL` ;
2. receipt batch `O_EXCL` ;
3. événement global séquence 1.

Chaque receipt référence le **chemin complet de l’arbre scellé**,
`payload_manifest_sha256`, `seal_sha256`, le hash du manifest concerné et
assez d’identité pour une reprise sans inbox.

Les `receipt_id` utilisent le mapping `[a-p]` :

```text
collection domain = SIRETO-V412-FRESH-SYNTHETIC-COLLECTION-RECEIPT-ID\0
collection projection = synthetic_run_id, collection_id,
                        collection_manifest_sha256,
                        payload_manifest_sha256, seal_sha256
batch domain = SIRETO-V412-FRESH-SYNTHETIC-BATCH-RECEIPT-ID\0
batch projection = synthetic_run_id, collection_id, source_batch_id,
                   sealed_source_sha256, payload_manifest_sha256, seal_sha256
```

Les schemas exacts sont dans le plan. Aucun champ dynamique ne modifie un
receipt.

## 8. Scan ligne par ligne

Les six lignes restent toujours dans le dénominateur.

### 8.1 STOP

Filesystem, stabilité, hash/taille, manifeste incohérent, schéma parquet,
durabilité, receipt/event conflict ou corruption produisent `STOP`.

### 8.2 Quarantaine batch

UTF-8 invalide, BOM, NUL, fins de lignes mixtes, header ou forme CSV invalide
produisent un batch `QUARANTINED`, zéro ligne sûre et six preuves ordonnées
sans valeur brute.

### 8.3 Quarantaine ligne

Les problèmes de ligne sont cumulés, puis triés par `source_row_number` et
priorité :

```text
1 DUPLICATE_SOURCE_RECORD_ID
2 EMPTY_REQUIRED_PROVENANCE
3 PROVENANCE_MISMATCH
4 LOCATION_RULE_FAILED
5 UNICODE_DECIMAL_9_OR_14
```

Une ligne ayant plusieurs raisons produit une preuve par raison, dans cet
ordre. Le batch reste `INGESTED` avec les autres lignes sûres.

Le scan décimal projette NFKC sans altérer le brut, sur toutes les strings. Il
détecte les séquences autonomes de 9/14 caractères `isdecimal()` ASCII,
Unicode `Nd`, mixtes et superscript après NFKC.

## 9. Arbre de scan exact

Après parsing, deux builds indépendants avec mêmes fixture et
`logical_time_utc` doivent produire byte-for-byte le même arbre :

```text
safe_queries_preidentity.parquet
quarantine_proofs.parquet
source_identity_map.parquet
scan_integrity.json
scan_provenance.json
payload_manifest.json
seal.json
```

Schemas, types, nullabilité, tris et hashes logiques sont fixés dans le plan.
`quarantine_proofs.parquet` est row-level et ne stocke aucune valeur brute.
Toutes les colonnes string des trois parquets et deux JSON sont rescannées.
`query_id`, `opaque_batch_id`, `opaque_stratum_id`, run, attempt et receipts
sont `[a-p]{64}`.

Les domaines, projections et trois golden vectors opaques sont repris
exactement du plan fresh. Les douze golden vectors Unicode sont également
repris exactement ; leur projection canonique combinée avec les vecteurs
opaques a le SHA-256
`9af0547f33d4caf6aab89655b5a1e357068e0adc519e70135eab9a850425b64f`.

## 10. Automates et événements globaux

Batch :

```text
WAITING_STABLE -> RECEIPTED -> INGESTED
WAITING_STABLE|RECEIPTED -> STOPPED
RECEIPTED -> QUARANTINED
```

Collection :

```text
WAITING -> INGESTED
WAITING -> QUARANTINED
WAITING -> STOPPED
```

Ordre durable :

1. arbre entrée scellé ;
2. receipt collection ;
3. receipt batch ;
4. seq `00000001`, batch `WAITING_STABLE -> RECEIPTED` ;
5. scan + arbre scan scellé ;
6. seq `00000002`, batch `RECEIPTED -> INGESTED|QUARANTINED` ;
7. seq `00000003`, collection `WAITING -> INGESTED|QUARANTINED|STOPPED`.

Une collection est `INGESTED` si le batch est `INGESTED`, y compris avec
quarantaines ligne. Elle est `QUARANTINED` si le batch entier l’est.
Une erreur d’intégrité rend collection et batch `STOPPED`.

Chaque événement porte `entity_kind=BATCH|COLLECTION`, `entity_id`, une
séquence globale zero-padded à huit chiffres et un timestamp audit RFC3339 UTC
seconde. Le premier `previous_event_sha256` et le premier
`previous_manifest_sha256` valent `null`. Chaque event est `O_EXCL`, nommé par
son hash, puis fermé dans un manifest générationnel exact. Les maps
`manifest_hashes` et `tree_hashes` ont les clés exactes du plan.

## 11. Recovery et crash matrix

Après le premier receipt collection, l’inbox est définitivement interdite.
La reprise suit seulement :

`collection receipt -> batch receipt -> dernière génération complète ->
arbres annoncés`.

Préfixes :

- seal seul : valider seal, créer collection receipt ;
- collection receipt seul : créer batch receipt depuis l’arbre enregistré ;
- deux receipts sans event : créer seq 1 ;
- seq 1 manifesté sans scan : parser l’arbre scellé et construire le scan ;
- scan scellé sans seq 2 : rattacher sans recalcul ;
- seq 2 sans seq 3 : dériver l’état collection et créer seq 3 ;
- event/temp non manifesté : conserver comme orphan, ignorer ;
- conflit référencé : `STOP`.

Aucun effet durable n’est rejoué, aucun fichier n’est supprimé ou remplacé.

## 12. Verdicts

S0 s’arrête toujours à :

- `INGESTED_SYNTHETIC_SCANNER_SEALER_V412` ;
- `QUARANTINED_SYNTHETIC_SCANNER_SEALER_V412` ;
- `STOP_SYNTHETIC_SCANNER_SEALER_V412`.

Il ne qualifie aucune identité et n’autorise aucun intake réel.

## 13. Fermeture P0 normative

Cette section remplace toute formulation antérieure incompatible.

### 13.1 Identité non circulaire et chemins

Le control manifest porte `fixture_spec_sha256`, hash du JSON canonique compact
UTF-8 sans LF de l'objet `fixture` exact du plan, indépendant de tout run.
`synthetic_run_id` est dérivé de
`H({fixture_spec_sha256,plan_sha256})`. Le producteur déterministe construit
ensuite les cinq fichiers exacts, puis le lock pinne leurs octets et le control
manifest complet avant tout démarrage du scanner. Il n'existe aucun arrêt
optionnel statistique ni aucune reconstruction adaptative de la fixture.

Chemins exacts :

```text
<root>/sealed/<run>/input
<root>/scan/<run>/output
<root>/quarantine/<run>/batch
<root>/inbox/<run>/package
<root>/tmp/<run>
<root>/audit/<run>/receipts/{collections,batches}
<root>/audit/<run>/{events,events_manifests}
```

`run` est un composant `[a-p]{64}`. Le sandbox n'autorise que ces sous-arbres
du run et les FDs payload synthétiques explicitement transmis. Il interdit
l'énumération du parent et tout autre arbre. L'audit ne peut écrire que dans
les quatre sous-arbres exacts `receipts/collections`, `receipts/batches`,
`events` et `events_manifests`.

### 13.2 Classification ferme

Les raisons batch, évaluées jusqu’à la première dans cet ordre, sont :

```text
CSV_ENCODING_INVALID
CSV_BOM_FORBIDDEN
CSV_NUL_FORBIDDEN
CSV_MIXED_LINE_ENDINGS
CSV_HEADER_DRIFT
CSV_ROW_SHAPE_DRIFT
```

Elles produisent l’arbre distinct exact
`{batch_quarantine_proof.json,payload_manifest.json,seal.json}`. La preuve
porte raison unique, hashes de l’entrée scellée et
`expected_source_row_count=6`, sans locator ni valeur brute.

`quarantine_proofs.parquet` concerne seulement les lignes parsables. Les
raisons ligne s’accumulent dans l’ordre préenregistré. Une provenance vide
supprime le `PROVENANCE_MISMATCH` redondant du même champ. Toute occurrence
d’un identifiant dupliqué est quarantinée.

Avant receipt, un STOP ne crée aucun receipt ou événement autoritaire ; seuls
des temp orphelins ignorés peuvent rester. Après seq 1, une erreur d’intégrité
crée seq 2 batch `STOPPED`, puis seq 3 collection `STOPPED` seulement si le
journal est sain. Un journal corrompu interdit tout append et préserve le
dernier préfixe valide.

### 13.3 Maps d’événement

Toutes les maps ont leurs clés exactes ; chaque valeur vaut hex64 minuscule ou
`null`. Seq 1 possède uniquement les hashes entrée non nuls. Seq 2/3
`INGESTED` possèdent entrée+scan et quarantine nul. `QUARANTINED` possède
entrée+batch-quarantine et scan nul. `STOPPED` conserve entrée et tout arbre
déjà validé, avec les autres clés explicitement nulles.

### 13.4 Sorties et hashes

L’anti-fuite rescane uniquement toutes les strings de
`safe_queries_preidentity.parquet`, futur payload query. Un hit est `STOP`.
Les maps privées et digests ne sont pas soumis à ce scan ; leurs hashes sont
validés par `^[0-9a-f]{64}$`.

`safe_queries_preidentity.parquet` contient exactement :

```text
query_id, opaque_batch_id, opaque_stratum_id, reference_date,
crm_name_raw, crm_address_raw, crm_postcode_raw, crm_city_raw, crm_insee_raw
```

Il ne contient aucun identifiant source, numéro de ligne ou hash privé.
`source_identity_map.parquet` conserve en privé numéro 1-based, collection,
batch, record, system, portfolio, source hash, raw hash et les trois IDs
opaques. Les IDs sont calculés pour les six lignes parsables, y compris les
lignes quarantinées ; seules les lignes sûres sont émises dans le payload.

`raw_row_sha256` est le SHA-256 du JSON canonique ARRAY des neuf strings
décodées, dans l’ordre des colonnes, compact UTF-8 sans LF. Le locator privé
utilise :

```text
domain = SIRETO-V412-FRESH-SYNTHETIC-ROW-LOCATOR\0
projection = synthetic_run_id, source_batch_id, source_row_number,
             raw_row_sha256
```

`reason_counts` possède toutes les clés batch+ligne préenregistrées et des
valeurs `uint64`. `logical_hashes` possède exactement les cinq noms de
payload scan et des valeurs hex64.

### 13.5 Reproductibilité et writer

Les arbres payload entrée et scan sont reproductibles byte-for-byte pour les
mêmes fixtures et `logical_time_utc`. Les timestamps d’audit peuvent différer
et restent hors payloads reproductibles.

Writer parquet : pyarrow `23.0.1`, format `2.6`, zstd niveau `9`,
data page `1.0`, row group `65536`, dictionnaire faux, statistiques vraies,
`store_schema=true`, aucune métadonnée applicative, rechunk d’un chunk par
colonne.

### 13.6 Fixture fermée

Le CSV LF/RFC4180/sans BOM contient six lignes préenregistrées avec provenance
commune `source_batch_id=batch-synthetic-001`,
`source_system=SYNTHETIC`, `portfolio_id=PORTFOLIO-SYNTHETIC` :

1. `record-001`, nom `Mairie de Test`, adresse `1 rue Alpha`, CP `75001`,
   ville `Paris`, INSEE `75101` — sûre ;
2. `record-002`, nom `École Démo`, adresse `2 avenue Bêta`, CP `69002`,
   ville `Lyon`, INSEE `69382` — sûre ;
3. `record-003`, nom `Client 12345678901234` — fuite ASCII14 ;
4. `record-004`, adresse `Rue ١٢٣٤٥٦٧٨٩` — fuite ArabicIndic9 ;
5. `record-005`, nom `Mixte 12345６７８٩٠1234` — fuite mixed/fullwidth14 ;
6. `record-006`, ville `¹²³⁴⁵⁶⁷⁸⁹` — fuite superscript9 après NFKC.

Les champs de localisation non cités des lignes 3–6 reprennent respectivement
les valeurs sûres des lignes 1, 2, 1 et 2. Résultat exact : lignes sûres
`[1,2]`, preuves `[3,4,5,6]`, `UNICODE_DECIMAL_9_OR_14=4`, toutes les autres
raisons `0`. Evidence : zéro ligne, un row group.

La projection golden exacte est l’objet canonique
`{opaque:[vecteurs ordonnés],unicode:[vecteurs ordonnés]}` du plan S0 ; son
SHA-256 est
`9af0547f33d4caf6aab89655b5a1e357068e0adc519e70135eab9a850425b64f`.

Tous les objets JSON et Parquet suivent la notation de schema typée du plan :
`[nom,type,nullable]`, listes ordonnées et enums fermées.

### 13.7 Constantes exactes et octets CSV

La fixture impose :

```text
collection_id = collection-synthetic-001
export_snapshot_id = export-synthetic-001
reference_date = 2026-07-30
period_start_utc = 2026-07-30T00:00:00Z
period_end_utc = 2026-07-30T00:00:00Z
export_cutoff_utc = 2026-07-30T00:00:00Z
producer_created_at_utc = 2026-07-30T00:00:00Z
logical_time_utc = 2026-07-30T00:00:00Z
population_name = SIRETO_V412_SYNTHETIC_S0
population_definition = EXACT_SIX_PREREGISTERED_SYNTHETIC_ROWS
producer_manifest_id = producer-synthetic-001
authority_type = NONE_SYNTHETIC
v411_service_id_equivalence_attested = SYNTHETIC_ONLY_NOT_REAL_EQUIVALENCE
lineage_attestation_reference = SYNTHETIC_FIXTURE
```

Le champ `fixture.csv.exact_utf8_text` du plan constitue les octets normatifs
de `crm_safe.csv` : UTF-8, aucune cellule citée inutilement, quoting minimal
RFC4180, séparateur virgule, fins de lignes LF et un unique LF final.

### 13.8 Manifest payload et seal communs

Chaque `payload_manifest.json` possède exactement, dans cet ordre :

```text
schema_version, package_kind, synthetic_run_id, collection_id,
source_batch_id, logical_time_utc, ordered_payload_records, payload_count,
payload_tree_sha256
```

`package_kind` est l'enum fermé
`SEALED_INPUT|SCAN_OUTPUT|BATCH_QUARANTINE`. Chaque record possède exactement
`relative_path:string`, `size_bytes:uint64`, `sha256:hex64`.
`payload_count` égale la longueur du tableau.
`payload_tree_sha256` vaut le SHA-256 du tableau
`ordered_payload_records` en JSON canonique compact UTF-8, dans son ordre
exact, sans LF.

Chaque `seal.json` possède exactement les champs du plan : identités et temps
logique, `package_kind`, taille et SHA-256 des octets du payload manifest, et
le même `payload_tree_sha256`. Le payload manifest ne se hash jamais lui-même :
son SHA-256 est une valeur externe portée par le seal, les receipts, les
événements et le lock selon leur contrat.

### 13.9 Typage des receipts, événements et générations

Les deux receipts n'admettent aucun `null`. Leur `receipt_kind` et leur
`schema_version` sont des constantes fermées ; tailles en `uint64`, hashes en
hex64 minuscule, IDs en `[a-p]{64}` et chemins liés au run exact. Le receipt
batch possède un `source_batch_id` non vide ; le receipt collection n'a pas ce
champ.

Chaque événement possède tous les champs et toutes les clés des deux maps
fermées du plan. `source_batch_id` est non nul pour `BATCH` et exactement
`null` pour `COLLECTION`. `previous_event_sha256` est `null` uniquement en
séquence 1. Les enums d'entité et d'état sont fermées par le plan ; toutes les
valeurs de hash optionnelles sont explicitement hex64 ou `null`.

Un manifest générationnel est cumulatif : génération `n` contient exactement
les records ordonnés des événements 1 à `n`, sans trou. `generation`,
`event_count` et la séquence du dernier record sont égaux.
`head_event_sha256` est le hash de ce dernier record.
`previous_manifest_sha256` vaut `null` pour la génération 1, puis exactement
le SHA-256 des octets du manifest générationnel complet précédent.

### 13.10 Reprise fermée des deux branches

Après seq 1, un arbre `SCAN_OUTPUT` scellé sans seq 2 et un arbre
`BATCH_QUARANTINE` scellé sans seq 2 suivent la même discipline : valider
d'abord `package_kind`, arbre exact, manifest, seal, identités, temps logique
et liaison aux hashes d'entrée de seq 1 ; ensuite seulement rattacher seq 2,
sans recalcul ni remplacement. Chaque préfixe durable complet, les orphelins
non manifestés et les conflits référencés sont des cas distincts obligatoires
du plan et de la crash matrix.

### 13.11 Hash logique non récursif

Le hash logique de `scan_integrity.json` est calculé sur la projection exacte
du plan qui exclut le champ `logical_hashes`. Le document qui contiendrait son
propre hash n'entre jamais dans l'entrée de calcul : aucune équation de
point-fixe ni hash auto-référent n'est admis.

### 13.12 Matrice négative obligatoire

La suite future doit comporter au moins un test indépendant pour chaque raison
batch et chaque raison ligne, les mutations et substitutions pendant la
stabilité, les conflits d'arbre/manifest/seal/package/binding, chaque préfixe
de crash et chaque canari du sandbox. Une réussite du seul golden path ne
suffit pas à autoriser l'exécution. S0 reste
`SPECIFICATION_ONLY_DO_NOT_EXECUTE`.

### 13.13 Schémas JSON fermés complémentaires

Le receipt batch référence comme source scellée
`sealed_source_relative_path=crm_safe.csv` ; sa taille et son hash portent sur
les octets de ce fichier, jamais sur `source_manifest.json`.

Les manifests collection, source et evidence, la preuve de quarantaine batch,
`scan_integrity.json` et `scan_provenance.json` suivent les maps de types
exhaustives du plan. Tous leurs champs sont non nuls. Leurs `schema_version`
sont respectivement les constantes :

```text
sireto-v4.12-fresh-synthetic-collection-source-manifest-1
sireto-v4.12-fresh-synthetic-source-manifest-1
sireto-v4.12-fresh-synthetic-evidence-source-manifest-1
sireto-v4.12-fresh-synthetic-batch-quarantine-proof-1
sireto-v4.12-fresh-synthetic-scan-integrity-1
sireto-v4.12-fresh-synthetic-scan-provenance-1
```

Les enums, constantes, booléens, `uint64`, hashes hex64, IDs `[a-p]{64}`,
tableaux et maps fermées sont ceux du plan, sans coercition implicite.
`source_record_id_semantics` vaut exactement
`UNIQUE_WITHIN_BATCH_OPAQUE_SOURCE_IDENTIFIER`.

### 13.14 Canonicalisation, hashes Parquet et chemins autoritaires

Le plan final est lui-même sérialisé en JSON canonique : clés triées,
séparateurs compacts, UTF-8 et un unique LF final.

Le hash logique de chaque Parquet est le SHA-256, sans LF, du JSON canonique
compact d'un ARRAY de ROW OBJECTS. Les clés de chaque row object suivent
exactement l'ordre des champs du schéma Parquet et les lignes exactement le
tri déclaré. Les seules valeurs JSON admises sont string, integer, boolean et
null, sans coercition. Une table vide se sérialise exactement `[]`.

Les chemins relatifs autoritaires sont exactement :

```text
receipts/collections/<receipt_id>/receipt.json
receipts/batches/<receipt_id>/receipt.json
events/<sequence:08d>-<event_sha256>.json
events_manifests/<generation:08d>-<generation_manifest_sha256>.json
```

Il existe exactement un manifest générationnel complet par numéro de
génération. Toute concurrence ou coexistence conflictuelle produit `STOP` ;
aucun timestamp ne départage des candidats.

### 13.15 Provenance du code pré-lock

Avant l'introduction du lock, `builder_source_sha256` n'est jamais un sentinel.
Il vaut le SHA-256 sans LF du JSON canonique compact portant exactement
`producer_sha256` et `scanner_sha256`, chacun calculé sur les octets du fichier
source correspondant. `tests_sha256` vaut le SHA-256 des octets exacts de
`tests/test_v412_fresh_intake_synthetic_scanner_sealer.py`. La reprise
recalcule ces valeurs et refuse tout arbre scan construit par d'autres octets.
