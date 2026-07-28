# V4.12 — Contrat du moteur unitaire de retrieval et de sa parité

Statut : préenregistré après `GO_V412_STRICT_STORES_SANDBOX`, avant toute
implémentation du moteur, toute sortie candidat et toute ouverture de
l'oracle par un évaluateur.

## 1. Objet

Construire le premier chemin honnête :

```text
une requête CRM sûre
→ une partition géographique
→ le TF-IDF gelé
→ au plus 100 SIRET actifs
```

Le jalon possède deux phases strictement séquentielles :

1. un worker aveugle produit et scelle les listes de candidats ;
2. après sa terminaison, un contrôleur sans oracle vérifie leur parité exacte
   avec le retrieval V4.11 historique.

L'oracle dev, les modèles et le test final restent fermés pendant les deux
phases. Une évaluation Recall séparée ne sera autorisée qu'après le verdict
de ce jalon.

Verdicts :

- `GO_V412_UNIT_RETRIEVAL_PARITY` ;
- `STOP_V412_UNIT_RETRIEVAL`.

Ce `GO` prouve que le moteur requête par requête reproduit le retrieval
historique. Il ne certifie ni sa justesse métier, ni la précision AUTO, ni la
production.

## 2. Entrées autorisées au worker

### CRM sûr

Build :

```text
ca0b22e79cd2e92a32c009266e6d967b4ea48654de8736bca2b0ea7fdc9f8d6e
```

Fichier :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/inputs/v4_12_unit_engine/
ca0b22e79cd2e92a32c009266e6d967b4ea48654de8736bca2b0ea7fdc9f8d6e/
queries_dev.parquet
```

SHA-256 :

```text
b1fef6ba72e4a557175a60c7e21e658f8fbb739b1d6607b77e5f70da873a031f
```

Le worker lit physiquement et exclusivement :

```text
query_id,crm_name,crm_address,crm_postcode,crm_city,crm_insee
```

Il exige 1 456 IDs uniques dans l'ordre déjà scellé par
`SHA-256("v412-unit-engine:" + query_id), query_id`. Aucun identifiant
SIRET/SIREN CRM, label, split, score, preuve ou décision n'est disponible.

### Stores certifiés

Certification Gate A :

```text
build_id =
9a99cd246d6d1a118dea064ab1458afe7c3bcb8a9bb28a1da6009d6bc42b4ee4

manifest SHA-256 =
6854b115f06eb1b1d0dacc8dfc44e2d7d6d470427d80628253ca09a79a443d2d

run_spec SHA-256 =
c53d08dafb0d0c32a60de51737199191b5aad5d2b39a1649a4a76d58508138bc

lookup descriptor SHA-256 =
3e58872c94b17b5e19dee762e3c33f2bcf7407f948c902aa398b74705d34ce6c
```

Le worker réutilise sans modification :

- `StrictPartitionStore` ;
- `_build_aligned_pool` ;
- `StrictVerifiedTfidfCache` ;
- `StrictSnapshotLookup`.

Le blob déjà certifié `src/xgb_matcher/v412_strict_stores.py` ne sera pas
modifié pour ce jalon. Sa source vérifiée est copiée dans l'espace privé du
worker, comme lors du Gate A.

La liste blanche data reste exactement celle du Gate A : 648 partitions,
648 pickles, 648 sidecars et une base lookup, soit 1 945 fichiers.

Le cache est épinglé par :

```text
namespace =
296c7891107249a073c00d93c7310c55a652243de4bcfa7165d09dbfc3349a82

tfidf_config_artifact_hash =
92b68d1f7aa386f181edbede280e58df72f8583d7663419d77da88300d241c61

sparse_config_hash =
aeaa671959fc00dcec2e8a5393976d1e68da9dfa5ae48ef4d836e9dbdc3c564e
```

Son tuple exact reste `name_vec,name_mat,names,char_vec,char_mat,addr_vec,
addr_mat`. Un miss donne `STOP`; rebuild et écriture sont impossibles.

## 3. Routage géographique

Le routage reste identique au Gate A :

1. normaliser `crm_insee` ;
2. si sa partition inventoriée existe, choisir `<insee>_` ;
3. sinon normaliser `crm_postcode` ;
4. si sa partition inventoriée existe, choisir `_<postcode>` ;
5. sinon arrêter le run, jamais retourner implicitement un pool vide.

La normalisation est celle du contrat Gate A. Les attendus sont :

- 1 449 routes INSEE ;
- 7 routes code postal ;
- 648 clés distinctes ;
- zéro clé manquante ;
- payload de routage de 20 491 octets, SHA-256
  `41477bbcc9dea2cee8d49922679064995c46e49f60db1560e5d8a0033adc79bf`.

## 4. Algorithme sparse figé

Le nouveau moteur n'importe ni `retrieval.py`, ni `v41_retrieval.py`, ni le
builder V4.11. Il réimplémente localement les seules opérations ci-dessous et
les compare différentiellement à leurs références historiques avant le run.

### 4.1 Pool aligné

Pour la partition routée :

1. conserver l'ordre physique Parquet ;
2. retirer les lignes sans aucun des neuf champs de nom gelés ;
3. retirer seulement `etat_admin == "F"` ;
4. former `siret=str(value or "").strip()` et retirer les vides ;
5. dédupliquer par affectation dictionnaire « dernière valeur gagnante »,
   tout en gardant l'ordre de première insertion de chaque SIRET.

Le cache TF-IDF doit concorder avec ce pool. Tout miss, écart de dimensions,
rebuild ou écriture donne `STOP`.

### 4.2 Prétraitement CRM minimal

Le moteur reproduit exactement les opérations historiques nécessaires au
retrieval :

- le TF-IDF nom et le rescue numérique reçoivent `crm_name` brut, sans
  retrait de commune, code postal ou INSEE ;
- normalisation de l'adresse ;
- extraction du numéro de voie en tête ;
- retrait du numéro et des suffixes `BIS|TER|QUATER|B|T|Q` pour obtenir la
  voie ;
- extraction des tokens numériques du nom.

Le prétraitement historique plus large est reproduit en test différentiel,
mais ses variantes de nom nettoyé et sa commune n'entrent pas dans l'identité
du pool de ce retrieval. Il ne calcule aucune feature de modèle.

Avant `vectorizer.transform`, le nom brut et l'adresse normalisée passent
séparément par `normalize_text_for_tfidf` :

1. normaliser texte, accents, tirets et espaces comme au Gate A ;
2. remplacer la regex exacte `[^\w\s]` par un espace, puis réduire les
   espaces ;
3. ajouter à la fin l'acronyme compact de chaque suite d'au moins deux tokens
   alphabétiques d'une lettre ;
4. dédupliquer les tokens en conservant leur première apparition ;
5. pour chaque token alphabétique d'au moins cinq lettres, ajouter si absente
   la variante `AUX→AL`, sinon retirer un `S` final sauf `SS`, sinon retirer
   un `X` final.

Le score de chaque canal est calculé directement par :

```text
(vectorizer.transform([requête normalisée]) @ matrice.T).getrow(0)
```

Les vectorizers et matrices gelés sont utilisés sans refit, normalisation ou
cast. Toute valeur non finie dans la requête vectorisée, le produit sparse ou
les scores avant classement donne `STOP_V412_UNIT_RETRIEVAL`.

### 4.3 Canaux

Configuration exacte :

```text
include_closed=false
drop_unnamed=true
tfidf_name_mode=bag
sparse_retrieval_enabled=true
dense_retrieval_enabled=false
prefilter_k=500 par canal
prefilter_trigger_size=1
retrieval_budget=100
min_candidates=50
fusion_mode=rrf
sparse_channel_fusion_mode=separate_rrf
rrf_k=60
rescue_addr_hash=true
rescue_numeric_tokens=true
mega_insee_policy=full_insee
siren_siblings=false
```

Les trois canaux sont :

1. `sparse_name` :
   - appliquer `_rank_sparse_scores(..., 500)` séparément au produit mot et
     au produit caractères ;
   - initialiser le dictionnaire par les 500 hits mot ;
   - pour les 500 hits caractères, conserver le maximum par index ;
   - trier l'union en Python par `(-score, index)` et garder 500 ;
2. `sparse_address` : retourner directement
   `_rank_sparse_scores(produit_adresse, 500)`, sans second tri par index ;
3. `rescue` : union du rescue adresse et du rescue numérique, puis SIRET
   croissants avant conversion vers leur index.

Le ranking sparse d'une row non vide :

- si plus de 500 valeurs sont non nulles, sélection par
  `sel=numpy.argpartition(data,-500)[-500:]`, appliquée à `indices` et
  `data` ;
- `order=numpy.argsort(data,kind="stable")[::-1]` ;
- retour de `indices[order],data[order]`.

Pour `sparse_address`, l'égalité suit donc l'ordre inversé du buffer sparse
retenu par cette opération, et non un index croissant inventé. Pour
`sparse_name`, le second tri `(-score,index)` stabilise seulement l'union des
deux top-500 déjà sélectionnés.

Le rescue adresse est exact :

- CRM : numéro de voie en tête et voie extraite de l'adresse ;
- candidat : `numeroVoie`, sinon `street_number`; voie
  `typeVoie + libelleVoie`, avec les aliases `street_type + street_name` ;
- normaliser la voie puis retirer exactement
  `RUE,AV,AVE,AVENUE,BD,BOULEVARD,CHE,CHEMIN,IMP,IMPASSE,ALL,ALLEE,PL,PLACE,SQ,SQUARE,ROUTE,RTE,QUAI,SENTIER,ZA,ZI,ZAC` ;
- hash `numero|voie` si le numéro existe, sinon `voie`; aucun hash si la voie
  est vide.

Le rescue numérique extrait par `\b\d{1,6}\b` les tokens du `crm_name` brut
et du seul `primary_name` candidat. Ce nom primaire est le premier nom
normalisé, non numérique et de longueur supérieure à deux selon la priorité :

```text
enseigne1
→ denomination
→ enseigne2 puis enseigne3
→ sigle_ul
→ denomination_usuelle_ul
→ denomination_ul
→ pm_dirigeant_names dans leur ordre
→ prenom_usuel_ul + nom_ul si cj_ul est un code personne autorisé
```

La normalisation du nom primaire est celle déjà figée au Gate A, dont le
retrait des formes juridiques et le plafond de 100 caractères.

### 4.4 Fusion et padding

Pour chaque canal et chaque candidat de rang `r` :

```text
score += 1 / (60 + r)
```

La fusion intermédiaire trie par :

```text
score décroissant,
meilleur rang de canal croissant,
str(index de row) croissant
```

Elle garde au plus 100 index. Si elle en contient moins que
`min(100, taille du pool aligné)`, le moteur complète par les index encore
absents dans l'ordre stable du pool. Le padding possède un score RRF nul.

Même si aucun canal ne retourne un index, la branche RRF complète par padding
jusqu'à `min(100, taille du pool aligné)`. Pour un pool de taille zéro ou un,
le trigger `taille > 1` n'est pas pris : le pool reste inchangé, son score
implicite vaut zéro et le tri final SIRET s'applique.

Le moteur hydrate alors ces seuls SIRET par le lookup, en un appel d'au plus
100 :

- un SIRET absent du lookup est omis comme dans le `LEFT JOIN` historique et
  incrémente `lookup_missing_count` dans l'intégrité agrégée ;
- seul `candidate_state == "A"` exactement est conservé ;
- tri final par score RRF décroissant puis SIRET croissant ;
- plafond final de 100 ;
- rangs réattribués de 1 à `K`.

Il n'existe aucun canal dense, SIREN global, identifiant CRM, positif injecté,
fallback web ou padding depuis la vérité.

## 5. Sortie minimale du worker

Racine :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/runs/
v4_12_unit_retrieval/<build_id>
```

Fichiers runtime exacts :

### `query_status.parquet`

| Colonne | Type | Nullable |
|---|---|---|
| `query_id` | string | non |
| `candidate_count` | uint8 | non |

Contraintes :

- exactement 1 456 lignes et les 1 456 IDs CRM dans le même ordre ;
- une ligne existe même si `candidate_count=0` ;
- `0 <= candidate_count <= 100` ;
- aucune metadata Arrow.

### `candidates_top100.parquet`

| Colonne | Type | Nullable |
|---|---|---|
| `query_id` | string | non |
| `candidate_rank` | uint8 | non |
| `candidate_siret` | string | non |

Contraintes :

- ordre des requêtes CRM, puis rang croissant ;
- rangs contigus `1..candidate_count` ;
- SIRET conforme à `^[0-9]{14}$` ;
- SIRET unique dans chaque requête ;
- au plus 145 600 lignes ;
- aucun score, feature, vérité, hit, label, décision ou metadata Arrow.

### Contrôle

`integrity.json` et `manifest.json` publient uniquement identité du build,
hashes/schémas/volumes, compteurs agrégés, runtime, déclarations anti-fuite et
verdict technique. Aucun identifiant de miss ou résultat par requête n'est
recopié dans un JSON ou un log.

`worker_build_id` est le SHA-256 du JSON canonique possédant exactement :

```text
schema_version,worker_policy_sha256,worker_lock_projection_sha256,
parent_runner_sha256,worker_source_hashes,
safe_input_build_id,safe_runtime_manifest_sha256,
safe_queries_dev_sha256,strict_stores_build_id,
strict_stores_manifest_sha256,retrieval,tfidf_cache,runtime
```

`worker_policy_sha256` couvre uniquement les projections `prerequisite`,
`safe_input`, `retrieval`, `tfidf_cache`, `expected_routing`, `outputs`,
`runtime` et `max_rss_bytes` du plan. La projection du verrou contient
uniquement les sources/inputs/profils/racines/runtime du worker. Le plan et le
verrou complets ne sont jamais transmis au worker.

Le worker reçoit ces deux hashes de projection comme valeurs opaques, jamais
leurs objets JSON. Son identité n'embarque directement aucun oracle, label,
attendu de parité, chemin ou hash de la référence historique. Elle reste liée
par des empreintes opaques à la certification Gate A et à ses artefacts
ancêtres ; cette dépendance de provenance ne fournit au worker aucune
information de comparaison. Le plan complet reste lié uniquement dans la
provenance parent-only.

`integrity.json` possède exactement :

```text
schema_version,worker_build_id,query_count,candidate_count,
minimum_pool_size,maximum_pool_size,under_ceiling_query_count,
empty_query_count,lookup_missing_count,candidate_payload_bytes,
candidate_payload_sha256,status_payload_bytes,status_payload_sha256,
sandbox_checks,peak_rss_bytes,durations_ns,declarations
```

`sandbox_checks` possède exactement
`allowed_read,oracle_denied,oracle_audit_denied,historical_denied,
model_denied,network_denied,write_denied`.
`durations_ns` possède exactement
`retrieval,lookup,serialization,total`, entiers non négatifs.

Le manifeste runtime possède exactement :

```text
schema_version,worker_build_id,safe_input_build_id,
strict_stores_build_id,files,runtime,declarations,verdict
```

`files` scelle exactement `query_status.parquet`,
`candidates_top100.parquet` et `integrity.json`, avec pour chaque Parquet
hash, taille, nombre de rows, schéma complet et metadata `null`. Le verdict
worker est `SEALED_V412_UNIT_RETRIEVAL`, jamais le verdict de parité.

Les déclarations exactes du worker sont :

```text
labels_opened=false
oracle_opened=false
historical_candidates_opened=false
models_opened=false
network_used=false
writes_outside_staging=false
cache_rebuild_attempted=false
positive_injection=false
```

La preuve sensible séparée contient un ledger exhaustif des fichiers ouverts,
la provenance Git et le hash du manifeste runtime, sous :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/
v4_12_unit_retrieval_worker/<build_id>
```

Elle contient exactement `open_ledger.parquet`, `provenance.json` et
`manifest.json`. La provenance possède exactement :

```text
schema_version,worker_build_id,git_commit,parent_source_hashes,
worker_source_hashes,lock_sha256,plan_sha256,runtime,
data_input_count,runtime_manifest_sha256,declarations
```

Le manifeste d'audit scelle exactement le ledger et la provenance.
Il possède exactement `schema_version,worker_build_id,files`; `files`
contient uniquement les records `open_ledger.parquet` et `provenance.json`
avec `path,size_bytes,sha256`.

Après la fin du worker, son audit est publié avant la publication finale de
l'artefact runtime, selon le protocole pending/final déjà validé au Gate A.
Racines finales `0555`, fichiers `0444`.

## 6. Sandbox du worker

Le profil reprend la frontière certifiée au Gate A :

- `deny default`, réseau et fork interdits ;
- runtimes système explicitement admis ;
- sources du moteur transmises par descripteurs ou copies privées scellées ;
- `queries_dev.parquet`, run-spec et descriptor ouverts
  `O_RDONLY|O_NOFOLLOW`, hashés puis consommés par descripteurs hérités ;
- lecture data limitée aux 1 945 fichiers du Gate A ;
- écriture limitée aux espaces privés `output` et `tmp` ;
- oracle, audit oracle, référence historique, modèles, challenges, final et
  tests explicitement interdits.

Sous le même processus worker, les sentinelles doivent prouver :

- lecture d'une entrée autorisée réussie ;
- ouverture de l'oracle refusée avec `EPERM` ;
- ouverture de l'audit oracle refusée avec `EPERM` ;
- ouverture de la référence historique refusée avec `EPERM` ;
- ouverture d'un modèle existant refusée avec `EPERM` ;
- réseau refusé ;
- écriture hors staging refusée.

Le worker doit être terminé et tous ses descripteurs fermés avant le démarrage
du contrôleur de parité.

## 7. Parité historique cryptographique post-seal

Le contrôleur de parité est un exécutable distinct, sans accès à l'oracle.
Il ouvre uniquement la sortie finale scellée et les IDs sûrs, par des
descripteurs ancrés `O_RDONLY|O_NOFOLLOW`, avec hashes et identités
avant/après. Il ne reçoit jamais le Parquet historique.

Le fichier suivant est une source de provenance **parent-only** utilisée
avant le gel du contrat pour calculer les attentes, puis indépendamment par
les contre-auditeurs :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_input_blind/
ec4326ec57e4411d/candidates_sparse_top100.parquet

SHA-256 =
78b2f78ddeac863ac39ca64301d42312c7fb766ac51e2b5d19dde5c5910aedac
```

Projection physique exclusive :

```text
query_id,retrieval_rank,candidate_siret
```

Ce fichier contient physiquement `is_ground_truth` et des features. Pour
cette raison, ni le worker ni le contrôleur ne peuvent l'ouvrir, même avec
une projection promise. Seuls le préenregistrement et les audits de confiance
l'ont projeté sur ces trois colonnes, filtré par semi-jointure aux 1 456 IDs
sûrs, puis ordonné selon l'ordre CRM et `retrieval_rank`.

Attendus préenregistrés :

| Contrôle | Valeur |
|---|---:|
| Requêtes | 1 456 |
| Lignes candidates | 145 236 |
| Pool minimum | 46 |
| Pool maximum | 100 |
| Requêtes avec moins de 100 candidats | 13 |
| Requêtes sans candidat | 0 |

Payload candidat :

```text
query_id + 0x00 + candidate_siret + 0x00
+ candidate_rank décimal ASCII + 0x0A
```

- taille : 3 629 947 octets ;
- SHA-256 :
  `1689a2f363cac7385dcfec10606c32e89d3f904e3990cda00989c13feb87ab00`.

Payload statut :

```text
query_id + 0x00 + candidate_count décimal ASCII + 0x0A
```

- taille : 16 110 octets ;
- SHA-256 :
  `65e662c0df3df6bd022da55cde1e2d6b13254e16314c71dad5cc063dde6d5518`.

Le contrôleur valide d'abord les schémas, IDs, rangs, unicités, compteurs et
ordres ligne par ligne dans la sortie. Il sérialise ensuite chaque séquence
dans les deux payloads canoniques ci-dessus. L'égalité des tailles et SHA-256
préenregistrés constitue la parité cryptographique de toutes les séquences
ordonnées. Un écart unique modifie le payload et donne
`STOP_V412_UNIT_RETRIEVAL`; il n'existe aucune tolérance ou acceptation d'un
« résultat meilleur » dans ce jalon.

Le contrôleur ne peut pas identifier un candidat attendu lors d'un échec,
puisqu'il ne connaît pas la référence ligne à ligne. Il publie uniquement le
type de contrôle et les hashes/compteurs obtenus. Tout diagnostic ultérieur
exige un nouveau contrat parent-only ; il n'est pas improvisé après le run.

Son profil sandbox autorise exactement :

- les deux Parquet et manifests de la sortie worker scellée ;
- `queries_dev.parquet` et son manifeste sûr ;
- son run-spec assaini, le seul hash du verrou et ses propres sources
  scellées ;
- son staging privé.

Il interdit explicitement le Parquet historique, l'oracle et son audit, les
stores, modèles, challenges, final, évaluations précédentes, réseau et fork.
Il teste sous le même profil les refus de l'oracle, de son audit, de la
référence historique, d'un modèle et du réseau.

Le contrôleur reçoit un `parity_run_spec.json` assaini, sans chemin
historique ni oracle. Il contient seulement les identités et hashes des
inputs autorisés, les tailles et hashes attendus des deux payloads, les
compteurs attendus, son staging privé et les déclarations.

`parity_build_id` est le SHA-256 du JSON canonique possédant exactement :

```text
schema_version,worker_build_id,worker_manifest_sha256,
worker_file_hashes,parity_run_spec_sha256,parity_source_hashes,
parity_profile_sha256,lock_sha256,runtime
```

Il est distinct du `worker_build_id`.

`parity.json` possède exactement :

```text
schema_version,parity_build_id,worker_build_id,query_count,
candidate_count,minimum_pool_size,maximum_pool_size,
under_ceiling_query_count,empty_query_count,
candidate_payload_bytes,candidate_payload_sha256,
expected_candidate_payload_bytes,expected_candidate_payload_sha256,
status_payload_bytes,status_payload_sha256,
expected_status_payload_bytes,expected_status_payload_sha256,
checks,sandbox_checks,declarations,verdict
```

`checks` possède exactement
`schemas,metadata,query_population,query_order,counts,ranks,sirets,
candidate_payload,status_payload`. `sandbox_checks` possède exactement
`allowed_read,oracle_denied,oracle_audit_denied,historical_denied,
model_denied,stores_denied,network_denied,write_denied`.

Les déclarations exactes du contrôleur sont :

```text
oracle_opened=false
oracle_audit_opened=false
historical_candidates_opened=false
models_opened=false
stores_opened=false
network_used=false
writes_outside_staging=false
```

Le verdict exact est `GO_V412_UNIT_RETRIEVAL_PARITY` ou
`STOP_V412_UNIT_RETRIEVAL`. L'audit parité contient exactement
`parity.json`, `provenance.json` et `manifest.json`; son manifeste scelle les
deux autres fichiers, et la provenance lie commit, sources, verrou, runtime
et manifeste worker.

La provenance parité possède exactement :

```text
schema_version,parity_build_id,worker_build_id,git_commit,
parity_source_hashes,lock_sha256,parity_run_spec_sha256,
worker_manifest_sha256,runtime,declarations
```

Le manifeste parité possède exactement :

```text
schema_version,parity_build_id,worker_build_id,files,
runtime,declarations,verdict
```

`files` contient uniquement `parity.json` et `provenance.json`, chacun avec
`path,size_bytes,sha256`.

## 8. Publication et verrou

Ordre obligatoire :

1. contre-audit exact du présent contrat et du plan encore non suivis ;
2. commit isolé des deux fichiers sans autre modification ;
3. réception du blob du commit et confirmation du même contenu ;
4. implémentation moteur, runner, contrôleur et tests ;
5. contre-audit indépendant du code et mini-publication ;
6. commit isolé du code ;
7. verrou externe épinglant commit, sources, inputs, Gate A et runtime ;
8. deux contre-audits du verrou ;
9. unique run worker ;
10. validation et scellement des sorties ;
11. contrôle de parité post-seal ;
12. contre-audit indépendant des deux artefacts ;
13. rapport et handover.

Le verrou n'existe qu'après le commit code audité. Il n'a aucun flag de
bypass et fixe au minimum :

- les sources Git du moteur, runner, contrôleur et tests ;
- les six fichiers du paquet CRM sûr ;
- la certification Gate A et sa preuve ;
- les 1 945 fichiers data autorisés ;
- le hash de provenance parent-only de la référence historique, inaccessible
  au worker et au contrôleur ;
- les deux profils sandbox et leurs exécutables ;
- le runtime scientifique exact ;
- les racines output/audit/temp ;
- un pic RSS inférieur ou égal à 8 Gio.

Chaque lecture sensible est résolue composant par composant sans symlink,
ancrée depuis un FD de répertoire, ouverte avec `openat` et `O_NOFOLLOW`, puis
consommée par Arrow ou JSON via le même FD. `fstat`, taille et SHA-256 sont
contrôlés avant et après sur ce FD ; aucun second `open(path)` n'est utilisé
pour la consommation.

Publication worker :

```text
staging privé
→ pending runtime privé et validé
→ audit worker final
→ runtime final
```

Publication parité :

```text
staging privé
→ pending parité validé
→ audit parité final
```

Les états final/pending/audit et leur reprise suivent les mêmes règles
fail-closed et APFS que le Gate A. Le parent publie ; les processus sandboxés
n'écrivent jamais dans une racine finale. Le parent attend explicitement la
fin du worker, vérifie son code retour et ferme tous les FDs hérités avant de
préparer ou lancer le contrôleur.

## 9. Tests minimaux

- normalisation CRM différentielle sur accents, codes numériques, communes
  dans le nom, numéros de voie et suffixes ;
- ranking sparse différentiel : zéro score, égalités, plus de 500 valeurs,
  ordre CSR et float non fini ;
- rescues adresse et numérique ;
- RRF, tie-break intermédiaire, padding et tri final ;
- pool aligné et cache strict inchangés ;
- lookup absent, non actif, doublon et appel à 101 ;
- requête sans candidat matérialisée dans `query_status` ;
- refus de 101 candidats, rang troué, rang dupliqué et SIRET dupliqué ;
- schémas Arrow exacts et absence de metadata ;
- worker incapable d'ouvrir oracle, historique, modèle ou réseau ;
- contrôleur incapable d'ouvrir l'oracle ;
- worker terminé avant parité ;
- mutation, substitution/restauration, symlink et TOCTOU ;
- publication pending/final, reprise APFS et immutabilité ;
- parité exacte sur fixtures historiques indépendantes ;
- mini-publication limitée à des fixtures synthétiques, sans dev réel, store
  réel ou historique brut ;
- suite complète du dépôt verte.

## 10. Portée du prochain geste

Seul `GO_V412_UNIT_RETRIEVAL_PARITY` autorisera le contrat d'un évaluateur
oracle séparé et la mesure de latence appariée.

Le résultat Recall déjà publié par les benchmarks historique/V2/V3 reste la
référence retrieval officielle tant que ce futur évaluateur n'a pas exécuté
sa procédure. Le présent jalon ne republie pas une métrique Recall isolée et
ne remplace pas la règle exigeant de publier historique, V2 et V3 ensemble.

Le ranker, le decider, le risk model, l'accepteur et le test final restent
gelés.
