# V4.12 — Contrat de l'évaluateur oracle séparé du retrieval unitaire

## 1. Objet, autorisation et limite

Le verdict `GO_V412_UNIT_RETRIEVAL_PARITY` autorise une seule nouvelle
opération : évaluer les candidats scellés du worker V4.12 en les joignant à
l'oracle V4.12 déjà scellé.

L'évaluateur mesure le retrieval. Il ne modifie ni les requêtes, ni l'oracle,
ni les candidats et ne choisit aucun seuil. Il n'importe aucun code de
matching, n'entraîne aucun modèle et ne relance pas le worker.

Les identités jointes sont exactement :

```text
worker_build_id =
d2915fe7747b9b219e7a0dce400052847c913417bd240c7d94df6fb8bafedd1a

oracle_build_id =
c4045da8ad1e0b9af35f3d7552176dec76ee2ba36fa759ee2dc0664c93d2fa70

parity_build_id =
d587937b60f3b4c8f43003bc384d4e61b5933999cba7cf557debebfef02205f5
```

Cette mesure porte sur le dev historique. Elle n'est ni une mesure
indépendante, ni une certification de production, ni une ouverture du test
final. Ranker, decider, risk model, accepteur et tous les modèles restent
fermés.

## 2. Entrées gelées

### 2.1 Runtime worker

Racine unique autorisée :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/runs/v4_12_unit_retrieval/
d2915fe7747b9b219e7a0dce400052847c913417bd240c7d94df6fb8bafedd1a
```

| Rôle | Fichier | SHA-256 |
|---|---|---|
| candidats | `candidates_top100.parquet` | `7332453ffc7281972678d7e7c29973d5eeb55ce7dd3dbcaa009b416d59d4d783` |
| statuts | `query_status.parquet` | `38e6ef0cc56ffe8f10d3fc63c062dbe32bc0da90e7ebb2d7dfad9a44ccfd2f82` |
| intégrité | `integrity.json` | `6aa4e1bb0f6e14578dd1ba65707bbfdc6353abbb24368dcf7f9ef59c009a42c0` |
| manifeste | `manifest.json` | `c660d5b55c1812323790b8031c0fc7c8451d42136dead2bce30b7eeb892f236f` |

Le manifeste doit republier le même `worker_build_id`, le verdict
`SEALED_V412_UNIT_RETRIEVAL`, 1 456 statuts et 145 236 candidats. Le statut
et les candidats doivent respecter leurs schémas déjà scellés. Toute requête
a au plus 100 candidats ; `100` est un plafond absolu, jamais une moyenne.

L'évaluateur vérifie également, sans ouvrir le ledger worker :

- le manifeste d'audit worker, SHA-256
  `870a4db2d9dec394595dc3fc794e9fcdfbbfa52847a9e4df9c0ec14916b8911a` ;
- le manifeste de parité, SHA-256
  `1234d5ed1913a52fed7d5dd9f9867fb69bae1227864dce7f6212bb067292ebef` ;
- `parity.json`, SHA-256
  `f60d8eae2ee61c9e7853cc51654a6ec941009c7218dc0fd766b3b56a9f55808f` ;
- la provenance de parité, SHA-256
  `448dfc70326cfdcba5d0c3ee4eb56fed38b28468134fcacfbdc98f71093bc343`.

La parité doit conclure exactement `GO_V412_UNIT_RETRIEVAL_PARITY`.

### 2.2 Oracle

Racine unique autorisée :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/oracles/v4_12_unit_engine/
c4045da8ad1e0b9af35f3d7552176dec76ee2ba36fa759ee2dc0664c93d2fa70
```

| Rôle | Fichier | SHA-256 |
|---|---|---|
| vérités | `oracle_dev.parquet` | `e0f8c66756aec95e9f54cfe180b59609869a927985f51853b676f93bbe409d6d` |
| intégrité | `integrity.json` | `54b1a8c624e06e03d306a26ec2ad2532b5c1490160731f180810e2907a8ad497` |
| manifeste | `manifest.json` | `e201a407e968566e5f75c2072c0d441581c9a6d1314698804fb916bd3c36fd25` |

Le manifeste d'audit oracle est ancré par le SHA-256
`a10fd6e4054a8d6e067c75e7f4c853c3ec795c0715ca705b9692992667a705b7`.
L'évaluateur n'ouvre ni les labels ni le split historiques ayant servi à
construire l'oracle.

Les cardinalités exactes sont :

| Population | Total | `MATCH_EXACT` | `AMBIGUOUS` |
|---|---:|---:|---:|
| `threshold_dev` | 710 | 583 | 127 |
| `comparison_dev` | 746 | 634 | 112 |
| global | 1 456 | 1 217 | 239 |

Le hash logique des vérités reste
`9b25bf45b7ff074a1db8e6ca166569eec9754e34b29f032aebae63925d71ff42`.

## 3. Jointure et définition des résultats

La jointure est un `LEFT JOIN` exact de chaque ligne oracle vers :

1. `query_status.parquet` sur la chaîne `query_id` ;
2. `candidates_top100.parquet` sur la chaîne `query_id` ;
3. pour les seuls `MATCH_EXACT`, égalité byte-for-byte entre
   `candidate_siret` et `ground_truth_siret`.

Aucune normalisation, conversion numérique, correction, alias, agrégation
SIREN ou rapprochement approximatif n'est autorisé.

Avant toute métrique :

- les 1 456 IDs oracle et statut doivent être uniques et égaux ;
- aucune ligne extra ou manquante n'est tolérée ;
- le nombre de candidats observé doit égaler `candidate_count` ;
- les rangs doivent être contigus de 1 à `candidate_count` ;
- les SIRET candidats doivent être uniques par requête et comporter
  14 chiffres ;
- `candidate_count <= 100` doit être vrai pour chaque requête ;
- le pool vide reste matérialisé par son statut ;
- l'oracle doit reproduire exactement les populations gelées.

Pour un `MATCH_EXACT`, `exact_rank` est le rang du candidat dont le SIRET est
exactement la vérité. S'il n'existe pas, `exact_rank` est nul et la requête
est une erreur à tous les rangs. Une vérité absente du pool est donc une
erreur end-to-end, sans exception ni réinjection.

Les 239 `AMBIGUOUS` sont conservés dans les volumes, la couverture et la
sortie ligne à ligne. Ils sont exclus de tous les dénominateurs Recall et
n'obtiennent jamais artificiellement un hit.

## 4. Métriques préenregistrées

Les populations publiées sont obligatoirement, dans cet ordre :

```text
global
threshold_dev
comparison_dev
```

`threshold_dev` et `comparison_dev` sont des partitions historiques
préexistantes, pas des seuils choisis par l'évaluateur.

La seule politique métrique autorisée est l'objet `evaluation_spec` du plan.
Son keyset exact, sans clé absente ou supplémentaire, est :

```text
schema_version
population_order
population_counts
reference_order
frozen_references
join
recall_k
coverage_definition
recall_definition
proportion_output_keyset
confidence_interval
gate
latency
tuning
```

La projection est reconstruite exclusivement par
`{key: plan["evaluation_spec"][key] for key in evaluation_spec_keys}`, dans
l'ordre de la liste `identity_projections.evaluation_spec_keys`. Son keyset
doit être strictement égal au keyset ci-dessus et sa valeur doit être
strictement égale à l'objet complet du plan. Elle est ensuite encodée en JSON
canonique UTF-8, clés triées, séparateurs compacts, `allow_nan=false`, LF
final. Son SHA-256 devient `evaluation_spec_sha256`.

Il est interdit au code, au verrou ou au runner de redéfinir, compléter ou
remplacer un champ de cette projection. Le verrou externe épingle
`evaluation_spec_sha256`; une divergence est un `STOP` avant ouverture de
l'oracle.

Pour chaque population :

- couverture identifiable :
  `MATCH_EXACT / (MATCH_EXACT + AMBIGUOUS)` ;
- `Recall@1`, `Recall@10`, `Recall@50`, `Recall@100` SIRET exact :
  `count(MATCH_EXACT avec exact_rank <= K) / count(MATCH_EXACT)`.

Chaque proportion publie :

```text
success_count
denominator_count
rate
wilson_95_low
wilson_95_high
wilson_99_low
wilson_99_high
```

Les nombres bruts ont autorité. Les taux et bornes sont calculés en double
précision, sans arrondi avant le calcul. Les constantes normales sont :

```text
z95 = 1.959963984540054
z99 = 2.5758293035489004
```

Pour `x` succès sur `n > 0`, avec `p=x/n`, les bornes de Wilson sont :

```text
centre = (p + z²/(2n)) / (1 + z²/n)
rayon = z * sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)
borne_basse = max(0, centre - rayon)
borne_haute = min(1, centre + rayon)
```

Un dénominateur nul est un `STOP`, jamais un `NaN`.

## 5. Références gelées à republier ensemble

Le rapport ne doit jamais montrer la seule mesure V4.12. Il publie dans la
même table les trois références du build V3 gelé
`ab8343817551c0a5`, issues de
`docs/retrieval_selective_recall100_contract.md` :

| Référence gelée | Couverture brute | Couverture | Recall@100 brut | Recall@100 |
|---|---:|---:|---:|---:|
| historique, toutes requêtes | 2 565 / 2 565 | 100 % | 2 495 / 2 565 | 97,271 % |
| V2 exact | 2 400 / 2 565 | 93,567 % | 2 343 / 2 400 | 97,625 % |
| V3 exact identifiable | 2 104 / 2 565 | 82,027 % | 2 095 / 2 104 | 99,572 % |

L'évaluateur recalcule leurs taux et intervalles de Wilson depuis ces nombres
bruts, sans ouvrir les artefacts historiques.

La mesure `V4.12 unit oracle` est ajoutée séparément avec 1 456 requêtes et
1 217 `MATCH_EXACT`. Les références gelées portent sur 2 565 requêtes et ne
constituent pas le même échantillon. Aucun delta, test de supériorité ou
revendication d'amélioration appariée n'est permis. Le rapport distingue
explicitement :

- `FROZEN_REFERENCE` pour historique, V2 et V3 ;
- `V412_MEASUREMENT` pour global, `threshold_dev` et `comparison_dev`.

## 6. Latence descriptive

La seule source de temps autorisée est `durations_ns` du fichier
`integrity.json` worker scellé :

| Champ | Nanosecondes |
|---|---:|
| `retrieval` | 620 340 646 980 |
| `lookup` | 398 384 905 282 |
| `serialization` | 117 570 208 |
| `total` | 1 030 157 672 584 |

L'évaluateur republie ces nombres et leur conversion descriptive en secondes.
Il peut publier `total / 1 456` sous le nom explicite
`mean_wall_seconds_per_query_from_aggregate`.

Il n'existe aucun temps par requête. Il est donc interdit de publier ou
d'inférer p50, p90, p95, p99, variance, distribution de latence ou SLA. Le
rapport porte obligatoirement :

```text
latency_source = WORKER_INTEGRITY_AGGREGATE
per_query_timing_available = false
p95_available = false
latency_gate_evaluated = false
```

## 7. Gates et verdict

Les deux gates produit principaux sont évalués sur la population globale :

```text
gate_statistic = OBSERVED_RATE_FROM_RAW_COUNTS
couverture identifiable >= 0.80
Recall@100 SIRET exact >= 0.99
```

`OBSERVED_RATE_FROM_RAW_COUNTS` signifie que chaque gate compare exactement
`success_count / denominator_count` au seuil correspondant. Les bornes de
Wilson sont publiées mais ne sont pas la statistique du gate. Il est interdit
d'utiliser une borne basse, une valeur arrondie, une estimation calibrée ou
une autre statistique pour décider `GO` ou `PIVOT`.

Les valeurs `threshold_dev` et `comparison_dev` sont publiées intégralement
comme contrôles de stabilité, mais ne changent pas les seuils et ne servent à
aucun tuning.

Verdicts exclusifs :

- `GO_V412_UNIT_RETRIEVAL_EVALUATION` si toutes les validations techniques
  passent et les deux gates globaux passent ;
- `PIVOT_V412_UNIT_RETRIEVAL_EVALUATION` si la mesure est valide mais au
  moins un gate global échoue ;
- `STOP_V412_UNIT_RETRIEVAL_EVALUATION` si un hash, une identité, un schéma,
  une cardinalité, une jointure, le plafond 100, une règle de calcul ou la
  sécurité diverge.

Un échec technique ne devient jamais un zéro métrique. Un `PIVOT` ne peut
jamais être transformé en `GO` par changement de seuil ou seconde variante.

## 8. Sorties, identité et preuve

Racine de l'évaluation :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/evaluations/
v4_12_unit_retrieval/<evaluator_build_id>
```

Fichiers exacts :

- `query_outcomes.parquet` ;
- `metrics.json` ;
- `integrity.json` ;
- `manifest.json`.

`query_outcomes.parquet` contient exactement :

| Colonne | Type | Nullable |
|---|---|---|
| `query_id` | string | non |
| `dev_partition` | string | non |
| `label_kind` | string | non |
| `candidate_count` | uint8 | non |
| `exact_rank` | uint8 | oui |
| `hit_at_1` | bool | oui |
| `hit_at_10` | bool | oui |
| `hit_at_50` | bool | oui |
| `hit_at_100` | bool | oui |

Les champs `exact_rank` et `hit_at_*` sont nuls pour `AMBIGUOUS`. Aucune
vérité SIRET, aucun score et aucune feature ne sont recopiés dans la sortie.

L'ordre physique des 1 456 lignes est exactement l'ordre physique de
`oracle_dev.parquet`; aucun tri par `query_id`, partition, rang ou hit n'est
autorisé.

Le payload logique de `query_outcomes.parquet` est la concaténation sans
header ni BOM des lignes encodées ainsi :

```text
UTF8(query_id)                    0x00
ASCII(dev_partition)             0x00
ASCII(label_kind)                0x00
ASCII_DECIMAL(candidate_count)   0x00
ASCII_DECIMAL(exact_rank) ou \N  0x00
BOOL(hit_at_1) ou \N             0x00
BOOL(hit_at_10) ou \N            0x00
BOOL(hit_at_50) ou \N            0x00
BOOL(hit_at_100) ou \N           0x0A
```

`ASCII_DECIMAL` est la représentation base 10 sans signe et sans zéro de
remplissage. `BOOL(true)` est l'octet ASCII `1`, `BOOL(false)` l'octet ASCII
`0`. Le token nul est exactement les deux octets ASCII `0x5c 0x4e`
représentant `\N`. Il n'y a aucun séparateur `0x00` après la dernière
colonne ; la ligne se termine directement par un unique LF `0x0A`.

Pour un `MATCH_EXACT` absent du pool, `exact_rank=\N` et les quatre hits
valent `0`. Pour un `AMBIGUOUS`, `exact_rank` et les quatre hits valent
`\N`. Les champs string doivent déjà satisfaire les interdictions NUL/LF du
contrat oracle.

`query_outcomes_payload_bytes` est le nombre exact d'octets de cette
concaténation et `query_outcomes_payload_sha256` son SHA-256. La formule
exacte de `missing_truth_count` est :

```text
COUNT(label_kind == "MATCH_EXACT" AND exact_rank IS NULL)
```

Comme le plafond est 100, elle doit aussi être strictement égale à :

```text
1217 - v412_measurements["global"].recall_at["100"].success_count
```

Toute divergence entre ces deux calculs est un `STOP`.

### 8.1 Keysets JSON exacts

Les schema versions sont exactement :

```text
query_outcomes = sireto-v4.12-unit-retrieval-evaluator-outcomes-1
metrics = sireto-v4.12-unit-retrieval-evaluator-metrics-1
integrity = sireto-v4.12-unit-retrieval-evaluator-integrity-1
evaluation_manifest = sireto-v4.12-unit-retrieval-evaluator-manifest-1
ledger = sireto-v4.12-unit-retrieval-evaluator-ledger-1
provenance = sireto-v4.12-unit-retrieval-evaluator-provenance-1
audit_manifest = sireto-v4.12-unit-retrieval-evaluator-audit-manifest-1
```

`metrics.json` possède exactement :

```text
schema_version
evaluator_build_id
attempt_id
worker_build_id
oracle_build_id
parity_build_id
population_order
reference_order
v412_measurements
frozen_references
latency
gates
verdict
declarations
```

`population_order` vaut exactement
`["global","threshold_dev","comparison_dev"]`.
`v412_measurements` est un tableau de trois records dans cet ordre. Chaque
record possède exactement :

```text
population
measurement_type
total_query_count
match_exact_count
ambiguous_count
coverage
recall_at
```

`measurement_type` vaut `V412_MEASUREMENT`. `recall_at` est un objet dont les
clés exactes sont, dans l'ordre logique publié, `"1","10","50","100"`.
`coverage` et chaque valeur de `recall_at` utilisent le record proportion
exact :

```text
success_count
denominator_count
rate
wilson_95_low
wilson_95_high
wilson_99_low
wilson_99_high
```

`reference_order` vaut exactement
`["historical_all","v2_exact","v3_exact_identifiable"]`.
`frozen_references` est un tableau de trois records dans cet ordre, chacun
avec exactement :

```text
name
reference_type
source_build_id
source_population_count
coverage
recall_at_100
```

`reference_type` vaut `FROZEN_REFERENCE`, `source_build_id` vaut
`ab8343817551c0a5`, et les deux proportions utilisent le même record exact
ci-dessus.

`latency` possède exactement :

```text
latency_source
durations_ns
durations_seconds
query_count
mean_wall_seconds_per_query_from_aggregate
per_query_timing_available
p95_available
latency_gate_evaluated
```

Les objets `durations_ns` et `durations_seconds` possèdent exactement les
clés `retrieval,lookup,serialization,total`.

`gates` possède exactement :

```text
gate_statistic
population
coverage_minimum
recall_at_100_minimum
coverage_observed
recall_at_100_observed
coverage_pass
recall_at_100_pass
all_pass
```

`gate_statistic` vaut exactement `OBSERVED_RATE_FROM_RAW_COUNTS`.

`integrity.json` possède exactement :

```text
schema_version
evaluator_build_id
attempt_id
worker_build_id
oracle_build_id
parity_build_id
query_count
match_exact_count
ambiguous_count
candidate_count
minimum_pool_size
maximum_pool_size
under_ceiling_query_count
empty_query_count
missing_truth_count
query_outcomes_payload_bytes
query_outcomes_payload_sha256
metrics_sha256
input_snapshot_count
opened_input_count
gate_statistic
declarations
verdict
```

Le manifeste d'évaluation possède exactement :

```text
schema_version
evaluator_build_id
attempt_id
worker_build_id
oracle_build_id
parity_build_id
files
runtime
declarations
verdict
```

Son objet `files` contient exactement les records
`query_outcomes.parquet`, `metrics.json` et `integrity.json`.
Le record Parquet possède exactement
`path,size_bytes,sha256,row_count,schema,metadata`, avec `metadata=null`.
Chaque record JSON possède exactement `path,size_bytes,sha256`.

Dans `metrics.json`, `integrity.json`, le manifeste d'évaluation et la
provenance, `declarations` possède exactement :

```text
historical_development_only
independent_measurement
production_certified
models_opened
historical_sources_opened
challenge_or_final_opened
tuning_performed
```

Les valeurs exactes sont respectivement
`true,false,false,false,false,false,false`.

Racine de preuve :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/
v4_12_unit_retrieval_evaluator/<evaluator_build_id>
```

Fichiers exacts :

- `open_ledger.parquet` ;
- `provenance.json` ;
- `manifest.json`.

`provenance.json` possède exactement :

```text
schema_version
evaluator_build_id
attempt_id
git_commit
source_hashes
plan_sha256
lock_sha256
input_hashes
evaluation_spec_sha256
runtime
data_input_count
evaluation_manifest_sha256
declarations
```

Le manifeste d'audit possède exactement
`schema_version,evaluator_build_id,attempt_id,files`. `files` contient
exactement `open_ledger.parquet` et `provenance.json`; chaque record possède
exactement `path,size_bytes,sha256`.

Le ledger possède exactement sept colonnes non nullables, dans cet ordre :

```text
role: string
absolute_path: string
projection: string
size_bytes_before: uint64
sha256_before: string
size_bytes_after: uint64
sha256_after: string
```

Il possède exactement douze rôles, dans l'ordre UTF-8 suivant :

```text
oracle_audit_manifest
oracle_dev
oracle_integrity
oracle_manifest
parity_manifest
parity_provenance
parity_result
worker_audit_manifest
worker_candidates_top100
worker_integrity
worker_manifest
worker_query_status
```

Les projections Parquet sont limitées aux colonnes contractuelles utilisées.
La projection des JSON est `FULL_JSON_EXACT_KEYSET`. Taille et SHA-256
avant/après doivent être égaux au verrou. Aucun exécutable, source Git ou
fichier non déclaré n'est ajouté au ledger de données ; ils sont ancrés par
le verrou et la provenance.

Le `evaluator_build_id` est le SHA-256 du JSON canonique ayant exactement :

```text
schema_version
plan_sha256
lock_sha256
source_hashes
input_hashes
worker_build_id
oracle_build_id
parity_build_id
evaluation_spec
runtime
```

Dans ce payload, `schema_version` vaut exactement :

```text
sireto-v4.12-unit-retrieval-evaluator-build-identity-1
```

Le keyset et cette valeur sont aussi épinglés dans
`identity_projections.build_identity_keys` et
`identity_projections.build_identity_schema_version` du plan. Le payload ne
contient aucune autre clé et `evaluation_spec` est exactement la projection
canonique décrite en section 4, pas une copie reconstruite autrement.

Les JSON utilisent UTF-8, `sort_keys=true`, séparateurs compacts,
`allow_nan=false` et un LF final. Audit final avant publication finale,
permissions immuables et renommage atomique sur le même filesystem.

## 9. Verrou externe, sandbox et lectures

Le verrou d'exécution n'est créé qu'après commit et audit du futur code. Il
épingle :

- le commit Git complet et tous les blobs source ;
- le présent contrat et le plan ;
- tous les chemins et hashes d'entrée déjà préenregistrés ;
- le profil sandbox et les exécutables ;
- le runtime scientifique ;
- les racines staging, évaluation et audit ;
- un plafond RSS de 8 Gio.

Le processus evaluator s'exécute avec `python -B`, réseau et fork interdits.
Il reçoit une allowlist de fichiers exacts, jamais une racine sensible
générale. Sont interdits :

- `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets` ;
- `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models` ;
- `/Users/nathanjullia/Documents/Projets/SIRETO/models` ;
- `/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges` ;
- `/Volumes/CATNAT_DATA/SIRETO_RECALL100/final` ;
- tout oracle autre que les trois fichiers exacts du build `c4045...` ;
- tout audit autre que les manifests/parité exacts préenregistrés ;
- toute sortie de ranker, decider, risk model ou accepteur.

Chaque entrée est un fichier régulier sans symlink, résolu composant par
composant, ouvert avec `openat` et `O_NOFOLLOW`, puis consommé depuis le même
FD. `fstat`, taille et SHA-256 sont contrôlés avant et après. Toute mutation,
substitution, restauration ou lecture non prévue est un `STOP`.

L'évaluateur écrit seulement dans un staging privé sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/tmp/v4_12_unit_retrieval_evaluator`.
Le parent valide puis publie ; le processus sandboxé n'écrit jamais dans une
racine finale.

## 10. Reçu durable et tentative unique

Deux identités sont calculées avant toute ouverture oracle.

Les schema versions sont exactement :

```text
measurement_slot_identity =
sireto-v4.12-unit-retrieval-evaluator-measurement-slot-1
attempt_identity =
sireto-v4.12-unit-retrieval-evaluator-attempt-identity-1
receipt = sireto-v4.12-unit-retrieval-evaluator-receipt-1
state = sireto-v4.12-unit-retrieval-evaluator-attempt-state-1
event = sireto-v4.12-unit-retrieval-evaluator-attempt-event-1
```

`measurement_slot_id` est le SHA-256 du JSON canonique possédant exactement :

```text
schema_version
purpose
worker_build_id
oracle_build_id
parity_build_id
```

Il réserve de manière permanente la mesure pour le triplet worker/oracle/
parité. `attempt_id` est le SHA-256 du JSON canonique possédant exactement :

```text
schema_version
plan_sha256
lock_sha256
input_hashes
evaluation_spec_sha256
worker_build_id
oracle_build_id
parity_build_id
```

Racine parent-only :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/
v4_12_unit_retrieval_evaluator_attempts/<measurement_slot_id>/
```

Arborescence exacte :

```text
receipt.json
state.json
events.jsonl
```

`receipt.json` est immuable et possède exactement :

```text
schema_version
measurement_slot_id
attempt_id
created_at_utc
plan_sha256
lock_sha256
input_hashes
evaluation_spec_sha256
worker_build_id
oracle_build_id
parity_build_id
policy_immutable
```

`policy_immutable` vaut `true`. Le parent crée la racine sans clobber, crée
`receipt.json` avec exclusivité, écrit l'événement initial et `state.json`,
appelle `fsync` sur chaque fichier puis sur chaque répertoire ancêtre jusqu'à
la racine d'attempt. Cette séquence est entièrement terminée avant que
l'oracle ne puisse être ouvert ou qu'un FD oracle ne soit transmis.

Après le reçu, et immédiatement avant `ORACLE_OPEN_COMMITTED`, le parent
revalide depuis zéro :

1. les octets canoniques, hashes et snapshots du plan et du verrou externe ;
2. le schéma/keyset du verrou, son commit Git, ses blobs source, son
   `evaluation_spec_sha256`, son runtime et ses racines ;
3. exactement les douze rôles non-oracle suivants contre les chemins et
   hashes du verrou :

```text
git_executable
parity_manifest
parity_provenance
parity_result
python_framework_app
python_framework_library
sandbox_executable
worker_audit_manifest
worker_candidates_top100
worker_integrity
worker_manifest
worker_query_status
```

Chaque fichier est régulier, sans symlink, ouvert par `openat/O_NOFOLLOW`,
`fstat` et hashé depuis le même FD. Plan, verrou et les douze snapshots sont
contrôlés une deuxième fois, inchangés, avant la transition.

Seulement si tous ces contrôles passent, le parent append l'événement
`ORACLE_OPEN_COMMITTED`, appelle `fsync` sur `events.jsonl` et son répertoire,
puis reconstruit et `fsync` le cache `state.json`. Seulement après cette
durabilité, les quatre rôles oracle sont ouverts dans cet ordre :

```text
oracle_manifest
oracle_integrity
oracle_audit_manifest
oracle_dev
```

Ils sont ouverts sans symlink, `fstat` et hashés contre le verrou, consommés
depuis ces mêmes quatre FDs, puis `fstat` et rehashés depuis ces mêmes FDs.
Aucun fichier oracle ne peut servir à la revalidation pré-commit.

La même séquence s'applique à toute reprise qui atteint pour la première fois
la frontière oracle. Si la chaîne contient déjà
`ORACLE_OPEN_COMMITTED=true`, aucune réouverture ou recomputation oracle
n'est autorisée : la reprise est exclusivement une validation/promotion
d'octets complets déjà produits.

Si le slot existe déjà, son reçu doit épingler exactement le même
`attempt_id`, plan, verrou, inputs et spécification. Une identité différente
est un `STOP`; créer un autre slot, effacer le reçu ou modifier le verrou
pour contourner ce contrôle est interdit.

`state.json` possède exactement :

```text
schema_version
measurement_slot_id
attempt_id
sequence
state
phase
oracle_open_committed
evaluator_build_id
reason_code
updated_at_utc
```

Chaque ligne JSON canonique de `events.jsonl` possède exactement :

```text
schema_version
measurement_slot_id
attempt_id
sequence
state
phase
oracle_open_committed
evaluator_build_id
reason_code
timestamp_utc
previous_event_sha256
```

Les séquences commencent à zéro, sont contiguës et chaque événement chaîne
le SHA-256 des octets de la ligne précédente, LF inclus ; la première valeur
`previous_event_sha256` est nulle. Un nouvel événement est appendé et
`fsync` avant remplacement atomique et `fsync` de `state.json`. Le processus
sandboxé n'a aucun droit d'écriture sur cette racine : seul le parent tient
le reçu et le journal.

`events.jsonl` est l'unique autorité de l'état. `state.json` est seulement le
cache canonique dérivé du dernier événement valide : il ne peut autoriser ni
annuler une transition.

À chaque démarrage ou recovery, le parent :

1. valide le reçu immuable ;
2. lit toute la chaîne d'événements jusqu'au LF final ;
3. vérifie keyset, schema version, identité, séquences contiguës, hash
   précédent, transitions permises et monotonie de
   `oracle_open_committed` ;
4. dérive l'état depuis le dernier événement ;
5. compare `state.json` à cette projection.

Si `state.json` est absent, il est reconstruit depuis la chaîne valide par
écriture d'un fichier temporaire exclusif, `fsync`, renommage atomique sans
suivre de symlink vers `state.json`, puis `fsync` du répertoire. S'il est
périmé, il n'est
reconstruit que si son identité et son contenu correspondent exactement à la
projection d'un événement antérieur de la même chaîne. Dans les deux cas,
`events.jsonl` n'est jamais réécrit.

Un cache en avance sur la chaîne, une identité divergente, une ligne partielle
ou sans LF, un hash ou une transition invalide, ou un cache affirmant
`oracle_open_committed=true` sans événement correspondant est un `STOP`.
Cette règle interdit qu'un événement `ORACLE_OPEN_COMMITTED` durable soit
perdu, masqué ou ramené à `false`. Dès sa première valeur `true`, tous les
événements suivants doivent conserver `true`.

Avant de rendre l'oracle accessible, le parent append et `fsync` l'événement
de phase `ORACLE_OPEN_COMMITTED` avec `oracle_open_committed=true`. Dès cet
instant, le protocole considère que l'oracle a pu être ouvert, même si le
processus s'arrête immédiatement.

Les seuls états sont :

| État | Définition exacte |
|---|---|
| `STARTED` | reçu durable ; exécution pas encore publiable. Avant `ORACLE_OPEN_COMMITTED`, la même tentative peut reprendre après revalidation intégrale. Après ce commit, un crash n'est pas rejouable. |
| `RECOVERABLE` | les deux arbres staging ou pending sont complets, liés au même build et validés, ou l'audit final est valide avec l'évaluation pending valide. Seules validation et promotion sont permises ; aucun calcul, aucune réouverture oracle. |
| `FINAL` | audit final puis évaluation finale existent, sont immuables, liés au même attempt/build et passent la postvalidation commune. |
| `STOPPED` | état terminal après divergence, conflit, violation, ou crash post-ouverture sans deux arbres complets et valides. Aucun rerun ni effacement. |

Les seules phases sont `RECEIPT_DURABLE`, `ORACLE_OPEN_COMMITTED`,
`COMPUTED_STAGING_VALID`, `PENDING_BOTH_VALID`, `AUDIT_FINAL`,
`EVALUATION_FINAL`, `TERMINAL`. `evaluator_build_id` et `reason_code` sont
nuls tant qu'ils ne sont pas définis par la phase. Toute autre combinaison
état/phase est un `STOP`.

Un verdict métrique `GO` ou `PIVOT` peut finir avec l'état protocolaire
`FINAL`. Le verdict métrique et l'état de publication ne sont jamais
confondus.

## 11. Machine d'états de publication

Les deux arbres sont construits sous :

```text
<temp_root>/<attempt_id>/evaluation.stage
<temp_root>/<attempt_id>/audit.stage
```

Puis promus vers :

```text
<evaluation_root>/.pending-<evaluator_build_id>-<attempt_id>
<audit_root>/.pending-<evaluator_build_id>-<attempt_id>
```

et enfin :

```text
<evaluation_root>/<evaluator_build_id>
<audit_root>/<evaluator_build_id>
```

Ordre exact :

1. valider complètement les deux arbres staging ;
2. journaliser `COMPUTED_STAGING_VALID` ;
3. promouvoir l'audit pending, `fsync` ;
4. promouvoir l'évaluation pending, `fsync` ;
5. valider les deux pending et journaliser `PENDING_BOTH_VALID` ;
6. promouvoir l'audit final, le rendre immuable, le revalider et journaliser
   `AUDIT_FINAL` ;
7. promouvoir l'évaluation finale, la rendre immuable, revalider les deux
   racines et journaliser `EVALUATION_FINAL` ;
8. journaliser `FINAL/TERMINAL`.

Chaque promotion est un renommage atomique sur le même filesystem, avec
création exclusive de la destination. Aucun fichier ou répertoire existant
n'est remplacé, fusionné, nettoyé ou supprimé. Une destination existante
n'est acceptée que si l'arbre complet est byte-for-byte celui attendu pour
le même `attempt_id` et le même `evaluator_build_id`.

Règles de crash et de reprise :

| État disque observé | Décision |
|---|---|
| reçu durable, oracle non committé | reprendre la même tentative ; revalider plan, verrou et les 12 rôles non-oracle, puis append+fsync du commit avant les 4 ouvertures oracle |
| oracle committé, arbres incomplets ou invalides | `STOPPED`, conservation intégrale, aucun recalcul |
| deux staging ou deux pending complets et valides | `RECOVERABLE`, promotion seulement |
| audit final valide, évaluation pending valide | `RECOVERABLE`, promotion de cette évaluation seulement |
| audit final valide, aucune évaluation complète | `STOPPED` sauf si une évaluation staging/pending complète et déjà validée existe |
| évaluation finale présente sans audit final | `STOPPED`, évaluation orpheline conservée et déclarée invalide ; aucune publication réparatrice |
| deux finaux présents mais ordre ou chaîne d'événements invalide | `STOPPED`, aucune réécriture |
| deux finaux valides et chaîne valide | `FINAL` idempotent |

Une évaluation finale sans audit final ne peut donc jamais être considérée
comme `RECOVERABLE` par recalcul. La seule reprise admise après ouverture
oracle promeut des octets complets déjà validés. Tout état non listé est
`STOPPED`.

## 12. Aucun tuning et ordre d'autorisation

Ordre obligatoire :

1. contre-auditer le présent contrat et le plan ;
2. committer ces deux fichiers dans un commit isolé ;
3. implémenter evaluator, runner, profil sandbox et tests synthétiques ;
4. contre-auditer le code, puis le committer ;
5. construire et contre-auditer le verrou externe ;
6. exécuter une fois l'évaluation officielle ;
7. publier les artefacts, puis effectuer un audit indépendant ;
8. publier les métriques et conclure `GO`, `PIVOT` ou `STOP`.

Il est interdit entre les étapes 1 et 8 de modifier :

- populations, dénominateurs, rangs K ou formules ;
- seuils 80 % et 99 % ;
- références gelées ;
- inputs, hashes, candidats ou oracle ;
- règles de verdict.

Avant création du reçu seulement, un défaut de code ou de verrou peut être
corrigé, documenté, recommitté et reverrouillé. Dès que le reçu existe, plan,
verrou, inputs et `evaluation_spec` sont immuables. Avant
`ORACLE_OPEN_COMMITTED`, seule la reprise du même `attempt_id` est autorisée.
Après ce commit, aucun recalcul ni rerun n'est autorisé, que des métriques
aient été publiées ou non.

Le test final reste fermé pendant tout ce jalon. Ce dev historique sert à
décider si l'architecture retrieval mérite un `GO`, un `PIVOT` ou un `STOP`;
il ne remplace pas l'évaluation finale unique exigée par la directive active.

## 13. Tests minimaux du futur code

- refus de tout build ID, hash, schéma ou cardinalité divergent ;
- refus d'un ID oracle/statut manquant, extra ou dupliqué ;
- refus de rangs troués, candidat dupliqué, SIRET invalide ou 101 candidats ;
- vérité absente comptée comme miss à 1/10/50/100 ;
- `AMBIGUOUS` conservé mais exclu du Recall ;
- exactitude des agrégats global, `threshold_dev`, `comparison_dev` ;
- payload logique outcomes byte-for-byte, ordre oracle et formule des miss ;
- Wilson 95/99 vérifié sur cas limites `0/n`, `n/n` et cas intermédiaires ;
- références historique/V2/V3 toujours publiées ensemble ;
- aucune prétention p95 à partir des durées agrégées ;
- sandbox refusant historique, modèles, challenge, final et oracle non prévu ;
- mutation, TOCTOU, symlink, write-scope, RSS, publication et reprise ;
- ledger exhaustif et manifests à keysets exacts ;
- reçu et événement `fsync` avant sentinelle réelle d'ouverture oracle ;
- aucun des quatre rôles oracle ouvert avant l'événement durable ;
- recovery pré-oracle répétant les douze revalidations non-oracle ;
- `state.json` absent/périmé reconstruit depuis la chaîne autoritaire valide ;
- cache en avance, chaîne partielle et régression du commit oracle refusés ;
- tentative concurrente, autre verrou et autre politique refusés par le slot ;
- transitions `STARTED/RECOVERABLE/FINAL/STOPPED` et crash à chaque promotion ;
- évaluation finale sans audit final conservée mais déclarée `STOPPED` ;
- aucune fixture réelle dev/oracle dans les tests ;
- aucun import ou chargement de ranker, decider, risk model ou accepteur.
