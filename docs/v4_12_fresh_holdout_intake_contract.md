# Contrat V4.12 — Intake autonome d’un holdout CRM frais

## 1. Objet, preuve visée et frontières

Ce contrat prépare le **premier export CRM complet, indépendant et soumis à
l’intake après son scellement**. « Premier » désigne l’ordre des soumissions
complètes au protocole, pas un ordre de création ou de réception impossible à
prouver. Une sémantique « premier export produit/reçu » n’est admise que si le
producteur fournit un journal monotone scellé, exhaustif et vérifiable et si
l’intake possède un receipt d’arrivée durable pour chaque export antérieur.
L’intake est autonome, sans validation utilisateur, et ne contient aucun
modèle de matching.

Trois preuves sont strictement séparées :

1. l’intake mesure la capacité à construire un holdout indépendant et à
   qualifier ses lignes ;
2. l’évaluation one-shot mesure le **Recall SIRET exact du retrieval**, avec
   au plus 100 candidats par requête ;
3. une éventuelle certification `AUTO_MATCH` à 99,8 % de précision est une
   expérience sélective distincte. Avec Clopper–Pearson exact unilatéral à
   99 %, elle exige au minimum **2 301 décisions AUTO indépendantes auditées
   sans erreur** : 2 300 donnent seulement `0,997999755`, sous 99,8 %. Une
   borne de Wilson unilatérale demanderait 2 701 cas. Le holdout retrieval ne
   certifie ni cette précision ni sa couverture.

Avant l’état `READY`, il est interdit d’ouvrir ou d’exécuter retrieval,
ranker, decider, risk model, accepteur, candidats, prédictions, hit, rang ou
score. Après `READY`, le scorer ne reçoit que les requêtes sûres. L’oracle est
ouvert par un évaluateur distinct uniquement après scellement des résultats.

## 2. Sampling frame préenregistré et absence d’optional stopping

L’unité d’échantillonnage est un enregistrement du prochain export CRM
opérationnel complet. Le `collection_source_manifest.json`, produit par le
système source avant toute ingestion, fixe :

- `export_snapshot_id` et `reference_date` ;
- `population_name` et la définition de la population ;
- `period_start_utc`, `period_end_utc` et `export_cutoff_utc` ;
- la liste exhaustive des `portfolio_id` inclus ;
- `expected_batch_count`, la liste ordonnée des `source_batch_id` et le
  nombre total attendu de lignes ;
- la règle d’inclusion `ALL_SOURCE_RECORDS_IN_FRAME` ;
- les exclusions métier éventuelles, définies par le producteur avant
  qualification, avec leur justification et leur nombre.

Le premier manifeste complet soumis et stable admissible verrouille cette frame. Tous ses
batches préannoncés sont inclus exhaustivement, dans l’ordre déclaré. Aucun
batch, portefeuille, service ou enregistrement ne peut être ajouté, retiré ou
remplacé en fonction du nombre de `MATCH_EXACT`, de la couverture, d’un résultat
modèle ou d’une borne statistique.

L’accumulation concerne uniquement les shards préenregistrés du même export.
Atteindre 657 exacts n’autorise pas un arrêt anticipé ; ne pas les atteindre
n’autorise pas à prolonger la période ou à ajouter un autre export. Les gates
ne sont calculés qu’après le cutoff, réception de tous les batches et
qualification exhaustive. Un volume insuffisant produit `PIVOT`, jamais une
collecte opportuniste.

## 3. Racines SSD et paquets d’entrée

Les racines absolues, distinctes et préenregistrées sont :

| Rôle | Racine |
|---|---|
| Inbox | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake/inbox` |
| Quarantaine | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake/quarantine` |
| Batches scellés | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake/sealed` |
| Audit intake | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake/audit` |
| Temporaire privé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake/tmp` |
| Registres | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake/registry` |
| Freeze READY | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake/ready` |
| Ledger évaluation | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_evaluation_ledger` |

Une collection contient exactement un `collection_source_manifest.json` et
les batches qu’il énumère. Chaque batch sépare deux côtés :

- côté requête : `source_manifest.json` et exactement un CRM sûr `.csv` ou
  `.parquet`, sans SIRET/SIREN ;
- côté oracle : `evidence_source_manifest.json` et zéro ou un
  `evidence_source.parquet` scellé, inaccessible au scorer.

Le paquet evidence est le mécanisme non circulaire recommandé : il peut porter
une relation contractuelle autoritative `source_record_id → SIRET`, avec sa
provenance et sa période de validité. Son absence est autorisée, mais une ligne
sans preuve autoritative indépendante restera `UNRESOLVED`. Archives,
classeurs, JSON de données, répertoires imbriqués, symlinks, FIFOs et hard
links multiples sont refusés.

Le manifeste source canonique contient au minimum :

```text
schema_version
export_snapshot_id
source_batch_id
source_system
portfolio_id
reference_date
period_start_utc
period_end_utc
export_cutoff_utc
source_filename
source_format
source_row_count
source_size_bytes
source_sha256
producer_manifest_id
producer_created_at_utc
source_record_id_semantics
v411_service_id_equivalence_attested
lineage_attestation_reference
```

`reference_date` est obligatoire et identique pour tous les batches de la
collection. Taille, hash, format, batch, période et provenance doivent être
cohérents avec le manifeste collection.

Le manifeste evidence canonique contient au minimum :

```text
schema_version
export_snapshot_id
source_batch_id
source_system
portfolio_id
reference_date
evidence_filename
evidence_row_count
evidence_size_bytes
evidence_sha256
authority_type
producer_manifest_id
producer_created_at_utc
```

Le parquet evidence a pour clé `source_record_id` et peut contenir
`authoritative_siret`, `authority_reference`, `valid_from`, `valid_to`,
`evidence_observed_at_utc`, `evidence_payload_sha256` et un code de provenance.
Il ne contient aucun hit, rang, score, candidat ou sortie SIRETO.

## 4. Formats et schéma CRM brut

Formats autorisés :

- CSV UTF-8, en-tête, virgule, quoting RFC 4180, fins de ligne LF ou CRLF ;
- Parquet plat, sans type imbriqué, partition implicite ni métadonnée
  applicative non déclarée.

Colonnes exactes, dans cet ordre :

1. `source_batch_id` ;
2. `source_record_id`, non vide et unique dans le batch ;
3. `source_system` ;
4. `portfolio_id` ;
5. `crm_name_raw` ;
6. `crm_address_raw` ;
7. `crm_postcode_raw` ;
8. `crm_city_raw` ;
9. `crm_insee_raw`.

Les quatre champs de provenance doivent correspondre au manifeste. Chaque
ligne fournit nom ou adresse, et au moins code postal, commune ou INSEE.
Toutes les lignes source restent dans le dénominateur principal, y compris
celles qui seront invalides, contaminées, ambiguës ou non résolues.

Toute colonne supplémentaire est refusée. Les tokens `siret`, `siren`,
`target`, `label`, `match`, `candidate`, `rank`, `score`, `prediction` et
`confidence` sont interdits dans les noms. Chaque valeur string est d’abord
projetée en Unicode NFKC pour le scan seulement, sans remplacer le CRM brut.
Une séquence autonome de longueur exactement 14 ou 9 dont chaque caractère
répond à `isdecimal()` — chiffres ASCII, Unicode `Nd` ou mélange des deux —
exclut la ligne des requêtes sûres et déclenche une preuve de quarantaine.
Les bornes doivent être absentes ou non décimales. Le scanner possède des
golden vectors positifs et négatifs pour ASCII, chiffres arabes/pleine chasse
et séquences mixtes. Aucun SIRET/SIREN CRM n’est transmis au retrieval.

## 5. Ouvertures, stabilité et durabilité macOS

Chaque composant de chemin est traversé depuis `/` avec `openat`,
`O_NOFOLLOW`, `O_CLOEXEC` et `O_DIRECTORY` pour les répertoires. Le terminal
est un fichier régulier avec un seul hard link. Deux observations de `device`,
`inode`, `size`, `mtime_ns` et `ctime_ns`, espacées d’au moins 60 secondes,
doivent être identiques.

Les descripteurs des racines et répertoires parents restent ouverts et ancrent
toutes les ouvertures enfants. Immédiatement après `open`, puis après la
dernière lecture, `fstat` vérifie type, UID effectif attendu, device, inode,
taille, `mtime_ns`, `ctime_ns` et nombre de liens. Le device et l’UUID du
volume sont épinglés dans le futur lock ; toute traversée ou promotion vers un
autre volume est refusée.

Le fichier est ouvert une seule fois. La lecture va explicitement jusqu’à EOF,
avec compteur d’octets ; ce compteur doit être identique à la taille `fstat`
initiale, finale et à `source_size_bytes` du manifeste. Taille et SHA-256
avant/après sont calculés sur le même descripteur. Une mutation, substitution,
lecture courte, donnée après l’EOF attendu, dérive UID/device/volume ou dérive
de manifeste produit `STOPPED`.

Toute écriture durable suit :

1. fichier temporaire exclusif dans la racine cible et sur le même volume ;
2. écriture complète, `fsync`, puis `F_FULLFSYNC` sur macOS ;
3. promotion non-clobber avec `renameatx_np(..., RENAME_EXCL)` ;
4. `fsync` du répertoire parent et `F_FULLFSYNC` lorsque supporté.

Les receipts, événements, manifests et artefacts scellés ne sont jamais
supprimés ou remplacés. L’absence de `F_FULLFSYNC` requis produit `STOPPED`.

## 6. Hashes, manifests, receipts et ledgers

Le JSON canonique est UTF-8, clés triées, séparateurs compacts, valeurs non
finies interdites et un LF final.

- `source_sha256` porte sur les octets exacts du CRM ;
- `raw_row_sha256` porte sur les neuf champs bruts ;
- `query_id` porte sur `collection_id`, `source_batch_id`,
  `source_record_id`, `source_sha256` et `raw_row_sha256` ;
- chaque arbre utilise une convention non auto-référente :
  `payload_manifest.json` ferme exactement les fichiers de payload, en
  excluant `payload_manifest.json` et `seal.json`; `seal.json`, stocké hors du
  payload logique, ferme les octets exacts de `payload_manifest.json`, son hash
  d’arbre et l’identité du paquet. Aucun manifeste ne contient son propre hash.

Les identifiants exposés `query_id`, `opaque_batch_id` et
`opaque_stratum_id` sont reproductibles et opaques. Pour chaque domaine, le
builder calcule SHA-256 sur `domain || canonical_json_without_final_lf`, puis
remplace chaque nibble hexadécimal `0..f` par `a..p`. La sortie contient donc
exactement 64 lettres ASCII `[a-p]`. Les domaines distincts sont
`SIRETO-V412-FRESH-QUERY-ID\0`,
`SIRETO-V412-FRESH-BATCH-ID\0` et
`SIRETO-V412-FRESH-STRATUM-ID\0`. Les projections canoniques sont
respectivement : identité collection/batch/record et hashes source/ligne ;
identité collection/export/batch ; identité collection/source/portfolio.
Des golden vectors query, batch et stratum, avec sorties attendues, sont
obligatoires et scellés avant toute ouverture CRM. La table inverse reste
privée dans l’audit.

Le `holdout_id` n’est ni choisi ni incrémental. Après scellement des payloads,
il est le SHA-256 du JSON canonique contenant la sampling frame, les hashes des
manifests collection et source ordonnés et les hashes logiques non
auto-référents des arbres queries, oracle et audit pré-identité. Tous les
champs, seals, chemins ou entrées d’index qui incluent déjà `holdout_id` sont
exclus de cette projection. Un ledger global
append-only, indexé par ce contenu, associe chaque `holdout_id` à cette
projection et à ses manifests ; une même projection ne peut produire qu’une
entrée et une collision d’identité produit `STOPPED`.

Le ledger intake append-only contient rôle, chemin, device, inode, taille et
hash avant/après, projection, timestamp, export, batch, record, source,
portfolio, manifeste, receipt et événement. Le ledger d’évaluation est une
racine et une chaîne distinctes : il ne peut être écrit par l’intake et ferme
les descripteurs de requêtes, résultats, candidats et oracle utilisés lors du
one-shot.

Le receipt de batch est créé exclusivement avant parsing sous
`audit/receipts/batches/<source_sha256>/receipt.json`. Le receipt collection
est créé sous
`audit/receipts/collections/<collection_manifest_sha256>/receipt.json`.
Un receipt existant interdit tout rerun ou renommage de contournement.
Le receipt initial est créé `O_EXCL` une seule fois et ne contient que
l’identité reçue et la liaison vers la source déjà scellée : identifiants de
collection/batch, chemin absolu scellé, taille/hash, chemin/hash du manifeste
source et timestamp de création. Il est immuable et ne contient jamais de
dernier événement, checkpoint ou état dynamique.

La qualification écrit séparément des checkpoints immuables par shard et par
`query_id`, avec hash de l’entrée, résultat, preuve et builder. Après crash,
la reprise part du receipt puis reconstruit l’état depuis les générations
d’événements et les checkpoints. Le cas `RECEIPT_EXISTS_NO_EVENT` ouvre
uniquement le chemin scellé enregistré dans le receipt, le revalide et le
rehash, puis crée le premier événement ; il ne reparcourt jamais l’inbox.
La reprise rattache les artefacts scellés annoncés et ne recalcule que les
queries sans checkpoint durable.

## 7. Registres de base immuables et entrées interdites

La fermeture historique réelle est épinglée, sans inférence :

| Registre | Pin | Volume |
|---|---|---:|
| V4.11-A manifest | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/registries/v4_11_consumed_population/fd25d1922040d585/manifest.json` — `77711f91fda8dffec3210c49b3df8404e46ff540f30f9597fc7fe7722f2d6962` | 23 384 consommés |
| V4.11-A source registry | `source_registry.parquet` — `3fda773e3712b53aad017c2380471452c91e63fdf8a127a1fa09a46e8575e28b` | 23 609 total |
| V4.11-A consumed | `consumed.parquet` — `bad97c3769a621a6a32b4c27ce1a0b8c15cd1f3877f2718ab0b3ab6c8759fe32` | 23 384 |
| V4.11-A unseen | `unseen.parquet` — `63ff648f6e326721e0646b0101de079f9a6feadb6e02c0474066c1288d8025a3` | 225 |
| Challenge 225 sanitized | manifest `449bed70276f31728357c173a5d17a3f646c3975306a2488dabd95083cc7dae3` | 225 |
| Challenge 225 qualification | manifest `17c7915725cea978278f1699832e5c17405dbab8cd21ef407f6d96916a5c89e7` | 225 |
| Challenge 225 execution | manifest `37f4957052493b3aa1e8b2e3ba5f156816cb33121aa5915f88c9b581306c71e6` | 225 |

La fermeture est `23 384 + 225 = 23 609`, sans recouvrement ni ligne
disponible. Tous ces dossiers sont développement/diagnostic consommé.

`config/v4_12_development_inputs.json` et
`config/v4_12_forbidden_artifacts.json` sont des catalogues immuables à
épingler par hash brut dans le futur lock d’intake. Leurs artefacts, ainsi que
train/dev/test historiques, modèles, challenges et sorties finales, sont
interdits au processus de qualification, sauf les projections minimales des
registres de contamination explicitement autorisées.

Un registre `consumed_sirens` n’existe pas encore avec une fermeture
suffisante. Sa spécification est portée par
`docs/v4_12_consumed_sirens_registry_contract.md` et
`config/v4_12_consumed_sirens_registry_plan.json`, épinglés respectivement à
`90e3587f49f2cb84d3903e6120e9def22bcbf3affde64d5942b03077f582b4d5`
et `9ab3bdff52d65fe18a001ad4ca9f24857b0785cc809ad744a456ba7af53c187d`.
Un pinner séparé doit le construire et le sceller **avant toute
ouverture du nouveau CRM** à partir des oracles/labels consommés listés par
les deux catalogues. Le lock d’intake doit épingler les SHA-256 finaux de la
spécification, puis le `build_id`, le manifest et le payload logique du build
réel. Aucun pin ne peut être déduit d’un chemin « latest ». Tant que ces pins
manquent, l’intake ne peut pas ouvrir la collection.

## 8. Registre de compatibilité et anti-chevauchement

La normalisation ne sert jamais à améliorer une requête. Les règles locales
`v411_exact_fingerprint_sha256` et leur catalogue ad hoc sont obsolètes :
l’intake ne réimplémente aucun builder historique. La source unique est le
registre défini par
`docs/v4_12_consumed_compatibility_registry_contract.md` et
`config/v4_12_consumed_compatibility_registry_plan.json`, épinglés
respectivement à
`7413952f31b40da6e647e907dd3a5bd5b611d50884e3a1e0590da9a89d1c110f`
et `8c4f31ce4ebaee86724148e6b69638a0f4d6876f3a71e27ac23876f26700a258`.

Avant toute ouverture du nouveau CRM, ce registre doit être réellement
construit et scellé. Le lock d’intake épingle au minimum ses SHA-256 finaux de
contrat et plan, `build_id`, `payload_manifest_sha256`, `seal_sha256` et les
hashes des quatre keysets privés. Il épingle aussi `hmac_key_id` et
`hmac_key_sha256`. La clé HMAC de production est la même que celle du build du
registre : elle reste dans le Keychain macOS, n’est jamais placée dans Git, un
argument, une variable d’environnement, un fichier temporaire, un log ou un
manifest. Le processus d'intake la lit dans son propre processus via
`SecItemCopyMatching`, avec la même fiche Keychain épinglée et
`kSecUseAuthenticationUIFail`. Il vérifie identifiant, longueur et hash contre
le lock avant toute comparaison, puis efface sa copie mémoire mutable.

- `service_id_keyset.parquet` ;
- `siret_masked_keyset.parquet` ;
- `fuzzy_historical_keyset.parquet` ;
- `input_siret_lineage_keyset.parquet`.

La disjonction compare `source_sha256`, `raw_row_sha256` et, sans règle locale
supplémentaire, les clés `service_id_lineage_hmac_sha256`,
`siret_masked_fingerprint_sha256`,
`fuzzy_historical_fingerprint_sha256` et
`input_siret_lineage_hmac_sha256` définies par ce registre. La dernière n’est
calculée que si l’entrée autoritative correspondante existe ; son absence
n’est jamais remplacée par une prédiction. Toute collision interdit la ligne
sûre et est publiée. `source_system` et `portfolio_id` ne sont jamais des clés
d’exclusion : ils servent uniquement à la représentativité et aux strates.

La lignée historique `SERVICE ID` compare `source_record_id` uniquement si le
manifeste source atteste que sa sémantique est stable et équivalente à
`service_id_norm` V4.11, avec une référence d’attestation vérifiable. Sans
cette preuve, la lignée est `UNVERIFIABLE` et l’intake termine
`STOP_UNPROVABLE_LINEAGE` ; il ne suppose jamais une équivalence.

Après qualification, la disjonction SIREN compare au registre
`consumed_sirens` chaque SIREN autoritativement connu : ceux des
`MATCH_EXACT`, mais aussi ceux des `AMBIGUOUS` lorsqu’une preuve indépendante
établit le SIREN sans permettre de choisir un SIRET. Les trois disjonctions
sont obligatoires.

## 9. Catalogue de preuves et builder de qualification gelés

Avant ouverture du nouveau CRM, un `evidence_catalog_manifest.json` ferme :

- types de preuves autorisés, priorités et dates d’éligibilité ;
- snapshots officiels et documents administratifs avec chemins et hashes ;
- règles `MATCH_EXACT`, `AMBIGUOUS`, `UNRESOLVED` ;
- règles temporelles relatives à `reference_date` ;
- builder, tests, configuration, commit Git, runtime et hashes sources ;
- schémas exacts d’entrée, preuve et oracle.

La qualification s’exécute dans un processus isolé deny-by-default, sans
réseau, fork, modèle, cache retrieval, partition, candidat, résultat, registre
de développement ou oracle historique. Il reçoit uniquement par descripteurs
scellés et par canaux séparés : lignes CRM sûres + manifeste source d’un côté,
paquet evidence source + catalogue de preuves + registre `consumed_sirens`
minimal de l’autre. Le processus d'intake charge directement la clé depuis le
Keychain sans UI, vérifie les `hmac_key_id` et `hmac_key_sha256` du lock et ne
la sérialise jamais. Les permissions macOS strictement nécessaires à cette
lecture native doivent être gelées et testées avant toute ouverture du CRM.
Le processus parent garantit aussi que le paquet evidence et l’oracle produit
ne sont jamais transmis au scorer.

- `MATCH_EXACT` : preuves convergentes vers un unique SIRET éligible à
  `reference_date`, SIREN cohérent et au moins une preuve autoritative ;
- `AMBIGUOUS` : plusieurs SIRET compatibles ou SIREN seul ;
- `UNRESOLVED` : aucune preuve autorisée concluante.

Un snapshot SIRENE et une similarité nom/adresse ne peuvent jamais, à eux
seuls, créer `MATCH_EXACT`. Ils peuvent contrôler l’éligibilité temporelle
d’un SIRET déjà fourni par une preuve autoritative, ou révéler une
contradiction. Sans mapping contractuel, document administratif ou identifiant
source autoritatif indépendant, la ligne reste `UNRESOLVED`. Cette règle évite
de sélectionner comme vérité les seuls cas que le retrieval sait déjà
retrouver.

Hit, rang, score, candidat ou prédiction ne peuvent ni entrer dans le sandbox
ni apparaître dans le ledger. Les sorties sont rehashées et validées par le
parent avant promotion.

## 10. Métriques de couverture et gates d’intake

La métrique principale est :

`MATCH_EXACT / toutes les lignes source de la frame immuable`.

Son dénominateur inclut invalides, quarantaines, collisions, doublons,
`AMBIGUOUS` et `UNRESOLVED`. Le gate principal exige au moins 80,0 % observé.

Métriques secondaires, toujours séparées :

- `MATCH_EXACT / lignes structurellement admissibles` ;
- `(MATCH_EXACT + AMBIGUOUS) / toutes les lignes source` ;
- `(MATCH_EXACT + AMBIGUOUS) / lignes admissibles` ;
- taux `UNRESOLVED`, quarantaine, collision et fuite ;
- distributions par batch, source, portfolio et période.

Chaque publication donne obligatoirement les nombres bruts
`succès / dénominateur`, l’estimation observée et les intervalles de Wilson
bilatéraux à 95 % **et** 99 % pour la couverture et le Recall@100. Le gate
retrieval porte sur l’observation `Recall@100 >= 99,0 %`. Il ne constitue pas
une affirmation statistique `borne basse 99 % >= 99,0 %`. Cette seconde
affirmation n’est autorisée que si la borne calculée la franchit ; avec zéro
miss, elle demande au moins 657 cas indépendants.

`READY` exige aussi au moins 657 `MATCH_EXACT`, toutes les disjonctions, le
pin `consumed_sirens`, les catalogues gelés, les arbres/receipts/ledgers
valides et zéro fuite dans les requêtes sûres. Avec 657 exacts et zéro futur
miss, la borne basse **bilatérale** de Wilson à 99 %, avec
`z = 2,5758293035489004`, vaut environ `0,990002` et dépasse 99,0 %. Cette
taille permet ce claim seulement en cas de zéro miss ; le gate observé peut
être évalué sans ce claim.

## 11. Automates collection et batch, journal et recovery

Automate collection public :

`WAITING → INGESTED → QUALIFIED → READY`

`WAITING|INGESTED|QUALIFIED → STOPPED`

Automate batch :

`WAITING_STABLE → RECEIPTED → INGESTED → QUALIFYING → QUALIFIED`

Tout état non terminal peut aller vers `QUARANTINED` ou `STOPPED`. Une
collection ne passe `INGESTED` qu’après ingestion de tous les batches
préannoncés, et `QUALIFIED` qu’après qualification exhaustive.

Le journal autoritaire est un répertoire immuable. Chaque événement canonique
est créé exclusivement avec `O_EXCL` sous
`events/<sequence>-<event_sha256>.json`, puis `fsync`/`F_FULLFSYNC`. Son
contenu porte collection, batch, séquence, état précédent/nouveau, hashes des
manifests/arbres et `previous_event_sha256`. Le nom doit correspondre au hash
des octets exacts.

Après chaque frontière durable, un manifeste canonique créé `O_EXCL` sous
`events_manifests/<generation>-<manifest_sha256>.json` scelle la liste
ordonnée des événements, leurs tailles, hashes et le hash de tête. Aucune
génération existante n’est modifiée. `events_head.json` est seulement un cache
atomique dérivé pointant vers la dernière génération complète ; sa perte ou
son retard n’affecte pas l’autorité du journal. `state.json` est également un
cache dérivé.

La reprise de crash commence par le receipt, sélectionne la dernière génération
complète et valide, puis rejoint receipt → événement → checkpoint → artefact
scellé depuis des descripteurs ancrés. Un fichier temporaire ou un événement
créé mais non encore référencé par un manifeste complet est conservé comme
orphan et ignoré : il ne rend pas un préfixe sain irrécupérable. Un artefact
scellé annoncé par un événement durable peut être rattaché sans recomputation.
La reprise ne reparcourt pas l’inbox et ne rejoue aucun effet déjà checkpointé.
Une chaîne référencée conflictuelle, un hash faux ou une transition impossible
produit `STOPPED`.

## 12. Sorties séparées et scoring freeze manifest

Sorties :

- `ready/queries/<holdout_id>/queries.parquet` : `query_id` et identifiants de
  batch/strate opaques composés uniquement de lettres ASCII, plus date et CRM
  brut ; aucun `source_batch_id`, `source_record_id`, `source_system` ou
  `portfolio_id` n’est exposé au scorer ;
- `ready/evidence/<holdout_id>/` : paquets evidence source scellés, privés et
  jamais visibles du scorer ;
- `ready/oracle/<holdout_id>/oracle.parquet` : `query_id`, classe, SIRET/SIREN
  exacts éventuels, date, références/hashes de preuves et justification ;
- `ready/audit/<holdout_id>/` : manifests, provenance, événements, receipts,
  ledger intake, couvertures, disjonctions et catalogues ;
- `ready/scoring_freeze/<holdout_id>/scoring_freeze_manifest.json`.

Tous les arbres de sortie privés — queries CRM, evidence, oracle, audit,
registres anti-chevauchement, receipts, événements et ledger — sont créés avec
`umask 0077`, répertoires `0700` et fichiers `0600`. Les modes sont vérifiés
par descripteur avant promotion ; un mode plus permissif produit `STOPPED`.

La correspondance entre IDs opaques et provenance source reste dans
`ready/audit/<holdout_id>/source_identity_map.parquet`, sous permissions
privées. Avant promotion, **toutes** les colonnes string de `queries.parquet`,
y compris IDs et champs libres, sont scannées après NFKC avec la règle Unicode
`isdecimal()` exacte de 9/14 caractères et bornes non décimales. Les scans
ASCII historiques restent des contrôles complémentaires, pas la définition
de sécurité. Toute occurrence produit `STOPPED`. Les IDs opaques utilisent
exactement 64 lettres de l’alphabet `[a-p]`.

Le scoring freeze manifest est créé avant `READY` et ferme exactement :

- `holdout_id`, sampling frame, manifests queries/oracle/audit et row counts ;
- build/commit, sources, plan, lock, runtime, inputs et politique du retrieval ;
- plafond absolu `candidate_count <= 100`, jamais une moyenne ;
- tie-break, ordre, schémas et hash logique des candidats ;
- définitions coverage et Recall@1/@10/@50/@100 ;
- références `historical`, `V2`, `V3` avec manifests, populations et hashes ;
- règle « vérité absente du pool = miss end-to-end » ;
- règle one-shot, ledger d’évaluation distinct et séparation oracle ;
- seuils et verdicts `GO`, `PIVOT`, `STOP`.

Avant la moindre ouverture des requêtes par le scorer, celui-ci crée
exclusivement `OPENING.json` avec `O_EXCL` dans le ledger d’évaluation, puis
un événement durable `SCORING_OPEN_COMMITTED`. Une seconde création échoue et
interdit le rerun. Après scellement des résultats, l’évaluateur crée de même
son événement `ORACLE_OPEN_COMMITTED` avant d’ouvrir l’oracle. `OPENING.json`,
les deux événements et les descripteurs ouverts sont fermés par le ledger
d’évaluation et ses manifests générationnels.

Les références développement sont reprises sans retuning du plan évaluateur
V4.12, hash brut
`87bc2601f96307304a191b44afc8b2f356bc4a47a7cd90ddbe3b0270c4aa6c2d`,
build `ab8343817551c0a5` :

| Référence | Couverture | Recall@100 |
|---|---:|---:|
| historique | 2 565 / 2 565 | 2 495 / 2 565 |
| V2 exact | 2 400 / 2 565 | 2 343 / 2 400 |
| V3 exact identifiable | 2 104 / 2 565 | 2 095 / 2 104 |

Ces nombres sont des références publiées conjointement, pas le holdout frais
et pas un seuil appris sur lui.

Verdicts du scoring retrieval :

- `GO` : intégrité complète, plafond respecté, métrique principale intake
  ≥ 80 %, Recall SIRET exact @100 observé ≥ 99,0 %, et publication conjointe
  fresh/historique/V2/V3 ;
- `PIVOT` : protocole intègre mais volume, couverture ou Recall insuffisant ;
- `STOP` : fuite, dérive, rerun, optional stopping, arbre/ledger invalide,
  candidat >100 ou violation one-shot.

Un résultat `GO` retrieval ne déverrouille pas automatiquement le ranker ou
l’accepteur et ne revendique jamais `AUTO_MATCH @ 99,8 %`.

## 13. Verdicts intake

- `READY_FRESH_HOLDOUT_V412` : frame exhaustive, gates intake franchis et
  scoring freeze durable ;
- `PIVOT_FRESH_HOLDOUT_INTAKE_V412` : protocole sain mais population,
  couverture ou volume insuffisant au cutoff ;
- `STOP_FRESH_HOLDOUT_INTAKE_V412` : violation d’intégrité, indépendance,
  anti-fuite, durabilité, catalogue, automate ou sampling frame.
- `STOP_UNPROVABLE_LINEAGE` : aucune attestation vérifiable ne permet de
  comparer `source_record_id` à la lignée historique `SERVICE ID`.

Ce contrat n’autorise aucune lecture du prochain CRM avant scellement du plan,
des vrais registres, des catalogues, du builder et de leur lock d’exécution.
