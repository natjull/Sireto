# V4.12 — Contrat du registre historique de compatibilité anti-chevauchement

## 1. Objet et frontière

Ce contrat préenregistre la construction, **avant toute ouverture d’un futur
CRM**, d’un registre privé permettant de détecter la réutilisation de l’une des
23 609 lignes historiques déjà consommées.

Le registre ne qualifie aucun nouveau dossier, ne produit aucune vérité terrain
et n’ouvre aucun modèle, candidat, score, rang, prédiction ou futur holdout. Il
transforme uniquement la lignée historique déjà consommée en keysets
déterministes utilisables par l’intake V4.12.

Verdicts possibles :

- `GO_V412_CONSUMED_COMPATIBILITY_REGISTRY` : build complet, scellé et conforme ;
- `STOP_UNPROVABLE_LINEAGE` : une entrée, une projection ou une équivalence
  nécessaire ne peut pas être démontrée ;
- `STOP_V412_COMPATIBILITY_REGISTRY` : dérive de hash, de schéma, de volume,
  de normalisation, de golden vector, d’arbre ou de durabilité.

Un `GO` autorise seulement l’utilisation des keysets par l’intake. Il
n’autorise ni lecture du futur CRM, ni qualification, ni scoring tant que les
autres prérequis de l’intake ne sont pas eux-mêmes gelés.

## 2. Population fermée

La population effective contient exactement les 23 609 lignes de
`data/entrainements.csv` :

- 23 384 lignes consommées par le benchmark fermé historique ou V4-Fresh ;
- les 225 lignes restantes, consommées ensuite par le challenge descriptif
  V4.11.

Les 225 ne sont pas une population disponible. Le registre de compatibilité
porte donc `effective_consumed=true` sur les 23 609 lignes, indépendamment de
l’ancien champ `population_status`.

## 3. Entrées réelles épinglées

### 3.1 Registre V4.11-A

Racine :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/registries/v4_11_consumed_population/fd25d1922040d585`

| Fichier | Lignes | SHA-256 |
|---|---:|---|
| `manifest.json` | — | `77711f91fda8dffec3210c49b3df8404e46ff540f30f9597fc7fe7722f2d6962` |
| `source_registry.parquet` | 23 609 | `3fda773e3712b53aad017c2380471452c91e63fdf8a127a1fa09a46e8575e28b` |
| `consumed.parquet` | 23 384 | `bad97c3769a621a6a32b4c27ce1a0b8c15cd1f3877f2718ab0b3ab6c8759fe32` |
| `unseen.parquet` | 225 | `63ff648f6e326721e0646b0101de079f9a6feadb6e02c0474066c1288d8025a3` |

Le `source_registry.parquet` est la source structurée primaire. Le builder doit
vérifier qu’il transporte exactement, en plus des colonnes de lignée V4.11, les
huit champs historiques :

```text
SITE
CODE_POSTAL
CODE_INSEE
SERVICE ID
COMMUNE
SIRET
SITE_CLI_ADRESSE
SITE_CLI_COMMUNE
```

### 3.2 Source brute de contre-vérification

Le CSV historique est une seconde entrée obligatoire :

`/Users/nathanjullia/Documents/Projets/SIRETO/data/entrainements.csv`

SHA-256 :

`f770215cd0d0fcc654b750b90dbba835acbf4efb5c74ed269d339e046c2b049d`

Il doit commencer par exactement un marqueur UTF-8 `EF BB BF`. Le builder
exige ces trois octets à l'offset zéro, refuse un second marqueur immédiatement
après, retire uniquement le premier puis décode strictement le reste en UTF-8.
Un caractère `U+FEFF` situé dans une valeur CRM reste une donnée et n'est pas
interdit par cette règle d'en-tête. Le séparateur est `;`, toutes les colonnes
sont des chaînes et les valeurs vides sont conservées. Son ordre et ses huit
colonnes doivent reproduire exactement les 23 609 lignes brutes du
`source_registry.parquet`. Le builder recalcule aussi `service_id_norm`,
`input_siret_norm` et `row_fingerprint_sha256` V4.11 et exige une égalité ligne
à ligne avec le registre.

Si la source brute, les huit champs du registre ou cette parité ne sont pas
disponibles, le verdict est `STOP_UNPROVABLE_LINEAGE`. Aucune approximation à
partir d’un nom de fichier, d’un numéro de ligne ou d’un hash global ne suffit.

### 3.3 Fermeture des 225

Les trois manifests suivants prouvent la consommation ultérieure des 225 :

| Étape | Build | SHA-256 du manifeste |
|---|---|---|
| CRM assaini | `1c994c852c10acaf` | `449bed70276f31728357c173a5d17a3f646c3975306a2488dabd95083cc7dae3` |
| qualification | `4f9ef46516b89ab8` | `17c7915725cea978278f1699832e5c17405dbab8cd21ef407f6d96916a5c89e7` |
| exécution | `ddb7336e8c2e042d` | `37f4957052493b3aa1e8b2e3ba5f156816cb33121aa5915f88c9b581306c71e6` |

Le builder ne lit pas leurs données. Il valide seulement les manifests
épinglés et applique la fermeture aux 225 lignes désignées par
`unseen.parquet`.

## 4. Normalisations communes

Toutes les fonctions sont pures, sans locale implicite.

### 4.1 Texte V4.11

`canonical_v411(value)` :

1. valeur absente → chaîne vide ;
2. conversion en chaîne ;
3. Unicode NFKC ;
4. trim ;
5. toute suite Unicode de whitespace devient un espace ASCII ;
6. majuscules Unicode.

`service_id_norm = canonical_v411(SERVICE ID)`.

`input_siret_norm` conserve uniquement les chiffres Unicode après
`canonical_v411`; la valeur est conservée si et seulement si elle contient
exactement 14 chiffres, sinon elle devient vide. Le SIRET historique n’est
jamais une vérité ou une feature : c’est une clé de lignée privée.

### 4.2 Empreinte exacte V4.11

Le builder recalcule l’empreinte V4.11 originale sur les huit champs, sans
masquage, avec JSON UTF-8 `ensure_ascii=false`, clés triées et séparateurs
compacts. Cette empreinte sert uniquement à vérifier la parité de l’entrée
épinglée ; elle n’est pas le mécanisme principal compatible avec un futur CRM
sans SIRET.

### 4.3 Empreinte commune SIRET-masked

La projection historique contient les mêmes huit clés dans le même domaine
canonique V4.11, mais force toujours `SIRET=""` avant sérialisation :

```text
SITE
CODE_POSTAL
CODE_INSEE
SERVICE ID
COMMUNE
SIRET = ""
SITE_CLI_ADRESSE
SITE_CLI_COMMUNE
```

`siret_masked_fingerprint_sha256` est le SHA-256 des octets JSON canoniques de
cette projection.

Mapping du futur schéma V4.12, gelé dès maintenant :

| Clé historique | Champ V4.12 |
|---|---|
| `SITE` | `crm_name_raw` |
| `CODE_POSTAL` | `crm_postcode_raw` |
| `CODE_INSEE` | `crm_insee_raw` |
| `SERVICE ID` | `source_record_id` uniquement sous attestation d’équivalence ; sinon `""` |
| `COMMUNE` | `crm_city_raw` |
| `SIRET` | constante `""` |
| `SITE_CLI_ADRESSE` | `crm_address_raw` |
| `SITE_CLI_COMMUNE` | `crm_city_raw` |

Une collision exacte exclut la nouvelle ligne. Une absence d’attestation ne
supprime pas le contrôle fuzzy décrit ci-dessous, mais elle interdit de déclarer
la lignée de service vérifiée et conduit finalement à
`STOP_UNPROVABLE_LINEAGE`.

### 4.4 Empreinte fuzzy historique sans identifiant

Cette empreinte protège notamment les anciennes lignes sans `SERVICE ID` et
les réexports dont l’identifiant technique aurait changé. Elle reste une
égalité de hash, jamais une similarité.

`fuzzy_text(value)` :

1. valeur absente → chaîne vide ;
2. Unicode NFKD puis suppression des caractères de catégorie `Mn` ;
3. minuscules Unicode ;
4. chaque caractère non alphanumérique devient un espace ;
5. compactage des espaces.

Projection de base :

```text
name      = fuzzy_text(SITE)
address   = fuzzy_text(SITE_CLI_ADRESSE)
postcode  = fuzzy_text(CODE_POSTAL)
insee     = fuzzy_text(CODE_INSEE)
city      = une seule valeur fuzzy normalisée
```

Pour une ligne historique, le builder calcule l'ensemble trié des valeurs non
vides distinctes de `fuzzy_text(COMMUNE)` et
`fuzzy_text(SITE_CLI_COMMUNE)`. Il émet **une observation et une empreinte par
ville singleton**. Si les deux villes sont vides, il émet exactement une
observation avec `city=""`. Une ligne historique produit donc une ou deux
observations, jamais une empreinte contenant un tableau de plusieurs villes.

La sérialisation est un JSON UTF-8, clés triées, séparateurs compacts,
`ensure_ascii=false`. Le futur schéma émet exactement une projection avec
`name=fuzzy_text(crm_name_raw)`, `address=fuzzy_text(crm_address_raw)`,
`postcode=fuzzy_text(crm_postcode_raw)`, `insee=fuzzy_text(crm_insee_raw)` et
`city=fuzzy_text(crm_city_raw)`. Une ligne future à une seule ville peut ainsi
collisionner chacune des villes historiques distinctes ; il n'existe aucun
problème de cardinalité `1 → N`.

`fuzzy_historical_fingerprint_sha256` est le SHA-256 de ces octets. Toute
collision avec l'une des observations exclut la nouvelle ligne, même si les
identifiants techniques diffèrent. Aucun suffixe juridique, token métier ou
numéro de voie n’est retiré au-delà de la normalisation ci-dessus.

## 5. Golden vectors obligatoires

Le plan canonique contient les entrées historiques et V4.12 de golden vectors,
ainsi que les hashes attendus. Les tests doivent au minimum démontrer :

- accents composés et décomposés ;
- whitespace Unicode et espaces multiples ;
- ponctuation d’adresse ;
- SIRET historique non vide sans effet sur l’empreinte masked ;
- équivalence du mapping ancien/nouveau lorsque `source_record_id` est attesté ;
- changement de `source_record_id` détecté par le keyset service et compensé
  par l’empreinte fuzzy ;
- ordre et déduplication déterministes des deux communes historiques ;
- deux villes historiques distinctes produisant deux empreintes singleton, dont
  chacune est reproductible par une ligne V4.12 portant cette seule ville ;
- chaîne vide et valeur absente ;
- chiffres Unicode de catégorie `Nd` acceptés par `str.isdigit()` ;
- caractère Unicode tel que `²`, accepté par `str.isdigit()` mais refusé par
  `str.isdecimal()` sur l'entrée, puis transformé en chiffre ASCII par le NFKC
  de `canonical_v411` avant calcul du HMAC, afin de pinner sans ambiguïté le
  comportement V4.11.

Le payload logique des golden vectors et le fichier de tests sont épinglés dans
le lock d’exécution. Un seul écart donne
`STOP_V412_COMPATIBILITY_REGISTRY`.

## 6. Sorties privées

Racine immuable :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake/registry/consumed_compatibility/<build_id>`

Payload :

1. `compatibility_rows.parquet`, exactement 23 609 lignes :
   `source_row_number`, `effective_consumed`, `consumption_reason`,
   `service_id_present`, `input_siret_present`,
   `v411_row_fingerprint_sha256`,
   `siret_masked_fingerprint_sha256`,
   `fuzzy_fingerprint_count` et les HMAC de lignée ; aucune valeur CRM brute ;
2. `service_id_keyset.parquet` :
   `service_id_lineage_hmac_sha256`, `row_count`, sans identifiant en clair ;
3. `input_siret_lineage_keyset.parquet` :
   `input_siret_lineage_hmac_sha256`, `row_count`, audit-only, permissions
   privées ;
4. `siret_masked_keyset.parquet` :
   `siret_masked_fingerprint_sha256`, `row_count` ;
5. `fuzzy_historical_observations.parquet` :
   `source_row_number`, `city_ordinal`,
   `fuzzy_historical_fingerprint_sha256` ;
6. `fuzzy_historical_keyset.parquet` :
   `fuzzy_historical_fingerprint_sha256`, `row_count` ;
7. `provenance.json` : pins, versions, mapping, commit, runtime, code et tests,
   ainsi que `hmac_key_id` et `hmac_key_sha256`, jamais la clé ;
8. `rejected_rows.parquet` :
   `source_row_number`, `reason_code`, sans valeur brute ;
9. `integrity.json` : volumes, unicités, multiplicités, parités et invariants.

Les clés de lignée utilisent exclusivement :

```text
HMAC-SHA256(K, "SIRETO-V412-SERVICE-ID-LINEAGE\0" + service_id_norm UTF-8)
HMAC-SHA256(K, "SIRETO-V412-INPUT-SIRET-LINEAGE\0" + input_siret_norm UTF-8)
```

Un SHA-256 simple est interdit car les espaces de `SERVICE ID` et SIRET sont
énumérables. `K` est une clé aléatoire privée d'au moins 256 bits, conservée
dans le Keychain macOS. Le builder la lit dans son propre processus avec
`SecItemCopyMatching`, en épinglant le service
`com.sireto.v412.compatibility-hmac`, le compte `SIRETO`, une seule fiche
`GenericPassword`, un retour `CFData` et
`kSecUseAuthenticationUIFail`. Pour la lecture et le transport de la clé, il
n'appelle ni le CLI `security`, ni un launcher, ni un sous-processus et échoue
sans afficher de demande d'autorisation. Le plan fixe un `hmac_key_id`; le
lock fixe le SHA-256 des octets exacts de `K`. La clé n'est jamais présente
dans Git, un argument CLI, une variable d'environnement, un fichier
temporaire, un log, un manifest ou une sortie. Le builder vérifie
indépendamment `key_id`, longueur minimale, hash de clé et golden vectors
HMAC, puis efface sa copie mémoire mutable dans un `finally`. Les copies
internes transitoires de Security.framework, CPython et HMAC ne sont pas
présentées comme effaçables par le builder.

Toutes les sorties sont des fichiers mode `0600` sous des répertoires mode
`0700`. Les valeurs sources en clair ne quittent jamais le processus builder.
Le registre, ses keysets et les fingerprints sont interdits au retrieval, au
ranker, à l’accepteur et au scorer ; seul le processus d’intake anti-overlap,
dans un sandbox deny-by-default lisant la même fiche Keychain par l'API
native sans UI, peut lire les projections minimales nécessaires.

## 7. Schémas physiques et reproductibilité byte-for-byte

Les trois parquets V4.11 d'entrée possèdent le même schéma Arrow physique,
épinglé dans le plan par sa sérialisation IPC SHA-256. Il contient, dans
l'ordre, `source_row_number:int64`, les huit champs CRM `string`, quatre champs
de lignée `string`, quatre indicateurs de match `bool`, deux indicateurs de
consommation `bool`, puis `population_status:string` et
`consumption_sources:string`. Tous les champs sont nullable. Toute différence
de champ, ordre, type, nullabilité ou métadonnée de schéma produit
`STOP_INPUT_DRIFT`.

Le plan fixe également la taille exacte de chaque entrée, le nombre de row
groups et les schémas Arrow complets des sorties. Le builder utilise
exclusivement Python `3.14.3` et PyArrow `23.0.1`, sans pandas pour sérialiser.
Les parquets sont écrits avec :

```text
format_version = 2.6
compression = zstd
compression_level = 9
use_dictionary = false
write_statistics = true
data_page_version = 1.0
row_group_size = 65536
store_schema = true
```

Avant écriture, toutes les tables sont triées selon l'ordre déclaré dans le
plan, rechunkées en un chunk par colonne et castées vers le schéma exact.
Les JSON suivent la sérialisation canonique du plan. La reproductibilité
byte-for-byte n'est revendiquée que sous le runtime, la plateforme, les options
writer et l'ordre épinglés. Deux builds indépendants avec les mêmes entrées et
la même clé doivent produire les neuf fichiers payload, leur manifest et leur
seal byte-for-byte identiques. Le runner réalise effectivement ces deux
écritures dans deux arbres `O_EXCL`, compare chaque fichier, ferme un événement
`BYTE_REPRODUCIBILITY_VALIDATED`, puis promeut seulement l'arbre primaire.
L'arbre de reproduction complet reste un orphan d'audit conservé.

## 8. Manifest et seal sans autoréférence

La fermeture utilise trois niveaux :

1. `payload_manifest.json` énumère exactement les neuf fichiers payload
   ci-dessus par chemin relatif, taille et SHA-256. Il ne s’énumère pas
   lui-même ;
2. `seal.json` contient le hash et la taille de `payload_manifest.json`, le
   `build_id`, le hash de la spécification de build et le hash de tête logique
   du payload. Il ne contient pas son propre hash ;
3. le SHA-256 brut de `seal.json` est publié par le ledger de promotion et par
   le futur lock d’intake.

Tout fichier supplémentaire, absent ou symbolique invalide le build. Cette
construction ne requiert aucun point fixe cryptographique.

Le `build_id` est le SHA-256 complet de la spécification canonique comprenant
schéma, normalisations, mappings, pins d’entrée, golden vectors, commit, hashes
du builder et des tests. Le répertoire porte ce hash complet.

## 9. Durabilité, receipts et reprise après crash

Toutes les traversées partent de `/` et conservent les descripteurs de
répertoires parents. Chaque composant est ouvert avec `openat`,
`O_NOFOLLOW|O_CLOEXEC` et `O_DIRECTORY` pour un répertoire. Les fichiers
terminaux doivent être réguliers, avoir un seul hard link, l'UID attendu et le
device/UUID de volume épinglé. `fstat` immédiatement après ouverture et après
la dernière lecture doit reproduire device, inode, taille, mtime, ctime, UID et
link count. Chaque entrée est lue jusqu'à EOF et hashée avant/après sur le même
FD ; lecture courte, octet supplémentaire ou dérive produit `STOP`.

Pour un nouvel attempt, après validation en mémoire de la clé et avant toute
lecture sémantique d'une source historique, un receipt d'attempt immuable est
créé avec `O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC`, mode `0600`. Les
événements canoniques sont ensuite créés individuellement avec `O_EXCL` et
chaînés par `previous_event_sha256`. Des manifests générationnels immuables
`events_manifests/<generation>-<sha256>.json` ferment chaque préfixe valide.
Le receipt reste statique ; l'état courant appartient uniquement à cette chaîne.

L'autorisation d'exécution est portée par un unique JSON canonique de schéma
`sireto-v4.12-consumed-compatibility-execution-lock-1`, lui-même appelé par
chemin et SHA-256. Il contient sans exception le chemin et le hash du plan, le
commit et le hash du builder, le chemin et le hash des tests, l'identifiant et
le hash de la clé HMAC, l'identifiant d'attempt, ainsi que les pins UID, device
et UUID de volume de chaque fichier lu et de la racine de sortie. Ces valeurs
ne peuvent pas être remplacées individuellement sur la ligne de commande.
Le CLI n'accepte aucun secret, chemin de secret ou descripteur de clé.

Chaque fichier temporaire est créé `O_EXCL` dans la racine cible sur le même
volume. L'ordre durable est : écriture complète, `fsync`, `F_FULLFSYNC`,
validation, puis promotion de l'arbre complet par
`renameatx_np(..., RENAME_EXCL)`, `fsync` et `F_FULLFSYNC` du parent.

Le FD de la racine de sortie est ouvert et validé une seule fois, puis conservé
jusqu'à la fin de l'exécution. La création des deux arbres, toutes les
écritures, relectures, validations, comparaisons et la promotion sont effectuées
uniquement avec des noms relatifs à ce FD (`mkdirat`, `openat`,
`renameatx_np`). Une substitution ultérieure du chemin texte de la racine ne
peut donc ni détourner une écriture ni changer la cible promue.

Après crash, le runner valide le receipt, la dernière génération complète et
un éventuel arbre temporaire complet. Il peut uniquement promouvoir cet arbre
déjà validé ; il ne relit pas partiellement les sources, ne rejoue pas un effet
et ne complète pas un payload. Un événement ou temporaire non référencé reste
un orphan conservé. Une chaîne conflictuelle, un arbre partiel ou un hash faux
produit `STOP`.

La validation de l'arbre recalcule aussi la provenance depuis le plan et le
lock, le `build_id` depuis la spécification de build et tous les volumes,
bornes, multiplicités et invariants du plan. Un arbre de test réduit ne peut
donc jamais être scellé avec le plan de production. Le device et l'inode de la
racine validée sont transmis à la promotion et revérifiés sur le FD retenu
juste avant `renameatx_np`, ce qui ferme la substitution de la racine entre
validation et renommage. Seul le dernier préfixe fermé par un manifest
générationnel fait autorité ; tout événement postérieur non fermé reste un
orphan conservé et n'est jamais appliqué.

Au démarrage, après validation des seuls artefacts de contrôle (lock, plan,
contrat, builder, tests et commit), le runner recherche le receipt exact de
l'attempt avant de charger la clé ou une source historique. Si ce receipt
existe, il valide sa chaîne, son dernier préfixe fermé et l'arbre temporaire ou
déjà promu correspondant. Il promeut exclusivement un arbre complet lié à
`TREE_VALIDATED`, ou retourne l'arbre déjà promu ; il ne recrée pas le receipt,
ne relit aucune source et ne relance aucun build. Un état antérieur non
récupérable produit `STOP`.

En l'absence d'attempt existant, le runner lit et valide la clé dans le
Keychain avant de créer le receipt et avant toute lecture d'une source
historique. Un échec Keychain, une interaction nécessaire, un identifiant
différent, une clé trop courte ou un hash différent ne crée donc ni receipt,
ni payload, ni arbre de build. Après validation de la clé, le receipt est
rendu durable avant la première lecture historique.

## 10. Rejets et échec fermé

Une ligne est inscrite dans `rejected_rows.parquet` si une parité V4.11,
projection, normalisation ou clé obligatoire est invalide. Un build de succès
exige zéro rejet. Un rejet ou une entrée insuffisante produit un artefact
d’échec séparé sous une racine `attempts`, puis
`STOP_UNPROVABLE_LINEAGE` ou `STOP_V412_COMPATIBILITY_REGISTRY`; il n’est
jamais promu comme registre.

Les doublons de fingerprints masked ou fuzzy ne sont pas des rejets
historiques : les keysets les agrègent avec `row_count`. Ils rendent le contrôle
du futur intake plus conservateur.

## 11. Invariants bloquants

Le build promu exige :

- hashes de toutes les entrées strictement conformes ;
- 23 609 lignes source, numéros exactement `1..23609`, sans perte ni ajout ;
- huit colonnes historiques brutes strictement égales entre CSV et registre ;
- 23 384 anciens `CONSUMED`, 225 anciens `UNSEEN`, puis 23 609
  `effective_consumed=true` ;
- 225 lignes `V411_CHALLENGE_225`, toutes issues exactement de
  `unseen.parquet` ;
- 659 `service_id_norm` vides et 22 950 non vides ;
- égalité ligne à ligne des `service_id_norm`, `input_siret_norm` et empreintes
  V4.11 recalculés ;
- 23 609 empreintes masked valides et entre 23 609 et 47 218 observations
  fuzzy singleton valides de 64 hexadécimaux ;
- zéro rejet ;
- golden vectors tous conformes ;
- chaque keyset trié lexicographiquement, unique par clé et doté d’une
  multiplicité strictement positive ;
- somme des multiplicités de chaque keyset égale au nombre de lignes portant
  la clé correspondante ;
- chaque keyset est exactement égal au `Counter` recalculé depuis les colonnes
  de `compatibility_rows.parquet` pour service, SIRET d'entrée et masked, ou
  depuis `fuzzy_historical_observations.parquet` pour fuzzy ; une clé remplacée
  avec multiplicité inchangée reste donc un `STOP`, même après recalcul du
  manifest et du seal ;
- somme des multiplicités fuzzy égale au nombre d'observations fuzzy, et chaque
  ligne possède exactement une observation par ville normalisée distincte ;
- clé HMAC lue uniquement par l'API Keychain native en processus, sans UI,
  `hmac_key_id` et hash conformes au lock, aucun SHA simple de service/SIRET
  et aucune clé secrète dans les sorties ;
- aucune valeur `SERVICE ID`, SIRET, nom ou adresse en clair dans les sorties ;
- schémas Arrow, runtime, compression, row groups, tris et bytes reproductibles ;
- modes `0600/0700`, receipts, événements, durabilité, recovery,
  payload manifest, seal et arbre exacts.

Tout invariant non démontrable ferme l’exécution. Aucun comportement permissif,
fallback fuzzy approximatif ou admission « best effort » n’est autorisé.

## 12. Ordre d’exécution et autorisation

Ordre obligatoire :

1. committer ce contrat et le plan canonique ;
2. implémenter builder et tests sans ouvrir le futur CRM ;
3. contre-auditer code, golden vectors, pins et sandbox ;
4. créer la clé privée dans le Keychain, pinner `hmac_key_id` et le hash de ses
   octets dans le lock, sans exposer la clé ;
5. produire un lock d’exécution avec tous les hashes, runtime, schémas, options
   writer, UID, device et UUID de volume ;
6. construire une seule fois le registre historique ;
7. vérifier indépendamment payload, keysets, seal et invariants ;
8. pinner le `build_id` complet et le SHA-256 de `seal.json` dans le lock
   d’intake V4.12 ;
9. faire lire à l'intake la même fiche Keychain par l'API native sans UI et
   seulement ensuite autoriser l’observation stable du prochain export CRM.

Le présent contrat et son plan ne construisent pas le registre et
n’autorisent pas encore l’étape 8.
