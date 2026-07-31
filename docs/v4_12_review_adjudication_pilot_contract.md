# V4.12-R30 — Contrat du pilote d'adjudication des REVIEW

Statut : préenregistré avant construction du docket, avant lecture de toute
preuve publique nouvelle et avant toute adjudication.

Identifiant : `V412_REVIEW_ADJUDICATION_PILOT_30`.

## 1. Question testée

Le retrieval V4.12 conserve le bon SIRET dans 100 candidats sur les 1 217
requêtes `MATCH_EXACT` du dev historique. Le Ranker C place le bon SIRET en
première position dans 1 216 cas. Le service gelé V4.12-G produit néanmoins
279 `REVIEW` sur les 1 456 requêtes de ce dev déjà consommé.

Le pilote répond à une seule question :

> Les REVIEW déjà consommés peuvent-ils fournir, de manière autonome et
> traçable, assez de labels difficiles fiables pour justifier une
> adjudication plus large puis une expérience OOF de l'accepteur ?

Le pilote ne mesure pas la précision produit. Il n'entraîne aucun modèle, ne
choisit aucun seuil et ne permet pas d'ouvrir un test final.

## 2. Entrées immuables

La population vient exclusivement de la référence de parité service :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/references/v4_12_service_parity/b4b7fef24c5e7036`

| Fichier | SHA-256 |
|---|---|
| `manifest.json` | `cbcb3303107cd00f895561b49b8ad3a26e5c8e3df8a07777817e7a6ed97f2340` |
| `queries.parquet` | `70ded26776bfd56c96501c6033e0e322a6dd11ed296c3309ad89bd9deec84cf9` |
| `guard_reference.parquet` | `fee3880a9d3b485abdcca2417952a19baaf70cf35d5dc60fb882378b10f42cca` |
| `query_evidence.parquet` | `3ec693b0258b1b1988be226a9aa803656de20e0fcd8aec7feaa960c4fa13e4a8` |
| `ranker_reference.parquet` | `418c8cffec21f030f08baa59e292240e7f4bffbbdc2dcb79b50e83052db48df7` |
| `scenes_reference.parquet` | `9bc4a5f5528f5f4a04126ad3078bcb950e9538b85907dfb4fedd8bd32a8e660c` |

Les exclusions anti-fuite utilisent uniquement les identifiants et rôles du
registre V4.8, jamais ses labels :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_8_acceptor_partitions/1c78764d5263afca`

| Fichier | SHA-256 |
|---|---|
| `manifest.json` | `f0e255b891dfb6b24d57f3b7423dd64a227908dbf68559b2da4572ea37791d33` |
| `partition_assignments.parquet` | `f828249172c36ce33a3279d294dfc5030e6d8eeb58baee9cf9e08130f13593b9` |

L'exclusion des dossiers déjà adjudiqués utilise seulement leurs identifiants
dans la consolidation V4.7 :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_7_current_adjudications/4cc5420fb5da0683`

| Fichier | SHA-256 |
|---|---|
| `manifest.json` | `634ad13c1c2eda0abd7c2921e94ebc1631c070cae8cb3b480514bbfba59e3a8c` |
| `current_labels.parquet` | `e5e592d4dcd540273378dada7128f957b1d335df63fbc88f4c1377c0f9337bd2` |

Le builder projette physiquement la seule colonne `query_id` de ce dernier
Parquet. Ouvrir une colonne de label, de cible, de SIRET, de score ou de
preuve pendant la sélection est interdit.

Le registre SIRENE utilisé pour vérifier un identifiant découvert est :

`/Users/nathanjullia/Documents/Projets/SIRETO/data/StockEtablissement_utf8.parquet`

- SHA-256 :
  `c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845` ;
- 42 322 035 lignes et 42 322 035 SIRET uniques ;
- son rôle est de confirmer l'existence, l'état et les attributs d'un SIRET
  découvert, jamais de créer seul la relation entre le CRM et ce SIRET.

Avant sélection, le builder doit reproduire exactement :

- 1 456 requêtes uniques ;
- 1 177 `AUTO_MATCH` et 279 `REVIEW` V4.12-G ;
- 40 REVIEW avec plusieurs sites directs d'un même SIREN ;
- 199 REVIEW avec collision directe entre plusieurs SIREN ;
- 40 autres REVIEW.

Tout écart donne `STOP_INPUT_INTEGRITY`.

## 3. Sélection aveugle des 30 dossiers

La sélection utilise seulement les requêtes CRM, la décision `REVIEW`, les
agrégats de preuve directe et les identifiants de séparation. Elle ne lit
aucun label, aucune vérité SIRET, aucun résultat de test et aucune
adjudication antérieure.

Les trois strates sont mutuellement exclusives, dans cet ordre :

1. `SAME_SIREN_MULTISITE` :
   `same_siren_direct_multisite == true` ;
2. `CROSS_SIREN_COLLISION` :
   `cross_siren_direct_collision == true` et strate 1 fausse ;
3. `OTHER_REVIEW` :
   les deux indicateurs précédents sont faux.

Dans chaque strate, les dossiers sont ordonnés par :

```text
SHA256(UTF8("SIRETO-V412-R30-SELECTION\0" + query_id))
```

puis par `query_id` croissant. Les dix premiers admissibles sont retenus.
Il est interdit de remplacer un dossier après ouverture des preuves, y
compris s'il devient difficile ou non résolu.

La liste ordonnée canonique des 30 `query_id`, sérialisée comme un tableau
JSON UTF-8 compact sans tri supplémentaire, concatène les dix lignes
`SAME_SIREN_MULTISITE`, puis les dix `CROSS_SIREN_COLLISION`, puis les dix
`OTHER_REVIEW`. Dans chaque bloc, l'ordre reste
`(selection_digest, query_id)` croissant. Elle doit avoir le SHA-256 :

`ec481d8db07165185fecc61bf437d868bfcbe4db6f4938a62b6c344e7000c2ee`

Sont exclus avant tri :

- tout identifiant ou composante lié aux partitions `random_sealed`,
  `hard_dev_locked` ou `descriptive_locked`, ou aux rôles `random_sealed`,
  `historical_random_excluded`, `hard_dev_locked` ou
  `descriptive_locked` ; les composantes interdites sont d'abord dérivées
  des seules colonnes `query_id`, `component_id`, `partition` et `role`,
  puis tous leurs identifiants sont exclus ;
- toute population challenge, holdout, unseen, test final ou collection
  fraîche non consommée. Les `query_id` préfixés `fresh:` déjà inclus dans la
  référence dev V4.12 sont un namespace historique de V4-Fresh consommé :
  ils restent admissibles uniquement via les Parquets de référence épinglés.
  Ce préfixe n'autorise l'ouverture d'aucune racine ou source V4-Fresh ;
- toute requête absente de la référence dev V4.12 ;
- tout dossier déjà adjudiqué dans V4.4 ou V4.7.

Le docket est scellé avant la première collecte de preuve. Il contient le CRM
utile à la recherche, le top-1 gelé, au plus les 100 candidats déjà présents,
la strate et les empreintes de provenance. Il ne contient aucun label.

La collecte est effectuée en deux passes séparées :

1. `IDENTITY_DISCOVERY` ne voit que le CRM et la strate. Elle ne reçoit ni
   top-1, ni candidat, ni rang, ni score et construit un dossier d'identité ;
2. `FROZEN_CANDIDATE_COMPARISON` n'est ouverte qu'après scellement du dossier
   d'identité. Elle compare ses SIRET supportés au top-100 gelé.

Cette séparation interdit d'utiliser le top-1 comme unique terme de recherche
et réduit le biais de confirmation.

## 4. Plan de collecte borné

Avant tout accès réseau, le builder matérialise pour chaque dossier les trois
requêtes exactes suivantes après échappement littéral des valeurs CRM :

1. `"<crm_name>" "<crm_postcode>"` ;
2. `"<crm_name>" "<crm_city>" "<crm_address>"` ;
3. `"<crm_name>" "<crm_city>" (SIRET OR établissement)`.

Elles sont exécutées dans cet ordre. Pour chaque requête, seuls les cinq
premiers résultats du moteur sont journalisés et seuls les deux premiers
résultats admissibles peuvent être ouverts. Un domaine déjà ouvert pour le
dossier est sauté sans augmenter le quota de deux groupes indépendants. Le
registre SIRENE local épinglé est consulté une fois, séparément, pour chaque
SIRET découvert.

La collecte d'un dossier s'arrête uniquement après les trois requêtes et
l'ouverture de leurs résultats admissibles, ou lorsqu'elle a archivé six
pages admissibles hors SIRENE. Une erreur réseau est journalisée et ne permet
ni remplacement de requête, ni essai supplémentaire. Une URL hors des
familles autorisées ci-dessous est archivée comme résultat observé mais ne
peut produire aucun fait. À l'issue du quota, les preuves insuffisantes
donnent obligatoirement `UNRESOLVED`.

Familles admissibles, dans l'ordre de groupe :

- `SIRENE_REGISTRY` : snapshot SIRENE local épinglé ;
- `PUBLIC_ADMINISTRATION` : domaine officiel `.gouv.fr`, collectivité ou
  établissement public identifiable ;
- `ENTITY_OFFICIAL_SITE` : site publié comme officiel par l'entité ;
- `OFFICIAL_SECTOR_DIRECTORY` : annuaire sectoriel porté par une autorité
  publique ;
- `DATED_PUBLIC_DOCUMENT` : PDF ou page datée émise par l'entité ou une
  autorité publique.

Un moteur de recherche, un agrégateur commercial ou une copie du registre
SIRENE ne forme jamais un groupe de preuve. Aucun service payant, compte
authentifié ou API à coût variable n'est autorisé.

Le plan scellé publie pour chaque dossier les trois requêtes, leur ordre et
leurs quotas. Changer une requête ou continuer après le quota produit
`STOP_INTEGRITY`, jamais une adjudication plus favorable.

## 5. Schéma des faits et règle d'adjudication

Le protocole reprend la frontière V4.7 :

- snapshot SIRENE épinglé comme registre officiel ;
- au moins une autre source publique indépendante : site officiel de
  l'entité, annuaire public sectoriel officiel, API publique officielle ou
  document public daté ;
- URL ou référence, date de collecte, contenu archivé, SHA-256 et faits
  contrôlables conservés ;
- deux restitutions du même enregistrement SIRENE forment un seul groupe ;
- une adresse commune, une similarité, un score modèle, un rang ou l'absence
  de résultat web ne constitue jamais une preuve suffisante.

Chaque dossier reçoit exactement un statut :

- `TOP1_CORRECT` : les preuves identifient sans contradiction le top-1 ;
- `TOP1_WRONG` : les preuves contredisent le top-1 ;
- `AMBIGUOUS` : plusieurs SIRET restent légitimement possibles ;
- `UNRESOLVED` : les preuves disponibles ne permettent pas de conclure.

Une décision fiable exige au moins deux groupes indépendants, dont le
registre officiel, une preuve du site précis lorsqu'un SIREN a plusieurs
établissements possibles et aucune contradiction non résolue.

`NO_MATCH` est interdit : le snapshot local ne porte pas une temporalité
autoritative suffisante pour prouver l'absence d'un SIRET éligible à la date
du CRM.

Pour `TOP1_WRONG`, deux champs distincts sont publiés :

- `exact_alternative_known` : un SIRET alternatif exact est prouvé ;
- `alternative_naturally_in_top100` : cet exact alternatif était déjà dans
  le pool gelé, sans injection.

Un `TOP1_WRONG` sans ces deux propriétés peut entraîner un accepteur à
refuser, mais jamais un ranker à choisir un positif.

### 5.1 Faits atomiques

`facts.parquet` contient exactement :

```text
query_id: string
proof_id: string
independence_group: enum des cinq familles autorisées
source_url_or_snapshot_ref: string
collected_at_utc: timestamp
content_sha256: 64 hex
fact_type:
  SIRET_IDENTIFIER | SIREN_IDENTIFIER | LEGAL_NAME |
  SITE_NAME | SITE_ADDRESS | ENTITY_SITE_RELATION
fact_value_normalized: string
related_siret: string|null
related_siren: string|null
site_specific: bool
source_excerpt_sha256: 64 hex
extractor_rule_id: string
```

Le texte source et l'extrait exact sont archivés séparément. L'extrait doit
contenir la valeur brute dont `fact_value_normalized` est la normalisation.
Les colonnes `verdict`, `correct`, `wrong`, `ambiguous`, `target`,
`ground_truth`, `candidate_rank` et `model_score`, ainsi que leurs synonymes,
sont interdites dans les faits.

Un groupe non SIRENE `supports(siret)` pour un dossier si une même archive
contient un `SIRET_IDENTIFIER` égal au SIRET et remplit simultanément :

- au moins un `LEGAL_NAME` ou `SITE_NAME` dont la forme normalisée est égale
  au nom CRM normalisé, ou satisfait la règle directe de nom gelée
  `active-direct-current-v4.0` ;
- au moins un `SITE_ADDRESS` qui partage le code postal CRM et le numéro de
  voie lorsqu'il existe, ou satisfait la règle directe d'adresse de cette
  même politique ;
- si plusieurs établissements actifs du SIREN satisfont ces conditions, un
  `ENTITY_SITE_RELATION` ou `SITE_ADDRESS` marqué `site_specific=true`
  distingue le SIRET exact.

`SIRENE_REGISTRY supports(siret)` seulement si le SIRET existe exactement,
est actif dans le snapshot et si ses champs reproduisent les faits nom/site
de l'archive indépendante. Le snapshot ne peut donc jamais être les deux
groupes à lui seul. Deux URLs de la même famille comptent toujours pour un
seul groupe.

### 5.2 Table de décision exhaustive

Après fermeture des faits, le comparateur calcule l'ensemble des SIRET ayant
au moins deux groupes indépendants, dont `SIRENE_REGISTRY`.

| Ensemble supporté | Relation au top-1 | Statut |
|---|---|---|
| exactement un SIRET | égal au top-1 | `TOP1_CORRECT` |
| exactement un SIRET | différent du top-1 | `TOP1_WRONG` |
| au moins deux SIRET | toute relation | `AMBIGUOUS` |
| aucun SIRET | — | `UNRESOLVED` |

Une contradiction explicite entre deux faits d'un même groupe invalide ce
groupe. Une contradiction entre groupes laisse plusieurs SIRET supportés et
produit `AMBIGUOUS`; si aucun SIRET ne conserve deux groupes, le résultat est
`UNRESOLVED`. Aucune saisie libre ne peut remplacer cette table.

## 6. Sorties et intégrité

Le jalon produit dans un répertoire immuable :

- `docket.parquet` : 30 dossiers, dix par strate ;
- `candidate_context.parquet` : candidats gelés, maximum 100 par dossier ;
- `collection_plan.parquet` : requêtes, ordre et quotas scellés ;
- `facts.parquet` : faits atomiques reconstruits depuis les archives ;
- `evidence.parquet` : preuves archivées et faits contrôlés ;
- `adjudications.parquet` : statuts, raisons et références ;
- `gate_report.json` : nombres bruts et verdict ;
- `manifest.json` : hashes des entrées, du contrat, du code et des sorties.

Le code doit reconstruire les adjudications depuis les faits conservés. Une
table de verdicts libre ou une conclusion uniquement rédigée par un LLM
n'est pas canonique.

### 6.1 Exécuteur fermé

Le launcher est scellé avant la collecte. Son hook d'audit et ses wrappers
d'ouverture :

- autorisent en lecture seulement les onze entrées épinglées de ce contrat,
  le snapshot SIRENE et les fichiers d'archives créés par le run ;
- autorisent en écriture seulement un staging privé sous la racine du run ;
- interdisent toute racine dont un composant normalisé contient
  `test`, `final`, `holdout`, `challenge`, `unseen`, `fresh`, `random` ou
  `locked`, hors les fichiers d'entrée explicitement épinglés. La présence du
  préfixe historique `fresh:` dans une valeur `query_id` n'est pas un chemin
  et ne crée aucune exception à cette denylist ;
- interdisent les symlinks, les hardlinks multiples, les sous-processus et
  tout fichier local non allowlisté ;
- journalisent chaque ouverture locale et chaque URL avant son accès.

Avant publication, un validateur recalcule le journal et exige zéro ouverture
hors allowlist. Une violation donne `STOP_INTEGRITY` et aucun dossier n'est
remplacé. Les tests doivent injecter au moins une tentative d'ouverture pour
chaque token interdit et prouver son refus avant lecture.

## 7. Gates préenregistrés

Le verdict est `SCALE_ADJUDICATION` si et seulement si :

- les 30 dossiers sont présents exactement une fois ;
- au moins 18/30 décisions sont fiables ;
- chaque strate possède au moins quatre décisions fiables ;
- au moins six décisions fiables sont `TOP1_WRONG` ou `AMBIGUOUS` ;
- `SAME_SIREN_MULTISITE` et `CROSS_SIREN_COLLISION` possèdent chacune au
  moins un `TOP1_WRONG` ou `AMBIGUOUS` fiable ;
- aucune décision fiable ne repose sur moins de deux groupes indépendants ;
- aucun label antérieur n'a été transporté ;
- aucun random, holdout, challenge, unseen, test final, ni racine ou
  population fraîche non consommée n'a été ouvert ;
- aucun candidat positif n'a été injecté.

Le verdict est `STOP_LABEL_PATH` si l'intégrité est saine mais qu'un seuil de
volume ou de preuve échoue. Toute fuite, mutation, substitution de dossier ou
ouverture interdite donne `STOP_INTEGRITY`.

`SCALE_ADJUDICATION` atteste uniquement le rendement et la diversité du
processus de labellisation. Il autorise l'adjudication d'un lot historique
plus large sous un contrat séparé. Il n'atteste aucune amélioration de
précision ou de couverture et n'autorise pas encore un entraînement.

## 8. Décision de modélisation ultérieure

Après un éventuel lot élargi :

- un corpus composé de `TOP1_CORRECT`, `TOP1_WRONG` et `AMBIGUOUS` fiables
  peut justifier une expérience accepteur query-level en prédictions OOF ;
- le ranker ne peut être réentraîné que si un nombre préenregistré de
  `TOP1_WRONG` possède un SIRET alternatif exact, prouvé et naturellement
  présent dans le top 100 ;
- tout modèle est comparé au bundle V4.12-G gelé sur train/dev consommés ;
- le test final et toute future cohorte indépendante restent fermés jusqu'au
  gel complet du candidat.

Ce pilote améliore la matière d'apprentissage potentielle. Seule une nouvelle
collection CRM indépendante pourra mesurer la North Star produit à 99,8 %.
