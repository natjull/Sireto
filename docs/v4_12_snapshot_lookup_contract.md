# V4.12 — Contrat du lookup SIRENE pour l'inférence

Statut : préenregistré avant tout build du lookup, toute parité retrieval
end-to-end et toute mesure de latence V4.12.

## 1. Pourquoi ce composant est nécessaire

Le chemin V4.11 gelé hydrate les candidats par un scan bulk du snapshot
SIRENE. Il n'existe pas encore de chemin canonique
« une requête CRM → une décision » : le helper par requête attend une méthode
`get_candidate_scene_details()` que le store d'état ne fournit pas.

Mesurer les parquets historiques ou répartir le temps d'un scan bulk entre
les requêtes ne satisferait pas le gate de latence d'inférence du contrat
V4.12. La prochaine brique est donc un lookup local, indexé et en lecture
seule, strictement équivalent à l'hydratation bulk V4.11.

Ce composant :

- ne change ni le retrieval, ni le ranker, ni les features ;
- ne contient aucun label ;
- n'ajoute aucune source externe ;
- est construit sur le Mac et stocké sur `/Volumes/CATNAT_DATA` ;
- ne peut choisir, réordonner ou injecter un candidat.

## 2. Source et projection gelées

Source unique :

```text
/Users/nathanjullia/Documents/Projets/SIRETO/data/StockEtablissement_utf8.parquet
SHA-256 = c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845
```

La table `candidate_details` contient exactement :

| Colonne lookup | Colonne snapshot | Coercition |
|---|---|---|
| `siret` | `siret` | `CAST AS VARCHAR` |
| `candidate_state` | `etatAdministratifEtablissement` | `upper(trim(CAST AS VARCHAR))` |
| `enseigne1` | `enseigne1Etablissement` | `CAST AS VARCHAR` |
| `enseigne2` | `enseigne2Etablissement` | `CAST AS VARCHAR` |
| `enseigne3` | `enseigne3Etablissement` | `CAST AS VARCHAR` |
| `denomination_usuelle` | `denominationUsuelleEtablissement` | `CAST AS VARCHAR` |
| `activity_code` | `activitePrincipaleEtablissement` | `CAST AS VARCHAR` |

Les six valeurs métier et leurs valeurs nulles doivent être identiques au SQL
de `bulk_hydrate_snapshot()` dans
`scripts/build_v411_input_blind_dataset.py`. Aucun nettoyage supplémentaire,
fallback ou enrichissement n'est autorisé.

## 3. Build

Le build produit un répertoire immuable sous :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/v4_12_snapshot_lookup/<build_id>
```

Sorties minimales :

```text
candidate_details.duckdb
manifest.json
integrity.json
timing.json
```

La base :

- possède exactement une table métier `candidate_details` ;
- possède exactement les sept colonnes de la section 2, dans cet ordre ;
- contient une seule ligne par SIRET ;
- refuse tout SIRET vide, non numérique ou différent de 14 caractères ;
- refuse tout doublon, même si les sept valeurs sont identiques ;
- possède un index unique sur `siret` ;
- est ouverte en `read_only=True` par le moteur d'inférence ;
- renvoie au plus les SIRET demandés, sans compléter les absents.

Le builder :

1. vérifie le hash du snapshot et les sources gelées ;
2. vérifie au moins 50 Gio libres avant staging ;
3. construit dans un répertoire temporaire du même volume ;
4. contrôle schéma, cardinalité, unicité et index ;
5. ferme puis rouvre la base en lecture seule ;
6. calcule les hashes après fermeture ;
7. vérifie un pic RSS inférieur ou égal à 8 Gio ;
8. `fsync` les fichiers et répertoires ;
9. publie par renommage atomique ;
10. refuse une cible existante.

Le refus de tout doublon est un prérequis d'intégrité renforcé, pas une
nouvelle coercition métier. Il est compatible avec le snapshot gelé, qui
contient exactement 42 322 035 lignes, 42 322 035 SIRET uniques valides et
aucun SIRET invalide. Si ces nombres changent, le build s'arrête avant de
comparer le lookup à l'ancien `SELECT DISTINCT`.

Paramètres DuckDB gelés :

```text
version = 1.4.3
PRAGMA memory_limit = '6GB'
PRAGMA threads = 4
temp_directory = <staging>/duckdb_tmp
CREATE UNIQUE INDEX candidate_details_siret_uidx
    ON candidate_details(siret)
```

Le builder contrôle l'index via `duckdb_indexes()`, exécute `CHECKPOINT`,
ferme toute connexion et refuse un fichier `.wal` avant hash et publication.

Le `build_id` dépend du snapshot, du SQL/projection, des sources, du runtime et
du verrou. Aucun timestamp n'entre dans son identité.

## 4. Parité obligatoire du lookup

Le lookup n'est utilisable qu'après deux comparaisons bit-exactes.

### 4.1 Tous les candidats retrieval historiques

Source exacte :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_input_blind/ec4326ec57e4411d/candidates_sparse_top100.parquet
SHA-256 = 78b2f78ddeac863ac39ca64301d42312c7fb766ac51e2b5d19dde5c5910aedac
698 892 lignes
508 081 SIRET uniques
```

La seule projection physique autorisée est `candidate_siret`; la colonne
`is_ground_truth` reste interdite.

Pour tous les 508 081 SIRET uniques :

1. les SIRET demandés sont matérialisés sans autre colonne ;
2. la référence est recalculée par un unique scan bulk du snapshot, avec une
   jointure sur ces SIRET et les mêmes sept expressions SQL que la section 2 ;
3. cette référence temporaire est fermée avant d'ouvrir le lookup ;
4. le lookup est interrogé par lots d'au plus 100 SIRET ;
5. présence, valeurs, types logiques et nullité doivent être identiques ;
6. aucun SIRET supplémentaire ne peut être renvoyé.

Il est interdit de rescanner le snapshot une fois par lot ou d'utiliser le
lookup lui-même pour produire la référence.

### 4.2 Échantillon indépendant du snapshot

Après `DISTINCT CAST(siret AS VARCHAR)`, prendre les 10 000 plus petits
couples :

```text
(SHA-256("v412-lookup-parity:" + siret), siret)
```

sur les SIRET valides du snapshot, sans utiliser de résultat modèle. Comparer
les sept valeurs snapshot/lookup exactement comme en 4.1.

L'ordre est calculé par DuckDB `sha256`, puis `siret` comme second critère. Le
SHA-256 des 10 000 SIRET ordonnés, chacun suivi de `\n`, est gelé :

```text
58c9700d2a1ed2bb433e4f7a25a845ba236d63cfe633dcd64f9156469777f945
```

Un seul écart donne `STOP_V412_LOOKUP_PARITY`.

## 5. Store d'inférence

Le futur store expose uniquement :

```text
get_candidate_scene_details(sirets: Sequence[str])
    -> dict[siret, {
         candidate_state,
         enseigne1,
         enseigne2,
         enseigne3,
         denomination_usuelle,
         activity_code
       }]
```

Contrat :

- l'entrée est une `Sequence[str]` d'au plus 100 éléments ;
- chaque élément est de type Python `str` et respecte exactement
  `^[0-9]{14}$`, sans `strip`, `zfill` ni conversion implicite ;
- `None`, `NaN`, bytes, entiers, whitespace et chaînes invalides font échouer
  l'appel ;
- les doublons sont dédupliqués en conservant la première occurrence ;
- requête paramétrée, jamais de SQL construit depuis une valeur CRM ;
- aucun cache mutable persistant ;
- aucune écriture DuckDB ;
- sortie déterministe ordonnée par SIRET avant conversion en dictionnaire ;
- les SIRET valides absents du snapshot sont omis du dictionnaire.

Le retrieval V4.11 produit déjà des chaînes canoniques de 14 chiffres. Cette
API fail-closed empêche qu'une normalisation nouvelle modifie silencieusement
un pool.

La parité retrieval end-to-end doit ensuite démontrer, sur les 1 456 requêtes
dev, que lookup par requête et hydratation bulk produisent exactement les
mêmes pools, rangs, features, scores et décisions V4.11.

## 6. Benchmark ultérieur

Le lookup ne franchit aucun gate de latence à lui seul. Après sa parité :

- population : les 1 456 requêtes dev, sélectionnées sans labels ;
- moteur persistant, ordre hashé préenregistré ;
- p95 `nearest-rank`, sur durées entières par requête ;
- preuve V4.12 recalculée, jamais lue depuis l'artefact scellé ;
- cache TF-IDF vérifié en lecture seule, zéro miss ;
- premier passage processus neuf publié comme diagnostic seulement ;
- passage warm filesystem utilisé pour les gates ;
- `p95(full V4.12) < 2 × p95(full V4.11)` ;
- `p95(preuve + garde) <= p95(retrieval V4.11)` ;
- RSS inférieur ou égal à 8 Gio.

La population est reconstruite depuis `split_assignments.parquet`
(`33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193`)
avec la projection exacte
`query_id, siren_component_id, split, oof_fold`. Elle contient toutes les
lignes `split == "dev"` : 710 `threshold_dev` et 746 `comparison_dev`, la
partition dev étant recalculée par
`SHA-256("v411-threshold:" + siren_component_id)` comme en V4.11. Aucun label,
scène ou résultat modèle n'entre dans cette sélection.

Un débit batch ou un temps amorti ne peut pas être présenté comme latence
d'inférence.

## 7. Autorisation et anti-fuite

Le builder et le validateur exigent, avant toute exécution réelle :

1. commit du présent contrat et du plan JSON ;
2. commit séparé du builder, du store et des tests ;
3. verrou externe épinglant commits, sources, snapshot, inputs, runtime et
   chemin de sortie ;
4. audit indépendant `GO_BUILD_V412_SNAPSHOT_LOOKUP`.

Sont interdits :

- les trois challenges V4.11 consommés, par chemin et par hash ;
- tout label, vérité CRM ou `is_ground_truth` ;
- scènes, décisions ou scores pour construire le lookup ;
- réseau, API, LLM, GPU loué ou dépense externe ;
- toute modification du snapshot source.

Le parquet de candidats retrieval est ouvert après build uniquement avec la projection
physique `candidate_siret` pour la parité 4.1.

## 8. Verdicts

- `GO_V412_SNAPSHOT_LOOKUP` : build intègre et deux parités exactes ;
- `STOP_V412_LOOKUP_BUILD` : intégrité, ressources ou publication invalides ;
- `STOP_V412_LOOKUP_PARITY` : au moins un écart de présence, valeur, type ou
  nullité.

Seul `GO_V412_SNAPSHOT_LOOKUP` autorise la construction du moteur
d'inférence V4.12 et le benchmark apparié. Il ne certifie pas la production.
