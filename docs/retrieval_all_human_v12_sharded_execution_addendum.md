# Addendum physique v12 — OOF fold-sharded exact

Statut : **PRÉENREGISTRÉ — PARITÉ W1/W3 EN ATTENTE — OOF NATIONAL
INTERDIT**.

Ce document complète l’addendum fold-fused v12 sans le remplacer. Il ne
change ni la population, ni la génération, ni les features, ni le sampling,
ni les modèles, ni les métriques, ni les seuils scientifiques. Il fixe
uniquement l’ordonnancement multi-processus du même calcul.

Le PREFIT non circulaire du moteur fold-fused a passé ses quatre contrôles sur
les 512 requêtes scellées :

- manifeste :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_prefit_parity_v12_7590ca67/manifest.json` ;
- SHA-256 :
  `0da156d88928c732d4e226dc7266159f713c8c5c9af131cfbe5cfdfe3c0a6ed0` ;
- phase : `PREFIT_NON_CIRCULAR` ;
- statut : `PASS` ;
- population : 512 requêtes et 1 536 contextes ;
- fold 0 ouvert : non ; fold 1 ouvert : non ;
- injection positive : non.

Cette preuve valide le noyau fold-fused contre la référence historique. Elle
ne valide pas encore l’ordonnanceur shardé décrit ici. Une preuve physique
supplémentaire W1/W3 est obligatoire avant l’OOF national.

## 1. Autorités immuables

Le plan JSON associé est
`config/retrieval_all_human_v12_sharded_execution_plan.json`. Il épingle :

- le contrat et la configuration scientifique v12 ;
- la configuration et l’addendum du noyau fold-fused ;
- la SceneIR, son index et la source candidate ;
- le ledger all-human, le sidecar opérationnel, les readers officiels et mmap ;
- les six modèles parent nested OOF ;
- le sample de parité 512 et le PREFIT passé.

Le plan shardé ne peut lire que les folds 2, 3 et 4. Les folds 0 et 1 restent
fermés. Une différence de hash, de cardinalité, d’ordre ou de schéma impose
`STOP` avant tout calcul national.

## 2. Partition physique gelée

La SceneIR réelle est déjà ordonnée par fold. Ses frontières de folds tombent
exactement sur des frontières de row-groups Parquet. Les trois shards sont
donc fixés ainsi, avec bornes de fin exclusives :

| shard | fold | requêtes SceneIR | row-groups source | lignes candidates |
|---|---:|---:|---:|---:|
| `fold2` | 2 | `[0, 7767)` | `[0, 210)` | `[0, 13 750 106)` |
| `fold3` | 3 | `[7767, 15734)` | `[210, 423)` | `[13 750 106, 27 705 638)` |
| `fold4` | 4 | `[15734, 23571)` | `[423, 635)` | `[27 705 638, 41 590 480)` |

Cardinalités respectives :

- fold2 : 7 767 requêtes et 13 750 106 références ;
- fold3 : 7 967 requêtes et 13 955 532 références ;
- fold4 : 7 837 requêtes et 13 884 842 références.

Les shards sont disjoints, contigus et couvrent exactement 23 571 requêtes,
635 row-groups et 41 590 480 références. Aucun hash-partitioning, shuffle ou
tri ultérieur par `query_id`, SIREN ou SIRET n’est autorisé.

## 3. Modèle de processus

Le start method est obligatoirement `spawn`. Un worker ouvre ses propres
handles read-only SceneIR, DuckDB, mmap et XGBoost ; aucun handle ouvert dans
le coordinateur ne peut être hérité.

Le nombre maximal de workers simultanés est choisi une seule fois avant
l’ouverture du premier shard :

1. mesurer `psutil.virtual_memory().used` ;
2. lire deux fois le compteur `Swapouts` de `vm_stat`, à cinq secondes
   d’intervalle ;
3. choisir W3 uniquement si la RAM machine utilisée est inférieure ou égale à
   14 Gio **et** si `swapouts_end - swapouts_start == 0` ;
4. choisir W2 dans tous les autres cas, y compris si une mesure est
   indisponible ou invalide.

La décision, les valeurs brutes, la taille de page, les deux instants de
mesure et la raison doivent être scellés avant le premier `spawn`. Il est
interdit de changer le nombre de workers après ouverture d’un shard, de
relancer automatiquement avec une autre valeur ou de choisir W3 à partir de
métriques de qualité du retrieval.

W2 et W3 exécutent les trois mêmes shards. Ils ne changent que le nombre de
shards calculés simultanément. Les workers natifs de prédiction sont plafonnés
à quatre threads par processus uniquement après parité bitwise ; les trois
fits LambdaMART conservent les paramètres scientifiques gelés, dont
`n_jobs=0`.

## 4. Une requête, trois contextes

Une requête ne peut jamais être divisée entre plusieurs workers. Le worker qui
la possède :

1. compile une fois la scène label-blind ;
2. produit ensemble les trois contextes outer 2, 3 et 4 ;
3. effectue le lookup officiel mutualisé une seule fois puis applique les
   masques de contexte indépendants ;
4. écrit le Master heldout `outer == fold_requête` avant toute lecture du
   label ;
5. calcule les statistiques sur chacun des deux Masters train complets avant
   sampling ;
6. ouvre ensuite le label uniquement pour sélectionner naturellement au plus
   128 lignes par contexte train ;
7. n’injecte jamais le positif absent.

Le cache d’un worker ne peut mémoriser que des résultats label-blind. Il ne
peut ni partager une décision entre contextes, ni changer l’ordre des
candidats.

## 5. Fusion train et cross-fitting exact

Chaque shard écrit des groupes de requêtes contigus et ordonnés par ordinal
SceneIR. Le coordinateur refuse un shard incomplet, dupliqué, chevauchant,
hors ordre ou dont le manifeste ne correspond pas aux entrées épinglées.

La fusion des matrices train est fixée, sans tri supplémentaire :

- `outer2 = fold3 ⊕ fold4` ;
- `outer3 = fold2 ⊕ fold4` ;
- `outer4 = fold2 ⊕ fold3`.

`⊕` signifie concaténation physique des groupes dans l’ordre source. Ainsi :

- le modèle outer2 n’apprend que sur les folds 3 et 4 ;
- le modèle outer3 n’apprend que sur les folds 2 et 4 ;
- le modèle outer4 n’apprend que sur les folds 2 et 3.

Les identifiants de groupes, l’ordre intra-groupe, les 319 features `float32`,
les labels et les poids sont identiques au chemin W1. Les trois modèles site
sont ajustés séquentiellement par le coordinateur, dans l’ordre outer2,
outer3, outer4. Aucun fit distribué, concurrent, incrémental ou par shard n’est
autorisé.

## 6. Replay heldout et sorties

Après scellement des trois modèles, les spools heldout sont rejoués en
parallèle :

- le shard fold2 utilise exclusivement le modèle outer2 ;
- le shard fold3 utilise exclusivement le modèle outer3 ;
- le shard fold4 utilise exclusivement le modèle outer4.

Les workers de replay n’ouvrent ni SceneIR, ni sidecars, ni modèles parent. Le
Master complet, jusqu’à 2 000 candidats, est toujours scoré. Le top500 et le
top100 conservent les mêmes scores, tie-breaks et raisons de sélection.

Les sorties finales sont concaténées fold2, puis fold3, puis fold4, dans
l’ordre ordinal interne de chaque shard. Les gates sont calculés une seule
fois sur les 23 571 outcomes fusionnés. Une moyenne de métriques par shard ou
par fold est interdite.

Le manifeste final doit lier les trois manifests de shards, les matrices
fusionnées, les trois modèles, les trois spools heldout, les sorties fusionnées
et leur ordre global. Tout artefact temporaire réside sous
`/Volumes/CATNAT_DATA` ; le plafond de 20 Gio reste inchangé.

## 7. Parité physique W1/W3 obligatoire

Avant l’OOF national, le sample immuable de 512 requêtes doit être exécuté par
le chemin monolithique W1 et le chemin shardé W3. Cette preuve est purement
physique et ne peut publier aucune métrique de qualité ni aucun `READY`.

Les contrôles sont exacts :

- population, folds, ordinals et ordre global ;
- parents, scores et rangs bit pour bit ;
- Masters, payloads, canaux et ordre byte pour byte ;
- statistiques du Master complet ;
- identifiants et ordre des lignes de fit ;
- matrice train, groupes et 319 valeurs `float32` bit pour bit ;
- spool heldout et ordre de replay ;
- hashes des trois modèles d’essai produits par les deux chemins ;
- scores site, PreRank500, top100, raisons, outcomes et agrégats byte pour
  byte.

La preuve doit aussi exécuter des tests négatifs : shard manquant ou dupliqué,
chevauchement, inversion de fusion, mauvais modèle outer, fold 0/1, hash
altéré, label lu avant scellement du Master, candidat injecté, sortie hors
ordre et gate calculé par moyenne de shards.

W2 utilise exactement les mêmes trois shards et le même merge que W3 ; seule
la concurrence maximale change. Une parité fixture W1/W2/W3 est obligatoire,
mais la preuve réelle scellée reste W1/W3 sur les 512 requêtes.

Toute divergence donne `STOP_SHARDED_PARITY`. Il n’existe aucune correction
automatique, aucun ajustement de seuil et aucun basculement vers l’OOF
national.

## 8. Gates scientifiques inchangés

Le sharding ne modifie aucune cible :

- Master@2000 reste diagnostique ;
- PreRank@500 doit être au moins 99,4 % ;
- Recall SIRET exact@100 doit être au moins 99,2 % ;
- la borne Wilson 95 % doit être au moins 99,0 % ;
- aucun outer fold ne peut être sous 98,8 % ;
- caps absolus : Master 2 000, fit 128, PreRank 500, produit 100 ;
- SLO chaud : p95 au plus 1 s et p99 au plus 2 s.

L’OOF reste `NO_READY` jusqu’à la parité POSTFIT utilisant les trois modèles
exacts du run, puis jusqu’au profil runtime séparé. Le fold1 ne peut être
ouvert que sur autorisation one-shot explicite après tous les gates.
