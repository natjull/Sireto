# Addendum d’exécution v12 — compilateur OOF fold-fused

Statut : **préenregistré, non exécuté**.

Cet addendum remplace uniquement le plan physique d’exécution du cycle v12.
Il ne modifie ni la population, ni les candidats logiques, ni les features, ni
les modèles, ni les folds, ni les caps, ni les métriques et seuils du contrat
`retrieval_all_human_v12_contract.md`.

L’exécuteur historique reste une référence de parité sur un échantillon borné.
Il ne doit pas être lancé nationalement : il matérialise trois Masters larges,
répète les mêmes lectures et lookups de sidecar et appelle le ranker parent sur
de très nombreux petits `DMatrix`. Ce coût est une propriété de son plan
physique, pas de la recette scientifique.

## 1. Plan physique gelé

Le nouvel exécuteur se nomme `FOLD_FUSED_CROSSFIT_COMPILER_V12_1`.

1. Une `SceneIR` label-blind et content-addressée est compilée une seule fois
   depuis les trois shards v11. Elle contient les références légères, les
   signaux de canal, la géographie, les relations officielles et la matrice de
   features parent, mais aucune vérité humaine, aucun équivalent opérationnel
   et aucun résultat d’évaluation.
2. Le pool parent naturel complet de chaque requête est construit avant que le
   label-side puisse sélectionner au plus 128 parents de fit. Les six modèles
   parent et leur routage nested OOF restent exactement ceux du contrat v12.
3. Pour chaque bloc de requêtes, les features parent sont calculées une fois.
   Les modèles requis par les trois contextes outer sont évalués par prédiction
   matricielle. Il n’existe ni ensemble de modèles ni moyenne entre contextes.
4. Les top parents des trois contextes sont unionnés uniquement pour effectuer
   un lookup officiel batché unique par requête. Un masque de contexte rejoue ensuite chaque
   expansion et chaque round-robin indépendamment ; un parent retenu dans un
   contexte ne peut ajouter un site à un autre contexte.
5. Pour une requête du fold `k`, le Master complet est conservé pour le seul
   contexte `outer=k`. Dans les deux autres contextes, les statistiques sont
   calculées sur le Master complet naturel, puis seules les 128 lignes de fit
   prévues par la recette sont matérialisées.
6. Les trois LambdaMART SIRET sont ajustés sur ces matrices compactes. Les
   Masters heldout complets sont ensuite scorés en blocs pour produire le
   PreRank@500, le top100, les outcomes et la taxonomie de pertes.

La table de routage reste immuable :

| fold requête | outer2 | outer3 | outer4 |
|---|---|---|---|
| 2 | fit34 | fit4 | fit3 |
| 3 | fit4 | fit24 | fit2 |
| 4 | fit3 | fit2 | fit23 |

## 2. Équivalences obligatoires

- mêmes références naturelles et mêmes expansions officielles ;
- mêmes caps Master 2 000, PreRank 500 et sortie 100 ;
- même ordre des features parent et SIRET, en `float32` ;
- même normalisation calculée sur le pool complet avant sampling ;
- même sampling naturel de fit et aucun positif injecté ;
- mêmes tie-breaks déterministes, notamment score décroissant puis SIRET
  croissant ;
- même nested OOF, sans lecture des folds 0 et 1 ;
- même dénominateur all-human et mêmes métriques exactes.

Le score parent effectué après expansion dans l’ancien exécuteur est supprimé
uniquement lorsqu’il est démontré qu’il est écrasé sans être consommé. La
prédiction batch remplace la prédiction singleton seulement si ses sorties
sont bit-identiques avec les six modèles réels.

## 3. Gate de parité avant OOF national

Sur les 512 requêtes scellées par le ledger v12, ancien et nouvel exécuteurs
doivent produire, pour chacun des trois contextes :

- les mêmes parents, scores et rangs ;
- les mêmes SIRET du Master, rangs, canaux et payloads officiels ;
- le même ordre d’expansion et le même hydration rank diagnostique ;
- les mêmes 319 valeurs de features SIRET en `float32` ;
- les mêmes identifiants et le même ordre des lignes de fit ;
- les mêmes PreRank@500 et top100, byte pour byte.

Le lookup sidecar mutualisé est comparé aux trois lookups indépendants. Un test
de perturbation vérifie qu’une modification d’un modèle outer n’affecte aucun
autre contexte. Toute divergence arrête le run ; elle ne peut être corrigée en
changeant un seuil, un cap ou une règle après lecture des labels.

## 4. Stockage et runtime

Les temporaires et matrices sont écrits exclusivement sur
`/Volumes/CATNAT_DATA`. Le disque système ne reçoit aucun spool, cache Arrow,
matrice XGBoost ou sortie OOF. Les trois Masters complets ne sont jamais
matérialisés simultanément.

Le lookup de production utilise une projection mmap dérivée du sidecar
`siren_local_sites`, avec les mêmes lignes et le même ordre que la vue
Parquet/DuckDB. Ce changement d’accès ne peut être activé qu’après parité
exacte contre le reader courant.

La cible d’ingénierie est un OOF complet en 2 à 4 heures après compilation,
avec un plafond temporaire de 20 Gio. Cette cible de durée n’est pas un gate
scientifique et ne peut pas modifier la recette si elle est manquée.
