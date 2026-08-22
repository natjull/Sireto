# Contrat v11 — retrieval sur tous les labels humains

Statut : **préenregistré, non exécuté**. Ce contrat remplace uniquement le
périmètre expérimental du prochain cycle retrieval. Il ne modifie aucun
résultat historique et n'autorise ni l'ouverture du fold 1, ni un entraînement
aval.

## 1. Vérité principale et dénominateur

La cible exacte canonique est `human_ground_truth_siret` :

1. `ground_truth_siret` lorsqu'il contient exactement quatorze chiffres ;
2. sinon `historical_ground_truth_siret` lorsqu'il contient exactement
   quatorze chiffres ;
3. sinon la ligne est exclue faute de label humain SIRET exploitable.

C'est la seule exclusion autorisée. `MATCH_EXACT`, `AMBIGUOUS`, `UNRESOLVED`,
les accords avec le dossier officiel, l'état administratif et les résultats
du retrieval ne peuvent pas réduire le dénominateur principal.

Sur le fold 0 déjà audité, cette règle correspond à 7 005 labels utilisables
sur 7 009 scènes. Les quatre autres scènes n'ont ni cible courante ni cible
historique exploitable. Cette référence est descriptive ; le builder v11
recalcule et publie les volumes par fold depuis son entrée scellée.

Le SIRET humain demeure inchangé même si un autre SIRET du même SIREN et du
même site est opérationnellement acceptable. Aucune qualification ne corrige
silencieusement le label humain.

## 2. Exact et opérationnel sont physiquement séparés

Les Parquet de benchmark exact ne contiennent pas
`acceptable_sirets_operational`. Le builder publie des fichiers label-side
distincts, reliés uniquement par `query_id` :

- le SIRET exact humain y est toujours présent ;
- les alternatives doivent être des SIRET valides du même SIREN ;
- ces alternatives ne changent jamais la métrique exacte ;
- elles servent seulement à la métrique opérationnelle secondaire et à
  l'exclusion des faux hard negatives.

La politique détaillée reste
`docs/siret_operational_equivalence_policy.md`.

## 3. Cloisonnement des folds

- folds 2/3/4 : construction et validation croisée de la recette ;
- fold 0 : diagnostic seulement, sans sélection de variante, de seuil ou de
  gate ;
- fold 1 : test fermé.

Le builder peut sceller le contenu du fold 1 dans des fichiers marqués
`locked`, mais aucun fichier de développement ne peut contenir une ligne du
fold 1. Le manifeste doit déclarer `fold1_opened=false` et publier le SHA-256
de chaque fichier verrouillé. Aucun runner générique de train ou de dev ne
doit accepter ces fichiers.

Fold 0 ayant déjà été observé lors des cycles v8–v10 et des diagnostics
post-hoc, il n'est plus autorisé pour choisir une variante ou un seuil. Il
reste une vue diagnostique facultative, pas une preuve indépendante et pas un
prérequis à l'ouverture autorisée du fold 1. Seul le fold 1 pourra fournir la
preuve finale après gel de la recette sur les folds 2/3/4.

## 4. Artefacts obligatoires du builder

Le builder v11 consomme un benchmark humain content-addressé et publie :

- `development.parquet` : folds 0/2/3/4, métrique exacte ;
- `test_locked.parquet` : fold 1 uniquement, métrique exacte ;
- `operational_labels_development.parquet` ;
- `operational_labels_test_locked.parquet` ;
- `missing_human_labels_development.parquet` ;
- `missing_human_labels_test_locked.parquet` ;
- `manifest.json`.

Le manifeste lie les SHA-256 de la source, du contrat, de la configuration et
du builder. Il publie les lignes source, utilisables et exclues, ainsi que les
comptages utilisables/manquants pour chacun des folds 0 à 4. Les listes de
labels manquants sont conservées ; aucune ligne ne disparaît sans trace.

Invariants fail-closed :

- un `query_id` unique ;
- folds exclusivement dans `{0,1,2,3,4}` ;
- aucune ligne synthétique ;
- zéro fold 1 dans les fichiers de développement ;
- zéro autre exclusion que l'absence des deux cibles humaines ;
- aucune alternative opérationnelle d'un autre SIREN ;
- aucun positif injecté et aucun hit/rang/score utilisé par le builder.

## 5. Gates du prochain cycle

Les métriques principales utilisent tous les `human_ground_truth_siret` :

- oracle exact de l'union interne, plafond 2 000 ;
- Recall@100 SIRET exact, plafond absolu 100.

Avant toute ouverture du fold 1 :

- oracle OOF folds 2/3/4 >= 99,5 % ;
- Recall@100 exact OOF folds 2/3/4 >= 99,2 % ;
- borne basse Wilson à 95 % du Recall@100 OOF >= 99,0 % ;
- Recall@100 exact de chacun des trois folds externes >= 98,8 % ;
- maximum observé de 2 000 puis 100 ;
- latence chaude p95 <= 1 000 ms et p99 <= 2 000 ms ;
- nombres bruts et résultats actif/fermé, origine CRM et géographie publiés ;
- exact et opérationnel publiés côte à côte.

La référence actuelle fold 0, uniquement pour expliquer l'écart à combler,
est 6 919/7 005 à l'oracle 2 000 et 6 654/7 005 au top 100. Elle ne sert pas à
choisir les paramètres v11.

## 6. Budgets gelés de la recette

Le benchmark v11 accompagne une seule recette bornée :

- référence brute maximale de 10 000 candidats, disponible pour mesurer le
  plafond informationnel mais jamais exposée au ranker final ;
- union de travail maximale de 2 000 et sortie maximale de 100 ;
- 700 places au plus pour les expansions de parents SIREN ;
- 32 graines au plus pour les relations officielles ;
- 20 places protégées pour les ancres directes et 20 pour le consensus ;
- trois SIRET au plus d'un même SIREN dans la liste finale de 100.

Un seul rescue contrastif conditionnel est autorisé par requête. Il n'est
appelé que lorsque le retrieval sparse ne fournit pas un soutien parental
suffisant ; il ne peut pas être exécuté systématiquement, répété ou utilisé
pour ajuster un seuil sur fold 0. Ses textes et features restent exempts de
SIRET, SIREN, identifiants et alias CRM.

## 7. Interdictions

- aucun alias CRM→SIREN/SIRET mémorisé ;
- aucune feature identifiant SIRET, SIREN, requête ou annonce ;
- aucune donnée synthétique pour régler le retrieval ;
- aucune source web payante ;
- aucune injection du positif ;
- aucun déplacement post-hoc du dénominateur ;
- aucun nouveau `GO` avant exécution conforme du cycle préenregistré.
