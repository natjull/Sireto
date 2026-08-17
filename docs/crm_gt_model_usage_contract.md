# Contrat d'utilisation des GT CRM humains

## Population de référence

La population `crm_gt_v2_commercial_certified_population/4b07f3b3d245358e`
contient **37 218 labels humains** : 17 097 historiques et 20 121 nouveaux.
Le SIRET a été saisi dans le CRM par un assistant commercial lors de la
création du site. Il reste la vérité SIRET exacte d'origine. Les contrôles
automatiques vérifient l'existence SIRENE et la cohérence de commune par INSEE,
avec repli postal lorsque l'INSEE manque ; ils ne réadjudicent pas le label.

Une revue LLM ou un modèle ne peut pas modifier ou supprimer seul un label
humain. Il peut uniquement produire un signal de suspicion destiné à une revue
humaine séparée.

## Rôles immuables pendant la sélection des modèles

Les composantes SIREN sont indivisibles entre folds afin d'empêcher toute fuite
d'identité :

| Rôle | Folds | Lignes | Usage |
|---|---:|---:|---|
| Apprentissage | 2, 3, 4 | 23 587 | Ajustement des paramètres |
| Développement prospectif | 0 | 7 009 | Choix de méthode, hyperparamètres et seuils |
| Test final | 1 | 6 622 | Une seule mesure après gel complet |

Les 37 218 lignes sont donc toutes valorisées. Une ligne de développement ou
de test apporte une mesure indépendante plutôt qu'un gradient d'entraînement.
Elle n'est pas perdue.

## Application aux modèles

Le même contrat s'applique à **XGBoost, BGE, CamemBERT et au modèle de fusion** :

- XGBoost n'apprend que sur les folds 2/3/4 pendant la sélection ;
- BGE et CamemBERT ne voient aucune requête ni identité des folds 0/1 durant
  leur entraînement ou leur minage de négatifs ;
- les siblings du même SIREN et du même site ne sont jamais des négatifs ;
- le fusionset d'apprentissage utilise uniquement des prédictions strictement
  out-of-fold des modèles sources, jamais leurs scores in-sample ;
- le fold 0 sert à choisir la fusion et ses seuils ;
- le fold 1 reste fermé jusqu'au choix définitif et n'est ouvert qu'une fois.

Les 20 000 exemples synthétiques sont une augmentation d'apprentissage
séparée, pondérée et traçable. Ils n'entrent jamais dans les jeux humains de
développement ou de test.

## Deux vérités publiées, sans réécriture du label

La métrique principale conserve le SIRET saisi comme vérité exacte. Une vue
opérationnelle secondaire accepte également un autre SIRET du même SIREN au
même site physique, conformément à
`docs/siret_operational_equivalence_policy.md`. Les deux métriques sont toujours
publiées séparément. Une équivalence opérationnelle n'écrase jamais le label
historique.

## Réentraînement de production

Après sélection sur le développement et évaluation unique sur le test :

1. un modèle de production peut être réentraîné sur train + développement,
   soit **30 596 lignes** ;
2. il peut ensuite être réentraîné sur les **37 218 lignes** si l'on accepte de
   consommer le holdout actuel ;
3. dans ce dernier cas, les métriques officielles restent celles du modèle gelé
   évalué avant le refit, et les prochaines données CRM doivent constituer un
   nouveau holdout prospectif.

Il est interdit de présenter une évaluation sur des lignes déjà utilisées pour
ajuster le modèle comme une performance indépendante.
