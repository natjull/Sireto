# V4.12 — validation aveugle de l'accepteur clean-target

Date : 31 juillet 2026

## Labels métier

| Verdict | Nombre |
|---|---:|
| `MATCH_EXACT` actif courant | **28** |
| `AMBIGUOUS` | **2** |
| `UNRESOLVED` | **0** |

Les labels et preuves ont été produits avant ouverture des scores dans
[`v412_clean_target_independent_labels_30.csv`](v412_clean_target_independent_labels_30.csv).

## Candidat figé

- ranker corrigé, poids `0,5` ;
- accepteur XGBoost monotone clean-target ;
- poids difficile `10` ;
- seuil `0.9940522313117981` ;
- aucun ajustement après l'audit.

## Résultats indépendants

| Mesure | Résultat |
|---|---:|
| Bons top 1 du ranker | **24/28 (85,71 %)** |
| `AUTO_MATCH` | **13/30 (43,33 %)** |
| AUTO corrects | **13/13** |
| Erreurs AUTO | **0** |
| Ambiguïtés AUTO | **0** |

Les quatre erreurs du ranker sont FNAC Carré Sénart, Audika Gennevilliers,
Capstone Limonest et Croix-Rouge Grabels. Elles restent toutes en REVIEW.

Le détail ligne par ligne est dans
[`v412_clean_target_independent_results_30.csv`](v412_clean_target_independent_results_30.csv).

## Interprétation

La cible nettoyée corrige bien le défaut d'abstention totale observé avec le
même modèle sur le lot précédent. Le candidat automatise ici 13 cas sans
erreur, tout en rejetant les deux ambiguïtés et les quatre erreurs du ranker.

Ce résultat reste une estimation de développement consommé, pas une
certification : avec seulement 13 AUTO sans erreur, la borne statistique est
très loin de 99,8 %. Le candidat reste figé et doit être mesuré sur les 69
REVIEW historiques non adjudiqués restants avant toute nouvelle décision
d'architecture.

Verdict : **`GO_EXTEND_BLIND_MEASUREMENT`**, sans déploiement et sans ouverture
du test final.
