# V4.12 — bundle trusted-label figé

Bundle :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/bundles/v4_12_trusted/c2a01c6bca43a468`.

Le manifeste lie par hash :

- le ranker trusted-label poids `0,5` ;
- l'accepteur XGBoost monotone poids `10` ;
- le seuil `0.9886879324913025` ;
- les ordres des 45 features ranker et 80 features accepteur ;
- la taxonomie de fonction de site ;
- le dataset, les 279 labels fiables et les sources d'entraînement ;
- le retrieval V4.2, plafond 100 et absence d'injection positive.

Le bundle déclare explicitement `final_test_opened=false` et
`production_promotion_authorized=false`. Les hashes des trois artefacts copiés
ont été revérifiés après écriture.

La prochaine preuve obligatoire est `NEW_INDEPENDENT_CRM_HOLDOUT`. Aucun ancien
holdout, random ou challenge consommé ne peut être réutilisé.
