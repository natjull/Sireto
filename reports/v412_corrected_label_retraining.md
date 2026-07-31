# V4.12 — réentraînement sur les labels REVIEW corrigés

Date : 31 juillet 2026

## Données corrigées

L'overlay gelé contient 143 dossiers difficiles déjà adjudiqués :

- 133 `MATCH_EXACT` ;
- 10 `AMBIGUOUS` ;
- dont 56 exacts et quatre ambiguïtés issus des 60 REVIEW supplémentaires ;
- aucun dossier du test final ;
- toutes les prédictions des dossiers difficiles sont produites hors
  échantillon sur cinq folds.

Overlay : [`v412_corrected_review_overlay_60.csv`](v412_corrected_review_overlay_60.csv)

## Ranker candidat

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_corrected_label_ranker/9fea31939cff7fea`

| Mesure | Ranker précédent | Ranker corrigé |
|---|---:|---:|
| Dossiers exacts difficiles | 133 | 133 |
| Vérités présentes dans le pool | 131 | 131 |
| Bons top 1 hors échantillon | 69 | **110** |
| Hit@1 | 51,88 % | **82,71 %** |
| Dossiers corrigés | — | 43 |
| Dossiers régressés | — | 2 |

Sur les 1 175 contrôles exacts hors composantes difficiles, les deux rankers
restent à 1 175/1 175. Le gain net est donc de 41 dossiers difficiles, sans
régression observée sur ce contrôle. Les deux misses retrieval restent comptés
comme erreurs end-to-end.

## Accepteur sélectif

Scènes :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_corrected_label_stack/aae2ad5814ecfb5b`

Ablation famille/poids :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_corrected_acceptor_family_weight/c88e443950d188cf`

La régression logistique compacte est excellente sur le développement
classique corrigé (615/669 AUTO, zéro erreur observée), mais elle accepte une
mauvaise décision parmi trois AUTO difficiles hors échantillon. Elle n'est donc
pas sûre.

Le seul candidat respectant les deux contrôles est l'accepteur XGBoost monotone
avec un poids difficile de 10 :

| Population | AUTO | Erreurs AUTO | Ambiguïtés AUTO | Précision observée |
|---|---:|---:|---:|---:|
| Comparaison classique | 602/669 | 0 | 0 | 100 % |
| Difficiles hors échantillon | **3/143** | 0 | 0 | 100 % |

Cette sécurité observée ne constitue pas une certification : trois décisions
difficiles seulement donnent un intervalle beaucoup trop large. Surtout, la
couverture difficile reste à **2,10 %**. Le problème de l'accepteur n'est donc
pas résolu par le nettoyage des labels ni par la pondération.

## Verdict

Verdict technique : **`GO_NEW_INDEPENDENT_ACCEPTOR_DOCKET`** pour mesurer le
candidat sur de nouveaux REVIEW historiques non adjudiqués.

Verdict produit : **pas de déploiement**. Le ranker progresse fortement, mais
l'accepteur reste le goulot d'étranglement de la couverture. Le prochain lot
doit être traité de façon autonome et ne peut servir ni à modifier le modèle ni
à choisir le seuil avant sa mesure.
