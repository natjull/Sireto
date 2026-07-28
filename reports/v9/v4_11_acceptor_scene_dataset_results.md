# V4.11 — Dataset de scènes accepteur

## Verdict

**`GO_FREEZE_V411_ACCEPTOR_PLAN`**

Le dataset compact de l'accepteur est construit à partir des prédictions OOF
du ranker C sur le fit et de son modèle complet sur le dev. Aucun accepteur,
score de confiance ou seuil n'a encore été entraîné ou choisi.

Artefact initial, désormais supersédé :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_acceptor/e9570f621216f3fd`

Artefact corrigé et retenu :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_acceptor/52ea3faba9a56aff`

L'ancien build `e9570f621216f3fd` est supersédé : ses scènes sont valides,
mais son manifeste n'épinglait pas l'implémentation transitive
`v49_site_function.py`.

## Volumes

| Population | Total | `MATCH_EXACT` | `AMBIGUOUS` |
|---|---:|---:|---:|
| Fit OOF | 5 547 | 4 666 | 881 |
| `threshold_dev` | 710 | 583 | 127 |
| `comparison_dev` | 746 | 634 | 112 |
| Total | 7 003 | 5 883 | 1 120 |

Les cibles query-level contiennent :

- 5 877 top-1 exacts corrects, cible 1 ;
- six top-1 faux sur des labels exacts, cible 0 ;
- 1 120 ambiguïtés, cible 0.

Les six erreurs exactes correspondent aux cinq erreurs OOF fit et à l'unique
erreur dev du ranker C. Elles restent dans les scènes ; aucune n'est retirée
ou transformée.

## Contrat de features

- 80 features uniques et finies ;
- 34 binaires non standardisées pour la logistique ;
- 46 continues ou compteurs standardisés sur le fit uniquement ;
- vecteur monotone : 49 contraintes positives, six négatives et 25 nulles ;
- calcul query-level à partir du top-1, du top-2, de la distribution des
  scores, des agrégats SIREN, des preuves candidat et des rôles/NAF ;
- aucun `input_siret`, `input_siren` ou signal dérivé dans l'ordre modèle.

## Intégrité

- hash du parquet scènes conforme :
  `c0f3d670e50cb43cdd6fed3b976c95e51d70f0313ae07ed0e1e2ed01eca5bed3` ;
- zéro composante SIREN partagée entre fit et dev ;
- zéro composante partagée entre `threshold_dev` et `comparison_dev` ;
- les 5 547 scènes fit proviennent uniquement de `ranker_c_oof` ;
- les 1 456 scènes dev proviennent uniquement de `ranker_c_dev` ;
- toutes les prédictions ranker sont hors-échantillon pour la scène
  concernée ;
- le manifeste épingle exactement le retrieval `ec4326ec57e4411d`, le ranker
  `e13eb3ac7498256e`, ses 698 892 prédictions, le contrat, la taxonomie,
  `v411_scene.py` et `v49_site_function.py` ;
- hash du manifeste corrigé :
  `8faaf2761bb280f1ba559ea3f2c579fd5d91531a202b6b54dff79e38f0d2757e` ;
- le parquet corrigé est bit-à-bit identique au parquet supersédé ;
- le validateur officiel de l'artefact passe ;
- le contre-audit indépendant conclut `GO_FREEZE_PLAN` ;
- aucun jeu random, holdout, test final ou challenge inédit n'a été ouvert.

## Prochaine étape autorisée

Geler dans Git un plan d'entraînement contenant :

- les hashes du dataset et des scènes ;
- l'ordre des 80 features et le vecteur monotone ;
- exactement `COMPACT_LOGIT` et `MONOTONIC_XGB` ;
- la politique de seuil à 99,8 % et le gate de couverture à 80 % ;
- les versions du runtime ;
- la baseline V4.1 épinglée.

Un verrou d'exécution externe devra ensuite lier ce plan au commit exact du
runner avant le premier fit.
