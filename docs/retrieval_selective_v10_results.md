# Rétractation — diagnostic retrieval sélectif v10 sur fold 0

## Statut corrigé

Le `GO Recall@100` publié précédemment est **rétracté**. Le résultat
5 649/5 700 = 99,1053 % est conservé pour l'audit, mais il a été obtenu sur un
sous-ensemble défini après plusieurs lectures de fold0. Il ne peut donc pas
servir de résultat principal ni franchir le gate.

Statut exact :
**`RETROSPECTIVE_SELECTIVE_DIAGNOSTIC_INVALID_FOR_PRIMARY_GATE`**.

Fold1 et le test sont restés fermés. Aucun nouvel entraînement, calcul ou
ajustement n'a été exécuté pour cette rétractation.

## Métriques principales corrigées

| Périmètre | Succès / total | Recall@100 | Statut |
|---|---:|---:|---|
| Labels humains utilisables | 6 654 / 7 005 | **94,9893 %** | métrique principale |
| Ancien périmètre qualifié exact | 6 085 / 6 307 | **96,4801 %** | vue secondaire historique |
| Sous-ensemble sélectif post-hoc | 5 649 / 5 700 | **99,1053 %** | diagnostic non admissible au gate |

Le maximum observé reste 100 candidats. Ce contrôle d'intégrité ne transforme
pas le diagnostic post-hoc en validation scientifique.

## Explication auditable des dénominateurs

- **7 009** est le nombre total de scènes humaines de fold0.
- **7 005** disposent d'un SIRET humain historique exploitable. Les quatre
  autres sont des lignes `AMBIGUOUS` sans `ground_truth_siret` ni
  `historical_ground_truth_siret` : `10395` (`PROMOTRANS LYON`), `9115`
  (`SANDVIK MINING AND CONSTRUCTION`), `fresh:FR033088`
  (`DECO INVESTISSEMENTS`) et `fresh:FR037897` (`Innergex`).
- **6 307** ont été marquées `MATCH_EXACT` par la qualification antérieure.
  Les 702 autres sont 581 `UNRESOLVED` et 121 `AMBIGUOUS`. Pour 698 d'entre
  elles, le SIRET humain reste présent dans `historical_ground_truth_siret` ;
  elles ne sont pas des labels humains moins valides.
- **5 700** est un sous-ensemble des 6 307. La qualification v4 en a exclu
  607 supplémentaires parce que le CRM et le dossier officiel présentaient
  moins d'accord direct. Ces 607 labels ne sont pas moins valides : le top100
  en retrouvait 436 et en manquait 171.

Les 1 309 lignes hors sous-ensemble post-hoc sont donc `702 + 607`. Aucune n'a
été supprimée des artefacts. Elles ont uniquement été retirées du dénominateur
du chiffre 99,1053 %, ce qui rend ce chiffre impropre comme mesure principale.

## Chronologie réelle de l'utilisation de fold0

1. Une première admission v8 a été lue sur fold0 : 5 805/6 307
   (92,0406 %).
2. Après correction de la compression des rangs, l'admission v9 a été lue sur
   fold0 : 5 826/6 307 (92,3736 %).
3. Une recette fortement régularisée a ensuite été choisie par validation
   croisée sur folds 2/3/4, mais cette étape intervenait déjà après observation
   des échecs v8/v9 sur fold0. Son évaluation v10 a produit 6 085/6 307
   (96,4801 %).
4. Après lecture de ce résultat v10, une première vue directe nom OU adresse a
   été mesurée sur fold0 : 5 466/5 510 (99,2015 %), avec seulement 78,6132 %
   de couverture.
5. Plusieurs règles conjointes nom/adresse ont ensuite été inspectées sur
   fold0. La règle finalement matérialisée ajoute les seuils
   `name_jaro_winkler >= 0.55`, `name_token_jaccard >= 0.10`,
   `address_jaro_winkler >= 0.50` et exige un numéro compatible.
6. Cette règle a été contrôlée a posteriori sur folds 2/3/4
   (80,5401 % de couverture), puis relue sur fold0, où elle donne
   5 649/5 700 (99,1053 %).

L'affirmation antérieure selon laquelle les seuils de qualification v4 avaient
été choisis uniquement sur folds 2/3/4 était donc fausse. Fold0 a influencé la
construction de la règle avant sa publication.

## Limite d'usage de la qualification v4

La qualification v4 consulte le SIRET humain, puis ouvre le dossier officiel
de ce SIRET pour mesurer l'accord nom/adresse. Elle n'utilise aucun hit, rang
ou score du retrieval, mais elle connaît tout de même le label humain.

Elle peut servir d'**audit rétrospectif d'identifiabilité**. Elle ne peut pas
être présentée comme un mécanisme de routage déployable : en production, le
vrai SIRET n'est précisément pas connu avant le retrieval.

## Conclusion corrigée

- Aucun `GO` ni `PASS` Recall@100 n'est annoncé.
- Le résultat principal courant reste **6 654/7 005 = 94,9893 %** sur les
  labels humains utilisables.
- Le résultat **6 085/6 307 = 96,4801 %** reste la vue de l'ancien périmètre
  exact.
- Le résultat **5 649/5 700 = 99,1053 %** est conservé uniquement comme
  diagnostic post-hoc.
- Fold1, le test et les modèles aval restent fermés.
- Toute stratégie suivante devra être préenregistrée et approuvée avant
  nouveau calcul ou ouverture de fold.

## Artefacts conservés

- correctif rangs + recette LambdaMART : commit `4f36ea6` ;
- qualification rétrospective : commit `7261f83` ;
- publication initiale désormais rétractée : commit `927c47f` ;
- modèle : `models/retrieval_ltr_v10_oof_shallow3`, ranker SHA-256
  `ae1632831717c61dfd0aac56f71660942f495cc1c40082de5d0b0ef1295a0920` ;
- qualification : `experiments/retrieval_selective_v4/qualification_development.parquet`,
  SHA-256 `5a947285b4a3451952e804a735a71e09773fbd8fbafbee796342472bf54965fc` ;
- diagnostic : `experiments/retrieval_selective_v4/dev0_v10_summary.json`,
  SHA-256 `1ffe12e59e7a99ae6b113ee88bfff84b8f4c3e7bcdd3bda6a554be18a4d379b5`.
