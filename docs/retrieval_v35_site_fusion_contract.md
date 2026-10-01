# V35 — fusion apprise parent/site, pilote borné

Préenregistrement du 1 octobre 2026, avant fit ou lecture de métriques du
pilote. Cette expérience n'ouvre ni fold 0 ni fold 1, ne touche pas aux modèles
aval et n'est pas un gate produit. Un seul modèle et une seule recette ; pas
de grille ni de sélection d'hyperparamètres sur les folds externes.

## Hypothèse et limite

Les indices nom/adresse/numéro/géographie doivent être conservés séparément :
une discordance de numéro n'annule pas automatiquement la preuve de nom et
de voie. Un LambdaMART SIRET apprend leur combinaison sur le Master naturel
de 2 000, produit un ordre unique, dont les préfixes 500 puis 100 sont publiés.
Pas de quota par SIREN, de protection de préfixe, ni de positif injecté.
Le plafond exact du Master v30 complet reste 98,96059 % ; ce travail ne peut
donc pas, seul, atteindre 99,2 % ou le gate 99,4 % du PreRank@500.

## Population indépendante des résultats

Les 32 premiers lots de 32 requêtes du manifeste de génération scellé v34
sont retenus : 1 024 requêtes en ordre SHA256 `v33-cost|query_id`, issu du
générateur. Sélection préexistante, sans labels, qualification, hit ou rang
de vérité. Toutes sont évaluées, y compris les vérités absentes. Les résultats
ne sont pas une estimation certifiée du complet et ne peuvent autoriser le
fold 1. Historique, V2, V3, folds et vues exacte/opérationnelle sont publiés.

## Cloisonnement nested-OOF réel

Pour le modèle SIRET évalué sur F, apprentissage sur les deux autres folds.
Les Masters d'apprentissage sont RECONSTRUITS à partir du réservoir officiel
label-blind, avec le parent `inner_trainG` où G exclut à la fois F et le fold
de la requête. Par exemple, outer2 : requêtes fold3 via inner_train4,
requêtes fold4 via inner_train3. Les anciens Masters outer3/4 ne peuvent pas
servir à cet entraînement : leurs upstream modèles ont vu des labels fold2,
ce qui affecte aussi la sélection des candidats, pas seulement les scores.
L'évaluation utilise outerF et exige une parité exhaustive de son Master
avec v30 scellé (identités, rangs, scores et preuves). Les hashes des modèles
et des splits sont vérifiés ; leurs folds et identités d'apprentissage sont
audités. Le pilote ne démontre pas la généralisation à d'autres CRM.

## Preuves et entraînement

Uniquement sources officielles existantes. Noms parent/site non recopiés aux
siblings ; accords textuels, numéro séparé et écart numérique, suffixe,
localité, état/siège, sources, ambiguïté du parent, rang upstream. Le numéro
et le code géographique bruts, identifiants et textes CRM bruts ne sont pas
des features. Les distances et incertitudes disponibles dans les voisins
BAN préexistants sont utilisées avec un indicateur de disponibilité ; un
non-voisin n'est jamais assimilé à une distance infinie. Pas de nouvel appel
BAN et pas de feature synthétique.

Features calculées sur le pool complet avant lecture des labels ou sampling.
Après scellement, entraînement sur positifs exacts naturellement présents,
64 premiers négatifs et 64 négatifs de strates régulières du reste. Les
équivalents opérationnels prouvés sont neutralisés, jamais négatifs, jamais
ajoutés comme positifs exacts. Groupes sans positif présents publiés et
exclus du fit uniquement, conservés dans tout dénominateur de mesure.

XGBoost local 3.1.3 CPU `rank:ndcg`, paires `mean`, 8 paires/échantillon,
200 arbres, profondeur 5, learning_rate 0,06, lambda 5, hist, max_bin 128,
subsample et colsample 1, seed 20261001, 2 threads, pas d'early stopping.
Le NDCG est un surrogate de classement, PAS la métrique métier ; mesurer
directement Recall exact @100/500/2000, pertes, gains/pertes appariés et Wilson.

## Coût et traçabilité

Un cache truth-blind unique projette les sites/noms nécessaires depuis les
sources scellées ; pas de rescan national par requête. DuckDB 2 threads,
3 Go de mémoire, spill sur CATNAT ; garde 6 Gio RSS, zéro nouveau swapout,
20 minutes pour l'expérience entière. Premier lot de 32 = contrôle technique
de parité et de coût, sans métrique qualité. Continuer les 31 lots suivants
uniquement si cette reconstruction prend au plus 30 s (hors cache commun).
Artefacts checkpoints nouveaux et immuables ; ancienne génération inchangée.
La source est isolée dans un nouveau script pour ne pas modifier l'inventaire
scellé `src/xgb_matcher/*.py` de v33. Pas de run complet automatique.

## Fondements, sans promesse SOTA

- [Ditto, article original](https://arxiv.org/abs/2004.00584) : preuves
  hétérogènes et difficulté des paires d'entity matching ; le présent modèle
  CPU n'est pas Ditto et n'hérite pas de ses performances.
- [XGBoost, learning to rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html) :
  LambdaMART groupé par requête et objectifs de classement. La sélection
  d'un modèle repose ici sur une recette préenregistrée, pas sur ses métriques
  externes.

Issue : résultat pilote avec gains/pertes et coût, puis orientation explicite
`PIVOT_GENERATION` (plafond Master insuffisant), complétée de l'état du
classement. Jamais `READY_FOR_FOLD1` à partir de cet échantillon.
