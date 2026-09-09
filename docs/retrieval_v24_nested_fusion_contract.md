# V24 — fusion calibrée uniquement en inner-OOF

Préenregistrement du 9 septembre 2026, avant mesure de cette recette.
Une expérience, sans grille, sans correction des 29 erreurs connues de v23.

## Hypothèse et recette figée

Le classement parent appris et le classement retrieval peuvent apporter des
preuves complémentaires. Réutiliser exactement le réservoir naturel v21,
les features et les six modèles parent v23 scellés. Aucun nouvel entraînement
LambdaMART, aucune source ou requête supplémentaire, aucun quota par dossier.

Pour chaque fold externe F, calibrer un unique poids `a` du canal appris,
sur les deux autres folds seulement. Chaque prédiction de calibration doit
elle-même provenir d'un modèle entraîné sur l'autre fold interne :

| Cible externe | Calibration | Modèles de calibration |
|---|---|---|
| 2 | 3, 4 | train4→3, train3→4 |
| 3 | 2, 4 | train4→2, train2→4 |
| 4 | 2, 3 | train3→2, train2→3 |

Dans chaque requête, convertir séparément les rangs parent retrieval et
appris en distributions : `p(parent) ∝ 1/(60 + dense_rank)`. Ces rangs ne
contiennent aucun identifiant comme preuve. Calibrer `a` par minimisation
convexe de la moyenne `-log((1-a)*p_retrieval(vrai_parent) +
a*p_appris(vrai_parent))`, avec barrière fixe
`-0.01*(log(a)+log(1-a))`. Résoudre la dérivée monotone par 80 dichotomies
sur (0,1), pas par essais de seuils sur le recall. Une requête = un poids.
La barrière symétrique évite la suppression totale d'un canal; son coefficient
est un choix de protocole, non validé par une grille. Les requêtes sans parent
positif naturel ne peuvent contribuer à cette vraisemblance conditionnelle;
elles sont comptées comme non calibrables et restent dans l'évaluation entière.

Sur le fold externe, appliquer ce seul poids aux preuves site déjà définies :
`direct_rrf + (1-a)*(score_v22-direct_rrf) +
a*(site_local ? 1/(60+rang_parent_externe+rang_site_conjoint_v23) : 0)`.
Classer globalement; départager par direct_rrf, score parent appris, SIRET
terminal seulement. Pas de somme doublant la preuve directe. Sortie ≤2000
SIRET par requête, sans préfixe protégé. Le poids est appris au niveau **parent**,
pas sur les labels SIRET du grand réservoir. Aucun modèle SIRET ne score ce
réservoir; aucun aval ni hydratation riche dégelé.

La vraisemblance parent est un **objectif de substitution**, pas le recall
SIRET@2000 : son transfert au mélange des preuves site est précisément
l'hypothèse à tester. Le poids inner provient de modèles entraînés sur un seul
fold, appliqué à un modèle externe entraîné sur deux : autre limite explicite.

## Intégrité et décision

Vérifier les hashes, les six splits, l'absence de labels cibles dans chaque
calibration, les populations et les plafonds. Tests de perturbation des labels
cibles, permutation des lignes, optimum convexe et absence de positif naturel.
Commit du protocole/code avant le run complet. Génération puis évaluation
séparée; publier gains **et** pertes contre v23 et tous les segments exacts /
opérationnels, historique / V2 / V3, avec sidecar officiel scellé.

Pas de choix oracle v22/v23 selon le résultat d'un dossier. Pas de deuxième
poids ou autre variante après lecture des résultats. Pas de national ni de
projection : les 1034 requêtes sont un panel biaisé et déjà réutilisé (522
échecs initiaux +512 contrôles). OOF ne supprime pas l'adaptation historique
de la stratégie. Validation cross-CRM seulement avec un export indépendant.

Fold0 interdit au développement, fold1 fermé. Mac/CATNAT, 30 min par phase,
RSS14 Gio, zéro swapout, temporaires externes. Verdict `PIVOT_GENERATION`
tant que les gates PreRank500/Recall100/Wilson/SLO ne sont pas établis;
défaut d'intégrité ou d'attribution = `STOP`. Aucun gain Master ne vaut gate.
