# V21 — correction structurelle, sans réglage sur le CRM

Préenregistrement du 9 septembre 2026, avant toute nouvelle métrique.

## Choix et limites

Une seule recette : restaurer la présence de tous les parents trouvés, sans
transformer leur identifiant en pertinence. Un parent présent uniquement dans
v17 conserve son appartenance, avec soutien RRF nul; le canal v17 n'apporte
aucun rang numérique. Tous les parents et tous les CRM suivent la même règle.

Deux propriétés d'intégrité accompagnent cette correction : une preuve
dupliquée au sein d'un même canal ne compte qu'une fois; les ex-aequo parent
et adresse ont le même rang de pertinence (`dense_rank`, sans identifiant).
SIREN/SIRET ne servent qu'aux jointures, regroupements, déduplication et au
départage final exact au plafond. Ce départage peut modifier l'identité d'un
candidat parmi des scores parfaitement égaux; il ne modifie jamais son score.

Le score v20 reste inchangé : soutien direct RRF +
`1/(60 + rang_parent + rang_adresse_dans_parent)` pour les sites locaux.
Plafonds 2000/1750, géographie, critères adresse, constante 60 et états sont
hérités, non réoptimisés. Aucun nouveau poids, seuil, secteur privilégié,
commune privilégiée, alias CRM, identifiant ou liste de dossiers à sauver.
Cette étape ne tente donc pas encore de résoudre le défaut d'allocation des
places par une nouvelle formule ajustée aux 112 évictions connues.

## Séparation de la génération et des vérités

La génération ne lit que les surfaces v20 sans labels, les canaux v18 et le
snapshot officiel des sites. Projections explicites et SHA des sources.
Les rangs de base proviennent du raw v11 conservé dans le réservoir v20,
jamais du résultat exact d'une requête. La génération ne lit aucun artefact
d'évaluation, même si sa référence figure dans la configuration commune.

Après scellement, un programme séparé compare le résultat aux vérités,
à l'attribution officielle et à l'ensemble opérationnel scellés par v20.
Exact, opérationnel, historique, V2/V3, folds, origines et régressions sont
publiés ensemble. Les qualifications vides restent visibles. La preuve
opérationnelle hérite des limites v20, dont les compléments non mesurés;
aucune précision top1 n'est déduite d'une présence dans le Master.

## Protection contre le surapprentissage

Les 1034 dossiers ont déjà été observés dans les versions précédentes.
Ni leur redécoupage ni le mot OOF ne les rendraient vierges. La recette
déterministe n'ajuste aucun paramètre sur eux, mais son choix est informé par
les diagnostics précédents : le résultat est **développement réutilisé**,
jamais une estimation non biaisée de généralisation.

Avant résultats, tests de propriétés sur fixtures artificielles de logiciel
(jamais données d'entraînement ni features) : invariance des scores aux
identifiants, ordre des lignes, doublons de preuve et présence de colonnes
interdites; isolation des requêtes; conservation du préfixe et plafond;
survie des témoins directs hors géographie; absence d'adresse non assimilée
à un accord. Les tests ne prouvent pas à eux seuls la qualité du retrieval.

Pas de grille, pas de meilleure variante choisie après mesure, pas de
nouveau cycle automatique après un mauvais résultat. Les gains ET les pertes
par origine et fold sont publiés, même si l'agrégat augmente. Les 512
contrôles sont présélectionnés comme anciens hits, pas un échantillon national.

Pour une preuve cross-CRM future, il faudra un export réellement nouveau,
indépendant des sources de développement, scellé avant accès aux labels :
échantillonnage indépendant du succès du retrieval, audit des doublons
requête/entreprise/site, publication des segments déjà vus/nouveaux et des
absences, recette entièrement gelée, une seule évaluation sans retuning.
Si les données permettent un apprentissage futur, les inner-splits doivent
grouper entreprises/sites; les seuils restent internes, les folds externes
ne servent pas à les choisir. Aucun nouveau CRM non consommé n'est identifié
dans ce protocole; la preuve externe est déclarée manquante, pas fabriquée.

## Exécution et décision

Préflight label-blind 1..16 puis un seul workbench 1034. Mac local,
30 minutes maximum par phase, RSS <=14 Gio, zéro swapout, aucun coût externe.
Pas de national ni de fold0/fold1; le « Go » de ce sprint n'est pas une
autorisation d'ouverture du fold1. Aucun modèle aval n'est dégelé.

Le Master ne prouve ni PreRank500 >=99,4 %, ni Recall100 >=99,2 %, ni Wilson,
ni les SLO. Le verdict reste `PIVOT_GENERATION` tant que ces étapes ne sont
pas établies; tout défaut de population, de hash ou d'attribution impose
`STOP`. La projection nationale reste explicitement conditionnelle.
