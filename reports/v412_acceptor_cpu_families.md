# V4.12 — accepteur CPU et politique North Star

## Périmètre corrigé

La comparaison réutilise les `104` features et les scènes role-aware de
`v4_12_acceptor_business_competition/35b7ca460456a40b`. La population difficile
comprend bien les `279` dossiers disponibles à l’inférence : `241` exacts,
`31` ambigus et `7` non résolus. Le ranker a un top 1 correct sur `227` des
241 exacts ; les 14 erreurs de ranker, 31 ambiguïtés et 7 non-résolus sont donc
les `52` négatifs de l’accepteur.

Pour chaque modèle, le résultat principal est un nested component-OOF : un
dossier externe n’est présent ni dans l’apprentissage du modèle, ni dans la
calibration de son seuil. Le test final n’a pas été ouvert.

## Modèles seuls — gate prudent

| Famille | AUTO corrects / 227 | Acceptation | Erreurs AUTO | dont non-résolus |
|---|---:|---:|---:|---:|
| XGBoost sans contraintes | **105/227** | **46,26 %** | **0** | **0** |
| ExtraTrees, feuille 10 | 117/227 | 51,54 % | 1 | 1 |
| ExtraTrees, feuille 5 | 115/227 | 50,66 % | 1 | 1 |
| ExtraTrees, feuille 3 | 107/227 | 47,14 % | 1 | 1 |
| Random Forest, feuille 3 | 63/227 | 27,75 % | 0 | 0 |
| Random Forest, feuille 5 | 61/227 | 26,87 % | 1 | 0 |
| Random Forest, feuille 10 | 54/227 | 23,79 % | 1 | 0 |

Aucune famille seule n’atteint `148/227` sans erreur. Le meilleur modèle sûr
est donc XGBoost sans contraintes. ExtraTrees accepte davantage de bons cas,
mais son erreur sur un dossier `UNRESOLVED` l’écarte du chemin prudent.

## Politique North Star explicable

Le candidat final de développement est l’union de ce XGBoost OOF et de huit
preuves directes, combinées par `OR` :

- avantage de similarité du nom légal ≥ `0,43` ;
- avantage de rôle métier ≥ `1` ;
- avantage de contenance du nom CRM ≥ `1` ;
- avantage de recouvrement des mots d’adresse ≥ `0,21` ;
- avantage IDF du nom ≥ `0,35` ;
- rôle explicite porté par le top 1 ≥ `1` ;
- avantage d’acronyme ≥ `1` ;
- avantage de date de début à la même adresse ≥ `0,01`.

Ces seuils arrondis sont directionnellement justifiables, mais ont été choisis
sur le développement consommé. Ils ne constituent donc pas une validation
indépendante.

### Résultat observé

- dossiers difficiles : `149/227` bons top 1 AUTO, soit `65,64 %`, avec
  `0/52` négatif AUTO ;
- contrôles positifs figés : `1 124/1 127` AUTO, sans erreur observée ;
- combiné : `1 273/1 406` AUTO, soit `90,54 %` de couverture globale ;
- précision observée : `1 273/1 273`, soit `100 %` ;
- couverture des `1 368` dossiers identifiables : `93,06 %` ;
- acceptation des `1 354` top 1 corrects disponibles : `94,02 %`.

Les deux gates de développement sont franchis :

- prudent : au moins `148/227` avec zéro erreur sur les 52 négatifs ;
- North Star : acceptation difficile entre 65 et 75 %, couverture globale
  entre 88 et 92 %, précision observée ≥ 99,8 %.

## Verdict et limites

Verdict : `GO_NORTH_STAR_DEV_ZERO_ERROR`.

Cela autorise à figer cette politique pour une évaluation réellement nouvelle.
Cela n’autorise ni un déploiement, ni une revendication de précision certifiée :
les 279 dossiers difficiles ont servi au choix de la famille, du seuil et des
règles, et les 1 127 contrôles ne contiennent que des top 1 corrects.

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_cpu_families/7270609bd3d59376`.
Il contient le modèle, les décisions individuelles avec noms CRM et raisons,
les comparaisons nested OOF, les contrôles et un manifeste de hashes.

