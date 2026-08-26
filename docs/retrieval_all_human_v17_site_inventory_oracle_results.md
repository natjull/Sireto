# Retrieval all-human v17 — résultats inventaire exhaustif des sites

Date : 26 août 2026  
Verdict : **`PIVOT_PARENT_DISCOVERY`**

## Expérience

Le pilote exécute uniquement les étapes 1 à 3 du contrat v17 sur la cohorte
scellée de 1 034 requêtes :

1. discovery parent par les 26 canaux texte v16, la géographie v16 et les
   adresses de tous les établissements officiels ;
2. sélection déterministe de 512 parents maximum ;
3. développement exhaustif de tous leurs SIRET, puis oracle exact et
   opérationnel séparé.

Le générateur n'ouvre aucune vérité, n'injecte aucun positif et ne touche pas
fold0/fold1. Il n'applique aucune profondeur par parent avant l'oracle.

Implémentation et contrat : commit `a37bab43`.

## Préflights

- 1 requête : 63 681 sites émis pour 512 parents ; le parent le plus grand
  porte 13 118 sites. Temps 4,172 s, RSS 1,02 Go, zéro swap.
- 16 requêtes : 431 949 sites, 4,807 s, RSS 1,89 Go, zéro swap.
- 23 tests v15/v16/v17 passent avant l'exécution.

## Génération label-blind complète

- 1 034 requêtes ;
- 75 139 signaux parents par adresse ;
- 910 402 présences parent brutes ;
- 1 490 340 lignes de candidats pour les trois variantes gelées ;
- **24 376 136 lignes SIRET développées exhaustivement** ;
- aucune divergence entre `source_site_count` et `emitted_site_count` ;
- temps 41,630 s ; RSS 12 140 265 472 octets ; zéro swapout.

Manifeste :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_site_inventory_v17_generation/69ddd52660b41f68/manifest.json`

SHA-256 :
`94f0715c37d1b3d8229c667205cac8a3062b85666d1d63d9a399bbfdc119965f`

## Contribution de chaque source au bon parent

Sur les 522 pertes v15 :

| Source | Bons parents présents |
|---|---:|
| Texte v16 | 258 |
| Géographie v16 | 131 |
| Adresse de tous les établissements | 47 |
| Adresse uniquement, absent du texte et de la géographie | **13** |
| Union brute des trois sources | **329** |

L'union brute vaut 132/107/90 par fold. Elle ajoute 13 parents au précédent
oracle texte+géo (316), mais reste très sous le minimum 464.

## Variantes parent@512

| Variante | Total misses | Fold 2 | Fold 3 | Fold 4 |
|---|---:|---:|---:|---:|
| Primaire texte256/adresse160/géo96 | **309** | 125 | 98 | 86 |
| Adresse lourde texte192/adresse224/géo96 | 307 | 125 | 96 | 86 |
| RRF texte/adresse/géo | 305 | 123 | 97 | 85 |
| Gate requis | **464** | **160** | **158** | **146** |

La variante primaire préserve les 512 contrôles dans le Master grâce à la base
top1750, même si son propre pool parent contient 503/512 parents contrôle.

## Exact versus opérationnel

La politique stricte même-site trouve :

- 58 requêtes possédant au moins un sibling opérationnel ;
- 25 cas `ACTIVE_SUCCESSOR_SAME_SITE` ;
- 5 cas `MULTIPLE_ACTIVE_EQUIVALENTS`.

Cependant, lorsque le bon parent est sélectionné, l'expansion exhaustive émet
aussi le SIRET exact historique. L'oracle de présence obtient donc ici la même
couverture exacte et opérationnelle : **309 rescues**, soit 125/98/86 par fold.
La valeur opérationnelle devra surtout intervenir au futur classement site :
elle permet de préférer un sibling actif au bon endroit sans le confondre avec
la vérité exacte.

## Projection Master et décision

À partir des 23 049 hits scellés :

- variante primaire : **23 358/23 571 = 99,0963 %** ;
- borne irréaliste avec tous les parents de l'union brute :
  **23 378/23 571 = 99,1812 %**.

La primaire reste à 25 hits de 99,2 % et 96 hits de 99,5 %. Même l'union brute
sans plafond reste à 5 hits de 99,2 % et 76 hits de 99,5 %.

Verdict terminal pour ce sprint : **`PIVOT_PARENT_DISCOVERY`**. Conformément au
contrat, les étapes LambdaMART parent/site et le run national ne sont pas
lancés. L'intuition « tous les SIRET d'un SIREN » est validée pour éliminer le
goulot site et pour identifier des successeurs opérationnels, mais elle ne
compense pas les SIREN encore absents.

Manifeste d'évaluation :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_site_inventory_v17_evaluation/94f0715c37d1b3d8/manifest.json`

SHA-256 :
`eadf53b260b0733f90c64b5269d6e27cd60a7efa2dd6cff31e868e78385107bd`
