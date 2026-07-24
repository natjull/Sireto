# Retrieval Recall@100 — courbe sparse dev

## Verdict intermédiaire

La baseline sparse actuelle atteint **92,75 % de Recall@100 SIRET**. Passer de
50 à 100 candidats ne gagne que 2,42 points et reste très loin de la cible
99,0 %.

Plus important : le store candidat V7 plafonne à **97,58 %** sur ce benchmark.
La cible 99 % est donc mathématiquement impossible sans corriger la source ou
élargir la génération géographique.

## Contrat

- benchmark : `c33b80855f560074`, split `dev`, 2 565 requêtes ;
- aucun SIREN partagé avec train ou test ;
- classement sparse calculé une fois jusqu'à 500 ;
- classement forcé dès que le pool contient plus de 50 lignes, afin que chaque
  préfixe soit stable ;
- aucune injection de vérité terrain ;
- artefact immuable :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/sparse_curve_dev_c33b80855f560074_ad6fdf3`.

Le préfixe @50 de chacune des 2 565 requêtes est exactement identique à
l'artefact sparse historique, candidats et indicateurs de hit compris.

## Courbe

| K | Bons SIRET | Recall SIRET | IC95 Wilson | Recall SIREN |
|---:|---:|---:|---:|---:|
| 50 | 2 317/2 565 | **90,33 %** | 89,13–91,42 % | 93,45 % |
| 100 | 2 379/2 565 | **92,75 %** | 91,68–93,69 % | 95,48 % |
| 200 | 2 415/2 565 | **94,15 %** | 93,18–95,00 % | 96,53 % |
| 500 | 2 457/2 565 | **95,79 %** | 94,94–96,50 % | 97,66 % |

Pour atteindre 99,0 %, il faut au moins 2 540 hits. La baseline @100 doit donc
récupérer **161 des 186 misses actuels**, tout en restant plafonnée à 100.

## Attribution des pertes

| Étape ou bucket | Requêtes |
|---|---:|
| Hit dans les 50 premiers | 2 317 |
| Rangs 51–100 | 62 |
| Rangs 101–200 | 36 |
| Rangs 201–500 | 42 |
| Absentes du top 500 sparse | 46 |
| Absentes du store/partition V7 | 62 |
| Perdues par filtre métier | 0 |
| Perdues par déduplication | 0 |

Les 62 vérités absentes du store V7 :

- sont toutes des établissements fermés ;
- sont absentes du store DuckDB V7 complet de 14 378 332 candidats ;
- existent toutes dans `StockEtablissement_utf8.parquet`, snapshot de
  42 322 035 établissements dont 25 475 873 fermés.

Il s'agit donc d'une perte lors de la constitution du référentiel candidat, pas
d'une erreur de localisation du runtime.

## Segments à K=100

| Segment | n | Recall@100 SIRET |
|---|---:|---:|
| Actifs | 2 078 | **96,34 %** |
| Fermés | 487 | **77,41 %** |
| Mégapoles | 165 | **86,06 %** |
| Multi-sites | 614 | **91,53 %** |
| CP seul | 39 | **94,87 %** |
| INSEE | 2 526 | **92,72 %** |

Les fermés expliquent 110 des 186 misses @100. Les autres misses comprennent
76 actifs, 23 mégapoles, 52 multi-sites et 2 localisations CP seules ; ces
catégories se recouvrent.

## Coût

- durée warm-cache : 985 s ;
- latence p50 : 143 ms ;
- latence p95 : 1 472 ms ;
- latence p99 : 3 522 ms ;
- le run diagnostique retourne jusqu'à 500 candidats ; il n'est pas éligible
  comme configuration finale.

## Conséquence

Deux chantiers sont obligatoires et distincts :

1. rendre les établissements historiques éligibles depuis le snapshot brut,
   au moins pour les scènes qui les nécessitent ;
2. remplacer le ranking sparse monolithique par une union de canaux capable de
   faire remonter dans le top 100 au moins 99 des 124 vérités présentes dans le
   store mais classées après la position 100.

Le prochain milestone mesure séparément nom-mots, nom-caractères, adresse,
clés exactes, rescue numérique, SIREN et géographie. Aucun modèle aval n'est
modifié.
