# Retrieval parent v18 — résultats du workbench de découverte

Date : 26 août 2026

## Verdict

`PASS_MASTER_99_2_RAW_FEASIBILITY_ONLY`

Le pool brut v18 retrouve le bon SIREN parent pour 389 des 522 pertes Master
de la cohorte de développement, contre 329 pour l'union brute v17. La
projection oracle correspondante est 23 438 / 23 571, soit 99,4357 %.

Ce résultat démontre que la cible 99,2 % n'est plus bloquée par la seule
découverte du parent. Il ne démontre pas encore un retrieval Master@2000 : le
pool v18 contient une médiane de 2 522 parents distincts par requête
(p95 3 773,5; maximum 5 713), au-dessus du plafond de 2 000. Aucune sélection,
aucun modèle et aucune expansion site n'ont été évalués dans ce workbench.

La cible oracle 99,5 % n'est pas franchie : elle demandait au moins 405
parents retrouvés sur 522. Le niveau de confiance préenregistré de 464, avec
les minima 160/158/146 par fold, n'est pas franchi non plus.

## Protocole anti-fuite

- 1 034 requêtes scellées : 522 pertes Master et 512 contrôles déterministes;
- folds 2/3/4 uniquement; folds 0/1 fermés;
- génération label-blind : seules les colonnes `query_id` et `fold` sont
  ouvertes;
- le SIREN est lu dans le champ officiel du document SIRET, jamais obtenu en
  tronquant le SIRET;
- vérité ouverte seulement dans l'évaluateur séparé;
- aucun positif injecté, aucun modèle entraîné, aucun filtrage post-hoc;
- résultat de développement same-CRM uniquement : aucune revendication de
  généralisation à un autre CRM, client ou millésime.

## Résultats par variante

| Variante | Parents / 522 | Fold 2 | Fold 3 | Fold 4 | Projection Master |
|---|---:|---:|---:|---:|---:|
| union brute v17 | 329 | 132 | 107 | 90 | 99,1812 % |
| + pool SIRET historique v11 | 377 | 145 | 123 | 109 | 99,3848 % |
| + relations structurées | 384 | 146 | 127 | 111 | 99,4145 % |
| v18 complet | **389** | **148** | **130** | **111** | **99,4357 %** |
| SIRET-native seul | 291 | 102 | 99 | 90 | 99,0199 % |

Les 512 contrôles sont tous retrouvés par le v18 complet. L'union brute v17
n'en retrouvait que 504 dans cette projection reconstruite.

## Stabilité récente / historique

| Origine | Dénominateur | v17 | v18 complet | Gain |
|---|---:|---:|---:|---:|
| `REAL_CRM_20260817` | 389 | 256 | 295 | +39 |
| `REAL_CRM_HISTORICAL` | 133 | 73 | 94 | +21 |

Le gain existe dans les deux origines internes, ce qui est préférable à une
amélioration portée uniquement par le sous-ensemble historique. Cela reste un
test dans une seule famille de CRM : les folds groupés par SIREN évitent de
mémoriser la même entité entre folds, mais ne simulent pas un changement de
client, de formulaire ou de période.

## Contributions brutes diagnostiques

Par rapport aux requêtes déjà couvertes par v17, les principaux canaux
retrouvent individuellement :

- pool SIRET historique : 48 pertes nouvelles;
- recherche SIRET adresse caractère : 24;
- recherche SIRET adresse mots : 18;
- nom courant caractère : 11;
- continuité économique SIRET : 7;
- adresse de support : 7;
- BM25 fieldé : 6;
- nom de site : 6;
- relation RNE sujet vers lié : 5;
- adresse historique : 4;
- cession BODACC : 2.

Ces nombres se recouvrent et ne doivent pas être additionnés. Après union de
toutes les familles, le gain net est +60.

## Ressources et autorités

- génération : 685,126 s, pic RSS 14 500 823 040 octets (13,50 Gio), zéro
  page swapout, statut ressources `PASS`;
- manifeste génération :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_parent_v18_discovery_generation/e270a60d520d7c7/manifest.json`;
- SHA-256 génération :
  `b7bd849e033ad3c34a107eef9c16e5922591985bb59cd3968ae8d3dde6d51b33`;
- manifeste évaluation :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_parent_v18_discovery_evaluation/b7bd849e033ad3c3/manifest.json`;
- SHA-256 évaluation :
  `f119d9413c269d1303d00186ce121dbd1547e21e7a58e8bb506fbadadd235b2a`.

## Étape suivante autorisée

Construire une fusion parent à plafond strict 2 000, sans utiliser fold1 :

1. figer les features et quotas à partir des seules preuves label-blind;
2. apprendre les poids uniquement dans des inner-splits groupés par SIREN au
   sein des folds 2/3/4, ou comparer à une fusion déterministe préenregistrée;
3. produire des candidats OOF externes pour chacun des folds 2/3/4;
4. exiger au minimum 334/522 à `parent@2000`, avec publication par fold et par
   origine;
5. conserver fold1 fermé pour une unique confirmation same-CRM;
6. exiger ensuite un shadow set entièrement nouveau et non retunable avant
   toute revendication cross-CRM.

Aucun run Master/site/national ne doit être lancé à partir du seul oracle brut
v18.
