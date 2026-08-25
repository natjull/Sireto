# Résultat du pilote Master@2000 v15

Date : 26 août 2026  
Verdict : **`PIVOT_PARENT_RETRIEVAL`**

## Exécution

Le pilote préenregistré a été exécuté une seule fois sur les 522 requêtes où
la vérité humaine est absente du top1750 v11 et sur 512 contrôles déterministes
où elle est présente. Le générateur n'a reçu que `query_id` et `fold`; la
vérité et le type de cohorte n'ont été ouverts qu'après scellement des
candidats. Folds 0 et 1 sont restés fermés et aucun positif n'a été injecté.

La génération a traité 1 034 requêtes en 325,736 s, avec un pic RSS de
3 367 305 216 octets et zéro swapout. Elle publie 1 724 094 lignes candidates,
dont 11 174 émissions rescue, ainsi que 44 120 parents. Le rang maximal
observé est 1 862 : la recette n'a jamais dépassé le plafond absolu de 2 000.

## Gate

| Vue | fold2 | fold3 | fold4 | Global |
|---|---:|---:|---:|---:|
| Misses top1750 | 182 | 175 | 165 | 522 |
| Rescues primaires | 31 | 32 | 35 | 98 |
| Minimum requis | 155 | 148 | 138 | 441 |
| Hits Master projetés | 7 616/7 767 | 7 824/7 967 | 7 707/7 837 | 23 147/23 571 |
| Recall projeté | 98,0559 % | 98,2051 % | 98,3412 % | **98,2012 %** |

Les 512/512 contrôles sont préservés. Les seuils de rescue par fold échouent et
aucune sensibilité ne passe :

- parents384 / rescue225 / profondeur16 : 23 147/23 571 = 98,2012 % ;
- parents512 / rescue200 / profondeur8 : 23 141/23 571 = 98,1757 %.

## Diagnostic causal

Le bon SIREN parent est retrouvé pour 103/522 misses (19,73 %). Lorsque ce
parent est présent, le bon site est émis dans 98/103 cas (95,15 %). Parmi les
424 échecs primaires restants, seulement cinq contiennent encore le bon parent.
Le goulot est donc le retrieval parent, pas l'expansion SIREN vers SIRET.

Sur les 98 rescues, 31 étaient déjà classés entre 1 751 et 2 000 dans v11 et
67 se trouvent au-delà de l'ancien top2000. La recette apporte donc un gain
réel, mais très insuffisant pour Master@2000 >=99,5 %. Elle est terminale et ne
doit pas être retouchée/rejointe sous le même identifiant. Aucun run national
Master, top500, top100 ou LambdaMART site n'est autorisé par ce résultat.

## Artefacts scellés

- cohorte :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_master_v15_feasibility_cohort/72896b6f70d7c9eb/manifest.json`,
  SHA-256 `9ac986a50295c67ed202581bacd79b1fe58bacc6126c3963552d6fc0870411d3` ;
- génération label-blind :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_master_v15_feasibility_generation/f900d9f9a1772cc4/manifest.json`,
  SHA-256 `9d62891d89eb96bc007694426a19b9df70037b6b725709baf030fcc7b6808012` ;
- évaluation :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_master_v15_feasibility_evaluation/9d62891d89eb96bc/manifest.json`,
  SHA-256 `cade37fff01dbfd2932bd4e243b14abb8269546e1072a49a711f52f4a00a60f4` ;
- recette et moteur : commit `87cebc3`.
