# Retrieval all-human v16 — résultats parent SIREN et géographie

Date : 26 août 2026  
Statut : **`PIVOT_PARENT_DISCOVERY`**

## Question testée

Le pilote v16 vérifie, sur la cohorte v15 scellée de 1 034 requêtes
(522 pertes Master et 512 contrôles, folds 2/3/4), si deux familles de signaux
peuvent rendre crédible l'objectif Master@2000 avant tout calcul national :

1. un retrieval natif de SIREN à 26 canaux (sources officielles, historique,
   rôles RNE, secteur, nom et adresse) ;
2. une side quest géographique fondée sur la résolution BAN de la requête et
   les coordonnées Lambert-93 des établissements SIRENE.

La génération de chaque branche est label-blind et scellée avant l'ouverture
du fichier de vérité. Aucun positif n'est injecté. Fold0 et fold1 ne sont pas
ouverts. La BAN est explicitement non décisionnelle (`score_eligible=false`) :
son absence n'est jamais une preuve négative et aucune distance n'élimine un
candidat.

## Exécution parent SIREN

- Génération : 1 034 requêtes, 589 144 signaux, 26 canaux.
- Temps : 128,695 s ; pic RSS : 12 194 529 280 octets ; swapout : 0.
- Manifeste génération :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_parent_v16_workbench_generation/3946aae4798b0ef9/manifest.json`
  (`97566d4d336b790dd248c62f56cbde16a87227aad30c6c67e7b9d4bb45210b83`).
- Manifeste évaluation :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_parent_v16_workbench_evaluation/97566d4d336b790d/manifest.json`
  (`f929f1bcdc69c337e14808402641dc3f0e6a695b47ad123b8016a77c80efb539`).

Le meilleur pool parent texte retrouve 258/522 parents absents de la recette
v15 à rang <=512, soit 108/82/68 sur les folds 2/3/4. C'est un gain absolu de
155 parents sur la baseline v15 (103/522), mais très inférieur au niveau de
confiance préenregistré 160/158/146 = 464. Le hard kill est donc déclenché et
aucune expansion site ou exécution nationale n'est autorisée.

## Exécution géographique

- Résolution BAN : 866/1 034 requêtes ; 168 non résolues restent au
  dénominateur.
- Génération : 1 724 094 paires candidat-géo, 433 066 voisins établissements
  et 432 443 parents distincts.
- Temps : 15,493 s ; pic RSS : 10 202 054 656 octets ; swapout : 0.
- Manifeste génération :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_geo_v16_generation/15b3d8c774edd5a7/manifest.json`
  (`2cbfb03c15fd82d0e57c72ece37ac23513dc429df857280ca8f7a2030b9694ff`).
- Manifeste évaluation OOF :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_geo_v16_evaluation/2cbfb03c15fd82d0/manifest.json`
  (`7c5ae2faa528ac84ae9c7e578f70b38213d4b83f91fb1a7283bbce3230f2c8a2`).

L'oracle géographique contient le bon parent pour 457/1 034 requêtes à 512,
soit 159/152/146 par fold. Sur les sorties SIRET, le voisinage brut atteint
293 exacts à 100 ; le modèle XGBoost OOF utilisant distance, qualité et
incertitude BAN en atteint 304. Il existe donc un signal apprenable réel.
Cependant, sur les 522 pertes v15, la géographie seule ne contient le bon
parent que pour 131 requêtes et ne produit que 47 rescues SIRET exacts nouveaux.

## Oracle d'union texte + géographie

La comparaison truth-side est effectuée uniquement après scellement des deux
générations. Elle mesure la présence du bon SIREN dans l'union des pools, sans
apprentissage ni sélection post-hoc :

| Population | Texte | Géo | Intersection | Union |
|---|---:|---:|---:|---:|
| 522 pertes v15 | 258 | 131 | 73 | **316** |
| fold 2 — 182 pertes | 108 | 50 | 30 | **128** |
| fold 3 — 175 pertes | 82 | 46 | 24 | **104** |
| fold 4 — 165 pertes | 68 | 35 | 19 | **84** |
| 512 contrôles | 473 | 326 | 301 | **498** |

Même un ranker oracle ne peut donc sélectionner que 316 bons parents parmi
les 522 pertes avec les candidats actuellement découverts. En supposant de
façon irréaliste que chaque parent retrouvé donne ensuite toujours le bon
SIRET, la borne supérieure Master serait 23 049 + 316 = **23 365/23 571 =
99,1260 %**. Elle reste sous 99,2 % (18 hits manquants) et sous le gate de
confiance Master@2000 99,5 % (89 hits manquants).

## Verdict et suite autorisée

Verdict : **`PIVOT_PARENT_DISCOVERY`**.

Les coordonnées et la qualité BAN doivent rester des features souples du futur
ranker, car l'ablation OOF prouve un gain. Mais entraîner maintenant un
LambdaMART plus complexe ne peut pas atteindre l'objectif : 206/522 bons
parents restent absents même de l'oracle d'union.

Le prochain sprint court doit donc augmenter la couverture des candidats
SIREN avant tout reranking : alias et dénominations historiques, enseignes et
établissements, variantes d'adresse, rôles RNE, puis canal BAN→SIRENE avec
incertitude explicite. Son premier gate est un nouvel oracle parent@512 ; aucun
run Master/site/national ne doit être lancé tant que cet oracle ne franchit pas
les seuils foldés préenregistrés.
