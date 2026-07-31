# V4.12 — Évaluation du stack ranker candidat + accepteur

Date : 31 juillet 2026  
Artefact : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_ranker_acceptor_stack/f6d3c21bd8a8359e`

## Question

Le ranker pondéré `0,5` sait désormais corriger des cas difficiles. L'accepteur V4.11 réentraîné sur ses nouvelles scènes sait-il transformer ces corrections en décisions `AUTO_MATCH` tout en maintenant la contrainte de précision SIRET exacte à 99,8 % ?

## Protocole sans fuite

- cinq rankers conjoints sont réentraînés en excluant à chaque fois le fold historique et le fold difficile évalués ;
- les scènes accepteur de 5 547 requêtes historiques et 83 dossiers difficiles sont donc toutes issues de prédictions ranker hors échantillon ;
- les 77 labels exacts et six ambiguïtés sont ajoutés au fit accepteur ;
- 665 scènes non adjudiquées servent uniquement à choisir le seuil ;
- 701 autres scènes non adjudiquées servent à comparer les accepteurs ;
- les sept dossiers du docket indépendant ne sont scorés qu'après sélection de la famille et du seuil ;
- aucun positif n'est injecté et les deux misses retrieval restent des erreurs ;
- le test final et V4-Fresh restent fermés.

## Comparaison sur le développement classique

| Accepteur | Seuil | AUTO | Couverture | Erreurs AUTO | Précision observée | Ambiguïtés AUTO | Éligible |
|---|---:|---:|---:|---:|---:|---:|---|
| Régression logistique compacte | 0,6633 | 619/701 | 88,30 % | 1 | 99,838 % | 1 | non |
| **XGBoost monotone** | **0,9893** | **592/701** | **84,45 %** | **0** | **100 %** | **0** | oui |

Le XGBoost monotone est sélectionné avant lecture des décisions indépendantes.

## Docket indépendant difficile

| Mesure | Résultat |
|---|---:|
| Dossiers | 7 |
| Top 1 exact du ranker | 6 |
| Ambiguïtés | 1 |
| AUTO_MATCH | **0** |
| REVIEW | **7** |
| Erreurs AUTO | 0 |
| Couverture | **0 %** |

Les scores accepteur des six bons top 1 sont compris entre `0,0595` et `0,2963`, très loin du seuil `0,9893`. L'ambiguïté PROMOTRANS reçoit `0,1653` et reste correctement en REVIEW.

## Interprétation

Le ranker n'est plus le seul goulot. Il corrige les six identités exactes du lot indépendant, mais l'accepteur ne reconnaît aucune de ces corrections comme suffisamment sûre. La couverture classique de 84,45 % masque donc une couverture nulle sur la population qui justifie précisément la refonte.

Ce n'est pas un problème que l'on peut régler en abaissant simplement le seuil : les bons dossiers difficiles et l'ambiguïté occupent la même zone basse. Un seuil ad hoc choisi sur sept cas créerait une fuite et ne serait pas crédible à 99,8 %.

## Décision

Verdict : **`PIVOT_ACCEPTOR_COVERAGE`**.

La suite autorisée est une expérience de pondération/OOF de l'accepteur sur les cas difficiles, sélectionnée uniquement sur les 83 scènes d'entraînement OOF et les splits de développement consommés. Les sept décisions indépendantes sont désormais consommées et ne pourront pas servir au choix de cette variante. Toute variante retenue exigera un nouveau docket indépendant parmi les 189 REVIEW encore non adjudiqués.
