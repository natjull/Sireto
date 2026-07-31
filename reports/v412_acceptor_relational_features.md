# V4.12 — ablation des preuves relationnelles de l'accepteur

## Verdict

`PIVOT_ACCEPTOR_FEATURES`

Rendre explicite l'avantage du nom légal sur le nom d'établissement ne
permet pas à l'accepteur de récupérer davantage de dossiers difficiles. Les
deux variantes testées restent à **1 AUTO sur 83** hors apprentissage, comme
la référence, et perdent légèrement de la couverture sur les 701 contrôles.

## Expérience

Le ranker, les pools et les 80 variables existantes restent inchangés. À poids
des dossiers difficiles fixé à `10`, trois variantes sont comparées :

1. accepteur de référence ;
2. ajout de deux écarts « nom légal moins nom d'établissement » pour le top 1
   et relativement au top 2 ;
3. ajout des deux écarts, de leur interaction avec l'écart du ranker et de la
   concurrence entre établissements du même SIREN.

Les décisions sur les 83 dossiers difficiles sont produites en cinq plis
SIREN hors apprentissage. Le seuil vient uniquement des 665 scènes de réglage
consommées ; les 701 scènes de comparaison servent au contrôle apparié. Les
sept adjudications précédentes et le test final ne participent pas au choix.

| Variante | AUTO sur 701 contrôles | Erreurs / ambiguïtés AUTO | AUTO sur 83 difficiles OOF | Erreurs / ambiguïtés AUTO |
|---|---:|---:|---:|---:|
| Référence poids 10 | 597 (85,16 %) | 0 / 0 | 1 (1,20 %) | 0 / 0 |
| Deux relations nom légal/établissement | 594 (84,74 %) | 0 / 0 | 1 (1,20 %) | 0 / 0 |
| Quatre relations nom/site | 595 (84,88 %) | 0 / 0 | 1 (1,20 %) | 0 / 0 |

## Conclusion métier

Le signal « nom légal contre enseigne » explique une partie des erreurs du
ranker, mais ne suffit pas à certifier ses corrections à 99,8 %. Continuer à
ajouter des combinaisons artisanales sur les mêmes 83 dossiers créerait surtout
du sur-ajustement.

Le meilleur candidat historique reste donc l'accepteur de référence. Il
franchit le gate de développement global sur les 701 contrôles, mais aucune
preuve produit nouvelle n'est disponible : l'ancien test final et le challenge
local sont consommés, et les 189 REVIEW restants ont déjà participé au réglage
ou à la comparaison via leurs labels historiques. Une vraie conclusion produit
exige une nouvelle cohorte CRM dotée de preuves SIRET indépendantes du matching.

Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_relational_features/81a976729f2140de`.

