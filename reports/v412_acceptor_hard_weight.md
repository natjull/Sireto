# V4.12 — pondération des dossiers difficiles dans l'accepteur

## Verdict

`PIVOT_ACCEPTOR_FEATURES`

Augmenter le poids des 83 dossiers métier déjà adjudiqués ne permet pas à
l'accepteur d'automatiser davantage de cas difficiles. Toutes les variantes
sûres acceptent exactement **1 dossier sur 83** hors apprentissage, comme la
variante de référence de poids `1`. Aucun modèle issu de cette expérience ne
doit donc être promu ni envoyé en validation indépendante.

## Population et absence de fuite

- 83 dossiers difficiles consommés : 77 `MATCH_EXACT` et 6 `AMBIGUOUS` ;
- 5 plis séparés par composante SIREN pour produire les décisions hors
  apprentissage sur ces 83 dossiers ;
- 665 scènes du lot de réglage du seuil et 701 scènes du lot de comparaison ;
- les 7 dossiers de la validation ranker précédente sont exclus de la
  sélection ;
- le test final reste fermé.

Les 189 REVIEW historiques encore non adjudiqués ne sont pas un jeu de preuve
indépendant pour l'accepteur : leurs anciennes cibles appartiennent déjà aux
lots de développement ayant servi à régler ou comparer le seuil. Ils restent
utiles pour corriger les labels et analyser les erreurs, mais pas pour annoncer
une validation indépendante.

## Résultats

| Poids des cas difficiles | AUTO sûrs sur 701 contrôles | Erreurs | AUTO sur 83 difficiles hors apprentissage | Erreurs / ambiguïtés AUTO |
|---:|---:|---:|---:|---:|
| 1 | 593 | 0 | 1 | 0 / 0 |
| 5 | 593 | 0 | 1 | 0 / 0 |
| 10 | 597 | 0 | 1 | 0 / 0 |
| 20 | 597 | 0 | 1 | 0 / 0 |
| 50 | 594 | 0 | 1 | 0 / 0 |

La meilleure couverture apparente sur les contrôles (`597/701`, soit 85,16 %)
ne constitue pas un gain sur le problème visé : elle ne récupère aucun dossier
difficile supplémentaire par rapport au poids `1`.

## Diagnostic métier

Parmi les 83 dossiers, le ranker candidat place le bon SIRET en tête pour 60
des 77 cas exacts. Pourtant l'accepteur ne laisse passer qu'un de ces 60 bons
classements. Les variables les plus différenciantes dans ce lot sont :

- l'avantage du nom légal de l'unité légale sur le nom d'établissement ;
- l'écart de ressemblance de nom entre le premier et le second candidat ;
- l'écart de score du ranker ;
- l'indicateur de siège et la concurrence entre sites.

Une ablation locale sans contraintes monotones a également été négative : elle
n'accepte que 0 à 1 dossier difficile selon le poids. Le blocage n'est donc pas
réparable par un simple changement de poids ou le retrait d'une contrainte.

## Suite autorisée

La prochaine expérience doit porter sur des variables relationnelles explicites
construites à partir des preuves déjà présentes, notamment « nom légal meilleur
que nom d'établissement » et la concurrence entre SIRET d'un même SIREN. Elle
doit rester sur les données de développement consommées. Un nouveau dossier de
validation ne sera gelé que si cette ablation apporte une couverture hors
apprentissage réellement supérieure sans erreur observée.

Artefact machine :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_hard_weight/a9bdb09ea504194e/evaluation.json`.

