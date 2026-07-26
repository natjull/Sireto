# Contrat E2 — Accepteur V4 exact-SIRET

## Objet

Apprendre à décider si la première proposition du ranker V4 peut devenir
`AUTO_MATCH` ou doit rester `REVIEW`.

Le ranker V4 est gelé. Cette expérience ne modifie ni le retrieval, ni ses
100 candidats, ni les features candidat, ni le ranker.

## Scènes

### Train

- 5 749 scènes `MATCH_EXACT` scorées par les prédictions OOF du ranker ;
- 966 scènes `AMBIGUOUS` du noyau V4 ;
- 142 scènes `AMBIGUOUS` de `fit_addition`.

Le ranker n’a jamais été entraîné sur les scènes ambiguës. Elles sont scorées
par le modèle final gelé et marquées `out_of_sample_ambiguous`.

### Dev

- 305 scènes `MATCH_EXACT` de `dev_new` ;
- 53 scènes `AMBIGUOUS` de `dev_new`.

Le dev est divisé déterministiquement en deux moitiés disjointes :
`calibration` et `threshold`.

### Cibles

- `MATCH_EXACT` : positif uniquement si le SIRET top-1 est exactement la
  vérité V4 ;
- `AMBIGUOUS` : toujours incorrect pour `AUTO_MATCH`, quel que soit le
  candidat choisi ;
- `UNRESOLVED` : exclu, car inconnu ne signifie pas faux ;
- aucun `NO_MATCH` n’est inventé.

## Reconstruction des candidats ambigus

Les mêmes canaux et la même admission gelée à 100 sont exécutés. Les SIRET
directs multiples de la qualification servent uniquement à confirmer le label
`AMBIGUOUS`; ils ne sont jamais utilisés pour modifier, compléter ou ordonner
la liste candidate.

Le runner de canaux exige techniquement un SIRET de diagnostic. Le premier
SIRET direct peut être transporté comme `diagnostic_probe_siret`, mais aucune
métrique de recall n’est publiée et cette valeur ne participe à aucun calcul
de retrieval ou de score.

## Features

Une scène reçoit les 80 features V9 déjà versionnées :

- scores top-1/top-2, écart, dispersion et entropie ;
- concurrence entre SIREN et entre établissements du même SIREN ;
- provenance du retrieval ;
- valeurs top-1, top-2 et différences des 20 preuves nom/adresse.

L’ordre des features doit être strictement identique au service.

## Modèles et calibration

Variantes fermées :

1. régression logistique standardisée ;
2. XGBoost avec les hyperparamètres V9 existants.

Pour chaque modèle :

1. score brut ;
2. calibration sigmoid ;
3. calibration isotonic.

Seed 42. Aucun tuning supplémentaire, règle segmentaire ou sélection de
features après lecture du dev.

## Sélection et gate

La moitié `calibration` ajuste uniquement la transformation du score. La
moitié `threshold` choisit le seuil.

Publier pour chaque variante la couverture maximale observée à :

- 99,0 % de précision SIRET exacte ;
- 99,5 % ;
- 99,8 %.

Le seuil exige au moins 25 décisions `AUTO_MATCH`.

- `GO_HOLDOUT_V4` si la meilleure variante atteint au moins 25 % de couverture
  à 99,8 % de précision observée sur la moitié `threshold` ;
- `PIVOT_ACCEPTEUR_V4` sinon.

Avec ce volume de dev, 99,8 % observé impose pratiquement zéro erreur. Le
résultat reste une estimation de développement, jamais une garantie.

## Interdictions

- aucun positif injecté ;
- aucune lecture de l’ancien test ;
- aucune lecture, génération de candidats ou prédiction sur
  `holdout_sealed` ;
- aucune location de GPU ou dépense externe ;
- aucun déploiement avant une évaluation finale autorisée et unique.
