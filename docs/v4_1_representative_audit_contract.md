# Contrat V4.1 — audit représentatif du CRM shadow

Statut : préenregistré avant lecture des preuves et avant adjudication.

## Objectif

Déterminer, sur un échantillon représentatif et difficile du CRM shadow, si
les erreurs viennent principalement :

1. du retrieval, quand le bon SIRET est absent des 100 candidats ;
2. du ranker, quand le bon SIRET est présent mais n'est pas premier ;
3. de l'accepteur, quand le premier choix correct est envoyé en revue ou qu'un
   mauvais choix est automatisé ;
4. de l'absence de vérité unique dans les données disponibles.

Aucun modèle, feature ou seuil n'est modifié pendant l'audit.

## Population

La population est le run shadow V4.1 immuable :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/shadow/v4_1/runs/v41_shadow_f1058826_20260727_v3`

Il contient exactement 19 025 `SERVICE ID` autorisés. Les 4 584 lignes
exclues du shadow restent interdites.

## Échantillon figé

L'échantillon contient 800 identifiants distincts. Les sélections sont
déterministes avec la clé :

`SHA-256("v4.1-representative-audit:42:" + SERVICE_ID)`

Les strates sont sélectionnées dans cet ordre, sans remplacement :

1. `RANDOM_POPULATION` : 250 cas tirés sur toute la population ;
2. `NO_ACTIVE_CANDIDATE` : 50 cas sans candidat actif ;
3. `AUTO_NEAR_THRESHOLD` : les 100 AUTO au score le plus proche du seuil ;
4. `AUTO_HIGH_SCORE` : 150 AUTO aux scores les plus élevés ;
5. `REVIEW_NEAR_THRESHOLD` : les 150 REVIEW au score le plus proche du seuil ;
6. `REVIEW_LOW_SCORE` : 100 REVIEW aux scores les plus faibles.

Les cas déjà retenus par une strate antérieure sont exclus des suivantes.
Les égalités de score sont départagées par la clé SHA-256.

## Aveuglement

Deux artefacts séparés sont produits :

- `sample_registry.parquet` conserve strate, décision, score, prédiction et
  raison de revue pour l'analyse postérieure ;
- `blind_cases.parquet` contient uniquement les données CRM brutes, l'état
  administratif du SIRET d'entrée et un identifiant d'audit aléatoire stable.

Pendant la construction des preuves et l'adjudication, il est interdit de lire
dans `sample_registry.parquet` :

- `decision` ;
- `confidence` ;
- `predicted_siret` ;
- `review_reason` ;
- tout rang ou score du retrieval, du ranker ou de l'accepteur.

La jointure avec le registre scellé n'est autorisée qu'après gel du fichier
d'adjudication.

## Sources de preuve autorisées

- CRM brut inchangé ;
- snapshot SIRENE local épinglé ;
- état actif ou fermé des établissements ;
- noms légaux, enseignes, sigles et noms antérieurs présents dans le snapshot ;
- adresses, communes, codes postaux et codes INSEE ;
- relation SIREN entre établissements actifs et fermés.

Les sorties du modèle peuvent être utilisées après adjudication pour l'autopsie
des étages, jamais pour choisir le label.

## Labels

Chaque cas reçoit exactement un statut :

- `MATCH_EXACT` : un seul SIRET actif est soutenu par les preuves ;
- `NO_MATCH` : aucun SIRET actif éligible dans le snapshot à la date de
  référence ;
- `AMBIGUOUS` : au moins deux SIRET actifs restent plausibles ;
- `UNRESOLVED` : les preuves disponibles ne permettent pas de conclure.

Une préannotation automatique ou locale doit être marquée `PROVISIONAL` et ne
constitue pas une validation humaine. Chaque label conserve les références de
preuve et un code de règle explicite.

## Mesures après gel des labels

Les métriques sont publiées avec nombres bruts :

- couverture identifiable :
  `MATCH_EXACT / 800` ;
- Recall@100 SIRET exact ;
- Hit@1 SIRET exact ;
- précision et couverture AUTO observées sur les cas adjudicables ;
- matrice AUTO/REVIEW par label ;
- familles d'erreur retrieval, ranker et accepteur ;
- résultats séparés pour le tirage aléatoire et les strates ciblées.

Les strates ciblées ne doivent jamais être pondérées comme si elles
représentaient naturellement le CRM. L'estimation populationnelle principale
vient uniquement des 250 cas `RANDOM_POPULATION`.

## Décision

- `GO_DIAGNOSTIC` : assez de `MATCH_EXACT` représentatifs pour localiser les
  erreurs et une famille dominante est démontrée ;
- `PIVOT_LABELS` : trop de cas restent non résolus pour comparer les étages ;
- `PIVOT_RETRIEVAL`, `PIVOT_RANKER` ou `PIVOT_ACCEPTEUR` : une brique concentre
  les erreurs démontrées ;
- `STOP` : les preuves montrent que la cible de précision/couverture n'est pas
  raisonnablement atteignable avec les données disponibles.

