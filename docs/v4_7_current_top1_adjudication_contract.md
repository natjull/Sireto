# Contrat V4.7 — adjudication exhaustive des top-1 courants

Statut : préenregistré avant toute nouvelle collecte de preuve et avant tout
score, entraînement ou choix de seuil d'accepteur.

Identifiant : `V47_CURRENT_TOP1_EXHAUSTIVE_ADJUDICATION`.

## 1. Question testée

La V4.5 a rejoué le retrieval V4.2-B avec le ranker A gelé sur les 172 cas
V4.4. Cent trente-cinq prédictions sont restées identiques et 37 ont changé.
Un label V4.4 juge uniquement l'ancien SIRET prédit : il est donc interdit de
le copier vers le nouveau SIRET.

V4.7 répond à une seule question :

> Parmi les 37 prédictions courantes différentes, lesquelles sont réellement
> correctes, incorrectes, ambiguës ou impossibles à résoudre avec des preuves
> publiques ?

Aucun modèle n'est entraîné dans V4.7. Le test final reste fermé.

## 2. Population immuable

Source :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_5_hard_scenes/21f8c0b0b172b907/scene_compatibility.parquet`

SHA-256 :
`72540dcdba6f33da0eb1875ef4bcdc8c44a2cd10083589b5e1683098cd954a08`

La population est exactement l'ensemble des lignes ayant
`scene_status=SCENE_DRIFT` :

- 37 dossiers uniques ;
- huit issus du tirage `RANDOM_POPULATION` ;
- 29 ciblés ;
- le SIRET jugé est toujours `replayed_top1_siret` ;
- le modèle jugé est le ranker A gelé appliqué au retrieval V4.2-B.

Tout écart de hash, de volume, d'identifiant ou de SIRET arrête V4.7 avec
`STOP_INPUT_INTEGRITY`.

Les 135 lignes `SCENE_COMPATIBLE` ne sont pas réadjudiquées. Leur label V4.4
reste transportable uniquement parce que le SIRET top-1 est strictement
identique.

## 3. Sources et règle de preuve

Les sources autorisées et les groupes d'indépendance sont ceux de
`docs/v4_4_autonomous_adjudication_contract.md` :

- snapshot SIRENE épinglé ;
- API Recherche d'entreprises et pages publiques officielles ;
- site officiel de l'entité ;
- annuaires publics sectoriels officiels ;
- documents publics datés avec URL et empreinte conservées.

Le SIRET CRM historique, l'ancien verdict V4.4, le score, le rang et une simple
adresse commune peuvent orienter la recherche mais ne constituent pas une
preuve.

Chaque nouveau top-1 reçoit exactement un statut :

- `TOP1_CORRECT` ;
- `TOP1_WRONG` ;
- `AMBIGUOUS` ;
- `UNRESOLVED`.

`evidence_validated=true` exige simultanément :

1. deux groupes de preuves indépendants du modèle ;
2. une concordance explicite d'identité, et du site lorsqu'un SIREN possède
   plusieurs établissements possibles ;
3. aucune contradiction non résolue ;
4. les références, dates de collecte, contenus utiles et empreintes dans
   l'artefact.

Deux restitutions du même enregistrement SIRENE ne forment qu'un seul groupe.
Une fermeture suivie d'un transfert ne permet de retenir le nouveau site que
si les preuves relient bien le CRM à ce site précis. L'absence de résultat web
n'est jamais une preuve négative.

## 4. Séparation et interdictions

- Les huit dossiers aléatoires restent marqués `random_sealed`.
- Ils ne servent ni à choisir une règle, ni à entraîner un modèle, ni à choisir
  un seuil.
- L'ancienne adjudication peut fournir des pistes de recherche, mais aucune
  conclusion n'est transportée automatiquement.
- Aucun dossier `REVIEW`, holdout ou test final n'est ouvert.
- Aucun positif n'est injecté.
- Retrieval, ranker A, features, accepteur et seuil sont gelés.
- Aucun GPU loué, service payant ou validation utilisateur n'est autorisé.
- Une décision sans deux preuves indépendantes reste `UNRESOLVED`.

## 5. Sorties canoniques

V4.7 produit :

- `docket.parquet` : les 37 dossiers et le SIRET courant à juger ;
- `evidence.parquet` : une ligne par preuve avec URL, famille, groupe
  d'indépendance, date, empreinte et fait extrait ;
- `adjudications.parquet` : les quatre statuts, justification et références ;
- `current_labels.parquet` : fusion des 135 labels compatibles et des 37
  nouveaux verdicts, sans cible pour `UNRESOLVED` ;
- `manifest.json` : hashes de toutes les entrées et sorties ;
- un rapport séparant toujours tirage aléatoire et sélection ciblée.

Les artefacts sont reconstruits par code à partir des preuves conservées. Un
fichier de verdict saisi manuellement ne constitue pas la sortie canonique.

## 6. Gates préenregistrés

### 6.1 Gate qualité du corpus courant

`CURRENT_LABEL_SET_USABLE` exige :

- les 37 dossiers traités exactement une fois ;
- au moins 150 labels fiables sur les 172 scènes courantes ;
- au moins 50 labels fiables sur les 57 dossiers aléatoires ;
- zéro label fiable avec moins de deux groupes indépendants ;
- zéro transport d'un ancien verdict vers un SIRET différent.

Sinon le verdict est `STOP_CURRENT_LABEL_QUALITY`.

### 6.2 Gate d'utilité pour l'accepteur

Après réussite du gate qualité :

- `GO_ACCEPTOR_FEASIBILITY` si la sélection ciblée courante contient au moins
  20 négatifs fiables (`TOP1_WRONG` ou `AMBIGUOUS`) et si le tirage aléatoire
  contient au moins trois négatifs fiables ;
- `KEEP_CURRENT_STACK_SHADOW` si les labels sont fiables mais qu'il existe
  moins de 20 négatifs ciblés : la collecte aura alors surtout démontré que le
  retrieval V4.2-B a réparé les anciennes erreurs, sans fournir assez de
  contre-exemples nouveaux pour justifier un accepteur ;
- `PIVOT_INDEPENDENT_EVALUATION` si les négatifs ciblés sont suffisants mais
  que le tirage aléatoire contient moins de trois négatifs : aucune variante
  d'accepteur ne peut être comparée honnêtement sur cette petite réserve.

Ces seuils autorisent seulement une expérience hors test. Ils ne certifient
jamais une précision de 99,8 %.

## 7. Décision après V4.7

En cas de `GO_ACCEPTOR_FEASIBILITY`, un contrat V4.8 distinct devra geler les
partitions anti-fuite, les variantes, les poids, le choix du seuil et les
critères de promotion avant tout entraînement.

Dans les deux autres verdicts non intègres, aucun accepteur n'est entraîné. Le
prochain travail porte alors sur une évaluation indépendante du stack courant,
pas sur l'ouverture opportuniste de davantage de dossiers historiques.
