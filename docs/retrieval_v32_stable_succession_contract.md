# V32 — stabilisation cohérente du témoin de succession

11 septembre 2026. Autorisation explicite de réparer la reproductibilité,
commune aux branches v23 et v30. Recette fixée avant tout nouveau score de
qualité. Les recettes/fichiers historiques restent immuables. Les nouvelles
branches s'appellent `v23_stable` et `v30_stable`.

## Décision sémantique

Une preuve de succession est une ligne officielle entière : seed SIRET,
SIRET lié, direction, date effective et identifiant de relation. Il est interdit
de choisir indépendamment ces champs et de fabriquer une preuve composite.

Conserver la priorité existante au seed le mieux classé. Parmi ses relations
admissibles vers un même parent, prendre la plus récente **par date effective
de l'événement**, pas par date d'import. Une date inconnue vient après une date
connue, à seed égal. En cas d'égalité restante, départager par SIRET lié, SIRET
seed, direction et identifiant de relation en ordre lexical croissant. Ce
dernier ordre est purement technique; les identifiants ne deviennent pas des
features de pertinence.

Ne pas favoriser a priori successeur plutôt que prédécesseur, ni actif plutôt
que fermé : le label exact peut désigner un ancien établissement. Ne pas
introduire de nouvelle préférence géographique, numéro, nom ou qualité BAN
dans cette réparation. Ces preuves sont déjà exploitées ailleurs; les ajouter
ici serait une nouvelle expérience de pertinence, pas une stabilisation.

Le rang du parent dans le canal reste celui de v18 : seed minimal, date
effective maximale de toutes ses relations admissibles, SIREN terminal. Cette
date agrégée est nommée `parent_latest_effective_date` dans le sidecar de
preuve; elle n'est pas présentée comme la date de la ligne témoin choisie.
La ligne témoin conserve sa propre `witness_effective_date`.

## Invariants

- Même ensemble de relations officielles scellées, continuité économique
  obligatoire, recherche dans les deux directions, seeds top250, canal top256.
- Une seule référence témoin par parent; aucun budget ni poids modifié.
- SIREN stocké dans la relation officielle, jamais déduit par le nouveau code.
- Doublons sémantiques neutralisés; aucun ordre physique de fichier, thread,
  processus, query_id ou shard ne choisit le témoin.
- Membership/rangs des parents et tous les autres canaux inchangés.
- Même réparation avant scoring pour les deux branches; modèles v23 gelés.
- Aucune consultation de labels pour choisir la règle, aucun positif injecté,
  aucun ajustement après résultats. Folds0/1 fermés, aval et caps2000/500/100 gelés.

## Validation avant rejeu complet

Tests unitaires de cohérence de ligne, seed/date, dates absentes, duplications,
directions et permutations. Comparaison SQL avec une référence Python séparée.
Sur les16 entrées du préflight v31 : plusieurs processus, threads1/2/8,
ordres inversés, shards différents et fichiers de relations permutés; mêmes
témoins, scores et rangs au Master de chaque branche. On attend des différences
avec les fichiers historiques, pas entre deux exécutions de la réparation.

Les contrôles sur16 réutilisent uniquement les canaux déjà reconstruits sans
labels pour isoler la réparation. Ce n'est pas le rejeu complet de production.
Le futur rejeu des23 571 entrées devra régénérer tous les canaux depuis les
champs CRM, contrôler ressources et isolation, et publier exact/opérationnel
séparément. Une absence catalogue reste une erreur. Pas de repondération des
522/512 ni de projection du petit panel vers le CRM complet.

Un préflight de stabilité réussi ne donne aucun nouveau recall ni
READY_FOR_FOLD1. Budget local14Gio, zéro swapout,30min par étape de validation.
Mac/CATNAT uniquement, aucun appel externe.
