# V33 — chaîne CRM complète et dimensionnement avant lancement

11 septembre 2026. Le contrôle part des champs CRM originaux, sans anciens
candidats, SceneIR ou labels comme entrée. Deux branches : `v23_stable` et
`v30_stable`, modèles parent gelés et même succession v32. Le scope complet
compte23 571 entrées, folds2/3/4; aucun fold0/1 ouvert.

## Implémentation et intégrité

- `41c03013` : runner complet, contrat et tests. Autorité d'inférence explicite
  couvrant les mêmes modèles que l'audit d'isolation SIREN v31, sans enlever
  la garde des anciens runners limités au panel.
- `738fc4f8`, `ac8e8493`, `a09ee041` : évaluation séparée après scellement,
  publication des absences de label, lancement chaîné génération/évaluation
  et contrôle strict d'équivalence opérationnelle. Aucun label réel ouvert
  pendant les contrôles techniques. Le launcher épingle le code d'évaluation.
- 47 tests passent : calcul sans features d'identité/GT, scope, caps, ordres,
  doublons, dates, égalité exhaustive, conservation des requêtes sans candidat,
  distinction exact/opérationnel et adresse stricte avec suffixe/commune multi-CP.

Les sources tabulaires, modèles et100 fichiers de relations sont hashés.
Les index sont contrôlés par manifeste/meta et inventaire stat/inode/mtime,
pas par un rehash intégral de tous leurs segments. Les champs officiels de
parent sont vérifiés, y compris les clés géographiques historiques.

## Coût initial et réparation physique

Le premier pilote32 a pris224,18s, dont120,96s pour le legacy et63,30s pour
géographie/relations. Sa projection45,87h dépassait la garde36h : aucun long
run n'a été lancé sur cette base.

EXPLAIN a identifié un `BLOCKWISE_NL_JOIN` : la sélection des relations
comparait chaque relation à de nombreux seeds pour une condition
prédécesseur OU successeur. `2fca6790` remplace uniquement cet opérateur par
deux semi-jointures d'égalité et un UNION distinct. Même ensemble de lignes
officielles, pas de nouvelle règle de pertinence. Le contrôle sur le pilote
réel retrouve3 743 relations strictement identiques en1,656s; les tests
couvrent aussi les NULL et doublons.

Après nouveau gel, les32 mêmes entrées prennent149,43s : legacy95,69s,
parent3,20s, géographie/relations12,83s, direct/conjoint19,67s,
Masters9,00/8,91s. La variation du legacy montre aussi des effets physiques
entre exécutions : ne pas attribuer tout le gain global à la jointure seule.

Les sorties avant/après optimisation sont comparées exhaustivement : raw,
preuves, structured/direct/joint et les deux Masters sont identiques, toutes
colonnes, scores et rangs compris. Aucun recall consulté pour cette optimisation.

## Contrôles réels et projection

La nouvelle parité16 donne exactement les Masters v32 de référence depuis
les champs CRM, en106,35s, pic RSS processus4,47Gio, zéro swapout.
Le pilote32 utilise le tri SHA256("v33-cost|" + query_id) sur tout le CRM,
pas le panel522/512. Pic RSS processus6,45Gio, zéro swapout.

Projection linéaire du pilote32 : **30,58h et8,80Gio de fichiers**. Petite
population, effets de cache et variation possible entre requêtes : ce n'est
ni une garantie de durée, ni un SLO de production. Les coûts de lancement et
de vérification ajoutent un peu de surcoût. Le plafond reste36h et chaque
lot est borné à30min/14Gio RSS/zéro swapout.

La répétition indépendante des32 entrées en ordre inversé, sous-lots8, passe :
214,32s, pic RSS processus3,89Gio, zéro swapout. Comparaison bidirectionnelle
exhaustive raw/proof/structured/direct/joint/pools/Masters : zéro différence,
toutes colonnes, scores et rangs compris. Le coût de cette répétition avec
petits lots n'est pas utilisé pour choisir un réglage de pertinence.

**PASS_READY_FOR_FULL_REPLAY**. Gate scellé avant lancement, SHA
`d350ec587d1e5e2c96f962147da519ac10fe608023c4d661e18cb45393b4d918`.
Le complet a ensuite été effectivement lancé sous
`fast/full_job/generation/`, suivi de l'évaluation séparée sous
`fast/full_job/evaluation/` seulement après scellement de la génération.
Le processus utilise `caffeinate -i` pour éviter la veille inactive pendant
le calcul; il ne change pas les préférences système et laisse l'écran dormir.
L'évaluation, son code et la politique métier sont épinglés au lancement.
`928bab7e` borne aussi son éventuel spill DuckDB aux artefacts sur CATNAT.

Suivi local : `fast/full_job/launch.json`, `generation/RUNNING.json`,
`generation/progress.json` après le premier lot, puis `status.json` final.
Une erreur reste visible dans STOP/status/logs, sans retry silencieux. À la
publication du rapport, le run complet est **en cours**, pas terminé; aucun
nouveau recall disponible. La projection31h reste indicative.

## Portée scientifique

Pas de nouvelle qualité mesurée à ce préflight. Les modèles ont été ajustés
sur le petit panel historique : même une mesure complète de ce CRM ne
prouve pas la généralisation à d'autres CRM. Les courbes de rang du Master
ne sont pas les gates d'admission apprise500/100. Pas de READY_FOR_FOLD1.

Le complet doit régénérer toutes les entrées (y compris celles des essais),
sceller les candidats, puis seulement ouvrir les labels dans un processus
séparé et reconstruire un sidecar officiel exhaustif. Chaque requête sans
candidat ou absente du catalogue demeure au dénominateur. L'évaluation
publiera exact/opérationnel, historique/V2/V3 et attribution des pertes.

Artefacts sous
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v33_full_replay/`.
Premières exécutions conservées; nouveau gel sous `fast/`.
SHA préparation physique :
`765635a3b751f02e1e63b4d7df0af1253abc25239b5b4a36de4621f3856136f8`.
