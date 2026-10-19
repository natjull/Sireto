# Brief14 — Lot A couvert, pas le full3000

Autorité : `sireto-site-target-14.md`, décision lead scientifique déléguée.
Mode : `.venv/bin/python -u scripts/train3000_frozen_v142_supervisor.py --covered-only`.
Code : `git log -1 --format=%H -- scripts/train3000_frozen_v142_covered.py`.

## Sélection sans labels
Tirage3000 `d54cec2c` inchangé, CRM8/folds2/3/4 et affectation entreprise OOF scellés.
Une zone doit disposer d'un corpus communal complet et de banques SITE/PARENT :
295 caches historiques, ou préparation smoke20 complète avec sortie possédée0,
corpus/SITE scellés, banques PARENT antérieures à la complétion puis hashées.
Les sources officielles intégrales des quatre familles et leurs fidélités sont exigées.
Les preuves conditionnées par chaque NOUVELLE requête restent à recalculer : aucune
matrice de scores d'une autre requête n'est transférée, aucun contexte526 emprunté.
Constat réel sans labels :440 dossiers/121 zones couverts, dont353/104 dans les caches
historiques ;2560 non couverts, exclus et comptés. L'écart vient des17 zones nouvelles
complètement préparées par le smoke, pas d'une nouvelle extraction dans ce lot.

## Budget et choix
Le code conserve les lots20 et leurs compagnons d'encodage du full canonique (smoke,
puis149 lots) ; filtrer une ligne ne doit pas changer le padding des lignes retenues.
Les recalculs query-only des quatre familles/global sources sont répétés par lot,
comme le code existant. Projection prudente du lot440≈139,64h, donc réduction AVANT labels.
Retenus51 :20 sorties smoke déjà possédées/réutilisées et31 nouveaux dossiers dans les
quatre lots canoniques20/25/49/58 (8/8/8/7 couverts).389 exclus budget,2949 exclus au total.
Ordre de choix : nombre de couverts par lot décroissant, puis ordinal ; pas un résultat
ou un label comme sélection. Coût estimé≈19467s=5,41h, réserve900s et marge25%, pas un SLO.
Échantillon couvert/budgeté géographiquement biaisé : aucune extrapolation de recall aux3000.

## Garde et preuve
Un gros worker/thread, même verrou v174, VM Stopped,8Gio/≥7,5Gio libre+inactive,
une escalade12Gio/≥9Gio pour préparation couverte ou génération ; marge256Mio,
RSS+Metal conservateurs, disque≥10Gio, TMP privé et tous les essais/reçus conservés.
Si hydraulic redémarre/incertaine : SIGTERM enfant possédé, sortie confirmée, attente60s
et réadmission, aucune action VM. L'autre agent ne doit pas être arrêté/modifié.
Interdictions exécutables : pas de Tantivy extraction communale, pas d'encodage SITE/PARENT ;
corpus complets copiés, banques existantes en lecture seule, seules requêtes encodées.
Avant les nouveaux dossiers : replay réel20/20 des features51 (dtype/bit à bit), activité
lossless et neuf contextes, comparaison aux références smoke déjà obtenues, STOP si écart.
Puis native100 national original82 772 691 documents, coefficients historiques affectés,
plafond absolu100 uniques, aucun positif injecté ni apprentissage/source de remplacement.

## Fin et audit descriptif
Chaque tâche reprend exclusivement un résultat SHA vérifié avec reçu enfant possédé0.
Un sceau complet du lot retenu est exigé avant d'ouvrir les seuls labels retenus :
ledger folds2/3/4 + IDs du lot, comparaison au sidecar scellé sans correction des vérités.
L'autorisation subset ne modifie PAS le gate3000 du mode full/évaluation historique.
Présences exact/SIREN/site/union exact-ou-site et rangs par dossier, dénominateur retenu
sans filtre. Coefficient-OOF seulement, pas encoder-OOF ni certification nationale.
Rapport automatique `docs/train3000_covered_result.{json,md}` + ligne handover + commit
limité à ces chemins, sans absorber code/index/handover non committé de l'autre agent.
Aucun --full, re-tirage3000, fit, web, nouvelle dépendance, modification526/pending/labels.
Un lancement détaché, une attente bloquante jusqu'à terminaison, sans polling agent.
