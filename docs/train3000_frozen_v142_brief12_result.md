# Brief12 — projection corrigée, STOP mémoire pendant préparation

## Fait / choix
`5592578c` corrige `pipeline.selected` avec la même projection Parquet que le
producteur scellé :8 colonnes CRM, filtres folds2/3/4 et IDs sélectionnés, sans labels.
Comparaison réelle20/20 aux CRM scellés en1,251s ; code compilé, recette inchangée.
Clés des tâches liées au code (anciens essais conservés), métriques préparation/génération
séparées à partir des reçus possédés, démarrage inclus et attente ressources exclue.

## Preuve réelle
- Quatre fidélités20/20/erreur0 conservées ; numérique51 bit à bit20/20.
- Cycle1/3 détaché PID82661 : une attente bloquante502,540s, sortie possédée1.
- SHA du gel :3681 fichiers/148479301179octets conformes,164,308s, sortie possédée0.
- Préparation336,869s PARTIELLE : activité achevée, numérique interrompu lors de
  `engine.prepare_zone/EXTRACTING_FULL_COMMUNE`, avant OSM/BANO et génération native.
- Pic RSS arbre échantillonné7165083648octets =6,673Gio ; comptabilité prudente
  RSS+Metal de la garde atteint le seuil7,75Gio (plafond8Gio moins256Mio).
  Maximum enregistré par télémétrie8307523584octets ; la garde peut agir entre
  échantillons. Pas de dépassement physique8Gio revendiqué, RSS/Metal pouvant se recouvrir.

## Reste / risques
Garde8Gio terminale selon la politique conservée : pas de cycle2, pas d'escalade
préparation12Gio, pas de réduction de commune ni de substitution de preuve.
Smoke0/20 : s/requête et pic natif complet NON mesurés, génération non démarrée.
ETA préparation/génération NON mesurée ; pas ready.json ni commande full validée.
Aucun3000 lancé, action VM, fit, web, push, changement de labels/source/pending/travail étranger.
Ce résultat ne prouve pas qu'une exécution équivalente sous8Gio soit impossible.
Reçus, télémétrie et hashes : `docs/train3000_frozen_v142_brief12_result.json`.
