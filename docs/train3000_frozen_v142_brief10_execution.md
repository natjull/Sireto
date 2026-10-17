# Brief10 — exécution gardée, pas encore une validation

Autorité : `sireto-site-target-10.md`. V142 `36c7a476` reste inchangé.
Code du jalon : auto-référence `git log -1 --format=%H -- scripts/train3000_frozen_v142_supervisor.py`.

## Code livré

- `scripts/train3000_frozen_v142_numeric.py` : V14/V18/V19 → V23/V32/V57/V68/V71/V75,
  modèles existants seulement, matrices de scores recalculées, banques documentaires
  SITE/PARENT distinctes et vérifiées. Gemma original2.14.1/Transformers5.19.0,
  padding/batches historiques conservés pour les pilotes ; tolérance1e-7 fixée avant run,
  égalité bit à bit et écart maximal publiés séparément. Nouveau cache parent mmap borné.
- `scripts/train3000_frozen_v142_osm.py` : constructeur national PBF/category V129,
  projection V143 et deltas pondérés, sans réemploi du spool des526.
- `scripts/train3000_frozen_v142_brief10.py` : fidélités sérielles activité → OSM → numérique51.
- `scripts/train3000_frozen_v142_contexts.py` : autres contextes obligatoires V88/V98/V130/V133/V137,
  corps historiques ; seuls scope/guards/I/O rebindés. COG local vérifié, aucun téléchargement.
  Les préconditions historiques d'exits526 ne sont pas maquillées en faux reçus :
  une fin de sous-préparation en processus est explicitement distinguée d'un exit enfant.
- `scripts/train3000_frozen_v142_pipeline.py` et `..._supervisor.py` : quatre préparations,
  autres contextes, recherche V142 native100 avec coefficients historiques affectés,
  smoke nouveau20, ETA puis rapport/commit isolé/handover automatiques.
  Évaluation distincte seulement après3000 sorties et reçus possédés code0 scellés.

## Ressources et gates

Fidélités/smoke :6Gio, admission libre+inactive≥7,5Gio ;8Gio si hydraulic confirmée
Stopped. Attente60s illimitée, aucun contrôle de la VM. Après arrêt préventif6Gio,
attendre l'admission8Gio plutôt que répéter un essai6Gio infructueux ;8Gio dépassés : STOP.
Un worker, un thread, marge256Mio, suivi RSS toutes50ms et mémoire Metal conservatrice,
disque≥10Gio, TMP privé, sources/reçus/essais conservés, aucun pending préexistant touché.
La politique ultérieure du full reste8Gio avec VM arrêtée, une seule escalade génération12Gio
si libre+inactive≥9Gio ; son dépassement est terminal. Aucune exécution full dans ce brief.

`--prepare-ready` exécute les trois fidélités restantes, vérifie BANO20/20 existant,
rehash complet du gel, puis uniquement les20 nouveaux smokes prédésignés. Les coûts
préparation/génération sont mesurés séparément ; extrapolation150 lots de20, hors attentes
illimitées, sans garantie de SLO ou de durée≤30h. Pas de label en préparation/génération.
Le `ready.json` et la commande future `--full` ne sont publiés qu'après tous ces gates.
`--full` refuse avant ce sceau ; il n'est jamais appelé automatiquement par `--prepare-ready`.

## État constaté avant remplacement du contrôleur initial

BANO20/20 contextes et deltas : preuve `33053f67` déjà mesurée, erreur0.
Autres familles : aucune comparaison achevée ; smoke0/20, ETA et écart numérique inconnus.
PID64619 a attendu au moins1080s, zéro admission/zéro enfant lourd ; arrêt propre pour
passer au superviseur unifié. Archive `brief10/controller_revisions/serial_to_unified/`.
Vérifications de démarrage uniquement :6 scripts compilés,20 autorités/8 corps compilés
sans global manquant, plus3 helpers Gemma/packing/query vérifiés. CLI `--help` CPU et Metal
code0 ; Torch2.14.1/Transformers5.19.0, MPS disponible. Ce n'est PAS une fidélité ni un smoke natif.

Continuation détachée depuis la racine du dépôt :
```bash
nohup env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/train3000_frozen_v142_supervisor.py --prepare-ready >> /Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v174/train3000_frozen_v142_20261017/extension_20261017/brief10_unified.log 2>&1 < /dev/null &
```
Rapport mesuré futur : `docs/train3000_frozen_v142_brief10_result.{json,md}`.
Ni ces fichiers ni un ETA ne sont préremplis comme succès. Les anciens rapports restent historiques.
