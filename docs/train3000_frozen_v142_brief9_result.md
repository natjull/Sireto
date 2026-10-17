# Brief9 — attente ressources avant4Gio, pas un échec de fidélité

1. BANO : **20/20 contextes et deltas pondérés identiques**, erreur maximale0 ; **7,5277s**, pic enfant réel **3,2387Gio** ; reçu possédé sortie0.
2. Activité : scan national19 années/4 177 176 propriétaires, puis arrêts préventifs mémoire avant comparaison : **3,5022Gio/156,18s**, puis **3,7673Gio/156,75s** (RSS échantillonné).
3. Le plafond4Gio n'est pas déclaré atteint : watchdog initial3,5 puis relance unique3,75, math/critère de fidélité inchangés ; marge conservatrice et preuves conservées. Cela ne prouve pas qu'une exécution<4Gio soit impossible.
4. Arrêt après la seconde tentative : activité0/20, OSM0/20, numérique510/20, smoke0/20 ; ETA préparation/génération non mesurées. Aucune action hydraulic, aucun run complet.
5. Le superviseur unifié n'est **pas prêt**, donc aucune commande de lancement complet fournie. PID61903 arrêté proprement, tous les enfants de ces minis terminés/reçus scellés.

## Livré / non livré

- `scripts/train3000_frozen_v142_fidelity.py` : contrôleur possédé des minis
  BANO/activité, un thread, logs/reçus, plafonds mémoire/disque ; pas de mode full.
- `scripts/train3000_frozen_v142_activity.py` : corps V112 inchangé, références
  nationales exactes, cutoff et IDF inchangés ; découpage projection V115 exact
  S9/float64. Le corps de construction a été exécuté, sans sortie complète publiée sous
  le seuil conservateur, **aucune comparaison ni projection activité achevée**.
  Seuls racine/scope/guard et thread DuckDB sont liés à l'exécution privée.
- `scripts/train3000_frozen_v142_osm.py` : adaptateur rédigé pendant le mini
  activité, **non exécuté et non branché au CLI** après l'arrêt précédent.
  Il décrit la reprise du handler PBF national V129 avec la transformation
  historique catégories séparées puis les deltas V143 ; ne pas le présenter
  comme validé ou prêt pour3000. Le vieux spool objets526 n'est pas un substitut.
- Numérique51 V14→V75 : adaptateur non réalisé, fidélité non exécutée.
- `scripts/train3000_frozen_v142_bano.py` : global `Counter` original ajouté au
  namespace du compilateur. Le mini réel l'avait fourni explicitement ; le
  compilateur standard résout maintenant le même global. Corps AST inchangé
  hors binding Counter, même SHA du corps historique et mêmes objets Python
  que ceux exécutés dans la preuve20 ; aucun nouveau mini après l'arrêt mémoire.
- Générateur/superviseur unifié, smoke20, ETA et commit automatique final :
  non réalisés. Les rapports `train3000_frozen_v142_result.*` restent historiques.

## Provenance et vérifications réellement exécutées

Commandes :

```bash
nohup env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/train3000_frozen_v142_fidelity.py --run bano ... &
nohup env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/train3000_frozen_v142_fidelity.py --run activity ... &
```

La seconde commande a été exécutée deux fois, une seule relance préventive,
jamais après une non-fidélité mesurée. La révision3,5→3,75 n'a changé que la
constante watchdog (AST normalisé comparé) ; plafond4Gio et tolérance activité
préfixée1e-7 inchangés. Les sources exactes de chaque tentative sont archivées.

Namespace :
`H/v174/train3000_frozen_v142_20261017/extension_20261017/brief9/`.

- `bano_20261017_135141_05f29fcc/{result,receipt}.json` : preuve20 et sortie0.
- `activity_20261017_135731_6545e8e6/receipt.json` : sortie-15 possédée,
  watchdog3,5 ; sources `implementation.py`, `adapter.py` archivées.
- `activity_20261017_140701_889ccb76/receipt.json` : sortie-15 possédée,
  watchdog3,75 ; mêmes archives, aucun troisième essai.
- `activity_watchdog_revision.json`, `bano_namespace_fix/revision.json` :
  chaînes de provenance non réécrites, origine Counter standard explicitée.

Le JSON frère contient les SHA exacts, métriques, reçus, seuils et champs nuls
pour les mesures absentes. **3 681 signatures/taille de fichiers du gel**
revérifiées ; pas de nouvelle lecture SHA intégrale148Go revendiquée.
Les workers ne lisent aucun label ; les matrices51 gelées sont des entrées
amont de la **preuve isolée BANO**, pas une reconstruction numérique51.
Ni source/indice/modèle/cache526, ni label, ni pending préexistant, ni travail
étranger modifié. Aucun fit/web/push, aucune suite de tests unitaires.

Commit BANO : `33053f67`. Commit du présent arrêt/adaptateurs et ligne handover :
`git log -1 --format=%H -- docs/train3000_frozen_v142_brief9_result.json`.
