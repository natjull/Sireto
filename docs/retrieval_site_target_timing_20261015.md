# Chronométrage site-target — 15 octobre 2026

Contrat préalable `996ce66b7c2ac0908a217690b26a99ebdeab50fc` ; namespace `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v173/site_target_timing_20261015`.
**Une partition interne (0), 20 recherches réelles ; aucun recall.**

| Phase | Paires / préparation (s) | Optimiseur (s) | Total interne (s) | Pic RSS Gio |
|---|---:|---:|---:|---:|
| v32 | 1442.63 | 0.036 | 1442.67 | 7.221 |
| v71 | 1129.04 | 0.052 | 1129.15 | 7.098 |

v32 : 284 requêtes train, 100949 paires, 17 cibles changées, 1 cibles sans signal numérique.

v71 : 284 requêtes train, 127173 paires, 17 cibles changées, 1 cibles sans signal numérique.

Génération : **20/20**, 2000 candidats, cap100 unique vérifié ; 7 requêtes des dix plus grandes communes du census.
Préparation des contextes : 18.84 s ; 20 processus de recherche, démarrages inclus : **509.26 s** ; kernels/préparateurs/recherche internes : 479.64 s.
Pic RSS toutes phases : **7.221 Gio** (garde supervisée 8 Gio, un thread, aucun processus lourd parallèle).
Signatures d’entrée revérifiées : 1439. Inventaire national : 352 fichiers conformes aux tailles/mtime historiques et inchangés au contrôle final. Les 28 nouveaux poids V32 sont inchangés dans le résiduel V71.

## Extrapolation conditionnelle : cinq modèles + 526 recherches

- Apprentissage : 56.6–267.9 min.
- Génération : 142.9–381.0 min.
- **Total : 199.5–648.9 min** ; pic attendu 7.22–8 Gio/processus, sans garantie que les 506 autres tiennent sous ce plafond.
- Basse : mining des paires partagé sur tous les éligibles, cinq optimiseurs, 0,75× coût de recherche stratifié grandes/autres communes. Haute : cinq minings complets ×1,25 et 2× coût stratifié de recherche.
- Scénarios d’ingénierie, pas IC de durée ni SLO ; état froid/chaud et autres charges non contrôlés. Préparation des contextes du panel complet et validation finale non mesurées séparément : prévoir marge dans la borne haute.
- Réutilisables : Gemma/adaptateur/vecteurs, index national, univers complets et 51 signaux publics, sources officielles et contextes. Aucun besoin de réencodage Gemma ou nouveau scan national de génération d’univers.
- À recalculer : paires avec nouvelle cible, 5 V32 puis 5 résiduels V71, bonus OSM/BANO dépendant des poids, requêtes Tantivy et sorties526 ; anciennes sorties V142 non réutilisables sous les nouveaux poids.
- Les paires déjà produites pour ce pilote peuvent être réutilisées ; estimation ci-dessus depuis les caches publics, sans déduire ce gain. Le reste des paires est indépendant du modèle interne et peut être partagé.

## Limites / reproduction

- Choix explicité : « au moins3 grandes communes » signifie trois communes distinctes ; rejet déterministe des tirages réservés avec commune répétée, puis 17 autres requêtes sans remise, même graine20261015, sans labels.
- Modèle interne0 fixe pour les20, pas de routage label-side, pas de prétention OOF ni qualité nouvelle mesurée.
- Partitions et exclusions historiques reprises ; concurrents re-minés avec la procédure et graines historiques ; L2=0,01.
- Les identifiants de cible n’entrent que dans la supervision ; aucun label réécrit, aucune injection en recherche.
- VM hydraulic : arrêtée selon inventaire local CatNat en lecture seule (`LIMA_HOME=/Volumes/CATNAT_DATA/catnat-lima`), aucune commande de démarrage/arrêt ; calcul conservateur un thread.
- Gros fichiers : signatures taille/mtime/inode + hashes hérités ; petits bindings vérifiés SHA256 ; pas de nouveau hash complet des segments.
- Commande réelle détachée : `.venv/bin/python -u scripts/retrieval_site_target_timing.py` ; publication : même commande avec `--report`. Ne pas relancer sur ce namespace existant.
- Suivi : `tail -n 20 /Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v173/site_target_timing_20261015/timing.log` ; résultats détaillés par requête dans `queries/*/result.json`.
- Recalcul complet et toute évaluation : **attendre accord Nathan**.
