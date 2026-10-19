# Lot A — COMPLETE

{"total": 3000, "cache_covered": 435, "missing_preparation": 2565, "excluded_budget": 385, "excluded_total": 2950, "retained": 50, "already_generated": 20}

Générés : 50 ; métriques observées : {"generation_seconds": 921.5729282922111, "generation_seconds_per_query": 18.431458565844224, "preparation_seconds": 9904.03004512377, "preparation_seconds_per_query": 198.08060090247542, "new_generation_seconds": 580.2303532068618, "new_generation_seconds_per_query": 19.34101177356206, "generation_peak_rss_bytes": 9814081536, "preparation_peak_rss_bytes": 5082202112, "elapsed_controller_seconds": 3740.3150783749297, "process_times_exclude_resource_waits": true}.
Présences descriptives exact/SIREN/site/union : {"exact": 49, "siren": 49, "site": 45, "site_union": 49}.
Rangs par dossier et références : `train3000_covered_result.json`.
Aucune extraction communale/SITE/PARENT ; scores et quatre contextes de requête recalculés depuis les sources officielles existantes.
Sélection géographiquement couverte puis budgetée, NON représentative des3000 ; exclus sans prédiction, pas des succès.
Coefficient-OOF seulement, pas encoder-OOF, aucun recall national/SLO promis ; full3000 non lancé.

## Vérification finale et écarts
50 sorties/reçus exit0 et5000 candidats uniques revérifiés :30 nouveaux dossiers,20 repris.
Exact top1 :35/50 ; rang exact maximal observé16, un absent ; rangs exact/SIREN/site/union par dossier dans le JSON.
Caches réellement complets :435/3000, après exclusion générique de cinq dossiers dont le corpus ne correspond pas au census national. Corpus incomplet conservé, jamais complété/pruné/forcé.
Budget estimé5,40h ; deux contrôleurs successifs ont duré10913,532s=3,032h (10887,682s=3,024h de workers, attentes ressources0s).
Écart : un second lancement après le STOP de complétude, une attente bloquante par lancement, sans polling agent ; toutes les tâches compatibles reprises par reçus, aucun changement des workers scientifiques.
Les temps du tableau automatique sont ceux des processus réussis (dont20 générations antérieures réutilisées). Pour les30 nouveaux :19,341s/requête ; cinq escalades8→12Gio ajoutent94,994s, soit22,507s/requête opérationnelle.
Pic RSS nouvelle génération9332473856octets=8,692Gio ; pic9,140Gio du lot inclut les20 reçus antérieurs ; préparation4,733Gio.
Replay20 : vecteurs51/dtypes bit à bit et activité/neuf contextes identiques, écart0 ; zéro nouvelle extraction/encodage SITE/PARENT confirmé pour les cinq préparations.
Preuves de fin et deux sorties contrôleur : référence `post_run_verification` dans le JSON ; aucun processus de ce lot restant, aucun3000 lancé.
