# Étape2 optionD — protocole pré-déclaré

Autorisation : `sireto-step2-5.md`. Expérience aval de développement, pas une
réouverture de la recherche V142 : son sceau `36c7a476`, ses526×100 candidats,
labels et métriques restent inchangés. Aucun accès/écriture au namespace v174,
aucune VM/web/installation ni nouvelle preuve Gemma/BANO/OSM/activité.

## G0, fixé avant toute mesure

Index national Tantivy `hierarchical_retrieval_v3/4b332899d8698354`, lecture seule.
Trois canaux locaux,100 chacun : BM25 nom (`names`), adresse complète
(`addresses`), et nom+adresse (`names`,`addresses`). INSEE si indexé, sinon CP,
via les primitives historiques. Texte CRM nettoyé des identifiants littéraux,
jamais de syntaxe de requête CRM exécutée. Fermés conservés.
Un quatrième canal retourne les100 points publics les plus proches dans un
rayon FIXE de5000m lorsque la géométrie CRM existe déjà : géométries v17 pour
le pilote, `ban.parquet` historique query-only pour les nouveaux dossiers.
Aucun nouveau géocodage. Scan exact des blocs Morton officiels v126 dont les
boîtes intersectent le rayon ; distances sur les coordonnées originales,
pas de qualité transférée entre sources. Élagage exact par borne inférieure
point/boîte et meilleur100 courant, égalités conservées ; aucun changement
des100 voisins par rapport au scan intégral des blocs du rayon. Aucun label
dans ces interfaces.
Fusion RRF somme, poids1, k60 ; égalités SIRET lexical, sortie≤100.
Si moins de100 sites naturels : compléter seulement avec BM25 national
nom+adresse (pas remplacer/réordonner les sites déjà émis). Pour une entrée
sans texte, repli géographique INSEE/CP, puis requête nationale sans texte ;
ce cas non informatif sera compté, jamais corrigé par un positif.
Objectif de coût2s/requête ; latence de recherche et coût complet des workers
(démarrage/index/géométrie inclus) séparés. G0 n'est pas une variante produit.

## Population et progression, fixées avant génération

Reproduire en lecture seule `train3000_frozen_v142_cohort.population()` sans
appeler son sealer :17427 dossiers,11483 entreprises, folds2/3/4 ; exclure les
405 entreprises pilote, test4323, control962, folds0/1 AVANT projection CRM.
Les autorités et exclusions sont celles du tirage `d54cec2c` ; ne jamais lire
les fichiers de membership/labels des jeux fermés. Copie du tirage3000 depuis
`docs/train3000_frozen_v142_sample.json`, hash des IDs
`f2a359afb834b02d5ca30d905808b62ea2d613eb038acbb9eaa1c82e8df2deb3`.
Les premiers500/1000 sont des préfixes imbriqués de ces3000, ordonnés par
SHA256(`20261017|content-curve|query_id`), indépendant des labels. Les autres
14427 suivent dans le même ordre SHA. Labels dans un sidecar séparé, pas
transmis aux workers de génération/features. Aucun positif ajouté.
Après les3000, étendre si temps déjà consommé +1,25×coût moyen observé×reste
≤6h ; plafond réel de6h de workers G0, attente concurrente comprise. Si le
budget ne le permet pas, `tout` désigne explicitement le volume effectivement
complet, sans présenter une courbe17427 inventée. Aucun choix selon recall.

## M_content, fixé avant fits

Recette R3 exacte : `rank:ndcg`,200 arbres, profondeur3, lr0,05,
min_child_weight10, L2=10, L1=1, hist/max_bin128, sous-échantillonnages1,
lambdarank mean/8, graine20261017, un thread. Pas de HPO/arrêt sur test.
85 features =96 v2 moins les11 dépendances d'ordre/score natif :
`v142_rank`, `v142_reciprocal_rank`, `v142_score`, `v142_score_relative`,
`v142_delta_best`, `v142_delta_next`, `v142_top1_top2_gap`,
`same_siren_as_native_top1`, `native_top1_siren_count`,
`v2_distance_minus_native_log`, `v2_siren_first_native_rank`.
TOUTES les autres features v2 sont conservées dans leur ordre initial ; aucune
feature rang/score G0 n'est ajoutée, afin de ne pas transférer une échelle G0
vers V142. Les agrégats de liste sans ancre native restent calculables pour
n'importe quelle liste. Les deux scores BAN probabilistes restent manquants.
Noms officiels publics, historique de fermeture officiel, stock public et
étude SIRENE identiques à v2 ; pas d'ID, alias CRM ni information label-side.
Matrices scellées avant ouverture du sidecar de cibles par le fit. Tous les
groupes de100 restent présents, y compris les groupes entièrement négatifs
(vérité absente), qui ne donnent aucun positif au LambdaMART.

## Empilement et évaluation, fixés avant fits

(a) Score M_content externe seul sur les100 V142, sans fit sur le pilote.
(b) R3 identique,21 features : sept natifs v142 ci-dessus, sept analogues du
score M_content (rang, inverse rang, score brut, score relatif min/max,
écart au meilleur, écart au suivant DANS L'ORDRE CONTENT, gap top1/top2), puis
les sept scènes V1 : `candidate_same_siren_count`,
`same_siren_as_native_top1`, `native_top1_siren_count`, `unique_siren_count`,
`same_address_count`, `name_gap_best_other`, `street_gap_best_other`.
Le M_content de chaque volume est entraîné exclusivement hors405 entreprises,
et fixé avant tous les fits pilote. Empilement en OOF entreprise folds2/3/4,
aucun entraînement ni normalisation sur les entreprises test.
Comparaisons aux mêmes sélections V142 brut353 et R3 v2 368/407 : cinq vues,
Wilson95 descriptif par requête, paires gains/pertes et McNemar bilatéral,
global/par fold. Holm sur exact/union des deux méthodes, tous volumes,
contre les deux références ; dépendance intra-entreprise non corrigée.
G0 pilote : présence exact/SIREN/union, cinq vues détaillées, intersections
avec V142, coûts, RSS. G0 et M_content ne modifient PAS le pool d'évaluation.

Confiance pour CHAQUE volume et chaque méthode : mêmes rôles entreprise
60/20/20 du diagnostic, seeds20261017+100+fold. Pour(b), ranking fit sur60%,
scène20 features historiques/logistique C1 sur scores3-plis OOF dans ces60%,
isotonique sur20%, seuil sur les20% restants, test externe jamais utilisé.
Pour(a), le ranking externe n'a vu AUCUNE entreprise du pilote : il reste
fixe ; les60% ne servent qu'au fit de la scène, sans réentraîner M_content.
Cible UNION pour scène/isotonique/seuil, ranking EXACT inchangé. Seuil de
couverture maximale à précision empirique union≥98%, sans minimum d'effectif
ou contrainte Wilson ajoutée après coup. Publier aussi précision exacte des
mêmes SUR, couverture et IC. Comparer marge et scène+isotonique sans choisir
la politique sur les tests.

## Ressources et custody

Namespace propre `H/step2_option_d_20261017/`, sous-processus séquentiels,
un thread, DuckDB512Mio, garde RSS5,75Gio (<6Gio), fichiers neufs/scellés.
G0 par256 requêtes ; hydratation/features regroupées par au plus1024 requêtes
pour amortir les scans officiels, sans agrégat entre requêtes.
Avant toute phase susceptible de dépasser2Gio, attendre60s tant que le
superviseur `train3000_frozen_v142_supervisor.py` OU un descendant a RSS>4Gio.
Aucun signal ni contrôle envoyé à l'autre agent/VM ; aucune écriture v174.
Smoke réel20 nouveaux dossiers et36 pilotes, puis complet ; pas de tests
unitaires. Sources SHA avant/après, modèles/cibles/partitions tracés, aucune
promotion production. Le pilote et les choix de features sont consommés,
l'amont V142 supervisé n'est pas full-OOF : aucune certification nationale.


## Résultats

**Développement, pas production ; aucune modification du retrieval V142.**

- G0 sur526 pilotes : exact 478, siren 486, operational 481, site_operational 393, union 486.
- Intersection moyenne avec V142 :44.01/100 ; recherche0.2213s/q, p950.4563s ; workers complets0.2513s/q ; pic3044.4Mio.
- Apprentissage :17427/17427 dossiers générés ; G0 workers4586.2s, projection préalable avec marge25%5832.7s ; extension=True.

| Volume | Méthode | Exact | SIREN | Opérationnel | Site | Union |
|---:|---|---:|---:|---:|---:|---:|
| — | V142 | 353 | 406 | 377 | 315 | 391 |
| — | R3_v2 | 368 | 419 | 392 | 326 | 407 |
| 500 | content | 359 | 397 | 378 | 301 | 384 |
| 500 | stack | 378 | 420 | 403 | 325 | 413 |
| 1000 | content | 358 | 398 | 378 | 296 | 383 |
| 1000 | stack | 379 | 420 | 406 | 323 | 414 |
| 3000 | content | 361 | 403 | 380 | 301 | 386 |
| 3000 | stack | 385 | 424 | 407 | 323 | 415 |
| 17427 | content | 363 | 408 | 382 | 298 | 389 |
| 17427 | stack | 373 | 414 | 394 | 315 | 405 |

### Positifs réellement disponibles

- 500 :454 positifs présents ;46 groupes sans positif, conservés dans les mesures/fit, sans injection.
- 1000 :915 positifs présents ;85 groupes sans positif, conservés dans les mesures/fit, sans injection.
- 3000 :2740 positifs présents ;260 groupes sans positif, conservés dans les mesures/fit, sans injection.
- 17427 :15956 positifs présents ;1471 groupes sans positif, conservés dans les mesures/fit, sans injection.

### Confiance, cible union98% en calibration seulement

| Volume/méthode | Politique | SUR | Union, Wilson95 | Exact, Wilson95 | Couverture, Wilson95 |
|---|---|---:|---|---|---|
| 500/content | margin | 37 | 36/37 (97.30 %, IC95 [86.18;99.52]) | 36/37 (97.30 %, IC95 [86.18;99.52]) | 37/526 (7.03 %, IC95 [5.15;9.55]) |
| 500/content | scene_isotonic | 0 | non définie | non définie | 0/526 (0.00 %, IC95 [0.00;0.73]) |
| 500/stack | margin | 89 | 83/89 (93.26 %, IC95 [86.06;96.87]) | 83/89 (93.26 %, IC95 [86.06;96.87]) | 89/526 (16.92 %, IC95 [13.96;20.36]) |
| 500/stack | scene_isotonic | 37 | 35/37 (94.59 %, IC95 [82.30;98.50]) | 35/37 (94.59 %, IC95 [82.30;98.50]) | 37/526 (7.03 %, IC95 [5.15;9.55]) |
| 1000/content | margin | 39 | 38/39 (97.44 %, IC95 [86.82;99.55]) | 38/39 (97.44 %, IC95 [86.82;99.55]) | 39/526 (7.41 %, IC95 [5.47;9.98]) |
| 1000/content | scene_isotonic | 59 | 57/59 (96.61 %, IC95 [88.46;99.07]) | 56/59 (94.92 %, IC95 [86.08;98.26]) | 59/526 (11.22 %, IC95 [8.80;14.20]) |
| 1000/stack | margin | 63 | 61/63 (96.83 %, IC95 [89.14;99.13]) | 61/63 (96.83 %, IC95 [89.14;99.13]) | 63/526 (11.98 %, IC95 [9.47;15.03]) |
| 1000/stack | scene_isotonic | 27 | 25/27 (92.59 %, IC95 [76.63;97.94]) | 23/27 (85.19 %, IC95 [67.52;94.08]) | 27/526 (5.13 %, IC95 [3.55;7.37]) |
| 3000/content | margin | 69 | 67/69 (97.10 %, IC95 [90.03;99.20]) | 67/69 (97.10 %, IC95 [90.03;99.20]) | 69/526 (13.12 %, IC95 [10.50;16.27]) |
| 3000/content | scene_isotonic | 19 | 18/19 (94.74 %, IC95 [75.36;99.06]) | 17/19 (89.47 %, IC95 [68.61;97.06]) | 19/526 (3.61 %, IC95 [2.32;5.57]) |
| 3000/stack | margin | 111 | 103/111 (92.79 %, IC95 [86.42;96.30]) | 102/111 (91.89 %, IC95 [85.31;95.68]) | 111/526 (21.10 %, IC95 [17.83;24.79]) |
| 3000/stack | scene_isotonic | 20 | 18/20 (90.00 %, IC95 [69.90;97.21]) | 16/20 (80.00 %, IC95 [58.40;91.93]) | 20/526 (3.80 %, IC95 [2.47;5.80]) |
| 17427/content | margin | 56 | 54/56 (96.43 %, IC95 [87.88;99.02]) | 54/56 (96.43 %, IC95 [87.88;99.02]) | 56/526 (10.65 %, IC95 [8.29;13.57]) |
| 17427/content | scene_isotonic | 39 | 37/39 (94.87 %, IC95 [83.11;98.58]) | 37/39 (94.87 %, IC95 [83.11;98.58]) | 39/526 (7.41 %, IC95 [5.47;9.98]) |
| 17427/stack | margin | 61 | 58/61 (95.08 %, IC95 [86.51;98.31]) | 58/61 (95.08 %, IC95 [86.51;98.31]) | 61/526 (11.60 %, IC95 [9.13;14.62]) |
| 17427/stack | scene_isotonic | 30 | 28/30 (93.33 %, IC95 [78.68;98.15]) | 26/30 (86.67 %, IC95 [70.32;94.69]) | 30/526 (5.70 %, IC95 [4.02;8.03]) |

### Paires finales et limites

- content vs V142 : exact +53/−43 p=0.3584, Holm=1, union +41/−43 p=0.9132, Holm=1.
- content vs R3_v2 : exact +36/−41 p=0.64879, Holm=1, union +24/−42 p=0.035583, Holm=0.74724.
- stack vs V142 : exact +41/−21 p=0.015134, Holm=0.33295, union +34/−20 p=0.075905, Holm=1.
- stack vs R3_v2 : exact +27/−22 p=0.56817, Holm=1, union +16/−18 p=0.86417, Holm=1.

### Recommandation unique

Ne pas promouvoir cet empilement ; privilégier ensuite un signal sémantique propre au site plutôt qu’augmenter encore ce même apprentissage de contenu.

- Cinq vues, folds, Wilson, McNemar et tous volumes publiés dans le JSON. Holm ne corrige pas la dépendance intra-entreprise ni la consommation du pilote. Pas de sous-ensemble choisi après résultats.
- Content reste externe à toutes les405 entreprises ; empilement et calibration sans entreprise partagée entre fit/calibration/test. Risque de décalage G0→V142 conservé, pas masqué.
- BAN ancien query-only réutilisé, aucun géocodage nouveau ; v17 pilote conservé. Dates et agrégats officiels identiques à v2. Les deux scores BAN probabilistes sont NaN.
- Pas de Gemma/BANO/OSM/activité dans G0, aucune écriture v174, VM/web/installation/positif injecté/jeu fermé. Index segments contrôlés par empreintes avant/après (pas nouveau SHA intégral98Gio), manifestes et sources de features/layout par SHA.
- Total5360.8s ; pic maximum des workers3342.1Mio ; sources et gel vérifiés avant/après. Poids, matrices, cibles et rôles scellés dans `H/step2_option_d_20261017/`.
- Reproduction : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_option_d.py run --output-dir <dossier_neuf_sous_H/step2_option_d_20261017>` ; ajouter `--smoke` pour20 nouveaux/36 pilotes.

## Rejeu indépendant

{"passed": true, "model_hashes_and_original_targets": 64, "company_fit_score_audits": 108, "primary_query_score_vectors_replayed": 4208, "confidence_queries_replayed": 4208, "union_target_role_records_verified": 24, "smoke_full_matching_feature_cells": 170000, "warnings": [], "implementation": {"path": "/Users/nathanjullia/Documents/Projets/SIRETO/scripts/step2_option_d_verify.py", "size_bytes": 17602, "mtime_ns": 1792395193076738880, "inode": 73464404, "sha256": "0e9d60729cb2ae0cac002958945b5633e77d260690edd271284873b79dd7aad6"}}
