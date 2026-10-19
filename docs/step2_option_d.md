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

En attente du smoke et du run ; aucun chiffre nouveau annoncé à ce stade.
