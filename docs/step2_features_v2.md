# Étape2 — features v2 et confiance métier

## Pré-déclaration avant toute mesure (17 octobre 2026)

Population526, folds2/3/4 ; listes ordonnées100 de V142 `36c7a476` intouchées.
Matrice V1 de46 features conservée. R3 `rank:ndcg`,200 arbres, profondeur3,
LR0,05, min_child_weight10, L2=10, L1=1, histogrammes128, seed20261017,
un thread : paramètres du diagnostic `a5361c93`, aucun HPO/early stopping.
Les ajouts ci-dessous testent des encodages et informations supplémentaires
aux46 ; leur nouveauté vis-à-vis de TOUT l'amont V142 n'est pas garantie.
Les ablations antérieures ne prouvent pas l'inutilité du nom/adresse/géo.

### Liste exacte et figée des features ajoutées

**Adresse précise —27** :
`v2_number_exact`, `v2_number_within2`, `v2_number_both_missing`,
`v2_number_one_missing`, `v2_number_log_gap`, `v2_suffix_equal`,
`v2_candidate_range`, `v2_street_exact`, `v2_street_jaro`,
`v2_street_token_overlap`, `v2_commune_match`, `v2_distance_log`,
`v2_distance_minus_native_log`, `v2_crm_ban_score`, `v2_candidate_ban_score`,
`v2_crm_street_similarity`, `v2_crm_log_uncertainty`,
`v2_candidate_log_distance_precision`, `v2_crm_house`, `v2_crm_street`,
`v2_crm_commune`, `v2_candidate_house`, `v2_candidate_street`,
`v2_candidate_commune`, `v2_crm_declared_weak`, `v2_geo_missing`,
`v2_geo_admissible`.
Normalisation locale du contrat `fb604665`, numéros positifs/suffixes séparés ;
plage candidat non assimilée à un numéro unique. ±2 inclut l'égalité, suffixe
traité séparément. Voie exacte/Jaro/Jaccard, INSEE égal non vide.
Distances Lambert93 finies compatibles, y compris points communaux pour ces
features uniquement ; types séparés et admissibilité200/500m en entrée.
Distance relative = log1p(distance)−log1p(distance du premier V142).
Métadonnées BAN CRM v17 et étude SIRENE juillet pour le candidat, sans
transfert de qualité entre sources. Scores BAN probabilistes absents restent
NaN : similarité voie, incertitude et distance_precision ne sont PAS des scores.

**Nom propre de l'établissement —10** :
`v2_enseigne_jaro_max`, `v2_usual_jaro`, `v2_site_best_jaro`,
`v2_site_name_present`, `v2_site_name_exact`, `v2_enseigne_token_overlap`,
`v2_usual_token_overlap`, `v2_enseigne_other_count`,
`v2_site_name_gap_same_siren`, `v2_site_vs_ul_jaro_gap`.
Enseignes1/2/3 distinctes de dénomination usuelle établissement ; maximum
séparé puis meilleur des deux. Nombre d'AUTRES candidats partageant au
moins une enseigne normalisée ; écart de similarité au meilleur AUTRE site
du même SIREN (NaN si aucun) et écart à la similarité UL V1. Aucun alias,
identifiant littéral incorporé au texte retiré, diffusion publique uniquement.

**Statut et dates —9** :
`v2_is_active`, `v2_is_closed`, `v2_is_siege`, `v2_closure_age_log_days`,
`v2_closure_recency_vs_siren_days`, `v2_closure_missing`,
`v2_active_same_siren_pool_count`, `v2_active_same_siren_crm_commune_count`,
`v2_state_missing`.
Fermeture = dernier début de période historique F avec indicateur officiel
de changement d'état vrai, au plus tard le snapshot2026-08-01, uniquement
si l'état courant est F. Le dateDebut du stock courant seul n'est PAS une
fermeture. Âge = log1p(jours depuis fermeture) ; récence relative = écart
(en jours) à la fermeture la plus récente des candidats du même SIREN.
Comptes d'actifs : dans100 et dans TOUT le stock public de la commune CRM,
via agrégation officielle sans retourner/ajouter de nouveaux candidats.
INSEE CRM absent → compte communal NaN. Fermeture inconnue → NaN + indicateur.

**Scène par SIREN —4** :
`v2_siren_first_native_rank`, `v2_siren_candidate_count`,
`v2_siren_is_nearest`, `v2_siren_distance_excess_log`.
Premier rang V142 du groupe, taille du groupe, plus proche parmi ses points
admissibles (ex æquo tous vrais), log1p(distance)−minimum log1p du groupe.
Distance non admissible/absente → indicateurs de proximité NaN, jamais infini.
IDs réservés aux jointures/groupes, aucun encodage ou hash d'ID comme feature.

**Gemma conditionnelle —1** : `v2_gemma_cosine`.
Incluse seulement si des vecteurs CRM et de SITE candidat déjà persistés,
issus du même encodeur avec mapping vérifiable, couvrent TOUTES les526×100
paires. Les vecteurs SIREN ne seront pas présentés comme vecteurs de site.
Sinon famille sautée intégralement, sans réencodage. Audit local des banques,
coût d'audit réel et extrapolation explicitement conditionnelle au débit
historique d'encodage, sans prétendre mesurer un coût512 tokens absent.
Les documents officiels en cache ne sont utilisés que pour cet inventaire.

### Mesures et confiance fixées avant fits

- R3 V1 seul ; V1+chaque famille disponible seule ; V1+toutes les familles.
  Pas de sélection de sous-ensemble après les résultats. Cibles ranking EXACTES
  immuables,19 requêtes sans positif conservées ; cinq vues globales/par fold,
  Wilson95, gains/pertes et McNemar exact bilatéral contre V1 et ordre V142.
  Holm sur les comparaisons exact/union contre V1 (deux par variante ajoutée).
- Confiance V1 ET V1+v2 pour comparaison honnête : mêmes entreprises60% fit
  propre/20% isotonique/20% seuil dans chaque train externe, seed20261017+100+fold.
  Scène identique aux20 features du diagnostic, logistique C=1/max_iter1000,
  OOF entreprise3-plis DANS les60%, cible **union** ; isotonique sur20% disjoints,
  puis seuil de marge/probabilité maximisant couverture à précision union
  empirique≥98% sur les20% seuil seulement. Pas de minimum d'effectif/Wilson
  au choix du seuil, aucun seuil appris sur test ; aucune calibration exacte
  ensuite rebaptisée métier. Modèles du socle commun distincts des full-fits.
- Même SUR : publier précision union, exacte et trois autres vues, couverture
  et Wilson95, global/par fold. Aucun SUR → précision non définie.
- Plafond : reprendre les compteurs du contrat `fb604665`, dont labels exacts
  loin du CRM ; uniquement diagnostic label-side APRÈS construction des features.
  Une distance au CRM n'est ni une erreur démontrée du label ni un plafond
  mathématique de précision exacte. Plafond global réel = présence exact@100 ;
  une précision98% sur un sous-ensemble SUR n'est pas exclue par ce plafond.
- Sources en lecture seule, SHA avant/après ; nouvelle matrice construite par
  interface sans labels et scellée avant lecture du ledger/sidecar d'évaluation.
  Garde RSS7,5Gio, DuckDB512Mio/1thread ; dossiers neufs, logs détachés, smoke réel.
  Aucun test unitaire, génération, jeu fermé, VM, installation, web ou push.


## Résultats

**Mesures de développement, aucune promotion production.**

526×100 candidats,46+50 features ; Gemma sautée. Cinq vues et folds, Wilson et paires détaillés dans le JSON.

| Variante | Exact | SIREN | Opérationnel | Site | Union | Gains/pertes exact vsV1 | p exact / union vsV1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 361 | 409 | 386 | 320 | 399 | +0/−0 | 1 / 1 |
| plus_address | 363 | 409 | 388 | 320 | 400 | +6/−4 | 0.7539 / 1 |
| plus_site_name | 357 | 405 | 382 | 319 | 394 | +2/−6 | 0.2891 / 0.125 |
| plus_status_dates | 370 | 417 | 394 | 328 | 407 | +14/−5 | 0.06357 / 0.09625 |
| plus_siren_scene | 359 | 404 | 384 | 318 | 396 | +3/−5 | 0.7266 / 0.4531 |
| v1_plus_v2 | 368 | 419 | 392 | 326 | 407 | +18/−11 | 0.2649 / 0.1338 |

**Bilan :** tous les ajouts donnent +7 exacts (+18/−11, p=0,265) et +8 union (+15/−7, p=0,134). Statut/dates est le signal descriptif le plus favorable (+9 exacts/+8 union), mais non confirmé : p bruts0,064/0,096, Holm0,636/0,866. Adresse +2/+1 ; nom de site −4/−5 ; scène SIREN −2/−3. Aucun bloc n'est promu à partir de ces comparaisons.

- Ajouts seuls = V1+UNE famille ; pas une recherche du meilleur sous-ensemble. Toutes les variantes annoncées sont publiées, aucun réglage après mesure.
- McNemar par requête exploratoire, Holm sur exact/union des variantes ajoutées ; ni correction des corrélations intra-entreprise ni validation indépendante d’un pilote déjà consommé.

### Confiance calibrée sur UNION

| Ranker / confiance | SUR | Précision union, Wilson95 | Précision exacte des mêmes SUR, Wilson95 | Couverture, Wilson95 |
|---|---:|---|---|---|
| v1/margin | 92 | 88/92 (95.65 %, IC95 [89.35;98.30]) | 86/92 (93.48 %, IC95 [86.49;96.98]) | 92/526 (17.49 %, IC95 [14.48;20.97]) |
| v1/scene_isotonic | 41 | 38/41 (92.68 %, IC95 [80.57;97.48]) | 30/41 (73.17 %, IC95 [58.07;84.31]) | 41/526 (7.79 %, IC95 [5.80;10.40]) |
| v1_plus_v2/margin | 149 | 142/149 (95.30 %, IC95 [90.62;97.71]) | 137/149 (91.95 %, IC95 [86.45;95.33]) | 149/526 (28.33 %, IC95 [24.64;32.32]) |
| v1_plus_v2/scene_isotonic | 30 | 28/30 (93.33 %, IC95 [78.68;98.15]) | 27/30 (90.00 %, IC95 [74.38;96.54]) | 30/526 (5.70 %, IC95 [4.02;8.03]) |

- Cible union utilisée pour le classifieur scène, l’isotonique ET le seuil. Les modèles ranking restent entraînés sur EXACT ; aucun label transformé/retiré.
- Les socles de confiance60% ne sont pas les full-fits du tableau top1 : v1 exact344, union383, v1_plus_v2 exact365, union403.
- **Cible union98% non atteinte pour les quatre politiques.** V2+marge augmente la couverture17,49→28,33%, mais la précision union reste95,30%. La scène calibrée ne justifie pas ici un étage supplémentaire.
- La précision empirique≥98% en calibration ne garantit pas98% sur test ; petits effectifs, Wilson descriptif par requête et pilote consommé. Aucun seuil appris sur les tests.

### Disponibilité réelle des informations

- Scores BAN probabilistes CRM/candidat : absents, deux colonnes intégralement NaN. Incertitude CRM et qualité/distance_precision SIRENE restent distinctes. Pas de nouveau géocodage, seuils union200/500m inchangés.
- Fermetures uniquement via transitions historiques officielles ; couverture parmi les occurrences candidates fermées : 29236/29881. DateDebut du stock non utilisée comme date de fermeture.
- Audit Gemma : 7.67s, 7058/46474 sites définitivement absents de la banque site inspectée ; couverture restante seulement une borne haute avant vérification du mapping tenseur. **Famille sautée**, zéro réencodage.
- Référence historique mesurée :526 vecteurs requête noms seuls en14.19s, chargement inclus. Extrapolation arithmétique conditionnelle aux sites manquants :190.4s AU MÊME débit noms seuls ; **pas un coût mesuré/prédit pour les textes512 tokens**, coût réel non établi et aucun encodage massif lancé.

### Plafond exact et désaccord adresse

- Contrat `fb604665` reproduit : 86/507 labels présents loin du CRM, dont 52 sièges ; 440 comparables, 67 non mesurables parmi les vérités présentes. En incluant les absences du pool : 97/526 labels loin (18.44%), 457 comparables et 69 non mesurables.
- Plafond exact global du pool : 507/526 (96.39 %, IC95 [94.43;97.68]). **98% global impossible avec ces100**, indépendamment du ranker.98% exact sur un sous-ensemble SUR reste possible en principe ; distance au CRM ≠ label erroné ou plafond exact supplémentaire.

### Recommandation unique

Priorité à un signal sémantique de site réellement distinct (activité/usage et nom propre officiels), avec un petit benchmark local autorisé avant tout encodage massif ; aucun gain v2 confirmé justifiant une promotion.

### Vérifications et limites

- Smoke réel puis run526 ; 42 fits ranking, 70.52s, pic RSS1460.5Mio (<7,5Gio). Sources SHA avant/après ; inventaire Gemma par empreintes taille/mtime/inode.
- Rejeu indépendant :3156 vecteurs de scores de requête reproduits exactement,1052 sorties de confiance (probabilités/seuils/décisions),42 audits sans entreprise commune, cibles exactes/union et SHA des42 modèles conformes. Les345600 cellules du smoke36 sont identiques aux mêmes lignes du complet. Trace : `H/step2_features_v2_20261017/full/verification.json`.
- V1 R3 reproduit à l’identique par requête ; nouveaux champs scellés avant lecture du ledger/sidecar. SHA des cibles et des modèles, matrices et rôles de calibration disponibles dans les reçus. Rappel100 et19 absences conservés, cinq vues inchangées.
- Amont supervisé consommé/non full-OOF et routage humain inchangés ; aucun nouveau candidat, alias, ID comme feature, positif injecté, jeu fermé, VM, installation, web ou push.
- Reproduction : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_features_v2.py --output-dir <dossier_CATNAT_neuf>` ; `--smoke`36 vrais dossiers, ne publie pas les docs.
