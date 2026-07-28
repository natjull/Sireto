# Contrat V4.11-B — stack aligné et aveugle au SIRET CRM

Statut : préenregistré avant construction du dataset V4.11, avant tout
réentraînement et avant ouverture des 225 lignes inédites.

Identifiant : `V411_INPUT_BLIND_ALIGNED_STACK`.

## 1. Question testée

V4.10b a échoué avec un accepteur de 641 features construit sur des scènes
hétérogènes. V4.11 teste une refonte plus simple :

```text
CRM sans SIRET/SIREN d'entrée
  → retrieval sparse V4.2, 100 candidats maximum
  → ranker C entraîné sur ces mêmes pools
  → scène compacte top-1/top-2
  → accepteur query-level
  → AUTO_MATCH ou REVIEW
```

L'hypothèse est que la cohérence entre retrieval, ranker et accepteur apporte
plus que l'ajout de centaines de features ou de règles postérieures.

## 2. Correction de frontière : la réponse n'est plus une entrée

Le SIRET et le SIREN historiques du CRM servent à construire ou auditer les
labels. Ils ne sont jamais visibles dans le chemin de prédiction V4.11.

Sont interdits :

```text
input_siret_exact_match
input_siren_exact_match
candidate_from_input_siret
candidate_from_input_siren
candidate_from_closed_alias
```

Sont également interdits comme entrée de retrieval :

- l'ajout direct du SIRET CRM ;
- l'expansion des établissements du SIREN CRM ;
- toute branche dont l'existence dépend du SIRET/SIREN historique.

Cette décision ne prétend pas que V4.2-B était frauduleux : il traitait un
problème de validation/réparation d'identifiants existants. V4.11 mesure un
problème plus strict et conforme au produit cible : retrouver le SIRET à
partir du nom et de l'adresse d'un CRM sale.

Diagnostic ayant motivé la correction, non utilisé comme preuve de
promotion :

- sur les 5 883 labels exacts V4.6, 5 882 restent présents parmi les
  candidats marqués `candidate_from_sparse=1` ;
- en mettant à zéro les quatre signaux directs d'identifiant dans le ranker B,
  5 882/5 883 top-1 restent corrects ;
- le résultat élevé ne semble donc pas dépendre fortement de ces signaux, mais
  leur présence empêche une interprétation propre.

## 3. Entrées de développement épinglées

| Rôle | Chemin | SHA-256 |
|---|---|---|
| requêtes V4.6 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_6_aligned_b/301b24f47820f992/queries.parquet` | `6a12f1c4ca9ec33636ebcf7748c208595c6168d7cdb8c068e1434af3fe22abb0` |
| labels V4.6 | même répertoire, `labels.parquet` | `69032b745817959422ef26e4c0c1228686260c1daa272ca5d6aba1d7be087b04` |
| candidats V4.2-B | même répertoire, `candidates_v42b.parquet` | `0b7fc90e045da10033f0ae4b598963505d76c16710e2efc9dbe728a93a6536dc` |
| splits/folds | même répertoire, `split_assignments.parquet` | `33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193` |
| prédictions ranker B diagnostiques | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_6_aligned_ranker/421f2cd0cc436af7/predictions_b_oof_dev.parquet` | `f708a51aed9842e236ff9bd6c752079d0ccd127ccabc86d975beaa921c3853db` |
| modèle ranker B diagnostique | même répertoire, `ranker_b/ranker.json` | `ffa0014e1650f679651da91b4b52ef53636eb4fee804666afb8f7756a90c50d7` |
| métadonnées ranker B | même répertoire, `ranker_b/metadata.json` | `39eb014b8c833c79cd50027db110b63144ad482e7466c7f97f8fcdd98b519f11` |
| registre consommé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/registries/v4_11_consumed_population/fd25d1922040d585/manifest.json` | `77711f91fda8dffec3210c49b3df8404e46ff540f30f9597fc7fe7722f2d6962` |
| manifeste retrieval V4.2 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_2_retrieval_integrity_7c4b957/manifest.json` | `63b52c3a1466070410881b0ea61b833ff5d413262239920abbc6b04e3f153f54` |
| snapshot SIRENE établissement | `data/StockEtablissement_utf8.parquet` | `c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845` |
| taxonomie de fonctions | `config/v4_9_site_function_taxonomy.json` | `48bbb7e1795a0731f1f12df41aeb971667c10d03c879bf06d5ba15b65f8b121d` |
| implémentation de fonctions | `src/xgb_matcher/v49_site_function.py` | `8463086d2ce404e5c83140df8ea7351cfb363793edfa7e74db95fe202d9c54e2` |
| baseline accepteur V4.1 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_1/f938abf6b8a87155/acceptor/acceptor_model.joblib` | `16283b8aba5ed135846a74e9040c79e9f863f7e2bd658ca642ad444174b9a3fa` |
| métadonnées baseline V4.1 | même répertoire, `metadata.json` | `73199451b2de6ae383c9c0c58b10ab9c7393994a4efdec45f9c8e1e9f150691c` |
| scènes baseline V4.1 | même répertoire parent, `acceptor_scenes.parquet` | `8f3bc4633ada9eb6347e47a1029f0e69fa8946b1c3c1df38c72232f572088dc9` |

Les candidats V4.6 servent à reproduire le développement. Le builder V4.11
doit reconstruire le retrieval sparse à partir de la configuration V4.2
épinglée, sans branche identifiant, afin de récupérer un vrai top-100 sparse
et pas seulement filtrer après coup le top-100 fusionné.

Le sous-ensemble `candidate_from_sparse=1` du parquet V4.6 sert de contrôle de
parité. Il ne remplace pas la reconstruction.

## 4. Retrieval V4.11

Le retrieval reçoit uniquement :

- nom CRM ;
- adresse CRM ;
- code postal ;
- commune ;
- code INSEE.

Il réutilise la normalisation et les canaux sparse de V4.2 :

- sparse courant ;
- nom mot ;
- nom caractères ;
- adresse mot ;
- exact nom ;
- exact adresse.

L'admission, les poids et les tie-breaks restent ceux de V4.2, après retrait
du canal `siren_head` et de toute branche identifiant. Le pool est tronqué à
100 SIRET uniques exactement ; 100 est un plafond absolu.

La configuration sparse source est celle embarquée dans le manifeste V4.6 :
signature
`021f928e21e2360186217862b4310be90fe0f705c1bfbf43b39a8b41e644e40c`.
Le filtre est `include_closed=false`, le snapshot d'état est celui épinglé
ci-dessus et le tie-break final est le SIRET croissant.

Avant tout ranker, publier sur fit et dev :

- nombre de requêtes ;
- taille min/médiane/moyenne/max des pools ;
- Recall@1/10/50/100 SIRET exact ;
- Recall@100 SIREN ;
- misses complets, sans réinjection positive ;
- comparaison avec V4.2-B complet et avec son sous-ensemble sparse.

Gate retrieval de développement :

- Recall@100 SIRET exact ≥ 99,0 % sur fit et dev ;
- zéro pool >100 ;
- zéro candidat fermé ;
- zéro positif injecté ;
- une vérité absente comptée comme erreur end-to-end.

Sinon verdict `PIVOT_INPUT_BLIND_RETRIEVAL`.

## 5. Ranker C

Le ranker C est un unique `XGBRanker` SIRET. Il utilise uniquement les labels
`MATCH_EXACT` dont le positif est réellement dans le pool.

Les folds existants restent groupés par composante SIREN. Chaque scène
d'accepteur du fit reçoit une prédiction ranker hors-échantillon ; le dev
reçoit une prédiction d'un modèle entraîné uniquement sur le fit.

Ordre candidat autorisé :

```text
has_any_name
name_count
name_jaro_max
name_jaro_second
name_jaro_gap
name_levenshtein_max
name_token_overlap_max
idf_name
numeric_token_match
name_first_word_match_max
name_contains_crm_max
name_crm_contains_cand_max
acronym_match_max
name_sim_max_etab
name_sim_max_ul
name_sim_max_sigle
name_sim_max_pm_dirigeant
type_of_max_name
is_ul_name_max
is_sigle_max
name_length_max
has_person_name
person_name_jaro_max
name_city_overlap_max
name_is_city_like_max
addr_jaro
addr_levenshtein
postcode_match
city_match
street_number_diff
addr_token_overlap
address_density
street_name_jaro
name_addr_consistency
legal_form_category
is_siege
is_association
alias_match
token_overlap_ul
ul_vs_pm_indicator
is_crm_school
geo_exact_match
name_norm_exact
street_number_match
retrieval_rank_recip
```

Il y a exactement 45 features. `retrieval_rank_recip = 1 / retrieval_rank`
est recalculé sur le pool V4.11 final. Aucun score ou rang provenant du pool
fusionné V4.2-B n'est transporté.

Hyperparamètres uniques, sans grille :

```text
objective=rank:pairwise
eval_metric=ndcg@1
n_estimators=800
learning_rate=0.035
max_depth=6
min_child_weight=3
subsample=0.85
colsample_bytree=0.85
reg_lambda=5.0
random_state=42
n_jobs=-1
tree_method=hist
early_stopping=disabled
```

Chaque modèle reçoit toutes les lignes candidat des requêtes exactes
éligibles, positives et négatives. Il y a cinq modèles OOF et un modèle
complet entraîné sur les exacts du fit. Aucun `eval_set` ne déclenche d'arrêt
anticipé.

Le modèle et toutes les prédictions sont entraînés deux fois. Scores et rangs
doivent être bit à bit identiques.

Gate ranker de développement :

- Hit@1 SIRET dev ≥99,8 % sur `MATCH_EXACT` ;
- aucune régression supérieure à deux points sur un segment d'au moins 100
  exacts face au ranker B masqué ;
- métriques end-to-end, donc misses retrieval inclus.

Sinon verdict `PIVOT_INPUT_BLIND_RANKER`.

Le comparateur « ranker B masqué » est le modèle épinglé ci-dessus, appliqué
aux mêmes pools V4.11. Ses cinq features interdites
`input_siret_exact_match`, `input_siren_exact_match`,
`candidate_from_input_siret`, `candidate_from_input_siren` et
`candidate_from_closed_alias` valent zéro. Les features `admission_*` qui
n'existent plus dans V4.11 sont reconstruites uniquement lorsqu'elles
correspondent à un canal sparse présent ; toutes les autres valent zéro. Ce
comparateur reste diagnostique et ne peut être promu.

## 6. Accepteur compact

La cible est `1` uniquement lorsque le top-1 ranker est le SIRET exact d'un
label `MATCH_EXACT`. `AMBIGUOUS` a une cible `0`. `UNRESOLVED` n'est pas un
négatif prouvé : il est exclu du fit, du seuil, de la sélection et des
métriques principales. Son comportement est publié séparément et sa sortie
cible reste `REVIEW`.

Les 94 cas difficiles V4.10b, les 57 random V4.8, les quatre locked et tous
les anciens holdouts sont interdits au fit, au seuil et au gate.

L'ordre accepteur contient une représentation unique de chaque information.

### 6.1 Scène et concurrence

```text
candidate_count
ranker_gap_fraction
ranker_top3_gap_fraction
ranker_score_std_fraction
ranker_score_entropy
unique_siren_count
top1_siren_candidate_count
same_siren_top2
siren_gap_fraction
retrieval_rank_top1_recip
retrieval_rank_gap_recip
same_siren_best_sibling_gap_fraction
crm_is_school
```

### 6.2 Preuves candidat

Pour chacune des 30 bases suivantes, produire exactement deux colonnes :
`top1_<base>` et `delta_<base>`, où le delta vaut top-1 moins top-2.

```text
name_jaro_max
name_jaro_gap
name_token_overlap_max
idf_name
numeric_token_match
name_first_word_match_max
name_contains_crm_max
name_crm_contains_cand_max
acronym_match_max
name_sim_max_etab
name_sim_max_ul
name_sim_max_sigle
name_sim_max_pm_dirigeant
is_ul_name_max
is_sigle_max
person_name_jaro_max
name_is_city_like_max
addr_jaro
postcode_match
city_match
street_number_diff
addr_token_overlap
address_density
street_name_jaro
name_addr_consistency
geo_exact_match
name_norm_exact
street_number_match
is_siege
is_association
```

### 6.3 Fonction de site, sans one-hot massif

```text
role_crm_count
role_top1_count
role_crm_top1_conflict
role_top1_top2_conflict
same_siren_distinct_role_count
same_siren_role_plurality
naf_top1_top2_division_equal
```

Le détecteur de rôles et sa taxonomie sont épinglés avant le build. Ils sont
des features, jamais un veto. Aucune catégorie NAF ou rôle n'est développée
en centaines de colonnes.

Le parquet candidat transporte, comme colonnes de contexte non consommées
directement par le ranker C, exactement :

```text
enseigne1
enseigne2
enseigne3
denomination_usuelle
activity_code
```

Elles sont lues dans le snapshot SIRENE épinglé au même moment que l'état
administratif. Leur absence du schéma provoque `STOP_DATASET_INTEGRITY` ;
elles ne peuvent pas être remplacées silencieusement par des zéros. Pour
`naf_top1_top2_division_equal`, une division inconnue d'un côté ou de l'autre
donne zéro, jamais un faux accord `UNKNOWN == UNKNOWN`.

L'ordre contient donc exactement 80 features : 13 de scène, 60 preuves et
sept informations de fonction.

Sont explicitement absents :

- SIRET/SIREN CRM et dérivés ;
- provenance de population, split, fold ou source ;
- one-hot NAF, rôles, forme juridique ou géographie ;
- copies top-2 lorsqu'elles sont reconstructibles par `top1 - delta` ;
- score ranker dupliqué sous plusieurs noms ;
- features sémantiques neuronales ;
- cross-encoder.

### 6.4 Formules, absences et tie-breaks

Les candidats sont ordonnés par score ranker décroissant, puis rang retrieval
croissant, puis SIRET croissant. Tout score ranker doit être fini ; un score
manquant provoque `STOP_DATASET_INTEGRITY`. Les features candidat historiques
manquantes sont imputées à zéro comme dans le code de features candidat et
cette imputation est identique au train et à l'inférence.

Pour éviter de comparer les échelles absolues des cinq rankers OOF et du
modèle dev, les features de score sont normalisées dans chaque requête. Pour
les scores ordonnés `s1 >= s2 ... >= sn`, poser
`range = s1 - sn`.

- si `n=0`, les 13 features de scène valent zéro ;
- si `n=1`, les trois gaps normalisés valent 1, l'écart-type et l'entropie
  valent zéro ;
- si `n>1` et `range <= 1e-12`, tous les gaps et l'écart-type normalisés
  valent zéro, et l'entropie vaut 1 ;
- sinon :
  - `ranker_gap_fraction = (s1-s2)/range` ;
  - `ranker_top3_gap_fraction = (s1-mean(s2..smin(3,n)))/range` ;
  - `ranker_score_std_fraction = std_population(scores)/range` ;
  - normaliser chaque score par `(si-sn)/range`, appliquer un softmax de
    température 1, puis diviser son entropie par `log(n)` ;
  - `siren_score_gap` devient `siren_gap_fraction` : différence entre le
    meilleur score du SIREN top-1 et le meilleur score d'un autre SIREN,
    divisée par `range`; valeur 1 s'il n'existe aucun autre SIREN ;
  - `same_siren_best_sibling_score_gap` devient
    `same_siren_best_sibling_gap_fraction` : différence entre `s1` et le
    meilleur score d'un autre SIRET du même SIREN, divisée par `range`;
    valeur 1 sans frère.

Les noms définitifs dans l'ordre de scène sont donc :

```text
candidate_count
ranker_gap_fraction
ranker_top3_gap_fraction
ranker_score_std_fraction
ranker_score_entropy
unique_siren_count
top1_siren_candidate_count
same_siren_top2
siren_gap_fraction
retrieval_rank_top1_recip
retrieval_rank_gap_recip
same_siren_best_sibling_gap_fraction
crm_is_school
```

`retrieval_rank_gap_recip` vaut `1/rang(top1) - 1/rang(top2)` et vaut
`1/rang(top1)` sans top-2. `std_population` utilise `ddof=0`.

## 7. Deux accepteurs, pas de grille

Comparer exactement :

1. `COMPACT_LOGIT` : régression logistique L2, `C=0.1`,
   `solver=lbfgs`, `tol=1e-4`, `class_weight=None`, `max_iter=5000`,
   seed 42 ; standardisation des continues/comptages sur le train uniquement,
   binaires inchangées ;
2. `MONOTONIC_XGB` : `n_estimators=400`, `learning_rate=0.03`,
   `max_depth=2`, `min_child_weight=20`, `subsample=0.85`,
   `colsample_bytree=0.85`, `reg_lambda=10`, `reg_alpha=1`,
   `objective=binary:logistic`, `eval_metric=logloss`, `tree_method=hist`,
   `n_jobs=8`, sans early stopping, seed 42.

Les contraintes monotones XGBoost sont préenregistrées dans le plan
d'entraînement : positives uniquement pour les scores/similarités où une
hausse ne peut que renforcer la preuve, négatives pour l'entropie, les
distances et conflits explicites, nulles pour les autres. Aucun signe ne sera
choisi après observation des labels.

Les deux modèles sont rejoués deux fois et doivent produire des scores
identiques.

Le plan d'entraînement, commité et hashé avant le premier fit, doit publier
le vecteur de 80 contraintes. Les contraintes non nulles sont figées par
famille :

- `+1` pour `ranker_gap_fraction`, `siren_gap_fraction`,
  `same_siren_best_sibling_gap_fraction`, les deux preuves de rang
  retrieval et les `top1_`/`delta_` des similarités ou accords explicites
  (`jaro`, `overlap`, `match`, `contains`, `acronym`, `consistency`) ;
- `-1` pour `ranker_score_entropy`,
  `top1_street_number_diff`, `delta_street_number_diff`,
  `role_crm_top1_conflict`, `role_top1_top2_conflict` et
  `same_siren_role_plurality` ;
- `0` pour toutes les autres.

`ranker_top3_gap_fraction` a explicitement une contrainte nulle.

Les features binaires, non standardisées pour le logit, sont exactement :

```text
same_siren_top2
crm_is_school
top1_<base> et delta_<base> pour :
  name_first_word_match_max
  name_contains_crm_max
  name_crm_contains_cand_max
  acronym_match_max
  is_ul_name_max
  is_sigle_max
  name_is_city_like_max
  postcode_match
  city_match
  geo_exact_match
  name_norm_exact
  street_number_match
  is_siege
  is_association
role_crm_top1_conflict
role_top1_top2_conflict
same_siren_role_plurality
naf_top1_top2_division_equal
```

Les compteurs de rôle `role_crm_count`, `role_top1_count` et
`same_siren_distinct_role_count`, ainsi que toutes les autres features, sont
standardisés sur le train propre au modèle. `numeric_token_match` est
continue.

## 8. Développement, seuil et sélection

Les 5 547 lignes `fit` entraînent les modèles à partir des prédictions ranker
OOF.

Le dev historique de 1 456 lignes est déjà consommé et n'est pas une preuve
finale. Il est néanmoins séparé de façon déterministe par composante SIREN,
avant score accepteur :

- `threshold_dev` si le premier octet de
  `SHA-256("v411-threshold:" + siren_component_id)` est inférieur à 128 ;
- `comparison_dev` sinon.

`siren_component_id` est celui du parquet de splits épinglé. Il est présent
pour les 7 003 lignes, y compris `AMBIGUOUS`, et ne peut être recalculé à
partir du label. Une même composante ne peut apparaître des deux côtés ; cette
disjonction est un invariant bloquant.

Les volumes attendus, calculés avant tout score accepteur, sont :

| Sous-ensemble | Total | `MATCH_EXACT` | `AMBIGUOUS` | Composantes |
|---|---:|---:|---:|---:|
| `threshold_dev` | 710 | 583 | 127 | 637 |
| `comparison_dev` | 746 | 634 | 112 | 652 |

Le seuil de chaque accepteur est choisi uniquement sur `threshold_dev`.
Les seuils candidats sont les scores distincts observés plus
`nextafter(max_score, +inf)`. La décision est `AUTO` si `score >= seuil`.
Un seuil est sûr si `1000 * correct_auto >= 998 * auto_count`; le seuil sans
AUTO n'est pas éligible. Maximiser `auto_count`, puis retenir le seuil le plus
haut. Si aucun seuil n'est sûr, la variante est inéligible.

La sélection entre les deux familles utilise uniquement `comparison_dev`.
Un candidat est éligible si :

- précision SIRET exacte observée ≥99,8 % ;
- couverture AUTO ≥80,0 % des requêtes `MATCH_EXACT + AMBIGUOUS` ;
- zéro `AMBIGUOUS` automatisé ;
- aucune famille critique d'au moins 100 lignes ne régresse de plus de deux
  points absolus face à la baseline reproductible ;
- aucune violation d'intégrité.

La baseline est le stack V4.1 épinglé ci-dessus, au seuil gelé
`0.46313316267954524`, évalué sur les mêmes `query_id` de
`comparison_dev` avec son propre top-1 historique. Il s'agit d'une comparaison
produit end-to-end, pas d'une ablation du seul accepteur.

Les familles critiques préenregistrées sont :

- `input_siret_state` (`ACTIVE`, `CLOSED`, `NOT_FOUND`, audit seulement) ;
- `source_segment` ;
- `top1_siren_candidate_count > 1` contre `=1` ;
- `role_crm_count > 0` contre `=0`.

Pour une famille d'au moins 100 lignes, la précision ne peut pas être
inférieure à celle de la baseline et la couverture ne peut pas perdre plus de
deux points absolus. Les familles plus petites sont publiées sans gate.

`input_siret_state` et `source_segment` sont conservés uniquement comme
métadonnées d'audit dans le dataset ; ils ne figurent jamais dans les 45 ou
80 features. Les lignes `UNRESOLVED`, lorsqu'elles existent, sont forcées en
`REVIEW` et exclues du dénominateur de couverture principal.

Parmi les éligibles : plus grande couverture, puis moins d'erreurs, puis
`COMPACT_LOGIT`.

Si aucun n'est éligible : `PIVOT_COMPACT_ACCEPTOR`. Sinon :
`GO_FREEZE_V411_CANDIDATE`.

Le modèle gagnant reste celui entraîné sur `fit` et son seuil reste celui de
`threshold_dev`. Il n'est pas réentraîné après lecture de `comparison_dev`.

## 9. Challenge inédit de 225 lignes

Les 225 lignes du registre V4.11-A restent physiquement fermées jusqu'au
verdict `GO_FREEZE_V411_CANDIDATE` et à la publication des hashes du bundle.

Après gel :

1. ouvrir les entrées CRM uniquement pour la qualification, puis qualifier
   les 225 lignes sans SIRET CRM, sans retrieval et sans score ;
2. utiliser la politique active-direct-current V4 inchangée ;
3. produire `MATCH_EXACT`, `AMBIGUOUS` ou `UNRESOLVED` avec preuves locales
   traçables, puis geler et hasher ces labels ;
4. exécuter une seule fois le stack gelé ;
5. publier tous les nombres, erreurs et intervalles.

Cette mesure est `DESCRIPTIVE_UNSEEN_225`, jamais un gate de promotion ni une
certification représentative, car les 225 lignes ont toutes un identifiant de
service absent.

## 10. Preuve finale

Une décision produit exige un nouvel export CRM indépendant :

- aucune ligne, empreinte, identité de service, SIRET historique ou entité
  connue dans le registre consommé ;
- au moins 500 dossiers `MATCH_EXACT` identifiables ;
- qualification gelée avant retrieval ;
- bundle et seuil gelés avant ouverture des labels ;
- une seule évaluation ;
- précision SIRET exacte AUTO ≥99,8 % ;
- couverture AUTO ≥80,0 % ;
- zéro `AMBIGUOUS` automatisé ;
- intervalles de confiance et nombres bruts publiés ;
- conclusion explicite `GO`, `PIVOT` ou `STOP`.

Avec seulement 500 identifiables, 99,8 % reste une estimation observée. Une
revendication statistique forte exige environ 2 300 décisions AUTO
indépendantes auditées sans erreur.
