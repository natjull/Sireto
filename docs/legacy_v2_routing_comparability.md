# Routing janvier 2026 — comparabilité avec l'étape 2

## Verdict
**Non comparable en l'état. (a) est démontré ; (b) existe, sans gain V142 mesuré ; (c) subsiste pour les recouvrements et la provenance exacte.**
Le 74,52 % à 99,84 % n'est pas une preuve de précision SIRET exacte end-to-end sur la population actuelle. Ne pas reprendre ses poids, son seuil ni son audit comme garantie.

## 1. Population et exclusions
- Source historique : `b2389895:reports/ROUTING_SOTA_2026-01-25.md`, §3 : `crm_ok_gt.csv`=17 094 lignes ; `samples_v6_A.parquet`=16 725 requêtes ; split test=2 512. La couverture 97,8 % est celle des requêtes représentées, pas le recall national.
- Le « gold » venait d'un filtre géographique, pas d'une adjudication humaine : `scripts/audit_gt.py:103–117`, SIRET trouvé ET (`INSEE égal` OU `CP égal`) ; autres dossiers exclus. Confirmation : `reports/v9/v4_3_hard_labels_results.md`, « Audit du gold standard historique ».
- Pas de filtre `MATCH_EXACT` dans le générateur contemporain : `b2389895:scripts/generate_training_samples_v5fast.py:150–189`, SIRET de longueur14, split par SIREN 70/15/15, seed42 ; ce n'est pas le protocole actuel folds2/3/4.
- Métadonnées effectives : `data/old/samples_v6_A.json`, `generated=2026-01-24T18:45:45.592191`, `seed=42`, `variant=A`, `exclude_closed_gt=false`, `exclude_closed_candidates=false` ; les fermés n'étaient donc pas tous retirés.
- Un dossier sans candidats disparaît ; sans positif, `return []` ; le positif trouvé dans la partition est réajouté au sous-pool, pour TOUS les splits (`b2389895:.../generate_training_samples_v5fast.py:551–552,657–658,721–725`). L'évaluation aval est conditionnelle à cette construction, pas all-human sans injection.
- Le builder de benchmark score ces samples, pas une recherche live : `scripts/generate_topk_benchmark.py:72–90,127–134`, défaut `split=test`, `top-k=20`. L'évaluateur peut encore exclure les décisions sans GT (`scripts/evaluate_routing.py`, `evaluate_routing`, filtre `has_gt`). Nombre réellement exclu à cette dernière étape : inconnu.
- **Recouvrements non établis** avec test4323, control962 ou folds0/1 : aucun croisement d'identifiants effectué ni attestation de disjonction trouvée. Ne pas assimiler « test » janvier à « test4323 » et ne pas supposer l'indépendance des anciens poids.
- Les populations récentes ont d'autres autorités : `docs/train3000_frozen_v142_sample.md:12–25`, ledger23 571, test4323 par complément, contrôle962 issu de validation, folds2/3/4 ; `reports/v412_learned_unified_population.md`, 17 054 anciennes lignes liées à `crm_ok_gt`, labels remaniés et nouveaux plis. Racine historique commune possible, cardinalités actuelles différentes, intersection chiffrée inconnue.

## 2. Ce que signifiait « précision »
- Cible du risque = **bon SIREN**, pas bon site : rapport janvier §4.2, commande `--target siren` ; `models/routing_risk_meta.json`, `target=siren`, `features`=64, `threshold=0.9850000000000004`. Les 68 features/0,835 d'AGENTS ne décrivent donc pas fidèlement le bundle local actuel.
- Mesure brute annoncée = **SIRET strict** : rapport janvier §5.1, `AUTO=1872`, `FP=6`, précision=99,68 %, taux=74,52 % ; ce n'est pas une mesure purement SIREN.
- Audit §5.2 : BOULANGER maison mère vs CUSTOMER CARE devient « acceptable » ; clinique équine, GT déclarée fausse ; IN EXTENSO SOCIAL IDF à la bonne adresse vs SOCIAL ailleurs, choix déclaré correct. Trois autres erreurs restent FP.
- Ainsi 99,84 %=1869/1872 **après réinterprétation métier de trois cas**, dont une relation maison mère, sans preuve générale même SIREN/même site : ce n'est pas notre définition opérationnelle actuelle.
- Le rapport recommande ensuite de « corriger les 3 erreurs » (§9), mais ne prouve pas une modification effective des labels. Aucun procès-verbal scellé relié aux sorties n'est disponible dans l'audit versionné (`reports/v9/downstream_architecture_audit.md`, « Défaut principal »). Nous ne réadjugeons aucun de ces cas.
- Attention versions : `reports/latest_v6a_sota/benchmark_v6a_eval_dual.json`, `metric=strict_siret`, `total_records=2512`, `auto_metrics.count=1676`, `correct=1674`, `incorrect=2` : variante duale à66,72 %, pas l'artefact du74,52 %.

## 3. Pool et concurrence : explication majeure
- Partitions SIRENE locales INSEE, repli CP, fermés inclus ; pas de LLM/API/Places dans ce benchmark de samples. Sources : rapport janvier §6.3 ; générateur contemporain `load_candidates_for_loc`, puis `generate_split`.
- `data/old/samples_v6_A.json` : `prefilter_k=500`, `tfidf_name_mode=primary`, `siren_siblings=false`, `drop_unnamed_candidates=true`, `hard_negative_ranker=null` ; le ranker de l'architecture ne constitue donc pas une preuve de génération réelle pour ce benchmark.
- Surtout `max_same_siren_negatives=0` : le générateur **retire les négatifs du SIREN humain** (`b2389895:.../generate_training_samples_v5fast.py:554–567`). La concurrence entre sites que nous devons résoudre aujourd'hui était artificiellement réduite avec la vérité terrain.
- Samples = positifs + négatifs sélectionnés (hard/adresse/aléatoires), non tous les candidats naturels ; défaut50 négatifs dans le code, mais limite exacte du run non enregistrée dans cette méta. Puis seulement20 lignes par requête au benchmark, sans déduplication SIRET préalable.
- Audit déjà mesuré, **non rejoué ici** : `reports/v9/downstream_architecture_audit.md`, « Défaut principal »/« Où se situe la perte » : vérité@20=2490/2512 (99,124 %), top1 exact=2150/2512 (85,589 %). Ce recall est conditionnel aux samples avec positif réintroduit, pas au pool naturel.
- Le même audit compte1428/2512 scènes avec top1=top2 SIRET ; toutes AUTO. Hors ces scènes :444/1084 AUTO=40,959 %, dont438 corrects=98,649 %. Global brut :1866/1872=99,679 %. Ce sous-ensemble est diagnostique, pas une estimation corrigée de production.
- Les22 absences du top20 restent dans ces2512 ; les absences en amont de la construction des samples n'y sont pas représentées. Le recall du pool naturel avant injection n'est pas établi pour ce run.

## 4. Signaux et comparaison aux85 features actuelles
- Aucun top10 certifié du **risk janvier** trouvé. Son schéma =4 scores +20 preuves ×(top1,top2,delta), soit64 (`src/xgb_matcher/routing_risk.py:13–43`). Ne pas confondre importance du risque et du decider.
- Huit importances documentées du **decider V7, autre benchmark** : permutation intra-requête, baisse Hit@1 en points : `name_semantic_max`21,19 ; `address_density`2,63 ; `name_addr_consistency`2,28 ; `addr_token_overlap`2,24 ; `is_siege`1,61 ; `ul_vs_pm_indicator`1,49 ; `name_semantic_second`1,45 ; `name_token_overlap_max`1,45 (`reports/consultant_architecture_crm_siret.md:249–287`). Ce rapport local n'est pas versionné ; ces effets ne sont pas une attribution du74,52 % ni des gains transférables.
- Absents des85 : les trois similarités sémantiques MiniLM (`name_semantic_*`), IDF/rareté du nom, densité historique de l'univers géographique, comparaison noms de dirigeants/personnes morales et indicateur UL/PM, ainsi que les probabilités du decider historique. `models/xgb_two_stage_meta_20260124_210218.json`, `feature_order`=47, `semantic_enabled_samples=1` ; schéma actuel : `scripts/step2_xgb_ranker_v1.py:53–67`, `scripts/step2_features_v2.py:38–59`, moins11 colonnes natives (`docs/step2_option_d.md`, M_content).
- Des analogues existent déjà : lexical nom/adresse, interaction nom/adresse, siège, concurrence même adresse sur100. Ce n'est pas la même chose que `address_density`/`idf_name` calculés sur la partition ; ne pas remplir les manques avec des zéros ni importer `alias_match` CRM.
- La confiance actuelle exploite déjà marges/scène/isotonique ; nouveauté historique à tester surtout dans les **preuves candidat**, pas simplement « ajouter un risk model » (`docs/step2_g0_bench.md`, « Plis et confiance »).

## 5. Ce qu'on devrait emprunter — sans promettre un chiffre
1. Évaluer une similarité sémantique des noms officiels propres au candidat, avec ablation OOF entreprise ; aucun dirigeant/tiers utilisé comme preuve du site, aucun alias CRM.
2. Tester rareté lexicale et densité/concurrence sur un univers label-free explicitement défini ; conserver les preuves directes top1/top2 dans l'accepteur.
3. Garder cible exacte et vue opérationnelle séparées, tous les dossiers sans positif, SIRET dédupliqués et concurrents même SIREN ; calibration interne seulement. **Ne pas emprunter** injection, exclusion des frères, requalification opportuniste des FP ou seuil0,835.

## Contraste actuel et limites de cette lecture
`docs/step2_g0_bench.md`, résultats : base/margin=1682/17427 SUR (9,65 %), union1627/1682=96,73 %, exact1622/1682=96,43 % ; base/scène=1228 SUR (7,05 %), union97,23 %. Ce sont des entreprises externes OOF, toutes absences conservées ; G0 n'est pas V142. `docs/step2_option_d.md` distingue le transfert sur526 V142 consommés.
Donc définitions, population, injection, retrait des concurrents et duplication expliquent une non-comparabilité certaine, **pas une décomposition chiffrée exhaustive de l'écart**. Les signaux perdus sont plausiblement utiles, leur apport actuel reste à mesurer.
Lecture de code/docs/métadonnées et arithmétique seulement ; aucun modèle, parquet, membership ou label des jeux réservés chargé ; aucun web ni recalcul du benchmark historique. Les sources locales non scellées et les liens de provenance incomplets restent des limites explicites.
Vérification réelle : `PYTHONDONTWRITEBYTECODE=1 python3` (JSON locaux + AST des schémas + `git show b2389895:...`) :46+50−11=85 features, risque64/cibleSIREN, injection/exclusion sans positif/exclusion des frères confirmées ; RSS25 378 816 octets. `git diff --check` sans erreur. Le premier essai de garde `RLIMIT_AS` a été refusé par macOS avant lecture ; le smoke léger a ensuite mesuré sa RSS, sans garde AS revendiquée.
