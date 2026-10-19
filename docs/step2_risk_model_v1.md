# Étape2-10 — risque XGBoost G0, S1 + S2-BASE

## Avenant figé avant tout fit

Banc17427/11483, candidats100/labels exact et UNION inchangés, folds population2/3/4 seulement. CV entreprise0–4 strictement héritées du banc G0 `9a8ecdd6`. BASE seul ; adaptateur `65db35ba`/`21a464d2`/`da3390d7` non promu. Sources et codes scellés SHA avant/après ; aucun jeu fermé, web, installation, VM contrôlée, push, fichier étranger/pending modifié.

### Cache et budget

Vérifier les reçus, SHA, bindings BASE sans adapter et clés/textes du colocated (encodage268697 + confirmation) et du fine-tuning10h côté BASE seulement. Celui-ci conserve les17427 et526,1266170 textes ; réutiliser ses shards par index/texte vérifié et recalculer les cosines/deltas même adresse à partir de ces vecteurs, jamais lire ses vecteurs adaptés. Vérifier l'identité IDs/groupes/ordre candidats/NAF et les requêtes nom seul. Tout manquant est encodé uniquement par `H/runtime/bin/python`, mêmes prompts, BF16/eager/mean masqué/L2/768/max512, batch4, aucun adapter/gradient.

Mini réel100 entreprises entières hash `20261020|risk-smoke|SIREN`, partitions héritées, mêmes fits, calibration et526 réduits à36 descriptifs. Coûts mini inclus. Projection avant les fits complets : max(débit CPU mini par ligne, débit fits S1 complets historiques) × lignes des25fits R3 prévus, facteur1,5 ; encodage manquant × débit BASE mesuré ×1,5 ; préparation/risque/audit/publication réserve600s minimum. Plafond36000s CPU+Metal+I/O, attentes ressources documentées exclues. Si nécessaire préfixes d'entreprises hash `20261020|risk-budget|SIREN`, fractions1/.75/.5/.25/.125 ; toutes leurs lignes conservées, mêmes dossiers pour les trois recettes, aucune sélection sur score/erreur. Premier périmètre qui tient retenu ; sinon STOP. Aucun résultat réduit décrit comme17427.

Superviseur détaché, un seul gros worker, threads1/DuckDB512Mio. Sans processus de VM identifié (hydraulic annoncée arrêtée), garde RSS+driver7,75Gio sous8 ; si VM active/non identifiable garde5,5Gio sous5,75, attente60s si free+inactive<8Gio. Aucun contrôle VM. Arrêt propre sans publication de résultats partiels si CATNAT libre<10Gio ; aucun nettoyage de données. Namespace neuf `H/step2_risk_model_v1_20261020/`.

### Reclasseur et absence de fuite multiniveau

R3 `content85 + S1(7) + S2-BASE(2)`, mêmes200 arbres/profondeur3/lr.05/child10/L2=10/L1=1/hist128/mean8/seed20261017/jobs1, cible exacte. S1 CountVectorizer/NB et crossfit interne3 entreprises du code colocated inchangés, sérialisation/rejeu. Top1 principal OOF5 sur tout le train hors pli. Comparaisons vs base OOF existante et +S1 OOF existant, bootstrap10000 entreprises apparié, seed20261017 ; exact/UNION, IC95 ordinaires et simultanés.

**Ne pas** entraîner le risque sur les OOF globaux des autres plis : leurs rankers ont vu les labels du pli risque test. Empilement renforcé : dans chaque CV externe, train divisé75% proper/25% calibration par entreprise (permutation seed20261017+1000+CV, ceil25%). Trois R3 internes produisent les OOF proper sans aucune entreprise calibration/test ; un R3 proper produit la calibration hors entreprise. Risque appris sur OOF proper seulement, seuil sur calibration seulement ; test sur les scores du R3 externe entraîné sur tout le train externe. Ainsi aucun label test/calibration ne remonte dans le fit du risque. Le refit final R3 augmente le volume par rapport au R3 de calibration : transport explicite, mesuré hors pli, pas une garantie.25 fits R3 au total, aucun HPO/early stopping/choix de checkpoint.

### Liste finale des features risque (46)

Dans cet ordre :
1. `score_top1`, `margin12`, `margin13`, `entropy_top10`.
2. `native_rank_top1`, `same_siren_count_top1`, `same_address_count_top1`, `near50_count_top1`, `same_address_or_near50_count_top1`, `crm_top1_log_distance`, `crm_top1_distance_missing`.
3. Pour chacun, dans l'ordre `naf_div_logp`, `naf_group_logp`, `coloc_div_rank`, `coloc_group_rank`, `coloc_div_delta_best`, `coloc_group_delta_best`, `naf_missing`, `gemma_base_cosine`, `coloc_gemma_delta_best` : trois colonnes suffixes `_top1`, `_top2`, `_delta12`.
4. `name_jaro_top1`, `number_exact_top1`, `street_jaro_top1`, `postcode_match_top1`, `active_top1`, `closed_top1`, `siege_top1`, `commune_log_public_count`.

46 colonnes, aucun ID/nom/code géographique/label comme valeur. `native_rank` = rang G0 sur G0, rang V142 sur526 (le rang V142 n'existe pas pour17427 : ne pas l'inventer). Les comptes SIREN utilisent les identifiants uniquement comme groupes. Adresse stricte de colocated ; ≤50m uniquement entre coordonnées officielles Lambert93 typées numéro/voie, sites publics, jamais centroïdes communaux ; compte incluant top1 si mesurable. Distance CRM–top1 log1p issue des features typées existantes. Densité = proxy log1p(nombre d'établissements publics de la commune CRM, tous états), jointure SIRENE officielle sans label, pas un nombre/km² sans surface connue. Manquants NaN/indicateurs, jamais suppression de dossiers.

Risque `xgb.train`, binary:logistic,200rounds,depth3,eta.05,child10,lambda10,alpha1,subsample/colsample1,hist128,seed20261017,nthread1,base_score.5,logloss. Monotonie+1 sur `margin12` et `margin13`,0 ailleurs ; cible top1 correct UNION. Aucun poids de classe ni rééquilibrage. Modèles sérialisés et probabilités rejouées exactement.

### Seuils et comparaisons

Seuil risque/marge = plus petit score parmi seuils observés, ex aequo indivisibles, dont Wilson bilatéral95 inférieur≥cible98% sur calibration25% uniquement. Aucun seuil admissible => zéro SUR, précision/IC non définis. Même règle97/98/99 pour la courbe prospective précision–couverture ; pas de seuil choisi sur les tests. Wilson brut n'est ni simultané sur seuils adaptatifs ni corrigé de corrélation entreprise : métrique de calibration demandée, aucune garantie produit.

Scène existante :20 features `d.scene`, logistiqueC1/imputation/scaling, isotonique ; proper scindé80/20 entreprises seed20261017+2000+CV pour apprentissage logistique/isotonique, calibration25% commune pour seuil. Logistique reçoit les OOF proper ; iso reçoit ses scores hors fit. Seuil empirique UNION≥98% historique conservé sous `scene_legacy` ; comparaison additionnelle équitable `scene_wilson` avec même Wilson que risque/marge. Toutes les décisions test portent sur le MÊME top1 externe et mêmes dossiers. Rapporter exact/UNION, Wilson95, couverture, CV et folds population, matrices appariées et IC entreprise des deltas de couverture/erreurs par dossier.

Tableau des faux SUR du meilleur modèle descriptif : priorité aux politiques satisfaisant précision UNION poolée≥98%, puis couverture maximale ; si aucune, borne Wilson inférieure maximale puis couverture, ordre risque/marge/scène-legacy/scène-Wilson pour égalité. Sélection de reporting seulement, jamais réglage ni promotion. Recalcul des prédicats/priorités historiques `c37a1a7b` sur le choix actuel, catégorie `pool_union_absent` prioritaire si absence ; conserver aussi la famille historique du dossier ou `hors_erreurs_base`. Tous faux SUR, noms/adresses officiels et vérité conservés sans correction, aucune de ces informations en features.

### Confirmation526 et publication

526 déjà consommés, exclus de tout fit/calibration G0, aucune sélection sur ces526. Mêmes cinq modèles R3 externes appliqués aux526 et mêmes cinq modèles risque/logistique/iso/seuils sans refit. Top1 = argmax de la moyenne des cinq vecteurs R3 ; SUR seulement si les cinq top1 sont identiques et les cinq politiques acceptent leur top1 (sinon A_VERIFIER). Cette agrégation conservatrice est figée avant lecture des résultats. Rapporter aussi chaque paire de modèles séparément. Aucun empilement appris sur526 dans ce jalon.

Auto-publication et commit isolé `docs/step2_risk_model_v1.{json,md}` + ligne append-only `handover.md`, après vérification sources/code/modèles/rôles ; refuse chemins tiers sales ou édition concurrente du journal. Rapport français≤35lignes, chiffres uniquement mesurés, STOP explicite si incomplet. Aucune certification≥98% sans mesure suffisante, aucune promotion.

### Correctif de plomberie avant reprise du mini

Première tentative `511f1597`, STOP conservé `7ea0ca9a` : préparation complète vérifiée, aucun encodage nouveau ; premier fit mini achevé mais pas de reçu final, exception de chemin SHA dans l'assemblage scène. Corriger seulement `read(ref)` en `read(verify(ref))` pour les deux références métadonnées concernées. Même protocole/46features/paramètres/plis/seuils, aucun choix sur score. Namespace neuf `H/step2_risk_model_v1_20261020_b/`, ancien namespace immuable. Réutiliser la préparation candidate-only complète via liens en lecture seule après SHA/validation de son ancien code par Git ; recalculer uniquement les indices de population mini/complet. Coût de la première tentative et intervalle de reprise inclus conservativement dans le budget total10h. Aucun fit partiel sans reçu réutilisé.

## Résultats

## Fait
Statut STOP_INCOMPLETE, aucune promotion ; `scripts/step2_risk_model_v1.py`, docs `step2_risk_model_v1`, `handover.md`.
Durée réelle calcul 0.118h ; mur 0.119h ; pic 0.29Gio.
## Preuve
Arrêt : risk1 exit1; /Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/step2_risk_model_v1_20261020_b/mini/risk1/worker.log. Aucun résultat complet top1/SUR/526 validé.
## Écarts / choix
Aucun résultat partiel présenté comme complet.
## Reste / risques
Recommandation unique : conserver les modèles antérieurs sans promotion.

Détails par CV/fold, courbes97/98/99, IC simultanés, modèles/masques, calibration et faux SUR exhaustifs : JSON et namespace externe.
