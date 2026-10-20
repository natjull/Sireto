# Reclasseur v3 — avenant avant tout fit, 20 octobre 2026

## Autorités et invariants

G0 `9a8ecdd6`, risque `2cf01765`/code `d65c3dc7`, plafond `a5ab226d`, familles `c37a1a7b`, criblage `c8b9e405`. Banc entier17427 dossiers/11483 entreprises,100 candidats inchangés, cinq CV entreprise (0–4 distincts des folds population2/3/4). Folds population0/1, test4323, control962 fermés.526 consommés descriptifs seulement : aucun fit/calibration/sélection sur eux. Aucun label corrigé/positif injecté ; tous les dossiers et absences restent évalués. Aucun identifiant comme feature, aucune nouvelle source/alias/web/dépendance/encodage/adaptateur. S1 et S2-BASE et leurs OOF de base restent figés.

## Groupe candidat et 27 colonnes supplémentaires figées

Pour chaque candidat, voisinage non transitif parmi les100 : même adresse stricte scellée (commune/numéro/suffixe/voie, sans plage) OU distance officielle typée<=30m ; self toujours inclus. Coordonnées publiques uniquement, précisions numéro ou voie admissibles selon les helpers historiques, jamais centroïde communal/inconnu. Une position voie proche est un proxy, pas une preuve de co-occupation. Aucun accès label-side dans le calcul de ces features.

13 colonnes structurelles : `group_size`, `group_distinct_siren`, `group_active_count`, `group_closed_count`, `group_siege_count`, `group_active_fraction`, `group_only_active`, `group_active_name_match_count`, `group_only_active_name_match`, `group_only_matching_active`, `group_secondary_at_crm_hq_elsewhere`, `group_hq_has_secondary_at_crm_elsewhere`, `group_candidate_secondary`. Nom qui colle = maximum Jaro CRM/public>=0,92 (règle historique, pas un seuil ajusté). « seul actif dont le nom colle » et « seul actif au nom concordant parmi les actifs » distincts. Actif/fermé sont uniquement états officiels A/F ; inconnu jamais déclaré actif.

14 colonnes relatives : pour chacun de `name_ul`, `name_etab`, `name_enseigne`, `name_max`, `naf_div`, `naf_group`, `gemma_base`, deux champs `group_<signal>_rank` et `group_<signal>_delta_best`. Rang compétition=1+nombre de voisins strictement meilleurs, ex aequo indivisibles ; delta=candidat−meilleur, absent NaN, singleton valide rang1/delta0. Noms = colonnes historiques `name_ul_jaro`, `name_etab_jaro`, `v2_enseigne_jaro_max`, `name_jaro_max` ; S1 = deux logprobabilités NB OOF historiques ; S2 = cosine BASE vérifiée. Les similitudes nulles héritées restent nulles, pas des valeurs inventées.

Secondaire/siège relatifs : uniquement relations observées parmi les100, SIREN pour jointure interne et comptage, jamais sa valeur comme feature. Secondaire à l'adresse CRM stricte ET siège du même SIREN hors voisinage et hors adresse CRM ; indicateur réciproque pour un siège ayant un secondaire à l'adresse CRM hors de son groupe. Secondaire reconnu si site public connu et siège non vrai. Absence de siège parmi100 => indicateur observé faux, pas une preuve que le siège n'existe pas ; cette limitation est publiée. Aucun lookup de siège hors pool.

## Trois variantes et apprentissage

`base` = content85+S1sept+S2deux, scores R3 exacts des25 fits imbriqués risque v1 réutilisés/rejoués ; top1 attendu12275exacts/13382UNION. `group` = mêmes94 colonnes+27 ci-dessus. `noise` = même121colonnes, poids requête0 pour les FORT du criblage officiel `c8b9e405`,1 sinon.95 FORT attendus, définis par grille sans modèle (pas le signal modèle ni le juge). Masque joint par query_id et index G0 ; ni feature ni filtre de calibration/test. Tous les100 candidats et labels intacts, poids query-level XGBRanker (pas suppression de lignes). Poids0 appliqué aussi au nouveau risque de cette variante, uniquement proper. Les NB historiques restent immuables : leurs anciens fits ne sont pas rétroactivement débruités. Cette variante évalue un débruitage des nouveaux apprentissages LTR/risque, pas une correction globale du pipeline ni une preuve de bruit humain.

Même `rank:ndcg`, paramètres historiques200arbres/profondeur3/eta0,05/child10/L2=10/L1=1/hist128/mean8/seed20261017/threads1, aucun HPO. Pour chacun des cinq CV : trois rankers internes proper OOF, un proper→calibration25% entreprises, un outertrain→test+526, mêmes rôles que risque v1. Total50 nouveaux rankers,25 bases réutilisées, aucun fit NB/S2. Les NB OOF training et scoring de chaque fit sont rejoués depuis les modèles scellés, mêmes indices/entreprises et SHA de features ; aucune OOF globale contaminant calibration/test. Deux smokes réels sur100 entreprises entières du proper CV0 (SHA `20261020|group-smoke|<entreprise>`), scoring calibration original, modèles sérialisés/rejoués, aucune sélection basée sur le smoke.

## Risque et métriques

Pour les TROIS variantes, même risque v1 binaire UNION, mêmes46features recalculées sur leur propre top1, plus trois colonnes choisies A PRIORI, pas « meilleures » après mesure : `group_name_max_delta_best_top1`, `group_naf_group_delta_best_top1`, `group_gemma_base_delta_best_top1`. Choix motivé par les trois sources indépendantes nom/S1/S2 ; aucune sélection statistique de features.49colonnes,200arbres et tous paramètres conservés, monotonies+1 margins12/13 seulement,0 autres. Trois×cinq fits proper OOF, calibration commune inchangée, transport proper→refit outertrain explicite. La base confiance49 est distincte de l'ancienne46 ; les deux ne seront pas confondues.

Seuil Wilson bilatéral95 (z1,959963984540054), plus petit score calibration observé admissible, ex aequo indivisibles, cibles96/97/97,5/98%. Aucun seuil=>0SUR/précision indéfinie. Vérification indépendante par helper contractuel, aucune sélection sur test/526. Précision exacte/UNION et couverture, poolé, par CV et fold population ; tableau final quatre lignes et trois variantes.526 : moyenne des cinq vecteurs ranking ; acceptation seulement unanimité top1 SIRET et acceptation des cinq risques, aucun fit.

Top1 exact/UNION n/N, Wilson et IC bootstrap cluster entreprise10000, seed20261017, pondération par dossier, apparié. Deltas vsbase, contraste bruit vsgroup, IC ordinaires et simultanés via helper historique. Gains ET pertes sur toutes familles historiques `c37a1a7b`, en particulier co-adresse1046 et siège267 ; compléments absences et hors anciennes erreurs pour couvrir17427. Familles label-side descriptives sélectionnées sur anciennes erreurs, jamais features. Toutes les recettes évaluent les mêmes17427. Recommandation exploratoire prédéclarée : garderbase si aucun gainUNION avec IC95>0 ; sinon préférergroup sauf gain incrémental noisevsgroup avec IC95>0 ; jamais promotion produit ni choix sur526/SUR test.

## Ressources et publication

Namespace neuf `H/step2_ranker_v3_group_20261020`, un worker CPU séquentiel détaché, threads1, DuckDB512Mio/aucun spill, pas de Metal. Garde RSS stricte8Gio sans VM identifiée,5,75 sinon ; inspection de métadonnées seulement, jamais contrôle de hydraulic. Attente60s si gros workerV142>4Gio ou VM active et free+inactive<8Gio. Autre agent production léger<=4Gio et tous fichiers étrangers/pending intouchés. Arrêt propre CATNAT libre<12Gio, aucune suppression, estimation écriture<1,5Gio et vérification disque continue. Budget14400s depuis lancement hors attentes documentées, smokes/IO/rejeux/bootstraps/publication inclus ; réserve publication120s. Projection prudente après premier fit représentatif, STOP si hors budget ; pas réduction silencieuse du banc. Modèles et packs compacts, adjacency packbits, aucune nouvelle banque Gemma.

Code/avenant commis avant fit. SHA/liaisons sources/modèles/OOF/features avant/après, rejoués, rôles entreprise disjoints et poids inspectés ; smokes réels, pas tests synthétiques. Publication automatique et commit isolé `docs/step2_ranker_v3_group.{json,md}` + append-only `handover.md`, aucun push, refuse chemins étrangers sales/concurrents. En cas de STOP, rapport incomplet explicite, pas résultats partiels promus. Rapport final<=25lignes, tableau top1 trois variantes/IC, gains co-adresse/siège, tableauSUR quatre cibles, durée/commits/une recommandation. Développement exploratoire : Wilson adaptatif et corrélations entreprise ne certifient pas98% produit.

## Résultats

**Fait** —17427 dossiers/11483 entreprises, mêmes5CV/OOF ; docs/script `step2_ranker_v3_group`, `handover.md`.
**Preuve** —50LTR+15risques+2smokes,60seuils indépendants/rejeux/SHA/groupes ; durée 100.63min, pic 4.92Gio.

| Top1 | Exact %, IC95 entreprise | UNION %, IC95 entreprise | Δ exact [IC95] | Δ UNION [IC95] |
|---|---|---|---|---|
| base | 70.44 [69.15, 71.73] | 76.79 [75.51, 78.04] | — | — |
| group | 70.41 [69.11, 71.71] | 76.79 [75.51, 78.07] | -0.03 [-0.23, 0.17] | +0.01 [-0.18, 0.19] |
| noise | 70.45 [69.15, 71.74] | 76.85 [75.56, 78.11] | +0.01 [-0.18, 0.2] | +0.06 [-0.12, 0.23] |

Familles historiques, ΔUNION group/noise vsbase : colocation n=1046 : -0.19 [-1.29, 0.88] / +0.19 [-0.87, 1.26]; same_siren_siege n=267 : +1.50 [-0.7, 4.05] / +2.25 [0.0, 4.76].

| Cible Wilson calibration | base P UNION / couverture % | +groupe P/C % | +groupe+bruit P/C % |
|---|---|---|---|
| 96% | 96.49 / 42.17 | 96.17 / 43.43 | 96.60 / 43.09 |
| 97% | 97.81 / 25.45 | 97.22 / 25.61 | 97.66 / 24.82 |
| 97.5% | 98.50 / 11.06 | 97.82 / 14.47 | 98.24 / 16.27 |
| 98% | 99.26 / 1.55 | 98.24 / 8.47 | 98.44 / 7.34 |

526 descriptifs, top1 exact/UNION : base 387/410; group 382/405; noise 381/404.
**Écarts / choix** —3features risque choisies a priori (nom/S1/S2),49 pour les3recettes ; bruit poids0 des95FORT en train LTR/risque seulement, NB figés.
**Reste / risques** —Tout le banc évalué, labels/pools/jeux fermés/VM/tiers inchangés ; IC exploratoires, pas garantie98% ni promotion.
Recommandation unique : conserver la référence S1+S2-BASE, aucun gain UNION confirmé.

Détails par CV/fold, familles complètes, n/N, Wilson, gains/pertes et confirmation526 dans le JSON ; reçus/modèles/rejeux dans `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/step2_ranker_v3_group_20261020`.
