# Banc G0 — protocole pré-déclaré étape2-6

Banc de développement, pas validation produit. Sources optionD `7b0005a8`,
17427 requêtes /11483 entreprises, candidats G0 et85 features immuables.
Aucune génération, aucun nouveau géocodage/embedding, aucun jeu fermé,
aucun changement de label ou du pool V142. Folds population2/3/4 seulement.
Les indices CV0–4 ci-dessous sont de NOUVEAUX plis, pas les folds population0/1.

## Recettes figées avant fits

- Base :85 features de M_content, mêmes R3 `rank:ndcg`,200 arbres/profondeur3,
  lr0,05, min_child10, L2=10/L1=1, hist/max_bin128, mean/8, seed20261017,
  n_jobs1, sans HPO/early stopping. Aucun poids historique entraîné sur ces
  17427 réutilisé pour leur évaluation OOF.
- `plus_g0` : base+EXACTEMENT2 colonnes, rang G0 (1–100) et score RRF brut.
- `company_weight` : mêmes85 et cible exacte ; poids XGBoost PAR GROUPE DE
  REQUÊTE = n_fit/(nombre_entreprises_fit × requêtes_de_cette_entreprise_fit).
  Somme de poids identique par entreprise, moyenne1, recalculée dans chaque
  fit. Pas de pondération supplémentaire des arbres par candidat, de la
  scène/calibration ou des métriques principales, pour isoler ce changement.
- `union_target` :85 features ; cible binaire exact OU site, et non site seul.
  Tous les candidats déjà émis admissibles sont positifs : même SIREN ET
  (adresse stricte OU distance typée200/500m), plus exact toujours positif.
  Réutilisation directe de `targets_and_views`, contrat `fb604665`, points
  SIRENE étude juillet/adresses août et géométrie CRM déjà cachée. Les points
  du label ne servent jamais à la génération/features. Aucun label réécrit.
- Chaque groupe100, même sans positif, reste au fit et dans les métriques.
  Égalités départagées dans l'ordre G0 déjà émis, jamais selon la vérité.

## Plis et confiance

GroupKFold5(shuffle=True,random_state20261017), groupes=SIREN humain,
uniquement pour la partition/custody. Mêmes plis pour les quatre recettes.
Top1 : fit sur80% des entreprises, score sur20% externes.
Confiance : protocole historique interne60/20/20 des entreprises du train
externe ; graine20261017+100+indiceCV. Scène20 variables/logistiqueC1,
max_iter1000, imputation/standardisation fit uniquement sur train, sur scores
3-plis entreprise OOF dans le proper60%. Isotonique sur20%, seuil sur20%,
aucun test externe utilisé. Ranker de confiance distinct du full-fit80%.
Marge et scène+isotonique visent la précision UNION empirique≥98%, couverture
maximale, fin d'ex æquo ; aucun minimum d'effectif ni borne Wilson ajoutés.
0 SUR : précision non définie. Publier exact et union des mêmes SUR, couverture,
Wilson95, effectifs et plis pour CHAQUE variante, sans choisir sur les tests.
Le contexte scène utilise les46 variables V1 génériques recalculées sur G0,
avec les ancrages natifs G0 seulement pour la confiance, jamais pour le ranker
base85. Les neuf colonnes retirées des46 sont reconstituées sans labels.

## Incertitude, classement et limites

Wilson95 par requête descriptif, sans promettre ±0,7pt avant mesure. Delta
apparié query-weighted vs base : bootstrap des entreprises entières,10000
réplications, graine20261017, ratio différences de succès / nombre de requêtes
rééchantillonnées. Même tirage pour les six comparaisons (3 variantes×2 vues),
percentiles95% et simultanés Bonferroni (six comparaisons) publiés ; pas de
refit bootstrap, IC conditionnels aux prédictions OOF et pas une garantie
nationale. Recommandation préfixée : variante ayant le plus grand delta union
si borne simultanée>0 (égalité ordre déclaré), à confronter ensuite aux526
V142 ; sinon ne pas départager sur du bruit. Ne pas changer la recette après
résultats. Ce banc a un autre générateur, sa précision ne se transfère pas à
V142 ; les526 consommés restent un contrôle de transfert, pas un test vierge.

## Audit séparé et ressources

Faux SUR union content/margin et stack/margin au volume17427 et seuils publiés
optionD, pas les choix des full-fits. Rejouer scores, marges, seuil et sélection
sur les526, hydrater choix+label officiels, présenter CRM, nom/adresse, points
et distances typées, siège/site, état, même SIREN. Verdict humain prudent après
lecture des faits, jamais entraîné ni utilisé pour modifier le banc/labels.
Distance lointaine ≠ preuve de label faux ; indéterminé si preuve insuffisante.

Namespace neuf `H/step2_g0_bench_20261017/`. Workers séquentiels,1 thread,
DuckDB512Mio, garde RSS5,5Gio (<6), attendre le superviseur V142 ou descendant
RSS>4Gio avant phase lourde ; aucun signal/contrôle de l'autre agent/VM.
SHA sources/matrices avant/après, sceller features avant cibles, puis plan de
plis avant fits. Smoke réel :100 entreprises entières sélectionnées par SHA
indépendamment des résultats, puis complet17427. Pas de tests unitaires,
web/installation/Gemma/VM/push. Travaux tiers et pending préservés.


## Résultats

**Banc de développement, pas précision produit V142.**

17427 dossiers /11483 entreprises ;5 plis par entreprise, mêmes plis pour toutes variantes.
Présence@100 : exact15956, union16325 ; toutes absences conservées.

| Recette | Exact, Wilson95 | Union, Wilson95 | Δ exact, IC95 cluster | Δ union, IC95 cluster |
|---|---|---|---|---|
| base | 11824/17427 (67.85 %, IC95 [67.15;68.54]) | 12889/17427 (73.96 %, IC95 [73.30;74.61]) | — | — |
| plus_g0 | 11861/17427 (68.06 %, IC95 [67.36;68.75]) | 12964/17427 (74.39 %, IC95 [73.74;75.03]) | +0.212 [-0.355;+0.734]pt | +0.430 [-0.140;+0.946]pt |
| company_weight | 11790/17427 (67.65 %, IC95 [66.96;68.34]) | 12853/17427 (73.75 %, IC95 [73.09;74.40]) | -0.195 [-0.537;+0.158]pt | -0.207 [-0.547;+0.162]pt |
| union_target | 11528/17427 (66.15 %, IC95 [65.44;66.85]) | 12953/17427 (74.33 %, IC95 [73.67;74.97]) | -1.699 [-2.194;-1.208]pt | +0.367 [+0.011;+0.737]pt |

## Confiance union98% interne, évaluée sur entreprises externes

| Recette/politique | SUR | Précision union, Wilson95 | Exact mêmes SUR | Couverture, Wilson95 |
|---|---:|---|---|---|
| base/margin | 1682 | 1627/1682 (96.73 %, IC95 [95.77;97.48]) | 1622/1682 (96.43 %, IC95 [95.44;97.22]) | 1682/17427 (9.65 %, IC95 [9.22;10.10]) |
| base/scene_isotonic | 1228 | 1194/1228 (97.23 %, IC95 [96.16;98.01]) | 1173/1228 (95.52 %, IC95 [94.22;96.54]) | 1228/17427 (7.05 %, IC95 [6.68;7.44]) |
| plus_g0/margin | 1183 | 1149/1183 (97.13 %, IC95 [96.01;97.94]) | 1140/1183 (96.37 %, IC95 [95.14;97.29]) | 1183/17427 (6.79 %, IC95 [6.42;7.17]) |
| plus_g0/scene_isotonic | 1287 | 1240/1287 (96.35 %, IC95 [95.18;97.24]) | 1201/1287 (93.32 %, IC95 [91.82;94.56]) | 1287/17427 (7.39 %, IC95 [7.01;7.78]) |
| company_weight/margin | 1471 | 1433/1471 (97.42 %, IC95 [96.47;98.11]) | 1430/1471 (97.21 %, IC95 [96.24;97.94]) | 1471/17427 (8.44 %, IC95 [8.04;8.86]) |
| company_weight/scene_isotonic | 491 | 482/491 (98.17 %, IC95 [96.55;99.03]) | 481/491 (97.96 %, IC95 [96.29;98.89]) | 491/17427 (2.82 %, IC95 [2.58;3.07]) |
| union_target/margin | 1651 | 1600/1651 (96.91 %, IC95 [95.96;97.64]) | 1591/1651 (96.37 %, IC95 [95.35;97.17]) | 1651/17427 (9.47 %, IC95 [9.05;9.92]) |
| union_target/scene_isotonic | 592 | 578/592 (97.64 %, IC95 [96.07;98.59]) | 560/592 (94.59 %, IC95 [92.47;96.15]) | 592/17427 (3.40 %, IC95 [3.14;3.68]) |

- Socle des SUR = proper60%, distinct du ranking80% ci-dessus ; tous les seuils sont internes, cible union pour scène/isotonique/seuil.98% calibration ≠ garantie test.
- Pondération entreprise appliquée seulement au ranking, métriques query-weighted. Cible union=exact OU site, aucun label/candidat modifié.
- Bootstrap10000 entreprises entières, IC95 marginaux ci-dessus ; IC simultanés six comparaisons et plis dans JSON. Wilson descriptif sans correction cluster ; consommation du développement et changement de générateur empêchent une certification.

## Recommandation unique

Ne pas départager ces recettes sur le bruit ; rechercher un signal distinctif du site plutôt que promouvoir une variante non confirmée.

- 100 fits, 418248 vecteurs effectivement rejoués depuis poids sauvés ; garde5,5Gio, pic5.16Gio, temps2879.1s. Sources/matrices vérifiées avant/après.
- Audit séparé : `docs/step2_sur_errors_audit.md`. Aucun accès v174/VM/web/jeu fermé, changement V142 ou push. Les526 V142 restent un contrôle de transfert consommé, pas une validation indépendante.
- Reproduction : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_g0_bench.py run --output-dir <dossier_neuf_sous_H/step2_g0_bench_20261017>` ; `--smoke`100 entreprises réelles.
