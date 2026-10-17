# Premier reclasseur XGBoost V142 — paramètres fixes

**526 requêtes**,100 candidats inchangés chacune ; aucun modèle de recherche recalculé.
Statut : développement sur pilote consommé, **pas certification production**.
**Conclusion : CIBLE_SUR_NON_ATTEINTE_POLITIQUE_NON_VALIDEE.** Les statuts `SUR` ci-dessous sont expérimentaux, non promus.

## Protocole fixé avant les fits

- `rank:pairwise`,200 arbres, profondeur3, taux0,05, min_child10, L2=10, L1=1, max_bin128, un thread ; graine20261017. Aucune HPO/early stopping.
- Audit SIREN humain : folds2/3/4 disjoints par entreprise. Chaque test externe exclu de tous ses fits et seuils.
- Pour chaque test externe, prédictions OOF sur les deux folds d’apprentissage (deux fits internes). Seuil de marge brut maximisant la couverture à précision exacte empirique≥98 % sur ces prédictions internes uniquement.
- Si aucun seuil admissible : tous `A_VERIFIER`. Aucun minimum de couverture ni borne Wilson imposé ;98 % est une cible de calibration, **pas une garantie test**.
- Ex æquo : ordre V142 conservé. Requêtes sans positif gardées au fit et à toute évaluation ; aucun positif ajouté.
- Familles reprises des anciens XGB : noms officiels propres UL/établissement/sigle, Jaro/Levenshtein/tokens, adresse/numéro, distance admissible et précision, état/siège, rang/score, concurrence sur les100.
- Aucun nom tiers dirigeant, alias CRM, identifiant brut, variable de label ou vérité de site en entrée. IDs utilisés seulement pour jointures/groupes et comptages SIREN demandés.
- Features construites par interface sans labels sur TOUS les100 ; ledger/sidecar lus séparément pour cibles et métriques. Pas de TF-IDF global appris, d’encodeur ni de poids historiques réutilisés.

## Top1 avant / après (Wilson95, dénominateur intégral)

| Vue | Avant | Après | Gains / pertes |
|---|---|---|---:|
| SIRET exact | 353/526 (67.11 %, IC95 [62.98;70.99]) | 356/526 (67.68 %, IC95 [63.57;71.54]) | +23/−20 |
| SIREN | 406/526 (77.19 %, IC95 [73.41;80.57]) | 398/526 (75.67 %, IC95 [71.82;79.14]) | +7/−15 |
| Opérationnel historique | 377/526 (71.67 %, IC95 [67.68;75.36]) | 383/526 (72.81 %, IC95 [68.86;76.44]) | +22/−16 |
| Site opérationnel | 315/526 (59.89 %, IC95 [55.64;63.99]) | 311/526 (59.13 %, IC95 [54.87;63.25]) | +11/−15 |
| Union exact ∪ site | 391/526 (74.33 %, IC95 [70.43;77.88]) | 391/526 (74.33 %, IC95 [70.43;77.88]) | +16/−16 |

## Par fold

| Fold | N | Top1 exact avant | Après | Gains/pertes | Taux SUR | Précision exacte SUR |
|---|---:|---|---|---:|---|---|
| 2 | 178 | 124/178 (69.66 %, IC95 [62.55;75.94]) | 121/178 (67.98 %, IC95 [60.81;74.39]) | +4/−7 | 0/178 (0.00 %, IC95 [0.00;2.11]) | non définie |
| 3 | 162 | 107/162 (66.05 %, IC95 [58.46;72.89]) | 106/162 (65.43 %, IC95 [57.83;72.32]) | +6/−7 | 23/162 (14.20 %, IC95 [9.65;20.40]) | 20/23 (86.96 %, IC95 [67.87;95.46]) |
| 4 | 186 | 122/186 (65.59 %, IC95 [58.51;72.04]) | 129/186 (69.35 %, IC95 [62.39;75.53]) | +13/−6 | 5/186 (2.69 %, IC95 [1.15;6.14]) | 4/5 (80.00 %, IC95 [37.55;96.38]) |

## SUR / A_VERIFIER

- Taux SUR : **28/526 (5.32 %, IC95 [3.71;7.59])**.
- Précision SUR SIRET exact : 24/28 (85.71 %, IC95 [68.51;94.30]).
- Précision SUR SIREN : 24/28 (85.71 %, IC95 [68.51;94.30]).
- Précision SUR Opérationnel historique : 24/28 (85.71 %, IC95 [68.51;94.30]).
- Précision SUR Site opérationnel : 21/28 (75.00 %, IC95 [56.64;87.32]).
- Précision SUR Union exact ∪ site : 24/28 (85.71 %, IC95 [68.51;94.30]).
- Test fold 2 : seuil=4.896793842315674, calibration interne 1/348 acceptés ; précision exacte 1/1 (100.00 %, IC95 [20.65;100.00]).
- Test fold 3 : seuil=3.1224396228790283, calibration interne 53/364 acceptés ; précision exacte 52/53 (98.11 %, IC95 [90.06;99.67]).
- Test fold 4 : seuil=4.426482677459717, calibration interne 12/340 acceptés ; précision exacte 12/12 (100.00 %, IC95 [75.75;100.00]).

### Courbe descriptive sur prédictions externes (jamais utilisée pour choisir le seuil)

| Marge ≥ | SUR | Taux SUR | Précision exacte SUR |
|---:|---:|---|---|
| 0.0 | 526 | 526/526 (100.00 %, IC95 [99.27;100.00]) | 356/526 (67.68 %, IC95 [63.57;71.54]) |
| 0.1 | 501 | 501/526 (95.25 %, IC95 [93.08;96.76]) | 351/501 (70.06 %, IC95 [65.91;73.91]) |
| 0.25 | 468 | 468/526 (88.97 %, IC95 [86.01;91.37]) | 339/468 (72.44 %, IC95 [68.22;76.29]) |
| 0.5 | 428 | 428/526 (81.37 %, IC95 [77.82;84.46]) | 321/428 (75.00 %, IC95 [70.69;78.87]) |
| 1.0 | 351 | 351/526 (66.73 %, IC95 [62.59;70.62]) | 285/351 (81.20 %, IC95 [76.78;84.94]) |
| 2.0 | 191 | 191/526 (36.31 %, IC95 [32.32;40.51]) | 175/191 (91.62 %, IC95 [86.83;94.78]) |

## Appariement exact (query_id)

Gains : `5298, 5367, 6519, 7154, 9322, 9368, NEWCRM:15a9b2d94c7e899d96bccc72caadc4d0, NEWCRM:2e142b4936e6f3f578f92e7ac61935b8, NEWCRM:2f364487762c7d028ec01ad443c97e23, NEWCRM:321ee3f4d681f5e1da4c21ec82bc180b, NEWCRM:341cf49a3b9260fc33d9094367faad40, NEWCRM:3c1f76f851d6ed6e35937eacb790e0ff, NEWCRM:4f469e7ee22bb84bbb323838b31d19c4, NEWCRM:5e32afcf4b77124f08a7c3c791821324, NEWCRM:6b39723f7aa2e877a49de19e0f3339f4, NEWCRM:7d92ccf89904477aad39be1ab49ecdf9, NEWCRM:9a48d10fcdfe21537f64405ff75d8afe, NEWCRM:b2c971005cc22fd0e460bc3259af54fd, NEWCRM:b4ea4149620da9b1db339845d6aa7d37, NEWCRM:deb935b4c6a9bed658d5cbd7487f14b9, NEWCRM:dfdd81cf60782f63b5adfa815739f858, NEWCRM:e67936d685e6396cb122e5ff6a4a6871, NEWCRM:fd677d022a439203558565af3df3e40c`.
Pertes : `10222, 1294, 14847, 16478, 16824, 1731, 3005, 526, NEWCRM:4743886158fdbd64a21c2e6fd9cce6fb, NEWCRM:6155fa9b30c1648df753eaa5a7bf6b05, NEWCRM:65fb6a2e1f1c372bfc42d621844f4d6d, NEWCRM:68a8b68ca6397b415c737becffa1a1fb, NEWCRM:6bd10474375b3210f4305b7e911a2ad3, NEWCRM:6d1832aa6ea1091e257a051e3d17eee3, NEWCRM:bdd1aafce8e09b7050a2f1a512b87afb, NEWCRM:c2a0cc827604202fa09bacaf1e2a34b4, NEWCRM:cd9fa35a6dcb2c8c400ea66333d3b28f, NEWCRM:d83b771e3de6341c39541ff896e568d2, NEWCRM:f2f4a03be0319d2ff969edf869350ddd, NEWCRM:fec7a588d14966c763b95ae427b2e045`.

## Top10 importance (gain moyen normalisé des trois modèles externes)

- `v142_score_relative` : 0.257735.
- `same_siren_as_native_top1` : 0.242509.
- `v142_rank` : 0.141813.
- `v142_delta_best` : 0.071464.
- `v142_delta_next` : 0.026365.
- `street_tokens` : 0.025295.
- `name_levenshtein_max` : 0.023914.
- `is_active` : 0.018535.
- `number_log_diff` : 0.018076.
- `full_address_match` : 0.017822.

## Contrôle permutation

- Même matrice, paramètres et validation. Dans CHAQUE fit, permutation des vecteurs pertinence100 entre requêtes d’apprentissage, sans échantillon externe. Les labels sources restent immuables ; les positifs par rang sont conservés en distribution.
- Ce contrôle transfère les positions pertinentes, pas les SIRET entre pools (ce qui annulerait presque tous les positifs et rendrait le fit trivial). Seuils internes et test évalués sur les vérités originales.
- Top1 exact contrôle : **346/526 (65.78 %, IC95 [61.62;69.71])**, contre 353 avant ; gains/pertes +0/−7.
- Le rang natif reste informatif car amont supervisé gelé : le contrôle ne prouve pas une indépendance end-to-end. Les autres vues, seuils et reçus du contrôle sont dans le JSON.

## Plus d’exemples — comptage seulement

- Ledger folds2/3/4 : **23571** labels humains exploitables, dont **23045** hors pilote ; aucun statut ne filtre ce compte.
- Temps natif supplémentaire conditionnel : **56.3–163.0 h**, hors préparation/sources/attentes ;107 noyaux V142 réellement recalculés et20 processus V173 comparables (pas des coûts mesurés sur les nouvelles requêtes).
- Pic indicatif 7.22–8 Gio, pas garanti sous8 hors échantillon ; affectation V142 limitée526 et modèle production unique absent. Aucune génération lancée.

## Intégrité / limites

- Rappels@100 inchangés : exact507, SIREN511, opérationnel508, site404, union511 ; résultat vérifié requête par requête contre la baseline.
- Vue site : même SIREN ET (adresse concordante OU distance admissible200/500 m). Exact ne vaut pas automatiquement site, union=exact OU site. Aucune vérité modifiée.
- OOF strict concerne CE ranker ; amont consommé, non full-OOF et sélection de coefficients par SIREN humain inchangés. Ne pas présenter ces scores comme une validation indépendante de toute la chaîne.
- Wilson95 brut par requête est descriptif ; il ne corrige ni corrélation intra-entreprise ni consommation du pilote. Les seuils calibrés sur de très petits effectifs acceptés restent fragiles et les marges de fits internes/externes peuvent changer d’échelle.
- Le JSON contient métriques/IC des cinq vues par fold, paires gains/pertes, décisions526, ordre des features, hashes matrice/entrées/modèles et reçus de disjonction de chaque fit.
- Temps total : 26.64s ; pic RSS échantillonné : 567.1 Mio, garde7,5 Gio. Aucun jeu fermé, web, VM ou push.
- Reproduction : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_xgb_ranker_v1.py --output-dir <nouveau_dossier_CATNAT>` ; `--smoke` mesure6 vraies requêtes et ne remplace pas les résultats docs.
