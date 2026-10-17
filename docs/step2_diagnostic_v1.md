# Étape2 — diagnostic données / algorithme / features

**526 requêtes ×100 candidats ×46 features V1 inchangés.** Recherche gelée `36c7a476` ; pilote consommé, pas certification production.
Plan complet écrit avant fits dans le répertoire du run ; aucun réglage après résultats.

##1. Courbe d’apprentissage R2

| Part des entreprises train | Exact moyen / N | % moyen ± écart-type |
|---:|---:|---:|
| 25% | 349.40/526 | 66.43 ± 1.66 |
| 50% | 356.20/526 | 67.72 ± 1.05 |
| 75% | 355.80/526 | 67.64 ± 1.25 |
| 100% | 356.00/526 | 67.68 ± 0.00 |

- Sous-échantillons imbriqués par entreprises entières, cinq graines fixes20261017–21 ; graine du modèle constante. Fractions d’entreprises, pas quotas exacts de requêtes : effectifs réels détaillés par fold/graine en JSON.
- Écart-type échantillonnal entre graines, pas un IC ;100% réutilise le même fit déterministe, dispersion nulle attendue.
- Dernier incrément75→100% : **+0,04 ±1,25 points** (écart-type des différences entre graines), positif pour3/5 graines. Le signe moyen est positif mais très inférieur à la dispersion : **pas de croissance nette observée, plateau à cette résolution entre50 et100%**. Cela n’extrapole pas la performance à3000.

##2. Comparaison des algorithmes

- R1 : imputation médiane et standardisation fit uniquement sur train, logistique C=1, max_iter1000 (budget de convergence, pas HPO).
- R2 : V1 pairwise200 arbres/profondeur3 ; R3 : seule l’objective devient rank:ndcg ; R4 : binary:logistic, mêmes paramètres hors réglages lambdarank inapplicables.
- R5 : HistGradientBoostingClassifier par défaut, seule graine fixée ; early_stopping auto conserve son split interne par candidats, uniquement dans le train externe, pas un split entreprise interne.
- Aucun poids de classe, recherche de paramètres, arrêt sur test, positif injecté ou requête retirée. Départage stable selon l’ordre V142.

| Modèle | Exact | SIREN | Opérationnel | Site | Union | Gains/pertes exact vsR0 | p exact vsR0 / vsR2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 353 | 406 | 377 | 315 | 391 | +0/−0 | 1 / 0.7608 |
| R1 | 355 | 409 | 380 | 322 | 397 | +13/−11 | 0.8388 / 1 |
| R2 | 356 | 398 | 383 | 311 | 391 | +23/−20 | 0.7608 / 1 |
| R3 | 361 | 409 | 386 | 320 | 399 | +18/−10 | 0.1849 / 0.4583 |
| R4 | 360 | 410 | 385 | 323 | 402 | +12/−5 | 0.1435 / 0.5966 |
| R5 | 347 | 399 | 375 | 310 | 388 | +18/−24 | 0.4408 / 0.1877 |

- Toutes valeurs ont le dénominateur intégral,19 absences restent erreurs. IC Wilson95, cinq vues, folds et query_id des paires en JSON.
- McNemar exact bilatéral (discordances), exploratoire par requête : dépendances intra-entreprise/pilote consommé non corrigées. Holm sur les9 comparaisons exactes R1–5 vsR0 et R1/3/4/5 vsR2 pour la recommandation ; autres vues brutes descriptives.

##3. Ablations dans R3

| Retrait / variante | Top1 exact | Écart vsR3 |
|---|---:|---:|
| without_name | 363/526 | +2 |
| without_address | 364/526 | +3 |
| without_geography | 363/526 | +2 |
| without_state | 347/526 | -14 |
| without_native | 339/526 | -22 |
| without_scene | 346/526 | -15 |
| native_plus_scene_only | 348/526 | -13 |

- Les retraits nom/adresse enlèvent aussi interactions et deltas de scène dérivés ; retrait natif enlève aussi les deux ancres de scène top1 natif. Listes exactes dans le plan/JSON.
- La variante native+scène conserve14 features existantes, dont deltas nom/voie et densité adresse : ce n’est PAS une variante « scores seuls ». Pool et départage restent natifs dans toutes les ablations.
- Ablation ≠ importance causale : redondances, réentraînement et faible effectif peuvent changer le signe.

##4. Confiance : marge / scène calibrée / conforme

**Protocole de comparaison honnête, commun aux trois méthodes :** chaque train externe est séparé par entreprise en60% fit propre R3,20% isotonique,20% seuil/conforme. Le classifieur scène logistique C=1 est fit sur un OOF entreprise3-plis DANS les60%. Les20% isotonique ne servent pas au seuil. Aucun modèle R3 utilisé sur une calibration n’a vu les labels de ses entreprises.
Ce R3 est donc entraîné sur moins de données que le R3 complet de la comparaison ; on ne transfère pas son seuil à un full-fit.
Top1 exact du socle commun : 344/526 (65.40 %, IC95 [61.24;69.34]).

| Méthode | SUR | Couverture avec Wilson95 | Précision exacte avec Wilson95 | Précision opérationnelle |
|---|---:|---|---|---|
| margin | 92 | 92/526 (17.49 %, IC95 [14.48;20.97]) | 86/92 (93.48 %, IC95 [86.49;96.98]) | 88/92 (95.65 %, IC95 [89.35;98.30]) |
| scene_isotonic | 63 | 63/526 (11.98 %, IC95 [9.47;15.03]) | 59/63 (93.65 %, IC95 [84.78;97.50]) | 59/63 (93.65 %, IC95 [84.78;97.50]) |
| split_conformal | 0 | 0/526 (0.00 %, IC95 [0.00;0.73]) | non définie | non définie |

- Marge et probabilité : seuil sur20% seuil seulement, couverture maximale à précision exacte empirique≥98%, aucune acceptation si impossible. Aucune borne Wilson/minimum d’effectif imposés par la recette.
- Scène : marge, entropie et dispersion scores, accord natif, nombre de sites, nom/adresse, géographie et état du top1 ; uniquement les46 features et sorties du ranker, pas d’ID ni de vérité en entrée.
- Conforme : split conformal α=2%, quantile fini corrigé sur maximum par entreprise de1−softmax(vérité).101e classe OUTSIDE à probabilité0 pour les vérités absentes ; celles-ci ne sont jamais retirées. SUR seulement pour un singleton candidat sans OUTSIDE.
- Couverture de l’ensemble conforme (OUTSIDE inclus) : 526/526 (100.00 %, IC95 [99.27;100.00]) ; tailles des ensembles : `{'101': 526}`.
- **98% de couverture marginale d’un ensemble n’est pas98% de précision des SUR.** Une garantie conforme exige échangeabilité des entreprises, non certifiée ici ; une calibration avec absences peut imposer tous les101 labels et aucune décision.
- **Cible98% non atteinte** : la scène calibrée a presque la même précision que la marge (93,65 contre93,48%), avec moins de couverture (11,98 contre17,49%). Pas d’avantage observé justifiant une cascade décideur+risque ; cette cascade elle-même n’est pas testée. Un seul modèle de confiance reste le comparateur parcimonieux, pas une politique validée ; seuils/poids non promus.

##5. Résiduels R3 : vérité présente mais non première

| Famille exclusive | Effectif |
|---|---:|
| ferme_vs_actif | 19 |
| siege_vs_site | 26 |
| meme_siren_autre_site | 2 |
| homonyme | 2 |
| adresse_proche | 40 |
| autre | 57 |
| **Total** | **146** |

- Diagnostic label-side uniquement, jamais des règles de ranking : priorité fermé/actif même SIREN, siège différent même SIREN, autre site même SIREN, homonyme approché (deux Jaro≥0,95), adresse proche approchée (deux concordances ou deux distances admissibles≤200m du CRM), autre.
- Catégories exclusives et reproductibles, pas causes établies par audit humain ; « autre site » signifie ici autre SIRET, pas preuve d’un lieu physique distinct. Ambiguïté sémantique et qualité du label ne sont pas résolues. Les absences restent au dénominateur global, hors ce tableau explicitement conditionnel.

## Recommandation unique et FusionSet

**meilleures_features** : Pas de gain algorithmique confirmé après Holm et pas de croissance>=1 point sur chacun des deux derniers incréments. Priorité au signal distinctif, pas preuve de causalité ni exclusion d’un bénéfice de données supplémentaires.
Features à tester ultérieurement : preuve de nom propre du site distincte du nom UL, adresse/geolocalisation datée et fiabilité de la source, compatibilité sémantique activité/usage CRM et données officielles.
- Règle de décision écrite avant fits : gain algorithme confirmé vsR2 après Holm → changer ; sinon deux derniers incréments de courbe≥1 point chacun → données ; sinon features. C’est une priorité expérimentale, pas une identification causale certaine.
- **FusionSet non testé.** Les compteurs/deltas de scène regardent déjà les100, sans attention apprise. Avec526 dossiers (~350 train externe), aucune preuve ne justifie maintenant un réseau d’attention original. Gemma pourrait apporter un signal sémantique absent des46, mais cet apport de features est confondu avec l’architecture proposée : l’isoler avant un modèle hybride, puis réévaluer l’attention avec plus de groupes indépendants. Aucun gain ni volume minimal promis.

## Preuves / limites

- 93 fits candidats distincts (réemploi des100% et caches déterministes), reçus groupés/modèles sauvegardés ; durée67.56s, pic RSS609.5Mio, garde7,5Gio.
- Matrice et listes vérifiées par SHA avant/après ; cinq vues identiques à la baseline. R2 complet vérifié requête par requête contre V1. Features et labels immuables, pas d’embedding/calcul retrieval, jeux fermés, VM, web, dépense ou push.
- Validation OOF de ces seuls modèles aval : amont supervisé consommé/non full-OOF, routage par SIREN humain inchangé. Pas de précision nationale/production ni de garantie de SUR. Pas de déduction que « absence de significativité = égalité ».
- Commande : `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/step2_diagnostic_v1.py --output-dir <dossier_CATNAT_neuf>` ; `--smoke` utilise36 dossiers réels, mêmes réglages, sans écraser les docs.
