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

Statut **RUNNING_GROUPED_CV_RESULTS_PENDING** ; implémentation `a4b346e1`,
protocole initial `a4a4d451`. Le complet est lancé détaché, PID70288 :
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u scripts/step2_g0_bench.py run --commit`.

- Préparation complète vérifiée :17427 dossiers /11483 entreprises,
  présence dans les100 G0 **15956 exacts /16325 union**,8150 candidats
  supplémentaires positifs union. Ce sont des présences, **pas le top1**.
- Deux smokes réels indépendants :100 entreprises /134 dossiers,100 fits
  chacun ;100 modèles/cibles et3216 vecteurs vérifiés, mêmes octets modèles,
  features/cibles et décisions confiance dans les deux runs ; zéro écart.
  Second smoke57,67s, pic641957888octets (0,60Gio), rapport31 lignes généré.
- Audit complet reproduit :5 faux SUR /4 dossiers, faits SHA
  `892166e52c6e979829af2664443c21362875088fd577540de175d6e7999d1daf` identique
  entre les deux smokes et le complet ; voir `docs/step2_sur_errors_audit.md`.
- **Non encore mesurés/publiés** : top1 OOF complet base/3 variantes,
  SUR complet, deltas bootstrap/IC, recommandation issue du grand banc.
  Le calcul exécute les4 recettes sans choix sur les résultats partiels.

Suivi : `tail -f H/step2_g0_bench_20261017/full.log`, et progression CV dans
`full/<recette>/worker.log`. À succès, le script remplace ce checkpoint par
les résultats complets JSON/Markdown, produit `full/final_report.md` (≤35
lignes) et crée un commit docs/handover isolé ; `full/publication.json`
atteste le commit effectif. Si le journal partagé est modifié par un tiers,
la publication s'arrête sans écraser ces modifications. Aucun succès final
ni commit final n'est annoncé avant leurs artefacts.
