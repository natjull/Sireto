# Étape 2 — inventaire aval en lecture seule

Inventaire pour `docs/step2_choose_one_contract.md`. Aucun modèle chargé,
scoré ou appris ; lecture de code, métadonnées JSON et noms de fichiers seulement.
Aucun parquet d'ancienne population ni jeu fermé ouvert. Les folds0/1 cités
ci-dessous décrivent les **anciens protocoles**, pas une permission de lecture.

Abréviation **B** = `/Volumes/CATNAT_DATA/SIRETO_RECALL100`.
« Partiel » signifie code/architecture récupérable, **pas** poids/seuil validé
sur V142. Aucun modèle existant n'est directement validé pour ce nouveau pool.

## Entrées réellement disponibles

- V142 : `outputs.json["learned_native_ooc"]`, **526 × 100**, chaque candidat
  porte seulement **`siret`, `siren`, `score`**. Le rang vient de la liste,
  jamais d'un nouveau tri (y compris les ex æquo). SHA enregistré dans la baseline.
- Le panel scellé `$H/v14/representative_20261008_b/panel.json` fournit nom,
  adresse CRM décomposée, CP/ville/INSEE et fold. La géométrie BAN scellée
  `$H/v17/structural_20261010_a/geo.json` est disponible séparément.
- Il manque dans les sorties : noms propres des candidats, adresse, état,
  siège, NAF, nature juridique, provenance/rangs des anciens canaux, similarités.
  Les champs officiels peuvent être joints **localement sur les seuls100** ;
  les anciennes provenances ne peuvent pas être inventées à partir du score V142.
- Scores V142 = score conjoint natif, **pas** probabilités du decider historique,
  ni RRF/admission ; ne pas rebaptiser ces colonnes pour satisfaire un ancien modèle.
- Le ledger humain et le sidecar d'acceptabilité sont **évaluation/target uniquement**.
  La baseline hydrate des candidats du SIREN humain pour mesurer le site : cette
  sélection label-side et ses booléens ne sont jamais des features de serving.

## Éléments trouvés (une ligne par élément)

| Chemin(s) | Rôle et entrées d'apprentissage établies | Réutilisable sur V142 ? / pourquoi |
|---|---|---|
| `scripts/train_xgb_ranker.py` ; `models/xgbranker_20260124_210313.json` et `xgbranker_fast_20260124_210313.json` ; `models/xgb_two_stage_meta_20260124_210313.json` | XGB pairwise par query ; métadonnées : `data/samples_v6_C.parquet`, 47 features, variantes sémantiques/full et fast ; splits train/dev/test historiques. | **Partiel** : ranker groupwise utile, mais aucune des47 features n'est dans outputs ; hydratation/normalisation nécessaires. Exposition des526 aux anciens fits non établie ; ne pas transférer les poids comme preuve OOF. |
| `models/xgbranker_20260221_193026.json`, `xgbranker_20260221_224040.json` et métas voisines | Autres rankers XGB ; respectivement `samples_v7_ranker.parquet` (sémantique désactivée) et `samples_v7_decider.parquet` (activée), 47 features ; retrieval INSEE/CP, sparse+dense historique. | **Partiel** : même besoin d'adaptateur ; le méta224040 pointe aussi `decider_model` vers ce ranker, donc le nom de clé ne prouve pas un classifieur calibré. |
| `scripts/train_xgb_decider.py` ; `models/xgb_decider_20260124_210218.json` ; `models/xgb_decider_calibrator_isotonic_20260124_210218.pkl` ; méta210218 | Classifieur candidat + calibration ; `data/samples_v6_A.parquet`, 47 features, sémantique activée ; script peut restreindre au topK ranker (défaut50). | **Partiel** pour l'inférence adaptée, **non** pour le seuil/calibrateur tel quel. `filter_topk_by_ranker` réajoute le positif y compris dev/test : ne pas réutiliser cette logique d'évaluation. La promotion actif/fermé du script n'exige pas la preuve même site, donc pas la politique opérationnelle actuelle. |
| `models/xgb_decider_20260221_213148.json` et méta213148 ; `models/xgb_decider_v5fast_B_clean_hardneg_final.json` et méta homonyme | Variantes decider sur `data/samples_v7_decider.parquet` et `data/samples_v5fast_DECIDER_hardneg_clean.parquet`, 47 features ; rankers/pools historiques différents. | **Partiel** : pas de compatibilité native outputs, provenance/exposition à contrôler et confiance à recalibrer. Les variantes ne constituent pas des expériences indépendantes sur V142. |
| `src/xgb_matcher/features.py`, `naming.py`, `semantic.py` ; `scripts/infer_xgb_two_stage.py` | Primitives communes nom/adresse, IDF/densité, état/siège, sémantique ; pas d'apprentissage dans `make_features`, modèle sémantique séparé. | **Partiel** : calcul label-free possible après hydratation officielle. Code courant54 features contre47 dans plusieurs poids ; respecter l'ordre du méta, pas `FEATURE_NAMES` courant aveuglément. IDF/densité dépendent de l'univers historique. Retirer les entrées alias CRM, notamment `alias_match`, dans une nouvelle recette ; ne pas compléter les manques par des zéros arbitraires. |
| `src/xgb_matcher/routing_risk.py`, `scripts/build_routing_eval_dataset.py`, `scripts/train_routing_risk_model.py` | Métamodèle requête : score top1/top2/gap/ratio +20 signaux candidat ×3 ; prédiction de justesse top1. Dataset via `--dataset-path`, splits ou repli aléatoire historique. | **Partiel** : construction de scène utile sans labels, mais ce sont des scores decider et des features absents de V142 ; le repli aléatoire ne respecte pas le nouveau protocole groupé. |
| `models/routing_risk_{model,calibrator}.pkl`, `models/routing_risk_meta.json` ; `models/risk_v7/routing_risk_{model,calibrator}.pkl` et méta | Deux risques XGB isotoniques, **64** features dans les métas (pas68), cible **SIREN**, seuils0,985 et0,805 ; chemin exact du dataset non enregistré dans ces métas. | **Non en l'état** : bonne entreprise ≠ bon SIRET/site ; échelle des scores et distribution différentes. Ne pas reprendre le0,835 documentaire ni revendiquer99,84 % sur V142 ; provenance de fit insuffisamment traçable ici. |
| `scripts/train_v9_ranker.py` ; `B/models/ranker_v4/ranker_1aebeada820d92a7_6236365/ranker.json` et `metadata.json` | Ranker V9/V4 sur top100 sélectifs ; dataset `1aebeada820d92a7`, **55** features non sémantiques avec admission multicanal, labels qualifiés MATCH_EXACT, cinq folds groupés ; positif non injecté selon méta. | **Partiel** : architecture100 adaptée ; provenance de canaux manquante, population qualifiée différente de all-human526 ; ne pas réutiliser le filtre de dénominateur historique. |
| `src/xgb_matcher/v9_features.py` ; `scripts/build_downstream_selective_dataset.py`, `build_v412_learned_candidate_features.py` | Features sparse/dense/RRF et admission, hydratation à partir de partitions et pools officiels, labels séparés des matrices. | **Partiel** : rang réciproque et agrégations candidat calculables sans vérité ; canaux V7/overlay absents des sorties V142. Les builders par défaut lisent d'autres populations, dont test : **ne pas les exécuter tels quels**. |
| `src/xgb_matcher/v9_scene.py`, `v9_acceptor.py` ; `scripts/train_v9_acceptor.py` ; `B/models/acceptor_v4/acceptor_2b8a9c994e0944be_9ec88c8/{acceptor_model.joblib,acceptor_calibrator.joblib,metadata.json}` | Accepteur logistique standardisé/isotonique, scènes top1/top2/concurrence, cible exacte sur prédictions OOF ; métadonnées : dataset `2b8a9c994e0944be`, seuil1,0. `build_inference_scene` ne lit pas de label. | **Partiel** : contrat accept/review récupérable ; score/distribution/provenances différents, seuil non transférable. `retrieval_miss` est constant0 avec candidats, pas une feature autorisée déduite de l'absence du label. |
| `src/xgb_matcher/v41_features.py`, `v41_acceptor.py`, `v41_decision.py` ; `B/models/v4_1/` | Ancienne vérification d'un **SIRET CRM suspect** : relations input_siret/input_siren, provenance input/alias fermé, états et accepteur. | **Non en l'état** : identifiant CRM et signaux dérivés hors contrat, non présents dans le pilote V142. L'ancien nom « sans identifiant brut » n'autorise pas ces dérivés. |
| `scripts/run_v411_ranker_c_development.py` ; `B/models/v4_11_ranker_c/0d2e419158c7a4c0/ranker_c/{full_fit.json,metadata.json}` (autre build `e13eb3ac7498256e`) | Ranker C input-blind, **45** features =44 non sémantiques + rang réciproque ; dataset `ec4326ec57e4411d`, cinq anciens folds0–4, labels exacts et groupes SIREN. | **Partiel**, meilleur point de départ structurel : pas besoin des anciens canaux. Reste hydratation, suppression d'alias CRM, audit des groupes et protocole2/3/4 ; les modèles OOF historiques ne correspondent pas aux folds actuels. |
| `scripts/run_v48_acceptor_development.py`, `run_v410b_structured_acceptor_development.py` ; `src/xgb_matcher/v410b_acceptor.py` ; `B/experiments/v4_8_acceptor_development/`, `v4_10b_structured_acceptor_development/` | Accepteurs logistiques/XGB sur scènes historiques et cas difficiles consommés, pondérations, scaler sélectif ; plans/hashes propres à ces expériences. | **Partiel** pour preprocessing et protocole, pas exécutable directement sur V142. Les cohortes difficiles/pondérations historiques ne sont pas une autorisation de sélection client dans ce jalon. |
| `src/xgb_matcher/v411_scene.py`, `v411_acceptor.py` ; `scripts/run_v411_acceptor_development.py` ; `B/models/v4_11_acceptor/9d23bf3deb6b63de/bundle/{acceptor_model.joblib,metadata.json}` | Scène80 features, COMPACT_LOGIT (alternative monotonic XGB), gaps normalisés, rôles/NAF, justesse top1 ; métadonnées scène hashée et seuil0,8720916707, FORCE_REVIEW unresolved. | **Partiel** : concurrence et scaler label-free réutilisables après hydratation ; seuil et filtre unresolved non transposables aveuglément, ne jamais filtrer526. |
| `scripts/build_v412_learned_business_features.py`, `evaluate_v412_ranker_business_features.py` ; `B/datasets/v4_12_learned_business_features/a77a3a5b226cfbe6/` | Enrichissement source/relations : noms UL/établissement, siège, activité, dates, concurrence même adresse/SIREN ; tables officielles locales. | **Partiel** : signaux source calculables sans label après jointure ; ordre inclut anciennes features d'admission absentes et `parenthetical_alias_exact` à exclure selon contrat. Agrégats à recalculer sur exactement les100, sans label. |
| `scripts/train_v412_learned_oof_rankers.py` ; `B/experiments/v4_12_learned_oof_rankers/6f9b07472d2acb5f/models_{baseline_45,business_learned,business_weak_labels}/full.json` et `manifest.json` | Rankers pairwise BASELINE_45/BUSINESS, négatifs bornés, labels humains exacts ou weak selon variante ; ancien dataset métier, cinq folds0–4 imposés par runner. | **Partiel** pour squelette ; **non** comme apprentissage OOF directement conforme. Weak labels, folds0/1 et anciennes populations non autorisés ici ; pas de reprise des poids sans contrôle d'exposition. |
| `scripts/train_v412_learned_acceptor.py` ; `B/experiments/v4_12_learned_acceptor/cc3cc3ca41e4e783/models_{logistic,xgboost}/outer_fold_*` et `manifest.json` | Accepteur imbriqué : outer=f, calibration=(f+1)%5, train trois autres ; scènes top3/autre SIREN/absences CRM, cible top1 et seuil visant précision0,998 historique. | **Partiel** : séparation train/calibration/évaluation pertinente ; scènes absentes de V142 et runner cinq folds incompatible avec2/3/4. Adapter à groupes internes, pas ouvrir0/1. |
| `scripts/evaluate_v412_cross_encoder_reranker.py`, `evaluate_v412_neural_reranker.py`, `src/xgb_matcher/v9_cross_encoder.py` | Pistes texte pairwise/groupwise ; exemple mMARCO MiniLM local, ancien top20 ranker, variantes de mélanges. | **Partiel**, pas priorité : nécessite textes candidat hydratés et nouvelle validation ; ne pas relancer la grille ni les anciens jeux. Aucun score neuronal V142 mesuré ici. |
| `src/xgb_matcher/selective.py` | Courbes risque/couverture, choix de seuil et bornes de précision ; aucune donnée d'apprentissage intrinsèque. | **Oui pour les utilitaires** sur scores/labels OOF séparés ; seuil à choisir sur calibration seulement et Wilson95 à publier selon nouveau contrat. Aucune garantie transférée. |

## Points à lever avant tout fit

Les anciens `models/` sont non suivis par Git ; leur présence n'est pas une
provenance scellée. Les métadonnées donnent des fichiers/datasets, pas une
preuve d'absence des526 entreprises dans leurs fits. Aucun ancien poids n'est
promu, aucune ancienne précision AUTO n'est reprise. La compatibilité décrite
est une **inspection de schéma/code**, pas un smoke d'inférence aval.
La sémantique des noms UL, établissement et dirigeants doit rester distincte ;
aucun alias CRM ni nom d'un tiers copié comme preuve propre du site.

## Première expérience recommandée — une seule, non exécutée

**Un ranker XGB compact OOF sur les100 V142 figés, suivi d'un seuil de marge
calibré sur groupes internes**, sans empiler decider, risk et accepteur hérités.
Reprendre le squelette input-blind du ranker C, pas ses poids ni son runner
cinq-folds. Préenregistrer une petite liste de signaux nom/adresse officiels,
état/siège, rang et score natifs, distance BAN avec précision/manquants ;
exclure identifiants et alias CRM. Hydrater tous les100 sans utiliser la vérité.
Pour chaque fold2/3/4 évalué, fit et calibration séparés par entreprise sur
les deux autres ; exclure aussi toute entreprise partagée avec le fold évalué.
Les requêtes sans positif ne fournissent pas de paire positive au fit, mais
restent **toutes** au dénominateur et dans l'évaluation de confiance.
Choisir l'objectif de précision `SUR` et la règle de seuil avant le run ;
si aucune couverture ne le satisfait sur calibration, tout passe `A_VERIFIER`.

Pourquoi : **353/526 top1 exacts**, mais **154 autres labels sont déjà dans100**
(dont73 aux rangs2–3). Il existe donc un problème de choix mesuré, distinct des
19 absences que ce ranker ne peut résoudre. Ce protocole teste directement ce
besoin avec des primitives déjà présentes, sans coût neuronal ni réutilisation
d'un risque entraîné sur le seul SIREN. Publier gains/pertes contre353,
les cinq vues, `SUR` exact/opérationnel et Wilson95, par fold et au total.
Cela restera un résultat de développement avec amont consommé/non full-OOF,
pas une certification de production.
