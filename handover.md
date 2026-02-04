# Handover (Fenetre de contexte)

## Actions terminees dans cette fenetre
- Mise en place de la retrieval "Ultima" : TF-IDF sur noms + adresses (double indexation), plus rescue universel (hash d'adresse + tokens numeriques). (commit 4cee604)
- Suppression de la penalite de longueur TF-IDF (norm=None) pour corriger la dilution bag-of-names. (commit 4cee604)
- Alignement entrainement et inference sur le meme pipeline de retrieval (zero train/serve skew). (commit 4cee604)
- Generation de data/samples_v5fast_B_ultima.parquet et entrainement du Ranker B Ultima (run local, pas de commit) :
  - models/xgbranker_fast_20260131_v5fast_B_ultima.json
  - models/xgb_two_stage_meta_20260131_v5fast_B_ultima.json
- Verification de la couverture retrieval (mesure locale, pas de commit) :
  - Absolute recall ~90.6% sur tout le CRM.
  - Relative recall ~96.9% sur les requetes eligibles (pool non vide).
- Mise a jour de DECISIONS.md et SSOT.md pour consigner la strategie "Ultima". (commit e13298d)
- **Optimisation memoire generate_training_samples_v5fast.py** (commit pending) :
  - StreamingParquetWriter : ecriture incrementale avec flush configurable (--flush-every).
  - Traitement requete par requete (pas de batch accumulation).
  - GC explicite apres chaque loc_key.
  - Semantic late-compute : features semantiques calculees uniquement sur samples selectionnes.
  - Stats incrementales (plus de pd.read_parquet en fin de run).
  - Resume support (--resume) pour reprendre apres crash.
  - Knobs memoire : XGB_SEMANTIC_CACHE_SIZE (default 50k), XGB_SEMANTIC_BATCH_SIZE (default 128).
- **Entrainement Decider B Ultima Hardneg** (02/03/2026) :
  - Samples : data/samples_v5fast_B_ultima_hardneg.parquet
  - Decider : models/xgb_decider_20260203_v5fast_B_ultima_hardneg.json
  - Meta (decider only) : models/xgb_two_stage_meta_20260203_v5fast_B_ultima_hardneg.json
  - Metriques : hit@1=82.7% (test), AUC=0.983 (calibrated)
- **Consolidation du meta B Ultima Hardneg** (03/02/2026) :
  - Ajout ranker_model, ranker_fast_model, ranker_feature_order, ranker_fast_feature_order
  - Meta complet pour inference : models/xgb_two_stage_meta_20260203_v5fast_B_ultima_hardneg.json
- **Fix calibrator loading dans infer.py** :
  - Remplacement pickle.load direct par load_calibrator() pour eviter AttributeError
- **Script d'optimisation DS** (commit pending) :
  - Ajout de scripts/optimize_pipe.py pour calculer metrics top-k + sweep du threshold risk
- **Alignement inference/train** (commit pending) :
  - TF-IDF name retrieval aligne sur generate_training_samples_v5fast.py (ngram_range=(1,2), pas de max_df/norm)
  - SEMANTIC_GATE desactive par defaut pour matcher l'entraînement (XGB_SEMANTIC_GATE_ENABLED=0)
  - Partitions par defaut passees a data/candidates_v5_all
- **Alignement stage1_top_n inference/train** (commit pending) :
  - Default stage1_top_n passe a 50 (alignement avec l'entraînement)
  - Ajout flag --force-stage1-min-500 pour conserver l'ancien comportement
- **Alignement TF-IDF prefilter inference/train** (commit pending) :
  - Re-activation du TF-IDF prefilter en mode insee_then_postcode dans infer_xgb_two_stage.py

## Travail en cours
- **Orientation A terminée** : XgbInferenceEngine est maintenant le chemin canonique pour le mode `insee_then_postcode`
  - Le mode `multi` (parallèle) garde encore le code legacy pour l'instant
  - Le mode `debug_gt` garde aussi le code legacy pour le tracking détaillé

## Fichiers modifies dans cette fenetre
- src/xgb_matcher/infer.py : ajout infer_topk(), TopKRow, fix calibrator loading, ajout candidate_last_treatment_date
- scripts/infer_xgb_two_stage.py : refactoring du chemin séquentiel pour utiliser XgbInferenceEngine
- scripts/optimize_pipe.py : rapport d'optimisation et sweep de seuil risk
- src/xgb_matcher/blocking.py : alignement TF-IDF name avec train
- src/xgb_matcher/features.py : semantic gate default=0
- src/xgb_matcher/profile.py : partitions default v5_all
- models/xgb_two_stage_meta_20260203_v5fast_B_ultima_hardneg.json : ajout ranker paths

## Prochaines actions (dans l'ordre)
1. **Lancer inference top-k sur CRM complet** :
   - python scripts/infer_xgb_two_stage.py --crm-path data/crm_ok_gt.csv --meta-path models/xgb_two_stage_meta_20260203_v5fast_B_ultima_hardneg.json --output-path data/topk_B_ultima_hardneg.csv --pool-mode insee_then_postcode
2. **Construire dataset routing + entrainer risk model** :
   - scripts/build_routing_eval_dataset.py
   - scripts/train_routing_risk_model.py
3. **Benchmark final** : top-k + routing (cible ~75% AUTO @ 99.8% precision)
4. **(Optionnel) Refactoriser le mode multi** pour utiliser l'engine également

## Artefacts principaux
| Artefact | Chemin |
|----------|--------|
| Ranker Fast B Ultima | models/xgbranker_fast_20260131_v5fast_B_ultima.json |
| Decider B Ultima Hardneg | models/xgb_decider_20260203_v5fast_B_ultima_hardneg.json |
| Meta complet | models/xgb_two_stage_meta_20260203_v5fast_B_ultima_hardneg.json |
| Samples | data/samples_v5fast_B_ultima_hardneg.parquet |
| Candidates | data/candidates_v5_all/ (partitionne par INSEE/CP) |

## Contraintes cles a respecter
- Strict insee_then_postcode (pas de fallback departement, meme avec Places).
- Ranker fait le pruning ; decider voit top-200.
- Etablissements fermes autorises ; SIRET ouvert promu si meme SIREN.
- Strategie bag-of-names a conserver (besoin business).
