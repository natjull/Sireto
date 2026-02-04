# Handover (Fenetre de contexte)

## Actions terminees dans cette fenetre
- Mise a jour documentation de handover et regle de suivi. (commit 96849f1)
- Alignement inference/train sur la strategie ULTIMA (double TF-IDF nom+adresse + rescue universel), semantic gate desactive par defaut, partitions v5_all par defaut, et chargement calibrator robustifie. (commit 0aa5daa)
- Alignement infer_xgb_two_stage.py avec stage1_top_n=50 et reactivation du prefilter TF-IDF en insee_then_postcode. (commit 37299d5)
- Optimisation memoire de generate_training_samples_v5fast.py + nettoyage automatique des partitions v5 pour eviter les doublons. (commit e27729b)
- Nouveau ranker champion (run local, pas de commit) :
  - Ranker fast : models/xgbranker_fast_v5fast_B_clean_500neg.json
  - Meta ranker : models/xgb_two_stage_meta_v5fast_B_clean_500neg.json
- Entrainement Decider B Ultima Hardneg (02/03/2026, run local, pas de commit) :
  - Samples : data/samples_v5fast_B_ultima_hardneg.parquet
  - Decider : models/xgb_decider_20260203_v5fast_B_ultima_hardneg.json
  - Meta : models/xgb_two_stage_meta_20260203_v5fast_B_ultima_hardneg.json
  - Metriques : hit@1=82.7% (test), AUC=0.983 (calibrated)

## Travail en cours
- **Orientation A terminée** : XgbInferenceEngine est maintenant le chemin canonique pour le mode `insee_then_postcode`
  - Le mode `multi` (parallèle) garde encore le code legacy pour l'instant
  - Le mode `debug_gt` garde aussi le code legacy pour le tracking détaillé

## Problemes / points d'attention
- Couverture retrieval : ~90.6% des requetes CRM generent des samples (perte en amont non comptabilisee dans le hit@1 offline).
- Mesures non directement comparables entre v4 et v5fast : les negatifs v5fast sont beaucoup plus difficiles (addr_jaro P95=1.0).
- Duplication de positifs observee dans samples v5fast (meme SIRET repete par query) : a verifier et dedupliquer si besoin.
- max_same_siren_negatives=0 dans les samples v5fast actuels : risque de confusion entre etablissements d'un meme SIREN.
- Decider non satisfaisant : aucun champion valide pour Stage 2 a ce stade.

## Fichiers modifies dans cette fenetre
- src/xgb_matcher/infer.py : ajout infer_topk(), TopKRow, alignement retrieval ULTIMA, fix calibrator loading
- src/xgb_matcher/blocking.py : TF-IDF name aligne sur train (ngrams 1-2)
- src/xgb_matcher/features.py : semantic gate default=0
- src/xgb_matcher/profile.py : partitions default v5_all
- src/xgb_matcher/semantic.py : cache et batch size parametrables pour la memoire
- scripts/infer_xgb_two_stage.py : stage1_top_n=50, prefilter TF-IDF reactive
- scripts/generate_training_samples_v5fast.py : streaming Parquet + resume + semantic late-compute
- scripts/build_candidate_partitions_v5.py : nettoyage partitions avant ecriture
- AGENTS.md + handover.md : regle de suivi et contextes mis a jour

## Prochaines actions (dans l'ordre)
1. **Stabiliser un Decider champion** (samples propres + same-siren negatives si besoin).
2. **Verifier/dedupliquer les positifs dans samples v5fast** (impact sur decider).
3. **Confirmer la couverture retrieval** (TF-IDF + rescue) avec le pipeline ULTIMA aligne.
4. **Lancer inference top-k sur CRM complet** :
   - python scripts/infer_xgb_two_stage.py --crm-path data/crm_ok_gt.csv --meta-path models/xgb_two_stage_meta_20260203_v5fast_B_ultima_hardneg.json --output-path data/topk_B_ultima_hardneg.csv --pool-mode insee_then_postcode
5. **Construire dataset routing + entrainer risk model** :
   - scripts/build_routing_eval_dataset.py
   - scripts/train_routing_risk_model.py

## Artefacts principaux
| Artefact | Chemin |
|----------|--------|
| Ranker Fast Champion | models/xgbranker_fast_v5fast_B_clean_500neg.json |
| Meta Ranker | models/xgb_two_stage_meta_v5fast_B_clean_500neg.json |
| Decider (candidat) | models/xgb_decider_20260203_v5fast_B_ultima_hardneg.json |
| Meta complet | TBD (bloque tant que le Decider n'est pas valide) |
| Samples | data/samples_v5fast_B_ultima_hardneg.parquet |
| Candidates | data/candidates_v5_all/ (partitionne par INSEE/CP) |

## Contraintes cles a respecter
- Strict insee_then_postcode (pas de fallback departement, meme avec Places).
- Ranker fait le pruning ; decider voit top-50 (+ rescue).
- Etablissements fermes autorises ; SIRET ouvert promu si meme SIREN.
- Strategie bag-of-names a conserver (besoin business).
