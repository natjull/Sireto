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
- Passage des partitions de reference a v6 (defaults pipeline + docs). (commit TBD)
- Validation partitions v6 OK avec schema force en string dans le validateur. (commit TBD)

## Travail en cours
- **Orientation A terminée** : XgbInferenceEngine est maintenant le chemin canonique pour le mode `insee_then_postcode`
  - Le mode `multi` (parallèle) garde encore le code legacy pour l'instant
  - Le mode `debug_gt` garde aussi le code legacy pour le tracking détaillé
- Rebuild complet a relancer sur v6 (samples -> ranker -> decider). (commit TBD)

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
- src/xgb_matcher/profile.py : partitions default v6_all
- src/xgb_matcher/semantic.py : cache et batch size parametrables pour la memoire
- scripts/infer_xgb_two_stage.py : stage1_top_n=50, prefilter TF-IDF reactive
- scripts/generate_training_samples_v5fast.py : streaming Parquet + resume + semantic late-compute
- scripts/build_candidate_partitions_v5.py : sortie par defaut v6_all + normalisation codes string-safe
- scripts/validate_partitions_v5.py : validation v6 + schema force + Corsica conditionnel
- scripts/infer_xgb_two_stage.py : partitions default v6_all
- scripts/generate_training_samples_v5fast.py : partitions default v6_all
- scripts/debug_tfidf_norm_effect.py : partitions default v6_all
- AGENTS.md + handover.md : regle de suivi et contextes mis a jour

## Plan d'harmonisation Train/Serve (ULTIMA B)

1. **Geler la cible “Variant B / SSOT” en une config canonique**
   - Specifier tous les knobs (pool_mode, prefilter_k, char_top_k, min_candidates, stage1_top_n, drop_unnamed, include_closed, siren_siblings, semantic gate).
   - Definir la structure de `RetrievalConfigV1` et `RetrievalSignatureV1` (hash).

2. **Corriger la construction des partitions (Source de Vérité)**
   - Script: `scripts/build_candidate_partitions_v5.py`.
   - Forcer schema stable (types) + Deduplication *avant ecriture* par `siret`.
   - Normalisation deterministe de `insee` et `postcode` (string vs int32 handling).

3. **Ajouter un validateur de partitions (Stop-the-line)**
   - Nouveau script: `scripts/validate_partitions_v5.py`.
   - Verifier schema, absence de doublons SIRET et fonctionnement des filtres PyArrow.

4. **Unifier le “Loading Policy” INSEE/CP (Train = Serve)**
   - Script: `src/xgb_matcher/partitioned_store.py`.
   - Ajouter gestion "mega-commune" canonique (seuil 200k rows) avec fallback CP filtre INSEE.
   - Centraliser dedupe SIRET et `REQUIRED_COLUMNS`.

5. **Mettre `build_tfidf_index` au standard SSOT (norm=None)**
   - Script: `src/xgb_matcher/blocking.py`.
   - Fixer TF-IDF name: word (1,2), lowercase=False, `norm=None` (suppression penalite longueur).

6. **Refactor `src/xgb_matcher/retrieval.py` pour Variant B**
   - Supprimer dept fallback/semantic rescue.
   - Implementer double TF-IDF (name+addr) + char fallback + rescue universel + padding deterministe.
   - Exposer diagnostics riches (lost_gt reasons).

7. **Raccorder `XgbInferenceEngine` au retrieval SSOT**
   - Script: `src/xgb_matcher/infer.py`.
   - Supprimer `_build_candidate_pool` interne au profit du module unifie.

8. **Eliminer les overrides implicites de knobs en inference**
   - Script: `scripts/infer_xgb_two_stage.py`.
   - Regle: si `--meta-path`, les knobs viennent de la meta (strict). Aligner defaults CLI sur Variant B.

9. **Neutraliser le mode `multi` (Dept Fallback)**
   - Soit deprecier, soit re-implementer comme pure parallelisation sans elargissement de pool hors SSOT.

10. **Aligner `generate_training_samples_v5fast.py` sur le retrieval SSOT**
    - Utiliser le module unifie + activer reellement `char_top_k=200` en training.
    - Ecrire la signature/config dans le sidecar JSON des samples.

11. **Propager et valider la signature dans les metas modeles**
    - Scripts: `scripts/train_xgb_ranker.py` / `scripts/train_xgb_decider.py`.
    - Injecter config+signature dans les metas; verifier le hash au chargement en inference.

12. **Suppression du rescue post-ranker**
    - Aucun ajout de candidats hors top-N (train et inference alignes).

13. **Aligner les metriques offline sur le post-processing metier**
    - Appliquer `open > closed` promotion dans les evaluations offline pour coherence hit@1.

14. **Nettoyage des variants et legacy**
    - Verrouiller Variant B par defaut, supprimer les knobs morts (`rescue_semantic_k`).

15. **Rebuild complet du socle**
    - Partitions -> Samples Ranker -> Ranker -> Samples Decider -> Decider.

16. **Verification E2E sur `data/crm_ok_gt.csv`**
    - Rapports: Coverage, Hit@1, Lost GT reasons, Pool statistics.

17. **Mise a jour doc (Handover/SSOT)**
    - Tracer chaque bloc via commit IDs et specifier les decisions finales.

## Artefacts principaux
| Artefact | Chemin |
|----------|--------|
| Ranker Fast Champion | models/xgbranker_fast_v5fast_B_clean_500neg.json |
| Meta Ranker | models/xgb_two_stage_meta_v5fast_B_clean_500neg.json |
| Decider (candidat) | models/xgb_decider_20260203_v5fast_B_ultima_hardneg.json |
| Meta complet | TBD (bloque tant que le Decider n'est pas valide) |
| Samples | data/samples_v5fast_B_ultima_hardneg.parquet |
| Candidates | data/candidates_v6_all/ (partitionne par INSEE/CP) |

## Contraintes cles a respecter
- Strict insee_then_postcode (pas de fallback departement, meme avec Places).
- Ranker fait le pruning ; decider voit top-50 (no rescue post-ranker).
- Etablissements fermes autorises ; SIRET ouvert promu si meme SIREN.
- Strategie bag-of-names a conserver (besoin business).
