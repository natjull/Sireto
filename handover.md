# SIRETO Handover - 28 Fevrier 2026

## Etat des lieux
Le pipeline V7 reste la baseline stable. La branche V8/Route B (SIREN-first) est maintenant integree au code, avec branchement train+serve effectif. Le prochain jalon est la validation metrique complete (coverage, Hit@1, latence) puis la recalibration du Stage 3.

## Actions terminees (fenetre recente)
- **V8 features + hard negatives + hyperparams decider**: ajout de 7 features d'interaction, extension des hard negatives colocataires/homonymes/siblings, tuning decider (`lr=0.05`, `max_depth=7`, `400 rounds`). *(commit GitHub: `35fb441`)*
- **Route B (SIREN-first) implementee**: nouvel index global SIREN, nouveau module de retrieval SIREN, branchement conditionnel dans l'inference profile/engine. *(commit GitHub: `3e090b7`)*
- **Correctifs bloquants Route B**: fix DuckDB `:memory:`, fix champ CRM nom, fix filtre closed/open, ajout CLI `--siren-index` dans le generateur de samples. *(commit GitHub: `c356923`)*
- **Branchement Route B dans le retrieval partage (training)**: `build_candidate_pool()` supporte Route B via indices SIREN, propagation sequentielle + multiprocess dans `generate_training_samples_v5fast.py`. *(commit GitHub: `1305012`)*
- **Fix expansion SIREN en mode geo-only**: chargement des index dissocie (global vs geo) dans le generateur de samples pour eviter la desactivation silencieuse de l'expansion quand seul `siren_to_geo.parquet` est present. *(commit GitHub: `c961371`)*

## Historique structurant (deja en place)
- **Retrieval hybride sparse+dense + cache TF-IDF persistant + timing**: integration du socle P0/P1. *(commit GitHub: `9ab297e`)*
- **Ablation dense-only corrigee + flag sparse explicite**: alignement des modes retrieval et signature de config. *(commit GitHub: `35fc3a3`)*
- **Defaults partitions V7 + manifest INSEE O(1)**: bascule des chemins/scripts vers `data/candidates_v7_all`. *(commit GitHub: `a309a7c`)*
- **Priorisation mega-communes embeddings**: orchestration dense amelioree pour runs longs. *(commit GitHub: `66b5b87`)*

## Fichiers modifies recemment
- `src/xgb_matcher/features.py` *(commit GitHub: `35fb441`)*
- `scripts/generate_training_samples_v5fast.py` *(commits GitHub: `35fb441`, `c356923`, `1305012`, `c961371`)*
- `scripts/train_xgb_decider.py` *(commit GitHub: `35fb441`)*
- `scripts/build_siren_global_index.py` *(commits GitHub: `3e090b7`, `c356923`)*
- `src/xgb_matcher/siren_retrieval.py` *(commit GitHub: `3e090b7`)*
- `src/xgb_matcher/infer.py` *(commits GitHub: `3e090b7`, `c356923`)*
- `src/xgb_matcher/retrieval.py` *(commit GitHub: `1305012`)*
- `src/xgb_matcher/retrieval_config.py` *(commit GitHub: `3e090b7`)*
- `src/xgb_matcher/profile.py` *(commit GitHub: `3e090b7`)*

## Travail en cours
- **Build index SIREN global**: generer les artefacts `data/siren_index/` sur jeu complet.
- **Regeneration samples V8b (Route B)**: lancer dataset decider/ranker base sur nouveau pool SIREN-first.
- **Retrain Stage 1 + Stage 2**: entrainer nouveaux modeles sur distribution Route B.
- **Recalibration Stage 3**: seuil risk model a re-estimer sur nouvelles distributions de score (obligatoire).

## Points d'attention
- **Validation metrique manquante**: pas encore de benchmark consolide post-Route B.
- **Latence Route B**: a mesurer en reel sur index 7M SIREN (objectif operationnel <200ms pour la phase SIREN query).
- **Governance docs**: garder `handover.md` comme journal de commits (regle AGENTS).

## Artefacts cibles (V8b)
| Artefact | Chemin |
|----------|--------|
| Partitions candidates | `data/candidates_v7_all/` |
| Index SIREN global | `data/siren_index/` |
| Samples decider Route B | `data/samples_v8b_decider.parquet` |
| Ranker V8b | `models/v8b_ranker*.json` |
| Decider V8b | `models/v8b_decider*.json` |
| Meta two-stage | `models/xgb_two_stage_meta_*.json` |

## Prochaines etapes
1. Construire `data/siren_index/` avec `scripts/build_siren_global_index.py`.
2. Regenerer les samples via `scripts/generate_training_samples_v5fast.py --mode decider --siren-index data/siren_index/`.
3. Reentrainer ranker puis decider sur les nouveaux samples.
4. Refaire l'evaluation complete (coverage, Hit@1, segmentation erreurs).
5. Recalibrer Stage 3 (`routing_risk_model.pkl`) et fixer le nouveau seuil AUTO/REVIEW.

---
*Regle projet: chaque modification de code/metier doit citer son commit GitHub dans ce document.*
