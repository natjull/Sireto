# SIRETO Handover - 1 Mars 2026

## Etat des lieux
Le pipeline cible n'est plus "Route B full SIREN-first" comme chemin principal.
La trajectoire retenue est desormais **V8b = V7 + SIREN expansion post-prefilter**:
- prefilter V7 SIRET local (INSEE/CP) conserve comme seed robuste
- expansion SIREN ensuite (local + cross-partition) via `siren_to_geo.parquet`
- Stage 1/2/3 inchanges structurellement, mais a reentrainer/recalibrer

Route B (ranking SIREN global en phase 1) reste dans le code pour A/B tests, mais n'est plus la strategie par defaut.

## Actions terminees (fenetre recente)
- **Spikes architecture neurale (cross-encoder + dual-encoder)**: benchmark reproductible sur un holdout SIREN-disjoint de 400 requetes. Le cross-encoder court ne remplace pas XGBoost (51,75% vs 85,25% Hit@1 sur les memes scenes). Le dual-encoder structure atteint 74,50% Recall@1 et 96,00% Recall@50; l'union TF-IDF top-50 + dense top-50 atteint 99,25% Recall@50 (8 des 11 misses TF-IDF recuperes). Le modele semantic exporte declare a tort `BertTokenizer`; le chargement actuel via SentenceTransformer produit excessivement des tokens `<unk>`, donc les anciens benchmarks semantiques doivent etre revalides apres correction. *(commit GitHub: `7640772`)*
- **V8 features + hard negatives + hyperparams decider**: ajout de 7 features d'interaction, extension des hard negatives colocataires/homonymes/siblings, tuning decider (`lr=0.05`, `max_depth=7`, `400 rounds`). *(commit GitHub: `35fb441`)*
- **Route B (SIREN-first) implementee**: nouvel index global SIREN, nouveau module de retrieval SIREN, branchement conditionnel dans l'inference profile/engine. *(commit GitHub: `3e090b7`)*
- **Correctifs bloquants Route B**: fix DuckDB `:memory:`, fix champ CRM nom, fix filtre closed/open, ajout CLI `--siren-index` dans le generateur de samples. *(commit GitHub: `c356923`)*
- **Branchement Route B dans le retrieval partage (training)**: `build_candidate_pool()` supporte Route B via indices SIREN, propagation sequentielle + multiprocess dans `generate_training_samples_v5fast.py`. *(commit GitHub: `1305012`)*
- **Implementation V8b SIREN expansion (V7 + local + cross-partition)**: ajout Step 5 d'expansion apres prefilter, feature flag, cap pool dedie et telemetrie d'expansion. *(commit GitHub: `9c0e806`)*
- **Correctifs critiques V8b**: exclusion explicite Route B quand expansion activee, filtres metier expansion, recalc GT coverage/loss reason post-expansion. *(commit GitHub: `f1fbbb8`)*
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
- `src/xgb_matcher/retrieval.py` *(commits GitHub: `1305012`, `9c0e806`, `f1fbbb8`)*
- `src/xgb_matcher/retrieval_config.py` *(commits GitHub: `3e090b7`, `9c0e806`)*
- `src/xgb_matcher/profile.py` *(commit GitHub: `3e090b7`)*

## Travail en cours
- **Regeneration samples V8b (expansion)**: regenerer ranker/decider avec `--enable-siren-expansion`.
- **Retrain Stage 1 + Stage 2**: entrainer sur la nouvelle distribution de pool (prefilter + expansion).
- **Recalibration Stage 3**: reestimer le seuil risk model (obligatoire apres shift de distribution).
- **A/B de verification**: garder Route B disponible uniquement pour comparaison offline.

## Points d'attention
- **Clarification nomenclature**: "V8" dans les echanges = V8b (V7 + SIREN expansion), pas Route B full.
- **Validation metrique manquante**: pas encore de benchmark consolide post-V8b.
- **Latence expansion**: mesurer sur commune dense (impact des loads cross-partition via cache INSEE).
- **Tokenizer semantic casse dans l'export**: `models/semantic/siret-bert-deploy/tokenizer_config.json` declare `BertTokenizer` alors que `tokenizer.json` contient le tokenizer SentencePiece multilingue. Ne pas interpreter les anciens scores semantiques avant correction du loader/export et rebenchmark.
- **Decision architecture issue des spikes**: pas de remplacement immediat du coeur XGBoost par le cross-encoder teste. Priorite a un retrieval hybride sparse+dense appris avec embeddings candidats pre-calcules; conserver TF-IDF car le dense seul reste inferieur en Recall@50 (96,00% vs 97,25%).
- **Governance docs**: garder `handover.md` comme journal de commits (regle AGENTS).

## Artefacts cibles (V8b)
| Artefact | Chemin |
|----------|--------|
| Partitions candidates | `data/candidates_v7_all/` |
| Mapping geo SIREN (obligatoire V8b) | `data/siren_index/siren_to_geo.parquet` |
| Index SIREN global (optionnel A/B Route B) | `data/siren_index/word_matrix.npz` + `char_matrix.npz` |
| Samples decider V8b | `data/samples_v8b_decider.parquet` |
| Ranker V8b | `models/v8b_ranker*.json` |
| Decider V8b | `models/v8b_decider*.json` |
| Meta two-stage | `models/xgb_two_stage_meta_*.json` |

## Prochaines etapes
1. Corriger le tokenizer de l'export semantic et ajouter un test anti-`<unk>` avant tout nouveau benchmark dense.
2. Pre-calculer un index dense avec le dual-encoder structure, puis mesurer l'union sparse+dense a budget fixe sur toutes les tailles de communes.
3. Construire (au minimum) `siren_to_geo.parquet` (`--geo-only` possible).
4. Regenerer les samples `ranker` et `decider` avec `--enable-siren-expansion` et le retrieval hybride corrige.
5. Reentrainer Stage 1 puis Stage 2, refaire l'evaluation complete (coverage pool, Hit@1, latence, segments PRUNED/NIP), puis recalibrer Stage 3.

---
*Regle projet: chaque modification de code/metier doit citer son commit GitHub dans ce document.*
