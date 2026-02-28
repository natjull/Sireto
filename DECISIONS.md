# Decisions Log

## 2026-01-31 - Orientation cible: retrain + zero skew + commune/CP strict

Decision:
- Reentrainer les modeles (ranker/decider/routing) pour refleter le retrieval reel.
- Le ranker devient le pruning ML principal et doit rester leger/rapide.
- Zero train/serve skew: meme retrieval/pruning/features pour samples, training et inference; aucune injection de GT.
- Aucun fallback departemental: commune/CP strict (si CRM faux, c'est un cas REVIEW/NO_MATCH).

Rationale:
- Les metriques training ne sont pas reproductibles en inference tant que le retrieval/pruning differe.
- L'injection GT dans le training masque des pertes reelles en inference.
- Le besoin metier impose de ne pas depasser la commune/CP.

Consequences:
- Fixer le retrieval commune/CP (pool + pruning non destructif), puis regenerer samples et reentrainer.
- La precision AUTO reste non negociable; le routing sera recalibre sur la distribution d'inference.

## 2026-01-31 - Strategie hard negatives (ranker d'abord)

Decision:
- Le ranker est entraine sans hard negatives generes par un modele.
- Les hard negatives pour le decider et le routing sont generes par le ranker nouvellement entraine (meme retrieval commune/CP).

Rationale:
- Evite de biaiser le ranker par un ancien modele.
- Aligne les hard negatives sur la distribution reelle de Stage 1 en production.

Consequences:
- Generation de samples en deux temps: dataset ranker sans hard negatives, puis dataset decider avec hard negatives issus du nouveau ranker.

## 2026-01-31 - Fix Retrieval "Bag of Names" SOTA

Decision:
- Adoption de la strategie "Bag-of-Names" pour le retrieval (TF-IDF) ET l'entrainement.
- Suppression de la normalisation L2 dans TF-IDF (`norm=None`).
- Activation du "Universal Rescue" (Whitelist Adresse Hash + Numeric Tokens) pour tous les modes (y compris `insee_then_postcode`).

Rationale:
- La normalisation L2 pénalisait les candidats "riches" (filiales avec beaucoup de noms), causant une perte de Recall critique.
- Le mode "insee" strict manquait de robustesse pour les cas "Nom faux / Adresse exacte".
- Cette stratégie permet de "Ringardiser le marché" en trouvant des correspondances structurelles complexes (Siège/Filiale) que les méthodes classiques ratent.

Consequences:
- Modification de `blocking.py` (norm=None) et `infer_xgb_two_stage.py` (Rescue).
- Regeneration des samples avec ces nouveaux parametres pour supprimer le biais train/serve.

## 2026-02-01 - Opération "Ultima" : Double Indexation Retrieval

Decision:
- Implémentation du "Address First Retrieval" : double indexation TF-IDF sur le Nom ET l'Adresse.
- Fusion des résultats au niveau du préfiltre pour maximiser le Recall.

Rationale:
- Atteindre le 100% théorique de Recall en Top-500.
- Assurer que même si le Nom CRM est totalement différent du Nom SIRENE, le candidat est repêché si l'adresse (même approximative) matche.
- Renforcer la robustesse aux changements de dénomination (Siège/Filiale) et aux erreurs de saisie d'adresse.

Consequences:
- Modification de `blocking.py` (Address TF-IDF) et alignement de tous les scripts de génération et d'inférence.
- Nouveau record de Recall Retrieval Relatif à ~97% (hors communes inexistantes).

## 2026-02-04 - Unification retrieval + verrouillage des parametres (Variant B uniquement)

Decision:
- Extraire la logique de retrieval dans un module partage (`src/xgb_matcher/retrieval.py`) consomme par training et inference.
- Verrouiller les parametres retrieval dans une configuration unique (ex: `InferenceProfile`/YAML) avec validation runtime.
- Variant B uniquement (pas de Variant C / siblings) pour limiter le bruit dans le pool.

Rationale:
- Eviter le train/serve skew en supprimant la duplication entre scripts.
- Stabiliser la strategie metier + data science avec des parametres uniques et auditables.

Consequences:
- Refactor des scripts pour deleguer la construction du pool au module partage.
- Ajout de checks de configuration (prefilter_k, stage1_top_n, tfidf ngrams, rescue, pool_mode) avant execution.

## 2026-02-04 - Validation et verrouillage des seuils de cascade retrieval->ranker->decider

Decision:
- Conserver l'architecture en cascade `TF-IDF (k=500/1000) -> Ranker (topN=50) -> Decider (top1)` comme standard SSOT.
- Fixer et versionner les seuils (k et topN) via un sweep offline sur courbes `GT recall@prefilter`, `GT recall@stage1_top_n`, et `hit@1`.
- Interdire tout override implicite des knobs retrieval/stage par la CLI lorsque `--meta-path` est fourni.
- Standardiser la politique "mega-communes" (seuil de bascule CP filtré INSEE = 100 000 lignes) et l'appliquer strictement en train et en serve.
- Encadrer `siren_siblings` avec des caps déterministes (`max_siren_siblings`, `max_names_per_candidate`) et le maintenir OFF par défaut sauf preuve de gain net.

Rationale:
- Le Hit@1 final est borné par le recall des stages amont; l'ajustement des seuils sans mesure des courbes masque les goulots d'étranglement réels.
- Les divergences de seuils (skew) entre le training (souvent top-50) et l'inférence (parfois top-200/500) dégradent la calibration du Decider et du Risk Model.
- Les caps et la politique mega-commune garantissent la stabilité de la latence et de la mémoire sur les gros périmètres (Paris/Lyon/Marseille).

Consequences:
- Ajout d'une "retrieval signature" (hash des knobs + version) dans les métadonnées pour refuser l'exécution en cas de mismatch.
- Régénération des samples et réentraînement des modèles après verrouillage de la signature.
- Mise à jour du `PartitionedCandidateStore` pour intégrer le seuil mega-commune et la cohérence de typage INSEE.

## 2026-02-05 - Contrainte hardware cible

Decision:
- La cible d'execution est un MacBook Pro M4 Pro, 24 GB RAM (stabilite memoire/latence).

Rationale:
- Garantir que toutes les optimisations et caches restent compatibles avec cette contrainte.

## 2026-02-05 - Suppression du rescue post-ranker par adresse

Decision:
- Suppression du "rescue_by_address" post-ranker (top-50) en train et en inference.
- Le Ranker est l'unique source de pruning; aucun ajout de candidats hors top-N.

Rationale:
- Simplification et alignement strict train/serve.
- Le recall@50 du Ranker est juge suffisant pour se passer d'un filet adresse.

Consequences:
- Re-evaluer recall@50 et hit@1 apres alignement complet.

## 2026-02-05 - Optimisation "Turbo" pour la génération de samples

Decision:
- Implementation du mode "Turbo" dans `generate_training_samples_v5fast.py` via un negative sampling basé sur les rangs TF-IDF du retrieval.
- Limitation du calcul des features lourdes (Jaro, address parser, etc.) aux seuls candidats sélectionnés (GT + 50 négatifs) au lieu de l'intégralité du pool (~500).

Rationale:
- Réduire le temps de génération des samples de plusieurs heures à moins de 30 minutes.
- Permettre un cycle d'itération rapide pour la Data Science.

Consequences:
- Accélération massive de la production de datasets sans introduire de skew (le retrieval reste identique).

## 2026-02-08 - P0: Cache TF-IDF persistant + Parallelisation + Instrumentation

Decision:
- Implementation d'un cache TF-IDF persistant sur disque (`data/tfidf_cache/<config_hash>/<partition>.pkl`) invalide par signature `RetrievalConfigV1`.
- Parallelisation de la boucle `loc_key` dans `generate_training_samples_v5fast.py` via `ProcessPoolExecutor` (controlable par `XGB_SAMPLE_WORKERS` ou `--max-workers`).
- Instrumentation p50/p95 par etape (partition_load, tfidf_fit, tfidf_query, feature_compute, semantic_encode) via `PipelineTimer`.

Rationale:
- Le TF-IDF est reconstruit per loc_key per run (bottleneck principal en iteration). Le cache persistant l'elimine pour les runs suivants.
- Les loc_keys sont independants: la parallelisation donne un speedup x4-6 sur M4 Pro 12-core.
- L'instrumentation permet d'identifier les outliers (mega-communes) et de mesurer les gains.

Consequences:
- Aucun changement de comportement ML (identique bit-a-bit si meme config).
- Nouveaux fichiers: `src/xgb_matcher/tfidf_cache.py`, `src/xgb_matcher/timing.py`.
- `build_candidate_pool()` accepte desormais des kwargs optionnels `persistent_cache`, `dense_store`, `timer` (backward-compatible).

## 2026-02-08 - P1: Hybrid Dense+Sparse Retrieval (FAISS + TF-IDF)

Decision:
- Ajout d'un retrieval dense (FAISS ANN sur embeddings MiniLM/siret-bert) en complement du TF-IDF existant.
- Mode hybride: `pool = union(top_k_sparse, top_k_dense, whitelist_rescue)` avec budget constant.
- Le TF-IDF n'est pas retire; il est complete par le dense. La decision de retirer le sparse sera prise sur benchmark ablation.

Rationale:
- Le TF-IDF fait sauter 2-4% de GT (recall ceiling) sur les cas semantiques (acronymes, renommages, filiales).
- Le dense retrieval capture la similarite semantique que le lexical rate par construction.
- L'approche hybride conserve la robustesse lexicale tout en recuperant les cas perdus.

Consequences:
- Nouveaux fichiers: `src/xgb_matcher/dense_retrieval.py`, `scripts/precompute_embeddings.py`, `scripts/benchmark_retrieval.py`.
- `RetrievalConfigV1` enrichi de `dense_retrieval_enabled` et `dense_top_k`.
- Pre-requis: `pip install faiss-cpu sentence-transformers` (optionnel, graceful degradation si absent).

## 2026-02-05 - Evolution de la policy Méga-Communes vers "Full INSEE"

Decision:
- Remplacement de la policy `cp_filter_insee` par `full_insee` pour les communes dépassant le seuil méga (100k).
- Chargement de l'intégralité de la commune méga pour l'indexation TF-IDF locale au lieu de filtrer par code postal CRM.

Rationale:
- Le filtrage par CP CRM sur les grosses communes (Nice, Nantes, etc.) causait une perte de Coverage majeure (~60% des échecs NOT_IN_PARTITION) à cause des imprécisions de saisie CP.
- Maximiser le Recall théorique en donnant au TF-IDF la visibilité sur toute la ville.

Consequences:
- Amélioration immédiate du Retrieval Coverage (passage de ~90% à ~93%+).

## 2026-02-08 - Adoption du retrieval hybride sparse + dense

Decision:
- Intégrer la branche d'audit retrieval hybride dans `main` pour unifier sparse TF-IDF + dense FAISS.
- Conserver une activation opt-in du dense (`dense_retrieval_enabled=false` par défaut) pour ne pas dégrader la stabilité en production.
- Conserver le cache TF-IDF persistant et l'instrumentation timing comme standard de génération des samples.

Rationale:
- Le sparse seul couvre mal les écarts sémantiques de dénomination, alors que le dense améliore le recall sur ces cas.
- Le cache persistant réduit fortement les temps d'itération sans modifier le comportement ML.
- Le mode opt-in permet un déploiement progressif avec benchmark contrôlé.

Consequences:
- Dépendances retrieval enrichies (`faiss-cpu`, `sentence-transformers`) et pré-calcul d'embeddings requis pour activer le dense.
- Le script de génération des samples peut désormais être câblé pour exploiter directement le store d'embeddings pré-calculés.
- Références commits GitHub: `9ab297e` (feature retrieval hybride), `35fc3a3` (correctifs audit dense-only/sparse flag).

## 2026-02-09 - Bascule partitions V7 + lookup manifeste O(1)

Decision:
- Adopter `data/candidates_v7_all` comme défaut pour les scripts actifs de génération, benchmark et inférence.
- Utiliser `manifest/insee_counts.parquet` quand disponible pour obtenir les tailles INSEE en O(1), avec fallback automatique vers `count_rows` si absent.

Rationale:
- Réduire les coûts I/O et la latence sur les méga-communes.
- Aligner les workflows de train/serve/benchmark sur l’artefact de partition le plus complet.

Consequences:
- `PartitionedCandidateStore` est désormais compatible V7 optimisé tout en restant rétro-compatible V6.
- Les defaults de `generate_training_samples_v5fast.py`, `benchmark_retrieval.py`, `precompute_embeddings.py`, `infer_xgb_two_stage.py` et `src/xgb_matcher/profile.py` ciblent V7.
- Référence commit GitHub: `a309a7c`.

## 2026-02-09 - Orchestration embeddings V7 (Option A)

Decision:
- Étendre `scripts/precompute_embeddings.py` avec une stratégie de priorisation pilotée par manifeste V7 avant exécution longue.
- Ajouter un mode `--dry-run` pour valider la sélection de partitions sans calcul d'embeddings.

Rationale:
- Réduire le risque opérationnel des runs longs (~42M vecteurs) en validant d'abord la stratégie de sélection.
- Prioriser les méga-communes permet d'obtenir un gain produit plus tôt.

Consequences:
- Nouvelles options CLI: `--sort-by-size`, `--mega-first`, `--mega-threshold`, `--min-rows`, `--max-partitions`, `--log-every`, `--dry-run`.
- Ajout de tests d'ordre/filtrage pour la logique de sélection de partitions.
- Référence commit GitHub: `66b5b87`.

## 2026-02-28 - V8 feature set et hard negatives cibles erreurs metier

Decision:
- Ajouter 7 features d'interaction pour mieux separer colocataires, homonymes geographiques et multi-sites SIREN.
- Elargir la generation de hard negatives (same_addr, sibling SIREN, homonymes geo) pour entrainer Stage 2 sur les erreurs reelles.
- Passer le decider sur un regime d'apprentissage plus fin (`learning_rate=0.05`, `max_depth=7`, `400 rounds`).

Rationale:
- Les erreurs Stage 2 etaient concentrees sur des cas adresse-forte / nom-faible et sur des collisions intra-SIREN.
- Les features de base ne capturaient pas explicitement ces interactions.

Consequences:
- Les scripts de generation et d'entrainement produisent un dataset plus discriminant pour le decider.
- La baseline attendue evolue de V7 vers V8 avant rebase Route B.
- Référence commit GitHub: `35fb441`.

## 2026-02-28 - Adoption de Route B (SIREN-first global matching)

Decision:
- Adopter une architecture hierarchique SIREN-first:
  1) matching global au niveau SIREN (index TF-IDF global)
  2) retrieval des SIRET par geographie de ce SIREN
  3) ranking/scoring Stage 1 et Stage 2 inchanges structurellement.
- Conserver la compatibilite descendante: fallback V7 si index SIREN absent.

Rationale:
- Le retrieval SIRET-only pouvait perdre le GT sur `PRUNED_BY_TFIDF` et `NOT_IN_PARTITION`.
- Le niveau "entite legale" (SIREN) est un sous-probleme distinct du niveau "site" (SIRET).

Consequences:
- Nouveaux artefacts et modules: index global SIREN, mapping `siren -> geo`, retrieval SIREN.
- Inference profile et retrieval config enrichis de knobs Route B.
- Référence commit GitHub: `3e090b7`.

## 2026-02-28 - Corrections de blocage et branchement train/serve Route B

Decision:
- Corriger les blockers operationnels Route B:
  - connexion DuckDB memory mode,
  - champ CRM nom utilise en requete SIREN,
  - logique filtre etablissements fermes,
  - exposition CLI `--siren-index`.
- Brancher explicitement Route B dans le retrieval partage utilise par la generation de samples (y compris multiprocess).

Rationale:
- Sans ces correctifs, Route B restait partielle (code present mais non executable/non active en train).

Consequences:
- Route B devient executable de bout en bout pour la regeneration des samples et le retrain.
- Les workers de generation chargent leurs index SIREN localement (safety multiprocess).
- Références commits GitHub: `c356923`, `1305012`.

## 2026-02-28 - Recalibration obligatoire du Stage 3 apres rebase Route B

Decision:
- Rendre explicite la recalibration obligatoire du risk model (Stage 3) apres retrain Stage 1/2 sur pool Route B.
- Ne pas reutiliser tel quel le seuil historique `0.835` sans reevaluation.

Rationale:
- La distribution des scores top1/top2 change avec la recomposition du pool candidats.
- Le niveau de precision AUTO cible (99.8%+) depend du calibrage post-shift.

Consequences:
- Executer une nouvelle courbe coverage/precision AUTO sur dataset d'evaluation regenere.
- Versionner le nouveau seuil et la meta risk associee avant promotion.
