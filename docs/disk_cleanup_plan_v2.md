# Plan de ménage SSD SIRETO v2 — proposition, aucune suppression

Audit du 2026-10-21T20:54:20.946151+00:00. Base Git `531ac54b73d4db21ada33b12937d3be1a29399c5`. Politique du brief `sireto-disk-2.md`.
**B signifie archivable sur accord de Nathan, pas une suppression déjà faite.** Le script livré n’a jamais été exécuté, même avec `--dry-run`.
Abréviations : `$H=/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2`, `$R=/Volumes/CATNAT_DATA/SIRETO_RECALL100`, `$V9=/Volumes/CATNAT_DATA/SIRETO_V9`, `$PPLX=/Volumes/CATNAT_DATA/SIRETO_PPLX`.

## Totaux sans double compte

| Classe | Gio alloués | Décision |
|---|---:|---|
| A — INDISPENSABLE | 236.473957 | Ne jamais supprimer |
| B — ARCHIVABLE | 368.784691 | Suppression possible uniquement sur accord explicite |
| C — DOUTE | 53.307198 | Ne pas toucher |
| **Total** | **658.565845** | **911 unités disjointes, 45 racines attribuées** |

Les tailles `du -sk` du v1 sont réutilisées. Les unités « direct_files » ne couvrent que les fichiers directement dans le dossier, pas ses sous-dossiers comptés séparément. Les autres unités sont récursives.
Cinq mesures isolées ont recompté des liens physiques déjà portés par une autre unité : **2,923481 Gio non ajoutés au total**, retranchés conservativement de la part B. Les signatures des arbres conservent cependant leurs tailles isolées réelles pour les contrôles du futur script. Liens physiques, clones et snapshots APFS peuvent encore réduire le gain physique réel : **368,785 Gio ne sont pas une garantie de place libérée**.

## A — ce qui reste indispensable

| Motif prioritaire, exclusif pour l’addition | Gio |
|---|---:|
| Production : checkpoint BASE, runtime, index national, entrées exactes | 100.575169 |
| Fichiers V142 non déjà comptés dans A1/A3/A4 | 36.347733 |
| Banc/résultats étape2, serving, revues et autorités directes des caches | 16.516708 |
| Travail tiers, sources brutes et vérités historiques fermées | 83.034348 |

Priorité d’attribution en cas de chevauchement : A4 puis A3 puis A1 puis A2. Les parents minimaux ne protègent que le chemin du dossier, jamais tous ses descendants. Le JSON distingue racines A intégrales, fichiers A exacts et parents minimaux.

### Production : accès effectifs, pas mentions documentaires

- `scripts/sireto_serve_v1.py:33–36,191,227–305,339–355,491–497` et `docs/serve_v1_manifest.json` : modèles `serve_v1/models`, matrices/références dans `step2_*`, textes/binding/shards BASE de `step2_gemma_finetune_10h_20261019/full/base_encode`, cache partagé du serving. Le checkpoint est chargé **localement depuis `$H/base`**, pas depuis les hubs Hugging Face/PPLX proposés en B.
- `scripts/step2_option_d.py:175–182` ouvre l’index national `$R/indexes/hierarchical_retrieval_v3/4b332899d8698354` (**98,090 Gio**) et **tous** les tableaux `plan.layout` via `np.load`. Le JSON en donne chaque fichier exact, sans garder toute la famille d’audits.
- `scripts/step2_xgb_ranker_v1.py:100–117` hydrate réellement stock SIRENE, unités légales et géolocalisation. Entrées CRM/POP du manifeste gardées ; memberships test/validation conservés pour les exclusions du serving.
- BAN (`src/xgb_matcher/retrieval_query_ban_v327.py`) et NAF (`step2_colocated_signals.py`) sont sous Oùbusiness : dépendances déclarées depuis le code, **hors audit et hors suppression**, comme la géolocalisation Oùbusiness. Aucune opération sur CatNat, catnat-lima, hydraulic ou PID 8288.
- Les liens symboliques du runtime et des bancs ont été vérifiés pour ne pas archiver une cible réelle nécessaire. Les autorités **directes** des métadonnées étape2 sont gardées ; leurs anciens manifestes de provenance ne rendent pas automatiquement tous leurs ancêtres indispensables.

### Gel V142 : taille des fichiers exacts, indépendamment des recouvrements

**138.303017 Gio alloués sur SSD**, plus **0.000984 Gio dans le dépôt** ; 3 681 chemins listés, 3 680 présents, 1 déjà absent. Cette vue recouvre notamment les 98,090 Gio de l’index : **ne pas l’ajouter au total A**.
Les 528 fichiers gelés sous `$H/v75` représentent **25,336 Gio** : ils restent A, sans sanctuariser tout `v75`. Les autres fichiers gelés dans `v14`, `v112`, `v129`, `v141`, etc. restent également A au chemin exact.
`$H/v6/deadline_recovery_20261008_a/fit/best_adapter.pt` était déjà absent au v1 et l’est toujours. Ni restauration ni suppression ; chemin et parents réservés. Les SHA du gel sont conservés, **pas intégralement recalculés**. Les tailles des 3 680 présents sont revérifiées.

### A4 : sources brutes et travail tiers, chiffrage séparé

| Ensemble protégé | Gio alloués du v1 | Pourquoi |
|---|---:|---|
| `$R/datasets/synthetic_gt_corpus` | 19.981819 | Corpus tiers non commité, SQLite WAL/état/collecte : pas un cache jetable |
| `$R/sources/official_sync` | 26.796211 | Snapshots bruts officiels ; accès futur et vintage non garantis |
| `$R/sources/official_evidence` | 30.101860 | Sources/preuves officielles brutes et autorités |
| `$R/sources/sirene` | 4.425434 | Stocks SIRENE, notamment les fichiers réellement lus |
| `$R/sources/sirene_history_20260801` | 1.717613 | Trois stocks historiques officiels, pas déductibles du stock courant |
| `$V9/benchmarks/closed` | 0.011154 | Vérités/splits humains fermés : pas des intermédiaires de calcul |

Les autres entrées exactes tiers (dont `fold_assignments.parquet`) sont listées dans `exact_A_files`. Les fichiers étrangers du dépôt ne sont ni intégrés au commit ni modifiés. Les motifs A4 ne découlent pas d’une simple citation ancienne d’une expérience.

## B — top 40 archivable, net des exceptions

Chaque ligne renvoie à une publication **commitée** de résultats, comptages effectivement produits ou arrêt réel. Un STOP ou run interrompu n’est pas transformé en réussite. Pour les caches, la publication est celle de l’expérience/pipeline propriétaire, pas un résultat scientifique propre au cache.
Coûts ci-dessous : **ordres de grandeur non mesurés**, dépendant des ancêtres B à refaire ; ni délai garanti ni promesse de replay bit-identique. Les estimations détaillées et publications sont présentes pour **chaque** unité dans le JSON.

| # | Unité B (hors exceptions) | B Gio | Publication commitée | Coût estimé non mesuré |
|---:|---|---:|---|---|
| 1 | `$H/v174/train3000_frozen_v142_20261017/extension_20261017` | 45.233788 | `docs/train3000_covered_result.md:1` (blob `e44d80e1c4`) | 2–14 j CPU/Metal |
| 2 | `$R/indices/sirene_temporal_v1/506bb98cfdcccfbe` | 14.553886 | `docs/sirene_temporal_retrieval_results.md:1` (blob `21bf020373`) | 4–120 h CPU |
| 3 | `$R/audits/retrieval_v129_national_variants/projection_20261003_a` | 14.243629 | `handover.md:10181` (blob `da326fb0db`) | 4–120 h CPU |
| 4 | `$R/experiments/retrieval_all_human_v11` — fichiers directs seulement | 13.327003 | `handover.md:433` (blob `da326fb0db`) | 2–120 h CPU |
| 5 | `$R/audits/retrieval_v177_packed_addresses/compile_20261004_a` | 10.818508 | `handover.md:12544` (blob `da326fb0db`) | 4–120 h CPU |
| 6 | `$R/audits/retrieval_v206_population_initial/run_20261004_a` | 10.463600 | `handover.md:13952` (blob `da326fb0db`) | 2–120 h CPU |
| 7 | `$R/audits/retrieval_v134_packed_names/compile_20261003_a` | 9.448620 | `handover.md:10495` (blob `da326fb0db`) | 4–120 h CPU |
| 8 | `$R/audits/retrieval_v34_fast_resume/full_job` | 9.080589 | `docs/retrieval_v34_full_crm_evaluation_results.md:1` (blob `aff73b6cbe`) | 2–120 h CPU |
| 9 | `$R/models/huggingface/hub` | 8.745380 | `reports/v9/v9_go_pivot_stop.md:3` (blob `4cad983ce7`) | Retéléchargement/reconstruction : minutes–heures, débit inconnu |
| 10 | `$R/audits/retrieval_v131_name_access` | 7.908154 | `handover.md:10495` (blob `da326fb0db`) | 4–120 h CPU |
| 11 | `$R/audits/retrieval_v130_variant_dictionary` | 7.590630 | `handover.md:10329` (blob `da326fb0db`) | 4–120 h CPU |
| 12 | `$R/audits/retrieval_v168_address_access` | 7.429062 | `handover.md:12293` (blob `da326fb0db`) | 4–120 h CPU |
| 13 | `$R/stores/candidates_v7_current_crmgt_full_20260817` | 7.367760 | `reports/v9/benchmark_v4_current_snapshot_results.md:1` (blob `36704f9900`) | 2–120 h CPU |
| 14 | `$H/v14/representative_20261008_b/zones` | 6.588688 | `handover.md:16529` (blob `da326fb0db`) | 6–168 h CPU/Metal |
| 15 | `$R/audits/retrieval_fold_fused_national_oof_v14` | 6.350914 | `docs/retrieval_all_human_v14_national_oof_results.md:1` (blob `fda46f6aa4`) | 2–120 h CPU |
| 16 | `$R/experiments/retrieval_dossier_v3_full_human` | 6.294189 | `docs/retrieval_v34_full_crm_evaluation_results.md:1` (blob `aff73b6cbe`) | 2–120 h CPU |
| 17 | `$R/audits/retrieval_v154_address_store` | 6.226597 | `handover.md:11605` (blob `da326fb0db`) | 2–120 h CPU |
| 18 | `$R/audits/retrieval_v160_reverse_resume` | 6.132484 | `handover.md:11682` (blob `da326fb0db`) | 2–120 h CPU |
| 19 | `$H/v18` | 6.107746 | `handover.md:16908` (blob `da326fb0db`) | 6–168 h CPU/Metal |
| 20 | `$H/v129/osm_local_20261013_night_e` | 5.241966 | `handover.md:19258` (blob `da326fb0db`) | 6–168 h CPU/Metal |
| 21 | `$R/audits/retrieval_v98_national_postings` | 5.039608 | `handover.md:9224` (blob `da326fb0db`) | 4–120 h CPU |
| 22 | `$R/indexes/retrieval_v12_mmap_sidecar` | 4.844406 | `docs/retrieval_all_human_v12_sharded_benchmark_results.md:1` (blob `024d82a9ec`) | 4–120 h CPU |
| 23 | `$V9/dense` | 4.791618 | `reports/v9/v9_go_pivot_stop.md:3` (blob `4cad983ce7`) | 2–120 h CPU |
| 24 | `$H/v129/osm_local_20261013_night` | 4.768784 | `handover.md:19098` (blob `da326fb0db`) | 6–168 h CPU/Metal |
| 25 | `$R/audits/retrieval_v199_population_initial` | 4.587955 | `handover.md:13617` (blob `da326fb0db`) | 2–120 h CPU |
| 26 | `$R/audits/retrieval_v153_address_resume` | 4.436035 | `handover.md:11525` (blob `da326fb0db`) | 2–120 h CPU |
| 27 | `$H/v36` | 4.184307 | `handover.md:17622` (blob `da326fb0db`) | 6–168 h CPU/Metal |
| 28 | `$H/v37` | 4.168640 | `handover.md:17729` (blob `da326fb0db`) | 6–168 h CPU/Metal |
| 29 | `$H/v33` | 4.168301 | `handover.md:17548` (blob `da326fb0db`) | 6–168 h CPU/Metal |
| 30 | `$R/audits/retrieval_v82_scoped_succession_names` | 3.988834 | `handover.md:8907` (blob `da326fb0db`) | 2–120 h CPU |
| 31 | `$R/audits/retrieval_v197_population_initial` | 3.644852 | `handover.md:13474` (blob `da326fb0db`) | 2–120 h CPU |
| 32 | `$R/audits/retrieval_v95_national_names` | 3.456333 | `handover.md:9207` (blob `da326fb0db`) | 2–120 h CPU |
| 33 | `$H/v173` | 3.447716 | `docs/retrieval_site_target_full_result_20261016.md:1` (blob `c472482f75`) | 6–168 h CPU/Metal |
| 34 | `$R/audits/retrieval_v93_national_names` | 3.383091 | `handover.md:9101` (blob `da326fb0db`) | 2–120 h CPU |
| 35 | `$R/audits/retrieval_v202_population_initial` | 3.132378 | `handover.md:13737` (blob `da326fb0db`) | 2–120 h CPU |
| 36 | `$V9/stores` | 2.901886 | `reports/v9/v9_go_pivot_stop.md:3` (blob `4cad983ce7`) | 2–120 h CPU |
| 37 | `$H/v56` | 2.837036 | `handover.md:18024` (blob `da326fb0db`) | 6–168 h CPU/Metal |
| 38 | `$H/v95` | 2.751923 | `handover.md:18627` (blob `da326fb0db`) | 6–168 h CPU/Metal |
| 39 | `$R/audits/retrieval_v100_complete_word_metadata` | 2.724876 | `handover.md:9230` (blob `da326fb0db`) | 2–120 h CPU |
| 40 | `$H/v22` | 2.611858 | `handover.md:17234` (blob `da326fb0db`) | 6–168 h CPU/Metal |

Le run complet3000 n’a pas été exécuté ; le lot50/les smokes et leurs résultats publiés ne deviennent pas un résultat3000. L’extension abandonnée reste B sauf ses exceptions. Les anciens stores `candidates_v7_*` et modèles HF/PPLX ne sont pas lus par le serving actuel tracé ci-dessus.

## Exceptions : suppression fichier par fichier, jamais du dossier mixte

Dans **toutes** les unités B, le script énumère les fichiers puis retire uniquement les non-A/non-C ; il n’utilise ni glob de suppression ni `rm -rf`/`rmtree`. Les dossiers ne sont retirés que s’ils sont vides, non A, non parents minimaux d’un A et non pending.
La liste complète et explicite des fichiers gelés/production à garder est `exact_A_files` ; les sous-arbres A sont `full_A_roots`. Par unité B, `scan.explicit_A_exceptions`, `A_subtree_exceptions`, `missing_A_namespace_exceptions` et `scan.C_exceptions` donnent les exceptions exactes, y compris celles de taille nulle.

| Unité mixte proposée | A Gio | Fichiers A présents | Exceptions C/pending | C Gio |
|---|---:|---:|---:|---:|
| `$H/v174/train3000_frozen_v142_20261017/extension_20261017` | 0.000000 | 0 | 696 | 0.340485 |
| `$R/audits/retrieval_v34_fast_resume/full_job` | 0.011402 | 1431 | 0 | 0.000000 |
| `$H/v14/representative_20261008_b/zones` | 1.603928 | 693 | 0 | 0.000000 |
| `$H/v129/osm_local_20261013_night_e` | 1.645504 | 4 | 0 | 0.000000 |
| `$H/v36` | 0.000011 | 3 | 0 | 0.000000 |
| `$H/v33` | 0.000019 | 5 | 0 | 0.000000 |
| `$H/v173` | 0.000000 | 0 | 9928 | 0.000000 |
| `$R/audits/retrieval_v96_national_membership` | 0.327061 | 1 | 0 | 0.000000 |
| `$H/v141` | 1.678230 | 7 | 0 | 0.000000 |
| `$H/v19` | 0.000095 | 1 | 6 | 0.000046 |
| `$H/v2` | 0.002388 | 1 | 0 | 0.000000 |
| `$R/audits/retrieval_v33_full_replay` | 0.000336 | 44 | 0 | 0.000000 |
| `$H/v143` | 0.351448 | 532 | 0 | 0.000000 |
| `$H/v14/full_validation_20261008_a` | 2.843811 | 1 | 0 | 0.000000 |
| `$H/v21` | 0.064373 | 4 | 0 | 0.000000 |
| `$H/v72` | 0.000008 | 2 | 0 | 0.000000 |
| `$H/v134` | 0.018139 | 5 | 0 | 0.000000 |
| `$H/v4` | 0.000076 | 2 | 0 | 0.000000 |
| `$H/v71` | 0.000397 | 8 | 0 | 0.000000 |
| `$H/v132` | 0.000061 | 2 | 0 | 0.000000 |
| `$H/v57` | 0.082081 | 3 | 0 | 0.000000 |
| `$H/v17` | 0.007526 | 1 | 0 | 0.000000 |
| `$H/v68` | 0.000164 | 1 | 0 | 0.000000 |
| `$H/v23` | 0.026554 | 4 | 0 | 0.000000 |
| `$H/v32` | 0.010784 | 10 | 0 | 0.000000 |
| `$H/v142` | 0.290722 | 1587 | 0 | 0.000000 |
| `$H/v14/representative_20261008_b` | 0.000183 | 2 | 0 | 0.000000 |
| `$H/v88` | 0.202084 | 3 | 0 | 0.000000 |
| `$H/v99` | 0.000298 | 4 | 0 | 0.000000 |
| `$H/v81` | 0.000202 | 3 | 0 | 0.000000 |
| `$H/v75/published_supplier_context_20261013_a` | 0.000172 | 2 | 0 | 0.000000 |
| `$R/audits/retrieval_v34_fast_resume/parallel_repaired32` | 0.000015 | 2 | 0 | 0.000000 |
| `$H/v115` | 0.062958 | 534 | 0 | 0.000000 |
| `$R/audits/retrieval_v328_serving_panel` | 0.002335 | 1 | 0 | 0.000000 |
| `$H/v98` | 0.079094 | 3 | 0 | 0.000000 |
| `$H/v6` | 0.000004 | 1 | 0 | 0.000000 |
| `$H/v75/published_supplier_context_20261013_a/features` | 25.335388 | 526 | 0 | 0.000000 |
| `$H/v130` | 0.048172 | 13 | 0 | 0.000000 |
| `$H/v114` | 0.000057 | 2 | 0 | 0.000000 |
| `$H/v133` | 0.000099 | 5 | 0 | 0.000000 |
| `$H/v137` | 0.000359 | 6 | 0 | 0.000000 |
| `$H/v65` | 0.000008 | 2 | 0 | 0.000000 |
| `$H/v10` | 0.000648 | 2 | 0 | 0.000000 |
| `$H/v74` | 0.000011 | 2 | 0 | 0.000000 |
| `$H/v112` | 3.516598 | 3 | 0 | 0.000000 |
| `$H/v70` | 0.000004 | 1 | 0 | 0.000000 |
| `$R/audits/retrieval_v126_national_layout` | 0.830425 | 5 | 0 | 0.000000 |

**Tous les `*.pending.*` restent intouchés**, même s’ils paraissent anciens. Les liens vers une cible étrangère restent C, sans parcours de leur cible. Une unité contenant A est une enveloppe de proposition : ce ne sont jamais ses fichiers A qui sont autorisés à la suppression.

## C — doutes ≥ 1 Gio, une justification par unité

| Unité | C Gio | Pourquoi ne pas toucher |
|---|---:|---|
| `$R/sources/siren_dossier_v3/01eb624cca97c6f3` | 20.290302 | Corpus documentaire dérivé distinct du stock brut ; dépendances tierces et chaîne exacte de reconstruction non prouvées dans ce brief. |
| `$R/sources/siren_dossier_retrieval_v3_full` | 8.398903 | Source documentaire de constructions historiques ; pas un simple cache explicitement autorisé B et pas de preuve suffisante de non-réutilisation. |
| `$R/sources/siren_dossier_v1` | 7.005726 | Ancienne source documentaire ; caractère remplaçable et disponibilité de ses entrées non établis. |
| `$R/experiments/channel_audit_k5000_train_c33b80855f560074_aeeaf0f` | 1.785633 | Audit historique sans publication de résultat/STOP identifiée assez précisément pour autoriser B. |
| `$H/v157` | 1.555546 | Expérience Gemma sans résultat/arrêt commité retrouvé de façon fiable ; ne pas assimiler un contrat à un résultat. |
| `$R/host_cache` | 1.521736 | Origine, consommateurs et coût exact de reconstruction du cache hôte non établis ; laisser intact. |
| `$R/experiments/closed_overlay_channel_audit_k5000_train_c33b80855f560074_aeeaf0f` | 1.437958 | Audit historique sans publication de résultat/STOP identifiée assez précisément pour autoriser B. |

Les unités C plus petites et leurs raisons sont dans le JSON ; C inclut également les exceptions pending et liens étrangers à l’intérieur des propositions B. Aucun C n’est inclus dans l’allowlist de fichiers supprimables.

## Application future, pas exécutée ici

Le script contient **587 unités B explicites**, avec portée récursive ou fichiers directs. Par défaut : `--dry-run`. Seule la combinaison `--apply --confirmed-by-nathan` permet l’application ; tout autre mélange d’arguments est refusé.
Avant toute suppression : SHA du plan/handover/sources/docs, historique Git, statut tiers, autorités A exactes, inventaire des métadonnées étape2, snapshots complets de chaque unité B et tailles `du -sk` sont contrôlés. Un nouveau fichier, lien, pending, binding ou référence de code impose un nouvel audit. Aucun fichier A ni son espace de noms n’est supprimable.
Les liens physiques supprimés par le script sont suivis pour ne pas confondre leur décrément de compteur avec une écriture étrangère. Les arbres sont tous contrôlés avant la première suppression, puis chaque unité et chaque fichier juste avant l’opération. Le script ne contrôle/arrête aucun autre processus : **arrêter les écritures SIRETO manuellement**, sans intervenir sur CatNat/VM/PID8288.
Après accord uniquement, un humain pourra lancer `bash scripts/disk_cleanup_apply_v2.sh --dry-run`, puis éventuellement `bash scripts/disk_cleanup_apply_v2.sh --apply --confirmed-by-nathan`. **Aucune de ces commandes n’a été lancée pendant cet audit. Ne pas utiliser le script v1 pour appliquer la politique v2.**

**Ce qu’on perd en supprimant B : les caches, matrices, anciens index, reçus et sorties détaillées de calcul permettant le rejeu immédiat des anciennes expériences ; les résultats publiés, V142 exact, le serving, l’étape2 et les sources/travaux protégés restent gardés.**

## Vérifications

Vérifications réelles lancées : `python3 /tmp/sireto_disk_v2_verify.py` et `bash -n scripts/disk_cleanup_apply_v2.sh`. Résultat observé : **10/10 mesures `du -sk` du top B conformes**, **6 arbres réels / 42 entrées / 22 fichiers proposés** rejoués en lecture seule avec signatures et exceptions identiques, **587 unités explicites** cohérentes avec le JSON, **3 680/3 681 fichiers V142 présents aux tailles attendues**, **3 131 SHA de sources/documents** conformes et **24 publications commitées** vérifiées par blob. Totaux des 911 unités réconciliés ; syntaxe Bash et AST Python valides. Ni suite de tests unitaires ni script d’application exécuté.
Plan JSON SHA256 : `e905191c50046c167a600d6839a39af1fb33dc103066e59b2469e2002379ec9d`. Les fichiers JSON/MD/script sont autonomes : aucune dépendance aux utilitaires d’analyse `/tmp`.
Commit isolé de ce jalon : `git log -1 --format=%H -- docs/disk_cleanup_plan_v2.md` (pas de push).
