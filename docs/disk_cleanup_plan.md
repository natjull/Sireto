# Plan conservateur de ménage disque SIRETO

Audit lecture seule publié le 2026-10-21T20:24:55.388028+00:00. **Aucune suppression, aucun déplacement, aucune compression. Validation Nathan requise.**

`$H` = `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2` ; `$R` = `/Volumes/CATNAT_DATA/SIRETO_RECALL100` ; `$REPO` = `/Users/nathanjullia/Documents/Projets/SIRETO`.

## Synthèse chiffrée

- **Total SIRETO SSD : 658.566 Gio**, 45 racines explicitement attribuées SIRETO ; 911 postes disjoints.
- **OBSOLÈTE supprimable proposé : 0.003 Gio** (35 dossiers).
- **RÉGÉNÉRABLE potentiel : 0.000 Gio** ; non inclus dans le script.
- Les chiffres de l’étude précédente ne sont pas repris comme mesures actuelles. Le volume est vivant ; `df` ci-dessous ne décrit pas seulement SIRETO.

```text
Filesystem   1024-blocks       Used Available Capacity iused      ifree %iused  Mounted on
/dev/disk5s1  1953309744 1711956348 241060588    88% 2780987 2410605880    0%   /Volumes/CATNAT_DATA
```

| Classe | Dossiers/postes | Gio alloués | Décision |
|---|---:|---:|---|
| GARDER | 795 | 635.223 | Conserver |
| RÉGÉNÉRABLE | 1 | 0.000 | Conserver avant validation/coût de reconstruction |
| OBSOLÈTE | 35 | 0.003 | Liste explicite proposée, pas exécutée |
| DOUTE | 80 | 23.340 | Laisser |

Exemples du brief, **mesures actuelles** : `v42` = 0.053 Gio, `v73` = 0.075 Gio, `v89` = 0.045 Gio, `v28` = 0.049 Gio. Les 31/29/27/24 Gio de l’ancienne étude ne sont plus présents dans ces dossiers ; aucune cause ni suppression n’est attribuée par cet audit.

## Protections et méthode

- Gel V142 : **3 681 chemins résolus, 3680 présents de taille attendue, 1 absent, 0 écart de taille**. Chaque chemin, SHA déclaré, chemin résolu et contrôle de présence figure dans le JSON. SHA des 148 Go historiques **non recalculés** ; aucun contenu de label fermé ouvert.
  - **Déjà absent** : `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v6/deadline_recovery_20261008_a/fit/best_adapter.pt`. Ne pas présenter le gel comme intégralement disponible ; restauration hors de ce brief.
- Index national `/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/hierarchical_retrieval_v3/4b332899d8698354` protégé ; `serve_v1`, revue de bruit, runtime, checkpoint BASE et banques BASE réutilisées protégés.
- Références des quatre scripts demandés suivies statiquement, sans les exécuter ; clôture des imports : 146 scripts. Les plans/reçus résolvent les sources G0, S1/S2, risques/statuts et serving. Dépendance NAF Oùbusiness seulement citée, jamais inspectée.
- `git grep` dans tout le dépôt suivi, recherche par nom distinctif **et chemin absolu** ; 215 fichiers récents et **424 blobs distincts** avant/après depuis le 14/10/2026, y compris anciens passages du handover. Les mentions récentes d’inventaire sont aussi conservées, par prudence. Contrôle complémentaire des candidats insensible à la casse et aux séparateurs : les versions/caches ambiguës (V91/v91, etc.) passent en DOUTE. Une déclaration de racine de stockage seule n’est pas une dépendance de tous ses descendants.
- **77851 manifestes JSON inspectés**, 1133567 liens inter-postes conservés (chemins ou SHA). Manifeste de fichier direct d’un parent et références SHA sans chemin inclus. 68911 fichiers candidats hachés ; comparaison aux empreintes documentaires et des manifestes gardés. 42312 empreintes de métadonnées sans fichier résolu, y compris empreintes de tableaux/configurations ; elles ne constituent pas automatiquement des fichiers.
- Erreurs de parcours : 0 ; manifestes non lus/invalides : 0. Une lecture incomplète interdit de certifier une suppression non vide. Pending/lien/entrée spéciale interdisent la suppression du sous-arbre.
- Tailles : `du -k` pour la hiérarchie, **`du -sk` pour les 877 dossiers disjoints et la racine de staging `/Volumes/CATNAT_DATA/0`**, un seul parcours lourd à la fois. Les postes « fichiers directs » = parent moins enfants ; ils ne sont jamais des cibles. Dates = mtime du répertoire UTC, pas date récursive.
- Les totaux sont alloués, **pas une garantie de gain physique** (clones APFS, snapshots et liens durs). Aucun examen des autres projets/VM/PID 8288, aucun web.

## Traçage des entrées production / étape 2

- Serving (`sireto_serve_v1.py:33-36,241-305,403-405,464-470`) : `serve_v1`, `step2_option_d_20261017/full`, risque `_c/full`, ranker groupe, tiers CRM, banques `step2_gemma_finetune_10h_20261019/full/base_encode`, checkpoint `base`, runtime. Exports CRM bruts et métadonnées de population sous `$R/datasets` gardés.
- G0 (`step2_g0_bench.py:25-32,50-59,119-130`) : plan Option D, `training_queries`, `matrix/features/keys`, géométrie ; autorités officielles `$R/sources/sirene/2026-08-01/{StockEtablissement,StockUniteLegale,StockEtablissementHistorique}.parquet`. Référence géolocalisation Oùbusiness seulement lue dans le plan, source externe non inspectée.
- S1/S2 (`step2_colocated_signals.py:27-31,318-325`) : checkpoint BASE, manifests/bindings/vecteurs du colocated et du BASE full 10h ; NAF externe laissé entièrement hors périmètre.
- Risque/statuts (`step2_crm_columns_tiers.py:16-18,45-49,65-70`) : imports ranker groupe → risque `_c` → full 10h / G0 ; modèles et reçus récents conservés. Les liens de fichiers réels figurent dans les preuves JSON.

## Racines SSD (non additives avec leurs enfants)

| Racine | Gio | mtime UTC |
|---|---:|---|
| `/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2` | 163.381 | 2026-10-20T12:27:38.763926+00:00 |
| `/Volumes/CATNAT_DATA/SIRETO_M3_PYTHON_TCB` | 0.000 | 2026-07-31T12:47:50.029528+00:00 |
| `/Volumes/CATNAT_DATA/SIRETO_M3_TCB_TEST` | 0.000 | 2026-07-31T12:35:42.522141+00:00 |
| `/Volumes/CATNAT_DATA/SIRETO_PPLX` | 2.256 | 2026-10-08T16:47:18.395154+00:00 |
| `/Volumes/CATNAT_DATA/SIRETO_RECALL100` | 483.780 | 2026-10-10T10:44:22.594707+00:00 |
| `/Volumes/CATNAT_DATA/SIRETO_STORAGE_QUARANTINE` | 0.000 | 2026-10-20T17:27:45.363685+00:00 |
| `/Volumes/CATNAT_DATA/SIRETO_V9` | 9.146 | 2026-08-24T10:51:00.076283+00:00 |
| `/Volumes/CATNAT_DATA/sireto-audit-fhk4nszm` | 0.000 | 2026-07-31T10:29:50.900150+00:00 |
| `/Volumes/CATNAT_DATA/sireto-native-anchor.3u44bv62` | 0.000 | 2026-07-31T11:09:11.811150+00:00 |
| `/Volumes/CATNAT_DATA/sireto-native-anchor2.1ln9o_3x` | 0.000 | 2026-07-31T11:09:19.518702+00:00 |
| `/Volumes/CATNAT_DATA/sireto-native-audit.CMWbvw` | 0.000 | 2026-07-31T10:59:06.333910+00:00 |
| `/Volumes/CATNAT_DATA/sireto-native-forge.i1duf0kv` | 0.000 | 2026-07-31T11:00:24.786588+00:00 |
| `/Volumes/CATNAT_DATA/sireto-native-forge2.rxbp5hon` | 0.000 | 2026-07-31T11:00:32.750720+00:00 |
| `/Volumes/CATNAT_DATA/sireto-native-forge3.ejphutqx` | 0.000 | 2026-07-31T11:00:40.967002+00:00 |
| `/Volumes/CATNAT_DATA/sireto-native-name-audit.oFeIVK` | 0.000 | 2026-07-31T10:59:46.918242+00:00 |
| `/Volumes/CATNAT_DATA/sireto-native-nouuid.tp2FFe` | 0.000 | 2026-07-31T10:59:55.647194+00:00 |
| `/Volumes/CATNAT_DATA/sireto-native-rereview.dZecJ6` | 0.000 | 2026-07-31T11:09:03.206934+00:00 |
| `/Volumes/CATNAT_DATA/sireto-s0-error-selfcheck.EHxKkj` | 0.000 | 2026-10-13T20:58:24.436378+00:00 |
| `/Volumes/CATNAT_DATA/sireto-s0-error-selfcheck.LGVHwb` | 0.000 | 2026-10-13T20:58:24.433238+00:00 |
| `/Volumes/CATNAT_DATA/sireto-s0-fd-selfcheck.bQKNQZ` | 0.000 | 2026-07-30T16:28:19.061954+00:00 |
| `/Volumes/CATNAT_DATA/sireto-s0-final-selfcheck.NdaYs5` | 0.000 | 2026-07-30T16:29:42.616942+00:00 |
| `/Volumes/CATNAT_DATA/sireto-s0-selfcheck.MMNVt3` | 0.000 | 2026-07-30T16:17:22.732243+00:00 |
| `/Volumes/CATNAT_DATA/sireto-s0-stable-selfcheck.Qno6TZ` | 0.000 | 2026-07-30T16:30:32.130524+00:00 |
| `/Volumes/CATNAT_DATA/sireto-s0-stop-selfcheck.HrSLUw` | 0.000 | 2026-07-30T16:28:40.508492+00:00 |
| `/Volumes/CATNAT_DATA/sireto-systemvol-final.4rFcrc` | 0.000 | 2026-07-31T11:18:26.228131+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-diag.6jrpPN` | 0.000 | 2026-07-31T05:49:13.502923+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-diag.TnF6YR` | 0.000 | 2026-07-31T06:00:15.849298+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-diag.VmG6W9` | 0.000 | 2026-07-31T05:49:59.297958+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-diag.ymbnIR` | 0.000 | 2026-07-31T05:49:43.545449+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-diagtmp.4Smw9Z` | 0.000 | 2026-07-31T05:49:13.504569+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-diagtmp.a3NN4c` | 0.000 | 2026-07-31T06:00:15.850900+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-diagtmp.eUZBgJ` | 0.000 | 2026-07-31T05:49:59.299752+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-diagtmp.mFnd7r` | 0.000 | 2026-07-31T05:49:43.548810+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-parent.8pAlsx` | 0.000 | 2026-07-31T06:05:50.970865+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-parent.AsM8eD` | 0.000 | 2026-07-31T05:48:46.655435+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-parent.R1Va2b` | 0.000 | 2026-07-31T07:21:29.054525+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-parent.jEawJf` | 0.000 | 2026-07-31T05:56:06.258642+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-parent.o852O6` | 0.000 | 2026-07-31T06:31:17.842376+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-tmp.Ybor2A` | 0.000 | 2026-07-31T07:21:29.056257+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-tmp.cF4Del` | 0.000 | 2026-07-31T06:31:17.844019+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-tmp.cKedzd` | 0.000 | 2026-07-31T05:48:46.657061+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-tmp.xogeaK` | 0.000 | 2026-07-31T06:05:50.972751+00:00 |
| `/Volumes/CATNAT_DATA/sireto-v412-tmp.yhvyZj` | 0.000 | 2026-07-31T05:56:06.260327+00:00 |
| `/Volumes/CATNAT_DATA/sireto_m3_cli_smoke_20260731` | 0.002 | 2026-07-31T12:27:06.212729+00:00 |
| `/Volumes/CATNAT_DATA/0` | 0.000 | 2026-07-31T13:31:12.072332+00:00 |

## Dix plus gros OBSOLÈTE proposés

| Chemin | Gio | KiB exacts | Preuve |
|---|---:|---:|---|
| `/Volumes/CATNAT_DATA/sireto_m3_cli_smoke_20260731` | 0.002 | 2364 | Pas de référence récente/manifestes gardés ; contenu inventorié et haché. |
| `/Volumes/CATNAT_DATA/sireto-native-rereview.dZecJ6` | 0.000 | 128 | Pas de référence récente/manifestes gardés ; contenu inventorié et haché. |
| `/Volumes/CATNAT_DATA/sireto-native-audit.CMWbvw` | 0.000 | 80 | Pas de référence récente/manifestes gardés ; contenu inventorié et haché. |
| `/Volumes/CATNAT_DATA/sireto-native-name-audit.oFeIVK` | 0.000 | 80 | Pas de référence récente/manifestes gardés ; contenu inventorié et haché. |
| `/Volumes/CATNAT_DATA/sireto-native-nouuid.tp2FFe` | 0.000 | 72 | Pas de référence récente/manifestes gardés ; contenu inventorié et haché. |
| `/Volumes/CATNAT_DATA/sireto-native-anchor.3u44bv62` | 0.000 | 56 | Pas de référence récente/manifestes gardés ; contenu inventorié et haché. |
| `/Volumes/CATNAT_DATA/sireto-native-anchor2.1ln9o_3x` | 0.000 | 56 | Pas de référence récente/manifestes gardés ; contenu inventorié et haché. |
| `/Volumes/CATNAT_DATA/SIRETO_PPLX/probe_20261008_a` | 0.000 | 40 | Pas de référence récente/manifestes gardés ; contenu inventorié et haché. |
| `/Volumes/CATNAT_DATA/sireto-native-forge2.rxbp5hon` | 0.000 | 40 | Pas de référence récente/manifestes gardés ; contenu inventorié et haché. |
| `/Volumes/CATNAT_DATA/sireto-native-forge3.ejphutqx` | 0.000 | 40 | Pas de référence récente/manifestes gardés ; contenu inventorié et haché. |

## DOUTE notables / régénération

- **RÉGÉNÉRABLE** : `$H/torchinductor_nathanjullia`, **0 KiB actuellement**. Cache compilateur PyTorch recréable avec le code/runtime gardés ; coût estimé non mesuré : quelques secondes à des dizaines de minutes CPU selon les graphes, sans réencodage Gemma. Aucun espace positif n’est promis.
- **DOUTE**, `$R/models/huggingface/hub` (8.745 Gio) : Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite.
- **DOUTE**, `$R/stores/candidates_v7_current_crmgt_full_20260817` (7.368 Gio) : Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide.
- **DOUTE**, `/Volumes/CATNAT_DATA/SIRETO_PPLX/hf_cache` (2.233 Gio) : Cache : reconstruction hors ligne et absence de réutilisation non démontrées.
- **DOUTE**, `$R/host_cache` (1.522 Gio) : Cache : reconstruction hors ligne et absence de réutilisation non démontrées.
- **DOUTE**, `$H/uv_cache` (0.963 Gio) : Cache : reconstruction hors ligne et absence de réutilisation non démontrées.
- **DOUTE**, `$R/revoked_prelaunch_s0` (0.478 Gio) : Contenu label/jeu fermé possible : non ouvert, laissé en place.
- **DOUTE**, `$H/v91` (0.297 Gio) : Nom de version/cache cité dans un document récent avec casse ou délimitation ambiguë (ex. V91/v91) ; ne pas supprimer sur cette ambiguïté.
- **DOUTE**, `$R/datasets/synthetic_gt_model_features` (0.281 Gio) : Sources, jeux réservés ou autorités historiques ; conservation par précaution.
- **DOUTE**, `$R/stores/candidates_v7_current_crmgt_20260817` (0.187 Gio) : Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide.
- **DOUTE**, `$R/fresh_holdout_intake_synthetic` (0.160 Gio) : Contenu label/jeu fermé possible : non ouvert, laissé en place.
- **DOUTE**, `$R/datasets/v4_12_learned_candidate_features` (0.151 Gio) : Sources, jeux réservés ou autorités historiques ; conservation par précaution.
- **DOUTE**, `$R/fresh_holdout_intake_synthetic_r3` (0.145 Gio) : Contenu label/jeu fermé possible : non ouvert, laissé en place.
- **DOUTE**, `$R/diag-r3-successor-gate.9fy69i1l` (0.144 Gio) : Contenu label/jeu fermé possible : non ouvert, laissé en place.
- **DOUTE**, `$R/diag-r3-successor-gate.zi2oynzm` (0.144 Gio) : Contenu label/jeu fermé possible : non ouvert, laissé en place.
- **DOUTE**, `$R/diag-r3-successor-gate.3129l7j9` (0.144 Gio) : Contenu label/jeu fermé possible : non ouvert, laissé en place.

## Gros fichiers locaux non suivis (information, aucune suppression)

| Chemin | Gio (`du -sk`) | mtime UTC |
|---|---:|---|
| `$REPO/data/StockEtablissement_utf8.parquet` | 1.951 | 2025-12-03T21:21:12+00:00 |
| `$REPO/data/samples_v5fast_RANKER_500neg_clean.parquet` | 1.735 | 2026-02-04T08:19:40.291682+00:00 |
| `$REPO/models/semantic/siret-bert-v1/run_20251223_173124/checkpoint-1000/optimizer.pt` | 0.876 | 2025-12-23T16:39:33+00:00 |
| `$REPO/models/semantic/siret-bert-v1/run_20251223_173124/checkpoint-2808/optimizer.pt` | 0.876 | 2025-12-23T16:53:35+00:00 |
| `$REPO/data/StockUniteLegale_utf8.parquet` | 0.596 | 2025-12-04T08:48:56+00:00 |
| `$REPO/models/semantic/siret-bert-deploy/model.safetensors` | 0.438 | 2025-12-23T16:39:32+00:00 |
| `$REPO/models/semantic/siret-bert-v1/run_20251223_173124/checkpoint-1000/model.safetensors` | 0.438 | 2025-12-23T16:39:32+00:00 |
| `$REPO/models/semantic/siret-bert-v1/run_20251223_173124/checkpoint-2808/model.safetensors` | 0.438 | 2025-12-23T16:53:33+00:00 |
| `$REPO/models/semantic/siret-bert-v1/run_20251223_173124/final/model.safetensors` | 0.438 | 2025-12-23T16:53:36+00:00 |
| `$REPO/data/siren_index/siren_to_geo.parquet` | 0.353 | 2026-02-28T14:48:45.370120+00:00 |
| `$REPO/tmp/bano_public_20261013_lZ8b88/full.csv.gz` | 0.313 | 2026-10-13T23:04:33.533397+00:00 |

Les tailles locales ne sont pas ajoutées au total SSD. `.git` et `.venv` sont donnés uniquement comme tailles de contexte dans le JSON ; ni objets Git ni modèles/données locales ne sont proposés à la suppression.

## Résumés parents/versions (non additifs)

| Parent | Gio | mtime UTC |
|---|---:|---|
| `$H` | 163.381 | 2026-10-20T12:27:38.763926+00:00 |
| `$H/v129` | 11.812 | 2026-10-13T23:44:48.331659+00:00 |
| `$H/v14` | 9.798 | 2026-10-08T17:08:51.711076+00:00 |
| `$H/v14/representative_20261008_b` | 8.225 | 2026-10-09T08:26:19.800682+00:00 |
| `$H/v174` | 48.418 | 2026-10-17T11:12:33.320379+00:00 |
| `$H/v174/train3000_frozen_v142_20261017` | 48.418 | 2026-10-17T11:26:41.650915+00:00 |
| `$H/v75` | 25.364 | 2026-10-13T00:18:17.886758+00:00 |
| `$H/v75/published_supplier_context_20261013_a` | 25.364 | 2026-10-13T02:06:33.728237+00:00 |
| `/Volumes/CATNAT_DATA/SIRETO_PPLX` | 2.256 | 2026-10-08T16:47:18.395154+00:00 |
| `$R` | 483.780 | 2026-10-10T10:44:22.594707+00:00 |
| `$R/audits` | 160.415 | 2026-10-07T13:49:35.962419+00:00 |
| `$R/audits/retrieval_v129_national_variants` | 14.244 | 2026-10-03T13:12:53.196016+00:00 |
| `$R/audits/retrieval_v134_packed_names` | 9.449 | 2026-10-03T16:14:55.750733+00:00 |
| `$R/audits/retrieval_v177_packed_addresses` | 10.819 | 2026-10-04T05:48:13.044808+00:00 |
| `$R/audits/retrieval_v206_population_initial` | 10.464 | 2026-10-04T13:41:12.988137+00:00 |
| `$R/audits/retrieval_v34_fast_resume` | 9.369 | 2026-09-11T16:06:16.195841+00:00 |
| `$R/datasets` | 21.690 | 2026-08-24T11:46:59.106749+00:00 |
| `$R/datasets/synthetic_gt_corpus` | 19.982 | 2026-08-16T08:46:51.287045+00:00 |
| `$R/experiments` | 53.172 | 2026-08-24T12:45:03.068409+00:00 |
| `$R/experiments/retrieval_all_human_v11` | 13.328 | 2026-08-23T23:36:28.875618+00:00 |
| `$R/experiments/v4_12_bge_groupwise` | 10.667 | 2026-08-15T11:52:21.935885+00:00 |
| `$R/indexes` | 108.655 | 2026-10-02T15:55:10.551900+00:00 |
| `$R/indexes/hierarchical_retrieval_v3` | 98.090 | 2026-08-19T12:46:06.716886+00:00 |
| `$R/indices` | 17.669 | 2026-08-21T16:02:09.125240+00:00 |
| `$R/indices/sirene_temporal_v1` | 14.554 | 2026-08-20T21:41:55.371135+00:00 |
| `$R/models` | 8.953 | 2026-08-24T11:49:53.333894+00:00 |
| `$R/models/huggingface` | 8.745 | 2026-08-13T07:58:28.092796+00:00 |
| `$R/sources` | 100.433 | 2026-08-21T15:59:01.276145+00:00 |
| `$R/sources/official_evidence` | 30.102 | 2026-08-20T11:47:51.200885+00:00 |
| `$R/sources/official_sync` | 26.796 | 2026-08-18T12:55:23.440956+00:00 |
| `$R/sources/siren_dossier_v3` | 20.290 | 2026-08-18T20:35:03.577537+00:00 |
| `$R/stores` | 8.399 | 2026-08-17T15:29:14.881768+00:00 |
| `/Volumes/CATNAT_DATA/SIRETO_V9` | 9.146 | 2026-08-24T10:51:00.076283+00:00 |

## Tableau exhaustif des postes disjoints

Preuves détaillées (fichier, numéro de ligne, blob historique, manifeste source/cible) dans `disk_cleanup_plan.json`. Les mentions documentaires historiques récentes suffisent à conserver ; absence de référence seule ne prouve pas la reproductibilité.

| Dossier/poste | Gio | mtime UTC | Classe | Preuve / motif |
|---|---:|---|---|---|
| `/Volumes/CATNAT_DATA/0` | 0.000 | 2026-07-31T13:31:12.072332+00:00 | GARDER | STAGING_PARENT du worker SIRETO dans scripts/v412_review_python_worker_spike.py:39 ; racine actuellement vide. |
| `$H/base` | 1.421 | 2026-10-06T21:34:25.753948+00:00 | GARDER | protection explicite; récent `docs/serve_v1_manifest.json:348`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/result.json` |
| `$H/cached_20261007_a` | 0.000 | 2026-10-20T17:27:44.158688+00:00 | GARDER | récent `docs/step2_option_d.json:121`; manifeste gardé : `$H/v6/deadline_recovery_20261008_a/comparison/v1_native160_binding.json` |
| `$H/comparison_v1` | 0.006 | 2026-10-07T17:14:36.929113+00:00 | GARDER | récent `handover.md:15968` |
| `$H/hf_cache` | 0.000 | 2026-10-06T21:34:09.149321+00:00 | DOUTE | Cache : reconstruction hors ligne et absence de réutilisation non démontrées. |
| `$H/label_noise_review_20261020` | 0.287 | 2026-10-20T07:00:14.839355+00:00 | GARDER | protection explicite; récent `docs/label_noise_review_v2.json:3`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/result.json` |
| `$H/night_20261006_a` | 0.017 | 2026-10-20T17:27:43.856784+00:00 | GARDER | récent `docs/step2_option_d.json:58`; manifeste gardé : `$H/step2_option_d_20261017/smoke/result.json` |
| `$H/night_20261013_b` | 0.000 | 2026-10-13T21:09:02.404632+00:00 | GARDER | récent `handover.md:19029` |
| `$H/runtime` | 1.062 | 2026-10-06T21:33:48.187926+00:00 | GARDER | protection explicite; récent `docs/step2_colocated_signals.json:1865395`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/prepare_ad3920cee85b2a761e5e3302ff95ac58f7fd48f3209b48a63a692decf14d1324_8fce752d2e/numeric/v14/zones/62043/v3_official512/binding.json` |
| `$H/serve_v1` | 1.200 | 2026-10-20T13:17:01.684026+00:00 | GARDER | protection explicite; récent `docs/serve_v1_manifest.json:2` |
| `$H/step2_colocated_signals_20261019` | 1.328 | 2026-10-19T12:25:33.563001+00:00 | GARDER | protection explicite; récent `docs/step2_colocated_signals.json:15`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/plan.json` |
| `$H/step2_crm_columns_tiers_20261020` | 0.063 | 2026-10-20T12:07:14.371984+00:00 | GARDER | récent `docs/step2_crm_columns_tiers.json:716` |
| `$H/step2_diagnostic_20261017` | 0.039 | 2026-10-17T10:24:08.101085+00:00 | GARDER | récent `docs/step2_diagnostic_v1.json:161`; manifeste gardé : `$H/step2_v142_20261017/ranker_v1/full/result.json` |
| `$H/step2_error_families_g0_20261019` | 0.018 | 2026-10-19T10:42:42.958699+00:00 | GARDER | récent `docs/step2_error_families_g0.json:9081` |
| `$H/step2_features_v2_20261017` | 0.023 | 2026-10-17T10:46:13.199186+00:00 | GARDER | récent `docs/step2_features_v2.json:202`; manifeste gardé : `$H/step2_diagnostic_20261017/smoke/result.json` |
| `$H/step2_g0_bench_20261017` | 0.472 | 2026-10-19T09:21:23.491103+00:00 | GARDER | récent `docs/crm_structure_audit.json:1083`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/result.json` |
| `$H/step2_gemma_finetune_10h_20261019` | 7.678 | 2026-10-19T19:29:22.608356+00:00 | GARDER | protection explicite; récent `docs/crm_structure_audit.json:1128`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/result.json` |
| `$H/step2_gemma_finetune_20261019` | 0.575 | 2026-10-19T13:14:50.195485+00:00 | GARDER | récent `docs/step2_gemma_finetune.json:14`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/result.json` |
| `$H/step2_option_d_20261017` | 1.437 | 2026-10-19T07:40:27.865449+00:00 | GARDER | récent `docs/crm_structure_audit.json:1068`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/result.json` |
| `$H/step2_ranker_v3_group_20261020` | 0.358 | 2026-10-20T10:30:54.483345+00:00 | GARDER | récent `docs/serve_v1_pilot.json:50`; manifeste gardé : `$H/step2_crm_columns_tiers_20261020/result.json` |
| `$H/step2_risk_ceiling_v1_20261020` | 0.022 | 2026-10-19T20:51:55.445361+00:00 | GARDER | récent `docs/label_noise_screen_v1.json:86`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/result.json` |
| `$H/step2_risk_model_v1_20261020_b` | 0.003 | 2026-10-19T19:59:37.079551+00:00 | GARDER | récent `docs/step2_risk_model_v1.md:49`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/result.json` |
| `$H/step2_risk_model_v1_20261020_c` | 0.102 | 2026-10-19T20:24:01.937228+00:00 | GARDER | récent `docs/crm_structure_audit.json:1078`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/plan.json` |
| `$H/step2_risk_model_v1_20261020` | 0.024 | 2026-10-19T19:55:09.820194+00:00 | GARDER | récent `docs/step2_crm_columns_tiers.json:26963`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/result.json` |
| `$H/step2_v142_20261017` | 0.017 | 2026-10-17T08:42:59.610880+00:00 | GARDER | récent `docs/step2_diagnostic_v1.json:152`; manifeste gardé : `$H/step2_diagnostic_20261017/smoke/plan.json` |
| `$H/torchinductor_nathanjullia` | 0.000 | 2026-10-06T21:45:46.333910+00:00 | RÉGÉNÉRABLE | Cache compilateur PyTorch ; recréable par les opérations correspondantes avec runtime et code gardés. |
| `$H/uv_cache` | 0.963 | 2026-10-11T18:30:44.025610+00:00 | DOUTE | Cache : reconstruction hors ligne et absence de réutilisation non démontrées. |
| `$H/v100` | 0.000 | 2026-10-13T04:10:18.773340+00:00 | GARDER | récent `handover.md:18629` |
| `$H/v101` | 0.000 | 2026-10-13T04:34:24.913448+00:00 | GARDER | récent `handover.md:18650` |
| `$H/v102` | 0.000 | 2026-10-13T04:44:18.863253+00:00 | GARDER | récent `handover.md:18644` |
| `$H/v103` | 0.000 | 2026-10-13T04:48:40.646167+00:00 | GARDER | récent `docs/retrieval_v143_osm_gain_5130_causal_audit_20261014.json:153` |
| `$H/v104` | 0.000 | 2026-10-13T05:00:52.837778+00:00 | GARDER | récent `handover.md:18658` |
| `$H/v105` | 0.000 | 2026-10-13T05:09:19.728499+00:00 | GARDER | récent `handover.md:18659` |
| `$H/v106` | 0.000 | 2026-10-13T05:13:39.916537+00:00 | GARDER | récent `handover.md:18671` |
| `$H/v107` | 0.000 | 2026-10-13T05:16:55.253918+00:00 | GARDER | récent `handover.md:18679` |
| `$H/v108` | 0.000 | 2026-10-13T05:23:25.416693+00:00 | GARDER | récent `handover.md:18696` |
| `$H/v109` | 0.000 | 2026-10-13T05:26:35.101554+00:00 | GARDER | récent `handover.md:18715` |
| `$H/v10` | 0.001 | 2026-10-08T07:12:39.329987+00:00 | GARDER | récent `docs/step2_option_d.json:72`; manifeste gardé : `$H/step2_option_d_20261017/smoke/result.json` |
| `$H/v110` | 0.000 | 2026-10-13T05:30:40.468643+00:00 | GARDER | récent `handover.md:18722` |
| `$H/v111` | 0.000 | 2026-10-13T06:51:19.602019+00:00 | GARDER | récent `handover.md:18780` |
| `$H/v112` | 3.517 | 2026-10-13T06:55:25.417975+00:00 | GARDER | V142 : 3 artefacts; récent `docs/retrieval_frozen_v142.json:708`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v113` | 0.003 | 2026-10-13T06:58:24.966998+00:00 | GARDER | récent `handover.md:18791` |
| `$H/v114` | 0.000 | 2026-10-13T07:01:43.695447+00:00 | GARDER | V142 : 2 artefacts; récent `docs/retrieval_frozen_v142.json:732`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v115` | 0.074 | 2026-10-13T07:05:43.327679+00:00 | GARDER | V142 : 534 artefacts; récent `docs/retrieval_frozen_v142.json:748`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v116` | 0.478 | 2026-10-13T07:12:11.586419+00:00 | GARDER | récent `handover.md:18810` |
| `$H/v117` | 0.001 | 2026-10-13T07:20:26.359024+00:00 | GARDER | récent `handover.md:18814` |
| `$H/v118` | 0.000 | 2026-10-13T07:42:15.023172+00:00 | GARDER | récent `handover.md:18819` |
| `$H/v119` | 0.000 | 2026-10-13T07:46:53.762565+00:00 | GARDER | récent `handover.md:18872` |
| `$H/v11` | 0.001 | 2026-10-08T07:19:05.611176+00:00 | GARDER | récent `handover.md:288` |
| `$H/v120` | 0.008 | 2026-10-13T07:55:58.352043+00:00 | GARDER | récent `handover.md:18877` |
| `$H/v121` | 0.000 | 2026-10-13T08:10:06.986103+00:00 | GARDER | récent `handover.md:18851` |
| `$H/v122` | 0.000 | 2026-10-13T11:28:32.639131+00:00 | GARDER | récent `docs/retrieval_miss_access_vs_ranking_20261014.json:3737` |
| `$H/v123` | 0.009 | 2026-10-13T19:48:53.768428+00:00 | GARDER | récent `handover.md:18962` |
| `$H/v124` | 0.000 | 2026-10-13T19:58:41.587511+00:00 | GARDER | récent `handover.md:18975` |
| `$H/v125` | 0.029 | 2026-10-13T20:03:20.495656+00:00 | GARDER | récent `handover.md:18983` |
| `$H/v126` | 0.000 | 2026-10-13T20:02:15.162971+00:00 | GARDER | récent `docs/step2_option_d.md:18` |
| `$H/v127` | 0.000 | 2026-10-13T20:17:35.007908+00:00 | GARDER | récent `handover.md:18992` |
| `$H/v128` | 0.000 | 2026-10-13T20:24:07.127374+00:00 | GARDER | récent `handover.md:18998` |
| `$H/v129/bano_independent_20261013_night_a` | 0.013 | 2026-10-13T22:59:26.012692+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:5021` |
| `$H/v129/independent_anchor_audit_20261014_a` | 0.000 | 2026-10-13T22:56:52.454008+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:5021` |
| `$H/v129/operator_query_only_audit_20261014_night_a` | 0.000 | 2026-10-13T23:45:30.533333+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:5021` |
| `$H/v129/osm_local_20261013_night_b` | 0.000 | 2026-10-13T21:46:21.794007+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:5021` |
| `$H/v129/osm_local_20261013_night_c` | 0.000 | 2026-10-13T22:06:48.575809+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:5021` |
| `$H/v129/osm_local_20261013_night_d` | 0.143 | 2026-10-13T22:13:05.175090+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:5021`; manifeste gardé : `$H/v129/osm_local_20261013_night_e/category_storage_binding.json` |
| `$H/v129/osm_local_20261013_night_e` | 6.887 | 2026-10-13T22:54:59.172527+00:00 | GARDER | V142 : 4 artefacts; récent `docs/retrieval_frozen_v142.json:5021`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v129/osm_local_20261013_night` | 4.769 | 2026-10-13T21:21:38.474862+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:5021` |
| `$H/v129` **[fichiers directs]** | 0.000 | 2026-10-13T23:44:48.331659+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:5021` |
| `$H/v12` | 0.050 | 2026-10-08T07:21:51.188290+00:00 | GARDER | récent `handover.md:67` |
| `$H/v130` | 0.049 | 2026-10-13T22:59:27.661031+00:00 | GARDER | V142 : 13 artefacts; récent `docs/production_cost_study.json:125`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v131` | 0.026 | 2026-10-13T21:13:29.097723+00:00 | GARDER | récent `handover.md:19038` |
| `$H/v132` | 0.067 | 2026-10-13T22:42:06.574032+00:00 | GARDER | V142 : 2 artefacts; récent `docs/retrieval_frozen_v142.json:5157`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v133` | 0.000 | 2026-10-13T21:50:13.215160+00:00 | GARDER | V142 : 5 artefacts; récent `docs/retrieval_frozen_v142.json:5173`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v134` | 0.117 | 2026-10-14T02:01:59.671698+00:00 | GARDER | V142 : 5 artefacts; récent `docs/retrieval_engine_explained_20261014.md:269`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v137` | 0.001 | 2026-10-13T21:57:11.092933+00:00 | GARDER | V142 : 6 artefacts; récent `docs/retrieval_frozen_v142.json:5253`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v138` | 0.001 | 2026-10-13T22:14:34.576481+00:00 | GARDER | récent `handover.md:19191`; manifeste gardé : `$H/v143/local_evidence_empty_recovery_20261013_night/banatic_only/native/a3ab7bbaf7390c1faafb17a4075a3a633882f897fa93ace3976bb64b0712842c.seal.json` |
| `$H/v139` | 0.090 | 2026-10-13T22:22:09.658257+00:00 | GARDER | récent `handover.md:19210`; manifeste gardé : `$H/v143/local_evidence_empty_recovery_20261013_night/banatic_only/native/017ae987199a49a101758e9e4c69421ea66ad828511e1f76b14c17428f206878.seal.json` |
| `$H/v13` | 0.000 | 2026-10-08T10:14:14.209836+00:00 | GARDER | récent `handover.md:68` |
| `$H/v14/full_validation_20261008_a` | 2.970 | 2026-10-08T14:23:35.234409+00:00 | GARDER | V142 : 1 artefacts; récent `docs/retrieval_far_label_audit_20261015.json:26`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/covered-prepare_fc2a89a72cbd6bd79ce81de33f43bb485b265a6d4ee5b2370bfdfc0da97577ad_59fc77a558/numeric/v14/names/owners.seal.json` |
| `$H/v14/perplexity_paired_20261008_a` | 0.001 | 2026-10-08T17:07:47.402603+00:00 | GARDER | récent `docs/retrieval_far_label_audit_20261015.json:26` |
| `$H/v14/perplexity_paired_20261008_b` | 1.447 | 2026-10-08T18:27:03.014085+00:00 | GARDER | récent `docs/retrieval_far_label_audit_20261015.json:26`; manifeste gardé : `$H/v14/representative_20261008_b/zones/71076/documents.seal.json` |
| `$H/v14/representative_20261008_b/names` | 0.000 | 2026-10-08T14:23:40.074084+00:00 | GARDER | récent `docs/retrieval_far_label_audit_20261015.json:26` |
| `$H/v14/representative_20261008_b/tmp` | 0.000 | 2026-10-09T06:57:37.468204+00:00 | GARDER | récent `docs/retrieval_far_label_audit_20261015.json:26` |
| `$H/v14/representative_20261008_b/zones` | 8.193 | 2026-10-08T14:23:40.403301+00:00 | GARDER | récent `docs/retrieval_far_label_audit_20261015.json:26`; manifeste gardé : `$H/v14/perplexity_paired_20261008_b/zones/71076/documents.seal.json` |
| `$H/v14/representative_20261008_b` **[fichiers directs]** | 0.033 | 2026-10-09T08:26:19.800682+00:00 | GARDER | V142 : 2 artefacts; récent `docs/retrieval_far_label_audit_20261015.json:26`; manifeste gardé : `$H/step2_diagnostic_20261017/smoke/plan.json` |
| `$H/v140` | 0.174 | 2026-10-14T00:04:13.062887+00:00 | GARDER | récent `handover.md:19334`; manifeste gardé : `$H/v143/local_evidence_empty_recovery_20261013_night/banatic_only/native/017ae987199a49a101758e9e4c69421ea66ad828511e1f76b14c17428f206878.seal.json` |
| `$H/v141` | 2.350 | 2026-10-13T23:12:40.958635+00:00 | GARDER | V142 : 7 artefacts; récent `docs/retrieval_frozen_v142.json:5326`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v142` | 0.324 | 2026-10-14T02:09:58.307526+00:00 | GARDER | V142 : 1587 artefacts; récent `docs/production_cost_study.json:969`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/result.json` |
| `$H/v143` | 0.558 | 2026-10-13T23:25:08.876247+00:00 | GARDER | V142 : 532 artefacts; récent `docs/retrieval_engine_explained_20261014.md:270`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v144` | 0.119 | 2026-10-14T00:07:29.867126+00:00 | GARDER | récent `handover.md:19512` |
| `$H/v145` | 0.859 | 2026-10-14T00:30:48.657024+00:00 | GARDER | récent `handover.md:19558` |
| `$H/v147` | 0.014 | 2026-10-14T01:50:14.621856+00:00 | GARDER | récent `handover.md:19606` |
| `$H/v149` | 0.001 | 2026-10-14T00:56:19.920973+00:00 | GARDER | récent `scripts/embeddinggemma2_v149_topad_metadata_audit.py:15` |
| `$H/v14` **[fichiers directs]** | 0.000 | 2026-10-08T17:08:51.711076+00:00 | GARDER | récent `docs/retrieval_far_label_audit_20261015.json:26` |
| `$H/v151` | 0.018 | 2026-10-14T01:10:54.519151+00:00 | GARDER | récent `scripts/embeddinggemma2_v151_bodacc_context_geo_dependence.py:23` |
| `$H/v152` | 0.000 | 2026-10-14T01:23:14.780282+00:00 | GARDER | récent `scripts/embeddinggemma2_v152_bodacc_structure_scope_audit.py:21` |
| `$H/v154` | 0.454 | 2026-10-14T02:02:09.896888+00:00 | GARDER | récent `docs/retrieval_bodacc_business_name_lineage_20261014.json:3` |
| `$H/v156` | 0.109 | 2026-10-14T02:26:42.842622+00:00 | GARDER | récent `docs/retrieval_v156_sigle_source_milestone.json:4` |
| `$H/v157` | 1.556 | 2026-10-14T03:17:42.668369+00:00 | GARDER | récent `docs/retrieval_v157_geography_component_result_20261014.json:73` |
| `$H/v158` | 0.150 | 2026-10-14T02:50:25.341228+00:00 | GARDER | récent `docs/retrieval_v158_corporate_administration_source_milestone.md:9` |
| `$H/v159` | 0.161 | 2026-10-14T03:35:37.874667+00:00 | GARDER | récent `docs/retrieval_temporal_lineage_direction_20261014.md:5` |
| `$H/v160` | 0.002 | 2026-10-14T03:56:07.180178+00:00 | GARDER | récent `docs/retrieval_v160_explain_final_c_stop_20261014.json:4` |
| `$H/v161` | 0.001 | 2026-10-14T03:48:00.612706+00:00 | GARDER | récent `docs/retrieval_v161_osm_commune_suffix_preflight.md:4` |
| `$H/v162` | 0.429 | 2026-10-14T04:01:42.335595+00:00 | GARDER | récent `docs/retrieval_v162_commune_suffix_native_result.md:8` |
| `$H/v16` | 0.001 | 2026-10-09T16:37:49.683285+00:00 | GARDER | récent `handover.md:6003` |
| `$H/v170` | 0.000 | 2026-10-14T04:24:02.667955+00:00 | GARDER | récent `docs/retrieval_v170_global_xy_quality_compatibility_result_20261014.json:64` |
| `$H/v171` | 2.207 | 2026-10-14T04:36:33.039938+00:00 | GARDER | récent `docs/retrieval_night_20261014_direction.md:73` |
| `$H/v172` | 0.165 | 2026-10-14T17:32:05.382819+00:00 | GARDER | récent `docs/retrieval_miss_score_explain_20261014.json:49` |
| `$H/v173` | 3.448 | 2026-10-15T21:39:25.531536+00:00 | GARDER | récent `docs/retrieval_site_target_full_result_20261016.json:22`; manifeste gardé : `$H/step2_option_d_20261017/full/result.json` |
| `$H/v174/train3000_frozen_v142_20261017/.train3000_frozen_v142_publications` | 0.000 | 2026-10-17T11:13:00.979129+00:00 | GARDER | récent `docs/production_cost_study.json:336` |
| `$H/v174/train3000_frozen_v142_20261017/extension_20261017` | 45.574 | 2026-10-19T10:14:27.887916+00:00 | GARDER | récent `docs/production_cost_study.json:336`; manifeste gardé : `$H/v173/site_target_timing_20261015/v71/result.json` |
| `$H/v174/train3000_frozen_v142_20261017` **[fichiers directs]** | 0.000 | 2026-10-17T11:26:41.650915+00:00 | GARDER | récent `docs/production_cost_study.json:336` |
| `$H/v174` **[fichiers directs]** | 0.000 | 2026-10-17T11:12:33.320379+00:00 | GARDER | récent `docs/production_cost_study.json:336` |
| `$H/v17` | 0.058 | 2026-10-10T20:46:20.760248+00:00 | GARDER | V142 : 1 artefacts; récent `docs/retrieval_far_label_audit_20261015.json:31`; manifeste gardé : `$H/step2_diagnostic_20261017/smoke/plan.json` |
| `$H/v18` | 6.108 | 2026-10-10T21:16:53.424307+00:00 | GARDER | récent `docs/train3000_cost_analysis.md:22` |
| `$H/v19` | 0.470 | 2026-10-11T18:31:04.114970+00:00 | GARDER | V142 : 1 artefacts; récent `docs/retrieval_frozen_v142.json:22346`; manifeste gardé : `$H/v173/site_target_timing_20261015/v71/result.json` |
| `$H/v20` | 0.021 | 2026-10-11T19:29:50.735075+00:00 | GARDER | récent `handover.md:4910` |
| `$H/v21` | 0.177 | 2026-10-11T19:53:01.710697+00:00 | GARDER | V142 : 4 artefacts; récent `docs/retrieval_frozen_v142.json:22354`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v22` | 2.612 | 2026-10-11T22:31:12.295947+00:00 | GARDER | récent `handover.md:4924` |
| `$H/v23` | 0.072 | 2026-10-11T23:12:41.363330+00:00 | GARDER | V142 : 4 artefacts; récent `docs/retrieval_frozen_v142.json:22386`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/numeric51_20261017_185032_c95c7c63/reconstruction/v23/features/ea3a03b4971eeb62730e1de238225cc4e6145f0eb50ad28b1379f2a2ee71e16e.seal.json` |
| `$H/v24` | 0.089 | 2026-10-12T00:32:33.632121+00:00 | GARDER | récent `handover.md:6379` |
| `$H/v25` | 0.049 | 2026-10-12T00:37:53.141531+00:00 | GARDER | récent `handover.md:17362` |
| `$H/v26` | 0.041 | 2026-10-12T00:41:37.116670+00:00 | GARDER | récent `handover.md:6476` |
| `$H/v27` | 0.048 | 2026-10-12T00:46:38.163383+00:00 | GARDER | récent `handover.md:6526`; manifeste gardé : `$H/v29/source_robust_native_20261012_a/calibration/a8302321e60791ae50456d85f1bb8b3ef92fbfb23a081da45ef468be922ae9b1.seal.json` |
| `$H/v28` | 0.049 | 2026-10-12T00:50:33.367497+00:00 | GARDER | récent `docs/production_cost_study.json:424` |
| `$H/v29` | 0.048 | 2026-10-12T01:08:05.820744+00:00 | GARDER | récent `handover.md:6688`; manifeste gardé : `$H/v27/nonlinear_native_20261012_a/calibration/a8302321e60791ae50456d85f1bb8b3ef92fbfb23a081da45ef468be922ae9b1.seal.json` |
| `$H/v2` | 0.345 | 2026-10-07T19:07:25.206502+00:00 | GARDER | récent `docs/label_noise_review_v2.json:3`; manifeste gardé : `$H/step2_option_d_20261017/smoke/result.json` |
| `$H/v30` | 0.050 | 2026-10-12T01:12:56.590509+00:00 | GARDER | récent `handover.md:6772` |
| `$H/v31` | 0.033 | 2026-10-12T01:23:47.466898+00:00 | GARDER | récent `handover.md:6918` |
| `$H/v32` | 0.054 | 2026-10-12T01:33:25.821537+00:00 | GARDER | V142 : 10 artefacts; récent `docs/retrieval_frozen_v142.json:22418`; manifeste gardé : `$H/v173/site_target_timing_20261015/v71/result.json` |
| `$H/v33` | 4.168 | 2026-10-12T02:11:03.546510+00:00 | GARDER | récent `docs/step2_error_families_g0.json:8997`; manifeste gardé : `$H/v36/global_cached_parent_dense_forder_20261012_a/query_vector_binding.json` |
| `$H/v34` | 0.000 | 2026-10-12T02:10:14.031349+00:00 | GARDER | récent `handover.md:48` |
| `$H/v35` | 0.065 | 2026-10-12T02:21:42.009140+00:00 | GARDER | récent `handover.md:56` |
| `$H/v36` | 4.184 | 2026-10-12T02:21:42.720536+00:00 | GARDER | récent `docs/step2_error_families_g0.json:8993`; manifeste gardé : `$H/v33/global_cached_parent_dense_20261012_b/query_vector_binding.json` |
| `$H/v37` | 4.169 | 2026-10-12T02:39:36.242599+00:00 | GARDER | récent `handover.md:7171` |
| `$H/v38` | 0.000 | 2026-10-12T03:24:06.105386+00:00 | GARDER | récent `docs/retrieval_scientific_validity_all526_20261014.json:3` |
| `$H/v39` | 0.000 | 2026-10-12T03:31:33.249280+00:00 | GARDER | récent `handover.md:18` |
| `$H/v3` | 0.816 | 2026-10-07T22:48:25.692841+00:00 | GARDER | récent `docs/step2_ranker_v3_group.md:1`; manifeste gardé : `$H/v4/night_20261008_b/data/manifest.json` |
| `$H/v40` | 0.000 | 2026-10-12T03:31:52.391955+00:00 | GARDER | récent `handover.md:7434` |
| `$H/v41` | 0.000 | 2026-10-12T03:37:19.732508+00:00 | GARDER | récent `handover.md:7500` |
| `$H/v42` | 0.053 | 2026-10-12T03:51:13.276947+00:00 | GARDER | récent `docs/production_cost_study.json:377` |
| `$H/v43` | 0.079 | 2026-10-12T03:54:58.669805+00:00 | GARDER | récent `handover.md:17724` |
| `$H/v44` | 0.000 | 2026-10-12T03:56:51.647391+00:00 | DOUTE | récent `handover.md:7647` |
| `$H/v45` | 0.027 | 2026-10-12T04:09:51.391819+00:00 | GARDER | récent `handover.md:17742` |
| `$H/v46` | 0.000 | 2026-10-12T04:41:42.696598+00:00 | DOUTE | récent `handover.md:7738` |
| `$H/v47` | 0.000 | 2026-10-12T04:29:37.388884+00:00 | DOUTE | récent `handover.md:7758` |
| `$H/v48` | 0.000 | 2026-10-12T04:52:25.980750+00:00 | DOUTE | récent `handover.md:7763` |
| `$H/v49` | 0.001 | 2026-10-12T10:00:56.703167+00:00 | GARDER | récent `scripts/embeddinggemma2_v49_postseal_bodacc_roles.py:22` |
| `$H/v4` | 0.081 | 2026-10-07T23:53:35.056980+00:00 | GARDER | récent `docs/step2_option_d.json:100`; manifeste gardé : `$H/step2_option_d_20261017/smoke/result.json` |
| `$H/v50` | 0.000 | 2026-10-12T10:03:36.309505+00:00 | DOUTE | récent `handover.md:7800` |
| `$H/v51` | 0.000 | 2026-10-12T17:16:50.345342+00:00 | GARDER | récent `handover.md:17889` |
| `$H/v52` | 0.000 | 2026-10-12T17:33:46.174416+00:00 | GARDER | récent `scripts/embeddinggemma2_v151_bodacc_context_geo_dependence.py:22` |
| `$H/v53` | 0.001 | 2026-10-12T17:41:26.054905+00:00 | DOUTE | récent `handover.md:7858` |
| `$H/v54` | 0.000 | 2026-10-12T17:45:14.339070+00:00 | DOUTE | récent `handover.md:7858` |
| `$H/v55` | 0.000 | 2026-10-12T17:54:38.846272+00:00 | DOUTE | récent `handover.md:7964` |
| `$H/v56` | 2.837 | 2026-10-12T21:55:03.607764+00:00 | GARDER | récent `handover.md:17982`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/numeric51_20261017_190721_767b3f3f/reconstruction/v57/features/8a851b82d947d76187136eedfdf49e03d5270311307a4919e29692bb5c45d051.seal.json` |
| `$H/v57` | 0.143 | 2026-10-12T21:58:15.792787+00:00 | GARDER | V142 : 3 artefacts; récent `docs/retrieval_frozen_v142.json:22498`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/numeric51_20261017_190721_767b3f3f/reconstruction/v57/features/ea3a03b4971eeb62730e1de238225cc4e6145f0eb50ad28b1379f2a2ee71e16e.seal.json` |
| `$H/v58` | 0.000 | 2026-10-12T22:01:42.863541+00:00 | GARDER | récent `handover.md:18036` |
| `$H/v59` | 0.000 | 2026-10-12T22:03:08.517001+00:00 | DOUTE | récent `handover.md:8078` |
| `$H/v5` | 0.030 | 2026-10-08T00:26:28.601441+00:00 | GARDER | récent `handover.md:952` |
| `$H/v60` | 1.002 | 2026-10-12T22:05:59.981786+00:00 | GARDER | récent `handover.md:18075` |
| `$H/v61` | 0.000 | 2026-10-12T22:09:31.099238+00:00 | DOUTE | récent `handover.md:8079` |
| `$H/v62` | 0.978 | 2026-10-12T22:11:26.290034+00:00 | GARDER | récent `handover.md:18086` |
| `$H/v63` | 0.000 | 2026-10-12T22:12:02.329884+00:00 | DOUTE | récent `handover.md:8142` |
| `$H/v64` | 0.000 | 2026-10-12T22:13:15.087787+00:00 | GARDER | récent `handover.md:18097` |
| `$H/v65` | 0.000 | 2026-10-12T22:21:57.520298+00:00 | GARDER | V142 : 2 artefacts; récent `docs/retrieval_frozen_v142.json:22530`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v66` | 0.000 | 2026-10-12T22:28:01.883512+00:00 | GARDER | récent `handover.md:18177` |
| `$H/v67` | 0.066 | 2026-10-12T22:32:22.014604+00:00 | GARDER | récent `scripts/embeddinggemma2_v68_filled_site_geo.py:26` |
| `$H/v68` | 0.050 | 2026-10-12T22:36:31.476353+00:00 | GARDER | V142 : 1 artefacts; récent `docs/retrieval_frozen_v142.json:22547`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/numeric51_20261017_190721_767b3f3f/reconstruction/v68/features/ea3a03b4971eeb62730e1de238225cc4e6145f0eb50ad28b1379f2a2ee71e16e.seal.json` |
| `$H/v69` | 0.005 | 2026-10-12T22:47:03.360146+00:00 | DOUTE | récent `handover.md:8301` |
| `$H/v6` | 0.002 | 2026-10-08T04:55:56.327542+00:00 | GARDER | V142 : 1 artefacts; récent `docs/retrieval_frozen_v142.json:22522`; manifeste gardé : `$H/v8/site_pipeline_20261008_a/v3_official512_binding.json` |
| `$H/v70` | 0.000 | 2026-10-12T22:52:07.118129+00:00 | GARDER | V142 : 1 artefacts; récent `docs/retrieval_frozen_v142.json:22555`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v71` | 0.069 | 2026-10-12T23:55:36.501893+00:00 | GARDER | V142 : 8 artefacts; récent `docs/production_cost_study.json:407`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/generate_0aad86471e72e47cb354bda0d79d0994eb0d6702f6dc57b674589b86a52caef4_b34b88c7c4/result.json` |
| `$H/v72` | 0.103 | 2026-10-13T00:09:23.245563+00:00 | GARDER | V142 : 2 artefacts; récent `docs/retrieval_frozen_v142.json:22628`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v73` | 0.075 | 2026-10-13T00:11:43.949051+00:00 | GARDER | récent `docs/production_cost_study.json:389` |
| `$H/v74` | 0.000 | 2026-10-13T00:13:01.040127+00:00 | GARDER | V142 : 2 artefacts; récent `docs/retrieval_frozen_v142.json:22644`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/covered-prepare_fc2a89a72cbd6bd79ce81de33f43bb485b265a6d4ee5b2370bfdfc0da97577ad_59fc77a558/numeric/v75/source_hint_receipt.json` |
| `$H/v75/published_supplier_context_20261013_a/calibration` | 0.000 | 2026-10-13T00:58:46.979133+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:22662` |
| `$H/v75/published_supplier_context_20261013_a/features` | 25.337 | 2026-10-13T00:22:48.147265+00:00 | GARDER | V142 : 526 artefacts; récent `docs/retrieval_frozen_v142.json:22662`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/numeric51_20261017_190721_767b3f3f/reconstruction/v71/features/ea3a03b4971eeb62730e1de238225cc4e6145f0eb50ad28b1379f2a2ee71e16e.seal.json` |
| `$H/v75/published_supplier_context_20261013_a/native` | 0.008 | 2026-10-13T02:06:26.508036+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:22662` |
| `$H/v75/published_supplier_context_20261013_a` **[fichiers directs]** | 0.018 | 2026-10-13T02:06:33.728237+00:00 | GARDER | V142 : 2 artefacts; récent `docs/retrieval_frozen_v142.json:22662`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v75` **[fichiers directs]** | 0.000 | 2026-10-13T00:18:17.886758+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:22662` |
| `$H/v76` | 0.002 | 2026-10-13T00:24:35.164579+00:00 | GARDER | récent `handover.md:18307` |
| `$H/v78` | 0.029 | 2026-10-13T00:38:05.749236+00:00 | GARDER | récent `handover.md:18343`; manifeste gardé : `$H/v71/frozen_residual_20261013_a/calibration/a8302321e60791ae50456d85f1bb8b3ef92fbfb23a081da45ef468be922ae9b1.seal.json` |
| `$H/v79` | 0.000 | 2026-10-13T00:42:51.263066+00:00 | GARDER | récent `handover.md:18358` |
| `$H/v7` | 0.000 | 2026-10-08T05:24:41.370346+00:00 | GARDER | récent `handover.md:897` |
| `$H/v80` | 0.000 | 2026-10-13T00:45:11.978661+00:00 | GARDER | récent `handover.md:18363` |
| `$H/v81` | 0.027 | 2026-10-13T00:58:59.065164+00:00 | GARDER | V142 : 3 artefacts; récent `docs/retrieval_frozen_v142.json:26887`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v83` | 0.000 | 2026-10-13T01:36:24.418605+00:00 | GARDER | récent `handover.md:18473` |
| `$H/v84` | 0.000 | 2026-10-13T02:16:37.303365+00:00 | GARDER | récent `handover.md:18495` |
| `$H/v85` | 0.001 | 2026-10-13T02:14:27.031715+00:00 | GARDER | récent `handover.md:18509` |
| `$H/v86` | 0.000 | 2026-10-13T02:20:24.453615+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$H/v87` | 0.202 | 2026-10-13T02:21:11.742975+00:00 | GARDER | manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v88` | 0.231 | 2026-10-13T02:23:09.016508+00:00 | GARDER | V142 : 3 artefacts; récent `docs/retrieval_frozen_v142.json:26912`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v89` | 0.045 | 2026-10-13T02:29:40.235068+00:00 | GARDER | récent `docs/production_cost_study.json:398`; manifeste gardé : `$H/v94/coherent_address_recovery_20261013_a/calibration/a8302321e60791ae50456d85f1bb8b3ef92fbfb23a081da45ef468be922ae9b1.seal.json` |
| `$H/v8` | 0.009 | 2026-10-08T05:32:12.374557+00:00 | GARDER | récent `handover.md:1001` |
| `$H/v90` | 0.000 | 2026-10-13T02:39:59.087654+00:00 | GARDER | récent `handover.md:18450` |
| `$H/v91` | 0.297 | 2026-10-13T02:46:15.546698+00:00 | DOUTE | récent `handover.md:9101` |
| `$H/v92` | 0.000 | 2026-10-13T02:48:10.467359+00:00 | GARDER | récent `handover.md:18562` |
| `$H/v93` | 0.030 | 2026-10-13T02:48:36.352479+00:00 | GARDER | récent `handover.md:18554` |
| `$H/v94` | 0.029 | 2026-10-13T02:57:38.552264+00:00 | GARDER | récent `handover.md:18527`; manifeste gardé : `$H/v78/nested_residual_offset_20261013_a/calibration/inner_baselines/outer_4_inner_3.seal.json` |
| `$H/v95` | 2.752 | 2026-10-13T03:00:15.657998+00:00 | GARDER | récent `handover.md:18537`; manifeste gardé : `$H/v78/nested_residual_offset_20261013_a/calibration/inner_baselines/outer_4_inner_3.seal.json` |
| `$H/v96` | 0.000 | 2026-10-13T04:06:43.273488+00:00 | DOUTE | récent `docs/retrieval_candidate_xy_quality_schema_audit_20261014.json:9` |
| `$H/v97` | 0.000 | 2026-10-13T03:22:42.170927+00:00 | DOUTE | récent `handover.md:9217` |
| `$H/v98` | 0.082 | 2026-10-13T03:23:50.455966+00:00 | GARDER | V142 : 3 artefacts; récent `docs/retrieval_frozen_v142.json:26936`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/covered-prepare_fc2a89a72cbd6bd79ce81de33f43bb485b265a6d4ee5b2370bfdfc0da97577ad_59fc77a558/collector_contexts/initials/v_commune_2026.seal.json` |
| `$H/v99` | 0.029 | 2026-10-13T03:30:32.428788+00:00 | GARDER | V142 : 4 artefacts; récent `docs/retrieval_frozen_v142.json:26960`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$H/v9` | 0.000 | 2026-10-08T06:25:22.294947+00:00 | GARDER | récent `docs/legacy_v2_routing_comparability.md:9` |
| `$H` **[fichiers directs]** | 0.153 | 2026-10-20T12:27:38.763926+00:00 | GARDER | récent `handover.md:16500` |
| `/Volumes/CATNAT_DATA/SIRETO_M3_PYTHON_TCB` | 0.000 | 2026-07-31T12:47:50.029528+00:00 | DOUTE | Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide. |
| `/Volumes/CATNAT_DATA/SIRETO_M3_TCB_TEST` | 0.000 | 2026-07-31T12:35:42.522141+00:00 | DOUTE | Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide. |
| `/Volumes/CATNAT_DATA/SIRETO_PPLX/hf_cache` | 2.233 | 2026-10-08T16:42:30.441615+00:00 | DOUTE | Cache : reconstruction hors ligne et absence de réutilisation non démontrées. |
| `/Volumes/CATNAT_DATA/SIRETO_PPLX/hf_home` | 0.000 | 2026-10-08T16:42:47.400519+00:00 | DOUTE | Cache : reconstruction hors ligne et absence de réutilisation non démontrées. |
| `/Volumes/CATNAT_DATA/SIRETO_PPLX/probe_20261008_a` | 0.000 | 2026-10-08T16:45:49.413309+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/SIRETO_PPLX/probe_20261008_b` | 0.000 | 2026-10-08T16:47:44.900286+00:00 | GARDER | récent `handover.md:16601` |
| `/Volumes/CATNAT_DATA/SIRETO_PPLX/vision_overlay` | 0.023 | 2026-10-08T16:46:56.230927+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `/Volumes/CATNAT_DATA/SIRETO_PPLX` **[fichiers directs]** | 0.000 | 2026-10-08T16:47:18.395154+00:00 | DOUTE | Fichiers directs/métadonnées du parent : pas une cible de suppression. |
| `$R/aborted` | 0.001 | 2026-08-24T11:55:06.588805+00:00 | DOUTE | Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide. |
| `$R/archives` | 0.000 | 2026-08-18T21:46:59.635604+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/audits/claude-opus-review.UbgXWB` | 0.000 | 2026-09-10T17:53:39.966442+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_fold_fused_national_oof_v14` | 6.351 | 2026-08-25T12:18:50.931725+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_fold_fused_physical_benchmark_v12_ca4c4f3` | 1.299 | 2026-08-24T17:24:35.697361+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v13_stop_receipt/005fb19823721480/manifest.json` |
| `$R/audits/retrieval_fold_fused_physical_benchmark_v13_w3` | 0.145 | 2026-08-24T21:13:38.592502+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v13_stop_receipt/005fb19823721480/manifest.json` |
| `$R/audits/retrieval_fold_fused_physical_benchmark_v14_w3` | 0.146 | 2026-08-25T06:36:23.325216+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_fold_fused_national_oof_v14/manifest.json` |
| `$R/audits/retrieval_fold_fused_prefit_parity_v12_7590ca67` | 0.111 | 2026-08-24T14:27:49.048926+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_fold_fused_physical_benchmark_v13_w3/trial_w3/worker_build/fold2/manifest.json` |
| `$R/audits/retrieval_geo_v16_evaluation` | 0.012 | 2026-08-26T00:24:14.742282+00:00 | GARDER | récent `src/xgb_matcher/retrieval_geo_v16.py:5` |
| `$R/audits/retrieval_geo_v16_generation` | 0.055 | 2026-08-26T00:23:47.903974+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v28_geo_features/1f604bd8a92e9fd9/manifest.json` |
| `$R/audits/retrieval_geo_v16_preflight` | 0.000 | 2026-08-26T00:23:10.349577+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_master_v15_feasibility_cohort` | 0.000 | 2026-08-25T23:35:59.012669+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v28_geo_features/1f604bd8a92e9fd9/manifest.json` |
| `$R/audits/retrieval_master_v15_feasibility_evaluation` | 0.000 | 2026-08-25T23:42:07.591882+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_master_v15_feasibility_generation` | 0.013 | 2026-08-25T23:41:46.650954+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_pair_semantic_parity` | 0.000 | 2026-10-04T10:43:41.174385+00:00 | GARDER | récent `handover.md:13360` |
| `$R/audits/retrieval_parent_v16_workbench_evaluation` | 0.000 | 2026-08-26T00:21:55.860581+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_parent_v16_workbench_generation` | 0.004 | 2026-08-26T00:21:17.099957+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_parent_v18_discovery_evaluation` | 0.000 | 2026-08-26T17:42:09.693008+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_parent_v18_discovery_generation` | 0.057 | 2026-08-26T17:40:20.513261+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v30_master/8cb4ad779f11ac84/manifest.json` |
| `$R/audits/retrieval_parent_v19_features` | 0.060 | 2026-08-26T22:16:53.402020+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_parent_v19_oof/9494ea0ed4992900/manifest.json` |
| `$R/audits/retrieval_parent_v19_oof` | 0.042 | 2026-08-26T22:19:19.093813+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_site_inventory_v17_evaluation` | 0.000 | 2026-08-26T10:29:29.190532+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_site_inventory_v17_generation` | 0.573 | 2026-08-26T10:28:59.028197+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v29_consolidated/134f18bd1bae893a/manifest.json` |
| `$R/audits/retrieval_v100_complete_word_metadata` | 2.725 | 2026-10-03T05:07:33.953207+00:00 | GARDER | récent `handover.md:9236` |
| `$R/audits/retrieval_v101_national_paths` | 0.006 | 2026-10-03T05:47:32.200687+00:00 | GARDER | récent `handover.md:9273` |
| `$R/audits/retrieval_v104_frozen_initialisms` | 0.000 | 2026-10-03T06:00:18.320759+00:00 | GARDER | récent `handover.md:9389` |
| `$R/audits/retrieval_v105_exact_intervals` | 0.001 | 2026-10-03T06:34:03.141324+00:00 | GARDER | récent `handover.md:9424` |
| `$R/audits/retrieval_v106_native_cost` | 0.000 | 2026-10-03T06:39:46.052850+00:00 | GARDER | récent `handover.md:9458` |
| `$R/audits/retrieval_v107_split_initial` | 0.000 | 2026-10-03T06:51:43.586623+00:00 | GARDER | récent `handover.md:9503` |
| `$R/audits/retrieval_v108_conditional_initial` | 0.001 | 2026-10-03T06:59:51.924603+00:00 | GARDER | récent `handover.md:9541` |
| `$R/audits/retrieval_v109_physical_pages` | 0.001 | 2026-10-03T07:07:51.236699+00:00 | GARDER | récent `handover.md:9561` |
| `$R/audits/retrieval_v110_official_context` | 0.532 | 2026-10-03T07:29:30.634637+00:00 | GARDER | récent `handover.md:9601`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v111_activity_decoder` | 0.009 | 2026-10-03T07:40:33.416703+00:00 | GARDER | récent `handover.md:9629` |
| `$R/audits/retrieval_v112_activity_catalog` | 0.124 | 2026-10-03T07:59:51.195722+00:00 | GARDER | récent `handover.md:9672`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v113_activity_semantic` | 0.005 | 2026-10-03T08:10:10.220549+00:00 | GARDER | récent `handover.md:9706`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v114_activity_joint` | 0.002 | 2026-10-03T08:49:45.129475+00:00 | GARDER | récent `handover.md:9764`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v115_activity_initial` | 0.000 | 2026-10-03T09:07:30.146780+00:00 | GARDER | récent `handover.md:9813`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v116_activity_fair` | 0.000 | 2026-10-03T09:13:34.523779+00:00 | GARDER | récent `handover.md:9828` |
| `$R/audits/retrieval_v118_official_variants` | 0.016 | 2026-10-03T09:52:28.260103+00:00 | GARDER | récent `handover.md:9923`; manifeste gardé : `$R/audits/retrieval_v119_variant_factorization/probe_20261003_a/manifest.json` |
| `$R/audits/retrieval_v119_variant_factorization` | 0.006 | 2026-10-03T10:00:42.711830+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v118_official_variants/probe_20261003_a/manifest.json` |
| `$R/audits/retrieval_v120_variant_hashjoin` | 0.006 | 2026-10-03T10:04:54.471822+00:00 | GARDER | récent `handover.md:9956`; manifeste gardé : `$R/audits/retrieval_v118_official_variants/probe_20261003_a/manifest.json` |
| `$R/audits/retrieval_v121_coupled_blocks` | 0.001 | 2026-10-03T11:07:49.420384+00:00 | GARDER | récent `handover.md:10024` |
| `$R/audits/retrieval_v122_coupled_source_ledger` | 0.002 | 2026-10-03T11:15:22.601394+00:00 | GARDER | récent `handover.md:10032`; manifeste gardé : `$R/audits/retrieval_v123_tight_blocks/probe_20261003_a/manifest.json` |
| `$R/audits/retrieval_v123_tight_blocks` | 0.002 | 2026-10-03T11:28:36.499259+00:00 | GARDER | récent `handover.md:10049`; manifeste gardé : `$R/audits/retrieval_v128_variant_batches/probe_20261003_a/manifest.json` |
| `$R/audits/retrieval_v124_vector_blocks` | 0.002 | 2026-10-03T11:34:05.057676+00:00 | GARDER | récent `handover.md:10065`; manifeste gardé : `$R/audits/retrieval_v123_tight_blocks/probe_20261003_a/manifest.json` |
| `$R/audits/retrieval_v125_mapped_blocks` | 0.022 | 2026-10-03T11:52:15.569324+00:00 | GARDER | récent `handover.md:10114`; manifeste gardé : `$R/audits/retrieval_v123_tight_blocks/probe_20261003_a/manifest.json` |
| `$R/audits/retrieval_v126_national_layout` | 0.830 | 2026-10-03T12:03:53.364334+00:00 | GARDER | V142 : 1 artefacts; récent `docs/retrieval_candidate_xy_quality_schema_audit_20261014.json:6`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$R/audits/retrieval_v127_owner_blocks` | 0.532 | 2026-10-03T12:19:39.353812+00:00 | GARDER | récent `handover.md:10192`; manifeste gardé : `$R/audits/retrieval_v136_extended_catalog/compile_20261003_a/manifest.json` |
| `$R/audits/retrieval_v128_variant_batches` | 0.005 | 2026-10-03T12:45:03.543127+00:00 | GARDER | récent `handover.md:10206` |
| `$R/audits/retrieval_v129_national_variants/projection_20261003_a` | 14.244 | 2026-10-03T13:45:04.534094+00:00 | GARDER | récent `handover.md:8920`; manifeste gardé : `$R/audits/retrieval_v130_variant_dictionary/compile_20261003_a/plan.json` |
| `$R/audits/retrieval_v129_national_variants` **[fichiers directs]** | 0.000 | 2026-10-03T13:12:53.196016+00:00 | GARDER | récent `handover.md:10233` |
| `$R/audits/retrieval_v12_reader_parity_b18e4e75` | 0.000 | 2026-08-24T12:17:33.761680+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v130_variant_dictionary` | 7.591 | 2026-10-03T14:10:37.081743+00:00 | GARDER | récent `handover.md:10358`; manifeste gardé : `$H/v14/full_validation_20261008_a/names/binding.json` |
| `$R/audits/retrieval_v131_name_access` | 7.908 | 2026-10-03T15:32:10.871963+00:00 | GARDER | récent `handover.md:10436`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v132_name_owners` | 1.905 | 2026-10-03T15:46:39.606474+00:00 | GARDER | récent `handover.md:10453`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v133_all_succession` | 0.675 | 2026-10-03T15:54:57.429483+00:00 | GARDER | récent `handover.md:10476`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v134_packed_names/compile_20261003_a` | 9.449 | 2026-10-03T16:40:35.184605+00:00 | GARDER | récent `docs/retrieval_candidate_xy_quality_schema_audit_20261014.json:6`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v134_packed_names` **[fichiers directs]** | 0.000 | 2026-10-03T16:14:55.750733+00:00 | GARDER | récent `handover.md:10525` |
| `$R/audits/retrieval_v135_catalog_inventory` | 0.006 | 2026-10-03T16:33:47.490899+00:00 | GARDER | récent `handover.md:10555`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v136_extended_catalog` | 0.012 | 2026-10-03T17:16:36.228345+00:00 | GARDER | récent `handover.md:10602`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v137_proof_bridge` | 1.140 | 2026-10-03T17:38:58.655988+00:00 | GARDER | récent `handover.md:10634`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v138_national_names` | 0.000 | 2026-10-03T17:53:52.716542+00:00 | GARDER | récent `handover.md:10669` |
| `$R/audits/retrieval_v139_site_attributes` | 0.491 | 2026-10-03T18:18:50.263826+00:00 | GARDER | récent `handover.md:10716`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v13_immutable_projections` | 0.000 | 2026-08-24T21:02:11.084211+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v13_immutable_w1_io_projection` | 0.000 | 2026-08-24T18:36:48.501383+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v13_stop_receipt` | 0.000 | 2026-08-24T21:37:02.919400+00:00 | GARDER | récent `handover.md:208`; manifeste gardé : `$R/audits/retrieval_fold_fused_physical_benchmark_v14_w3/manifest.json` |
| `$R/audits/retrieval_v140_canonical_inventory` | 0.001 | 2026-10-03T18:38:56.233247+00:00 | GARDER | récent `handover.md:10746`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v141_append_catalog` | 0.003 | 2026-10-03T19:04:25.636601+00:00 | GARDER | récent `handover.md:10782`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v142_address_projection` | 1.312 | 2026-10-03T19:25:40.544024+00:00 | GARDER | récent `handover.md:10844`; manifeste gardé : `$R/audits/retrieval_v144_address_bootstrap/projection_20261003_a/plan.json` |
| `$R/audits/retrieval_v143_address_resume` | 0.000 | 2026-10-03T19:48:06.486994+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v144_address_bootstrap` | 0.590 | 2026-10-03T19:55:49.059294+00:00 | GARDER | récent `handover.md:10926`; manifeste gardé : `$R/audits/retrieval_v153_address_resume/projection_20261004_a/plan.json` |
| `$R/audits/retrieval_v149_name_rebind` | 0.001 | 2026-10-03T21:37:16.483271+00:00 | GARDER | récent `handover.md:11158`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v14_immutable_projections` | 0.000 | 2026-08-25T06:17:58.785882+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_fold_fused_physical_benchmark_v14_w3/logical_io/f7e52ec59e1947f4/manifest.json` |
| `$R/audits/retrieval_v150_rebound_names` | 0.000 | 2026-10-03T21:57:34.507216+00:00 | GARDER | récent `handover.md:11205` |
| `$R/audits/retrieval_v151_name_pages` | 0.000 | 2026-10-03T22:11:47.231145+00:00 | GARDER | récent `handover.md:11239` |
| `$R/audits/retrieval_v153_address_resume` | 4.436 | 2026-10-03T23:16:29.522586+00:00 | GARDER | récent `handover.md:11415`; manifeste gardé : `$R/audits/retrieval_v154_address_store/compile_20261004_a/plan.json` |
| `$R/audits/retrieval_v154_address_store` | 6.227 | 2026-10-03T23:58:59.153094+00:00 | GARDER | récent `handover.md:11583`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v157_address_resume` | 0.892 | 2026-10-03T23:49:32.510604+00:00 | GARDER | récent `handover.md:11551`; manifeste gardé : `$R/audits/retrieval_v153_address_resume/projection_20261004_a/current_addresses/part00031/manifest.json` |
| `$R/audits/retrieval_v159_address_stage` | 0.028 | 2026-10-04T00:32:14.421391+00:00 | GARDER | récent `handover.md:11656`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v160_reverse_resume` | 6.132 | 2026-10-04T00:47:14.002490+00:00 | GARDER | récent `handover.md:11718`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v162_address_access` | 0.000 | 2026-10-04T01:45:53.627037+00:00 | GARDER | récent `handover.md:11795` |
| `$R/audits/retrieval_v163_address_access` | 0.061 | 2026-10-04T01:54:04.641068+00:00 | GARDER | récent `handover.md:11823`; manifeste gardé : `$R/audits/retrieval_v166_address_access/compile_20261004_a/plan.json` |
| `$R/audits/retrieval_v164_address_channels` | 0.014 | 2026-10-04T02:06:07.720296+00:00 | GARDER | récent `handover.md:11866`; manifeste gardé : `$R/audits/retrieval_v166_address_access/compile_20261004_a/parts/part00000/manifest.json` |
| `$R/audits/retrieval_v165_address_access` | 0.064 | 2026-10-04T02:22:21.295547+00:00 | GARDER | récent `handover.md:11940`; manifeste gardé : `$R/audits/retrieval_v166_address_access/compile_20261004_a/parts/part00000/manifest.json` |
| `$R/audits/retrieval_v166_address_access` | 0.721 | 2026-10-04T02:43:39.133752+00:00 | GARDER | récent `handover.md:11999`; manifeste gardé : `$R/audits/retrieval_v176_packed_addresses/compile_20261004_a/plan.json` |
| `$R/audits/retrieval_v167_address_access` | 0.024 | 2026-10-04T03:16:48.309381+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v168_address_access` | 7.429 | 2026-10-04T03:20:06.538244+00:00 | GARDER | récent `handover.md:12101`; manifeste gardé : `$R/audits/retrieval_v176_packed_addresses/compile_20261004_a/plan.json` |
| `$R/audits/retrieval_v169_paged_names` | 1.050 | 2026-10-04T03:45:11.557533+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v170_paged_names` | 0.000 | 2026-10-04T03:54:42.760874+00:00 | GARDER | récent `handover.md:12196` |
| `$R/audits/retrieval_v171_paged_names` | 0.000 | 2026-10-04T04:26:05.371125+00:00 | GARDER | récent `handover.md:12242` |
| `$R/audits/retrieval_v172_direct_alias` | 0.000 | 2026-10-04T04:32:54.609001+00:00 | GARDER | récent `handover.md:12263` |
| `$R/audits/retrieval_v173_direct_alias` | 0.000 | 2026-10-04T04:36:31.042291+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v174_direct_alias` | 0.000 | 2026-10-04T04:40:25.680815+00:00 | GARDER | récent `handover.md:12310` |
| `$R/audits/retrieval_v175_address_seal` | 0.058 | 2026-10-04T05:00:38.612831+00:00 | GARDER | récent `handover.md:12340`; manifeste gardé : `$R/audits/retrieval_v176_packed_addresses/compile_20261004_a/plan.json` |
| `$R/audits/retrieval_v176_packed_addresses` | 0.123 | 2026-10-04T05:43:10.420283+00:00 | GARDER | récent `handover.md:12413`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v177_packed_addresses/compile_20261004_a` | 10.819 | 2026-10-04T06:29:36.519167+00:00 | GARDER | récent `handover.md:11583`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v177_packed_addresses` **[fichiers directs]** | 0.000 | 2026-10-04T05:48:13.044808+00:00 | GARDER | récent `handover.md:12446` |
| `$R/audits/retrieval_v179_paged_addresses` | 0.000 | 2026-10-04T06:31:52.123672+00:00 | GARDER | récent `handover.md:12566` |
| `$R/audits/retrieval_v184_staged_initial` | 0.652 | 2026-10-04T07:28:04.594758+00:00 | GARDER | récent `handover.md:12656`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v185_staged_initial` | 0.000 | 2026-10-04T07:38:02.375839+00:00 | GARDER | récent `handover.md:12708` |
| `$R/audits/retrieval_v186_staged_initial` | 0.000 | 2026-10-04T07:47:27.168509+00:00 | GARDER | récent `handover.md:12729` |
| `$R/audits/retrieval_v187_staged_initial` | 0.249 | 2026-10-04T08:17:03.248406+00:00 | GARDER | récent `handover.md:12788`; manifeste gardé : `$R/audits/retrieval_v191_early_catalog/probe_20261004_a/manifest.json` |
| `$R/audits/retrieval_v188_compact_merge` | 0.000 | 2026-10-04T08:31:21.125156+00:00 | GARDER | récent `handover.md:12830` |
| `$R/audits/retrieval_v189_stream_merge` | 0.000 | 2026-10-04T08:37:16.682177+00:00 | GARDER | récent `handover.md:12860` |
| `$R/audits/retrieval_v190_staged_merge` | 0.021 | 2026-10-04T08:46:57.169734+00:00 | GARDER | récent `handover.md:12926` |
| `$R/audits/retrieval_v191_early_catalog` | 0.021 | 2026-10-04T08:51:14.045733+00:00 | GARDER | récent `handover.md:12959` |
| `$R/audits/retrieval_v193_names_pair` | 0.745 | 2026-10-04T09:14:59.805008+00:00 | GARDER | récent `handover.md:13050` |
| `$R/audits/retrieval_v194_addresses_pair` | 0.547 | 2026-10-04T09:35:59.695016+00:00 | GARDER | récent `handover.md:13126` |
| `$R/audits/retrieval_v195_pair_initial` | 0.174 | 2026-10-04T09:59:17.454381+00:00 | GARDER | récent `handover.md:13212`; manifeste gardé : `$R/audits/retrieval_v196_pair_initial/probe_20261004_a/manifest.json` |
| `$R/audits/retrieval_v196_pair_initial` | 0.174 | 2026-10-04T10:32:24.103953+00:00 | GARDER | récent `handover.md:13326`; manifeste gardé : `$R/audits/retrieval_v195_pair_initial/probe_20261004_a/manifest.json` |
| `$R/audits/retrieval_v197_population_initial` | 3.645 | 2026-10-04T11:21:28.976215+00:00 | GARDER | récent `handover.md:13428` |
| `$R/audits/retrieval_v198_names_ram` | 0.373 | 2026-10-04T12:00:56.883402+00:00 | GARDER | récent `handover.md:13515` |
| `$R/audits/retrieval_v199_population_initial` | 4.588 | 2026-10-04T12:10:46.739174+00:00 | GARDER | récent `handover.md:13581` |
| `$R/audits/retrieval_v201_names_ram` | 0.373 | 2026-10-04T12:30:57.330275+00:00 | GARDER | récent `handover.md:13661` |
| `$R/audits/retrieval_v202_population_initial` | 3.132 | 2026-10-04T12:40:24.858786+00:00 | GARDER | récent `handover.md:13724` |
| `$R/audits/retrieval_v205_names_ram` | 0.373 | 2026-10-04T13:35:57.847838+00:00 | GARDER | récent `handover.md:13847` |
| `$R/audits/retrieval_v206_population_initial/run_20261004_a` | 10.464 | 2026-10-04T14:03:09.528679+00:00 | GARDER | récent `handover.md:13428` |
| `$R/audits/retrieval_v206_population_initial` **[fichiers directs]** | 0.000 | 2026-10-04T13:41:12.988137+00:00 | GARDER | récent `handover.md:13896` |
| `$R/audits/retrieval_v208_numeric_storage` | 0.000 | 2026-10-04T14:21:22.937718+00:00 | GARDER | récent `handover.md:13993` |
| `$R/audits/retrieval_v209_numeric_storage` | 0.000 | 2026-10-04T14:35:58.966127+00:00 | GARDER | récent `handover.md:14047` |
| `$R/audits/retrieval_v20_master_eval` | 0.000 | 2026-09-09T16:33:36.815716+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v24_evaluation/78d37b7cdebea46b/manifest.json` |
| `$R/audits/retrieval_v20_master` | 0.170 | 2026-09-09T16:30:57.994438+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v21_membership/177f6e1c35b10f85-canonical-external/manifest.json` |
| `$R/audits/retrieval_v20_sampling` | 0.007 | 2026-09-09T16:30:19.205387+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v210_structural_channels` | 0.000 | 2026-10-04T15:31:35.681316+00:00 | GARDER | récent `handover.md:14116` |
| `$R/audits/retrieval_v211_structural_channels` | 0.000 | 2026-10-04T15:38:20.105980+00:00 | GARDER | récent `handover.md:14163` |
| `$R/audits/retrieval_v212_structural_population` | 0.000 | 2026-10-04T15:59:47.440108+00:00 | GARDER | récent `handover.md:14205` |
| `$R/audits/retrieval_v214_structural_population` | 0.011 | 2026-10-04T16:13:39.342667+00:00 | GARDER | récent `handover.md:14295`; manifeste gardé : `$R/audits/retrieval_v215_postseal_evaluation/run_20261004_a/manifest.json` |
| `$R/audits/retrieval_v215_postseal_evaluation` | 0.000 | 2026-10-04T17:03:19.067679+00:00 | GARDER | récent `handover.md:14320` |
| `$R/audits/retrieval_v216_loss_evidence` | 0.000 | 2026-10-04T17:29:50.976466+00:00 | GARDER | récent `handover.md:14394` |
| `$R/audits/retrieval_v217_corroborated_population` | 0.009 | 2026-10-04T17:50:18.721657+00:00 | GARDER | récent `handover.md:14435`; manifeste gardé : `$R/audits/retrieval_v218_postseal_evaluation/run_20261004_a/manifest.json` |
| `$R/audits/retrieval_v218_postseal_evaluation` | 0.000 | 2026-10-04T18:07:06.451721+00:00 | GARDER | récent `handover.md:14436` |
| `$R/audits/retrieval_v219_score_diagnostic` | 0.001 | 2026-10-04T18:34:22.029278+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v21_membership_eval` | 0.000 | 2026-09-09T17:57:33.677899+00:00 | GARDER | récent `handover.md:6237`; manifeste gardé : `$R/audits/retrieval_v22_global_evaluation/f9a73fa4db0ce00e/manifest.json` |
| `$R/audits/retrieval_v21_membership_preflight` | 0.009 | 2026-09-09T17:56:43.621653+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v21_membership` | 0.176 | 2026-09-09T17:57:05.134857+00:00 | GARDER | récent `handover.md:6209`; manifeste gardé : `$R/audits/retrieval_v22_global_generation/f9a73fa4db0ce00e/manifest.json` |
| `$R/audits/retrieval_v220_corroborated_population` | 0.000 | 2026-10-04T18:41:43.935443+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v222_corroborated_population` | 0.000 | 2026-10-04T18:45:16.559642+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v224_corroborated_population` | 0.009 | 2026-10-04T18:51:04.153201+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v225_postseal_evaluation/run_20261004_a/manifest.json` |
| `$R/audits/retrieval_v225_postseal_evaluation` | 0.000 | 2026-10-04T19:07:17.992590+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v226_training_geography` | 0.000 | 2026-10-04T19:09:09.514074+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v227_national_training_negatives` | 0.007 | 2026-10-04T20:44:06.317627+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v228_national_place_training` | 0.000 | 2026-10-04T20:52:24.837110+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v229_corroborated_population` | 0.009 | 2026-10-04T20:55:43.799038+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v230_postseal_evaluation/run_20261004_a/manifest.json` |
| `$R/audits/retrieval_v22_global_evaluation` | 0.000 | 2026-09-09T18:10:55.085366+00:00 | GARDER | récent `handover.md:6284`; manifeste gardé : `$R/audits/retrieval_v23_evaluation/bb12058773f80d6f/manifest.json` |
| `$R/audits/retrieval_v22_global_generation` | 0.056 | 2026-09-09T18:10:37.344827+00:00 | GARDER | récent `handover.md:6282`; manifeste gardé : `$R/audits/retrieval_v22_global_evaluation/f9a73fa4db0ce00e/manifest.json` |
| `$R/audits/retrieval_v22_global_preflight` | 0.001 | 2026-09-09T18:10:21.120441+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v230_postseal_evaluation` | 0.000 | 2026-10-04T21:13:31.542125+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v232_actual_train_features` | 0.010 | 2026-10-04T21:18:17.808397+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v233_actual_initial_training` | 0.000 | 2026-10-04T21:29:55.147474+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v234_corroborated_population` | 0.009 | 2026-10-04T21:41:02.175653+00:00 | GARDER | récent `handover.md:14724`; manifeste gardé : `$R/audits/retrieval_v235_postseal_evaluation/run_20261004_a/manifest.json` |
| `$R/audits/retrieval_v235_postseal_evaluation` | 0.000 | 2026-10-04T22:08:27.473703+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v237_corroborated_population` | 0.010 | 2026-10-04T22:18:06.558460+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v238_postseal_evaluation/run_20261005_a/manifest.json` |
| `$R/audits/retrieval_v238_postseal_evaluation` | 0.001 | 2026-10-04T22:37:41.615460+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v23_evaluation` | 0.000 | 2026-09-09T18:29:38.225734+00:00 | GARDER | récent `handover.md:6347`; manifeste gardé : `$R/audits/retrieval_v24_evaluation/78d37b7cdebea46b/manifest.json` |
| `$R/audits/retrieval_v23_features` | 0.139 | 2026-09-09T18:28:20.214029+00:00 | GARDER | récent `handover.md:6343`; manifeste gardé : `$R/audits/retrieval_v30_master/8cb4ad779f11ac84/manifest.json` |
| `$R/audits/retrieval_v23_oof_master` | 0.140 | 2026-09-09T18:29:01.321611+00:00 | GARDER | récent `handover.md:6345`; manifeste gardé : `$R/audits/retrieval_v24_generation/78d37b7cdebea46b/manifest.json` |
| `$R/audits/retrieval_v23_preflight` | 0.002 | 2026-09-09T18:27:24.147206+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v30_master/preflight16/manifest.json` |
| `$R/audits/retrieval_v240_actual_initial_training` | 0.000 | 2026-10-04T22:39:17.652484+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v241_corroborated_population` | 0.009 | 2026-10-04T22:47:48.745624+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v242_postseal_evaluation/run_20261005_a/manifest.json` |
| `$R/audits/retrieval_v242_postseal_evaluation` | 0.000 | 2026-10-04T23:04:55.743312+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v244_corroborated_population` | 0.010 | 2026-10-04T23:07:24.218846+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v245_postseal_evaluation/run_20261005_a/manifest.json` |
| `$R/audits/retrieval_v245_postseal_evaluation` | 0.001 | 2026-10-04T23:27:16.671618+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v248_corroborated_population` | 0.004 | 2026-10-04T23:30:30.063416+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v24_evaluation` | 0.000 | 2026-09-09T19:05:49.765965+00:00 | GARDER | récent `handover.md:6395`; manifeste gardé : `$R/audits/retrieval_v24_independent_audit/78d37b7cdebea46b/manifest.json` |
| `$R/audits/retrieval_v24_generation` | 0.098 | 2026-09-09T19:05:28.058171+00:00 | GARDER | récent `handover.md:6393`; manifeste gardé : `$R/audits/retrieval_v24_evaluation/78d37b7cdebea46b/manifest.json` |
| `$R/audits/retrieval_v24_independent_audit` | 0.000 | 2026-09-09T19:08:21.087213+00:00 | GARDER | récent `handover.md:6397` |
| `$R/audits/retrieval_v24_preflight` | 0.000 | 2026-09-09T19:05:10.210309+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v251_actual_train_features` | 0.010 | 2026-10-04T23:47:06.015976+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v252_actual_initial_training` | 0.000 | 2026-10-04T23:53:32.042121+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v254_corroborated_population` | 0.009 | 2026-10-04T23:57:58.178875+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v255_postseal_evaluation/run_20261005_a/manifest.json` |
| `$R/audits/retrieval_v255_postseal_evaluation` | 0.000 | 2026-10-05T00:15:37.591809+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v258_corroborated_population` | 0.009 | 2026-10-05T00:18:17.590599+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v259_postseal_evaluation/run_20261005_a/manifest.json` |
| `$R/audits/retrieval_v259_postseal_evaluation` | 0.000 | 2026-10-05T00:36:24.546200+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v25_conjunctive_evaluation` | 0.000 | 2026-09-09T20:50:54.128592+00:00 | GARDER | récent `handover.md:6460`; manifeste gardé : `$R/audits/retrieval_v25_result_audit/2c11d2553743ea1e/manifest.json` |
| `$R/audits/retrieval_v25_conjunctive_generation` | 0.003 | 2026-09-09T20:48:39.342341+00:00 | GARDER | récent `handover.md:6458`; manifeste gardé : `$R/audits/retrieval_v25_result_audit/2c11d2553743ea1e/manifest.json` |
| `$R/audits/retrieval_v25_conjunctive_preflight` | 0.000 | 2026-09-09T20:48:01.337786+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v30_discovery/preflight16/manifest.json` |
| `$R/audits/retrieval_v25_discovery_audit` | 0.000 | 2026-09-09T20:38:06.234901+00:00 | GARDER | récent `handover.md:6414` |
| `$R/audits/retrieval_v25_result_audit` | 0.000 | 2026-09-09T20:53:01.480885+00:00 | GARDER | récent `handover.md:6462` |
| `$R/audits/retrieval_v261_corroborated_population` | 0.009 | 2026-10-05T00:39:58.009762+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v262_postseal_evaluation/run_20261005_a/manifest.json` |
| `$R/audits/retrieval_v262_postseal_evaluation` | 0.000 | 2026-10-05T00:56:21.113873+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v265_corroborated_population` | 0.000 | 2026-10-05T00:58:16.641072+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v267_official_communes_snapshot` | 0.003 | 2026-10-05T00:38:33.344342+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v279_municipal_structure/compile_20261005_a/manifest.json` |
| `$R/audits/retrieval_v26_geo` | 0.001 | 2026-09-10T09:53:55.375237+00:00 | GARDER | récent `handover.md:6511`; manifeste gardé : `$R/audits/retrieval_v27_geo/preflight16-final/manifest.json` |
| `$R/audits/retrieval_v26_names` | 0.000 | 2026-09-10T09:54:05.568009+00:00 | GARDER | récent `handover.md:6513`; manifeste gardé : `$R/audits/retrieval_v27_names/f6f0ed019069defc/manifest.json` |
| `$R/audits/retrieval_v270_corroborated_population` | 0.005 | 2026-10-05T00:59:09.352703+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v273_corroborated_population` | 0.000 | 2026-10-05T01:13:24.456838+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v277_corroborated_population` | 0.009 | 2026-10-05T01:14:24.681412+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v278_postseal_evaluation/run_20261005_a/manifest.json` |
| `$R/audits/retrieval_v278_postseal_evaluation` | 0.000 | 2026-10-05T01:32:50.600980+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v279_municipal_structure` | 0.003 | 2026-10-05T01:41:07.942086+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v27_geo` | 0.019 | 2026-09-10T16:04:25.214619+00:00 | GARDER | récent `handover.md:6566`; manifeste gardé : `$R/audits/retrieval_v28_geo_features/1f604bd8a92e9fd9/manifest.json` |
| `$R/audits/retrieval_v27_names` | 0.001 | 2026-09-10T16:03:15.402168+00:00 | GARDER | récent `handover.md:6573` |
| `$R/audits/retrieval_v280_public_role_training` | 0.000 | 2026-10-05T01:47:16.139497+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v281_official_region_snapshots` | 0.000 | 2026-10-05T01:56:52.373302+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v283_corroborated_population` | 0.000 | 2026-10-05T01:53:24.019299+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v287_corroborated_population` | 0.009 | 2026-10-05T02:03:31.917283+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v288_postseal_evaluation/run_20261005_a/manifest.json` |
| `$R/audits/retrieval_v288_postseal_evaluation` | 0.000 | 2026-10-05T02:21:02.971318+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v28_geo_features` | 0.139 | 2026-09-10T17:21:18.763953+00:00 | GARDER | récent `handover.md:6630`; manifeste gardé : `$R/audits/retrieval_v33_full_replay/fast/parity16/shard00000/manifest.json` |
| `$R/audits/retrieval_v28_geo_oof` | 0.066 | 2026-09-10T17:26:47.422136+00:00 | GARDER | récent `handover.md:6632`; manifeste gardé : `$R/audits/retrieval_v29_consolidated/134f18bd1bae893a/manifest.json` |
| `$R/audits/retrieval_v290_corroborated_population` | 0.009 | 2026-10-05T02:24:22.636020+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v291_postseal_evaluation/run_20261005_a/manifest.json` |
| `$R/audits/retrieval_v291_postseal_evaluation` | 0.001 | 2026-10-05T02:42:56.328111+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v293_corroborated_population` | 0.003 | 2026-10-05T02:54:25.375052+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v297_corroborated_population` | 0.003 | 2026-10-05T03:02:16.336079+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v299_parent_site_diagnostic` | 0.000 | 2026-10-05T02:46:35.996667+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v29_consolidated` | 0.409 | 2026-09-10T18:49:46.856121+00:00 | GARDER | récent `handover.md:6701`; manifeste gardé : `$R/audits/retrieval_v29_evaluation/134f18bd1bae893a/manifest.json` |
| `$R/audits/retrieval_v29_evaluation` | 0.000 | 2026-09-10T18:52:38.077065+00:00 | GARDER | récent `handover.md:6733`; manifeste gardé : `$R/audits/retrieval_v29_verification/134f18bd1bae893a/manifest.json` |
| `$R/audits/retrieval_v29_verification` | 0.000 | 2026-09-10T18:53:15.837574+00:00 | GARDER | récent `handover.md:6735` |
| `$R/audits/retrieval_v301_corroborated_population` | 0.009 | 2026-10-05T03:10:52.569233+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v302_postseal_evaluation/run_20261005_a/manifest.json` |
| `$R/audits/retrieval_v302_postseal_evaluation` | 0.001 | 2026-10-05T03:29:36.535029+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v303_data_es_source` | 0.125 | 2026-10-05T03:32:14.976701+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v306_corroborated_population` | 0.009 | 2026-10-05T05:44:50.022375+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v307_postseal_evaluation/run_20261005_a/manifest.json` |
| `$R/audits/retrieval_v307_postseal_evaluation` | 0.000 | 2026-10-05T06:02:50.337445+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v308_data_es_source` | 0.125 | 2026-10-05T05:34:05.954900+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v30_discovery` | 0.007 | 2026-09-10T20:40:57.819137+00:00 | GARDER | récent `handover.md:6784`; manifeste gardé : `$R/audits/retrieval_v30_master/8cb4ad779f11ac84/manifest.json` |
| `$R/audits/retrieval_v30_evaluation` | 0.000 | 2026-09-10T20:46:34.355332+00:00 | GARDER | récent `handover.md:6815`; manifeste gardé : `$R/audits/retrieval_v30_verification/8cb4ad779f11ac84/manifest.json` |
| `$R/audits/retrieval_v30_master` | 0.418 | 2026-09-10T20:45:05.197140+00:00 | GARDER | récent `handover.md:6814`; manifeste gardé : `$R/audits/retrieval_v30_verification/8cb4ad779f11ac84/manifest.json` |
| `$R/audits/retrieval_v30_verification` | 0.000 | 2026-09-10T20:47:09.407860+00:00 | GARDER | récent `handover.md:6815` |
| `$R/audits/retrieval_v310_culture_source` | 0.089 | 2026-10-05T03:52:00.108182+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v312_culture_availability` | 0.000 | 2026-10-05T03:57:18.484677+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v314_corroborated_population` | 0.001 | 2026-10-05T04:06:00.914101+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v317_corroborated_population` | 0.002 | 2026-10-05T05:36:32.279171+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v319_data_es_availability` | 0.000 | 2026-10-05T05:42:38.685240+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v31_replay_preflight` | 0.010 | 2026-09-10T22:14:45.694454+00:00 | GARDER | récent `handover.md:6882`; manifeste gardé : `$R/audits/retrieval_v35_site_fusion/pilot_20261001_a/manifest.json` |
| `$R/audits/retrieval_v320_bodacc_relations` | 0.335 | 2026-10-05T06:04:40.248002+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v322_large_panel` | 0.000 | 2026-10-05T13:35:39.248469+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v323_corroborated_population` | 0.001 | 2026-10-05T18:41:55.517100+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v328_serving_panel` | 0.006 | 2026-10-05T18:31:47.876975+00:00 | GARDER | récent `docs/step2_option_d.json:86`; manifeste gardé : `$H/step2_option_d_20261017/smoke/result.json` |
| `$R/audits/retrieval_v32_stability` | 0.010 | 2026-09-11T08:25:15.421725+00:00 | GARDER | récent `handover.md:6914`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/prepared/manifest.json` |
| `$R/audits/retrieval_v330_corroborated_population` | 0.002 | 2026-10-05T19:51:00.850847+00:00 | GARDER | récent `handover.md:15514` |
| `$R/audits/retrieval_v333_corroborated_population` | 0.005 | 2026-10-05T21:40:05.968928+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v336_persistent_launcher` | 0.000 | 2026-10-05T21:33:09.811360+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v337_persistent_launcher` | 0.000 | 2026-10-05T21:40:02.462574+00:00 | GARDER | récent `handover.md:15566` |
| `$R/audits/retrieval_v33_full_replay` | 0.316 | 2026-09-11T09:18:43.859035+00:00 | GARDER | récent `docs/step2_option_d.json:145`; manifeste gardé : `$H/step2_option_d_20261017/smoke/result.json` |
| `$R/audits/retrieval_v344_corroborated_population` | 0.523 | 2026-10-06T18:38:16.612490+00:00 | GARDER | récent `handover.md:15685`; manifeste gardé : `$R/audits/retrieval_v360_miss_evidence/run_20261007_a/manifest.json` |
| `$R/audits/retrieval_v347_persistent_launcher` | 0.000 | 2026-10-06T05:28:33.329996+00:00 | GARDER | récent `handover.md:15608` |
| `$R/audits/retrieval_v34_fast_resume/ban_audit` | 0.001 | 2026-09-11T15:54:06.590624+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/parallel_ban_audit/manifest.json` |
| `$R/audits/retrieval_v34_fast_resume/bounded32` | 0.051 | 2026-09-11T15:38:36.985221+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88` |
| `$R/audits/retrieval_v34_fast_resume/cached8` | 0.000 | 2026-09-11T15:28:03.586955+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88` |
| `$R/audits/retrieval_v34_fast_resume/full_job` | 9.092 | 2026-10-01T20:28:29.409366+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$H/step2_diagnostic_20261017/smoke/plan.json` |
| `$R/audits/retrieval_v34_fast_resume/gate_serial` | 0.000 | 2026-09-11T15:54:28.956652+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88` |
| `$R/audits/retrieval_v34_fast_resume/gate` | 0.000 | 2026-09-11T16:05:59.966568+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/full_job/generation/manifest.json` |
| `$R/audits/retrieval_v34_fast_resume/parallel_ban_audit` | 0.001 | 2026-09-11T16:05:45.257083+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/ban_audit/manifest.json` |
| `$R/audits/retrieval_v34_fast_resume/parallel_pilot32` | 0.013 | 2026-09-11T15:59:02.463391+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$H/step2_option_d_20261017/full/result.json` |
| `$R/audits/retrieval_v34_fast_resume/parallel_prototype32` | 0.013 | 2026-09-11T15:55:55.888858+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$H/step2_option_d_20261017/full/result.json` |
| `$R/audits/retrieval_v34_fast_resume/parallel_repaired32` | 0.013 | 2026-09-11T16:00:22.835821+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$H/step2_option_d_20261017/full/result.json` |
| `$R/audits/retrieval_v34_fast_resume/parallel_repeated32` | 0.014 | 2026-09-11T16:03:28.213617+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/gate/manifest.json` |
| `$R/audits/retrieval_v34_fast_resume/pilot32` | 0.012 | 2026-09-11T15:46:27.199184+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$H/step2_option_d_20261017/full/result.json` |
| `$R/audits/retrieval_v34_fast_resume/prepared` | 0.002 | 2026-09-11T15:41:34.317692+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/parallel_ban_audit/manifest.json` |
| `$R/audits/retrieval_v34_fast_resume/profile8` | 0.000 | 2026-09-11T15:24:46.543737+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88` |
| `$R/audits/retrieval_v34_fast_resume/projected8` | 0.024 | 2026-09-11T15:29:58.717004+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88` |
| `$R/audits/retrieval_v34_fast_resume/prototype32` | 0.051 | 2026-09-11T15:32:22.531434+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88` |
| `$R/audits/retrieval_v34_fast_resume/released32` | 0.058 | 2026-09-11T15:41:11.481602+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$H/step2_option_d_20261017/full/result.json` |
| `$R/audits/retrieval_v34_fast_resume/repaired32` | 0.012 | 2026-09-11T15:48:37.800135+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$H/step2_option_d_20261017/full/result.json` |
| `$R/audits/retrieval_v34_fast_resume/repeated32` | 0.013 | 2026-09-11T15:51:43.957391+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/parallel_repeated32/shard00016/manifest.json` |
| `$R/audits/retrieval_v34_fast_resume` **[fichiers directs]** | 0.000 | 2026-09-11T16:06:16.195841+00:00 | GARDER | récent `docs/retrieval_site_operational_metric_result_20261014.json:88` |
| `$R/audits/retrieval_v352_persistent_launcher` | 0.000 | 2026-10-06T15:51:15.987846+00:00 | GARDER | récent `handover.md:15684` |
| `$R/audits/retrieval_v358_postseal_evaluation` | 0.002 | 2026-10-07T01:36:59.655887+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v360_miss_evidence/run_20261007_a/manifest.json` |
| `$R/audits/retrieval_v359_persistent_launcher` | 0.000 | 2026-10-06T18:38:06.703364+00:00 | GARDER | récent `handover.md:15801` |
| `$R/audits/retrieval_v35_site_fusion` | 0.382 | 2026-10-01T21:43:14.404082+00:00 | GARDER | récent `handover.md:7155`; manifeste gardé : `$R/audits/retrieval_v36_evidence_ranker/pilot_20261002_a/feature_manifest.json` |
| `$R/audits/retrieval_v360_miss_evidence` | 0.001 | 2026-10-07T13:29:32.840986+00:00 | GARDER | récent `handover.md:15919` |
| `$R/audits/retrieval_v361_name_provenance` | 0.000 | 2026-10-07T13:34:48.771171+00:00 | GARDER | récent `handover.md:15931` |
| `$R/audits/retrieval_v362_address_channel` | 0.000 | 2026-10-07T13:39:52.920262+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v363_transfer_coverage` | 0.000 | 2026-10-07T13:43:34.918376+00:00 | GARDER | récent `handover.md:15933` |
| `$R/audits/retrieval_v364_address_words` | 0.000 | 2026-10-07T13:46:51.390311+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v365_address_words_budget` | 0.022 | 2026-10-07T13:49:35.962476+00:00 | GARDER | récent `handover.md:15943` |
| `$R/audits/retrieval_v36_evidence_ranker` | 0.646 | 2026-10-01T22:30:42.790607+00:00 | GARDER | récent `handover.md:7202`; manifeste gardé : `$R/audits/retrieval_v37_parent_evidence/pilot_20261002_a/manifest.json` |
| `$R/audits/retrieval_v37_parent_evidence` | 0.039 | 2026-10-01T22:38:59.845838+00:00 | GARDER | récent `handover.md:7205` |
| `$R/audits/retrieval_v38_initial_top100` | 0.003 | 2026-10-02T05:35:12.783932+00:00 | GARDER | récent `handover.md:7260` |
| `$R/audits/retrieval_v39_relational_proofs` | 0.207 | 2026-10-02T09:14:02.412289+00:00 | GARDER | récent `handover.md:7301` |
| `$R/audits/retrieval_v40_initial_top100` | 0.003 | 2026-10-02T12:03:49.421549+00:00 | GARDER | récent `handover.md:7466` |
| `$R/audits/retrieval_v41_learned_initial` | 0.001 | 2026-10-02T14:55:25.106735+00:00 | GARDER | récent `handover.md:7522` |
| `$R/audits/retrieval_v42_indexed_metric` | 0.007 | 2026-10-02T15:11:01.371926+00:00 | GARDER | récent `handover.md:7632` |
| `$R/audits/retrieval_v43_spatial` | 0.001 | 2026-10-02T15:55:47.392791+00:00 | GARDER | récent `handover.md:7657` |
| `$R/audits/retrieval_v44_index_pruning` | 0.000 | 2026-10-02T15:59:25.172884+00:00 | GARDER | récent `handover.md:7676` |
| `$R/audits/retrieval_v45_spatial_pruned` | 0.004 | 2026-10-02T16:02:26.951105+00:00 | GARDER | récent `handover.md:7688`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v46_joint` | 0.020 | 2026-10-02T16:41:04.238822+00:00 | GARDER | récent `handover.md:7747`; manifeste gardé : `$R/audits/retrieval_v50_serving_features/training_20261002_a/manifest.json` |
| `$R/audits/retrieval_v47_joint_fair` | 0.001 | 2026-10-02T16:51:02.695393+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v48_joint_point` | 0.001 | 2026-10-02T16:58:25.235196+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v49_primary_probe` | 0.002 | 2026-10-02T17:09:04.670594+00:00 | GARDER | récent `handover.md:7778` |
| `$R/audits/retrieval_v50_serving_features` | 0.000 | 2026-10-02T17:37:16.549343+00:00 | GARDER | récent `handover.md:7805` |
| `$R/audits/retrieval_v51_dense_training` | 0.004 | 2026-10-02T17:43:13.680557+00:00 | GARDER | récent `handover.md:7841` |
| `$R/audits/retrieval_v52_contrastive_training` | 0.154 | 2026-10-02T17:56:46.705202+00:00 | GARDER | récent `handover.md:7901`; manifeste gardé : `$R/audits/retrieval_v59_sparse_query/probe_20261002_a/manifest.json` |
| `$R/audits/retrieval_v53_nested_training` | 0.000 | 2026-10-02T18:04:12.125845+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v54_rowspace_training` | 0.007 | 2026-10-02T18:10:24.273186+00:00 | GARDER | récent `handover.md:7930` |
| `$R/audits/retrieval_v55_proof_metric` | 0.015 | 2026-10-02T18:38:16.776476+00:00 | GARDER | récent `handover.md:7982`; manifeste gardé : `$R/audits/retrieval_v56_exact_fields/training_20261002_a/manifest.json` |
| `$R/audits/retrieval_v56_exact_fields` | 0.015 | 2026-10-02T18:55:23.121929+00:00 | GARDER | récent `handover.md:7999`; manifeste gardé : `$R/audits/retrieval_v59_sparse_query/probe_20261002_a/manifest.json` |
| `$R/audits/retrieval_v57_native_proof` | 0.002 | 2026-10-02T19:02:45.076851+00:00 | GARDER | récent `handover.md:8015` |
| `$R/audits/retrieval_v58_raw_proofs` | 0.003 | 2026-10-02T19:23:00.820661+00:00 | GARDER | récent `handover.md:8094` |
| `$R/audits/retrieval_v59_sparse_query` | 0.009 | 2026-10-02T19:39:29.026634+00:00 | GARDER | récent `handover.md:8106`; manifeste gardé : `$R/audits/retrieval_v61_sparse_flat/bundle_20261002_a/manifest.json` |
| `$R/audits/retrieval_v60_sparse_initial` | 0.002 | 2026-10-02T19:46:49.046366+00:00 | GARDER | récent `handover.md:8114`; manifeste gardé : `$R/audits/retrieval_v61_sparse_flat/bundle_20261002_a/manifest.json` |
| `$R/audits/retrieval_v61_sparse_flat` | 0.002 | 2026-10-02T19:55:28.524991+00:00 | GARDER | récent `handover.md:8124`; manifeste gardé : `$R/audits/retrieval_v60_sparse_initial/bundle_20261002_a/manifest.json` |
| `$R/audits/retrieval_v62_sparse_cost` | 0.000 | 2026-10-02T20:09:14.236212+00:00 | GARDER | récent `handover.md:8132` |
| `$R/audits/retrieval_v63_joint_contrastive` | 0.000 | 2026-10-02T20:29:17.794098+00:00 | GARDER | récent `handover.md:8157` |
| `$R/audits/retrieval_v64_geo_postings` | 0.000 | 2026-10-02T20:36:45.696122+00:00 | GARDER | récent `handover.md:8189` |
| `$R/audits/retrieval_v65_initialisms` | 0.009 | 2026-10-02T21:02:54.651504+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v67_stored_initialisms/probe_20261002_a/manifest.json` |
| `$R/audits/retrieval_v66_head_initial` | 0.002 | 2026-10-02T21:09:53.322702+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v67_stored_initialisms` | 0.009 | 2026-10-02T21:30:40.541097+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v65_initialisms/probe_20261002_a/manifest.json` |
| `$R/audits/retrieval_v68_impact_cost` | 0.000 | 2026-10-02T21:38:14.421993+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v69_sparse_contrastive` | 0.000 | 2026-10-02T22:22:04.179064+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v70_continuous_sparse` | 0.004 | 2026-10-02T22:28:17.670562+00:00 | GARDER | récent `handover.md:8353`; manifeste gardé : `$R/audits/retrieval_v71_conditioned_sparse/probe_20261003_a/manifest.json` |
| `$R/audits/retrieval_v71_conditioned_sparse` | 0.004 | 2026-10-02T22:38:33.485005+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v70_continuous_sparse/probe_20261003_a/manifest.json` |
| `$R/audits/retrieval_v72_observation_cost` | 0.000 | 2026-10-02T22:48:53.727569+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v73_owner_observation` | 0.021 | 2026-10-02T22:56:37.763388+00:00 | GARDER | récent `handover.md:8445` |
| `$R/audits/retrieval_v74_coherent_proof` | 0.004 | 2026-10-02T23:18:08.119956+00:00 | GARDER | récent `handover.md:8492` |
| `$R/audits/retrieval_v75_history_site` | 0.174 | 2026-10-02T23:37:03.799982+00:00 | GARDER | récent `handover.md:8572` |
| `$R/audits/retrieval_v76_history_parent` | 0.108 | 2026-10-03T00:11:03.955023+00:00 | GARDER | récent `handover.md:8630` |
| `$R/audits/retrieval_v77_safe_sparse` | 0.010 | 2026-10-03T00:30:14.565905+00:00 | GARDER | récent `handover.md:8692`; manifeste gardé : `$R/audits/retrieval_v79_sparse_geo/training_20261003_a/manifest.json` |
| `$R/audits/retrieval_v78_safe_sparse_initial` | 0.003 | 2026-10-03T00:36:22.115222+00:00 | GARDER | récent `handover.md:8723`; manifeste gardé : `$R/audits/retrieval_v187_staged_initial/probe_20261004_a/audit-context/manifest.json` |
| `$R/audits/retrieval_v79_sparse_geo` | 0.000 | 2026-10-03T01:12:41.759574+00:00 | GARDER | récent `handover.md:8781` |
| `$R/audits/retrieval_v80_succession_site` | 0.000 | 2026-10-03T01:18:47.777357+00:00 | GARDER | récent `handover.md:8809` |
| `$R/audits/retrieval_v81_succession_bindings` | 0.756 | 2026-10-03T01:37:18.159528+00:00 | GARDER | récent `handover.md:8880`; manifeste gardé : `$R/audits/retrieval_v82_scoped_succession_names/projection_20261003_b/manifest.json` |
| `$R/audits/retrieval_v82_scoped_succession_names` | 3.989 | 2026-10-03T02:04:58.374934+00:00 | GARDER | récent `handover.md:8940`; manifeste gardé : `$R/audits/retrieval_v85_name_facts/compile_20261003_a/manifest.json` |
| `$R/audits/retrieval_v84_succession_csr` | 0.503 | 2026-10-03T02:22:48.891573+00:00 | GARDER | récent `handover.md:8982` |
| `$R/audits/retrieval_v85_name_facts` | 1.215 | 2026-10-03T02:40:01.394472+00:00 | GARDER | récent `handover.md:9023`; manifeste gardé : `$R/audits/retrieval_v86_name_postings/compile_20261003_a/manifest.json` |
| `$R/audits/retrieval_v86_name_postings` | 1.539 | 2026-10-03T02:47:43.179365+00:00 | GARDER | récent `handover.md:9041` |
| `$R/audits/retrieval_v87_scoped_paths` | 0.000 | 2026-10-03T03:24:47.395234+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v88_scoped_paths` | 0.000 | 2026-10-03T03:32:17.365731+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v89_scoped_paths` | 0.000 | 2026-10-03T03:38:35.922898+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v90_scoped_paths` | 0.001 | 2026-10-03T03:45:05.206863+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/retrieval_v91_scoped_paths` | 0.005 | 2026-10-03T04:18:59.203903+00:00 | GARDER | récent `handover.md:9116` |
| `$R/audits/retrieval_v93_national_names` | 3.383 | 2026-10-03T04:25:15.012473+00:00 | GARDER | récent `handover.md:9154` |
| `$R/audits/retrieval_v94_virtual_national_names` | 0.192 | 2026-10-03T04:30:10.792885+00:00 | GARDER | récent `handover.md:9179`; manifeste gardé : `$R/audits/retrieval_v96_national_membership/compile_20261003_a/manifest.json` |
| `$R/audits/retrieval_v95_national_names` | 3.456 | 2026-10-03T04:39:45.692195+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v97_complete_provenance/compile_20261003_a/manifest.json` |
| `$R/audits/retrieval_v96_national_membership` | 1.081 | 2026-10-03T04:43:33.689770+00:00 | GARDER | V142 : 1 artefacts; récent `docs/retrieval_frozen_v142.json:27001`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$R/audits/retrieval_v97_complete_provenance` | 1.642 | 2026-10-03T04:50:10.230986+00:00 | GARDER | récent `handover.md:9222`; manifeste gardé : `$R/audits/retrieval_v100_complete_word_metadata/compile_20261003_a/manifest.json` |
| `$R/audits/retrieval_v98_national_postings` | 5.040 | 2026-10-03T04:55:10.214807+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v100_complete_word_metadata/compile_20261003_a/manifest.json` |
| `$R/audits/retrieval_v99_national_paths` | 0.000 | 2026-10-03T05:19:47.044852+00:00 | GARDER | récent `handover.md:9248` |
| `$R/audits/v21-test-runtime.GkBAkd` | 0.001 | 2026-09-11T08:19:20.169836+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/v4_12_review_adjudication_pilot` | 0.000 | 2026-07-31T09:03:07.838565+00:00 | GARDER | récent `handover.md:1550` |
| `$R/audits/v4_12_snapshot_lookup` | 0.000 | 2026-07-28T14:42:08.887550+00:00 | GARDER | récent `handover.md:3259` |
| `$R/audits/v4_12_strict_stores` | 0.000 | 2026-07-28T19:00:09.792757+00:00 | GARDER | récent `handover.md:2773` |
| `$R/audits/v4_12_unit_inputs` | 0.001 | 2026-07-28T15:47:51.193006+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/v4_12_unit_oracle` | 0.000 | 2026-07-28T16:38:47.865082+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/v4_12_unit_retrieval_evaluator_attempts` | 0.000 | 2026-07-30T10:22:27.128091+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/v4_12_unit_retrieval_evaluator` | 0.000 | 2026-07-30T10:22:27.090093+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/audits/v4_12_unit_retrieval_parity` | 0.000 | 2026-07-28T21:41:55.725672+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/v4_12_unit_retrieval_worker` | 0.000 | 2026-07-28T21:41:54.640461+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/v4_1_representative_evidence` | 0.001 | 2026-07-27T08:20:17.058547+00:00 | GARDER | récent `handover.md:3805`; manifeste gardé : `$R/audits/v4_1_representative_summary/2d18ef172f32aefc/manifest.json` |
| `$R/audits/v4_1_representative_summary` | 0.000 | 2026-07-27T08:28:44.487769+00:00 | GARDER | récent `handover.md:3807` |
| `$R/audits/v4_1_representative` | 0.000 | 2026-07-27T08:01:24.992610+00:00 | GARDER | récent `handover.md:3804`; manifeste gardé : `$R/audits/v4_1_representative_summary/2d18ef172f32aefc/manifest.json` |
| `$R/audits/v4_3_hard_labels` | 0.007 | 2026-07-27T17:31:58.231474+00:00 | GARDER | récent `handover.md:3781`; manifeste gardé : `$R/audits/v4_4_adjudication_batch_inputs/a124489e092b2e61/manifest.json` |
| `$R/audits/v4_4_adjudication_batch_inputs` | 0.001 | 2026-07-27T23:26:02.300380+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/v4_4_adjudications/1e2c68337408c453/manifest.json` |
| `$R/audits/v4_4_adjudications` | 0.000 | 2026-07-27T23:26:12.409301+00:00 | GARDER | récent `handover.md:3524`; manifeste gardé : `$R/audits/v4_4_gate/6f5972fbdcf10043/manifest.json` |
| `$R/audits/v4_4_evidence_facts` | 0.000 | 2026-07-27T21:50:20.572109+00:00 | GARDER | récent `handover.md:3521`; manifeste gardé : `$R/audits/v4_4_sector_facts/6a08bff403154884/manifest.json` |
| `$R/audits/v4_4_gate` | 0.000 | 2026-07-27T23:26:24.083488+00:00 | GARDER | récent `handover.md:3532`; manifeste gardé : `$R/datasets/v4_5_hard_scenes/21f8c0b0b172b907/manifest.json` |
| `$R/audits/v4_4_known_contradictions_inputs` | 0.000 | 2026-07-27T22:12:06.505114+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/v4_4_adjudications/320fe62322e14d25/manifest.json` |
| `$R/audits/v4_4_official_evidence` | 0.000 | 2026-07-27T20:50:03.799841+00:00 | GARDER | récent `handover.md:3519`; manifeste gardé : `$R/audits/v4_4_sector_evidence/3149124f69dd7b1f/manifest.json` |
| `$R/audits/v4_4_sector_evidence` | 0.047 | 2026-07-27T21:59:50.700940+00:00 | GARDER | récent `handover.md:3522`; manifeste gardé : `$R/audits/v4_4_sector_facts/6a08bff403154884/manifest.json` |
| `$R/audits/v4_4_sector_facts` | 0.000 | 2026-07-27T22:08:18.810890+00:00 | GARDER | récent `handover.md:3523` |
| `$R/audits/v4_4_web_evidence` | 0.015 | 2026-07-27T23:21:57.928072+00:00 | GARDER | récent `docs/production_cost_study.json:176` |
| `$R/audits/v4_7_current_adjudications` | 0.013 | 2026-07-28T03:05:23.080214+00:00 | GARDER | récent `handover.md:3463`; manifeste gardé : `$R/audits/v4_12_review_adjudication_pilot/c7a9feecaf2d3c2a/manifest.json` |
| `$R/audits/v4_7_current_top1_docket` | 0.000 | 2026-07-28T02:45:01.301270+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/v4_7_current_adjudications/4cc5420fb5da0683/manifest.json` |
| `$R/audits/v4_7_official_evidence` | 0.000 | 2026-07-28T02:46:27.533335+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/v4_7_current_adjudications/4cc5420fb5da0683/manifest.json` |
| `$R/audits` **[fichiers directs]** | 0.000 | 2026-10-07T13:49:35.962419+00:00 | GARDER | récent `docs/production_cost_study.json:176`; manifeste gardé : `$R/audits/retrieval_v13_stop_receipt/005fb19823721480/manifest.json` |
| `$R/benchmarks` | 0.018 | 2026-08-13T15:51:57.939284+00:00 | GARDER | récent `handover.md:3882`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/bundles` | 0.002 | 2026-07-31T17:00:33.854314+00:00 | GARDER | récent `handover.md:1285` |
| `$R/certification` | 0.000 | 2026-07-25T23:31:50.020989+00:00 | GARDER | récent `handover.md:3973` |
| `$R/certifications` | 0.001 | 2026-07-28T18:24:04.472250+00:00 | GARDER | récent `handover.md:2773` |
| `$R/challenges` | 0.005 | 2026-07-28T11:20:14.410187+00:00 | GARDER | récent `handover.md:3120`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/datasets/acceptor_v4` | 0.082 | 2026-07-26T14:20:17.230880+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:43` |
| `$R/datasets/crm_gt_exact_admission` | 0.017 | 2026-08-17T15:01:57.247837+00:00 | DOUTE | Sources, jeux réservés ou autorités historiques ; conservation par précaution. |
| `$R/datasets/crm_gt_v2_certified_population` | 0.006 | 2026-08-17T15:06:16.324320+00:00 | GARDER | récent `handover.md:5375` |
| `$R/datasets/crm_gt_v2_commercial_certified_population` | 0.024 | 2026-08-17T17:43:01.919994+00:00 | GARDER | récent `docs/crm_structure_audit.json:1058`; manifeste gardé : `$H/step2_crm_columns_tiers_20261020/result.json` |
| `$R/datasets/crm_gt_v2_population` | 0.015 | 2026-08-17T14:43:28.662063+00:00 | GARDER | récent `handover.md:451` |
| `$R/datasets/crm_gt_v2_retrieval_input_commercial` | 0.005 | 2026-08-17T17:43:40.556258+00:00 | DOUTE | manifeste gardé : `$R/experiments/crm_gt_v2_v7_current_channels_5a9c2437a2a3d817_dev/manifest.json` |
| `$R/datasets/crm_gt_v2_retrieval_input` | 0.000 | 2026-08-17T15:10:17.054385+00:00 | DOUTE | manifeste gardé : `$R/experiments/crm_gt_v2_overlay_channels_eab0d23205256a4a_dev/manifest.json` |
| `$R/datasets/crm_gt_v2_sanitized_population` | 0.008 | 2026-08-17T15:02:36.722129+00:00 | GARDER | récent `handover.md:5367` |
| `$R/datasets/crm_gt_v3_retrieval_input_all_human` | 0.019 | 2026-08-18T22:24:39.405943+00:00 | GARDER | récent `docs/production_cost_study.json:87`; manifeste gardé : `$R/experiments/retrieval_selective_v4/qualification_development.parquet.manifest.json` |
| `$R/datasets/crm_new_data_20260817` | 0.042 | 2026-08-17T13:09:16.892345+00:00 | GARDER | récent `docs/crm_structure_audit.json:1048`; manifeste gardé : `$H/step2_crm_columns_tiers_20261020/result.json` |
| `$R/datasets/downstream_failed` | 0.143 | 2026-07-26T01:07:13.341435+00:00 | DOUTE | Sources, jeux réservés ou autorités historiques ; conservation par précaution. |
| `$R/datasets/downstream` | 0.144 | 2026-07-26T01:07:34.420491+00:00 | GARDER | récent `docs/retrieval_composite_admission_static_audit_20261014.json:18` |
| `$R/datasets/ranker_v4` | 0.051 | 2026-07-26T12:59:29.521348+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:41` |
| `$R/datasets/retrieval_all_human_v11_benchmark` | 0.009 | 2026-08-22T10:15:22.750648+00:00 | DOUTE | manifeste gardé : `$R/audits/retrieval_v34_fast_resume/prepared/manifest.json` |
| `$R/datasets/retrieval_all_human_v12_label_ledger` | 0.001 | 2026-08-24T11:46:59.174794+00:00 | GARDER | récent `docs/retrieval_far_label_audit_20261015.json:43`; manifeste gardé : `$H/step2_diagnostic_20261017/smoke/plan.json` |
| `$R/datasets/synthetic_augmented_model_mix_v1` | 0.119 | 2026-08-17T02:53:39.736268+00:00 | DOUTE | Sources, jeux réservés ou autorités historiques ; conservation par précaution. |
| `$R/datasets/synthetic_gt_agentic_loop_v2` | 0.000 | 2026-08-15T10:47:23.791436+00:00 | DOUTE | Sources, jeux réservés ou autorités historiques ; conservation par précaution. |
| `$R/datasets/synthetic_gt_corpus/.balanced_selector_context_index` | 0.290 | 2026-08-17T02:05:28.278350+00:00 | GARDER | récent `handover.md:620` |
| `$R/datasets/synthetic_gt_corpus/agentic_loop_v2` | 0.129 | 2026-08-15T14:00:17.306926+00:00 | GARDER | récent `handover.md:620` |
| `$R/datasets/synthetic_gt_corpus/agentic_pilot` | 2.661 | 2026-08-15T02:23:29.275803+00:00 | GARDER | récent `handover.md:620` |
| `$R/datasets/synthetic_gt_corpus/balanced_v1` | 3.086 | 2026-08-17T02:17:22.588388+00:00 | GARDER | récent `handover.md:620`; manifeste gardé : `$R/datasets/synthetic_gt_corpus/official_context_v8_closed_extended.jsonl.manifest.json` |
| `$R/datasets/synthetic_gt_corpus/composite_v2` | 0.017 | 2026-08-15T15:48:58.752359+00:00 | GARDER | récent `handover.md:620` |
| `$R/datasets/synthetic_gt_corpus/composite_v3` | 0.031 | 2026-08-15T21:11:36.525306+00:00 | GARDER | récent `handover.md:620` |
| `$R/datasets/synthetic_gt_corpus` **[fichiers directs]** | 13.767 | 2026-08-16T08:46:51.287045+00:00 | GARDER | récent `handover.md:620`; manifeste gardé : `$R/datasets/synthetic_gt_corpus/balanced_v1/P028_seed_input.jsonl.manifest.json` |
| `$R/datasets/synthetic_gt_model_features` | 0.281 | 2026-08-17T02:53:20.563942+00:00 | DOUTE | Sources, jeux réservés ou autorités historiques ; conservation par précaution. |
| `$R/datasets/synthetic_gt_model_retrieval_input_v1` | 0.019 | 2026-08-17T02:26:02.894283+00:00 | DOUTE | manifeste gardé : `$R/experiments/synthetic_gt_overlay_channels_799bf5b289a0e943/manifest.json` |
| `$R/datasets/v4_10_structured_acceptor` | 0.007 | 2026-07-28T04:33:09.397299+00:00 | GARDER | récent `handover.md:3380` |
| `$R/datasets/v4_10b_structured_acceptor` | 0.007 | 2026-07-28T04:54:55.747511+00:00 | GARDER | récent `handover.md:3336` |
| `$R/datasets/v4_11_acceptor` | 0.003 | 2026-07-28T08:05:07.460046+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:47`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/datasets/v4_11_input_blind` | 0.041 | 2026-07-28T07:41:39.224787+00:00 | GARDER | récent `handover.md:3031`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/datasets/v4_12_bge_training_groups` | 0.013 | 2026-08-14T22:10:56.980712+00:00 | GARDER | récent `handover.md:1078` |
| `$R/datasets/v4_12_direct_evidence` | 0.001 | 2026-07-28T12:20:56.558986+00:00 | GARDER | récent `handover.md:3194`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/datasets/v4_12_learned_business_features` | 0.290 | 2026-08-13T16:41:00.420764+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:48` |
| `$R/datasets/v4_12_learned_candidate_features` | 0.151 | 2026-08-13T16:13:59.372728+00:00 | DOUTE | Sources, jeux réservés ou autorités historiques ; conservation par précaution. |
| `$R/datasets/v4_12_learned_scenes` | 0.008 | 2026-08-13T17:15:16.698406+00:00 | DOUTE | Sources, jeux réservés ou autorités historiques ; conservation par précaution. |
| `$R/datasets/v4_12_learned_unified_population` | 0.005 | 2026-08-13T15:47:05.422400+00:00 | GARDER | récent `handover.md:1193`; manifeste gardé : `$R/evaluations/v4_12_learned_unified_retrieval/cce1bc83f82a1c3f/manifest.json` |
| `$R/datasets/v4_12_neural_text_corpus` | 0.076 | 2026-08-13T21:55:05.029161+00:00 | GARDER | récent `handover.md:1142` |
| `$R/datasets/v4_12_neural_training_groups` | 0.006 | 2026-08-13T22:21:29.085821+00:00 | GARDER | récent `handover.md:1156` |
| `$R/datasets/v4_1` | 0.037 | 2026-07-26T23:01:52.188537+00:00 | GARDER | récent `handover.md:3818`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/datasets/v4_5_hard_scenes` | 0.002 | 2026-07-27T23:42:37.188279+00:00 | GARDER | récent `handover.md:3496`; manifeste gardé : `$R/audits/v4_7_current_adjudications/4cc5420fb5da0683/manifest.json` |
| `$R/datasets/v4_6_aligned_a` | 0.037 | 2026-07-28T02:35:55.232759+00:00 | GARDER | récent `handover.md:3479`; manifeste gardé : `$R/models/v4_6_aligned_ranker/421f2cd0cc436af7/manifest.json` |
| `$R/datasets/v4_6_aligned_b` | 0.037 | 2026-07-28T02:35:55.232678+00:00 | GARDER | récent `handover.md:3480`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/datasets/v4_8_acceptor_partitions` | 0.001 | 2026-07-28T03:18:59.427171+00:00 | GARDER | récent `handover.md:3446`; manifeste gardé : `$R/audits/v4_12_review_adjudication_pilot/c7a9feecaf2d3c2a/manifest.json` |
| `$R/datasets` **[fichiers directs]** | 0.000 | 2026-08-24T11:46:59.106749+00:00 | DOUTE | Fichiers directs/métadonnées du parent : pas une cible de suppression. |
| `$R/diag-r2-helper.Bg9z6m` | 0.000 | 2026-07-30T19:18:13.716031+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/diag-r2b-gate.5uQLVz` | 0.143 | 2026-07-30T19:38:37.914189+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/diag-r2b-gate.oVHNuc` | 0.146 | 2026-07-30T19:37:10.341252+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/diag-r2b-smoke.hOQzRB` | 0.000 | 2026-07-30T19:24:19.713214+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/diag-r2b-trace.84M3F3` | 0.146 | 2026-07-30T19:38:07.636620+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/diag-r3-successor-gate.3129l7j9` | 0.144 | 2026-07-30T20:26:10.988780+00:00 | DOUTE | Contenu label/jeu fermé possible : non ouvert, laissé en place. |
| `$R/diag-r3-successor-gate.8uvl2olp` | 0.000 | 2026-07-30T20:12:27.728598+00:00 | DOUTE | Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide. |
| `$R/diag-r3-successor-gate.9fy69i1l` | 0.144 | 2026-07-30T20:15:20.559223+00:00 | DOUTE | Contenu label/jeu fermé possible : non ouvert, laissé en place. |
| `$R/diag-r3-successor-gate.sfzj9buk` | 0.144 | 2026-07-30T21:12:03.575472+00:00 | GARDER | récent `handover.md:2592` |
| `$R/diag-r3-successor-gate.yww2qf5m` | 0.144 | 2026-07-30T20:26:50.571914+00:00 | GARDER | récent `handover.md:2555` |
| `$R/diag-r3-successor-gate.zi2oynzm` | 0.144 | 2026-07-30T20:12:58.362085+00:00 | DOUTE | Contenu label/jeu fermé possible : non ouvert, laissé en place. |
| `$R/evaluations` | 0.018 | 2026-08-17T20:12:47.471382+00:00 | GARDER | récent `handover.md:1182`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/experiments/admission_diagnostic_dev_c33b80855f560074_5a0e67f` | 0.003 | 2026-07-24T09:00:16.727761+00:00 | GARDER | récent `handover.md:4050`; manifeste gardé : `$R/evaluations/v4_12_learned_unified_retrieval/cce1bc83f82a1c3f/manifest.json` |
| `$R/experiments/admission_diagnostic_test_c33b80855f560074_eb0e6a3` | 0.003 | 2026-07-25T23:31:10.386425+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/evaluations/v4_12_learned_unified_retrieval/cce1bc83f82a1c3f/manifest.json` |
| `$R/experiments/admission_train_c33b80855f560074_734f792` | 0.011 | 2026-07-26T00:46:41.155323+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/evaluations/v4_12_learned_unified_retrieval/cce1bc83f82a1c3f/manifest.json` |
| `$R/experiments/admission_v4_dev_amb_af5ce0b` | 0.000 | 2026-07-26T14:16:34.574699+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/admission_v4_dev_e566c25` | 0.000 | 2026-07-26T12:51:36.236978+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/admission_v4_final_7dbd552` | 0.000 | 2026-07-26T14:43:41.197455+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/admission_v4_fit_amb_af5ce0b` | 0.000 | 2026-07-26T14:16:34.938380+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/admission_v4_fit_e566c25` | 0.001 | 2026-07-26T12:51:40.356730+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/channel_audit_dev_c33b80855f560074_69f6f7e` | 0.036 | 2026-07-24T07:59:22.441314+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/channel_audit_k5000_dev_c33b80855f560074_d4255de` | 0.395 | 2026-07-24T08:47:31.417826+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_diagnostic_dev_c33b80855f560074_5a0e67f/manifest.json` |
| `$R/experiments/channel_audit_k5000_test_c33b80855f560074_eb0e6a3` | 0.402 | 2026-07-25T23:25:09.985327+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_diagnostic_test_c33b80855f560074_eb0e6a3/manifest.json` |
| `$R/experiments/channel_audit_k5000_train_c33b80855f560074_aeeaf0f` | 1.786 | 2026-07-26T00:45:28.266309+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_train_c33b80855f560074_734f792/manifest.json` |
| `$R/experiments/channel_audit_k5000_v4_dev_amb_af5ce0b` | 0.010 | 2026-07-26T14:15:44.561191+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_v4_dev_amb_af5ce0b/manifest.json` |
| `$R/experiments/channel_audit_k5000_v4_dev_e566c25` | 0.046 | 2026-07-26T12:49:52.684408+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_v4_dev_e566c25/manifest.json` |
| `$R/experiments/channel_audit_k5000_v4_final_7dbd552` | 0.054 | 2026-07-26T14:43:25.134136+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_v4_final_7dbd552/manifest.json` |
| `$R/experiments/channel_audit_k5000_v4_fit_amb_af5ce0b` | 0.023 | 2026-07-26T14:16:14.036389+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_v4_fit_amb_af5ce0b/manifest.json` |
| `$R/experiments/channel_audit_k5000_v4_fit_e566c25` | 0.128 | 2026-07-26T12:45:16.624191+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_v4_fit_e566c25/manifest.json` |
| `$R/experiments/channel_audit_siren_dev_c33b80855f560074_d4255de` | 0.060 | 2026-07-24T08:31:22.373815+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/closed_overlay_channel_audit_dev_c33b80855f560074_d71d3cb` | 0.031 | 2026-07-24T08:16:45.144644+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/closed_overlay_channel_audit_k5000_dev_c33b80855f560074_d4255de` | 0.323 | 2026-07-24T08:51:29.920852+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_diagnostic_dev_c33b80855f560074_5a0e67f/manifest.json` |
| `$R/experiments/closed_overlay_channel_audit_k5000_test_c33b80855f560074_eb0e6a3` | 0.325 | 2026-07-25T23:30:24.252120+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_diagnostic_test_c33b80855f560074_eb0e6a3/manifest.json` |
| `$R/experiments/closed_overlay_channel_audit_k5000_train_c33b80855f560074_aeeaf0f` | 1.438 | 2026-07-26T00:23:50.956764+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_train_c33b80855f560074_734f792/manifest.json` |
| `$R/experiments/closed_overlay_channel_audit_k5000_v4_dev_e566c25` | 0.037 | 2026-07-26T12:51:17.138079+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_v4_dev_e566c25/manifest.json` |
| `$R/experiments/closed_overlay_channel_audit_k5000_v4_fit_e566c25` | 0.101 | 2026-07-26T12:47:34.753036+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_v4_fit_e566c25/manifest.json` |
| `$R/experiments/crm_gt_v2_overlay_channels_5a9c2437a2a3d817_dev` | 0.487 | 2026-08-17T17:52:27.585156+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/crm_gt_v2_overlay_channels_eab0d23205256a4a_dev` | 0.109 | 2026-08-17T15:13:00.375046+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/crm_gt_v2_retrieval_eval_5a9c2437a2a3d817_dev` | 0.004 | 2026-08-17T18:08:56.623323+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/crm_gt_v2_retrieval_eval_eab0d23205256a4a_dev` | 0.001 | 2026-08-17T15:15:15.460852+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/crm_gt_v2_v7_channels_eab0d23205256a4a_dev` | 0.128 | 2026-08-17T15:14:55.671408+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/crm_gt_v2_v7_current_channels_5a9c2437a2a3d817_dev` | 0.649 | 2026-08-17T18:08:29.270511+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/downstream_architecture_audit_a59fb0f` | 0.000 | 2026-07-26T00:04:17.756450+00:00 | GARDER | récent `handover.md:3952` |
| `$R/experiments/official_evidence_retrieval_v1` | 0.000 | 2026-08-18T11:18:34.415510+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/overlay_channel_audit_k5000_v4_dev_amb_af5ce0b` | 0.008 | 2026-07-26T14:16:09.961133+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_v4_dev_amb_af5ce0b/manifest.json` |
| `$R/experiments/overlay_channel_audit_k5000_v4_final_7dbd552` | 0.044 | 2026-07-26T14:40:02.299812+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_v4_final_7dbd552/manifest.json` |
| `$R/experiments/overlay_channel_audit_k5000_v4_fit_amb_af5ce0b` | 0.018 | 2026-07-26T14:16:25.177947+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_v4_fit_amb_af5ce0b/manifest.json` |
| `$R/experiments/qualification_stability_train_dev_111b07c` | 0.000 | 2026-07-25T23:40:27.504802+00:00 | GARDER | récent `handover.md:3962` |
| `$R/experiments/ranker_v4_e1_250a05f` | 0.000 | 2026-07-26T13:13:15.418447+00:00 | GARDER | récent `handover.md:3861` |
| `$R/experiments/retrieval_all_human_v11/invalid_pre_b265713` | 0.001 | 2026-08-23T15:57:12.119315+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_all_human_v11/oracle_oof_train234` | 0.001 | 2026-08-24T01:05:41.291461+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_all_human_v11` **[fichiers directs]** | 13.327 | 2026-08-23T23:36:28.875618+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/audits/retrieval_v23_features/bb12058773f80d6f/manifest.json` |
| `$R/experiments/retrieval_dossier_v3_full_human` | 6.294 | 2026-08-19T19:36:37.772878+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/retrieval_ltr_v9_true_union/train_features_v9.parquet.manifest.json` |
| `$R/experiments/retrieval_dossier_v3_temporal_dev` | 1.498 | 2026-08-20T13:48:15.737960+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_dossier_v3_temporal_truthful_dev` | 1.401 | 2026-08-20T23:43:28.669459+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_fold_fused_parent_v12_arrow` | 0.002 | 2026-08-24T12:45:03.067679+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_ltr_v10_oof_shallow3` | 0.016 | 2026-08-21T22:35:53.344707+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_ltr_v6` | 0.407 | 2026-08-21T17:46:41.465906+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/retrieval_ltr_v7/train_features_v7.parquet.manifest.json` |
| `$R/experiments/retrieval_ltr_v7` | 0.073 | 2026-08-21T18:21:59.436078+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_ltr_v8_fullneg` | 0.092 | 2026-08-21T20:33:10.037240+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_ltr_v9_true_union` | 0.087 | 2026-08-21T21:47:59.020638+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_official_v4` | 0.019 | 2026-08-21T10:22:32.944915+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_open_data_v5` | 0.040 | 2026-08-21T16:16:15.894747+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_open_data_v6` | 1.823 | 2026-08-21T20:32:20.116596+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_selective_v4` | 0.001 | 2026-08-21T23:26:46.119088+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_v41_dev_85f7674` | 0.002 | 2026-07-26T22:14:40.011482+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/retrieval_v41_dev_feede27` | 0.002 | 2026-07-26T22:38:13.446060+00:00 | GARDER | récent `handover.md:3817`; manifeste gardé : `$R/shadow/v4_1/runs/v41_shadow_f1058826_20260727_v3/manifest.json` |
| `$R/experiments/siren_dossier_smoke_v1` | 0.000 | 2026-08-18T12:37:21.377853+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/site_label_audit_dev_c33b80855f560074_ac971e0` | 0.000 | 2026-07-24T13:35:04.868622+00:00 | GARDER | récent `handover.md:4011`; manifeste gardé : `$R/benchmarks/qualification_v2/522351669d5313dc/manifest.json` |
| `$R/experiments/site_label_audit_test_c33b80855f560074_eb0e6a3` | 0.000 | 2026-07-25T22:59:22.911211+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/benchmarks/qualification_v2/356308b941bbc402/manifest.json` |
| `$R/experiments/site_label_audit_train_c33b80855f560074_a68f679` | 0.001 | 2026-07-24T13:57:42.803188+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/benchmarks/qualification_v2/f8af7e1da18fa94a/manifest.json` |
| `$R/experiments/sparse_curve_dev_c33b80855f560074_963160b` | 0.012 | 2026-07-24T07:23:05.749985+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/sparse_curve_dev_c33b80855f560074_ad6fdf3` | 0.012 | 2026-07-24T07:43:06.612271+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_diagnostic_dev_c33b80855f560074_5a0e67f/manifest.json` |
| `$R/experiments/sparse_curve_test_c33b80855f560074_eb0e6a3` | 0.012 | 2026-07-25T23:17:45.506060+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/admission_diagnostic_test_c33b80855f560074_eb0e6a3/manifest.json` |
| `$R/experiments/synthetic_augmented_bge_comparison_v1` | 0.000 | 2026-08-17T01:00:26.772563+00:00 | GARDER | récent `handover.md:756` |
| `$R/experiments/synthetic_augmented_bge_v1` | 2.139 | 2026-08-17T01:00:12.957210+00:00 | GARDER | récent `handover.md:755` |
| `$R/experiments/synthetic_augmented_xgb_v1` | 0.042 | 2026-08-17T02:54:40.378624+00:00 | GARDER | récent `handover.md:749` |
| `$R/experiments/synthetic_gt_overlay_channels_6281c9d1470f3913` | 0.987 | 2026-08-16T17:33:14.456388+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/synthetic_gt_overlay_channels_799bf5b289a0e943` | 1.000 | 2026-08-17T02:35:31.653338+00:00 | GARDER | récent `handover.md:829` |
| `$R/experiments/synthetic_gt_v7_channels_6281c9d1470f3913` | 1.283 | 2026-08-16T17:22:43.461605+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/synthetic_gt_v7_channels_799bf5b289a0e943` | 1.293 | 2026-08-17T02:41:04.677332+00:00 | GARDER | récent `handover.md:828` |
| `$R/experiments/v4_10b_structured_acceptor_development` | 0.002 | 2026-07-28T05:19:39.061010+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:46` |
| `$R/experiments/v4_12_acceptor_business_competition` | 0.024 | 2026-08-13T08:47:49.688921+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/v4_12_acceptor_clean_target` | 0.000 | 2026-07-31T16:14:57.671655+00:00 | GARDER | récent `handover.md:1413` |
| `$R/experiments/v4_12_acceptor_conservative` | 0.000 | 2026-07-31T16:58:38.289031+00:00 | GARDER | récent `handover.md:1305` |
| `$R/experiments/v4_12_acceptor_cpu_families` | 0.014 | 2026-08-13T09:03:11.270662+00:00 | GARDER | récent `handover.md:1236` |
| `$R/experiments/v4_12_acceptor_hard_weight` | 0.000 | 2026-07-31T15:19:34.223263+00:00 | GARDER | récent `handover.md:4645` |
| `$R/experiments/v4_12_acceptor_relational_features` | 0.000 | 2026-07-31T15:23:12.487333+00:00 | GARDER | récent `handover.md:4668` |
| `$R/experiments/v4_12_bge_groupwise/01e1049c16af2600` | 2.139 | 2026-08-15T04:52:38.102372+00:00 | GARDER | récent `handover.md:590` |
| `$R/experiments/v4_12_bge_groupwise/2b424777fbf2f02e` | 2.132 | 2026-08-15T06:59:04.251093+00:00 | GARDER | récent `handover.md:590` |
| `$R/experiments/v4_12_bge_groupwise/487186b0a9faaed4` | 2.131 | 2026-08-14T22:14:05.687234+00:00 | GARDER | récent `handover.md:590` |
| `$R/experiments/v4_12_bge_groupwise/9c5091071d727cb6` | 2.132 | 2026-08-15T09:24:25.840074+00:00 | GARDER | récent `handover.md:590` |
| `$R/experiments/v4_12_bge_groupwise/a79c8c3adb3ca3bc` | 2.132 | 2026-08-15T11:52:21.935671+00:00 | GARDER | récent `handover.md:590` |
| `$R/experiments/v4_12_bge_groupwise` **[fichiers directs]** | 0.000 | 2026-08-15T11:52:21.935885+00:00 | GARDER | récent `handover.md:590` |
| `$R/experiments/v4_12_bge_operational_secondary` | 0.000 | 2026-08-15T11:57:46.397392+00:00 | GARDER | récent `handover.md:596` |
| `$R/experiments/v4_12_bge_xgb_stack` | 0.004 | 2026-08-15T11:53:12.060625+00:00 | GARDER | récent `handover.md:594` |
| `$R/experiments/v4_12_conservative_ensemble` | 0.001 | 2026-08-13T08:27:39.635437+00:00 | GARDER | récent `handover.md:4712` |
| `$R/experiments/v4_12_corrected_acceptor_family_weight` | 0.000 | 2026-07-31T16:02:35.053216+00:00 | GARDER | récent `handover.md:1466` |
| `$R/experiments/v4_12_corrected_acceptor_hard_weight` | 0.000 | 2026-07-31T16:00:22.270072+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/v4_12_corrected_label_ranker` | 0.002 | 2026-07-31T15:57:27.416531+00:00 | GARDER | récent `handover.md:1462` |
| `$R/experiments/v4_12_corrected_label_stack` | 0.001 | 2026-07-31T16:00:08.649878+00:00 | GARDER | récent `handover.md:1464` |
| `$R/experiments/v4_12_cross_encoder_reranker` | 0.001 | 2026-08-13T08:19:02.434291+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/v4_12_generic_business_rules` | 0.009 | 2026-08-13T08:44:47.891830+00:00 | GARDER | récent `handover.md:1258` |
| `$R/experiments/v4_12_hard_label_ranker_replica` | 0.002 | 2026-07-31T15:00:02.976196+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/v4_12_hard_label_ranker` | 0.008 | 2026-07-31T14:57:54.118105+00:00 | GARDER | récent `handover.md:4576` |
| `$R/experiments/v4_12_learned_acceptor` | 0.019 | 2026-08-13T17:19:03.234712+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:50` |
| `$R/experiments/v4_12_learned_cross_encoder_pilot` | 0.458 | 2026-08-13T17:38:02.810465+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/v4_12_learned_fresh_admission` | 0.000 | 2026-08-13T15:54:46.853430+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/evaluations/v4_12_learned_unified_retrieval/cce1bc83f82a1c3f/manifest.json` |
| `$R/experiments/v4_12_learned_fresh_channels_overlay` | 0.006 | 2026-08-13T15:54:27.301578+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/v4_12_learned_fresh_admission/1465a04be44b13c3/manifest.json` |
| `$R/experiments/v4_12_learned_fresh_channels_v7` | 0.014 | 2026-08-13T15:54:03.781961+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/v4_12_learned_fresh_admission/1465a04be44b13c3/manifest.json` |
| `$R/experiments/v4_12_learned_oof_rankers` | 0.671 | 2026-08-13T17:03:09.680045+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:49` |
| `$R/experiments/v4_12_neural_groupwise_cross_encoder` | 2.519 | 2026-08-14T12:23:42.340946+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/v4_12_neural_groupwise_qwen` | 0.001 | 2026-08-14T12:24:10.237574+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/v4_12_neural_zero_shot_external_tmp_check` | 0.000 | 2026-08-14T08:33:54.771980+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/v4_12_neural_zero_shot` | 0.031 | 2026-08-13T22:04:05.815401+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/v4_12_ranker_acceptor_stack` | 0.003 | 2026-07-31T15:12:48.776061+00:00 | GARDER | récent `handover.md:4620` |
| `$R/experiments/v4_12_ranker_business_features_local241` | 0.166 | 2026-08-13T08:36:38.558662+00:00 | GARDER | récent `handover.md:4761` |
| `$R/experiments/v4_12_ranker_business_features` | 0.010 | 2026-08-13T07:53:35.484540+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/v4_12_trusted_acceptor` | 0.001 | 2026-07-31T16:54:06.497985+00:00 | GARDER | récent `handover.md:1321` |
| `$R/experiments/v4_12_trusted_label_ranker` | 0.004 | 2026-07-31T16:47:38.221271+00:00 | GARDER | récent `handover.md:1335` |
| `$R/experiments/v4_12_two_stage_business_reranker` | 0.003 | 2026-08-13T07:56:56.738323+00:00 | GARDER | récent `docs/production_cost_study.json:312` |
| `$R/experiments/v4_2_retrieval_integrity_7c4b957` | 0.001 | 2026-07-27T10:36:25.082144+00:00 | GARDER | récent `handover.md:3793`; manifeste gardé : `$R/datasets/v4_6_aligned_a/301b24f47820f992/manifest.json` |
| `$R/experiments/v4_8_acceptor_development` | 0.000 | 2026-07-28T03:29:05.554196+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:46`; manifeste gardé : `$R/experiments/v4_8_random_holdout/f1ac35f4f7450b6a/manifest.json` |
| `$R/experiments/v4_8_random_holdout` | 0.000 | 2026-07-28T03:41:37.663150+00:00 | GARDER | récent `handover.md:3418` |
| `$R/experiments/v4_9_site_function_retrospective` | 0.000 | 2026-07-28T03:51:15.803078+00:00 | GARDER | récent `handover.md:3404` |
| `$R/experiments` **[fichiers directs]** | 0.000 | 2026-08-24T12:45:03.068409+00:00 | GARDER | récent `docs/production_cost_study.json:312`; manifeste gardé : `$R/experiments/synthetic_gt_overlay_channels_799bf5b289a0e943/manifest.json` |
| `$R/final_evaluations` | 0.008 | 2026-07-26T14:43:54.239630+00:00 | GARDER | récent `handover.md:4457`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/final_holdout_inputs` | 0.000 | 2026-07-26T14:35:07.883615+00:00 | DOUTE | manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/fresh_holdout_intake_authorities` | 0.000 | 2026-07-31T01:45:48.882905+00:00 | DOUTE | Contenu label/jeu fermé possible : non ouvert, laissé en place. |
| `$R/fresh_holdout_intake_synthetic_r2` | 0.145 | 2026-07-30T19:49:50.288561+00:00 | GARDER | récent `handover.md:2492` |
| `$R/fresh_holdout_intake_synthetic_r3` | 0.145 | 2026-07-30T21:18:26.221859+00:00 | DOUTE | Contenu label/jeu fermé possible : non ouvert, laissé en place. |
| `$R/fresh_holdout_intake_synthetic` | 0.160 | 2026-07-30T18:40:11.294018+00:00 | DOUTE | manifeste gardé : `$R/revoked_prelaunch_s0/f918b8af6c9dc47bc61bcb6ab36d0808704206a28865f8fa32b629b1a32d59e2/runtime/komapnlggeolpcjifjmnifnpbgjkmkbmcenjpggeddpokkcgomdgibgoekldonkc/private_runtime_manifest.json` |
| `$R/fresh_holdout_intake` | 0.014 | 2026-07-30T13:24:26.605533+00:00 | DOUTE | Sources, jeux réservés ou autorités historiques ; conservation par précaution. |
| `$R/gates` | 0.000 | 2026-07-27T23:45:08.789592+00:00 | GARDER | récent `handover.md:3498` |
| `$R/host_cache` | 1.522 | 2026-10-10T10:44:22.594795+00:00 | DOUTE | Cache : reconstruction hors ligne et absence de réutilisation non démontrées. |
| `$R/imports` | 0.014 | 2026-08-17T13:31:45.448249+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/indexes/hierarchical_retrieval_v1` | 0.000 | 2026-08-18T21:47:33.038590+00:00 | GARDER | récent `docs/production_cost_study.json:214` |
| `$R/indexes/hierarchical_retrieval_v3/4b332899d8698354` | 98.090 | 2026-08-19T12:46:06.716397+00:00 | GARDER | V142 : 353 artefacts; protection explicite; récent `docs/retrieval_engine_explained_20261014.md:71`; manifeste gardé : `$H/v173/site_target_timing_20261015/queries/41d6ee66737219abd20c35cd7b9fd294a55619d7b5dacdc3f1830ea5a2dc4702/binding.json` |
| `$R/indexes/hierarchical_retrieval_v3` **[fichiers directs]** | 0.000 | 2026-08-19T12:46:06.716886+00:00 | GARDER | récent `docs/retrieval_engine_explained_20261014.md:71`; manifeste gardé : `$R/experiments/retrieval_dossier_v3_temporal_dev/raw_union_dev_temporal.parquet.manifest.json` |
| `$R/indexes/official_evidence_overlay_full_v3` | 0.000 | 2026-08-19T22:08:00.746480+00:00 | GARDER | récent `docs/production_cost_study.json:214` |
| `$R/indexes/official_evidence_overlay_history_v1` | 0.000 | 2026-08-19T22:07:24.997444+00:00 | GARDER | récent `docs/production_cost_study.json:214` |
| `$R/indexes/official_evidence_overlay_sirene_history_v1` | 0.000 | 2026-08-20T12:27:50.441995+00:00 | GARDER | récent `handover.md:8530`; manifeste gardé : `$R/experiments/retrieval_dossier_v3_temporal_dev/raw_union_dev_temporal.parquet.manifest.json` |
| `$R/indexes/official_evidence_overlay_v1` | 0.037 | 2026-08-18T09:22:52.749889+00:00 | GARDER | récent `handover.md:5508`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/prepared/manifest.json` |
| `$R/indexes/official_evidence_overlay_v2` | 0.059 | 2026-08-18T09:32:57.023101+00:00 | GARDER | récent `handover.md:5521`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/prepared/manifest.json` |
| `$R/indexes/retrieval_sidecars_v11` | 1.701 | 2026-08-22T09:40:19.730210+00:00 | GARDER | récent `docs/production_cost_study.json:214`; manifeste gardé : `$R/audits/retrieval_v23_features/bb12058773f80d6f/manifest.json` |
| `$R/indexes/retrieval_v12_mmap_sidecar` | 4.844 | 2026-08-24T11:47:44.131077+00:00 | GARDER | récent `docs/production_cost_study.json:214` |
| `$R/indexes/retrieval_v12_parent_signal_capability` | 0.000 | 2026-08-24T12:12:38.881963+00:00 | GARDER | récent `docs/production_cost_study.json:214` |
| `$R/indexes/retrieval_v12_scene_ir` | 0.000 | 2026-08-24T11:47:37.673674+00:00 | GARDER | récent `docs/production_cost_study.json:214`; manifeste gardé : `$R/audits/retrieval_v23_features/bb12058773f80d6f/manifest.json` |
| `$R/indexes/retrieval_v43_spatial` | 1.377 | 2026-10-02T15:55:10.552037+00:00 | GARDER | récent `handover.md:7657` |
| `$R/indexes/v4_12_snapshot_lookup` | 2.545 | 2026-07-28T14:00:02.861192+00:00 | GARDER | récent `handover.md:3259` |
| `$R/indexes` **[fichiers directs]** | 0.000 | 2026-10-02T15:55:10.551900+00:00 | GARDER | récent `docs/production_cost_study.json:214` |
| `$R/indices/dossier_temporal_tail_v1` | 0.321 | 2026-08-21T16:02:50.305139+00:00 | GARDER | récent `handover.md:5890`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/prepared/manifest.json` |
| `$R/indices/hierarchical_v2_smoke_legacy` | 0.002 | 2026-08-18T14:10:19.125730+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/indices/hierarchical_v3_national_smoke` | 0.003 | 2026-08-18T20:37:34.025804+00:00 | GARDER | récent `handover.md:5737` |
| `$R/indices/hierarchical_v3_smoke` | 0.002 | 2026-08-18T18:08:22.860989+00:00 | GARDER | récent `handover.md:5680`; manifeste gardé : `$R/smokes/retrieval_v3_union/tahiti.parquet.manifest.json` |
| `$R/indices/official_sector_data_es_v1` | 0.169 | 2026-08-21T15:52:36.154330+00:00 | GARDER | récent `handover.md:5884`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/prepared/manifest.json` |
| `$R/indices/official_sector_v1` | 1.054 | 2026-08-21T08:51:50.009182+00:00 | GARDER | manifeste gardé : `$R/audits/retrieval_v34_fast_resume/prepared/manifest.json` |
| `$R/indices/rne_legal_entity_roles_v1` | 1.563 | 2026-08-21T09:06:11.641342+00:00 | GARDER | récent `docs/retrieval_frozen_v142.json:29835`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/prepared/manifest.json` |
| `$R/indices/sirene_temporal_v1/506bb98cfdcccfbe` | 14.554 | 2026-08-20T21:41:55.370790+00:00 | GARDER | récent `handover.md:5840`; manifeste gardé : `$R/audits/retrieval_v34_fast_resume/prepared/manifest.json` |
| `$R/indices/sirene_temporal_v1_smoke` | 0.001 | 2026-08-20T21:24:35.994104+00:00 | GARDER | récent `handover.md:5824` |
| `$R/indices/sirene_temporal_v1` **[fichiers directs]** | 0.000 | 2026-08-20T21:41:55.371135+00:00 | GARDER | récent `handover.md:5840` |
| `$R/indices` **[fichiers directs]** | 0.000 | 2026-08-21T16:02:09.125240+00:00 | DOUTE | Fichiers directs/métadonnées du parent : pas une cible de suppression. |
| `$R/inputs` | 0.001 | 2026-07-28T15:47:40.090487+00:00 | GARDER | récent `handover.md:2657`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/logs` | 0.000 | 2026-08-14T22:16:43.028592+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/models/acceptor_v4` | 0.002 | 2026-07-26T14:23:36.480410+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:43` |
| `$R/models/contrastive_siren_rescue_v1` | 0.001 | 2026-08-21T09:16:36.307952+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/models/downstream` | 0.078 | 2026-07-26T01:36:22.831817+00:00 | GARDER | récent `docs/retrieval_composite_admission_static_audit_20261014.json:18` |
| `$R/models/huggingface/hub` | 8.745 | 2026-08-13T21:58:57.532594+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/models/huggingface/xet` | 0.000 | 2026-08-13T07:58:28.270621+00:00 | DOUTE | récent `handover.md:15846` |
| `$R/models/huggingface` **[fichiers directs]** | 0.000 | 2026-08-13T07:58:28.092796+00:00 | DOUTE | Fichiers directs/métadonnées du parent : pas une cible de suppression. |
| `$R/models/ranker_v4` | 0.030 | 2026-07-26T13:11:11.953259+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:41` |
| `$R/models/retrieval_fold_fused_parent_v12` | 0.000 | 2026-08-24T11:55:06.591015+00:00 | DOUTE | Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide. |
| `$R/models/retrieval_ltr_v10_oof_shallow3` | 0.000 | 2026-08-21T22:35:29.604427+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/models/retrieval_ltr_v6_score_first` | 0.001 | 2026-08-21T16:47:30.324411+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/models/retrieval_ltr_v6` | 0.001 | 2026-08-21T16:59:55.246277+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/models/retrieval_ltr_v7_union_rank` | 0.001 | 2026-08-21T18:21:06.018170+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/models/retrieval_ltr_v8_fullneg` | 0.002 | 2026-08-21T19:26:10.504676+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/models/retrieval_ltr_v9_true_union` | 0.002 | 2026-08-21T21:47:44.777569+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `$R/models/v4_11_acceptor` | 0.000 | 2026-07-28T08:10:00.173257+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:47`; manifeste gardé : `$R/evaluations/v4_12_guard_historical/fedcd1d512bfd269/manifest.json` |
| `$R/models/v4_11_ranker_c` | 0.045 | 2026-07-28T07:53:02.889635+00:00 | GARDER | récent `docs/step2_downstream_inventory.md:45`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/models/v4_1` | 0.028 | 2026-07-26T23:04:10.351194+00:00 | GARDER | récent `handover.md:3819`; manifeste gardé : `$R/evaluations/v4_12_guard_historical/fedcd1d512bfd269/manifest.json` |
| `$R/models/v4_6_aligned_ranker` | 0.018 | 2026-07-28T02:38:23.872417+00:00 | GARDER | récent `handover.md:3482`; manifeste gardé : `$R/models/v4_11_ranker_c/0d2e419158c7a4c0/manifest.json` |
| `$R/models` **[fichiers directs]** | 0.000 | 2026-08-24T11:49:53.333894+00:00 | DOUTE | Fichiers directs/métadonnées du parent : pas une cible de suppression. |
| `$R/oracles` | 0.000 | 2026-07-28T16:38:47.724769+00:00 | GARDER | récent `handover.md:2689`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/pytest-s0-audit-final-c64` | 0.004 | 2026-07-30T16:34:18.370020+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-audit-final` | 0.004 | 2026-07-30T16:31:23.886163+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-audit-full` | 0.003 | 2026-07-30T16:21:53.113820+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-audit-latest2` | 0.001 | 2026-07-30T16:25:15.586450+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-audit-latest` | 0.001 | 2026-07-30T16:24:25.253203+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-audit` | 0.000 | 2026-07-30T16:21:44.007659+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-core-final2` | 0.000 | 2026-07-30T16:40:04.658039+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-core-final` | 0.000 | 2026-07-30T16:39:57.849147+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-evaluator-2` | 0.003 | 2026-07-30T16:18:20.398809+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-evaluator-3` | 0.000 | 2026-07-30T16:19:51.000948+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-evaluator-4` | 0.000 | 2026-07-30T16:23:06.744993+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-evaluator-5` | 0.001 | 2026-07-30T16:25:21.309960+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-evaluator-final` | 0.004 | 2026-07-30T16:33:30.951679+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-evaluator-stable` | 0.004 | 2026-07-30T16:31:27.679362+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/pytest-s0-evaluator` | 0.003 | 2026-07-30T16:17:44.801018+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `$R/references` | 0.008 | 2026-07-31T03:45:00.944042+00:00 | GARDER | récent `handover.md:1803`; manifeste gardé : `$R/audits/v4_12_review_adjudication_pilot/c7a9feecaf2d3c2a/manifest.json` |
| `$R/registries` | 0.011 | 2026-07-30T13:23:11.381748+00:00 | GARDER | récent `handover.md:3311`; manifeste gardé : `$R/fresh_holdout_intake/registry/consumed_compatibility/48851668dd2f173686f3240ecc62e30fcbfdb96d8abf0ced498eb29891d8a490/provenance.json` |
| `$R/releases` | 0.000 | 2026-07-26T14:34:57.317117+00:00 | GARDER | récent `handover.md:4456` |
| `$R/retrieval_v4` | 0.006 | 2026-07-26T12:51:54.315291+00:00 | GARDER | récent `handover.md:3872` |
| `$R/revoked_prelaunch_s0` | 0.478 | 2026-07-30T18:39:56.132195+00:00 | DOUTE | Contenu label/jeu fermé possible : non ouvert, laissé en place. |
| `$R/runs` | 0.042 | 2026-07-31T05:48:49.800939+00:00 | GARDER | récent `handover.md:1739`; manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `$R/shadow` | 0.606 | 2026-07-26T21:55:47.040344+00:00 | GARDER | récent `handover.md:3821`; manifeste gardé : `$R/audits/v4_1_representative_summary/2d18ef172f32aefc/manifest.json` |
| `$R/smokes` | 0.006 | 2026-08-18T21:50:38.366937+00:00 | GARDER | manifeste gardé : `$R/indices/hierarchical_v3_national_smoke/3be273936b9f971b/manifest.json` |
| `$R/sources/dossier_temporal_tail_v1` | 0.050 | 2026-08-21T16:01:27.110897+00:00 | GARDER | récent `handover.md:5890`; manifeste gardé : `$R/indices/dossier_temporal_tail_v1/66f38cc146c51da6/manifest.json` |
| `$R/sources/official_config` | 0.000 | 2026-08-18T09:28:24.301528+00:00 | GARDER | récent `docs/production_cost_study.json:234` |
| `$R/sources/official_evidence/bodacc_yearly_v1` | 4.710 | 2026-08-18T10:55:37.916108+00:00 | GARDER | récent `handover.md:5651`; manifeste gardé : `$H/v152/bodacc_structure_scope_20261014_night_b/source_sample_manifest.json` |
| `$R/sources/official_evidence/official_2026-08-18_v2` | 0.005 | 2026-08-18T09:32:35.112298+00:00 | GARDER | récent `handover.md:5518`; manifeste gardé : `$R/indexes/official_evidence_overlay_v2/338cd52006a9c30f/manifest.json` |
| `$R/sources/official_evidence/official_history_2008_2026_v1` | 4.569 | 2026-08-18T10:58:03.645838+00:00 | GARDER | récent `handover.md:5651`; manifeste gardé : `$R/sources/siren_dossier_v3/01eb624cca97c6f3/manifest.json` |
| `$R/sources/official_evidence/rne_2026-08-18_v1` | 0.002 | 2026-08-18T09:22:34.303173+00:00 | GARDER | récent `handover.md:5651`; manifeste gardé : `$R/sources/siren_dossier_v3/01eb624cca97c6f3/manifest.json` |
| `$R/sources/official_evidence/rne_bulk_formalities_20260304_v3` | 3.720 | 2026-08-18T18:24:33.692943+00:00 | GARDER | récent `handover.md:5651`; manifeste gardé : `$R/audits/retrieval_v74_coherent_proof/probe_20261003_a/extraction_manifest.json` |
| `$R/sources/official_evidence/sirene_history_20260801_v1_partitioned` | 7.431 | 2026-08-20T10:08:49.621335+00:00 | GARDER | récent `handover.md:5651` |
| `$R/sources/official_evidence/sirene_history_20260801_v1` | 8.799 | 2026-08-20T00:55:13.531139+00:00 | GARDER | récent `handover.md:5651`; manifeste gardé : `$R/sources/official_evidence/sirene_history_20260801_v2_partitioned/manifest.json` |
| `$R/sources/official_evidence/sirene_history_20260801_v2_partitioned` | 0.865 | 2026-08-20T11:47:51.200793+00:00 | GARDER | récent `handover.md:5651`; manifeste gardé : `$R/indexes/official_evidence_overlay_sirene_history_v1/5bb04044578179db/manifest.json` |
| `$R/sources/official_evidence` **[fichiers directs]** | 0.000 | 2026-08-20T11:47:51.200885+00:00 | GARDER | récent `handover.md:5651`; manifeste gardé : `$R/sources/official_evidence/official_history_2008_2026_v1/manifest.json` |
| `$R/sources/official_projection` | 0.001 | 2026-08-18T09:22:13.579902+00:00 | GARDER | récent `docs/production_cost_study.json:234`; manifeste gardé : `$R/sources/official_evidence/official_2026-08-18_v2/manifest.json` |
| `$R/sources/official_sync/bodacc` | 8.537 | 2026-08-18T10:20:52.767656+00:00 | GARDER | récent `handover.md:5610`; manifeste gardé : `$H/v152/bodacc_structure_scope_20261014_night_b/source_sample_manifest.json` |
| `$R/sources/official_sync/rne-ftp-bulk` | 18.099 | 2026-08-18T13:27:24.627690+00:00 | GARDER | récent `handover.md:5610`; manifeste gardé : `$R/audits/retrieval_v39_relational_proofs/rne_projection_20261002_a/manifest.json` |
| `$R/sources/official_sync/rne_backfill_v1` | 0.000 | 2026-08-18T12:36:02.994231+00:00 | GARDER | récent `handover.md:5610` |
| `$R/sources/official_sync/rne_ftp_inventory` | 0.000 | 2026-08-18T12:53:13.942288+00:00 | GARDER | récent `handover.md:5610` |
| `$R/sources/official_sync/rne` | 0.160 | 2026-08-18T12:53:38.912952+00:00 | GARDER | récent `handover.md:5610`; manifeste gardé : `$R/sources/official_sync/rne_backfill_v1/receipt.json` |
| `$R/sources/official_sync` **[fichiers directs]** | 0.000 | 2026-08-18T12:55:23.440956+00:00 | GARDER | récent `handover.md:5610` |
| `$R/sources/open_data_v5` | 0.331 | 2026-08-21T15:51:20.051136+00:00 | GARDER | récent `docs/production_cost_study.json:234`; manifeste gardé : `$R/audits/retrieval_v308_data_es_source/compile_20261005_a/manifest.json` |
| `$R/sources/rne_account_deposits_v1` | 0.382 | 2026-08-18T13:40:25.742875+00:00 | GARDER | récent `docs/production_cost_study.json:234` |
| `$R/sources/rne_legal_entity_roles_v1` | 0.116 | 2026-08-21T09:04:17.673454+00:00 | GARDER | V142 : 1 artefacts; récent `docs/retrieval_frozen_v142.json:29835`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/covered-prepare_fc2a89a72cbd6bd79ce81de33f43bb485b265a6d4ee5b2370bfdfc0da97577ad_59fc77a558/numeric/v57/context_receipt.json` |
| `$R/sources/sector_official_v1` | 0.124 | 2026-08-21T08:47:14.849950+00:00 | GARDER | V142 : 2 artefacts; récent `docs/retrieval_frozen_v142.json:29843`; manifeste gardé : `$H/v174/train3000_frozen_v142_20261017/extension_20261017/brief10/covered-prepare_fc2a89a72cbd6bd79ce81de33f43bb485b265a6d4ee5b2370bfdfc0da97577ad_59fc77a558/numeric/v32/parent_source_receipt.json` |
| `$R/sources/siren_dossier_retrieval_v2_smoke` | 0.000 | 2026-08-18T14:10:07.342180+00:00 | GARDER | récent `docs/production_cost_study.json:234`; manifeste gardé : `$R/indices/hierarchical_v2_smoke_legacy/f66f58ca2b635a6f/manifest.json` |
| `$R/sources/siren_dossier_retrieval_v3_full` | 8.399 | 2026-08-19T10:24:20.231049+00:00 | GARDER | récent `docs/retrieval_miss_access_vs_ranking_20261014.json:4590`; manifeste gardé : `$R/indices/official_sector_data_es_v1/37ea905013a83660/manifest.json` |
| `$R/sources/siren_dossier_retrieval_v3_smoke` | 0.000 | 2026-08-18T18:08:08.248682+00:00 | GARDER | récent `docs/production_cost_study.json:234`; manifeste gardé : `$R/indices/hierarchical_v3_smoke/8317d02c56ca3587/manifest.json` |
| `$R/sources/siren_dossier_v1` | 7.006 | 2026-08-18T12:36:28.571506+00:00 | GARDER | récent `handover.md:5555` |
| `$R/sources/siren_dossier_v3/01eb624cca97c6f3` | 20.290 | 2026-08-18T20:35:03.577157+00:00 | GARDER | récent `handover.md:5726`; manifeste gardé : `$R/audits/retrieval_v94_virtual_national_names/projection_20261003_a/manifest.json` |
| `$R/sources/siren_dossier_v3` **[fichiers directs]** | 0.000 | 2026-08-18T20:35:03.577537+00:00 | GARDER | récent `handover.md:5726` |
| `$R/sources/sirene_history_20260801` | 1.718 | 2026-08-19T21:58:20.227945+00:00 | GARDER | récent `docs/production_cost_study.json:234`; manifeste gardé : `$R/sources/official_evidence/sirene_history_20260801_v1/manifest.json` |
| `$R/sources/sirene_temporal_v1_smoke` | 0.000 | 2026-08-20T21:24:18.635246+00:00 | GARDER | récent `handover.md:5824` |
| `$R/sources/sirene_temporal_v1` | 0.693 | 2026-08-20T21:30:31.066031+00:00 | GARDER | récent `handover.md:5840`; manifeste gardé : `$R/indices/sirene_temporal_v1_smoke/b9f252b234f5499b/manifest.json` |
| `$R/sources/sirene` | 4.425 | 2026-08-17T19:14:19.015095+00:00 | GARDER | V142 : 3 artefacts; récent `docs/label_noise_review_v2.json:147`; manifeste gardé : `$H/step2_ranker_v3_group_20261020/result.json` |
| `$R/sources` **[fichiers directs]** | 0.000 | 2026-08-21T15:59:01.276145+00:00 | GARDER | récent `docs/production_cost_study.json:234` |
| `$R/stores/candidates_v7_current_crmgt_20260817` | 0.187 | 2026-08-17T15:17:09.830525+00:00 | DOUTE | Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide. |
| `$R/stores/candidates_v7_current_crmgt_full_20260817` | 7.368 | 2026-08-17T15:29:14.882007+00:00 | DOUTE | manifeste gardé : `$R/experiments/crm_gt_v2_v7_current_channels_5a9c2437a2a3d817_dev/manifest.json` |
| `$R/stores/legacy_closed_overlay_c33b80855f560074_e39fddd` | 0.844 | 2026-07-24T08:08:33.340311+00:00 | GARDER | manifeste gardé : `$R/experiments/closed_overlay_channel_audit_k5000_v4_dev_e566c25/manifest.json` |
| `$R/stores` **[fichiers directs]** | 0.000 | 2026-08-17T15:29:14.881768+00:00 | DOUTE | manifeste gardé : `$R/experiments/synthetic_gt_overlay_channels_799bf5b289a0e943/manifest.json` |
| `$R/work` | 0.000 | 2026-07-28T11:20:14.533447+00:00 | DOUTE | Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide. |
| `$R/workspace_recovery` | 0.000 | 2026-08-15T23:29:37.650972+00:00 | DOUTE | Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide. |
| `$R` **[fichiers directs]** | 0.000 | 2026-10-10T10:44:22.594707+00:00 | DOUTE | manifeste gardé : `$R/diag-r3-successor-gate.sfzj9buk/runtime/kbfkbicacgcgabcddiiacogfkndicooigeebcdaghpdgklgebocfhkinnniladkl/app/config/v4_12_fresh_intake_synthetic_scanner_sealer_plan.json` |
| `/Volumes/CATNAT_DATA/SIRETO_STORAGE_QUARANTINE` | 0.000 | 2026-10-20T17:27:45.363685+00:00 | GARDER | récent `handover.md:11320` |
| `/Volumes/CATNAT_DATA/SIRETO_V9/benchmarks` | 0.011 | 2026-07-23T15:32:29.684677+00:00 | GARDER | manifeste gardé : `$R/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a/manifest.json` |
| `/Volumes/CATNAT_DATA/SIRETO_V9/comparisons` | 0.000 | 2026-07-24T00:17:46.258379+00:00 | DOUTE | Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide. |
| `/Volumes/CATNAT_DATA/SIRETO_V9/dense` | 4.792 | 2026-07-23T19:17:24.897684+00:00 | GARDER | manifeste gardé : `$R/evaluations/hierarchical_retrieval_v1/diagnostics/dense_rescue_37/global_siren_hybrid100/manifest.json` |
| `/Volumes/CATNAT_DATA/SIRETO_V9/experiments` | 0.008 | 2026-07-24T00:08:33.005206+00:00 | DOUTE | Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide. |
| `/Volumes/CATNAT_DATA/SIRETO_V9/geo` | 0.986 | 2026-07-23T22:51:05.329450+00:00 | GARDER | manifeste gardé : `$R/evaluations/hierarchical_retrieval_v1/diagnostics/dense_rescue_37/global_siren_hybrid100/manifest.json` |
| `/Volumes/CATNAT_DATA/SIRETO_V9/models` | 0.447 | 2026-07-23T16:07:00.958790+00:00 | GARDER | manifeste gardé : `$R/evaluations/hierarchical_retrieval_v1/diagnostics/dense_rescue_37/global_siren_hybrid100/manifest.json` |
| `/Volumes/CATNAT_DATA/SIRETO_V9/plans` | 0.000 | 2026-07-23T16:07:54.590556+00:00 | GARDER | SHA256 du contenu référencé par un document récent ou manifeste gardé. |
| `/Volumes/CATNAT_DATA/SIRETO_V9/quality` | 0.000 | 2026-07-23T22:46:19.744392+00:00 | DOUTE | Dossier sans référence nominative : origine et reproductibilité non établies ; ne pas déduire obsolescence d’un grep vide. |
| `/Volumes/CATNAT_DATA/SIRETO_V9/stores` | 2.902 | 2026-07-23T23:32:31.691474+00:00 | GARDER | manifeste gardé : `$R/evaluations/hierarchical_retrieval_v1/diagnostics/dense_rescue_37/global_siren_hybrid100/manifest.json` |
| `/Volumes/CATNAT_DATA/SIRETO_V9` **[fichiers directs]** | 0.000 | 2026-08-24T10:51:00.076283+00:00 | DOUTE | Fichiers directs/métadonnées du parent : pas une cible de suppression. |
| `/Volumes/CATNAT_DATA/sireto-audit-fhk4nszm` | 0.000 | 2026-07-31T10:29:50.900150+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `/Volumes/CATNAT_DATA/sireto-native-anchor.3u44bv62` | 0.000 | 2026-07-31T11:09:11.811150+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-native-anchor2.1ln9o_3x` | 0.000 | 2026-07-31T11:09:19.518702+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-native-audit.CMWbvw` | 0.000 | 2026-07-31T10:59:06.333910+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-native-forge.i1duf0kv` | 0.000 | 2026-07-31T11:00:24.786588+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-native-forge2.rxbp5hon` | 0.000 | 2026-07-31T11:00:32.750720+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-native-forge3.ejphutqx` | 0.000 | 2026-07-31T11:00:40.967002+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-native-name-audit.oFeIVK` | 0.000 | 2026-07-31T10:59:46.918242+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-native-nouuid.tp2FFe` | 0.000 | 2026-07-31T10:59:55.647194+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-native-rereview.dZecJ6` | 0.000 | 2026-07-31T11:09:03.206934+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-s0-error-selfcheck.EHxKkj` | 0.000 | 2026-10-13T20:58:24.436378+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `/Volumes/CATNAT_DATA/sireto-s0-error-selfcheck.LGVHwb` | 0.000 | 2026-10-13T20:58:24.433238+00:00 | DOUTE | Pending/lien/fichier spécial/manifeste non inspecté dans le sous-arbre : suppression interdite. |
| `/Volumes/CATNAT_DATA/sireto-s0-fd-selfcheck.bQKNQZ` | 0.000 | 2026-07-30T16:28:19.061954+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-s0-final-selfcheck.NdaYs5` | 0.000 | 2026-07-30T16:29:42.616942+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-s0-selfcheck.MMNVt3` | 0.000 | 2026-07-30T16:17:22.732243+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-s0-stable-selfcheck.Qno6TZ` | 0.000 | 2026-07-30T16:30:32.130524+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-s0-stop-selfcheck.HrSLUw` | 0.000 | 2026-07-30T16:28:40.508492+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-systemvol-final.4rFcrc` | 0.000 | 2026-07-31T11:18:26.228131+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-diag.6jrpPN` | 0.000 | 2026-07-31T05:49:13.502923+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-diag.TnF6YR` | 0.000 | 2026-07-31T06:00:15.849298+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-diag.VmG6W9` | 0.000 | 2026-07-31T05:49:59.297958+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-diag.ymbnIR` | 0.000 | 2026-07-31T05:49:43.545449+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-diagtmp.4Smw9Z` | 0.000 | 2026-07-31T05:49:13.504569+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-diagtmp.a3NN4c` | 0.000 | 2026-07-31T06:00:15.850900+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-diagtmp.eUZBgJ` | 0.000 | 2026-07-31T05:49:59.299752+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-diagtmp.mFnd7r` | 0.000 | 2026-07-31T05:49:43.548810+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-parent.8pAlsx` | 0.000 | 2026-07-31T06:05:50.970865+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-parent.AsM8eD` | 0.000 | 2026-07-31T05:48:46.655435+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-parent.R1Va2b` | 0.000 | 2026-07-31T07:21:29.054525+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-parent.jEawJf` | 0.000 | 2026-07-31T05:56:06.258642+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-parent.o852O6` | 0.000 | 2026-07-31T06:31:17.842376+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-tmp.Ybor2A` | 0.000 | 2026-07-31T07:21:29.056257+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-tmp.cF4Del` | 0.000 | 2026-07-31T06:31:17.844019+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-tmp.cKedzd` | 0.000 | 2026-07-31T05:48:46.657061+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-tmp.xogeaK` | 0.000 | 2026-07-31T06:05:50.972751+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto-v412-tmp.yhvyZj` | 0.000 | 2026-07-31T05:56:06.260327+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |
| `/Volumes/CATNAT_DATA/sireto_m3_cli_smoke_20260731` | 0.002 | 2026-07-31T12:27:06.212729+00:00 | OBSOLÈTE | Répertoire ancien ; aucune mention dans les documents/scripts récents ni leurs imports. Références historiques seules publiées séparément ; dépendances JSON vérifiées ci-dessous. |

## Vérification réelle et script non exécuté

- Smoke réel : **14 dossiers re-mesurés**, 3 681 chemins re-statés, partition = somme des racines (690556340 KiB), correspondance exacte des cibles OBSOLÈTE. Résultat dans `verification` du JSON.
- Contrôle final réel complémentaire : **35/35 cibles mesurées individuellement avec `du -sk`, total 3 008 KiB**, 10/10 SHA de manifestes rééchantillonnés conformes, aucune entrée cible modifiée depuis le 14/10 ; liste explicite AST = classe OBSOLÈTE du JSON et SHA du plan exact.
- Syntaxe seulement : `bash -n scripts/disk_cleanup_apply.sh` et analyse AST du Python embarqué ; aucune exécution du script, même en dry-run, pendant cet audit.
- Le futur script est **dry-run par défaut**, et n’accepte que sa liste explicite sans joker. Il vérifie le SHA du plan/sceau/handover/manifestes et des sources protégées, les nouveaux commits de code/documentation, les chevauchements protégés, l’inventaire exact et les SHA du contenu ; refuse pending/liens/montages/fichiers nouveaux ou étrangers. `du -sk` doit être strictement égal avant chaque suppression.
- Après validation Nathan et arrêt des écritures SIRETO seulement : `bash scripts/disk_cleanup_apply.sh --dry-run`, puis éventuellement `bash scripts/disk_cleanup_apply.sh --apply --confirmed-by-nathan`. Aucun script d’application ne doit être exécuté automatiquement ; réaudit si les sources ou le code changent.
- SHA256 du JSON livré : `4d126ad17da8b531d8bdb445768f8acd9bbdfb41491658429f0bb5ae34892600`.
- Reproduction du smoke de taille, sans exécuter le script de suppression :

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import json, subprocess
from pathlib import Path
p = json.loads(Path('docs/disk_cleanup_plan.json').read_text())
for target in p['apply_targets']:
    actual = int(subprocess.check_output(['du', '-sk', target['path']], text=True).split()[0])
    assert actual == target['expected_kib'], target['path']
refs = p['frozen_v142']['artifacts']
present = sum(Path(r['path']).exists() for r in refs)
print(len(p['apply_targets']), 'cibles ;', sum(t['expected_kib'] for t in p['apply_targets']), 'KiB ;', present, '/', len(refs), 'autorités présentes')
PY
bash -n scripts/disk_cleanup_apply.sh
```

- Reproduction des recherches : extraire les noms/chemins distinctifs de `folders`, puis `git grep -l -I -P <alternance_bornee> -- .` (PCRE, arrêt au premier résultat par fichier) ; historique : `git log --since=2026-10-14T00:00:00+02:00 --raw --no-abbrev --format= -- docs scripts handover.md`, `git cat-file --batch` sur les blobs listés dans `git_reference_audit.historical_blobs`.
- Journaux de travail locaux : `/tmp/sireto-disk-{inventory,references,classify,complete}.log`. Les fichiers temporaires ne sont pas des dépendances du plan ni du script livré. Le contrôle réel a été effectué par `/tmp/sireto_disk_publish.py` ; le JSON conserve l’ensemble de ses comptes et les tailles échantillonnées.

## Commit

Commit isolé du plan, JSON, script et ligne handover : `git log -1 --format=%H -- docs/disk_cleanup_plan.json`. Pas de push.
