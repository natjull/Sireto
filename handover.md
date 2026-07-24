# SIRETO Handover - 23 Juillet 2026

## Etat des lieux
Le goal actif est **Retrieval SIRET Recall@100**: atteindre au moins 99,0 % de
Recall candidat au SIRET exact avec un plafond absolu de 100 candidats. Le
contrat est `docs/retrieval_recall100_contract.md`.

L'experience V9 sans GPU precedente est terminee avec une decision `PIVOT`.
Ses pools denses multicanaux ne sont pas promus. Le rapport reste
`reports/v9/v9_go_pivot_stop.md`.

V7/V8b et Route B restent physiquement disponibles comme baselines legacy.
Le ranker, le decider, le risk model et l'accepteur restent geles jusqu'au gate
Recall@100.

## Actions terminees (fenetre recente)
- **Préfixes Recall@K stabilisés et cache mutualisé**: un passage max-K ne
  classait pas les partitions de taille comprise entre 51 et K, rendant son
  préfixe @50 différent de la baseline. Ajout d'un seuil de déclenchement du
  ranking indépendant du budget final; le smoke reproduit désormais exactement
  les dix préfixes @50 historiques. Les matrices TF-IDF utilisent un hash
  d'artefact indépendant du cutoff avec fallback vers les 7,7 Go de cache
  legacy. Le premier run `..._963160b` est conservé mais déclaré diagnostic
  invalide pour les préfixes @50/@100/@200. Suite complète à 81 tests passants.
  *(commit GitHub: `bdc7ad4`)*
- **Instrumentation Recall@K et causes de perte**: séparation explicite des
  états avant filtre, après filtre et après déduplication dans le retrieval
  partagé; nouveau runner immuable calculant en un passage les préfixes
  @50/@100/@200/@500, intervalles, segments, latence, cardinalités et buckets
  de perte mutuellement exclusifs. Smoke réel et suite complète à 78 tests
  passants. *(commit GitHub: `5e3fd5f`)*
- **Contrat Recall@100 pre-enregistre**: cible SIRET exacte >=99,0 %, plafond
  absolu 100, courbes diagnostiques @50/@100/@200/@500, attribution obligatoire
  partition/filtre/deduplication/pruning, audit canal par canal, tuning
  train/dev et unique evaluation de la variante gelee sur test. `AGENTS.md`
  pointe desormais vers ce goal actif. *(commit GitHub: `8b77af3`)*
- **Decision finale V9 = PIVOT**: sparse + dense local perd 1,83 point de
  Recall@50 SIRET et sparse + dense global SIREN perd 2,61 points. Les deux
  regressions sont statistiquement nettes et violent les gates segmentaires.
  En revanche, leur Hit@1 SIRET brut gagne respectivement 7,33 et 11,31 points,
  avec des IC95 strictement positifs. Gate 3 ranker/accepteur, Gate 4 open-set
  et le cross-encoder ne sont pas ouverts sur le pool rejete. Le pivot propose
  une nouvelle ablation: pool sparse fixe + dense comme feature de scoring.
  Le Mac a execute l'ensemble sans GPU ni depense cloud. *(commit GitHub:
  `59ec78c`; lecture STOP preliminaire supersedee: `53ad3b3`)*
- **Hit@1 SIRET/SIREN publie par le runner**: les comparaisons appariées
  incluent désormais les Hit@1, leurs recuperations/deplacements, IC95
  bootstrap et tests de McNemar. Sur global, Hit@1 SIRET passe de 36,22 % a
  47,52 % et Hit@1 SIREN de 41,91 % a 53,92 %. Suite complete a 68 tests
  passants. *(commit GitHub: `de0079a`)*
- **Gate 2 dense global SIREN echouee sur dev**: sparse atteint 2 317/2 565 =
  90,33 % contre 2 250/2 565 = 87,72 % pour l'hybride global. Delta apparie
  −2,61 points, IC95 [−3,51; −1,72], McNemar p=1,47e-8; 37 misses recuperes
  mais 104 hits deplaces. Actifs −3,27 points et multi-sites −2,28 points.
  La latence p95 passe a 1,079x et le budget 50 est respecte. *(commit GitHub:
  `53ad3b3`)*
- **Audit de budget multicanal corrige**: deux sorties globales de 50 candidats
  etaient faussement marquees non conformes car leur seul pool local contenait
  41 ou 18 lignes. Le controle refuse maintenant les depassements de K et les
  underfills locaux, sans rejeter les candidats qui completent un pool court
  via un nouveau canal. Rapport global regenere avec zero violation; suite
  complete a 68 tests passants. *(commit GitHub: `bc49918`)*
- **Expansion globale SIREN bornee avant materialisation**: le store candidat
  v2 calcule la densite par zone et applique dans DuckDB un top-K par SIREN,
  ordonne par correspondance INSEE/CP puis densite locale, avant tout transfert
  vers Python. Le cap SQL est 40 et le cap metier final reste 20 SIRET par
  SIREN; le lecteur reste compatible avec les stores v1. Suite complete a
  66 tests passants. *(commit GitHub: `43f2c64`)*
- **Manifeste d'expérience dense fermé**: chaque run V9 référence désormais le
  hash du contrat de son store local, ANN global, géographie mmap et store
  candidat SIREN; les stores partitionnés sans manifeste racine sont liés par
  un hash agrégé déterministe de leurs  manifestes. Suite complète à 66 tests
  passants. *(commit GitHub: `fef3658`)*
- **Expansion globale SIREN rendue exécutable**: le smoke historique rechargeait
  jusqu'à des dizaines de partitions aléatoires par requête (p95 17,5 s sur
  cinq cas, contre 0,67 s sparse). Ajout d'un store DuckDB read-only indexé par
  SIREN qui récupère les 50 groupes de candidats en une requête, tout en
  conservant la priorité géographique et la limite de 20 SIRET par SIREN.
  La lecture Arrow conserve les `None`/listes des partitions, sans conversion
  pandas en `NaN`. Suite complète à 65 tests passants. *(commits GitHub:
  `13d66d2`, `654413f`)*
- **Lookup géographique SIREN compatible 24 Go**: remplacement optionnel du
  chargement legacy de 37,8 M lignes dans pandas/dict par un artefact trié,
  quatre tableaux NumPy mmap et une recherche binaire SIREN. Le builder DuckDB
  travaille sur SSD avec limite mémoire, publie hashes et cardinalité; le
  lecteur conserve la compatibilité avec le parquet historique. Suite complète
  à 64 tests passants. *(commit GitHub: `7781d31`)*
- **Index dense global SIREN construit sur Mac**: 28 982 797 unités légales
  encodées en CPU avec le modèle générique épinglé, puis indexées en IVFPQ
  4096/48; manifeste avec hash source, fingerprint modèle et hashes FAISS/IDs.
  Un contrôle reproductible échantillonne les row groups du parquet, vérifie
  l'intégrité des sorties et mesure self-recall@1/@50 et latence avant toute
  évaluation métier. Suite complète à 63 tests passants. *(commits GitHub:
  `2d74b2b`, `6718d8b`)*
- **Contrat dense global SIREN renforcé**: le builder publie désormais la
  progression d'encodage, le fingerprint intégral du modèle et les hashes des
  fichiers FAISS/IDs. Le benchmark et l'inférence V9 refusent un index construit
  avec un autre modèle avant même de charger FAISS. Suite complète à 62 tests
  passants. *(commit GitHub: `2d74b2b`)*
- **Gate 2 dense local échouée sur dev**: sparse atteint 90,33 % Recall@50
  SIRET contre 88,50 % pour sparse+dense local et 70,29 % pour dense seul.
  L'hybride récupère 45 misses mais déplace 92 hits: delta apparié −1,83 point,
  IC95 [−2,73; −0,94], p exact 0,000073. Budget et latence passent, mais actifs
  (−2,26), mégapoles (−3,03) et multi-sites (−2,28 points) violent le gate
  segmentaire. Les 168 misses sparse au niveau SIREN et 25 récupérations SIREN
  uniques par le dense justifient la dernière expérience globale SIREN, sans
  tuning opportuniste de RRF. *(commit GitHub: `71c68ef`)*
- **Store dense local dev complet**: les 871 partitions INSEE et 14 partitions
  CP du plan gelé ont été encodées sur CPU avec le MiniLM générique épinglé,
  soit 10 216 448 candidats dans 885 paires index/manifeste (3,0 Go sur SSD).
  La vérification exhaustive confirme un unique fingerprint modèle, le hash
  exact du plan, zéro fichier manquant/temporaire et 61 tests passants. Le
  builder cherchait initialement `cp_codes` au lieu du champ canonique
  `postcode_codes`; le défaut est corrigé et couvert par régression. *(commit
  GitHub: `8ec1881`)*
- **Comparateur apparié Gate 2**: validation des hashes de l'expérience et de
  l'alignement exact des requêtes, décompte des misses récupérés et hits
  déplacés, IC95 bootstrap apparié, test exact de McNemar, deltas par segment,
  ratio de latence p95 et refus explicite de toute violation du budget fixe.
  Le rapport JSON/Markdown produit est immuable et lié au manifeste de
  l'expérience; suite complète à 60 tests passants. *(commit GitHub:
  `86dea2c`)*
- **Dense local non contamine prepare**: fingerprint integral du modele
  semantique impose entre build et inference, revision generique MiniLM
  `86741b4e` copiee sans telechargement sur le SSD, reparation du tokenizer
  Unigram et plan de partitions immuable. Le plan dev couvre 871 partitions
  INSEE et 14 CP, environ 10,2 M de lignes physiques; aucune requete dev sans
  partition planifiable. *(commit GitHub: `10dd990`)*
- **Baseline sparse-50 V9 mesuree**: sur les 2 652 requetes test gelees,
  Recall@50 SIRET 88,54 % (2 348 hits, IC95 87,27–89,69), Recall@50 SIREN
  92,16 %, recall du pool geographique 98,00 %, zero violation de budget.
  Les 304 erreurs comprennent 53 absences de partition et 251 prunings; 96
  erreurs conservent le bon SIREN. Segments critiques: fermes 67,09 %,
  megapoles 77,01 %. Artefacts bruts hashes sur le SSD et rapport dans
  `reports/v9/retrieval_baseline_sparse50.md`. *(commit GitHub: `8adc5f3`)*
- **Runner retrieval V9 immuable**: execution sparse, hybride local, dense-only
  et hybride global SIREN avec budget final strict, preuves par requete,
  Recall SIRET/SIREN et Wilson 95/99 %, segments, latences p50/p95/p99, cache
  SSD borne en RAM et manifeste lie au commit. Le benchmark segmente v2
  `c33b80855f560074` remplace le build v1 pour les experiences; le v1 reste
  conserve. *(commit GitHub: `771beb6`)*
- **Benchmark ferme V9 gele**: reconstruction exacte du split V7 historique
  par SIREN (seed 42), validation contre les scenes positives V7, ajout des 692
  requetes historiquement absentes des scenes afin de compter les misses
  end-to-end, hash integral des 4 119 fichiers de partitions et des snapshots
  SIRENE. Build initial immuable `8967e72e07c9f4bf` puis revision segmentee
  `c33b80855f560074` sur le SSD externe: 11 837 train,
  2 565 dev, 2 652 test, zero SIREN partage. Les labels restent des verites CRM
  historiques non reaudites et le modele dense fine-tune local est declare
  contamine pour toute revendication finale sur ce corpus. *(commit GitHub:
  `b384509`)*
- **Gate 0 V9 sans GPU franchie**: cles d'index dense alignees sur les vraies
  partitions, refus des subsets mega-communes incompatibles, manifeste de
  cardinalite et d'ordre SIRET, isolation stricte de PyTorch et FAISS dans
  deux sous-processus persistants sans `KMP_DUPLICATE_LIB_OK`, builders local
  et global SIREN corriges, mode dense-only repare et entrypoints V9
  executables directement. Validation: 52 tests passants, smoke 512 lignes,
  index local reel de 17 462 candidats et index global SIREN de 1 000 entites
  construits/interroges avec succes sur CPU. *(commit GitHub: `88e97e0`)*
- **Contrat d'execution V9 sans GPU**: directive active `GO/PIVOT/STOP`
  placee en tete de `AGENTS.md`, ressources locales autorisees, ordre des
  experiences, gates et regles d'arret formalises dans
  `docs/v9_execution_contract.md`. Les descriptions V6/V7/V8 sont explicitement
  historiques et ne pilotent plus les travaux. *(commit GitHub: `72d2749`)*
- **Benchmark open-set, ablation cross-encoder et gates V9**: feuille
  d'adjudication stratifiee, validation humaine/evidence/snapshot obligatoire,
  gel adresse par hash, cross-encoder top-20 avec revision epinglee, gates
  retrieval/segments/latence/deploiement et guide d'execution. Les trois
  variantes cross-encoder produisent des predictions OOF compatibles avec le
  meme accepteur. *(commits GitHub: `c4cf99f`, `b82271e`)*
- **Ranker unique + accepteur selectif V9**: 54 features brutes partagees
  train/serve puis sous-ensemble manifeste, features retrieval/SIREN, ranker
  XGBoost avec predictions OOF, misses conserves, correction stricte SIRET,
  calibration et selection de seuil sur deux moities dev distinctes, comparaison
  logistique/XGBoost, moteur d'inference `AUTO_MATCH|REVIEW` compatible
  `routing_status`. L'injection de positifs est autorisee uniquement dans le fit
  ranker train et interdite dans les scenes/evaluations. *(commit GitHub:
  `db4ab27`)*
- **Retrieval hybride V9 a budget fixe**: RRF sparse/dense/rescue, vrais scores
  TF-IDF ordonnes, provenance/rangs par canal, configurations 50 et ablation 100,
  index dense global SIREN streaming avec manifeste/tokenizer, expansion limitee
  SIRET et benchmark p50/p95. *(commit GitHub: `36404ae`)*
- **Contrats et dataset canonique V9**: ajout du contrat public `AUTO_MATCH/REVIEW` avec mapping legacy, labels `MATCH_EXACT/NO_MATCH/AMBIGUOUS/UNRESOLVED`, split deterministe SIREN-disjoint, bundle parquet immuable adresse par hash, manifeste de provenance/config/tokenizer/features et registre explicite des artefacts legacy interdits aux entrypoints V9. *(commit GitHub: `afb0f3d`)*
- **Socle V9 semantique + prediction selective**: chargement lazy de SentenceTransformer, reparation runtime du tokenizer Unigram exporte comme BertTokenizer, healthcheck anti-`<unk>`, injection semantique partagee train/serve, remise en service du mining d'homonymes geographiques et primitives testees de courbe risque-couverture/certification binomiale. Suite de tests retablie a 20 tests passants. *(commit GitHub: `fcfc33f`)*
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
- `src/xgb_matcher/features.py` *(commits GitHub: `35fb441`, `fcfc33f`, `db4ab27`)*
- `scripts/generate_training_samples_v5fast.py` *(commits GitHub: `35fb441`, `c356923`, `1305012`, `c961371`, `fcfc33f`, `db4ab27`)*
- `scripts/train_xgb_decider.py` *(commit GitHub: `35fb441`)*
- `scripts/build_siren_global_index.py` *(commits GitHub: `3e090b7`, `c356923`)*
- `src/xgb_matcher/siren_retrieval.py` *(commit GitHub: `3e090b7`)*
- `src/xgb_matcher/infer.py` *(commits GitHub: `3e090b7`, `c356923`, `36404ae`)*
- `src/xgb_matcher/retrieval.py` *(commits GitHub: `1305012`, `9c0e806`, `f1fbbb8`, `36404ae`)*
- `src/xgb_matcher/retrieval_config.py` *(commits GitHub: `3e090b7`, `9c0e806`, `36404ae`)*
- `src/xgb_matcher/profile.py` *(commit GitHub: `3e090b7`)*
- `src/xgb_matcher/v9_dataset.py` *(commits GitHub: `afb0f3d`, `db4ab27`)*
- `src/xgb_matcher/v9_scene.py`, `v9_acceptor.py`, `v9_infer.py`
  *(commit GitHub: `db4ab27`)*
- `src/xgb_matcher/fusion.py`, `v9_features.py` *(commit GitHub: `36404ae`)*
- `src/xgb_matcher/v9_adjudication.py`, `v9_cross_encoder.py`,
  `v9_evaluation.py` *(commit GitHub: `c4cf99f`)*

## Travail en cours
- Mesurer en un seul passage les courbes Recall@50/@100/@200/@500 sur dev et
  attribuer chaque perte au premier stage responsable.
- Auditer ensuite les canaux nom, caracteres, adresse, cles exactes, rescue,
  SIREN et geographie avant toute nouvelle union.
- Les artefacts et baselines sont conserves; aucun nettoyage massif du SSD
  n'a ete effectue.

## Points d'attention
- **Plafond absolu 100**: les mesures @200/@500 sont diagnostiques et ne
  constituent jamais une configuration eligible.
- **Test**: sa baseline sparse historique est connue, mais aucune nouvelle
  variante n'y sera executee avant gel complet sur train/dev.
- **Modeles aval geles**: aucun changement ranker/decider/risk/accepteur avant
  Recall@100 dev >=99,0 %.
- **Decision PIVOT scopee**: elle invalide l'admission/fusion des candidats
  denses V9 testes, pas leur signal de scoring ni le pipeline sparse/XGBoost.
- **Comparaison retrieval uniquement a budget constant**: un gain avec 100
  candidats ne justifie pas la promotion de la variante 50.
- **Precision strictement SIRET**: un bon SIREN mais mauvais etablissement est
  une erreur pour l'accepteur.
- **NO_MATCH temporel**: toujours rattache au snapshot SIRENE et a la date de
  reference.
- **Cross-encoder conditionnel**: aucune promotion sans +1 point de couverture
  a precision cible et gates segments/latence. Il reste hors chemin critique et
  aucune location de GPU n'est autorisee.
- **Certification**: avant environ 2 300 AUTO independants audites sans erreur,
  publier une estimation observee, jamais une garantie a 99,8 %.
- **Governance docs**: garder `handover.md` comme journal de commits (regle AGENTS).

## Artefacts cibles (V9)
| Artefact | Chemin |
|----------|--------|
| Partitions candidates | `data/candidates_v7_all/` |
| Bundle canonique | `data/v9/<build_id>/{queries,labels,candidates}.parquet` |
| Manifeste dataset | `data/v9/<build_id>/manifest.json` |
| Mapping geo SIREN | `data/siren_index/siren_to_geo.parquet` |
| Index dense SIREN | `data/v9_indices/siren_dense_<snapshot>/` |
| Ranker + predictions OOF | `models/v9/ranker_<build_id>/` |
| Accepteur + calibration | `models/v9/acceptor_<build_id>/` |
| Benchmark open-set gele | `data/v9_open_set/<benchmark_id>/` |

## Prochaines etapes
1. Instrumenter les stages partition, filtre, deduplication et ranking.
2. Produire la courbe sparse dev @50/@100/@200/@500 avec hashes et erreurs.
3. Mesurer chaque canal seul puis les unions cumulatives sous plafond 100.
4. Ne consulter la nouvelle variante sur test qu'apres franchissement du gate
   dev et gel de son manifeste.

---
*Regle projet: chaque modification de code/metier doit citer son commit GitHub dans ce document.*
