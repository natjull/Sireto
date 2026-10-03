# V99 — recherche initiale100, noms et sites nationaux complets

Contrat avant mesure : exact99,2%, heap100 maximum, aucun pool2000/500 ni
reclassement aval. Folds2/3/4 seuls, fold0 non lu, fold1 fermé, aval gelé.
Aucun gain garanti. Comparateur V91 fixé avant mesure.

V91 limitait ses chemins de noms aux12531976 endpoints du graphe V84.
V99 emploie TOUS43896818 sites V96 pour les preuves directes et TOUS les
sites de chaque parent, sans cap de fanout ni rejet des fermés. Succession
V84 un hop typé, grades1/.5/.25 inchangés : pas identité ni co-validité
prouvée. Tous endpoints/parents V84 sont validés contre le registre complet.

Les57804962 tuples V97 et109305599 postings V100 proviennent des162537492
observations officielles intégrales. Mots réunis seulement dans le même
alias ; contexte SIREN SIGN et vides conservés sans postings, IDs masqués.
IDF sur tout le corpus, jamais sur le CRM/labels. Nom site non transmis aux
siblings. Observation de site absent du registre conservée mais sans site
fabriqué ; accès trié/ponctuel concordants. Dates/types/provenance/ordinals
scellés V97/V100 accessibles, NON utilisés comme co-validité dans le score.
Propriétaires/postings dans le dossier V98 référencé par V100, pas V100.

Score : max(natif V78 gelé,64 × preuve nom scopée × preuve XY).64 hypothèse
V87 non ajustée sur14 misses ;11 features natives et8 sémantiques OOF V77
gelées. Nom, numéro, voie, CP, commune/INSEE et vues officielles restent
actifs dans le natif. BAN/XY absents annulent seulement la nouvelle branche,
pas le natif ; BAN faible facteur.5 et distance Lambert93 inchangés. Activité,
catégorie et dates ne sont pas annoncées comme features actives inventées.

Trois flux exhaustifs adaptatifs avec borne globale certifiée, completion
ponctuelle exacte et heap100. Pages256/completion512 buffers physiques, pas
pools/caps métier. Dictionnaire3,6M mots compacté uint8/uint32 ; accès et
arithmétique identiques V86 (oracle bit exact).

Même128 CRM query-only V78, panel développement réutilisé NON holdout ni
fullOOF. Ni labels ni anciens candidats en génération ; évaluation séparée
seulement APRÈS128 scellés. Exact/op séparés, dénominateur all-human,
ledger/sidecar officiel hashés, labels inexploitables publiés séparément.
Préflight16 sans labels moyenne<=4s ; worker80s/6GioRSS/zéro swapout,
superviseur90s, contrôleur900s, requête20s. Aucun budget élargi ni recall128
partiel. Sources serving/codes hashés avant/après. Inventaire Tantivy vérifié,
gros segments non entièrement rehashés comme V38. Hashes/démarrage séparés
des temps requête ; timings V91 historiques non appariés, cache non contrôlé.

Oracles indépendants registre fixture complet, direct/parent/siblings,
relations, hors graphe, fermés, BAN faible/absent, accès trié/ponctuel. Données
synthétiques uniquement tests, jamais source/feature métier. Même100% pilote
ne valide pas fullOOF, Wilson/généralisation, attribution exhaustive ni
p95<=1s/p99<=2s. Aucun READY_FOR_FOLD1, aucune revendication SOTA.
