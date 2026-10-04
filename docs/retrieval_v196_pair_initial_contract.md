# V196 — différer les pré-hash CSR répétés, pas les supprimer

Protocole avant mesure. Même expérience195 positions0/1, population128 entière
vérifiée, corpus44403034sites/98804872faits/34346446variantes d'adresse,
quatre mêmes spans physiques. Mêmes preuves193/194, formule30features182/146,
kernel183, models114 et ownvector/head192 ; mêmequery/geo/num/CP/INSEE,
nomsparent/site/historique/onehop149, XY139 et étatfermé conservé. Aucun pool,
rescore, anciencandidat, GT, aliasCRM/IDfeature ou modèleaval modifié.

Le constructor178 FULL-hash les1024arrays CSR à CHAQUE ouverture de scorer,
via177.phase_record. Il relit11,6Go malgré l'audit complet après TOUSles scores,
et cela contribue aux~18-20s d'ouverture195. Nouveau DeferredHashAddresses196,
pas de patch des modules mesurés : même source réelle readonly, mêmes
1024arrays et256packs, backend177/load/ledger93507/STOP, rootseal/phase
certificatsoriginaux FULLhashés, véritables parents0/preflights/report153,
namespaces/counts/typedshapes/CSRfirst-last/epochs contrôlés intégralement.
Les certificats des phases et leurs quatre références doivent correspondre
EXACTEMENT au root177 scellé, pas à une reconstitution depuis les résultats.

Seule différence : les quatre GROS datahashes de chaquephase ne sont pas lus
avant le scoring. Ils restent des références originales immuables dans les
targets, avec leurs checksums complets. Reader en QUARANTAINE pendinghash ;
les statistiques de snapshot disent explicitement que ces hashes sont
différés, jamais source_hashes_checked_at_open=True. Chaque nouvelle lecture
réarme la quarantaine. verify_after hérité178 FULL-hash TOUS1024arrays et
revérifie les contrôles/epochs/STOP avant/après ; seul ce succès enlève la
quarantaine du lecteur. Mutation mêmeinode/size/mtime n'est PAS prétendue
détectée par epoch : un test démontre son rejet parFULLhashaprès.

Au niveau national, paquetcommun et audits195 gardent TOUSles mêmes56462
targets/epochs source/proofs193/194/attrs/querycontrols/3heads. Workers
FULLhash fact/postings/context APRÈS TOUSquatre scores, puis deuxréductions
identiques<=100 et ledger93507 aprèsmerge ; admission195-style liée aux
vrais parents et à toutes les receipts/producers/packets. Pas d'admission
fondée sur pendingFalse seul, pas de stat/headers substitués à unFULLhash,
pas de cible supprimée. Aucun candidat initial émis avant toutes ces étapes.

Budgetinchangé1Gio/80s, score30s, quiet30/p1/free>=50%, swapglobal télémétrie
non attribuée. Mac/CATNAT seuls, STOPimmuable/sansretry, pascloud/spend/purge/
autresapps/filesources supprimées. Publier coût total incluant lespreuves
193/194 et audits, même cacheincontrôlé, pas réduction promise avant mesure.
La sous-classe196 conserve aussi la query math178 pour l'oracle scalar tiny,
mais196national ne refaitpas lesrequêtes de preuve193/194.

Tests prévus : oracle entierfloat64/bool inchangé, touscertificats/typedCSR/
shapes/offsets/phaseparent actual0, toutesarrays réellementFULLhashées après
query, payloadsameepoch rejeté, falseseal/partial/truth/STOP/closed/timeout,
et intégritéadmission/paire/source/folds/budget/cap identiques195. Les petits
packs de tests sont réels ; supervision/pression fake explicitement, pasune
preuve nationale. Sources réelles mesurées seulement aprèscommit isolé.

Ce pilote n'a toujours que DEUX recherches initial100. Aucun recall nouveau
ni GT avant128sorties scellées/évaluateur séparé. Cible99,2%exactinitial et
fullOOF/Wilson/généralisation/label-sidecar/SLO intacts. Fold0/1fermés,
avalgelé, dernier89,0625%exact inchangé. Aucun READY/SOTA/garantie de99%.
