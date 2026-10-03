# V134 — Même accès complet, audit CSR vectorisé uint64

Contrat avant mesure,3 octobre2026. V131 tout staging COMPLETE scellé,
57530682 noms/2015657202 postings exacts words/char3/char4/exact-feature.
Rien à reprojeter,renormaliser,reclasser,caper ou sélectionner. Réutiliser
ce staging entier immuable et ses512 projections feature sans changement.

V131 bucket0 word/char3/char4 sont PASS,mais exact-feature STOP80,984678s
interne,zéro swapout,pic435404800. Trace pointe la recherche binaire scalaire
np.searchsorted(posts,uint64array/Pythonint) : NumPy2.4.1 peut convertir et
copier TOUT le tableau par lookup. Ce STOP/partiel reste immuable,jamais
scellé comme accès complet ni réécrit. Les512 parties complètes restent une
autorité valide indépendante du compactage arrêté.

V134 remplace seulement cette recherche par un vecteur expliciteuint64 de
positions de débuts de runs,un appel natif par batch131072. Tous postings,
vocabulaire,arrays,noms et ordre restent BIT-identiques à l'algorithme
original sur tests. Chaque token/alias/boundary se reconstruit toujours sur
SECONDE lecture COMPLETE de la source ordonnée : aucun audit échantillonné
ou désactivé. Une position dépassant2^53 est testée sans passagefloat64.
Les gros vocabulaires et postings traversant des batches sont testés.

Toutes4×64 phases depuis TOUS leurs fichiers V131. Hashes sources entiers
avant/après workers,versions DuckDB1.4.3/NumPy2.4.1 épinglées,phases immuables
et seal uniquement si tous256 terminaux/audités.80s interne/90s superviseur,
3GioRSS,zéro nouveau swapout,1thread DuckDB768Mio,spill2Gio,512Mio/phase et
32Gio incluant les sorties staging originales. AUCUN budget élargi ni quota
métier modifié,pas de reprise automatique d'un STOP.

Pas de CRM/GT/fold0/fold1,fit,embeddings,dense,RRF/reranker/decider modifiés.
Accès lexical national seul,PAS index riche/moteur100 ni recall/SLO. Dernier
pilote V10189,0625% exact inchangé ; pas de garantie99,2%/READY_FOR_FOLD1.
