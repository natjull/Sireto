# V106 — isoler le coût natif sans ouvrir les labels

Avant mesure. V105 a certifié16 sorties, puis STOP_COST_PREFLIGHT : moyenne
5,510441s>4s.7305 cibles dominées prouvées, aucun point natif requis. Les
index/membres/fanouts ne sont plus le coût dominant de ces16 cas. Aucun
recall128 mesuré, aucune reprise élargie. La recette V105 reste immuable.

Deux premières requêtes de l'ordre128 CRM déjà fixé, folds234 uniquement.
Mesurer trois ablations de coût sur l'index natif NATIONAL complet :
LEXICAL11_PLUS_WORDS (poids sigles annulés pour diagnostic seulement),
CURRENT_INITIALISM_ONLY (prédicat global sigle courant),
CURRENT_INITIALISM_AT_PLACE (même prédicat intersecté commune/CP).
Ordre alterné par requête, une passe, cache non contrôlé. Les coefficients
modifiés ne sont jamais une nouvelle recette qualité ni un choix de fit.
Chaque appel retourne au plus100 hits internes ; ni leurs clés ni leurs
contenus ne sont écrits ou réutilisés pour une recherche métier.

Modèles et expansions query-only V105 vérifiés. Aucun label/sidecar, fit,
lookup de vérité, transfert entre alias ou corpus réduit. Métadonnées/index
cardinalité/inventaire vérifiés avant/après, limite historique non-rehash des
gros segments publiée.80s/4Gio/zéro swap, threads2, supervision90s.
Rapporter temps/construction/retour non vide/RSS/swap. Aucun recall ou SLO
national annoncé à partir de deux ablations ; fold0/1 et aval gelés.

Si le prédicat sigle coûte cher, prochaine action : compiler au build
ses postings officiels avec oracle langage actuel/corpus complet. Aucun
gros build national déduit d'une petite vitesse V68 ni suppression du
signal appris pour forcer artificiellement le passage du contrôle coût.
