# V133 — Toutes les successions officielles brutes

Contrat avant mesure,3 octobre2026. Source complète9730972 observations du
stock SIRENE liens de succession,hash entier vérifié avant/après. V84 limité
aux7512820 observations précédemment liées reste immuable,historique.

Toutes lignes gardées,dans l'ordre source original. Aucune déduplication de
dates/endpoints,pas de filtre de continuité économique,état,commune,query ou
catalogue. Identifiants ASCII14 typésuint64,leading zeros reconstructibles ;
datesdate32/timestampsus avec validité explicite,booléens UNKNOWN/FALSE/TRUE
en codes0/1/2. Owners hors catalogue et self-links restent présents. Un
self-link est contexte de source,jamais expansion. Continuité économique
n'est pas identité juridique ni preuve du même site physique historique.

CSR bidirectionnel endpoint→incidences : chaque observation r a deux
orientations physiques r et n+r,y compris les self-links. Offsetsuint64,
incidencesuint32,keysuint64. Ordre stable des endpoints complets,CSR parent
vers ranges contigus d'endpoints (préfixe9digits). Pas de produit nom×sites,
cap de voisins/membres ou filtre de statut. Identifiants joints seulement,
pas des features de similarité. L'intégration query/scoring est ultérieure :
pas de propagation transitive ni d'assimilation d'une relation à un alias.

Audit indépendant de TOUTES lignes : chaque string,leading zero,date/null,
booléen est reconstruit exactement depuis les arrays. Certificat complet
permutation des2n orientations,puis reconstruction de CHAQUE endpoint/
parent par CSR. Pas d'oracle hash/échantillonné pour la sémantique. Toutes
références source brutes restent stables via l'ordinal de l'observation.

80s interne/90s superviseur,3GioRSS,zéro nouveau swapout,sorties<=1Gio.
Artefacts immuables,implémentation hashée avant/après,tous handles terminaux
avant usage ; pas de reprise/élargissement automatique d'un STOP. Aucune
lecture CRM/GT/fold0/fold1,fit,dense/reranker/decider/risk/accepteur modifiés.

Le graphe compilé ne vaut pas retrieval100/recall/SLO/READY_FOR_FOLD1.
Dernier recall V10189,0625% exact inchangé. Ce stock ne couvre pas par lui-
même toutes fusions/TUP BODACC ni des preuves historiques XY/adresse absentes.
