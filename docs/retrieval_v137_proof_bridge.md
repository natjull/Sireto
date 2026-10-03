# V137 — Preuves nationales vers positions candidat, sans pool

Après V136, raccorder TOUS105564761 facts nominaux officiels V132 et
TOUTES9730972 observations de succession V133 au namespace44323289 sites.
Les sources/observations/périodes/scopes d'origine restent immuables.
Chaque ancien ID garde sa position. Les identifiants sont des joints opaques,
jamais des features ni des mesures de similarité.

Deux tableaux par bucket nominatif : parent position et site position.
SIREN→preuve parent, SIRET→preuve de site uniquement. Aucun produit aliases×
membres, ni cap sur noms, sites, membres ou relations. Un parent officiel
sans site dans ce namespace conserve ses facts et une position UNKNOWN ;
pas d'identité site fictive. Chaque fact site doit être admis sinon STOP.
Reconstruction indépendante exacte de CHAQUE tuple via les positions.

Graphe : parent de CHAQUE candidat, site/parent de CHAQUE endpoint officiel,
positions predecessor/successor de CHAQUE observation. Les dates, booléens
TRUE/FALSE/UNKNOWN, IDs observation et provenance restent dans V133.
Pas d'état, adresse, siège, activité ou XY transféré entre sites.

Le noyau query-side agrège le maximum d'une variante ENTIÈRE, jamais un
sac de tokens issu de plusieurs noms. Exact-current exige à la fois le
match exact de cette variante et l'explicite flag current de sa source.
La propagation un-hop utilise uniquement un snapshot de preuves directes,
dans les deux sens : grade0,5 si continuité explicitement TRUE,0,25 sinon.
Self-links contextuels non expansés ; doublons max, pas de boost par nombre
d'observations. La preuve propagée ne devient jamais exact-current et reste
attachée à l'endpoint, pas tous membres du SIREN cible. Pas de propagation
A→B→C dans la même requête. Hypothèse issue du noyau V121/V122, pas résultat
appris ni garantie de pertinence de chaque lien.

80s internes/90s superviseur/3GioRSS/zéro swapout par phase, séquentiel,
groupes physiques8 buckets (pas une limite de recherche), contrôleur1200s,
768Mio maximum par sortie phase,1,5Gio sorties totales. Tous tableaux V132/
V133 et les quatre dictionnaires de joints V136 entiers rehashés à init ;
sources utilisées rehashées avant/après chaque phase,
et toutes sorties au seal. Toute dérive/incomplétude/budget : STOP, pas de
relance au même root ni budget relevé. Implémentation/config figées avant
mesure, modèles aval inchangés, aucun CRM/GT/fold0/fold1/fit/recall ouvert.

Ce bridge et ces noyaux ne sont PAS le moteur national initial100 complet :
lookup lexique V134 et adresses/attributs restent à intégrer. Pas de recall
ni SLO revendiqué ici. Pas de dense/RRF/LightGBM ajouté, pas de pool fixe.
La prochaine vraie mesure exige sortie initiale≤100 pour TOUTES requêtes
du pilote128 scellées AVANT lecture labels, puis OOF complet pour le gate.
