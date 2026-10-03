# V124 — Calcul exact des feuilles par bloc

Contrat avant mesure, 3 octobre 2026. V123 a gardé exactement les sorties
V122 et l'oracle complet sur32768 sites. Recherche0,313 à0,993s sur quatre
requêtes query-only, visites8094 à21036 sites. Ce coût et cette
taille partielle ne permettent pas d'extrapoler une recherche nationale.

V124 reprend strictement le score V121, les bornes V123, le source adapter
V122, groupes, corpus, poids transférés V114, absence de fit et collecteur100.
Les scores des textes dédupliqués sont calculés une seule fois par requête
et partagés entre les bornes et les feuilles. Chaque bloc dispose des
attaches site local→texte ; maximum lexical et maximum current-exact
gardent des grades distincts. Les objets de source/provenance restent
intacts, même lorsque des attaches de score identiques sont factorisées.

Les30 features sont calculées par tableaux NumPy pour un bloc seulement ;
le cache est propre à la requête et contient au plus un bloc physique.
La même somme float64, heap100, gestion globale des égalités et arrêt
certifié V121 sont conservés. Aucun produit de candidats à trier ensuite,
aucun pruning approximatif, aucun filtre état/localité/XY inconnu.

L'oracle utilise explicitement les fonctions SCALAIRES V121, pas la méthode
vectorisée testée. Les tests comparent chaque feature et borne bit à bit,
les sorties, les visites, les états de requêtes intercalées, les grades
current/historique séparés,250 variantes et325 égalités globales.

Même probe source-only et mêmes limites V123 : quatre premières requêtes
du panel128 préfixé, folds2/3/4,32768 sites, familles officielles4096 lignes
nom/adresse chacune.80s/90s,3GioRSS,20s par recherche,zéro nouveau swapout,
32Mio sorties. Activité désactivée. Aucun label/GT/fold0/fold1, aucun alias
CRM appris, aucun modèle aval changé, aucun GPU/cloud/dépense externe.

Ce prototype hérite encore des objets Python et des postings redondants
V121/V123 ; ce n'est PAS un index national compact. SHA sources complets
hérités déclarés, stats/footer contrôlés avant/après ; petites autorités
hashées. Ce probe d'équivalence et de coût ne mesure pas un recall ni un
SLO national. Dernier recall exact publié V101114/128=89,0625%, pilote
réutilisé,99,2% et tous gates finaux encore NON atteints.
