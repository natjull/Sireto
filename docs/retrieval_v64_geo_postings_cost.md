# V64 — coût d'une géographie native jointe, sans qualité mesurée

Préenregistré avant exécution. V63 étaye un intérêt conjoint de la sémantique
et de la géographie sur contrôles source, pas le recall national. Éviter un
énorme build dense avant preuve. Tester un chemin physique nouveau : résoudre
les prédicats de distance dans le KDTree officiel complet, puis compiler leur
membership dans la requête Tantivy via les clés de jointure SIRET. Un seul
collecteur natif100, zéro pool fixe, reranking ou lookup GT. Les sites hors
rayon restent recherchables par les preuves lexicales globales.

IMPORTANT : bases radiales BINAIRES inclusives25/100/500/2000m, pas le noyau
continu V63. Pour ce test de coût uniquement, poids V63 natifs19 réutilisés,
cosines NON servies. Ce n'est donc PAS une exécution équivalente de V63 ni
une recette métier apprise ou validée. Toute qualité ultérieure nécessite
refit OOF/parité sur cette représentation exacte et son propre protocole.

Site lists : tous les points compatibles dans chaque rayon, y compris
co-localisés/fermés si présents dans le corpus spatial officiel ; pas de
nearest-k coupé. IDs seulement jointure prédicat spatial->document indexé,
pas signal d'identité appris ou alias CRM. Garde200000 points impose STOP
si dépassée, jamais une troncature. BAN faible : bases fiables désactivées,
pas de filtre dur de région ; BAN incompatible/absente : zéro bonus spatial,
corpus lexical intégral conservé. Couverture XY officielle limitée à26951280
sites sur43896818 sites indexés, autres sites NON exclus.

Population :8 premières requêtes du panel scellé V45, ordre indépendant des
labels, folds2/3/4 ; lecture de queries/geo seulement, aucune évaluation
V45/V57/V63 externe. Comparateur coût : même requête lexicale11 sans geo,
mêmes poids, ordre lexical/geo alterné. Sorties jetées ; publier comptes,
temps construction/query, tailles des prédicats et RSS/swap. Aucun candidat
persisté, aucun label, recall ou READY_FOR_FOLD1, cache non contrôlé donc
PAS SLO. Vérification hashes des sources spatiales/manifestes et inventaire
lexical size/mtime ; gros segments lexicaux toujours non entièrement rehashés.

Budget80s/6Gio/zéro swap, supervision90s, CPU2 ; STOP immuable, pas de relance
élargie. Aucune nouvelle collecte distante, dépense ou index national créé.
