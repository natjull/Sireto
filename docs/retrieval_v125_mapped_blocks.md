# V125 — Moteur persistant de recherche initial100

Contrat avant mesure du 3 octobre2026. V124 est exact sur son sample, mais
ses objets Python, dictionnaires complets et bornes en boucle Python ne
constituent pas une représentation nationale maîtrisée.

V125 exporte le MÊME score V121 et les MÊMES bornes V123 dans des tableaux
sans objets NumPy/pickle. Textes ASCII normalisés stockés en bytes+offsets,
lookup binaire avec index d'ordre ; postings words/char3/char4 en CSR.
Toutes attaches alias→bloc en CSR, exact courant et preuve site séparés.
Les grades current-exact et lexical restent distincts. Originales sources,
variantes, masques, états et toutes observations/provenances restent dans
les autorités source hashées ; les tableaux sont une projection de score,
pas une réécriture des observations officielles.

Runtime NumPy mmap read-only, aucun dictionnaire Python par alias ni objet
par site, pas de constructeur V124 dans le processus de recherche. Scores
de toutes variantes lexicalement présentes, scatter des attaches positives
par chunks mémoire, aucun top-alias/fanout cap. Bornes19/30 calculées en
tableaux avec les mêmes marges flottantes, pas de pruning approximatif.
Heap<=100, arrêt strict certifié et égalités globales conservés.

Même fixture de sources officielles32768 sites et familles4096 lignes
nom/adresse, quatre premières requêtes query-only128 folds2/3/4. Compilation
du sample via V124, export persistant, puis oracle SCALAIRE sur tous32768
sites scellé séparément. Nouveau processus Python recherche dans les
tableaux ; aucun oracle/ancien candidat/label lu pendant la génération.
Comparaison exacte scores/ordre seulement après scellement des quatre
sorties complètes dans le superviseur. Cache disque non contrôlé : nouveau
processus ne veut PAS dire disque froid.

Limites inchangées par worker80s/90s superviseur,3GioRSS,zéro nouveau swapout,
20s/recherche,32Mio total sorties. Pas de relance automatique élargie.
Activité désactivée, poids own-fold V114 transférés sans fit, aval gelé.
Toute dérive d'oracle ou dépassement STOP. Aucun fold0/fold1/GT. Aucun
dictionnaire d'alias CRM, coût externe/GPU/cloud, ni métrique post-ranker.

L'exporter accepte encore un fixture Python V124 : il n'est PAS un compiler
national streaming. Le corpus partiel et ses ratios ne permettent pas de
prédire volume, recall ou latence nationale. Couverture officielle complète,
compiler streaming, tests de montée d'échelle et véritable mesure nationale
restent nécessaires. Gros SHA sources hérités déclarés/stats+footer contrôlés,
pas rehash global ; petites autorités et chaque tableau complet vérifiés.

Dernier recall exact reste V101114/128=89,0625%, pilote réutilisé. Aucun
recall/SLO/gate final validé par ce test de persistance/équivalence/coût.
