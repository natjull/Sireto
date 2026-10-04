# V173 — distinguer première exécution et répétitions chaudes

Protocole avant mesure, 4 octobre 2026. Mêmes sources, lot49 entier,
requête et kernel172 exacts que V172. Aucun nouveau score ni candidat.

V172 donne direct0,460077s contre oracle0,071772s, mais direct est exécuté
en premier et paie potentiellement l'initialisation DuckDB/Arrow. Ne pas
attribuer ce différentiel à l'algorithme seul sans autre mesure. V173 appelle
le MÊME kernel trois fois, sans modifier/précharger les textes, et publie les
trois timings séparément. Même requête, tous les65536alias, ordre fixé avant
mesure. Pas de meilleur résultat choisi ni de moyenne cachant le premier.
Comparer chaque ID/float64/bool entre répétitions ET aux postings exhaustifs.

Reprendre intégralement les protections du contrat172 : autorité fullINIT,
lot certifié/trueexit0, hashes complets de chaque artifact avant/après,
oracle quatre lecteurs, folds234 query-only contrôlés avant sélection,
GT/anciens candidats/aval/fold0/1 fermés. Worker1Gio/80s, pression1/free50,
préflight30s, swap global télémétrie non attribuée. Root173 fresh immutable,
pas de retry172. Ce n'est ni une mesure nationale, ni recall, ni SLO.
État de cache disque non contrôlé ; première utilisation runtime et cache
ne sont pas causalement isolés. Aucun temps national extrapolé comme preuve.
