# V116 — même score V115, bornes de sources sans famine

Contrat complémentaire avant mesure ; V115/V114 restent immuables.
Reprendre le contrat V115 en entier pour sources, score, exclusions, caps,
évaluation et limites. Seule priorité de lecture change : lire le flux avec
le moins de lectures déjà achevées, départage par plus grand frontier.
Chaque flux doit pouvoir réduire son majorant ; ce ne sont pas quotas de
candidats, tous sites restent admissibles et aucune profondeur fixe.

V115 s'est arrêté20s sur PREMIÈRE query sans label : natif7 lectures12160
observations, géographie0, frontière natif1,8532 et XY1,1053, cutoff2,6485.
Le majorant XY jamais interrogé rendait toute nouvelle page native coûteuse
sans lever cette inconnue. V116 corrige cette famine, PAS poids/features,
top100/score exact, taille des pages, epsilon, délais ou corpus.
L'oracle doit rester identique V115 dans toutes variantes caps/pages/ties.
Une fixture avec20000 lignes au plateau certifie par1 lecture de CHAQUE
flux,128 lignes physiques par lecture ; jamais un plafond128 de génération.

Toujours128 queries query-only234, coût16 mean<=4s, worker80s/6Gio/zéro swap,
superviseur90s, contrôleur900s,20s/query. STOP interdit qualité partielle et
relance budget accru. Nouveau répertoire, modèle V114 frozen. Folds0/1
interdits/fermés ; aval figé. Pas garantie de coût/recall ; dernier recall
V10189,0625%, pas fullOOF. Limites relationnelles et critères99,2% inchangés.
