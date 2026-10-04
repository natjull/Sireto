# V183 — phases physiques de la même recherche initiale à 100

Protocole avant mesure, 4 octobre 2026. Ce kernel prépare une exécution
supervisée par phase, sans élargir 1 Gio/80 s V153/V164 ni la sortie <=100.

1. Le snapshot noms V170 est rassemblé pour TOUS les 44 403 034 sites dans
   deux fichiers readonly float64/bool. Le gather garde bit à bit couverture
   et exact-direct, portée parent/site et successions one-hop. Pas d'héritage
   d'adresse, géographie, enseigne de site ou exact via succession.
2. L'adresse V178 reste entière sur les 34 346 446 alias. Les sources V181
   conservent tous les 98 804 872 faits originaux cohérents.
3. Les intervalles physiques V160 peuvent être partagés entre des workers.
   Chaque site et fait appartient à exactement un intervalle. Chaque worker
   calcule le MÊME score V182/V146 pour TOUTES ses feuilles, puis garde ses
   100 meilleurs avec les mêmes ties par identité officielle.
4. La réduction finale exige tous les intervalles contigus, tous les sites
   et faits, la même signature source/requête/fold/poids. Elle vérifie les
   identités et leur appartenance physique, puis réduit les scores inchangés
   à 100. Il n'existe AUCUN second score, reranker ou réservoir 2000/500.

Les sorties des partitions ne sont PAS des sorties produit. Aucun résultat
initial ne sort si un worker est incomplet, STOP, source mutable ou recette
différente. Le caller national doit vérifier chaque vrai parent_exit0,
V153 report, hashes/sources et protections AVANT réduction/publication.
Ces kernels et leurs fixtures ne certifient pas cette chaîne nationale.

Temps global = toutes les phases de requête et d'assemblage, jamais seulement
la dernière réduction. Ne pas prétendre tenir les SLO1s/2s grâce au découpage
physique. État disque non contrôlé n'est pas un benchmark froid. Les fichiers
intermédiaires scellés restent accessibles readonly et ne sont pas supprimés
par leur consommateur. Les propres leases sont fermés, pas les autres apps.

Objectif all-human initial10099,2% exact avant aval inchangé ; fold0/1/GT
fermés pendant génération, aucun positif injecté ou alias CRM appris, aval
gelé, exact/opérationnel séparés, fullOOF/Wilson/généralisation/SLO exigés.
Une requête nationale ou le pilote128 réutilisé ne vaut pas READY_FOR_FOLD1.
