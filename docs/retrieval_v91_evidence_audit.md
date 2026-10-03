# Audit séparé après génération V91

L'audit ouvre les labels humains uniquement APRÈS vérification des128 sorties
V91 intégrales, certifiées, hashées et déjà évaluées. Folds2/3/4 seulement,
aucun fold0/1 ; les labels ne modifient ni sources ni sorties ni features.
Toutes128 requêtes sont auditées, pas seulement les14 échecs. Aucun fit,
réglage du poids64 ou sélection de recette à partir de ces observations.

Accès point à la vérité : score natif de la requête GELÉE, preuve officielle
du chemin V84/V86, XY/BAN puis MAX exact V91. Une vérité manquée au-dessus
de la frontière100, ou un hit dont le score ponctuel diffère du score émis,
invalide l'audit par STOP : il ne faut pas supposer le certificat correct.

Les motifs publiés concernent uniquement LA NOUVELLE BRANCHE SCOPÉE : hors
endpoints V84, aucune preuve de mot littéral dans ces sources, BAN inconnue,
XY cible absente, ou preuve positive mais score insuffisant. Ils ne prouvent
jamais une absence dans toutes les sources officielles. Les champs individuels
restent publiés pour éviter de confondre motif de branche et cause globale.

Budget80s/6GioRSS/zéro nouveau swapout ; supervision externe90s. Hashes des
sources et implémentations serving vérifiés avant/après, comme génération.
Index Tantivy national : même contrôle d'inventaire/métadonnées, pas rehash
intégral des gros segments. Une garde bloque l'audit, jamais élargissement
automatique. Le pilote réutilisé n'est ni fullOOF ni preuve de généralisation.
Le rapport est label-side seulement et ne peut devenir une feature serving.
