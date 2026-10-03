# V119 — Même projection, barrières physiques et reconstruction alignée

Contrat avant mesure du 3 octobre 2026. V118 a terminé sous ses gardes mais
54,876353s regroupent factorisation **et** reconstruction complète. Le temps
de chacune de ces sous-phases n'a pas été instrumenté ; ne pas affirmer que
la reconstruction seule serait responsable de tout ce temps.

V119 matérialise une fois les observations et les dictionnaires dans des
tables DuckDB locales, au lieu de réévaluer des vues. Aucun nom, rôle,
scope, champ, période, ordre ou règle de génération n'est changé. Même
source physique immuable V118, hashes complets des petits fichiers vérifiés
avant/après. La limite des hashes des grosses sources V118 reste héritée.

Audit équivalent exact, sans EXCEPT ALL de larges tuples répété : prouver
nombre de lignes, unicité et égalité des indices avec les positions physiques
de sortie ; comparer ensuite **chaque ligne et chaque champ** à la même
position d'origine, avec IS DISTINCT FROM pour les nulls/tableaux. Vérifier
toutes les références de nom et tous les owners/scopes depuis les fichiers
publiés, ainsi que l'unicité des IDs et clés des dictionnaires. Toute ligne
ajoutée/perdue/déplacée ou toute différence de champ => STOP. Aucun contrôle
approximatif par checksum, échantillon, somme ou seules colonnes d'identité.

Comparaison avec V118 : mêmes clés et IDs des dictionnaires/faits contrôlés
exhaustivement ; les observations sont chacune déjà prouvées exactement
égales à la même source. Fixtures comparent en plus les trois tables entières.
Tests de corruption d'un champ, date, liste de champs, preuve, référence ou
owner ; permutation et observation déplacée refusées.

Mesurer les sous-phases physiques séparément. Budgets inchangés80s/3Gio/
zéro swapout, supervision90s, DuckDB512Mo mono-thread/spill512Mo ; nouvelle
sortie≤64Mio. Aucun relèvement de budget pour faire passer le test.

C'est une optimisation de préparation de données, **pas de retrieval**.
Toujours cost-only source sample, pas publication nationale, pas nouveau
recall, pas modèle, aucun CRM/GT/fold0/fold1 ouvert. Aucun gain de recall ni
SLO de requête à déduire d'un éventuel gain de vitesse de factorisation.
Les champs source non nominaux et liens restent dans les samples V118.
