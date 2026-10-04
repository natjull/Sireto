# V182 — égalité des numéros par dictionnaire, mêmes scores initiaux

Protocole avant mesure nationale, 4 octobre 2026. Cette optimisation remplace
la construction/validation répétée de 98 millions de chaînes de numéro par
une validation du dictionnaire original de chaque intervalle, puis un gather
booléen. Le dictionnaire V147 reste verbatim : ne jamais le renormaliser.
La requête seule reçoit exactement la normalisation V118 existante.

Les slices CSR, le texte complet d'adresse, CP/INSEE, état courant et portée
site sont inchangés. Aucun tri/filtre/cap de preuves. NUM demeure l'égalité
exacte non vide du même numéro dans la même preuve, pas son code ordinal.
Les 30 features et poids gelés, float64, ordre des opérations/sommes/maxima,
fallback courant, géographie souple et UNKNOWN demeurent V146. La métrique
exacte reste distincte de l'équivalence opérationnelle.

Le binding exige V181, conserve sa traversée complète et ses contrôles de
source et hash final du snapshot. Le collecteur V146 consomme tous les sites
avant une sortie de 100 au maximum : aucun pool fixe 2000/500 ni reranker.
Le modèle aval reste gelé. Une exception ne publie aucun résultat partiel.

Parité bit-à-bit à établir sur les numéros vides, Unicode de compatibilité,
anciens numéros à neuf chiffres, CP/INSEE inconnus/Corsica/DOM, preuves
contradictoires, géocodage fiable ou non, sites sans coordonnées et tous les
poids/features. Cette preuve sur fixtures n'est pas une preuve nationale.

Toute première exécution nationale exige le superviseur V153/V164 : 1 Gio,
80 s par worker, pression 1 et free >=50 %, Mac/CATNAT sans dépense. Objectif
99,2 % exact humain all-human OOF2/3/4, Wilson/fullOOF/généralisation/SLO.
Fold0/1 et GT fermés pendant génération, aucune injection ou alias CRM appris
ici. Pas de recall ou READY annoncé depuis ces seuls tests.
