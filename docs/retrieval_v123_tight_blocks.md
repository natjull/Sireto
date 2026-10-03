# V123 — Bornes de blocs par variantes cohérentes

Contrat de coût avant mesure, 3 octobre 2026. V122 a produit exactement le
top100 exhaustif des mêmes32768 sites pour les quatre premières requêtes
query-only préfixées, mais a visité9465 à31736 sites. Il ne mesure aucun recall.

V123 conserve intégralement les sites, groupes, preuves, poids cross-fit
V114 transférés, formule de score V121, grades, géométrie et collecteur100.
Seul le calcul de la borne supérieure des preuves de nom et d'adresse change.
L'union des tokens de variantes différentes à l'intérieur d'un bloc faisait
paraître possibles des similarités qui n'existent pour aucune variante.

Chaque texte normalisé unique et toutes ses attaches aux blocs sont indexés.
Les postings words/char3/char4 calculent exactement la couverture de chaque
variante entière ; chaque bloc prend le maximum des couvertures pondérées
par les grades d'attache. Il ne choisit aucun top-alias ni ne plafonne les
variantes, sites, membres ou relations. Le score feuille est inchangé, avec
les mêmes protections flottantes et arrêt certifié. Pas de pool fixe suivi
de reclassement. Aucun nouvel entraînement ni dictionnaire d'alias CRM.

Source adapter V122 inchangé : les self-links officiels restent contexte
auditable, les autres successions sont des hypothèses, pas des identités.
Fermés, XY inconnus, activité inconnue et données masquées restent conservés.
Le masque ne devient jamais une preuve lexicale. Aucune adresse ancienne
inventée ni preuve de site copiée à tous les sites frères.

Le probe reprend exactement les sources et budgets V122 : échantillon
source-only, familles SIRENE_CURRENT/BODACC/RNE4096 lignes chacune pour nom
et adresse, quatre premières requêtes du panel128 préexistant folds2/3/4.
Les modèles own-fold ont été fixés avant cette mesure ; leurs poids sous
cette nouvelle formule ne constituent pas une équivalence au score V114.
Les grades d'activité sont explicitement désactivés pour ce probe.
L'oracle trie tous les32768 sites du même sample avec la même formule ;
chaque résultat doit être exactement égal, scores et ordre compris.

Budget80s interne/90s superviseur,3GioRSS, zéro nouveau swapout,20s par
recherche,32Mio de sorties. Un dépassement impose STOP, pas de relance
automatique élargie. Les gros SHA sources sont hérités déclarés ; stats et
footers contrôlés avant/après, pas rehash complet. Petites autorités vérifiées.

Ce prototype Python possède encore les structures V121 héritées ; il ne
valide ni une représentation nationale compacte, ni le SLO, ni le recall.
Aucun label, fold0 ou fold1 n'est ouvert. Aucun modèle aval changé, aucune
dépense externe, aucune compilation nationale autorisée par ce test seul.
Dernier recall mesuré reste V101114/128=89,0625% exact, pilote réutilisé.
99,2%, Wilson, OOF complet, généralisation et attribution restent à franchir.
