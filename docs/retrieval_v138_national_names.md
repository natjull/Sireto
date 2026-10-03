# V138 — Lecteur lexical national et preuves nominatives, coût seulement

Toutes les57530682 variantes raw V130/V132/V134, tous les2015657202 postings
words/char3/char4/exact et tous les105564761 facts/9730972 liens restent
accessibles. Les64 buckets sont du stockage DuckDB hash1.4.3, pas des zones
ou plafonds de recherche. Pas de filtre géographique/date/état, pas de pool
de seeds ou de fanout cap, pas de dense/RRF/LightGBM.

Projection d'une requête CRM : max des couvertures words/char3/char4 calculées
sur CHAQUE variante entière, bit-identique au noyau V121. Les collisions
feature-text de différents raw alias IDs restent TOUTES accessibles en
exact. La preuve exact-current dépend toujours du flag officiel V132.
Scatter vers owners V137, sans produit alias×members. Optimisation physique
exacte : seuls les facts de variantes avec couverture>0 OU match exact sont
gathered ; sauter une preuve nulle ne change aucun maximum/flag. Un seul
alias populaire avec10001 refs est entièrement parcouru, par chunks RAM.
Même snapshot un-hop V137 ; pas de contagion multi-hop/exact ni autres sites.

Sonde : quatre PREMIÈRES requêtes du panel query-only128 préexistant hashé,
ordre inchangé, folds2/3/4 seulement, uniquement crm_name comme texte.
Tous IDs/traits CRM autres que le nom non utilisés ici ; aucun label/ancien
hit/modèle/source de vérité n'est ouvert. Dictionnaire/scatter comparés à
l'oracle scalaire dans des tests synthétiques, sans entraîner une feature.

Les sources utilisées sont entièrement rehashées avant/après la sonde, mais
pas à chaque requête. Aucune reconstruction d'index pendant les requêtes.
80s/90s par worker,20s maximum par requête,3GioRSS et zéro swapout. STOP si
budget/dérive/incomplétude ; root immuable et pas de budget relevé. Cache
disque non contrôlé, PAS benchmark froid ni SLO de production.

Ce noyau ne produit AUCUN top100 et ne prétend PAS mesurer recall ni temps
retrieval : seules preuves nominatives et relationnelles sont calculées.
Adresses, attributs/XY, upper bounds et collecteur100 restent à raccorder.
Ce test court identifie le coût national réel avant un replay long/all-human.
Tout gate99,2% exige sorties initiales≤100 scellées avant labels, puis OOF
complet/Wilson/SLO/sidecar. Fold0 jamais lu,fold1 fermé,aval gelé.
