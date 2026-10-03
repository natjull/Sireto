# V126 — Blocs nationaux de tous les sites officiels

Contrat avant mesure,3 octobre2026. V125 prouve une représentation mmap
exacte du score sur sample ; son exporter Python n'est pas un compiler
national. V126 constitue la première brique de compilation nationale pour
la nouvelle famille de blocs, SANS encore noms, adresses, graphe ou modèles.

Lire toutes les43896818 lignes du stock officiel courant SIRENE et faire
une jointure typée exacte vers toutes les identités de l'autorité V96.
Aucun WHERE/localité/état/rayon/CRM/GT/graph-membership ne retire une ligne.
Une absence, identité malformée ou dupliquée impose STOP, pas une exclusion.
Source seule, chaque site exactement une fois ; les fermés restent inclus.

XY reprend précisément le périmètre métropolitain officiel L93 V121 :
codeCommune non vide et non97/98, abscisse0..1300000, ordonnée6000000..7300000,
coordonnées finies. Toute autre ligne reste dans le corpus, XY UNKNOWN.
Ce sont les coordonnées officielles courantes, PAS d'anciennes coordonnées
attribuées aux noms historiques, pas de géocodage massif ni API externe.

Morton à la maille du mètre uniquement pour ordonner physiquement les sites
connus ; la matrice de score conserve les doubles XY originaux, non quantisés.
Égalités Morton stables selon l'ordre préexistant V96. Sites UNKNOWN rangés
dans des blocs dédiés sans exclusion. Blocs<=128 et bboxmin/max exacts.
Le layout ne score ni ne sélectionne des candidats. Une nouvelle attache
de toutes les preuves aux nouveaux blocs restera nécessaire.

Audits : bijection complète source→V96, bitmap de couverture exactement
une fois et cardinalité de tous les sites du layout, bboxes calculées sur
tous les membres. Comparer aussi toutes32768 lignes du sample source-only
V118 aux fonctions scalaires source_coordinate V121 et à la matrice nationale.
Ce sample vérifie la révision physique, pas un recall ni une représentativité.

Budget80s interne/90s superviseur,3GioRSS,zéro nouveau swapout,DuckDB256Mo
mono-thread,sans spill,1Gio sorties. Un STOP reste terminal pour cette
révision ; pas de relance automatique augmentée. Champs source/parquet
footers/stats vérifiés avant/après. SHA complet rawSIRENE hérité DÉCLARÉ,
pas rehash des2,2Go ; tableaux d'identité V96 et sorties hashés intégralement.

Aucun CRM/label/fold0/fold1/fit/aval/cloud/GPU/dépense externe. Pas de
recall/SLO national ni index initial100 complet revendiqué. Tous gates
finaux restent à franchir ; dernier pilote exact@100 V10189,0625% inchangé.
