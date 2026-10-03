# V115 — premier serving national du score activité/site V114

Avant mesure. Modèle V114 figé après sélection INNER et contrôles TRAIN :
native11 + géographie8 + activité x preuve de site11. Ce test change le score
initial de V101, il n'en est PAS une accélération bitwise. Les preuves RNE/
filiation/BODACC et sigles V67 ne sont PAS toutes intégrées à cette formule ;
aucune promesse de conservation des gains V101 ou exploitation totale des
sources. Comparateur V101 séparé, lu après seal seulement.

Deux flux EXHAUSTIFS : natif Tantivy national43896818 SIRET, géographie
officielle V43 parmi les26951280 sites aux coordonnées compatibles connues.
Pas exclusion des sites sans XY, activité, noms, actifs ou fermés. Activité
catalogue V112 : jointure exacte version/code,2140 descriptions V113,
unknown0 preuve uniquement. Tous les sites gardent le score natif. Aucun
code/identifiant comme feature, aliasCRM ou bibliothèque synthétique.

La requête CRM NOM encodeur figé V52/V78 sert au décodeur V114 propre au fold.
Tous2140 libellés scorés ; aucune shortlist d'activités ou de parents.
Maximum global d'activité fournit les coefficients natifs/XY majorants.
Chaque site lu est complété avec les11 preuves indexées réelles,8 distances
et SON activité source. Le heap de classement contient<=100 SIRET seulement.
Les pages physiques128 croissant à4096 ne plafonnent PAS une génération ;
lecture continue jusqu'à borne des sites jamais visités < score du100e.
En cas d'égalité, continuer pour garantir l'ordre canonique global des IDs.
Pas pool fixe puis reranker. Le callback complète le MÊME score initial.

Les scores triés sont des majorants, pas valeurs finales ou minorants.
Somme des frontiers natif+XY borne tout site non visité. Natifs f32 :
enveloppe conservative gamma_n des opérations positives,512+64 par
token/champ, facteur2 pour float du score et des preuves point. Native
1.000008 par feature,2.000008 pour somme nom+adresse ; coefficient f32,
arrondi de sortie et cast couverts, jamais clip de score exact. Si n*u>=.1
STOP. Géométrie : marge64eps64(sum coefficients+1). Une violation de borne,
identité/parent/source/complétude/coût STOP, aucun retrait de requête.

Oracle synthétique DE TEST UNIQUEMENT, jamais business : classement exhaustif
450 documents, différents caps/pages, gagnant sous100 dans CHAQUE flux,
égalité globale325, missing/unknown, parent drift et garde budget. Cela ne
prouve pas encore les bornes du runtime Rust sur corpus réel ; chaque visite
réelle vérifie le score contre les majorants source et global avant seal.

Population128 query-only V78 inchangée, folds234 uniquement, autre TRAIN512
disjoint. Panel développement réutilisé, pas holdout. Parité encoder poids/
préfixe NOM/longueur128 avec TRAIN, modèles cross-fit sans famille tenue.
BAN préexistante query-only, CRS/fiabilité inchangés ; aucune réparation GT.

Worker80s/6Gio/zéro swap ; superviseur90s et contrôleur900s.16 queries
coût avant128, moyenne<=4s obligatoire ; budget expérimental pas SLO.
STOP n'autorise pas évaluation de sorties partielles ni relance plus large.
Generation128 intégrale scellée avant évaluateur distinct all-human exact/
opérationnel/Wilson/segments/ledger/label-side. Dernier vrai recall V101
89,0625% tant que nouveau replay complet non scellé/mesuré. Même100% pilote
ne valide pas99,2% fullOOF, Wilson bas99,0%, chaque fold98,8%, généralisation,
attribution et p95<=1s/p99<=2s. Fold0 jamais lu, fold1 fermé, aval gelé.
