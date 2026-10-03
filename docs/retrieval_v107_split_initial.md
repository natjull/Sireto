# V107 — quatre flux exhaustifs, un seul top100 initial

Avant mesure. V105 STOP16mean5,51s malgré suppression de tous ses points.
V106 deux ablations nationales sans labels : lexical seul1,45/2,61s, sigle
seul0,13/0,49s. Cache non contrôlé/limit100vs256, aucune vitesse garantie.
Cette observation motive la séparation physique, pas un nouveau fit.

Lire quatre flux triés sur le corpus NATIONAL inchangé : lexical11+termes
OOF, quatre sigles courants pondérés V67, nom officiel-scopé-chemin et XY.
Score initial=max(L+A,64*N*S), heap100 unique, sans top100 par canal préalable
fusionné/reclassé ni réservoir2000/500. Pages256→4096 sont des buffers de
lecture ; continuer jusqu'à preuve que toute donnée non vue reste sous
cutoff, y compris égalités canoniques. Fanouts/sites/relations complets.

L et A sont scores float32 des deux requêtes natives avec EXACTEMENT les
mêmes coefficients et grammaire V67. Leur somme float64 est explicitement
la métrique V107, même fonction mathématique mais pas identité bitwise avec
l'association de sommes V104/V105. Aucun ajustement aux14 misses. Oracle
composants V67 à tolérance et oracle top100 entier exact de cette nouvelle
arithmétique indispensables. Pas de revendication de supériorité SOTA.

Pour un site non encore vu dans un composant, borne de ce flux ; si flux
exhaustivement terminé et site absent, zéro PROUVÉ. Autrement inconnu=None,
jamais imputé. Native borné par UL+UA, chemin par64*frontièreN*frontièreS.
Chaque score admis exact ; dominance ou chemin>=borne évite un point.
Pour les autres sites, points réels sur les seuls composants manquants.
Incohérence scores/parent/source/borne implique STOP. Borne initialement
infinie conservée ; absence d'un positif n'est jamais réparée via les labels.

Mêmes query-only128/234/V67 projections/vocabulaires, sources43,9M sites,
57,8M tuples, graphes/direct/une-hop/BAN. Aucune activité/dates nouvelles
présentées comme servies. Aval/fold0/fold1 gelés ; all-human et sidecar.
Même contrôle16mean<=4s,20s/requête,80s/6Gio/zéro swap,90s supervision,
900s contrôleur.128 scellées avant évaluateur séparé/comparateurV101.
Pas de recall partiel, extension de STOP ou suppression du sigle appris.
Pilote non fullOOF, pas READY_FOR_FOLD1, Wilson/généralisation/SLO requis.
