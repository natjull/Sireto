# V95 — facteurs de nom nationaux complets et provenance exhaustive

Factorisation de TOUTES162537492 observations de la vue V94 scellée, aucune
sélection de graph/owner/CRM/label. Clé exacte (subject,owner,parent,scope,
normalized_value), NULL et vide distincts, aucun bag inter-alias. Noms vides
et SIREN enseigne restent facteurs context-only à score lexical zéro ; toutes
leurs observations et leur provenance restent accessibles, jamais supprimées.
Grain invalide impose STOP ; V94 national a compté zéro grain invalide.

Ordinal d'origine162m -> facteur, puis facteur -> TOUS ordinals. Chaque
source assignée exactement une fois avec garde de doublon, nombre total,
sentinelle absente et inverse exhaustif. Aucun plafond d'alias ou parent/site.
Dates/état/source/priorité/textes/IDs de preuve restent dans le fichier officiel
V94 référencé et hashé ; l'inverse donne ses vrais ordinals, pas un ordre de
projection matérialisée substitutif. Le signal d'ID sera masqué au posting,
pas utilisé comme similarité ; IDs ici uniquement grain/joins/provenance.

16 buckets owner%16 UNIQUEMENT stockage, pas pools de candidats. Phase
partition puis16 phases de tuple puis inverse, processus successifs80s/4Gio/
zéro swapout chacun, supervision90s, contrôleur900s, deux threads,
DuckDB1GB/spill2GB. Inverse par intervalles de facteur évite un tri global
des162m ordinals mais ne limite pas les observations. Sources originales
pleinement hashées avant/après au contrôleur et à partition ; chaque bucket
revérifie son partitionnement complet hashé, tous codes hashés par phase.
Plafond physique total10Gio préfixé pour162m observations, sans copie des
5,5Go source ; aucun budget augmenté après échec. STOP conserve l'historique.

Ce build ne produit AUCUN candidat et ne mesure PAS recall. Modèles aval
gelés, pas folds0/1, pas source CRM ni label, pas fit/dépense externe. Suite :
postings/owners et membership nationaux complets, oracle serving puis
préflight aveugle et mesure initial100. Cible99,2% exacte/OOF/généralisation/
SLO inchangée. Les463 vides et275422 observations enseigne contextuelles
V94 ne doivent pas être retirés de la source sous prétexte de performance.
