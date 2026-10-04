# V205 — max original, ORFALSE exact et profiling de diffusion

Protocole avant mesure. V204bench a rejeté le tri/groupement des maxima
(plus lent). V205 conserve STRICTEMENT DiskVector.maximum V170, y compris
signedzero. Seul flags204 est utilisé : valider toutes références, types,
shapes/epochs et fichier propre entier puis ne pas écrire ORFALSE (identité),
True dédupliqué en blocs131072. Pas retrait de refs/alias, nouvelle feature,
score, troncature, cap de candidats ou positif injecté.

Le query complet est celui201 : mêmes RAMcounter198 <=64Mio et fallback
intégral170, mêmes propriétaires/patches149, parents/site séparés, relations
figées1hop V170 en blocs131072/toutes entrées/deux sens/grades .5/.25. Max
et flags des quatre vecteurs sont chronométrés séparément ; counts d'updates
et true-flags dans checkpoints de buckets et stats. Ces diagnostics ne sont
jamais des features ni une sélection de sources. Timer globalquery20s reste
unique, profiling compris, aucun reset entre étapes.

Tests contre scalar150/lexique138, BYTES de tous vecteurs/gather, counts
et relations sur plusieurs textes/chunks et champs vides. Spy interdit
GroupedProofVector.maximum, garantissant qu'il n'est PAS activé. Fausse ref
hors bornes avec flagsFalse, mauvais source/hash/epoch/process refusés.
Toute petite autorité/process/pression de fixture est explicitement simulée,
pas validation nationale. Aucun label réel lu.

Probe coût fixé q4 parce que STOP202 vient de39054006refs/temps, pas recall.
ALL128/vecteurs/heads/folds2/3/4 validés avant sélection. NationalENTIER
57530682aliases/44403034sites, FULLaudit séparé1701sources+preuve avec
epochs pré/post et parents réels0. Refs canoniques path+SHA, tailles
originales requises lorsqu'elles existent, pas erreur de format V198.

Fresh CATNAT205, aucun artefact/capsule/scorer de scopes STOP ou anciens
candidats ouvert/réutilisé. Budget1Gio/80s/query20/flatten20/quiet30/p1/
free>=50 inchangé, swapGLOBAL nonattribué. Pas dépenses/cloud/purge/autre
app arrêtée. Codes/config/contrat/tests commités avant mesure puis gelés.
Si STOP, conserver intact et attribuer avec profile ; pas hausse de budget
ou réduction de population. Pas autre128long avant cette preuve de coût.

Ce probe ne mesure ni recall/SLO/OOFcomplet/READY. Toute génération future
reste ALL128fresh, <=100 initial avant modèles aval gelés, labels seulement
après ROOT128SEALED. Dernier recall exact non remplacé par un gain de coût.
