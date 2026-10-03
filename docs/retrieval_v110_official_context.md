# V110 — contexte officiel complet, avant apprentissage

Préparation source-only du nouveau volet initial100, pas un résultat de
retrieval, une nouvelle baseline ou un reranker. Le dernier recall mesuré
reste celui de V101 (114/128,89,0625%, panel de développement réutilisé).

Les43896818 établissements du dossier officiel sont alignés exactement à
V96 par jointures SIRET/SIREN typées, jamais par ordinal supposé identique.
Pas de restriction aux actifs, à un graphe, à un territoire ou à une vérité.
Les identifiants servent seulement à la jointure et restent hors des features.

Champs conservés : activité brute, état brut, siège (inconnu/faux/vrai),
date de création et début de période courante en jours date32, NULL séparé.
NULL et chaîne vide sont DISTINCTS dans les dictionnaires uint16 d'activité
et état. Aucun trim, changement de casse, assimilation de nomenclatures,
fusion de codes ou traduction métier. Les dates ne prouvent pas la validité
historique d'un nom, d'une adresse ou d'une filiation. Aucun de ces nouveaux
champs n'est encore utilisé en serving ni appris ; ne pas annoncer de gain.

Deux lectures Arrow en lots : dictionnaires bruts exhaustifs, puis contexte
joint au registre national sorted uint64. Vérification de tous les formats
ASCII14/9, préfixe parent, existence, unicité dans/entre lots, état V96,
cardinalité et couverture complète. Une contradiction implique STOP, jamais
UNKNOWN imputé ou ligne supprimée. Les tableaux sont mmap compacts, sans
nouveau pool candidat ni copie riche du dossier. Autorités de source et du
registre entièrement rehashées avant/après, code/config/contrat hashés.
Chaque valeur écrite est relue et décodée pendant le lot, puis comparée à
la valeur source exacte, NULL compris, pour toutes les lignes (pas un sample).

Guard80s/4GioRSS/zéro swap, superviseur90s, deux threads au plus, sortie1Gio.
STOP conservé sans relance élargie. Tous les folds/CRM/labels sont absents
de ce compilateur. Fold0 contaminé et fold1 fermé, modèles aval gelés.

Étape suivante : tester une preuve de contexte apprise uniquement en TRAIN
OOF folds2/3/4, protections familles et hard negatives naturels, jamais
dictionnaire alias CRM ni règles tirées des misses. Son incorporation devra
se faire dans une recherche nationale à heap100 unique et bornes exactes,
pas par reclassement d'un pool fixe. Score/architecture à figer séparément
avant tout nouveau run. L'objectif reste99,2% exact all-human, fullOOF,
Wilson/généralisation/attribution/SLO ; ce catalogue ne franchit aucun gate.
