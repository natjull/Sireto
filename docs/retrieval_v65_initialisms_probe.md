# V65 — sigles issus des alias officiels, sans dictionnaire CRM

Préenregistrement avant mesure. Nouveau signal de NOM : reconnaître un sigle
via les initiales de mots d'UN alias officiel courant complet. Compiler des
RegexQuery sur les champs RAW exacts legal/trade/site déjà peuplés du corpus
national. Pas de corpus réduit, nouvelle table de synonymes ou gros build.
Tantivy RegexQuery matche des termes entiers du dictionnaire, pas une regex
appliquée à un pool hydraté : https://docs.rs/tantivy/latest/tantivy/query/struct.RegexQuery.html.
Tester la parité locale sur la version installée0,25,1.

Règle générique figée : au plus2 tokens alphabetiques2..6 lettres du nom CRM,
dans son ordre, suffixe numérique éventuel ignoré pour ce SEUL signal. SIREN/
SIRET retirés par signal() comme précédemment. Articles/prépositions usuels,
formes juridiques et mots institutionnels communs exclus ; aucune marque ou
sigle de cas CRM nommé dans le code. Alias normalisé uppercase, formes légales
et articles initiaux ignorables, particules françaises entre mots ignorables
SAUF EN, initiales en ordre, fin descriptive autorisée. Chaque preuve reste
sur un seul alias : ne pas fusionner les initiales d'alias différents.
L'inférence ne contient aucune paire alias CRM->nom officiel stockée.

Ajouter4 preuves : sigle global, sigle+commune/CP, sigle+numéro+commune/CP,
sigle+couverture adresse+commune/CP. Plus le composant lexical sémantique V59
de<=8 mots officiels pondérés, et11 preuves natives V57. Score16 non négatif,
compilable en un seul collecteur natif100 : pas de reranker/pool fixe.
Les corpus lexical et alias restent globaux, les gates de preuves ne sont
pas des filtres globaux de commune ou d'activité. Sites fermés conservés.

Probe TRAINING-only :512 scènes V52/V56, négatifs naturels, alias officiels
lus des documents indexés, vocabulaire<=2048 dérivé UNIQUEMENT des documents
officiels du train sûr de CHAQUE split (DF>=3, DF<=15%). Exclure positifs ET
négatifs des entités du contrôle de la construction du vocabulaire aussi.
Pas réutiliser le vocabulaire global V59, sélectionné sur tous les documents
teacher dont des positifs externes. Décodeur ridge10 fixé de la recette
interne V59, vocabulaire/projection refit sur le train sûr interne puis externe.
Jamais réutiliser un décodeur ayant vu les labels du contrôle interne.
Poids appris par perte de tête par requête V63, ridge0,005, bornes0..64,
180 itérations, convergence requise. Sélection interne seulement : BASELINE11
logistique refitée, HEAD_NATIVE, HEAD_WORD, HEAD_ACRONYM, HEAD_FULL. Critère
erreur de tête, nombre moyen de concurrents devant GT, puis moindre capacité.
Positifs ET négatifs des entités internes/externes interdits au train.
Mesures externes après choix ; test de non-influence du contrôle externe.

Les8 premières scènes d'apprentissage dans l'ordre source avec pattern
non vide donnent une parité offline/native sur les4 preuves nouvelles,
docIDs joins TRAINER seulement. Pas un positif injecté dans une évaluation.
Si manque d'alias/document/parité : STOP, pas de réduction du dénominateur.
Budget80s/4Gio/zéro swap, supervision90s, CPU2. STOP immuable, pas de relance
élargie. Contrôles réutilisés PAS holdout inédit, PAS recall100 national ni
SLO. Les modèles aval restent gelés, folds0/1 non lus. Aucun pilote de qualité
automatique sans protocole/scellement100 complet puis labels séparés.

Limites : preuve de sigle, pas équivalence juridique ni identité certaine ;
homonymie et expansions parasites possibles, d'où contrôle et poids appris.
Champs exacts uniquement COURANTS, pas faux champs exacts historiques. Cet
essai n'incorpore pas les coordonnées/BAN, relations ou temporalité causale ;
il complète l'investigation identité/site, ne prétend pas la remplacer.
L'audit V58 choisi par vérité ne fournit aucun alias ou overlay de production.
