# V112 — nomenclature explicite retrouvée, aucune version devinée

Le stock SIRENE original utilisé par le dossier V3 contient
`nomenclatureActivitePrincipaleEtablissement`. La projection V3 ne l'avait
pas retenue, et avait coalescé l'activité NULL en vide. Cette préparation
source-only retrouve la version et le code originaux pour TOUS les43896818
sites, actifs ET fermés, par jointures typées V96. Aucun CRM ou label lu.

Comparer la projection documentée NULL->vide à V110, puis conserver NULL
original distinct dans le NOUVEL artefact ; anciens manifests immuables.
Nom de nomenclature officiel exact, pas d'inférence par forme du code ou
dates. Toute contradiction de source, jointure, unicité, couverture,
dtype, nomenclature nouvelle ou activité inconnue provoque STOP. Toutes les
valeurs écrites sont relues/décodées exactement ligne par ligne.

Libellés versionnés des tables dBASE INSEE :
[NAF2008](https://www.insee.fr/fr/information/2120875),
[NAF2003](https://www.insee.fr/fr/information/2408180),
[NAF1993](https://www.insee.fr/fr/information/2408178).
Les URLs et bytes complets sont hashés avant mesure. Lecture stricte
dBASE III, deux champs caractères, counts732/712/696, aucun deleted ignoré,
ni record tronqué ou code dupliqué. Les seules espaces retirées sont celles
de padding des champs dBASE, pas celles des codes SIRENE. Les encodages
OEM CP850/CP437 doivent être strictement IDENTIQUES sur chaque texte utilisé,
sinon STOP au lieu de supposer un accent. Parsing indépendant du classeur,
dont le runtime fourni était indisponible ; aucun package installé ni
classeur modifié. Référence structure des champs :
https://blogs.embarcadero.com/dbase-dbf-file-structure/.

Jointure `(nomenclature, code)` exacte, sans transformer un code ancien en
NAF actuelle ou fusionner deux définitions. Publier toute la couverture et
chaque couple non résolu, PAS un corpus limité aux sites décrits. La NAP1973
figure dans le stock mais ses descriptions ne sont pas encore chargées :
UNKNOWN de ce signal seulement, jamais exclusion de sites. Source primaire
à poursuivre : https://www.insee.fr/fr/information/3582824.

Garde80s/4Gio/zéro swap, superviseur90s, sortie256Mio, downloads officiels
<=1Mio/source avec timeout10s. Pas d'élargissement d'un ancien STOP.
Pas de nouvel entraînement, serving, candidat ou recall dans cette compilation.
L'essai suivant doit comparer un signal sémantique de libellés aux contrôles
TRAIN V111, sélection interne/protection familles, puis seulement tester la
combinaison noms/adresse/XY/BAN/relations dans une recherche initiale à100.
Fold0/1 fermés, aval gelé. Gates99,2% exact all-human/OOF/Wilson/généralisation/
attribution/SLO restent entiers et non démontrés.
