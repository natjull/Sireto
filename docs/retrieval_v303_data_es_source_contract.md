# V303 — rendre le snapshot national Data ES utilisable, sans aucune cible

Avant exécution. Le CSV national existant sur CATNAT (331746052 octets,
SHA b7c49db1667af423e3ee381cdcd7371eb0a5cc86cd895f4b1f88abbbc2ca6a28)
est l'unique entrée. Pas de requête par client, de téléchargement supplémentaire,
de lecture d'ancien candidat, de label, de TRAIN ou de fold. L'ancien overlay
et ses métriques ne servent ni d'autorité nouvelle ni de validation de rappel.
Les cinq entrées129 n'incluent pas explicitement ce référentiel sectoriel.

Toute ligne CSV parsée dans son ordre original reste dans records, sans filtre
d'état, nom, zone, équipement, identifiant ou historique. Deux observations
nominales par ligne, INSTALLATION et EQUIPMENT, vides incluses. Ordinals exacts,
quinze champs bruts nécessaires (noms, identifiants, adresse, localisation,
propriétaire/type, gestionnaire/type, hors-service, mise à jour et coordonnées)
restent verbatim dans SQLite ; aucun autre champ source n'est détruit : tous
restent dans le CSV intégral hashé avant/après. Ce n'est pas une projection
du label ni une preuve de complétude nationale indépendante du fichier publié.

CSV UTF8-BOM, point-virgule, quotes/multilignes stricts, entêtes uniques et tous
les champs nécessaires, pas de cellule manquante/extra silencieuse ; taille
de champ1MiB est une garde d'intégrité, pas une troncature. Normalisation118
pour les deux noms seulement, originaux conservés, masques/IDs textuels exclus.
SIRET publié accepté seulement si14 chiffres ASCII après whitespace externe,
parent non000000000 ; scientifique, caractères, blancs et mauvais formats
restent UNKNOWN avec valeur brute. Aucun SIREN inféré à partir du nom du
propriétaire ou du type Commune. Pas de validation Luhn ou de membership natif
revendiquée ; cette dernière sera une jointure catégorielle séparée, pas feature.

Second parcours intégral du CSV et comparaison exacte de CHAQUE champ projeté,
ordinal, SIRET parsé et deux alias normalisés avec SQLite. Compteurs source,
UNKNOWNs, alias et SIRET distincts publiés, integrity_check, DB readonly après
seal, cache faible, aucun cap de noms/parents/sites par source. Limite sortie
256MiB, premier dépassement STOP ; aucun fragment/réessai/hausse de limite.

Producer214 après301/302 TERMINAUX seulement : quiet30, pression1/free50,
2GiB/80s/mono-thread et propre sortie exit0 réellement confirmée. Hashes avant/
après source331Mo et DB, epochs, versions Python/SQLite/Unicode et implémentation
liés au manifest. Un fichier
existant sur CATNAT ne devient pas une source scellée par simple présence.

Pas de génération, fit, recall, SLO, READY ou généralisation. Dates de mise à
jour ne deviennent pas intervalles de validité ni dates de fermeture, champs
hors-service ne remplacent pas l'état SIRENE. La déclaration SIRET Installation
ne certifie pas l'identité physique ou l'opérateur de chaque équipement. Toute
future recette devra conserver les scopes/source/provenance et les métriques
exact/op séparées, sans élargir les 100 ni attribuer automatiquement à une mairie.

Documentation primaire consultée, pas promesse de couverture de notre snapshot :
[Data ES](https://data.sports.gouv.fr/explore/dataset/equipements-sportifs/)
et [recensement Sports](https://www.sports.gouv.fr/recensement-des-equipements-sportifs-data-es-671).
La base Culture dispose de champs identifiants documentés mais son API nécessite
un accès ; aucune demande ni identifiant envoyé. Son export public est une piste
distincte non collectée/non intégrée à ce jalon.
