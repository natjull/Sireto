# V204 — mises à jour exactes max/OR, bench numérique non national

Protocole avant mesure. Goulot observé V202q4 : diffusion de39054006refs,
13,869s, seuil global20s dépassé malgré RSS conforme. V204 ne change aucune
source, preuve, score, label ou profondeur. Il prépare un vecteur privé
avec mêmes règles dtype/shape/position/epoch que DiskVector V170.

Pour maximum, toutes les refs sont d'abord validées, puis dans de petits
blocs physiques131072 : tri stable des positions, maximum de chaque groupe,
maximum avec valeur existante. Tous IDs restent traités, dernier bloc
conservé, pas top-k/re-score/cap business. Float64 exact ; valeurs nonfinies
ou hors[0,1] refusées. Cas -0.0 conserve l'ancienne méthode pour préserver
les bits de tie signés, pas seulement une égalité numérique approchée.

Pour flags, ORFALSE est l'identité : après validation de TOUTES positions
et ouverture/validation du fichier propre ENTIER, écrire True aux seules
positions où le flag est True (dédupliquées). Faux flag n'autorise jamais
à ignorer référence hors limites, mauvais dtype/shape, epoch ou vecteur clos.
Les preuves de couverture et tous compteurs propriétaires restent dues.

Tests contre V170 sur appels répétés, doublons/ordre différent/types/chunks,
dernier bloc131079, flagsTrue/False et signedzero, comparaisons de BYTES.
Aucune feature métier synthétique en production. Classe pas encore intégrée
au querykernel complet ni validée sur sources nationales à ce jalon.

Bench fixé numérique : vecteurs privés16000000 positions,1048576 mises à
jour, quatre appels, trois essais avec ordre original/grouped alterné.
Seed204, float64/bool, toutes valeurs du vecteur final comparées en blocs,
pas seulement IDs touchés. Ces chiffres sont des FIXTURES de performance,
pas corpus SIRENE/CRM/GT ni preuve de recall/national/SLO. Coût de création
et égalité séparé des temps max/flags. Cache non contrôlé explicitement.

Budget153 inchangé1Gio/80s/p1/free>=50%, quiet30, swapGLOBAL nonattribué.
Fresh répertoire private/tmp uniquement, suppression des seuls quatre
fichiers de scratch numériques détenus par essai APRES égalité complète ;
pas suppression de sources ou résultats anciens. Report/préflight conservés.
Pas cloud/dépense/application arrêtée. Code/contrat/tests/bench commité avant
mesure. Si accélération crédible, futur kernel205 complet bit-exact puis
probe q4 national fresh avant nouvelle génération128. Aucun label ouvert.
