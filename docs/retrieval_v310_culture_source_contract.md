# V310 — source Culture complète, avant mesure

Volet source indépendant, pas une relance Data ES V303 STOP. La correction
V308 reste non exécutée en attente d'autorisation explicite. Aucune sortie
rejetée, requête CRM, GT, candidat historique, fold0/1 ou modèle aval utilisé.

Source publique du ministère de la Culture :
[Base des lieux culturels ouverts](https://www.data.gouv.fr/datasets/base-des-lieux-culturels-ouverts).
Licence déclarée ODbL, lacunes/erreurs possibles selon l'éditeur. Le
[dictionnaire officiel](https://basedeslieux.culture.gouv.fr/dictionnaire-des-donnees)
sépare nom, nom usuel, entité juridique, SIRET et adresse publique/administrative.
L'API authentifiée n'est ni contournée ni sollicitée. Aucun contact envoyé.

GET global unique du CSV public lié par DataGouv, URL objet explicitement
figée dans le code, sans identifiants privés, cookies, query ou CRM transmis.
HTTP200, URL sans redirection, text/csv, identity, Content-Length42496931
confirmé par lecture d'en-tête indépendante avant ce contrat. Réponse entière
conservée et hashée, garde64MiB sans troncature, timeout réseau20s. Changement
de source/schema/taille ou téléchargement partiel impose STOP sans retry.

ALL lignes CSV UTF8-BOM/semicolon, quoting/multilignes stricts, ALL50 champs
bruts en SQLite, ordre physique/ordinal conservé, champs vides et valeurs
invalides incluses. Deux observations normalisées118 par ligne : nom du lieu
et nom usuel, même si vides. Aucune déduction de SIREN/SIRET depuis le nom de
l'entité juridique ; SIRET publiés14 chiffres ASCII et parent nonzero seulement,
sinon UNKNOWN en conservant le brut. Cette vérification de forme n'affirme ni
existence SIRENE, ni identité physique, ni continuité légale, ni équivalence.

Second parcours CSV complet vérifie CHAQUE champ/ordinal/SIRET/alias SQLite,
sans sélectionner sur commune, activité, nom, ID ou qualité. Runtime lié
Python/SQLite/Unicode. Integrity_check, lecture readonly, SHA256 avant/après,
epochs originaux et garde256MiB de sortie ; les autres colonnes ne servent
pas pour l'instant à la recherche. Données de contact conservées dans ce
snapshot officiel local seulement, aucun message/usage d'envoi autorisé.

Producer propre214 : quiet30s, pression normale1/free>=50%, RSS<=2GiB,
worker<=80s, un thread, vrai exit0/PASS contrôlé par superviseur avant seal.
Mac/CATNAT uniquement, pas de dépense ou nouveau modèle. Toute faute STOP
interdit l'admission, la réutilisation partielle et la relance automatique.
Le téléchargement complet est une collecte publique, pas du compute externe.

Source-only : aucun candidat émis, fit, label, recall, READY, fullOOF ou SLO.
Une recette native100 distincte devra être figée et mesurée après source seal.
