# V74 — Preuve couplée RNE dans les contrôles source, avant mesure

Objectif limité : vérifier si une interaction nom/adresse appartenant à UNE
observation officielle de site ajoute de la discrimination aux 23 preuves
actuelles, incluant XY/BAN V56 et sigles officiels V67. Ce test n'est pas le
retrieval national, ne mesure aucun recall@100 et ne peut valider le gate.

Population inchangée : les 512 scènes TRAIN V52, tous positifs et tous négatifs
naturels. Extraction RNE uniquement pour les SIRET ET parents de ces exemples
d'apprentissage. C'est une extraction label-selected TRAIN, jamais une source
admissible pour construire un index national ou un corpus d'évaluation. Toutes
les scènes restent présentes, RNE absent => bonus zéro, pas exclusion.
Le teacher historique conserve ses biais de sélection et exclusions des
équivalents opérationnels ; il ne définit PAS le dénominateur all-human final.

Lire toutes les observations RNE de ces parents et des SIRET d'apprentissage
explicitement demandés dans le Parquet canonique national. Conserver tous
parents SIREN correspondants pour retrouver le propriétaire explicitement lié
par source_record_id=formalité + ':site:' + SIRET. Seuls noms LEGAL de CETTE
formalité transférés, pas noms commerciaux, adresses parent, siblings, relation
asset-transfer ou dates devinées. Le lien juridique ne prouve pas une période
commune, dates UNKNOWN restent UNKNOWN. Contenu official_evidence allowlisté.

Bonus fixe avant mesure : max sur observation, alias individuel, alternative
d'adresse de `max(coverage_nom, sigle_officiel) * coverage_adresse *
(1 + lieu + numéro_au_lieu) / 3`. Tokens et IDs masqués comme V72 ; sigle
V67 sur UN alias complet, lieu = INSEE ou CP égaux sans filtre, numéro_au_lieu
= numéro égal ET lieu égal. Adresse CRM utilise numéro/suffixe/voie, pas son
label. Le max se prend APRES l'interaction, pas produit de maxima provenant
de différentes observations. Bonus [0,1], jamais un gate dur. Les anciennes
preuves natives restent un comparateur séparé agrégé, pas annoncées cohérentes.

Deux familles UNIQUEMENT : NATIVE23 et NATIVE_PLUS_COHERENT24. Contrastes
InfoNCE égal par requête avec tous négatifs naturels safes, ridge .005 autour
du prior V69 (nouveau coefficient initial zéro), poids positifs [0,64], L-BFGS-B
180 itérations/ftol1e-8/gtol1e-5. Non convergé exclu, refit/comparateur non
convergé STOP. Pas de relâchement automatique ou grille ajustée après mesure.
Nested V53 par entreprises, registre COMPLET folds2/3/4. Parents outer et inner
retirés de l'apprentissage positif ET négatif. Choix inner seulement, par
head_error puis mean_outrank_count puis nombre de paramètres ; refit train+
inner puis contrôles outer. Publier toutes512 lignes, changements appariés,
couverture RNE positives/négatives, familles et poids. Aucune métrique de
rang dans un jeu de contrastes n'est un recall initial dans43,9m SIRET.

Deux workers distincts : extraction puis fit, chacun 80s/4Gio/zéro nouveau
swapout, supervision externe90s, threads2. DuckDB mémoire1Gio, lecture filtrée
par clés TRAIN et sortie streaming ; au-delà500000 observations STOP, pas
troncature/évaluation partielle. Manifestes/hash source et implémentation,
taille/inode/mtime/footer Parquet avant/après. Gros fichier non rehashé en
entier, limite publiée. L'extraction scellée doit être COMPLETE avant fit.
Les limites de récupération/quarantaine du manifeste RNE source restent
héritées : cette extraction ne certifie pas l'exhaustivité juridique du RNE.
Aucun worker relancé automatiquement avec budget élargi. Code/contrat commités
avant mesure, milestone et résultat consignés dans handover.md.

Fold0 non lu/fold1 fermé, aval gelé, absence de données synthétiques ou alias
CRM comme features. Pas de lookup de vérité en génération : il n'y a AUCUNE
génération ni index serving ici. Ce contrôle prépare seulement la décision
sur une représentation officielle à exécuter ensuite sur le corpus national
complet, avec collecteur100 et sources indépendantes de tous labels.
