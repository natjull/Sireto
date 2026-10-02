# V68 — plan physique sparse direct100, pas un nouveau recall

Préenregistrement du2octobre2026 avant exécution. Changer la représentation
et son exécuteur, pas renommer un pool fixe ni déclarer un succès sur les
contrôles V63/V67. Le retrieval doit rester initial100, exact99,2% all-human,
Wilson/SLO et protections v38 inchangés. Fold0 non lu/fold1 fermé, aval gelé.

## Hypothèse physique à départager

Un index d'impacts non négatifs accepte un score appris query·document et
des preuves composites précompilées. Des bornes par blocs permettent de
sauter des documents incapables de battre le100e, sans modifier score,
ordre ni plafond. Prototype Python float64, pas une reproduction SPLADE ni
un moteur production Block-Max WAND. L'élagage est conservateur : arrondi
vers+inf après chaque addition, les égalités de borne restent visitées.
Identifiants utilisés seulement comme clés et tie-break, jamais features.
Seul heap100 persistant, aucun réservoir préalable à reclasser. Le scoring
interne des blocs du corpus n'est pas un pool candidat fixe matérialisé.

V67 a révélé un signal de sigle précédemment masqué par le stockage de
champs exacts. Compiler les sigles au build depuis CHAQUE alias officiel
courant séparé : langage équivalent aux regex V67,2..6 lettres, formes
juridiques/particules optionnelles, aucun code construit en mélangeant
plusieurs alias. Une complexité excessive STOP, jamais une troncature.
Pas de table de marques CRM ni propagation libre de nom entre filiales.

## Population et coût, jamais qualité

32768 documents établissement obtenus par requête document_type=SIRET,
prefix physique du moteur indépendant des queries ET labels. Ce n'est
PAS un échantillon national aléatoire ni un corpus autour des vérités.
Source nationale/inventaire/hashes et digest des impacts produits publiés.
16 premières queries CRM v34, whitelist v38/folds2/3/4, aucune ouverture
ledger/sidecar/GT/candidats anciens.8 sentinelles de coût à un terme choisi
par fréquence des mots/sigles OFFICIELS du cohort pour ne pas présenter des
queries sans matching comme preuve de rapidité. Sentinelles non-CRM, pas
des données métier synthétiques ni des exemples d'entraînement/qualité.

Poids1 fixes de coût, mots nom/adresse, sigles courants, commune/CP et mots
ou sigles×commune/CP préindexés. Ce n'est PAS le score V67 ni une recette
apprise ; l'API d'impacts admet des poids appris ultérieurement. BAN/XY,
graphe typé et preuve période/source pas encore servis ; les mots restent
en bags dans cette ablation, ne pas annoncer la représentation cible comme
implémentée. Géographie administrative en bonus, aucune exclusion hors zone.

Comparateur exhaustif SUR TOUT ce cohort, même score/float64/ordre/additions.
3 passes par query, ordre des2 plans alterné selon passe. Exiger parité
exacte des100 couples clé-score et ordre ; publier queries non vides,
termes actifs, fraction du corpus scorée, postings lus, latences p50/p95,
arrays mémoire, temps lecture/encodage/build, RSS et swap. Résultats clés
retournés ne sont PAS écrits ; aucune mesure de recall humain autorisée.
Tout est RAM après build, cache non contrôlé : pas SLO national/SSD ni
extrapolation linéaire du temps à43,9millions de sites. Pruning peut être
plus lent que l'oracle vectorisé à cette échelle : résultat valide à publier.

80s/4Gio/zéro swapout, CPU2, superviseur90s ; STOP scellé sans élargissement
automatique/reprise qualité. Ne pas faire de gros build national à partir
de cette seule mesure. Ensuite seulement : apprendre des impacts cohérents
identité/site en OOF et développer leur indexation complète ; preuve de
recall exige recherche initiale sur corpus national complet, jamais ce cohort.

Appuis, pas garanties transférées : [SC-Block](https://arxiv.org/abs/2303.03132)
pour apprentissage contrastif du blocking ; [SPLADE](https://github.com/naver/splade)
pour représentations sparse apprises et index inversé ;
[Block-Max pruning](https://research.engineering.nyu.edu/~suel/papers/pulse-sigir24.pdf)
pour accélération des scores sparse appris. Pas de modèle anglais générique
présenté comme prêt pour l'entity matching français, pas de SOTA revendiquée.
