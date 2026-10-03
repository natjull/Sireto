# V104 — servir réellement le modèle sigles/sémantique gelé V67

Avant mesure. Objectif inchangé : recherche initiale100 exact99,2%, pas de
pool2000/500 ni ranker aval. V101 complet reste114/128 ; son extension de
corpus n'a pas donné de gain. Tour précédent PROGRESS par résultats/audits,
pas par un gain fictif. Fold0 non lu, fold1 fermé, aval gelé.

V67 a corrigé des alias exacts indexés mais non stockés. Ses trois folds
avaient choisi HEAD_FULL16 en contrôle interne, avant audit des14 pertes.
Ce modèle n'a jamais été évalué en recherche nationale initial100. V104
teste cette recette GELÉE sans nouveau fit, réglage de poids ou dictionnaire
des erreurs. Les16 poids, vocabulaires sûrs par fold et projections sont
ceux du manifeste V67 hashé. Les parents outer/inner sont exclus des
positifs ET négatifs du fit et du vocabulaire (code source V67 scellé).
Les contrôles512 ne valent pas recall national, ni dénominateur all-human.

Le natif compile11 preuves nom/adresse/numéro/commune/CP/exacts +4 preuves
de sigle courant (global,place,numéro+place,adresse+place), puis termes
officiels sémantiques OOF<=8. Les sigles viennent d'un alias courant officiel
entier via RegexQuery sur champs exacts indexés, pas d'alias CRM stockés.
Grammaire/STOP/maximum2 tokens initiaux sont les LIMITES préexistantes V67,
inchangées dans ce test, pas sigles historiques inventés ni compréhension
de toutes marques/équipements revendiquée. Le bonus géographique des preuves
conditionnelles ne filtre jamais tout le corpus.

Le score initial=max(natif16 V67,64 × meilleur nom-chemin V101 × XY).
Le poids64 et les grades1/.5/.25 restent V87, non ajustés sur14 erreurs.
Toutes57804962 tuples officiels V100,43896818 sites V96, memberships
parents exhaustifs et relations V84 un hop restent présents. Géométrie
BAN faible/absente et XY manquant inchangée, natif intact si inconnue.
Dates/activité disponibles restent NON actives dans ce score ; pas de
co-validité proclamée ni de champ absent silencieusement annoncé utilisé.

Même collecteur V101 à trois flux exhaustifs, accès ponctuel complet,
frontière certifiée et heap100, ties canoniques. Même buffers8192/65536,
pages256 physiques, pas caps de preuve/fanout. Oracles indépendant tout
corpus fixture et score natif16 comparé aux composants appris incluent
preuves sigle positives, pas seulement all-zero.

Réutiliser uniquement les embeddings NOM CRM V78, pas anciens candidats
ni expansions ; vérifier encodeur V52 entraînement identique, recalculer
les8 termes avec projections/vocabulaire V67 par fold.128 queries234
identiques au comparateur V101 gelé. Aucun label en génération. All-human,
exact/op, sidecar scellé, absences publiées, évaluation distincte après128.

Même gardes V101 : préflight16 mean<=4s,20s/requête,80s/6Gio/zéro swapout,
90s supervision,900s contrôleur.900s est le budget de cette nouvelle recette
comme V101, PAS reprise/extension du STOP V66 HEAD_WORD360s. Pas évaluation
de population partielle, élargissement automatique ou dépense externe.
Tantivy inventaire/mtime vérifié, gros segments non entièrement rehashés.

Pilote réutilisé pas holdout/fullOOF ; même100% ne valide pas Wilson>=99%,
folds/généralisation/attribution exhaustive/SLO p95<=1s/p99<=2s. Aucun
READY_FOR_FOLD1 ni garantie SOTA. Si sigles trop lents, nouveau plan physique
source-only, pas corpus autour des labels ni abandon des noms/adresses/XY.
