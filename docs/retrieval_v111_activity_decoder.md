# V111 — essai TRAIN-only de contexte d'activité, pas recall SIRET

Hypothèse nouvelle : un embedding NOM masqué et figé peut prédire une
pertinence d'activité brute officielle qui ne requiert pas un mot commun
entre CRM et nom officiel. Aucun dictionnaire d'alias CRM, mapping métier de
nomenclatures ou identifiant feature. Le contexte V110 complet et lossless
sert aux jointures TRAIN. Tous les codes sont catégoriels, jamais ordinaux.

512 scènes naturelles TRAIN préexistantes V52, folds2/3/4 exclusivement,
23749 négatifs naturels ; AUCUN candidat n'est émis/injecté par cet essai.
Les512 ne sont PAS une population de validation nationale. Familles de fold
tenu retirées du fit et des négatifs inner ; familles inner retirées du fit.
Les positifs manquants du catalogue ou les contrôles sans négatif sûr causent
STOP, pas suppression de ligne. Les inconnus/vide sont conservés dans le
dénominateur, mais ne génèrent pas une preuve d'identité par valeur manquante.

Recettes sélectionnées exclusivement sur contrôles inner : fréquence TRAIN
non conditionnelle, ou décodeur ridge1/10/100 des384 dimensions NOM et biais
vers tous les codes du catalogue officiel. Targets uniquement positifs du
TRAIN admissible ; aucun négatif/famille de contrôle n'est utilisé comme
target. Paramètres refités sans familles du fold tenu. Les scores sont
bornés0..1, PAS des probabilités calibrées, et tous les codes sont considérés
sans top8, seuil de vocabulaire ou filtre actifs. Aucune prédiction site ne
peut distinguer deux établissements portant la même activité : ties et
proportion de mêmes codes publiés, pas cachés.

Mesures : error pairwise avec égalité de poids par requête, head error,
nombre de négatifs au-dessus, activité exacte top1/top5, inconnus et égalités,
avant/après fréquence TRAIN, séparément par fold. Ces top1/top5 sont des
classes ACTIVITÉ et ne sont JAMAIS Recall SIRET@100. Aucun reranker, aval,
recherche nationale, modèle déployé ou garantie métier dans ce probe.

Guard80s/4Gio/zéro swap, superviseur90s, deux threads CPU ; sources/hash
code vérifiés avant/après. Fold0 jamais lu, fold1 fermé. Les champs état,
siège, dates sont conservés V110 mais ne sont pas encore appris ici.
La pertinence observée doit ensuite être testée dans une recherche nationale
initial100 intégrant noms officiels, adresses, XY/BAN et filiations avec
bornes correctes, sans pool fixe. Un bon contrôle TRAIN ne franchit aucun
gate99,2% exact all-human/OOF complet/Wilson/généralisation/attribution/SLO.
