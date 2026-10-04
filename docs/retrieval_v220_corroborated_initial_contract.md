# V220 — sigles composés et confiance des fragments, avant mesure

Objectif inchangé : recherche initiale <=100 SIRET exact99,2%, avant tout
modèle aval. V217/V218 restent immuables,118/128 exact et119/128 opérationnel.
V219 dix pertes diagnostiques scellées, TOUTES sous frontière, pas nouvelles
métriques. Les décisions ci-dessous sont du développement informé par ces
erreurs : le panel128 réutilisé n'est PAS un holdout ni une généralisation.

Deux changements génériques, sans dictionnaire CRM ni correction de labels :

1. Pour TOUS les tokens CRM alphabétiques2..6 lettres hors particules/formes
   juridiques/mots de contexte génériques, compiler TOUTES les partitions
   en au moins deux préfixes de mots. Une partition peut affecter plusieurs
   lettres au même mot. Match d'UN alias officiel courant entier dans UN des
   trois champs exacts legal/trade/site ; particules/formes juridiques
   explicitement permises, aucun suffixe arbitraire ni emprunt à un autre
   alias/site. Pas de tokens initiaux top-k, ni identifiant comme feature.
   La preuve sigle exige commune OU CP cohérent pour CE nouveau signal,
   pas de filtre global : tous les autres chemins nationaux restent ouverts.
   Nom = MAX(couverture historique V56, membership sigle1), preuves nom+place
   et nom+adresse+place recomposées. Cela change la sémantique du signal nom,
   pas les poids OOF11 gelés, pas l'exact-string ni un reranker.
2. Un alias entier ou fragment multiword conserve1. Un singleton contenu
   dans un CRM plus long vaut1/nombre de tokens de contenu CRM (particules
   et formes juridiques exclues). Un nom officiel solitaire égal au CRM
   conserve1. MAX avec l'exact original, pas de double comptage. La preuve
   exact-nom/adresse/place conserve ce grade, pas une promotion silencieuse1.

Ces deux changements sont testés ENSEMBLE : pas d'attribution causale ni
ablation annoncée. Aucun fit ou coefficient réglé sur128. Aucun alias CRM,
label-side V103/V219, identifiant humain ni ancienne sortie comme source.
Le langage ne couvre pas les sigles longs, les alias historiques absents
des trois champs exacts courants ou les liens marques/équipements→entités
qui demanderaient une preuve officielle supplémentaire. Aucune garantie99%.

Conserver V217 : nom-owner complet134/149, successions133 un hop, anneauxXY
complets natifs, confiance BAN, termes8 et tête11 OOF V77/V78 gelés. UNE
recherche native mondiale `limit=100`, aucun réservoir2000/500/fixe, aucune
expansion depuis100 suivie d'un rerank. Fermés accessibles. Corpus natif
43896818 SIRET/82772691 docs, pas44403034 append réindexés annoncés.

Tous128 CRM dans l'ordre query-only original folds234, labels fermés tant
que tous16 workers8 réels exit0/PASS214 et scellement global complet. Nouvelle
racine, STOP/LIVE non réutilisables. Budget214 autorisé2GiB/20s query ouverture
incluse/80s worker/quiet30 pression1/free>=50, un thread, swap global telemetry
non attribué non éliminatoire. Mean16 descriptive non gate. Sources utilisées
FULL SHA avant/après +epochs ; natif inventaire/méta, segments NON FULL SHA.

Évaluation distincte V221 APRES scellement intégral : exact/op séparés,
all-human et qualification non filtrante, absences, folds, Wilson, latences,
appariement V218 et V101. Aucun READY/fullOOF/SLO/généralisation même100% sur
ce panel réutilisé. Fold0 contaminé non lu, fold1 fermé, modèles aval gelés,
pas de dépense externe ni nouveau run23k sans une prochaine étape justifiée.
