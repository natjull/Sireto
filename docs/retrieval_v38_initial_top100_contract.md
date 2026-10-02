# Recherche initiale nationale à 100 candidats

Contrat v38 du 2 octobre 2026, avant mesure. L'utilisateur exige une recherche
initiale dont la sortie contient au plus100 SIRET et conserve le SIRET humain
exact dans au moins99,2% des requêtes exploitables. Les modèles aval sont
hors de cette mesure. Les anciens gates2000/500 ne définissent plus ce volet.
Leur historique n'est ni modifié ni comparé comme une recherche équivalente.

## Hypothèse limitée de ce premier test

Tester une requête BM25 multichamp directement sur l'index national existant,
sans pool candidat préalable, expansion depuis un SIREN appris ou reranker.
Le collecteur natif du moteur retourne100 documents SIRET ; seules ces
références sont matérialisées. Les documents SIREN ne sont pas admissibles.

Les champs nom courant, enseigne, nom de site, historique et supporting
restent séparés dans l'index. On calcule pour chacun la moyenne des scores
BM25 des mots de la requête puis le maximum entre ces vues, avec aussi la
vue ngrams de noms courants. Même principe pour les adresses courantes,
historiques et supporting et leurs ngrams courants. Les scores nom et adresse
sont additionnés. Au moins une preuve textuelle est obligatoire, mais aucune
obligation de correspondance simultanée du nom ET de l'adresse.

Commune et code postal donnent chacun un bonus constant1, combinés par
maximum et non par somme. Ils ne filtrent pas le corpus. Poids nom1,
adresse1, géographie1 ; aucun réglage après lecture des résultats.
Les vues ngrams utilisent la même moyenne BM25 que le canal conjoint v30.
Il n'y a ni exclusion par numéro ni bonus d'activité qui supprimerait les
vérités fermées. Les identifiants servent à produire/dédupliquer la sortie,
jamais de signal de similarité.

Il s'agit d'une première hypothèse lexicale avec géographie administrative
souple, PAS d'une recherche par distance XY. L'index interrogé n'a pas de
champs coordonnées. Son contenu historique n'inclut pas automatiquement
toutes les autres sources temporelles disponibles séparément. Le test ne
prétend donc pas exploiter déjà toutes les preuves ni toutes les relations.

Comparateur explicatif fixé : requête conjointe v30 nom courant ngrams ET
adresse ngrams, même corpus, filtre commune/CP historique, collecteur100
direct, sans la suite v23/v30. Ce comparateur n'est PAS toute l'ancienne
recherche multicanal. Aucun des anciens scores après fusion apprise n'est
présenté comme baseline de recherche initiale.

## Population et déroulement

Index full scellé `hierarchical_retrieval_v3/4b332899d8698354` :
82772691 documents, dont43896818 documents établissement. Vérifier manifeste,
métadonnées, inventaire des segments et nombre de documents avant/après.
Les gros segments ne sont pas entièrement rehashés pendant ce préflight :
publier cette limite. Aucun corpus réduit autour de la vérité.

Population fixée :128 requêtes des quatre premiers lots32 du replay v34,
dans son ordre hashé préexistant indépendant des labels, folds2/3/4 seulement.
C'est un sous-ensemble du panel de développement déjà utilisé, pas un
holdout inédit. Sources de requête uniquement CRM ; aucun label, ancien hit
ou candidat de vérité dans la génération. Les labels ne sont ouverts qu'après
scellement de l'intégralité des sorties128, dans un processus d'évaluation
séparé. Aucun choix de variante selon ces métriques.

D'abord16 requêtes sans labels pour le coût : deux lots8 séquentiels.
Continuer jusqu'à128 seulement si aucune erreur/intégrité/ressource, chaque
worker sous6GioRSS et zéro nouveau swapout, et temps total moyen des deux
recherches par requête <=4s sur16. Ce seuil est un budget expérimental,
pas un SLO de production. Worker90s au plus par supervision externe ;
garde interne80s. Contrôleur900s, processus successifs, RAYON_NUM_THREADS=2.
Une garde peut stopper le test : conserver les sorties partielles mais ne
pas les évaluer comme la population128. Pas de relance automatique élargie.

## Mesures et protections

Publier exact/opérationnel@100 séparément, caps, dénominateur complet,
Wilson95, gains/pertes appariés, folds, label historique, V2/V3, origine et
état. Métadonnées manquantes publiées UNKNOWN, jamais inventées. Les16
labels sources absents/inexploitables restent publiés séparément. Les
qualifications et les groupes ambigus ne filtrent aucune requête.

Reprendre et vérifier le ledger et le sidecar officiel label-side de
l'évaluation v34. Pour ce chemin sans expansion ni admission séparée,
distinguer HIT, OFFICIAL_CATALOG_ABSENT et NOT_RETRIEVED_AT100. Cette dernière
classe n'attribue pas sans preuve la perte à l'absence d'index, à un filtre
ou au score ; leur attribution fine reste non mesurée. Ne pas faire de
lookup de vérité vers le moteur pendant la génération.

Publier temps par requête, p50/p95/p99 avec qualification froid/chaud inconnue
si non contrôlée, temps total, RSS et swap. La séquence primaire/comparateur
alterne par requête pour réduire le biais de cache ; ce petit essai ne valide
pas les SLO. Pas d'hydratation riche, fit, modèle téléchargé ou cloud.

Ce pilote ne peut donner READY_FOR_FOLD1, même à100%. La validation finale
nécessitera un protocole OOF figé complet, recall exact>=99,2%, Wilson95
bas>=99,0%, chacun des folds>=98,8%, p95<=1s et p99<=2s, et attribution
officielle des pertes. Fold0 jamais lu, fold1 fermé jusqu'à autorisation
one-shot explicite. Verdict de ce test : STOP si intégrité/coût bloque,
PIVOT_GENERATION si rappel insuffisant ; aucune garantie SOTA.

## Appui scientifique et portée

[Sparkly](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-vldb2023.pdf)
justifie d'évaluer sérieusement une recherche lexicale top-k pour le blocking.
Ce test n'est pas une reproduction de Sparkly Auto et ne reprend pas ses
performances comme garanties. Les choix de multivue, poids égaux et géographie
souple ci-dessus sont nos hypothèses à tester, pas des conclusions du papier.
