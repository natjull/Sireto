# Résultats du pilote de classement SIRET v35

Le 1 octobre 2026, la fusion apprise parent/site améliore le recall exact à
100 candidats de **88,86719 % à 95,01953 % sur les mêmes 1 024 requêtes**.
Le gain net est de 63 cas, avec 81 gains et 18 régressions. Ce n'est ni le
résultat des 23 571 lignes, ni l'atteinte de l'objectif 99,2 %.

La recette, le runner et les tests ont été préenregistrés dans `c966cbb2`.
Le [contrat du pilote](retrieval_v35_site_fusion_contract.md) précise la
sélection indépendante des labels, le cloisonnement et les ressources.

## Résultats comparables sur le pilote

| Profondeur | Ordre heuristique v30 | Fusion SIRET v35 | Gains et pertes |
|---|---:|---:|---:|
| 100 | 910 / 1 024 = 88,86719 % | 973 / 1 024 = 95,01953 % | 81 / 18 |
| 500 | 996 / 1 024 = 97,26563 % | 1 000 / 1 024 = 97,65625 % | 13 / 9 |
| 2 000 | 1 018 / 1 024 = 99,41406 % | 1 018 / 1 024 = 99,41406 % | 0 / 0 |

Les vues opérationnelles sont publiées séparément dans le manifeste ; elles
coïncident ici avec les vues exactes à chacune des trois profondeurs. Aucune
promotion exclusivement opérationnelle à 100. Parmi les requêtes du pilote,
20 présentent plusieurs équivalents actifs et 50 ont un équivalent actif
dans le top100 alors que la vérité exacte est fermée. Ce dernier nombre
décrit l'ensemble de candidats, pas une prédiction top1 ni une précision.

La borne basse Wilson 95 % du recall exact à 100 est **93,51075 %**. Les
segments historique, qualification V2 et qualification V3 sont tous publiés
dans le manifeste sans filtrer le dénominateur. Les 16 labels absents ou
inexploitables de la population source restent publiés séparément par fold
(8, 3, 5) ; toutes les 1 024 requêtes préchoisies du pilote sont mesurées.

| Fold externe | Requêtes | Heuristique à 100 | Fusion à 100 | Fusion à 500 |
|---|---:|---:|---:|---:|
| 2 | 328 | 286 | 311 = 94,81707 % | 321 = 97,86585 % |
| 3 | 348 | 318 | 335 = 96,26437 % | 342 = 98,27586 % |
| 4 | 348 | 306 | 327 = 93,96552 % | 337 = 96,83908 % |

L'admission récupère 63 des 108 vérités que l'ancien cut100 perdait alors
qu'elles étaient présentes dans le Master, soit 58,3 % du déficit de
classement **sur ce pilote**. Elle ne résout pas le problème de couverture
du Master complet : son plafond reste 98,96059 %.

## Ce qui a réellement changé

Un seul LambdaMART SIRET combine 51 features légères : preuves de nom
parent/site, voie, numéro et écart numérique séparés, suffixe, localité,
état/siège, rangs et consensus upstream, ambiguïté du parent, distance BAN
disponible et qualité de géocodage. Le même score donne les préfixes 500 et
100 ; aucune règle de quota par SIREN ou de protection d'un ancien préfixe.

Le modèle utilise [le classement groupé de XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html).
Cette expérience CPU n'est pas un benchmark SOTA. Les preuves hétérogènes et
les paires difficiles sont des principes d'entity matching, illustrés par
[l'article Ditto](https://arxiv.org/abs/2004.00584), pas une garantie de gain
ni une reproduction de son modèle.

Les importances de gain, moyennées sur les trois modèles, privilégient le
rang parent appris (environ 65,5 %), puis le RRF parent (5,1 %) et l'état du
site (3,6 %). Elles suggèrent que la recombinaison du parent joue un rôle
important. Elles ne prouvent PAS que le relâchement du numéro ou la distance
est la cause du gain ; une ablation serait nécessaire pour cette attribution.

## Cloisonnement et contrôles

Les Masters de fit ont été reconstruits avec les modèles parent inner OOF,
pas empruntés aux anciens Masters outer des autres folds. Cela évite une
fuite des labels externes par les scores ET par la sélection des candidats.
L'outer Master reproduit exhaustivement le v30 scellé pour les 32 lots :
identités, rangs, scores parent/RRF et preuves conjointes identiques.

Toutes les features ont été scellées avant ouverture des labels. Les seuls
positifs de fit sont naturellement présents dans leurs Masters ; les
équivalents opérationnels sont retirés des négatifs. Les groupes sans
positif restent dans la mesure (5, 7 et 3 groupes non entraînables pour les
modèles outer2, outer3 et outer4). Les fits utilisent respectivement 691,
669 et 673 requêtes, sur les deux autres folds uniquement.

Le ledger officiel ne présente aucun SIREN humain sur plusieurs folds
2/3/4, vérifié aussi sur la population complète. Ceci protège la séparation
d'entités, mais ne démontre pas la généralisation à un autre CRM. Les folds
0/1 n'ont pas été ouverts, aucun identifiant n'a été présenté au modèle,
aucun modèle aval n'a été changé et aucun service externe n'a été appelé.

Les maxima observés sont 2 000 références scorées, un préfixe 500 et une
sortie 100. Ce sont des sélections légères offline, **pas une validation
d'hydratation riche ou des SLO end-to-end en production**.

## Pertes restantes et investigation

Sur les 51 échecs exacts à 100, 3 sont des parents absents, 3 des évictions
réservoir vers Master, 18 des évictions Master vers 500 et 27 des évictions
500 vers 100. L'attribution réutilise le sidecar officiel label-side scellé.

Diagnostic exploratoire sur les 45 pertes de classement : 38 ont une
similarité maximale de nom parent/site inférieure à 0,7, 26 une similarité
de voie inférieure à 0,7, 23 un numéro discordant lorsqu'il est fourni, et
37 n'ont pas de distance dans les voisins BAN conservés. Ces familles se
recouvrent ; les seuils servent uniquement à décrire, jamais à sélectionner
ou filtrer une évaluation.

Dans le stock officiel, 18 de ces 45 cas ont une distance calculable entre
coordonnées du site et position BAN, dont 8 à moins de 500 m. Une distance
au-delà des voisins 512 pourrait donc apporter une preuve supplémentaire,
sans garantie de gain. Les noms BODACC de rôle SUPPORTING et les noms
historiques existent mais sont ignorés par les features de nom actuelles.
Leur simple présence ne prouve pas qu'ils résoudront les erreurs : cette
projection label-side exploratoire ne doit jamais devenir un lookup de GT.

Suite recommandée : une recette unique enrichie de preuves officielles
manquantes avec provenance et incertitude, ou un test de montée en données
à hyperparamètres inchangés. Préenregistrer chaque expérience, préserver
la reconstruction nested, vérifier les 18 régressions et le faible gain
à 500 ; ne pas lancer une génération nationale inchangée. Aucun prochain
calcul lourd n'a été lancé automatiquement.

## Artefacts et coût

Racine nouvelle :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v35_site_fusion/pilot_20261001_a/`.

Manifeste résultat SHA256
`8000f1f8851d28af788708007b14c7fc2e2e5735d7fa8bbd34f46c27ca37112a`.
Manifeste des features SHA256
`806838ae9abe249c396612da7de9799074993732aa9eea128aac75571bcc8f4f`.
Prérequis : génération v34 SHA
`022b5c4e4436b335e78d217a81319c122390631e5698dd17c9d7db3f36c77e26`.

Durée sous garde **91,82 s**, pic RSS **2 216 607 744 octets = 2,06 Gio**,
zéro nouveau swapout, 2 threads CPU. Cache officiel unique en 12,39 s ;
premier lot de 32 en 2,03 s, puis environ 2 s par lot. Trois fits et leurs
évaluations en 3,35 / 3,33 / 3,38 s. 1 891 016 candidats outer scorés ; pas
de rerun des index nationaux. Douze tests ciblés passent.

Statut : `MEASURED_BOUNDED_NESTED_OOF_SITE_FUSION`. Conclusion active
**PIVOT_GENERATION**, avec progrès d'admission démontré uniquement sur le
pilote. Gates 99,4 % à 500, 99,2 % à 100, Wilson, folds et SLO non atteints
ou non mesurés. **Pas de READY_FOR_FOLD1.**
