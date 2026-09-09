# V20 — diagnostic de sampling et Master SIRET avec preuves conservées

Gel avant résultats : 9 septembre 2026. Développement uniquement; folds 0 et 1
fermés, aucune autorisation de national ni de `READY_FOR_FOLD1`.

## Diagnostic causal v19

Pour chaque outer fold exclu, les deux folds autorisés sont séparés en train et
validation interne par `SHA256(v20-inner|GT_SIREN) mod 5 == 0`. Les SIREN sont
disjoints. Comparaison appariée, mêmes hyperparamètres/seed/features/groupes :
512 premiers négatifs RRF historiques, ou 256 premiers et 256 représentants
uniformément répartis dans la queue. Aucun positif n'est ajouté au pool. Les
requêtes sans positif restent erreurs d'évaluation; elles ne peuvent entraîner
un ranker pairwise. Les folds externes ne servent pas à évaluer ces modèles.

Les features historiques incluent un rang v17 fabriqué en triant les SIREN.
Elles sont conservées uniquement pour isoler causalement le sampling, et sont
déclarées non conformes pour une nouvelle recette. Aucun modèle de cet audit
n'est un candidat à la production. Les résultats précédents ne sont pas modifiés.

## Nouvelle génération indépendante des labels

Le plafond vise 2 000 **SIRET**, pas 2 000 parents. Le préfixe v11 top1750 est
intégralement conservé; les places restantes sont remplies par un budget global
de rescue. Il ne s'agit pas d'un quota final top100 : aucun top500 ou top100
n'est produit dans cette étape de génération.

Sources existantes exclusivement : raw v11, witnesses SIRET v18, sidecar
officiel SIRENE. Les sources sont vérifiées par SHA. La projection CRM lit
uniquement les champs disponibles en production. Les labels sont ouverts par
un programme séparé après scellement du Master.

Le rang artificiel v17 n'intervient dans aucun score. Le soutien parent est la
somme des rangs réciproques des autres canaux. Chaque témoin direct reste
attaché à son SIRET. Les établissements locaux (INSEE ou CP CRM) de tous les
parents soutenus sont examinés avec des métadonnées légères. Un témoin direct
hors de cette géographie n'est pas éliminé. Aucun nombre fixe de sites par
parent n'est imposé. La preuve d'un témoin n'est pas copiée à tous ses siblings.

Score de rescue gelé : somme RRF des preuves directes du SIRET,
plus `1/(60 + rang_parent + rang_adresse_dans_parent)` pour les sites locaux.
Le rang adresse utilise accord conjoint numéro/suffixe/voie, puis INSEE,
numéro, suffixe, similarité voie, état actif. Identifiants seulement pour
jointures, déduplication et départage terminal exact, jamais comme feature.
Les scores sont heuristiques de génération; aucun LambdaMART SIRET ne score
le réservoir exploratoire. Le futur modèle SIRET recevra au plus 2 000 références.

## Mesure et arrêt

Évaluation des 1 034 requêtes entières, sans retrait de qualifications. Publier
exacts et opérationnels séparément, versions V2/V3/origine/fold comme segments.
Le sidecar officiel label-side est construit et scellé depuis le snapshot
SIRENE pour attribuer les pertes : catalogue absent, parent absent, site non
expansé, éviction du réservoir vers Master2000. Les étapes Master→500→100 sont
non mesurées, jamais supposées réussies.

La projection nationale `23049 + rescues` est conditionnelle à la conservation
du même préfixe partout; elle n'est pas un recall national mesuré. Les 512
contrôles ne remplacent pas un test national. Une progression exige d'abord
la marge pour PreRank@500 ≥99,4 % puis les gates top100 du contrat actif.
L'absence d'attribution officielle complète impose STOP.

Protocole sans grille ni retuning après résultats. Préflight 1..16 requêtes
puis un workbench complet, garde RSS 14 Gio / zéro swapout / 30 min par phase,
calcul Mac local, aucun service externe. Les phases réussies sont scellées;
un répertoire incomplet n'est pas un résultat et n'est pas repris silencieusement.
