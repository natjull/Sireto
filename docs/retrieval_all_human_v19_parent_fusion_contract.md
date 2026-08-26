# Contrat de développement v19 — fusion parent@2000

Date de gel : 27 août 2026

## Objet

Le v18 a démontré une borne oracle brute de 389/522 bons parents, mais son pool
contient une médiane de 2 522 SIREN distincts par requête. Le v19 mesure si ces
parents peuvent être conservés sous un plafond strict de 2 000 sans utiliser la
vérité du fold évalué.

Le v19 est un workbench de développement same-CRM. Il ne constitue ni une
évaluation finale, ni une autorisation de run national, ni une preuve de
généralisation à un autre client ou millésime.

## Protocole gelé

1. Les deux fichiers de signaux v18 sont rechargés par SHA-256.
2. Les 36 features parent sont matérialisées sans ouvrir aucune vérité : nombre
   de canaux, rangs réciproques globaux, RRF et présence/rang réciproque des 15
   canaux.
3. `query_id`, fold, SIREN, SIRET, NIC, origine, cohorte et toute information de
   vérité sont exclus des features.
4. Une baseline RRF fixe (`k=60`) est scellée à 2 000 avant ouverture des labels.
5. Pour chaque fold externe, LambdaMART apprend exclusivement sur les deux
   autres folds : F2←F3+F4, F3←F2+F4, F4←F2+F3.
6. Les SIREN vérité sont disjoints entre folds. Les requêtes sans positif naturel
   dans le pool sont exclues du fit, sans injection, mais restent des erreurs
   dans l'évaluation.
7. Au train seulement, le positif naturel et au plus 512 négatifs déterministes
   sont conservés par requête. Chaque requête a ainsi un poids de groupe voisin.
8. Le modèle prédit tous les parents du fold externe et ne garde jamais plus de
   2 000 parents. Aucun label externe n'est lu pendant le fit.

Les hyperparamètres XGBoost, les features, les canaux, le cap et les gates sont
figés dans `config/retrieval_all_human_v19_parent_fusion.json` avant exécution.
Aucun retuning post-résultat n'est autorisé dans ce milestone.

## Gates

- minimum utile : 334/522 parents à `parent@2000`, correspondant à la borne
  Master 99,2 %;
- palier suivant : 405/522, correspondant à 99,5 %;
- contrôle : 512/512 contrôles doivent rester couverts;
- absence de régression par origine relativement à l'union v17 : au moins
  256 cas récents et 73 historiques;
- confiance avant étape longue : 464 global et 160/158/146 par fold;
- ressources : 45 minutes, 14 Gio RSS, zéro page swapout.

Un passage à 334 autorise seulement la construction du prochain workbench
Master/site. Fold1 reste fermé. La revendication cross-CRM exige un shadow set
nouveau, temporel et non retunable.
