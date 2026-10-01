# V36 reconnaissance des entreprises et concurrence complète

Préenregistrement du 2 octobre 2026. Objectif actif inchangé : au moins
99,2 % exact à 100 candidats, gate préalable 99,4 % à 500. Les folds 0 et 1
restent fermés. Ce pilote explicatif sur les 1 024 requêtes v35 ne peut
ouvrir le fold 1, même s'il franchit ponctuellement ces valeurs.

## Diagnostic et hypothèse

Dans v35, le bon SIREN figure dans le top100 de 981 requêtes contre 973 pour
le SIRET exact. Sur 51 échecs exacts, seuls 8 ont donc le bon parent déjà
admis. Le rang parent v23 concentre 65,5 % de l'importance de gain des modèles
SIRET. Ces parents sont issus de petits fits historiques ; la fusion v35
ne prouve pas qu'une représentation riche de l'entreprise est épuisée.

Le fit v35 conserve 128 négatifs alors que l'inférence en rencontre jusqu'à
1 999. C'est une différence de concurrence, non une preuve que le sampling
est intrinsèquement mauvais. Les preuves de nom sont également réduites
aux noms courants ; les rangs des canaux indépendants sont principalement
écrasés dans leur somme RRF, et les distances dépendent d'un voisinage512.

Hypothèse principale : une représentation multivue officielle, combinée
à une concurrence naturelle complète, améliore la reconnaissance du parent
et la conservation du SIRET. Les signaux numériques viennent des documents
officiels ou de leurs retrievers, jamais d'alias CRM ni d'identifiants.

## Recette unique et ablation explicative

La recette primaire v36 conserve les 51 features v35 et ajoute :

- similarités JW, Jaccard, couverture des tokens de requête et disponibilité
  séparées pour noms parent courants, parent BODACC SUPPORTING, noms de site
  courants et historiques ; une preuve SIRET reste liée à son propre site ;
- distances Lambert93 calculées pour chaque site dont les coordonnées sont
  disponibles, avec indicateur de disponibilité, incertitude BAN et meilleur
  soutien géographique du parent ; aucune exclusion par rayon ;
- pour chaque canal officiel déjà généré, rang et score relatif au maximum
  de la requête/canal, séparément au niveau parent et site ; les relations
  structurées conservent leur provenance, les canaux ne sont pas inventés.

Les features sont calculées sur le réservoir label-blind avant jointure des
quatre contextes Master v35 scellés. Leurs statistiques ne dépendent pas de
l'union des Masters appris, ce qui empêcherait de garantir l'isolation.
Membership et rangs de génération v35 sont conservés. Les contexts de fit
inner OOF sont repris exactement ; la couverture Master reste fixe.

Fit sur tous les candidats naturellement présents des deux folds autorisés,
hors équivalents opérationnels neutralisés. Aucun sous-échantillonnage de
négatifs. Une requête sans positif reste au dénominateur d'évaluation et est
exclue uniquement du fit. Un LambdaMART unique score au plus2 000 références,
son ordre donne500 puis100. Aucun quota, choix oracle ou suppression d'un
label ambigu. Pas d'hydratation riche ni de changement des modèles aval.

XGBoost3.1.3 CPU,200 arbres,profondeur5,learning_rate0,06,lambda5,hist,
max_bin128,subsample1,colsample1,seed20261001,2 threads,`rank:ndcg`, paires
`topk` avec k100. Pas d'early stopping ou de choix par score OOF externe.
Le NDCG reste un surrogate ; seul le recall exact mesuré répond au contrat.

Une ablation préenregistrée utilise seulement les anciennes51 features avec
ce même fit complet/topk100. Elle explique l'effet de l'apprentissage par
rapport à la représentation. Elle ne peut remplacer la recette primaire si
elle la bat. Aucun balayage d'hyperparamètres, aucun réglage après métriques.

## Mesure et ressources

Même population que v35, tous les folds2/3/4, pour comparaison appariée.
Publier exact/opérationnel, historique/V2/V3, gains/pertes vs v35 et ancien
v30, pertes génération/préclassement/admission, caps et Wilson. Vérifier
les hashes des sidecars officiels label-side avant attribution. Conserver
les groupes sans positif, les absences et les équivalences dans l'audit.

Cache officiel projeté une fois, 2 threads, DuckDB3Go, garde globale6GioRSS,
zéro nouveau swapout et20minutes. Aucun index national relancé, aucun GPU
loué, aucun service externe, aucun téléchargement de modèle. Checkpoints
et modèles sont isolés sous une nouvelle racineCATNAT. Le code v35 et les
inventaires gelés v33/v34 ne sont pas modifiés.

Conclusion primaire imposée par le contrat : `PIVOT_GENERATION` si500
échoue ; `PIVOT_ADMISSION` si500 passe mais100 échoue ; `PIVOT_LATENCY`
seulement après mesures adéquates. Un résultat pilote reste non admissible
au fold1 et ne prouve pas la généralisation cross-CRM. Le plafond complet
98,96059 % exige encore une amélioration de génération pour99,2%@100.

## Références et portée

[Wang et Zhang, NAACL2024](https://aclanthology.org/2024.naacl-long.483.pdf)
étudient la généralisation de bloqueurs neuronaux et lexicaux : le modèle
dominant dépend du régime d'évaluation, du domaine et de la taille du pool.
Un modèle réputé SOTA ne certifie donc pas notre objectif CRM.

[Documentation XGBoost3.1.3](https://xgboost.readthedocs.io/en/release_3.1.0/tutorials/learning_to_rank.html)
décrit les objectifs et la construction de paires topk. Le choix k100 est
ici une hypothèse préenregistrée alignée sur le budget produit ; son effet
doit être mesuré, et la documentation ne garantit pas un gain.
