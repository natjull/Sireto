# Handover pour reprendre le retrieval SIRETO

Note du 1 octobre 2026, destinée à l'agent qui reprend ce chantier. Elle
résume les faits vérifiés, les difficultés et les investigations proposées.
Elle ne lance aucun travail et ne vaut pas autorisation de dégeler l'aval.

Le problème est double : le Master de 2 000 candidats ne contient pas encore
assez de vérités humaines, et son ordre perd beaucoup de bonnes réponses
quand on le coupe à 100. La génération complète est déjà calculée et son
évaluation est désormais terminée. Ne pas repartir sur 23 heures de calcul
pour obtenir les mêmes métriques.

Addendum du même jour : un [pilote de fusion SIRET v35](retrieval_v35_site_fusion_results.md)
est désormais terminé (`c966cbb2` préenregistrement, `9a0d8319` résultats).
Sur les mêmes 1 024 requêtes, le recall exact à 100 passe de 88,86719 % à
95,01953 % ; à 500, de 97,26563 % à 97,65625 %. Trois nouveaux modèles
SIRET de retrieval sont entraînés, les modèles aval restent gelés. Le reste
de cette note décrit l'état avant ce pilote. Il ne faut pas confondre son
résultat avec celui du complet, ni réutiliser les Masters outer d'autres
folds pour un fit : reconstruire les Masters de fit avec les parents inner
OOF, car l'ancienne sélection de candidats peut elle-même fuiter les labels.

## Objectif et règles à lire en premier

Lire complètement `AGENTS.md`, `docs/retrieval_all_human_v12_contract.md` et
`docs/siret_operational_equivalence_policy.md` avant toute action.

L'objectif principal est le Recall SIRET humain exact >= 99,2 % avec un
plafond absolu de 100 candidats. Le développement utilise uniquement les
folds 2/3/4 en outer OOF. Le fold 0 est contaminé, diagnostique historique
seulement, et ne doit pas être lu par le développement v12. Le fold 1 reste
fermé jusqu'à une autorisation explicite one-shot après gel hashé.

La cascade cible distingue les références légères, la shortlist riche et la
sortie produit : Master <= 2 000, PreRank et hydratation <= 500, sortie <= 100.
Le même ranker SIRET de retrieval doit scorer le Master et servir au
préclassement puis à l'allocation finale. Ne pas confondre ce modèle à
développer selon le contrat avec les modèles aval gelés.

Le gate PreRank@500 est >= 99,4 %. Le gate Recall@100 est >= 99,2 %, borne
basse Wilson 95 % >= 99,0 %, et chaque fold >= 98,8 %. Le contrat exige aussi
une latence chaude end-to-end p95 <= 1 000 ms et p99 <= 2 000 ms. Le Master
>= 99,5 % est une cible de marge diagnostique, pas un gate produit autonome.

Tous les labels humains exploitables restent au dénominateur, y compris
AMBIGUOUS/UNRESOLVED, fermés, absents des sources ou sans candidat. Aucun
positif injecté, identifiant SIREN/SIRET/query_id, alias CRM ou donnée
synthétique comme feature. Les identifiants officiels servent seulement aux
jointures, regroupements et sorties. Aucune règle par dossier ni choix oracle
entre deux recettes. Un sidecar officiel label-side scellé est obligatoire
pour attribuer les pertes, et ne doit jamais alimenter la génération.

Publier séparément exact et opérationnel même SIREN/même site. Un autre
SIRET au même site peut être métier-correct, notamment actif face à une vérité
fermée, mais ne remplace jamais la cible exacte ni ses gates. La proximité
géographique seule n'est pas une preuve d'équivalence. Publier ensemble les
vues historique, V2 et V3, même quand une qualification manque.

## État Git et environnement

Dépôt : `/Users/nathanjullia/Documents/Projets/SIRETO`.
Branche actuelle : `codex/v9-architecture`, pas `main`. Au dernier contrôle
elle avait 206 commits d'avance sur sa référence locale
`origin/codex/v9-architecture`, avant le commit de cette note. Cela n'est pas
une vérification live de GitHub. Aucun push n'est effectué pour cette note.

Les commits `e5c1138b` et `e61ab364` enregistrent le résultat complet et son
handover. Ne pas confondre leur présence locale avec une publication GitHub.
Il existe également des milliers d'artefacts non suivis ; ne pas faire
`git add .`, nettoyer globalement, supprimer des données ou pousser tout
l'historique sans résoudre le périmètre.

Ne pas toucher aux neuf modifications suivies préexistantes :

- `docs/synthetic_gt_corpus_contract.md`
- `scripts/__init__.py`
- `scripts/maps_assisted_smoke.py`
- `scripts/test_build_synthetic_gt_corpus.py`
- `scripts/test_maps_assisted_smoke.py`
- `scripts/v412_review_m3_business.py`
- `scripts/v412_review_m3_worker_protocol.py`
- `tests/test_v412_review_m3_business.py`
- `tests/test_v412_review_m3_worker_protocol.py`

Calcul autorisé uniquement sur le Mac M4 Pro et `/Volumes/CATNAT_DATA`, sans
GPU loué, API payante ni dépense externe. Le SSD peut être absent : vérifier
son montage avant toute conclusion ou lancement. Python local :
`/Users/nathanjullia/Documents/Projets/SIRETO/.venv/bin/python`.

## Le run complet existe et a été évalué

Population : 23 571 requêtes CRM originales avec un SIRET humain exploitable,
folds 2/3/4 = 7 767 / 7 967 / 7 837. Seize lignes sans label exploitable sont
exclues et publiées séparément, 8 / 3 / 5 selon ces folds. Ce n'est pas un
échantillon national d'entreprises ni une validation sur un CRM indépendant.

Le rejeu v33/v34 compare `v23_stable` et `v30_stable` depuis nom, adresse,
CP et INSEE CRM, sans anciens pools ni vérité en entrée. V30 ajoute le canal
conjoint nom/adresse ; modèles parent v23 gelés et succession stable v32
commune. Les calculs de génération communs sont partagés avant les deux
fusions. Lire `docs/retrieval_v33_full_replay_contract.md` et
`docs/retrieval_v34_physical_resume_contract.md` pour le protocole.

Racine :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v34_fast_resume/full_job/`.

La génération a fini le 12 septembre : 737 lots, 23 571 requêtes,
`generation/manifest.json` = `SEALED_FULL_PAIRED_GENERATION`, environ 23,1 h
d'orchestration. L'ancien `status.json` indique STOP parce que l'évaluateur
a ensuite échoué au démarrage avec `ModuleNotFoundError: No module named
'scripts'`. Ce STOP historique ne signifie pas une génération incomplète.

Le 1 octobre, les sorties ont été vérifiées intégralement : SHA256 de tous
les artefacts référencés, inputs identiques, scope exact sans doublon/omission,
candidats naturels, rangs contigus, scores finis et cap <= 2 000. Les deux
Masters représentent environ 43,3 millions de couples candidat/requête chacun,
pas autant d'établissements distincts. Le code/config et les autorités
d'inférence passent aussi leurs contrôles. Les gros segments Tantivy n'ont
pas été intégralement rehashés : leur contrôle reste manifest/meta/inventaire
stat, limite du protocole à conserver.

Seule l'évaluation a été reprise, avec l'évaluateur gelé et une invocation
en module depuis le dépôt. Résultats actuels :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v34_fast_resume/full_job/evaluation_resume_20261001/manifest.json`.
SHA256 : `b6d7123d2fb33616c76dc14ed4277552a7696d2bdb17b56f540b7f76edfec3c9`.
Statut : `MEASURED_FULL_CRM_MASTER_ONLY`. Les fichiers
`v23_stable_outcomes.parquet`, `v30_stable_outcomes.parquet`,
`official_catalog_sidecar.parquet`, `acceptable_operational.parquet` et
`label_side_manifest.json` accompagnent le manifeste.

La phase sous garde de l'évaluateur a pris 96,17 s, pic RSS ~1,12 Gio,
zéro nouveau swapout. Ce temps exclut réconciliation préalable et diagnostics.
DuckDB était borné à deux threads et 3 Go. L'évaluation est terminée ; aucun
nouveau retrieval n'a été lancé dans cette reprise.

## Mesures qui expliquent le blocage

Les trois lignes suivantes utilisent exactement les mêmes 23 571 labels.
Les points 100/500 sont des coupes de l'ordre actuel du Master, pas les
résultats d'une cascade SIRET top500/top100 nouvellement entraînée.

| Coupe du Master | Hits exacts v23 | Recall exact v23 | Hits exacts v30 | Recall exact v30 |
|---|---:|---:|---:|---:|
| 2 000 | 23 311 | 98,89695 % | 23 326 | 98,96059 % |
| 500 | 22 784 | 96,66115 % | 22 853 | 96,95388 % |
| 100 | 20 013 | 84,90518 % | 20 745 | 88,01069 % |

À 2 000, v30 gagne 23 requêtes et en perd 8 contre v23 : +15 nettes,
+0,06364 point. L'opérationnel est publié séparément : v23 98,90543 %, v30
98,97756 % à 2 000 ; à 100, v30 21 208/23 571 = 89,97497 %.

La décomposition des 2 826 échecs exacts v30 à la coupe100 est :

| Où disparaît la vérité | Requêtes |
|---|---:|
| Absente du Master@2000 | 245 |
| Présente à 2 000, évincée à la coupe500 | 473 |
| Présente à la coupe500, évincée à la coupe100 | 2 108 |
| Total absent à la coupe100 | 2 826 |

Il y a donc 2 581 vérités trouvées dans les 2 000 mais perdues à 100 : le
classement est un chantier majeur. Ce n'est pas une preuve qu'un XGBoost
les conservera toutes, ni que le modèle aval gelé est responsable de ces
coupes : il n'a pas été exécuté dans ce run.

Les 245 absences du Master se répartissent officiellement en 130 parents
non retrouvés, 3 sites non développés, 111 vérités présentes au réservoir
mais évincées des 2 000, et 1 vérité absente du catalogue. Le réservoir
naturel v30 contient 23 437 vérités, soit 99,43150 %, mais il est plus large
que le Master et n'est pas une entrée produit admissible telle quelle.

Un classement limité au Master actuel a un plafond de 98,96059 %. Il manque
au moins 57 hits pour atteindre 99,2 % et 104 pour 99,4 %, avant toute perte
aval. Le réservoir actuel ne laisse que sept hits de marge au-dessus du
minimum de 23 430 hits nécessaire à 99,4 % : la couverture aussi doit
progresser, pas seulement l'ordre de sortie.

Le manifeste conclut STOP car les gates PreRank500/Recall100/SLO n'ont pas
été mesurés. L'orientation actuelle est PIVOT_GENERATION. Aucun
READY_FOR_FOLD1, déploiement ni claim de généralisation cross-CRM.

## Ce que font réellement les modèles actuels

La dernière recette retenue pour ce retrieval est v23, implémentée dans
`src/xgb_matcher/retrieval_v23_joint_parent.py` et
`scripts/retrieval_v23_joint_parent.py` (`7ba9b699`, résultats `8e298f3c`).
Un LambdaMART XGBoost classe les sociétés parentes avec 16 features : noms
officiels actuels, meilleure preuve nom/adresse sur un site, géographie,
consensus de canaux, taille de groupe et champs manquants.

Les négatifs naturels sont 256 en tête plus 256 répartis dans la queue.
Les trois modèles externes sont entraînés sur les deux autres folds, avec
trois modèles inner-OOF supplémentaires. Aucun nom propre d'un établissement
n'est copié à ses siblings pour fabriquer une preuve de site.

Le classement SIRET du Master reste une fusion : `direct_rrf` plus, pour un
site local, `1/(60 + rang_parent_appris + rang_site_conjoint)`. Ce n'est pas
encore un LambdaMART SIRET apprenant directement à sélectionner les 100.
V30 n'a pas réentraîné les modèles parent. L'essai géographique appris
v28/v29 n'a pas été retenu après régression du Master ; ne pas le réactiver
en mélangeant arbitrairement les variantes.

Du code de cascade et d'admission SIRET v12 existe déjà :
`src/xgb_matcher/retrieval_top100_cascade_v12.py`,
`src/xgb_matcher/retrieval_hierarchical_admission_v12.py` et les modules
`retrieval_fold_fused_*_v12.py`. Les CLI correspondantes incluent
`scripts/run_retrieval_hierarchical_admission_v12.py` et
`scripts/run_retrieval_fold_fused_sharded_oof_v12.py`. Ce prototype possède
une chaîne nested OOF et 319 features, avec des tests et benchmarks physiques
historiques. Il n'a pas été exécuté comme admission sur les Masters v23/v30
de ce rejeu. Ne pas affirmer qu'aucun code de ranker SIRET n'existe, ni lancer
ses anciens runners sur de nouveaux artefacts sans contrôler leurs hashes,
leurs snapshots, leurs schémas et leurs autorisations. Commencer par évaluer
ce qui peut être réutilisé plutôt que réimplémenter toute la cascade.

L'ancien moteur aval two-stage `src/xgb_matcher/infer.py` score tous les
candidats au ranker sans sémantique, puis ses 50 meilleurs au decider avec
features sémantiques (`stage1_top_n=50` par défaut, configurable selon le
profil). Le seuil historique 0,835 est celui du risk model AUTO/REVIEW,
pas un seuil de nombre de candidats ni un seuil universel du decider.
Ce moteur n'est pas la cascade retrieval v12 et n'a pas été benchmarké ici.

Avant le gel, les expériences aval V4.12 avaient construit BUSINESS_LEARNED
avec 129 features contre une baseline45, puis testé scores BGE et synthétique.
Ces derniers essais ont terminé par STOP/PIVOT sans promotion de la baseline.
Ne pas importer leurs métriques top1, leur sélection historique fold0 ni leurs
labels qualifiés dans l'évaluation retrieval all-human actuelle. Ranker aval,
decider, risk model et accepteur restent gelés jusqu'au gate retrieval.

## Pistes proposées et autorité nécessaire

L'utilisateur veut essayer un classement XGBoost des 2 000 pour garder les
meilleurs avant le decider. C'est cohérent avec la cascade cible, mais aucun
nouveau fit SIRET ni nouveau test de cette idée n'a été lancé. Cette note est
un transfert de contexte, pas une instruction à exécuter un long run.

Pour préparer la prochaine étape, partir des résultats scellés et séparer :

1. Classement dans le Master : concevoir une expérience SIRET OOF avec des
   features disponibles en production, sans retuning sur les outer résultats.
   Utiliser les 2 581 vérités perdues comme audit de familles, jamais comme
   table d'exceptions ou signal d'inférence. Distinguer manque de preuve,
   sites frères, rang parent, géographie incertaine et concurrence locale.
2. Sélection vers le Master : expliquer les 111 évictions du réservoir avant
   de proposer une correction générale. Tout nouveau scorer et ordre de
   sélection doit respecter le plafond de 2 000 références du contrat ; ne
   pas scorer un réservoir illimité puis appeler cela une cascade conforme.
3. Découverte : attribuer les 130 parents absents aux mécanismes de recherche
   et aux preuves officielles réellement disponibles. Une meilleure admission
   à partir du Master ne peut pas résoudre ces absences.

Réutiliser les candidats enregistrés pour une expérience de classement ne
revient pas à injecter la vérité ni à utiliser un cache candidat en production.
Cela évite de refaire une génération inchangée. La validation produit doit
ensuite porter sur une recette unique gelée, reconstruite comme en production.
Préenregistrer les changements et préserver les modèles/sources historiques.

La géographie intéresse explicitement l'utilisateur : coordonnées X/Y, BAN
et sa qualité/incertitude, tous les sites officiels d'un SIREN, rues adjacentes
et sites réels différents du label commercial. Ce sont des hypothèses à
évaluer ; une BAN peu fiable doit rester un manque/incertitude, pas une
distance certaine. Ne pas limiter toute la génération à un rayon dur ni
remplacer la vérité exacte par un site proche.

## Attentes de coût et livrables

L'utilisateur perd patience devant les longs runs sur son Mac. Éliminer les
idées non viables par des contrôles courts et des expériences hors ligne
bornées. Une validation exhaustive peut être nécessaire, mais pas à chaque
petit changement. Ne pas annoncer un gain de vitesse 20x parce qu'il y a 20x
moins de candidats : features, accès aux sources, décodage et sémantique ont
des coûts distincts ; le decider historique reste à 50 dans les deux cas.

Avant un nouveau calcul lourd, proposer son hypothèse, son protocole OOF,
les ressources et un critère d'arrêt crédible. Ne pas considérer cette note
comme une autorisation de calcul payant, d'API externe, de délégation ou
d'ouverture de fold1. Expliquer les résultats en français simple.

Livrables attendus du prochain cycle autorisé : preuve d'intégrité, métriques
exactes et opérationnelles, toutes vues/segments, gains ET régressions,
attribution exhaustive des pertes, coût réel, limite de généralisation et
verdict explicite. Chaque milestone doit avoir son commit isolé référencé
dans `handover.md`, en préservant les autres changements locaux.

## Sources principales

- `docs/retrieval_v34_full_crm_evaluation_results.md` : reprise et mesures.
- `docs/retrieval_v23_joint_parent_results.md` : modèle parent et fusion.
- `docs/retrieval_v30_joint_search_results.md` : canal ajouté et premier
  résultat sur le panel historique, à distinguer du nouveau rejeu complet.
- `handover.md`, dernières entrées v23 à v34 : essais et rejets historiques.
- Le manifeste d'évaluation CATNAT cité plus haut : résultats numériques,
  segments, courbes, sidecar officiel et références hashées aux artefacts.
