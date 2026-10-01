# V34 — évaluation du rejeu complet CRM

1 octobre 2026. Évaluation terminée, génération conservée et inchangée.
Comparaison Master@2000 uniquement : aucun gate produit top500/top100 franchi.

## Reprise sans recalcul

Le rejeu du 11–12 septembre a terminé la génération : 737 lots, 23 571
requêtes folds 2/3/4, statut `SEALED_FULL_PAIRED_GENERATION`, 83 148,47 s
d'orchestration. Son étape d'évaluation a échoué avant tout scoring avec
`ModuleNotFoundError: No module named 'scripts'`.

À la demande utilisateur « vas-y lance alors », seule l'invocation de
l'évaluateur a été reprise depuis la racine du dépôt avec `python -u -m
scripts.evaluate_retrieval_v33_full_replay`. Aucun code, modèle, seuil,
source ou candidat n'a été modifié. Les preuves du premier arrêt restent
intactes, y compris son `status.json` STOP et son `evaluation.log`.

Le résultat actuel est dans une racine distincte :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v34_fast_resume/full_job/evaluation_resume_20261001/`.
Son `manifest.json` porte `MEASURED_FULL_CRM_MASTER_ONLY`, SHA256 :
`b6d7123d2fb33616c76dc14ed4277552a7696d2bdb17b56f540b7f76edfec3c9`.

## Intégrité et autorisations

- Tous les fichiers de sortie référencés ont été vérifiés par SHA256 ;
  couverture exacte des 23 571 entrées, sans doublon, ajout ni omission.
- Identité des inputs CRM, code/config gelés, évaluateur et politique
  épinglés au lancement ; contrôles stat/inode/taille/mtime des sources et
  inventaires des index conformes. Pas de nouveau hash intégral des segments
  Tantivy nationaux : limite du protocole conservée.
- Les deux Masters ont été contrôlés intégralement : candidats naturels,
  aucune requête étrangère, aucun doublon SIRET, rangs contigus, identifiants
  cohérents, scores finis et plafond absolu de 2 000 respecté.
- 43 334 072 lignes candidates v23 et 43 334 093 v30. Chaque version couvre
  les 23 571 requêtes. Ce sont des références candidat/requête, pas autant
  d'établissements distincts.
- Labels ouverts uniquement après réconciliation de la génération scellée.
  Sidecar officiel catalogue et équivalences opérationnelles scellés avant
  scoring ; empreintes et comptages des sorties d'évaluation revérifiés.
- Aucun fold 0/1 ouvert, aucun entraînement, aucun positif injecté. Les
  16 labels absents/inexploitables sont publiés séparément (folds 2/3/4 :
  8/3/5). Aucun statut de qualification ne filtre les labels exploitables.

## Résultats all-human

| Mesure sur 23 571 requêtes | v23 stable | v30 stable |
|---|---:|---:|
| SIRET humain exact dans Master@2000 | 23 311 / 98,89695 % | 23 326 / 98,96059 % |
| Échecs exacts end-to-end | 260 | 245 |
| Couverture opérationnelle Master@2000 | 23 313 / 98,90543 % | 23 330 / 98,97756 % |
| SIRET exact dans le réservoir naturel avant Master | 23 435 / 99,42302 % | 23 437 / 99,43150 % |

Comparaison appariée exacte : 23 requêtes gagnées, 8 perdues, soit +15 nettes
et +0,06364 point de pourcentage. Opérationnelle : 23 gagnées, 6 perdues,
+17 nettes. Ces gains sur ce CRM ne prouvent pas une généralisation cross-CRM.

| Attribution exhaustive des échecs exacts | v23 | v30 |
|---|---:|---:|
| Vérité absente du catalogue officiel | 1 | 1 |
| Parent non retrouvé | 132 | 130 |
| Site non développé | 3 | 3 |
| Site présent dans le réservoir, évincé du Master@2000 | 124 | 111 |
| Total | 260 | 245 |

| Fold de développement | Requêtes | Hits exacts v23 | Hits exacts v30 |
|---|---:|---:|---:|
| 2 | 7 767 | 7 698 / 99,11163 % | 7 699 / 99,12450 % |
| 3 | 7 967 | 7 871 / 98,79503 % | 7 881 / 98,92055 % |
| 4 | 7 837 | 7 742 / 98,78780 % | 7 746 / 98,83884 % |

Les vues historique, V2 et V3, états de vérité, origines de données et autres
segments sont publiées ensemble dans le manifeste. Exact et opérationnel
restent distincts. Les courbes de rang 100/500 de ce Master ne sont pas les
métriques de la cascade apprise top500/top100 du contrat v12.

## Coût et décision

La phase sous garde ressources de l'évaluateur a pris 96,17 s, avec un pic
RSS de 1 206 026 240 octets (~1,12 Gio), aucun nouveau swapout et
`pass_resource_policy=true`. Ce temps exclut la réconciliation préalable et
les contrôles diagnostiques ; aucune nouvelle génération n'a été lancée.
DuckDB reste borné à deux threads et 3 Go pour l'évaluation.

Le verdict de l'artefact est `STOP` / absence de gates PreRank500,
Recall100 et SLO. Orientation de travail : **PIVOT_GENERATION**. Le plafond
Master v30 à 98,96059 % ne permet pas d'atteindre 99,2 % exact en sortie100,
ni 99,4 % au préclassement500, par simple classement de ses candidats.
Il manque au minimum 57 hits pour 99,2 % et 104 pour 99,4 %, avant toute
perte aval. Les 111 évictions du réservoir et les 130 parents absents sont
les deux pertes principales à investiguer ; ne pas les convertir en règles
par identifiant ou en injection de positifs.

Pas de `READY_FOR_FOLD1`, pas d'autorisation de fold1 et pas de déploiement.
