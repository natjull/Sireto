# V4.12 — Audit du contrat des stores stricts et de la sandbox

## Verdict

`GO_CODE_V412_STRICT_STORES`

Ce verdict autorise l'implémentation. Il n'autorise ni le build réel, ni les
modèles, ni l'ouverture de l'oracle.

Versions auditées :

- contrat :
  `db9deb008618517ff86cf9c5ee9852ddc3885236e8e71f10ee7d294481e04002` ;
- plan :
  `a73e4b6d66a533c6d0ad250dcd71aca0bf72f57b15d2e6daf0b397ed3cf3ea14`.

## Refus successifs

Les revues ont refusé les versions qui :

1. rendaient le build dépendant d'un chemin temporaire aléatoire ;
2. ne fermaient pas les keysets des preuves et du verrou ;
3. transmettaient indirectement un manifeste historique au worker ;
4. confondaient clé de routage et valeur Hive ;
5. supposaient à tort que les caches TF-IDF étaient alignés sur les
   partitions brutes ;
6. autorisaient trop largement les lectures ou écritures dans la racine
   privée ;
7. attendaient `EACCES` alors que `sandbox-exec` renvoie `EPERM` sur ce Mac ;
8. n'épinglaient pas les deux exécutables Python Framework ;
9. laissaient les fonctions historiques de construction des noms hors de
   l'identité du build ;
10. ne donnaient pas les droits metadata nécessaires au `lstat` exhaustif.

## Preuves factuelles

- 1 456 requêtes dev, 1 449 routées par INSEE et 7 par code postal ;
- 648 clés distinctes, aucune partition absente ;
- 648 partitions, 449 454 881 octets et 8 030 285 rows physiques ;
- 648 pickles et 648 sidecars, 4 042 655 632 octets ;
- transformation exacte vers 4 764 472 rows filtrées et dédupliquées ;
- égalité exacte des 4 764 472 chaînes `names` ;
- dimensions et vocabulaires cohérents pour les 648 tuples TF-IDF ;
- lookup DuckDB de 42 322 035 rows, schéma et index unique conformes ;
- liste blanche enfant de 1 945 fichiers ;
- ledger parent exhaustif de 1 954 fichiers.

## Sandbox vérifiée sur le Mac

Un probe indépendant hors données projet a confirmé :

- lecture des fichiers littéralement autorisés ;
- écriture limitée aux répertoires privés `output` et `tmp` ;
- refus de lecture et d'écriture hors liste avec `EPERM`, errno 1 ;
- refus du réseau loopback et externe ;
- refus de `fork` ;
- import de Python 3.14.3, PyArrow et DuckDB ;
- fonctionnement des paramètres séparés `RUN_SPEC`, `RUN_OUTPUT` et
  `RUN_TMP`.

Les hashes de `/usr/bin/sandbox-exec`, du launcher Python Framework et du
second exécutable `Python.app` ont été redérivés et correspondent au plan.

## Portée

Le futur worker devra réimplémenter en privé la préparation du pool et la
construction des noms, sans importer les modules historiques. La parité
exhaustive avec les chaînes du cache reste un critère fail-closed. Le verrou
d'exécution ne pourra être créé qu'après implémentation, tests et nouvel audit
du code.
