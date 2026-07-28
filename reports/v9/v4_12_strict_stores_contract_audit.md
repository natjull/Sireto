# V4.12 — Audit du contrat et du code des stores stricts

## Verdict

`GO_CODE_V412_STRICT_STORES`

Ce verdict autorise le commit isolé de l'implémentation, puis la création et
le contre-audit du verrou d'exécution. Il n'autorise ni le build réel, ni les
modèles, ni l'ouverture de l'oracle par le worker.

Versions auditées :

- contrat :
  `d33d66ac50bf21be62f999d25b5ce0b8af7f5735a9b67f5838728d8b5882739e` ;
- plan :
  `4d449f70c2b5b7eba7bd56322c7248ff4b52930eca1a8c87250f4fb3f51a1f5f` ;
- profil sandbox :
  `d9195c35e78f11eadafd883acbd53996ab531dbc2e998d1efb21179f2556be77` ;
- certificateur :
  `dbcc653b99f1ffc63b46f2d3b65eb74f87103f6b72e00256bcfe4302b48f1918` ;
- stores :
  `6d6064b4b76df141b961be3ae8ff5512e8def08a33790194456d5c6b1132294b` ;
- tests :
  `b89e38a73b7b12d97cff2181d8316e581f1922c2543ea8a7d04c86ddd63d7093`.

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
11. rouvraient des manifestes JSON par pathname après vérification ;
12. ne rendaient pas la publication et sa reprise transactionnelles ;
13. utilisaient Git via un `PATH` hérité ;
14. rouvraient le code, le run-spec et le descripteur lookup par pathname ;
15. acceptaient des IDF infinies ou des indices sparse hors bornes ;
16. ne pouvaient pas nettoyer un runtime privé rendu non modifiable ;
17. présentaient à tort tout le runtime Homebrew comme scellé ;
18. épinglaient un hash Python framework tronqué à 63 caractères ;
19. utilisaient `Path.cwd()` sous une politique n'autorisant que les
    métadonnées de `RUN_ROOT`.

Le premier verrou dérivé, de hash
`76f00570941920dead5bc4ac966c6d6a23ec317a35ecf25172fc04bdc17ba5e3`,
a été révoqué avant exécution : 96 des 97 assertions indépendantes passaient,
mais le contrôle du fichier runtime réel a détecté le caractère manquant.
Le lock a été supprimé, les trois références ont été corrigées, puis un
contre-audit dédié `GO_CODE_PATCH` a validé 20 assertions sur 20.

Le premier lancement complet autorisé s'est ensuite arrêté avant
publication avec `EPERM`. Le smoke initial ne reproduisait pas la résolution
du cwd. Deux PoC sandboxés ont établi que `getcwd()` est interdit, tandis que
`lstat(".")` et `lstat(RUN_ROOT)` réussissent et désignent le même
`st_dev/st_ino`. Le verrou a de nouveau été révoqué. Le patch
`GO_CODE_CWD_PATCH` conserve la preuve du cwd par identité de répertoire,
fixe `JOBLIB_MULTIPROCESSING=0` et n'ajoute aucun droit sandbox.

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

Les hashes de `/usr/bin/sandbox-exec`, de `/usr/bin/git`, de
`Python.app` et de la bibliothèque `Python` du framework ont été redérivés
et correspondent au plan.

## Implémentation auditée

- `StrictPartitionStore` charge uniquement les partitions autorisées, vérifie
  schéma, taille, hash et alignement exact avec le cache historique ;
- `StrictVerifiedTfidfCache` vérifie sidecar, pickle, classes exactes,
  dimensions, IDF finies, vocabulaire bijectif et indices sparse ;
- `StrictSnapshotLookup` ouvre DuckDB via un descripteur `O_NOFOLLOW`, en
  lecture seule, refuse WAL/tmp et impose un budget de 0 à 100 SIRET ;
- source, run-spec et descripteur sont consommés via des descripteurs ancrés ;
- le profil effectif est transmis directement par `sandbox-exec -p` ;
- la publication est atomique, rescellée et récupérable uniquement si elle
  correspond encore aux entrées courantes ;
- le ledger parent couvre exactement les 1 954 fichiers attendus.

## Tests finaux

- `44 passed` sur `tests/test_v412_strict_stores.py` ;
- smoke sandbox macOS réel : `SMOKE_OK` ;
- `714 passed` sur la suite complète du dépôt ;
- AST Python valide et `git diff --check` propre.

## Frontière de confiance

Les données, le code du worker et ses trois contrôles sont protégés contre
une réouverture mutable entre vérification et consommation. `/System`,
`/usr` et `/opt/homebrew` restent le runtime local de confiance, dont les
versions et exécutables structurants sont enregistrés. La preuve ne prétend
pas résister à un attaquant concurrent disposant du même compte macOS ou des
droits administrateur. Cette limite est explicite et proportionnée au risque
scientifique du Gate A.

## Étape autorisée

Committer l'implémentation auditée, mettre à jour `handover.md`, puis créer un
verrou d'exécution séparé et le contre-auditer. Aucun build complet Gate A ne
doit être lancé avant le `GO_LOCK`.
