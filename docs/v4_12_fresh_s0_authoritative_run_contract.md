# Contrat autoritatif V4.12 S0 — exécution synthétique sous sandbox

## 1. Statut et portée

Ce document préenregistre l’unique trajectoire autorisée pour transformer le
core synthétique V4.12 en exécution S0 auditable.

Statut obligatoire :

`PREREGISTERED_DO_NOT_IMPLEMENT_UNTIL_AUDIT`

Ce statut n’autorise ni implémentation, ni construction de fixture, ni
scellement de lock, ni lancement. Une revue indépendante doit d’abord conclure
que le présent contrat et son plan canonique ferment correctement l’autorité.

S0 porte exclusivement sur les six lignes synthétiques déjà préenregistrées.
Il ne mesure aucune performance de matching et ne peut produire ni
`QUALIFIED`, ni `READY`, ni `GO` pour un traitement CRM réel.

Le core de référence est le commit Git exact :

`38287c18ae1caa73e062bde0506a6a665e8379ab`

Les artefacts core byte-for-byte à cette révision sont :

| Rôle | Chemin | SHA-256 |
|---|---|---|
| Producteur de fixture | `scripts/build_v412_fresh_intake_synthetic_fixture.py` | `d359994c9231b761c4cede45371f6202cc6471f52f524e1265b5cfe7e7f4f32f` |
| Scanner/sealer core | `scripts/run_v412_fresh_intake_synthetic_scanner_sealer.py` | `1902ccd08e513b4054ebc85635609fbb73ada111b113e38dbaf716b478c6280d` |
| Tests core | `tests/test_v412_fresh_intake_synthetic_scanner_sealer.py` | `b43309bbccbc37fced14c1b731956bad372c35c09d243b23df1d8efb9a6f72e1` |
| Plan core | `config/v4_12_fresh_intake_synthetic_scanner_sealer_plan.json` | `e8a55a999035183363c0bf7711280b09553a305434173286e41c696ea3e4772f` |
| Contrat core | `docs/v4_12_fresh_intake_synthetic_scanner_sealer_contract.md` | `ad8eed1bf5d8d8a280ea8b212d3d308eb5c8b048efb3ebefb567956b3eb60ca8` |

Toute divergence d’un de ces octets, même sans changement fonctionnel apparent,
invalide le run. Un nouveau préenregistrement est alors nécessaire.

## 2. Séquence fermée

L’unique séquence autorisée est :

```text
commit d’implémentation
  -> fixture
  -> execution lock immuable
  -> commit du launch authorization manifest
  -> launcher fixe
  -> lease flock + claim pré-spawn immuable
  -> worker distinct par FDs
  -> launch receipt immuable
```

Aucune étape ne peut être fusionnée avec la suivante :

1. un commit d’implémentation ferme launcher, sealer, worker, profil, tests,
   présent plan et présent contrat ;
2. le producteur fabrique uniquement la fixture et son control manifest ;
3. le lock sealer vérifie puis ferme l’autorité sans lancer le scanner ;
4. un commit ultérieur ajoute l’unique launch authorization manifest ;
5. le launcher sans argument vérifie cette autorisation, prend le lease,
   résout sous ce lease l'état claim/receipt, crée si autorisé le claim et
   lance un worker unique ;
6. le worker scanne les FDs transmis sans lancer de processus ;
7. le launcher revalide l’autorité et publie le launch receipt terminal.

Le lock sealer n’importe pas le launcher. Le launcher n’appelle pas le
producteur. Le worker n’appelle ni le producteur, ni le lock sealer, ni
`sandbox-exec`.

Le scanner core
`scripts/run_v412_fresh_intake_synthetic_scanner_sealer.py`, hash
`1902ccd08e513b4054ebc85635609fbb73ada111b113e38dbaf716b478c6280d`,
reste strictement immuable. Le worker futur distinct est
`scripts/run_v412_fresh_s0_worker.py`. Il importe ou adapte ses helpers fermés,
mais aucune modification du core n’est autorisée. Le lock épingle séparément
le core et le worker.

## 3. Aucun vrai CRM

La racine autorisée est exclusivement :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake_synthetic`

Le run ne reçoit que la fixture synthétique fermée par le plan core. Sont
interdits, y compris en lecture, métadonnées et énumération :

- tout CRM réel ou extrait dérivé ;
- `data`, `models`, `reports`, `challenges` et `registries` ;
- `final_holdout_inputs` et toute racine réelle `fresh_holdout_intake` ;
- les ledgers de qualification et d’évaluation ;
- les répertoires parents permettant de découvrir ces objets.

Les tests utilisent des canaris synthétiques portant ces noms et des contenus
inoffensifs. Aucun test ne crée, ne lit ou ne sonde un fichier réel pour
démontrer un refus.

Un canari de fichier est réussi seulement si l’ouverture par chemin et
l’énumération de son parent synthétique échouent depuis le worker. Le canari
réseau vise uniquement `127.0.0.1:9` et ne transporte aucune donnée. Keychain
et fork ne sont jamais appelés dynamiquement : ils sont prouvés par audit
statique du worker et règle explicite du profil. L’absence d’un vrai fichier
n’est pas une preuve d’isolation.

Le lock sealer crée avant le lock, avec `O_EXCL`, tous les fichiers et
répertoires canaris synthétiques, ainsi qu'un manifest canonique qui ferme leur
existence, identité, contenu ou absence attendue. Ce manifest est un input
hashé du lock et un FD retenu par le parent avant/après le worker.

## 4. Lock immuable et lease distinct

### 4.1 Execution lock

L’execution lock est un document JSON canonique immuable. Il constitue
l’autorité complète du run. Il est créé une seule fois avec `O_EXCL`,
`O_NOFOLLOW`, `umask 0077`, synchronisé par `fsync` et `F_FULLFSYNC`, puis son
répertoire parent est synchronisé.

Son SHA-256 est externe au document et est porté par le launch authorization
manifest. Le lock ne se hash pas lui-même et le launcher n’accepte aucun hash
ou chemin de lock par CLI.

Le parseur refuse :

- toute clé absente ou supplémentaire ;
- tout doublon JSON ;
- NaN, Infinity, coercition ou type implicite ;
- tout JSON non canonique ;
- tout chemin relatif, composant vide, `.` ou `..` ;
- toute valeur de hash autre que 64 hexadécimaux minuscules.

Le schéma exact du lock est défini dans le plan autoritatif. Il contient un
`implementation_commit` concret et les hashes des blobs de ce commit pour le
lock sealer, le launcher, le worker, le profil, les tests, le présent plan et
le présent contrat. Il épingle aussi le core immuable.

Le runtime privé est fermé par un inventaire ordonné sans hash récursif. Chaque
record porte `role`, `source_path`, `private_relative_path`, `size_bytes`,
`sha256` et `mode`. Le hash global porte sur la projection canonique des
records, qui ne contient pas ce hash global. L'inventaire inclut l’exécutable
Python, sa bibliothèque framework, la stdlib utilisée, PyArrow et les
bibliothèques Mach-O non système atteignables, le worker, le scanner core, le
fixture builder et leurs plan/contrat épinglés. Le layout conserve les chemins
absolus applicatifs sous un `rootfs` privé ; `DYLD_ROOT_PATH` redirige les
install names absolus et l'algorithme du plan résout `@rpath`,
`@loader_path`, `@executable_path` et `LC_RPATH`. Les racines, l'ordre, le
traitement des liens et les exclusions système sont fermés dans le plan. Une
dépendance absente ou supplémentaire produit `STOP`.

Une sentinelle `UNIMPLEMENTED` empêche le passage au run ; elle n’est jamais
une valeur valide dans un lock exécutable.

### 4.2 Lease `flock`

Le lease est un fichier opérationnel séparé du lock. Le launcher l’ouvre avec
`O_CREAT|O_RDWR|O_NOFOLLOW|O_CLOEXEC`, valide qu’il est régulier, mono-lien,
du bon UID et du bon volume, puis prend un `flock(LOCK_EX|LOCK_NB)` conservé
jusqu’après publication du receipt.

Le lease :

- ne complète pas l’autorité ;
- n’est pas inclus dans le hash du lock ;
- peut être réutilisé après libération ;
- ne permet jamais de modifier le lock ;
- empêche seulement deux launchers de traiter le même `attempt_id`.

Un lock immuable et un lease mutable ne peuvent jamais être le même inode ou
le même chemin.

### 4.3 Launch authorization manifest

L’ancre de confiance du launcher est exclusivement le fichier futur :

`config/v4_12_fresh_s0_launch_authorization.json`

Il est ajouté par un commit Git postérieur à la création du lock. Son schéma
exact est :

```text
schema_version
implementation_commit
authoritative_plan_sha256
authoritative_contract_sha256
execution_lock_absolute_path
execution_lock_sha256
synthetic_run_id
attempt_id
authorization_status
```

`authorization_status` vaut exactement `AUTHORIZED_SYNTHETIC_S0`. Le launcher
utilise ce chemin constant, sans argument. Il vérifie que les octets du
manifest sont exactement le blob du `HEAD` d’autorisation, que le HEAD ne
contient pas de modification des sources épinglées depuis
`implementation_commit`, puis que ces sources égalent les blobs de ce commit.
Un working tree sale sur un chemin fermé produit `STOP`.

### 4.4 Claim pré-spawn

Après acquisition du lease et avant tout spawn, le launcher crée avec `O_EXCL`
un claim immuable distinct du lock, du lease et du receipt :

```text
audit/<synthetic_run_id>/parent/claims/<attempt_id>.json
```

Champs exacts :

```text
schema_version
implementation_commit
authorization_manifest_sha256
execution_lock_sha256
synthetic_run_id
attempt_id
claimed_at_utc
claim_status
```

`claim_status` vaut `CLAIMED_PRE_SPAWN`. `claimed_at_utc` est audit-only. Si le
claim existe, seul un receipt terminal complet et byte-valid permet une lecture
idempotente. Un claim sans receipt complet produit
`STOP_NO_RERUN`; aucun nouvel `attempt_id` n’est inventé.

## 5. Identités et ouverture

Tous les chemins contrôlés sont parcourus depuis un FD de `/` par composants
avec `openat`, `O_DIRECTORY`, `O_NOFOLLOW` et `O_CLOEXEC`.

Chaque fichier d’autorité ou payload doit être :

- régulier ;
- mono-lien ;
- du `uid` épinglé ;
- sur le device et l’UUID de volume épinglés ;
- non accessible en écriture aux groupes et autres ;
- identique en device, inode, taille, `mtime_ns`, `ctime_ns`, mode et uid
  pendant sa période d’autorité.

Le launcher ouvre et conserve avant le spawn les FDs du lock, du control
manifest, des cinq payloads, du worker privé, du profil effectif et des
artefacts nécessaires à la preuve. Les payloads du worker sont transmis comme
FDs, jamais redécouverts par parcours.

Le parent calcule sur ses FDs une observation complète avant le spawn, puis
une seconde après la terminaison du worker. Pour chaque objet épinglé, les
deux lectures complètes jusqu’à EOF et les deux `fstat` doivent concorder avec
le lock. Toute dérive force le verdict terminal `STOP`.

## 6. Launcher unique

Pendant le lancement autoritatif, un seul fichier source futur est autorisé à
lancer un processus :

`scripts/launch_v412_fresh_intake_synthetic_scanner_sealer.py`

Son interface publique ne comporte aucun argument. Le chemin du launch
authorization manifest est une constante source exacte. Les racines, inputs,
outputs, temps logiques, identifiants, lock et politiques ne sont remplaçables
ni par CLI, ni par environnement.

Le launcher :

1. vérifie le blob du manifest d’autorisation au HEAD ;
2. vérifie le lock et son hash depuis cette autorisation ;
3. vérifie l’implementation commit et tous ses blobs ;
4. vérifie runtime, uid, device, UUID et chemins ;
5. ouvre et conserve les FDs ;
6. prend le lease puis traite exactement les états claim/receipt du plan ;
7. crée le claim pré-spawn seulement sous le lease ;
8. prépare l’arbre privé sur le même volume depuis le runtime manifest ;
9. revalide toutes les copies contre les FDs sources ;
10. lance un seul worker avec une table `pass_fds` fermée ;
11. attend le worker et collecte son résultat de contrôle ;
12. rehash les FDs parents ;
13. valide arbres terminaux, seals et tête du journal ;
14. publie le launch receipt immuable ;
15. libère le lease.

L’environnement enfant est fermé : `PATH` minimal constant,
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONHOME`, `PYTHONPATH`, `TMPDIR`,
`DYLD_ROOT_PATH`, `DYLD_LIBRARY_PATH` et `DYLD_FRAMEWORK_PATH` sont tous
dérivés du layout privé fermé. Toutes les autres variables sont supprimées,
notamment les proxies, variables Python utilisateur et secrets.

## 7. Worker par FDs

Le worker futur distinct reçoit l’interface interne :

```text
--worker-spec-fd <n>
--worker-control-fd <n>
```

Ce n’est pas une interface publique. Une invocation sans spec FD canonique,
sans canal de contrôle authentifié par le parent, ou avec un FD absent de la
table fermée produit `STOP`.

Le worker :

- utilise le core immuable uniquement comme bibliothèque de helpers fermés ;
- ne résout aucun input par chemin ;
- n’énumère aucun parent ;
- n’importe ni n’utilise `subprocess` ;
- n’ouvre ni réseau, ni Keychain ;
- ne fork et n’exécute aucun programme ;
- écrit uniquement via les FDs/racines exactes du run ;
- conserve les cinq FDs payload pendant les deux observations ;
- effectue réellement au moins 60 secondes monotones dans le même processus ;
- n’accepte aucun paramètre réduisant cette durée.

L’injection d’horloge ou d’attente reste limitée aux tests pytest et ne peut
être activée par le spec de production, une variable d’environnement ou un
flag public.

## 8. Sandbox et sous-arbres exacts

Le profil effectif est dérivé avant le lock depuis un template épinglé, par
remplacement ordonné de placeholders fermés et échappement Seatbelt défini
dans le plan. Tout placeholder inconnu ou résiduel produit `STOP`; le hash des
octets effectifs est épinglé par le runtime manifest et le lock.

Le profil est `deny default`. Il n’autorise en écriture que les sous-arbres
exacts du run :

```text
sealed/<synthetic_run_id>
scan/<synthetic_run_id>
quarantine/<synthetic_run_id>
audit/<synthetic_run_id>/worker
tmp/<synthetic_run_id>
```

`inbox` et `control` sont lecture seule par FDs retenus. La racine synthétique
complète ne reçoit jamais un `subpath` de lecture ou d’écriture.

Le worker n'a aucun droit d'écriture sur
`audit/<synthetic_run_id>/parent`. Claims, leases, observations d'autorité et
launch receipts vivent exclusivement dans ce sous-arbre parent. Les receipts
et événements produits par le core vivent exclusivement dans
`audit/<synthetic_run_id>/worker`.

Le runtime privé vit sous `runtime/<synthetic_run_id>`, hors de tous les
sous-arbres inscriptibles par le worker. Le profil peut autoriser son `subpath`
en lecture/exécution seulement parce que le parent a vérifié juste avant spawn
que cet arbre mono-lien correspond exactement au manifest, sans fichier
manquant ni supplémentaire.

Pour les ancêtres nécessaires au cheminement, seule la métadonnée du composant
exact peut être permise. La lecture du contenu d’un répertoire parent et son
énumération restent refusées.

Le profil refuse explicitement :

- réseau entrant et sortant ;
- fork et création de nouveaux processus ;
- accès Keychain et services d’identité ;
- écritures hors des cinq sous-arbres ;
- lecture/énumération des canaris interdits ;
- accès au dépôt de travail depuis le worker.

Le worker utilise une copie privée du code épinglé. Aucun import ne vient du
working tree.

## 9. Frontière runtime système macOS

La frontière applicative et la frontière runtime système sont distinctes.

La frontière applicative est fermée par hashes, FDs et profil : lock, worker,
profil effectif, control manifest, fixture et outputs exacts.

Sur macOS, `dyld`, Python et `sandbox-exec` dépendent de composants protégés et
de caches système que l’application ne peut pas raisonnablement copier,
énumérer et hasher exhaustivement. Le système courant impose donc une frontière
runtime en lecture seule :

- `/System`;
- `/usr/lib`;
- les caches dyld utilisés par le système ;
- `/dev/null`, `/dev/urandom` et `/dev/fd`;
- les lectures `sysctl` strictement requises au runtime.

Cette frontière est du TCB système, pas une entrée métier. Le lock épingle au
minimum architecture, version et build macOS, Python, sa bibliothèque
framework effectivement chargée, et `/usr/bin/sandbox-exec`. Les identités de
volume du dépôt, du run externe et du TCB système sont distinctes et toutes
explicites. Tout changement invalide le lock.

Le profil ne doit pas étendre cette frontière à `/Users`, `/Volumes`, `/opt`,
au dépôt ou à des gestionnaires de paquets. Une dépendance applicative non
système doit être copiée et hashée dans l’arbre privé.

## 10. Impossibilités macOS et compensations obligatoires

Les affirmations suivantes sont interdites car macOS ne permet pas de les
garantir comme formulées :

1. **Exécuter `sandbox-exec` depuis une copie privée.** Une copie peut perdre
   les propriétés attendues par AMFI. Compensation : exécuter le chemin
   canonique root-owned `/usr/bin/sandbox-exec`, non inscriptible, après
   vérification de son hash et de son identité.
2. **FD-pinner tout `dyld` et le runtime système.** Les shared caches et
   composants SIP ne forment pas une fermeture applicative stable.
   Compensation : frontière TCB système explicite, lecture seule, version/build
   épinglés et invalidation au moindre changement.
3. **Interdire tout `exec` dans le profil tout en démarrant Python sous ce même
   profil.** Le démarrage nécessite `process-exec` pour le Python privé.
   Compensation : autoriser uniquement cet exécutable, interdire le fork,
   fermer les imports, vérifier statiquement l’absence d’API d’exécution dans
   le worker et considérer qu’un `exec` sans fork vers ce même binaire ne peut
   pas être prouvé impossible par Seatbelt seul.
4. **Rendre un lease `flock` durable ou autoritatif.** `flock` est consultatif
   et disparaît avec le processus. Compensation : le lock immuable porte toute
   l’autorité ; le lease ne traite que la concurrence.
5. **Prouver l’absence d’énumération par une simple denylist.** Seatbelt et les
   permissions POSIX doivent être testés depuis le processus réellement
   sandboxé. Compensation : allowlist exacte, permissions de métadonnées
   minimales et canaris synthétiques d’ouverture et d’énumération.
6. **Publier un receipt si le parent est tué après le worker mais avant la
   publication.** Il existe une fenêtre de crash. Compensation : l’absence de
   receipt avec claim existant vaut `STOP_NO_RERUN`. La reprise ne relance
   jamais le worker et n’invente jamais un nouvel `attempt_id`.

Ces limites ne sont pas des exceptions silencieuses. Elles font partie du
verdict et doivent apparaître dans le launch receipt.

## 11. Protocoles FD et schémas fermés

Le plan ferme les clés, types, nullabilité et enums des objets suivants :

- chaque record de payload FD : `role`, `fd_number`, `identity`,
  `size_bytes`, `sha256`, `access`;
- `runtime.system` et chaque record du `private_runtime_manifest` ;
- chaque observation parent avant/après : rôle, identité complète, taille,
  hash, position restaurée et résultat EOF ;
- worker spec : identifiants, temps logique, minimum 60 secondes, table exacte
  des rôles FD, racines de sortie et hashes d’autorité ;
- worker spec : FD séparé, absent de sa propre table de payloads et donc sans
  auto-référence ;
- canal de contrôle : socketpair local pré-ouvert, distinct des records de
  fichiers, messages framed JSON canoniques `READY`, `RESULT` ou `STOP`, sans
  chemin ni FD reçu après spawn ;
- résultat worker : état, phase, reason code, observations de stabilité,
  autorités de sortie et hash du message ;
- chaque canari : code, opération, chemin synthétique, résultat `DENIED` et
  errno attendu ;
- receipt worker porté par le launch receipt : PID audit-only, exit/signal,
  stdout, stderr et résultat de contrôle.

Les nombres de FD sont des entiers non négatifs et uniques. Les rôles, leur
cardinalité et leur direction sont fermés dans le plan. Un socket n'est jamais
validé comme un fichier régulier hashable. Les maps ont exactement les clés du
plan. Aucun champ libre, optionnel implicite ou valeur non typée n’est admis.

Le résultat terminal ferme obligatoirement :

```text
sealed_input_payload_manifest_sha256
sealed_input_seal_sha256
terminal_tree_kind
terminal_tree_payload_manifest_sha256
terminal_tree_seal_sha256
journal_generation
journal_generation_manifest_sha256
journal_head_event_sha256
```

Pour un `STOP` antérieur à la création d’une autorité, le champ correspondant
est explicitement `null`; pour `INGESTED` ou `QUARANTINED`, aucun de ces champs
n’est nullable.

## 12. Launch receipt

Le launch receipt est le seul résumé terminal du lancement. Il est écrit par
le parent avec `O_EXCL`, JSON canonique, synchronisation fichier et parent,
après la seconde observation de tous les FDs.

Son schéma exact est fermé dans le plan. Il contient notamment :

- phase terminale et `reason_code` exact ;
- hash externe du lock et identité de son FD ;
- commit core et hashes des sources ;
- hash du profil sandbox effectif ;
- identité du runtime et de `sandbox-exec` ;
- `synthetic_run_id` et `attempt_id` ;
- identité et hash avant/après de chaque FD épinglé ;
- statut/exit du worker ;
- résultat terminal du scanner ;
- résultats de tous les canaris synthétiques ;
- observations honnêtes de durée, processus et FDs ;
- autorités exactes des arbres de sortie et de la tête du journal ;
- liste fermée des limites macOS reconnues ;
- verdict final `INGESTED`, `QUARANTINED` ou `STOP`.

`same_worker_process=true`, `same_five_payload_fds=true` et une durée monotone
d’au moins 60 secondes sont obligatoires seulement pour `INGESTED` et
`QUARANTINED`. Pour `STOP`, les valeurs réellement observées sont écrites,
éventuellement `false`, `null` ou inférieures à 60 selon le schéma. Un STOP
pré-spawn n’invente ni PID, ni durée, ni observations worker.

Les canaris sont tous présents, dans l'ordre fermé, pour `INGESTED` et
`QUARANTINED`. Un receipt `STOP` antérieur à la publication de la preuve
canari porte une liste vide ; il ne fabrique jamais de refus non observé. Dès
que `canaries.json` existe et est valide, le receipt doit reprendre la liste
complète, y compris pour `STOP`.

Les timestamps et PID sont audit-only. Ils n’entrent dans aucun identifiant ou
résultat reproductible.

Si le receipt existe déjà :

- octets identiques et état complet : lecture seule, aucun remplacement ;
- différence, corruption ou préfixe incomplet : `STOP`;
- jamais de suppression, troncature ou overwrite.

L’absence de receipt complet interdit toute revendication de run réussi. Les
états de reprise sont fermés :

- sous lease acquis, aucun claim et aucun receipt : création du claim puis
  lancement ;
- claim + receipt complet valide : lecture idempotente, aucun spawn ;
- claim sans receipt, ou receipt incomplet/invalide : `STOP_NO_RERUN` ;
- claim absent avec receipt présent : conflit `STOP`, aucun claim ni spawn ;
- claim invalide : conflit `STOP`, aucun spawn ;
- lease non acquis : conflit `STOP` sans création de claim ;
- lease acquis sans claim : le launcher résout alors l'état receipt puis peut
  créer le claim ;
- aucun état ne permet de changer de run ou d’attempt.

## 13. Gates avant implémentation et avant run

### Gate A — audit du préenregistrement

Une revue indépendante doit vérifier :

- schéma exact du lock et du receipt ;
- séparation lock/authorization/claim/lease/receipt ;
- absence de vraie donnée dans les tests ;
- sous-arbres sandbox exacts ;
- limites macOS déclarées ;
- cohérence des hashes et du commit core.

Verdict requis : `GO_IMPLEMENTATION`. Sinon `PIVOT` ou `STOP`.

### Gate B — audit du code

Après implémentation, mais avant toute fixture ou tout run :

- tous les futurs hashes sont remplacés par des valeurs concrètes dans un
  nouveau lock ;
- le commit d’autorisation post-lock contient un manifest exact et aucune
  modification des sources du commit d’implémentation ;
- tests unitaires, statiques et sandbox macOS sont verts ;
- pendant le run, le launcher est l’unique source utilisant `subprocess` ;
  avant le lock, le sealer peut invoquer uniquement `/usr/bin/otool` pour
  fermer les dépendances Mach-O, et ses sorties sont validées puis hashées ;
- aucun chemin réel n’est ouvert par les tests ;
- un audit indépendant conclut `GO_SYNTHETIC_RUN`.

### Gate C — run S0

Un run S0 est recevable seulement si :

- la séquence fermée est respectée ;
- le lease est tenu pendant toute l’exécution ;
- les observations parent avant/après concordent ;
- les canaris échouent comme attendu ;
- pour un succès, le worker observe au moins 60 secondes avec les mêmes FDs ;
- un receipt terminal immuable et complet existe.

Même réussi, ce gate conclut seulement
`INGESTED_SYNTHETIC_SCANNER_SEALER_V412`. Il n’autorise aucun CRM réel.
