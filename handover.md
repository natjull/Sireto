# SIRETO Handover - 30 Juillet 2026

## Etat des lieux
Le pivot V4.12 vers un holdout CRM réellement frais est désormais
préenregistré et contre-audité **`GO_CONTRACTS_FINAL`** sans ouverture d'un
nouveau CRM. Trois frontières séparées sont gelées :

Le successeur exécutable S1 est maintenant préenregistré et deux audits
indépendants rendent **`GO_S1_IMPLEMENTATION`**. L’admission est manifest-only,
Worker Q voit le CRM sans vérité, Worker E voit les preuves sans nom/adresse
CRM, et le scorer ne voit que les requêtes scellées. Les registres matériels,
schémas exacts, catalogues payload/seal, signatures Ed25519, ledger producteur
séquentiel, anti-relance durable, gates distincts et évaluation retrieval
one-shot sont fermés. La suite complète donne `1152 passed`. Ce GO autorise
uniquement la construction des catalogues et l’implémentation synthétique ; il
n’autorise aucune ouverture CRM réelle.
*(commits GitHub : réparation tests R3 `421ec40`, préenregistrement
`6cbd80a`, fermeture autorités `f7079ed`, ordre producteur `288a1de` ;
rapport : `reports/v9/v4_12_fresh_s1_preregistration_results.md`)*.

L'autorité locale Ed25519 du futur producteur S1 est préenregistrée, sans
accès au CRM et sans création de clé réelle. Le correctif ferme
l'appartenance de l'item Keychain par le SHA-256 exact du claim dans
`kSecAttrGeneric`, lie le claim au lock, à l'autorisation et à un nonce
aléatoire, interdit la synchronisation Keychain et impose
`AfterFirstUnlockThisDeviceOnly`. Un audit hostile a ensuite relevé que cette
accessibilité exige sur macOS le Data Protection Keychain : les dictionnaires
`SecItemAdd` et `SecItemCopyMatching` imposent maintenant
`kSecUseDataProtectionKeychain=true` et sont fermés séparément de la
projection des attributs retournés par l'OS. Les schémas imbriqués sont
désormais fermés, le commit de certification S1 est corrigé et l'autorisation précède
les audits de provisionnement. Les 7 tests ciblés et la suite complète
(`1159 passed`) sont verts. Deux audits indépendants rendent désormais
**`GO_S1_LOCAL_PRODUCER_IMPLEMENTATION`** sur le commit exact `c64c0c9`.
Ce verdict autorise uniquement le code et les tests synthétiques du
provisioner ; aucun item Keychain, root S1 ou CRM réel n'a été ouvert.
*(commits GitHub : préenregistrement initial `aacc76a`, fermeture ownership
et gates `28ef796`, Data Protection Keychain `c64c0c9`)*.

Le cœur synthétique du provisioner local S1 est désormais implémenté. Il
produit le claim durable, la clé/signature Ed25519 via un backend injecté, le
genesis, le payload, le seal et le receipt ; il reprend le même attempt après
crash, refuse tout item étranger ou artefact divergent et retourne un receipt
valide sans relire le secret. Vingt tests dédiés couvrent notamment le vecteur
RFC 8032, les contrats Data Protection Keychain, cinq frontières de crash,
l'idempotence et la corruption ; la suite complète donne `1179 passed`. Le
backend macOS réel reste volontairement fermé par
`NATIVE_KEYCHAIN_NOT_PINNED` jusqu'à son implémentation et son audit : aucun
accès Keychain réel, CRM ou `/Volumes` n'a eu lieu.
*(commit GitHub : cœur et tests synthétiques `97f7d0d`)*.

Le premier contre-audit du cœur synthétique rend **`NO_GO`** avant backend
natif. Il valide le one-shot, la cryptographie, la reprise et la gestion des
secrets, mais exige trois fermetures : validation récursive de tous les types
et pins du lock, reconstruction publique exacte du payload depuis plan+lock,
et store entièrement FD-ancré appliquant les pins root/device/volume. La
couche native et tout run restent fermés jusqu'au correctif et à deux nouveaux
audits. Le correctif est maintenant figé : validateur récursif des 13 schémas,
contrôles exhaustifs plan/contrat/implémentation/runtime/Keychain/device/UUID,
reconstruction publique du payload et du genesis, et store exclusivement
`openat`/`O_NOFOLLOW` avec identité avant/après. Les tests dédiés passent à 31,
dont mutations de chaque famille de pins, payload auto-cohérent falsifié,
symlink, hardlink et permissions ; la suite complète donne `1190 passed`.
Il reste soumis aux deux ré-audits avant toute couche native.
*(commit GitHub : fermeture des frontières de confiance `83efe3b`)*.

Le second ré-audit confirme ces trois corrections mais maintient `NO_GO` sur
deux derniers écarts : absence de preuve de concurrence et temps logique du
claim seulement typé, pas égal au plan. Le claim compare maintenant cette
valeur exacte ; un test concurrent synchronise deux launchers et prouve un
seul claim, item, arbre d'autorité et receipt. Les frontières de crash
`CLAIM_DURABLE`, `KEYCHAIN_QUERIED` et `SEED_GENERATED`, ainsi que les états
terminaux du receipt, sont aussi couvertes. Les tests dédiés passent à 38 et
la suite locale donne `1197 passed`; la reproduction d'audit collecte les
mêmes 1197 tests avec `1135 passed, 62 skipped` selon ses capacités
d'environnement. Deux audits indépendants rendent désormais
**`GO_SYNTHETIC_CORE_NEXT_NATIVE_BACKEND`**. Ce GO autorise uniquement
l'implémentation et les tests mockés du backend Data Protection ; un test
multiprocessus reste obligatoire avant tout run réel.
*(commit GitHub : concurrence, temps logique et crashs `3b38fe0`)*.

Le backend macOS Data Protection est maintenant implémenté en processus via
Security.framework et CoreFoundation, sans commande `security`, UI, argument,
environnement ou fichier temporaire. Il construit séparément les
dictionnaires exacts `SecItemCopyMatching` et `SecItemAdd`, vérifie uniquement
la projection persistée autorisée, copie la graine dans un buffer mutable et
libère tous les objets CF. Les 43 tests du provisioner utilisent des APIs
factices ; `main` prouve qu'il s'arrête sur l'absence du lock avant même de
construire le backend natif. La suite locale donne `1202 passed`. Aucun appel
Keychain réel ni création sous `/Volumes` n'a eu lieu ; le commit reste soumis
à deux audits et à un pré-vol lecture seule avant toute autorisation.
*(commit GitHub : backend Data Protection natif `7d40c85`)*.

- le registre de compatibilité ferme les 23 609 anciennes lignes avec des
  empreintes SIRET-masked/fuzzy et des clés de lignée HMAC privées
  *(commit GitHub : `96be59e`)* ;
- le registre `consumed_sirens` ferme uniquement les identités autoritatives
  déjà consommées, en excluant candidats, prédictions et sondes techniques
  *(commit GitHub : `0b47b4c`)* ;
- l'intake impose une frame exhaustive sans arrêt opportuniste, une couverture
  `MATCH_EXACT / toutes lignes source` >= 80 %, au moins 657 exacts, des
  preuves oracle-side séparées, un scoring retrieval one-shot à 100 candidats
  maximum et une certification AUTO 99,8 % distincte
  *(commit GitHub : `9c7eccd`)*.

Les deux registres préalables sont désormais réellement construits, scellés
et contre-audités **`GO_V412_CONTAMINATION_REGISTRIES`** :

- `consumed_sirens` ferme 64 618 observations et 19 754 SIREN uniques, sans
  candidat, prédiction ou sonde technique et avec zéro rejet. Build
  `fbc0b84d9c81b01a`, manifest
  `b220efd7c4dc89a980b9d0b5501e16fd286edcafdff61573ae6c5e8d8423c6ff`
  *(commits GitHub : code/tests `3b66fd7`, contrat/plan `9f74c00`,
  cross-pin intake `a20c704`)* ;
- `consumed_compatibility` ferme les 23 609 anciennes lignes, dont les 225
  cas du challenge, par des keysets privés HMAC/masked/fuzzy. Build
  `48851668dd2f173686f3240ecc62e30fcbfdb96d8abf0ced498eb29891d8a490`,
  seal `2068a5d18aac189b7bffc0515054fa31166cb5cd9e4d066f143d3c2d5bc3e976`,
  zéro rejet. La clé reste dans le Keychain et est lue en processus sans UI,
  argument, environnement, fichier temporaire ou log
  *(commits GitHub : identité volume `6de4585`, Keychain `4a5ac60`,
  contrat/plan `38b18d8`, cross-pin `4b8bd2a`)*.

Le premier lancement du registre de compatibilité s'est arrêté sans payload :
le CSV réel porte un BOM UTF-8 non déclaré. L'attempt
`v412-compat-8c4f31ce-attempt-01` reste immuable avec son seul receipt et
`ATTEMPT_RECEIPTED`. Le correctif épingle exactement le BOM initial, conserve
un éventuel `U+FEFF` dans une valeur CRM sale et prouve la parité réelle des
23 609 lignes. Le second attempt a été publié après reproduction
byte-for-byte et deux audits indépendants
  *(commits GitHub : ancien lock révoqué `213a3b0`, code BOM `47e9772`,
  contrat/plan `6f9ad7e`, cross-pin `63e45f1`, lock final `5516ba6`)*.

Le scanner/sealer d'arrivée S0 est maintenant préenregistré et deux audits
indépendants rendent **`GO pour commencer le code S0`**. Sa fixture de six
lignes est entièrement déterministe, l'identifiant de run n'est plus
circulaire, les trois types d'arbres scellés et le journal de reprise sont
fermés, et les tests négatifs couvrent stabilité, structure CSV, quarantaine,
conflits et crashs. Ce GO autorise uniquement l'implémentation sur synthétique :
aucun CRM réel ni run autoritaire ne peut être ouvert avant le sandbox, le
launcher, le verrou et le control manifest pinnés.
*(commit GitHub : `50333d3`)*.

Le cœur S0 est désormais implémenté et contre-audité
**`GO_CORE_PRELOCK`**. Le producteur déterministe et le scanner test-only
ferment la stabilité sur FDs, les arbres et receipts scellés, les
quarantaines, le journal et sa reprise, les bindings de provenance et les
sorties Parquet. La matrice défensive compte 62/62 tests verts sur le SSD ;
elle couvre aussi les métadonnées applicatives, liens, ancêtres de chemins,
reçus partiels ou concurrents et dates impossibles. Toute invocation hors du
répertoire pytest dédié reste refusée : le prochain geste est exclusivement
la matérialisation puis l'audit du sandbox, du launcher et du verrou.
*(commit GitHub : `38287c1`)*.

Le contrat autoritatif du lancement S0 franchit désormais
**`GO_IMPLEMENTATION`** après trois cycles de contre-audit. Il ferme un worker
FD-only distinct du core, le runtime Python/PyArrow privé, la sandbox
`deny default`, les autorités parent/worker disjointes, le protocole de
contrôle, l'automate lease puis claim anti-rejeu, les canaris synthétiques et
la cohérence complète résultat/exit/receipt. Le plan canonique amendé a pour
hash
`f73d855b9d6c76f6175cae5e04f2bd2bc61a19a5d78d356ebe99d3d6289f8596`
et épingle le contrat
`b969a8d552ba060e5b7e24bd1e295abbaf025f1dfbfb7e5683bd5853b689b5df`.
L'amendement `GO_AMENDMENT` interdit de fabriquer des preuves canaris lors
d'un STOP précoce : succès = liste complète, STOP avant preuve = liste vide.
Ce GO autorise seulement l'implémentation du launcher, du worker, du sealer et
du profil ; aucun run, fixture nouvelle ou CRM réel n'a été ouvert.
*(commits GitHub : contrat initial `46b1958`, amendement `7a3353f`)*.

Le bundle autoritatif S0 est implémenté et deux audits indépendants rendent
**`GO_CODE_BUNDLE`**, sans fixture autoritative ni run. Le sealer construit un
runtime privé de 1 528 fichiers et un lock sans suivre les liens ; le launcher
sans argument ferme lease, claim, reprise, receipts, TOCTOU, canaris et arbres
de sortie ; le worker n'accepte que les FDs, attend réellement 60 secondes et
réutilise le core immuable. Les 109 tests S0 passent. La suite complète donne
1 071 succès et le seul échec historique connu, causé par un test qui interdit
tout `/Volumes/CATNAT_DATA` alors que le `TMPDIR` obligatoire y réside. Ce GO
autorise uniquement la construction de la fixture puis du lock, suivie de leur
audit avant autorisation et lancement.
*(commit GitHub : `42d9027`)*.

Le premier lock autoritatif S0
`feeef92c7df4c24473d3850f0b074aa5e5f904ac79c507f674606d4b6057a598`
a été révoqué **avant autorisation et avant lancement** : son audit matériel
était intégralement vert (1 722 contrôles), mais un contre-audit de cohérence a
détecté que le launcher exigeait à tort une identité de fichier non nullable
pour le canari `EXISTING_DIRECTORY`, alors que le sealer et le schéma
autoritatif imposent trois valeurs nulles. Le launcher valide maintenant ce
répertoire par ouverture ancrée, sans lien symbolique, puis contrôle son
propriétaire, son volume et ses permissions. Deux audits indépendants rendent
`GO_PATCH`; les 110 tests S0 passent et le vrai manifeste de canaris est
accepté. L'ancien lock ne doit jamais être autorisé : il doit être archivé de
façon récupérable, puis la fixture et un nouveau lock doivent être reconstruits
sur le commit corrigé.
*(commit GitHub : `61a52c5`)*.

Le deuxième lock S0
`f918b8af6c9dc47bc61bcb6ab36d0808704206a28865f8fa32b629b1a32d59e2`
avait franchi deux audits statiques, puis le pré-vol exécutable a détecté un
second défaut avant tout worker : le helper de lecture imposait l'UID
utilisateur aux deux autorités macOS légitimement détenues par `root`
(`SystemVersion.plist` et `/usr/bin/sandbox-exec`). L'autorisation initiale
`0bcdb7a`, non canonique, puis sa correction `10b907e` sont toutes deux
révoquées avec ce lock et ne doivent jamais servir à un lancement. Le helper
accepte désormais un propriétaire explicite, limité à `uid=0` pour ces deux
fichiers système ; toutes les autorités privées restent obligatoirement
détenues par l'utilisateur. Deux audits rendent `GO_PATCH` et `GO_PATCH_2`,
les 111 tests S0 passent et le pré-vol runtime réel est vert. Le deuxième
environnement doit être archivé sans suppression, puis lock et autorisation
doivent être reconstruits sur le commit corrigé.
*(commit GitHub : `2bb2bc2`)*.

Le troisième lock S0
`d608a0e13334270a16a554f3ca676135b4cec671af3a52d468ec8a8a28a40e50`
a lui aussi été arrêté au pré-vol, sans autorisation mise à jour ni worker. Le
launcher mettait en cache l'UUID de volume par `st_dev`; or, sur ce Mac, le
dépôt (volume Data) et `/` (volume System) partagent le même `st_dev` tout en
ayant des UUID APFS distincts. Selon l'ordre de lecture, la frontière de
confiance système était donc remplacée par celle du dépôt. Le resolver relit
désormais l'UUID directement sur chaque FD avec contrôle d'identité
avant/après, sans cache ambigu. Deux audits indépendants rendent
`GO_APFS_PATCH` et `GO_APFS_PATCH_2`; les 112 tests S0 et les validations
runtime/volumes du lock réel passent. Ce troisième lock reste révoqué car il
n'épingle pas le blob corrigé ; reconstruire encore lock et autorisation avant
tout lancement.
*(commit GitHub : `75edb12`)*.

Le run autoritatif S0-R1 a été exécuté une seule fois et conclut **`PIVOT`**.
Son receipt immuable
`68d1267351447d6dd755cfca62cccec700715191b45a906e28ecc59b40bc6746`
rapporte `WORKER_CONTROL_INVALID`, enfant `exit=65`, aucune frame
`READY`, aucune sortie, aucun canari et aucune stabilité fabriqués ; les
13 autorités parent sont identiques avant/après. La cause est prouvée
byte-for-byte : le stderr de 62 octets, SHA
`1d24b61273dbf35a7162215eaa0aa2668c83773f003884a01e326b8065132cf7`,
est exactement
`sandbox-exec: /dev/fd/effective.sb: No such file or directory\n`.
Le transport verrouillé `sandbox-exec -f /dev/fd/<fd>` échoue donc avant
Python et avant le worker. Le claim et le receipt R1 restent immuables ; aucun
rerun, déplacement ou rebuild sous les mêmes identifiants n'est autorisé.
La suite exige une autorité S0-R2 préenregistrée avec nouvelle racine,
nouveaux `synthetic_run_id` et `attempt_id`, et transmission du profil par
`-p` depuis les octets relus et rehashés du FD retenu.
*(commit GitHub d'autorisation R1 : `37b453f`)*.

Le successeur S0-R2 franchit **`GO_R2_IMPLEMENTATION`** et
**`GO_R2_IMPLEMENTATION_2`**. Il conserve R1 immuable, utilise la racine
distincte `fresh_holdout_intake_synthetic_r2`, dérive un nouveau run
`bjpoibmapghmeklagcnddeamijgmlfijmifdobbmmanmohkknplbpolonjfjahlo`,
et impose un nouvel attempt. Le transport du profil devient `sandbox-exec -p`
depuis les octets relus et rehashés du FD parent, sans FD profil transmis. Un
smoke final sans payload doit réussir après construction du runtime et être
attesté dans un lock R2 schema-3 avant toute autorisation. Plan canonique :
`e05102a36b9aaf37ed3aa1052814a9e2bb8ff77a62d26cf135f9ff1f240abd27`;
contrat :
`2933d217f169b67d3eff399c5b270a91590a2ccec82430469a3ab8489a17a937`.
Ce GO autorise seulement l'implémentation R2.
*(commit GitHub : `4cf640e`)*.

Le Gate A R2 a ensuite découvert, avant création de la racine R2, que
`sandbox-exec` supprime les variables `DYLD_*` et rend inexécutable la copie
du stub Homebrew `bin/python3.14`. L'amendement R2-B franchit désormais deux
audits indépendants **`GO_R2B_IMPLEMENTATION`**. Il copie le vrai helper
`Python.app`, conserve stdlib et PyArrow privés, supprime tout `DYLD_*`, et
épingle comme unique exception hôte la bibliothèque framework exacte,
retenue et rehashée avant/après. Le smoke pré-lock doit importer
`encodings` et PyArrow 23.0.1 depuis le runtime privé, sans stdout/stderr.
`otool` reste limité au sealer pré-lock ; le launcher revalide l'install name
Mach-O en processus afin de conserver un seul child autoritatif. Plan
canonique :
`2ab9a1d5954588c01de22c54e21c721aa0e9da9a9e7f140d9f93950cb8b1abf4`;
contrat :
`66418a23ae6b166f253f7ef4bc220e3a47ce0655c2ee96c7e8a9db51e0519a42`.
La sonde homologue hors racine R2 réussit avec `exit=0`, stdout/stderr vides
et aucun accès général à `/opt`. La racine R2 reste absente ; ce GO autorise
uniquement l'implémentation et son Gate B avant toute création R2.
*(commit GitHub : `5fc7116`)*.

Le bundle R2-B franchit maintenant le Gate B **`GO_R2B_CODE_FINAL`**, confirmé
par deux contre-audits. Le runtime homologue construit hors racine R2 contient
1 525 records fermés ; son smoke réel charge `encodings` et PyArrow 23.0.1
depuis le privé avec `exit=0`, stdout/stderr vides, puis le launcher reconstruit
exactement la même attestation. Le Gate B a détecté et corrigé avant commit un
alias Homebrew de stdlib qui dupliquait le framework hôte, puis une capture
stdout/stderr initialement plafonnée après coup : la lecture est maintenant
bornée en continu à 65 536 octets par flux, avec kill/close/wait au
dépassement ou timeout et stdin sur `/dev/null`. La suite complète est verte :
1 090 tests passent sur le `TMPDIR` SSD. La racine R2 reste absente ; le
prochain geste autorisé est l'audit du commit, puis seulement la construction
de la fixture et du lock R2.
*(commit GitHub : `0afb010`)*.

L'unique exécution autoritative S0-R2 a ensuite conclu
**`PIVOT_R2_WORKER_IDENTITY`**. Son receipt canonique immuable, SHA-256
`6d9fb590bab4d205ce9004454954d47406de5e0d2ec74ad9390f01f6948f839e`,
atteste onze canaris refusés, le même processus et les mêmes cinq FD pendant
`60.005023459` secondes, puis `WORKER_CONTROLLED_STOP`, sans stdout, stderr
ni sortie `sealed`, `scan`, `quarantine` ou `tmp`. Le builder R2 dérive
correctement `bjpoib...` avec le domaine successeur et le receipt R1, mais le
worker recalcule encore l'ancienne identité cœur `komapn...`; il échoue donc
avant toute écriture sur l'invariant d'identité. Le catch global masque cette
cause sous un code générique. R2 est consommé et ne doit jamais être relancé.
Avant tout R3, corriger la dérivation, produire un STOP à phase/code fermés et
faire atteindre `INGESTED` au vrai `_process` dans un gate sandbox jetable.
Autorisation R2 : commit GitHub `5dbb2ff`. Rapport de pivot :
`reports/v9/v4_12_fresh_s0_r2_pivot.md`
*(commit GitHub : `648cd4f`)*.

Le gate jetable du successeur conclut désormais **`GO_PREREG_R3`**, sans
autoriser encore build, lock ou run R3. L'artefact probant
`diag-r3-successor-gate.yww2qf5m` possède un résultat persistant canonique,
SHA-256
`c86ad8bf1a4b8af0525c6870e05ddabb2f27c4208f9f07c8be07edebb52e212b`.
Sous le vrai Python privé et Seatbelt, le worker atteint `INGESTED`, conserve
les mêmes FD pendant `60.003147917` secondes, refuse les onze canaris avec
`EPERM`, garde stdout/stderr vides et publie trois générations de journal.
Une identité R1 complète est rejetée
`IDENTITY/EXECUTION_IDENTITY_SCHEMA_INVALID` sans mutation de sortie. Le
worker consomme désormais l'identité successeur du spec, et
`control-result-2` transporte un diagnostic fermé validé par le launcher.
Les 1 095 tests passent. Les gates antérieurs `zi2oynzm` et `9fy69i1l`
restent non promotables. Le prochain geste est exclusivement la
préinscription du contrat/plan R3 fermant la chaîne lock → spec → worker.
Rapport :
`reports/v9/v4_12_fresh_s0_r3_gate_results.md`
*(commit GitHub : `5d1820d`)*.

Le contrat et le plan S0-R3 franchissent maintenant deux audits indépendants
post-commit **`GO_R3_IMPLEMENTATION`**. Le plan canonique SHA-256
`ce7f8ed4a9d6236e61cffca72b92a1043d414afc69571ae79c94f191e6def1e2`
est lié au contrat
`247b41f60a39211f85431d141625bf0d8321ae88c701d17ffd380a04ef7a9353`.
L'overlay fermé applique 30 overrides et quatre suppressions au plan R2 :
les schémas R3 matérialisent intégralement champs, nullabilité et types,
`SANDBOX_EXEC` reste une autorité système hors des blobs Git, et aucune
identité R2 interdite ne pilote R3. Les 11 tests R3 et les 1 106 tests du
dépôt passent sur le SSD. La racine R3 reste absente. Ce GO autorise seulement
l'implémentation R3 ; ni build, ni autorisation, ni run, ni CRM réel ne sont
encore ouverts.
Préenregistrement : commits GitHub `7bf1ea4`, correctif `a483107`.
Rapport :
`reports/v9/v4_12_fresh_s0_r3_preregistration_audits.md`.

Le bundle S0-R3 franchit désormais deux contre-audits indépendants
**`GO_R3_CODE_BUNDLE`**. Le commit d'implémentation `8d8e0a3` relie le builder
R3, le sealer, le launcher receipt-3 et le worker à l'identité littérale
préenregistrée. Le gate Seatbelt jetable
`diag-r3-successor-gate.sfzj9buk` atteint `INGESTED`, refuse 11/11 canaris,
conserve les mêmes FD pendant `60.010024083` secondes, publie trois
générations de journal et rejette une identité R1 sans mutation. Son résultat
canonique a pour SHA-256
`556558d4372b003d23190b86ff8163e021e0a937b83539b0fcc1e4828b53185b`.
Les deux audits ont rehashé les 1 525 fichiers du runtime, les seals et le
journal. La suite donne 1 078 succès et 62 skips, zéro échec, sur le SSD. La
racine autoritative R3 reste absente ; le prochain geste autorisé est sa
construction unique, puis le lock et son audit, jamais le CRM réel.
Rapport :
`reports/v9/v4_12_fresh_s0_r3_code_gate_results.md`.

L'unique exécution autoritative S0-R3 est désormais doublement certifiée
**`INGESTED_R3_CERTIFIED`**. Le receipt-3 canonique SHA-256
`8061247794f403f52a692e41f19549dcf2803a6db744c74e9719cb824ad96a08`
lie l'autorisation `f686ffd9…`, le lock `de545687…`, le claim `c6a1c5…`,
le spec worker et les sorties. Le worker termine `exit=0`, stdout/stderr
vides, après `60.002720750` secondes avec le même processus et les mêmes cinq
FDs ; 11/11 canaris sont refusés, les 14 observations parent sont identiques,
les arbres sont scellés et le journal compte trois générations. Il existe
exactement un claim, un lease, un spec et un receipt ; le chemin idempotent
interdit désormais tout nouveau spawn. Aucun processus R3 n'est actif.
Autorisation initiale non canonique jamais lancée : commit `ffccb7e` ;
autorisation canonique utilisée : commit GitHub `b64133f`.
Rapport :
`reports/v9/v4_12_fresh_s0_r3_authoritative_results.md`.
Ce succès ouvre uniquement la construction puis la qualification aveugle du
CRM frais ; labels, retrieval, modèles et test final restent fermés.

Rapport complet :
`reports/v9/v4_12_contamination_registries_results.md`
*(commit GitHub : `a0b510a`)*. Le prochain geste autorisé est le
scanner/sealer d'arrivée sur paquets synthétiques uniquement. Aucun futur CRM
ne peut encore être ouvert et le ranker, le decider, le risk model et
l'accepteur restent gelés. Tout test ou build suivant doit utiliser le SSD
externe ; aucun nettoyage destructif n'a été effectué.

Le builder des entrées sûres du moteur unitaire V4.12 est implémenté et
contre-audité, sans build réel. Une première revue a rendu `STOP_CODE` malgré
18 tests verts et a exposé une mauvaise empreinte TF-IDF, un contournement du
plan interne, des sorties Parquet non revérifiées, une signature seulement
recopiée et une fenêtre TOCTOU CRM. Les défauts ont été reproduits puis
corrigés. Deux contre-audits indépendants rendent désormais `GO_CODE` et
`GO_CODE_2`; 27 tests ciblés et les 645 tests du dépôt passent. Les
inventaires réels partitions/cache et la signature historique ont été
recalculés exactement. Ce GO autorise seulement le verrou puis le build des
entrées physiquement aveugles ; il n'autorise ni oracle, ni worker, ni
benchmark. *(commits GitHub : builder `18eb76e`, audit `46868db`)*
Le séquencement du verrou a ensuite été fermé : le commit du verrou peut
suivre le commit audité sans rendre l'exécution impossible, tandis que chaque
source reste identique au worktree, au verrou et au blob du commit audité.
Le contre-audit rend `GO_LOCK_SEQUENCING`; 29 tests ciblés et 647 tests
complets passent. *(commit GitHub : `c97c737`)*
Le verrou d'exécution des entrées sûres est contre-audité `GO_LOCK`. Il
épingle le commit audité, les cinq sources, queries/split, les deux
inventaires complets, le runtime et les trois racines SSD. Son hash avant
commit est `e794c60f...e9def315`; aucun build n'avait encore été lancé lors
du gel. *(commit GitHub : `7c31051`)*
Le build réel des entrées sûres franchit **`GO_V412_UNIT_INPUTS`**, confirmé
indépendamment par `GO_V412_UNIT_INPUTS_AUDIT` et 21 161 assertions sans
import du builder. Les 7 003 requêtes, dont 1 456 dev, ne contiennent que les
six champs CRM autorisés ; les inventaires scellent 4 119 partitions et
1 454 paires TF-IDF. Le ledger séparé couvre exactement 7 029 fichiers.
Aucun label, oracle, modèle, résultat candidat ou chemin sensible n'est
présent dans le paquet runtime. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/inputs/v4_12_unit_engine/ca0b22e79cd2e92a32c009266e6d967b4ea48654de8736bca2b0ea7fdc9f8d6e`.
Rapport : `reports/v9/v4_12_unit_input_results.md`. Ce GO autorise seulement
le préenregistrement de l'oracle séparé. *(commit GitHub : `e5d01a9`)*
Le contrat de l'oracle dev V4.12 est préenregistré `GO_CONTRACT_ORACLE`.
L'oracle sera truth-only : 1 456 IDs, 1 217 `MATCH_EXACT` et 239
`AMBIGUOUS`, sans candidat, rang, score, preuve ou décision historique. Une
première revue a refusé la simple séparation de dossiers sur le même SSD ;
le contrat corrigé exige que le futur worker tourne sous `sandbox-exec`, avec
les racines oracle/audit interdites et une sentinelle d'ouverture réellement
refusée. L'oracle reste historique, non indépendant et non certifiant.
*(commit GitHub : `1dd7428`)*
Le contrat précise désormais le ledger exhaustif des huit fichiers réellement
ouverts par le builder oracle : six fichiers du paquet runtime sûr plus
labels/split. Les inventaires sont ouverts uniquement pour contrôler
l'intégrité du paquet, jamais comme résultats de retrieval ni pour former la
vérité. *(commit GitHub : `bbf31b9`)*
Le builder d'oracle et ses tests franchissent `GO_CODE_ORACLE` après quatre
refus d'audit : rescellation complète, sibling modifié à taille/mtime
restaurées, ledger incomplet puis ledger réordonné. Les trois PoC finaux sont
désormais bloqués ; 23 tests ciblés et 670 tests complets passent. Aucun
build réel n'avait encore été lancé. *(commits GitHub : builder `7eafad8`,
audit `02e954b`)*
Le verrou d'exécution oracle est contre-audité `GO_LOCK_ORACLE` avec 4 434
assertions : cinq sources Git, quatre inputs, six fichiers runtime sûrs,
populations, ordre, payloads, runtime et racines sont exacts. Hash du verrou :
`4d598cf1...f6d4c8b1`. *(commit GitHub : `04a22db`)*
L'oracle séparé franchit **`GO_V412_UNIT_ORACLE`**, confirmé par
`GO_V412_UNIT_ORACLE_AUDIT` et 4 430 contrôles sans import du builder. Il
contient 1 456 lignes : 1 217 `MATCH_EXACT` et 239 `AMBIGUOUS`, uniquement
issues des labels/split historiques. Le ledger ordonné couvre les huit
fichiers réellement ouverts. Aucun résultat retrieval, score, décision,
modèle, challenge ou test final n'a participé à la vérité. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/oracles/v4_12_unit_engine/c4045da8ad1e0b9af35f3d7552176dec76ee2ba36fa759ee2dc0664c93d2fa70`.
Rapport : `reports/v9/v4_12_unit_oracle_results.md`. *(commit GitHub :
`9d2c68e`)*

Le contrat Gate A des stores stricts et de la sandbox V4.12 est
préenregistré `GO_CODE_V412_STRICT_STORES`. Trois audits indépendants ont
fermé le routage, les 648 partitions, les 648 caches et le lookup snapshot,
ainsi que la liste blanche réelle de `sandbox-exec`. Une erreur de conception
a été interceptée avant code : les caches portent sur 4 764 472 rows
filtrées/dédupliquées et non sur les 8 030 285 rows physiques. La sandbox
épingle `sandbox-exec`, Git, `Python.app` et la bibliothèque framework,
refuse oracle/audit, réseau, fork et écritures hors espaces privés, et expose
exactement 1 945 fichiers à l'enfant. Le
ledger parent attendu en couvre 1 954. Ce GO autorise seulement
l'implémentation puis son audit, pas encore le build réel ni les modèles.
Rapport : `reports/v9/v4_12_strict_stores_contract_audit.md`. *(commit
GitHub : `0173d6b`)*

Le contrat Gate A a été durci après dix-sept refus successifs : contrôles
consommés par descripteurs ancrés, profil transmis en mémoire, Git absolu,
runtime Python privé, publication atomique et reprise rattachée aux entrées
courantes. La frontière de confiance est maintenant explicite : le runtime
local `/System`, `/usr` et `/opt/homebrew` est enregistré mais n'est pas
présenté comme un système intégralement scellé. *(commit GitHub : `d23c287`)*
Les trois stores stricts et leur certificateur sandbox sont implémentés et
contre-audités **`GO_CODE_V412_STRICT_STORES`**. Les contrôles couvrent
partitions, caches TF-IDF, lookup DuckDB via FD, refus sandbox, rescellation,
publication/recovery et nettoyage des espaces privés. Le smoke macOS réel,
les 41 tests ciblés et les 711 tests du dépôt passent. Aucun build Gate A,
verrou, oracle ou modèle n'a été ouvert par cette implémentation. Le prochain
geste autorisé est la création puis le contre-audit du verrou d'exécution.
*(commit GitHub : `e059148`)*
Le premier verrou candidat a été révoqué avant exécution : le contrôle
indépendant a détecté que le hash de la bibliothèque Python, cohérent entre
plan/code/lock, était tronqué à 63 caractères face au fichier réel. Le hash
64 caractères a été corrigé dans le contrat, le plan et le certificateur ;
`GO_CODE_PATCH`, 41 tests ciblés, le smoke réel et 711 tests complets
confirment le correctif. Aucun build n'a été lancé avec le verrou fautif.
*(commit GitHub : `c22d05a`)*
Le verrou corrigé, hash
`31aab729f33db26350da37e8d1fbf427d19a8153112d353973088df83e620b9f`,
franchit `GO_LOCK_V412_STRICT_STORES` et `GO_LOCK_2`. Les deux
contre-audits ont validé respectivement 7 901 et 342 contrôles, dont les
1 945 fichiers physiques (7 224 974 001 octets), les blobs Git, le runtime
réel, le routage, les subsets et l'absence d'inputs interdits. Le prochain
geste autorisé est désormais l'unique build Gate A sous sandbox, toujours
sans accès à l'oracle. *(commit GitHub : `775c3bb`)*
Le premier lancement Gate A s'est arrêté sans publication : `Path.cwd()`
recevait `EPERM` sous la politique metadata-only de `RUN_ROOT`, un chemin que
le smoke initial n'exerçait pas. Deux PoC ont isolé cette cause ; les refus
joblib `SemLock` et `mach-lookup` étaient du bruit non fatal. Le worker
compare désormais l'identité `st_dev/st_ino` de `.` et `RUN_ROOT`, sans
élargir aucun droit, et force joblib en série. `GO_CODE_CWD_PATCH`, 44 tests
ciblés, le smoke réel et 714 tests complets sont verts. Le lock `775c3bb`
est révoqué ; un nouveau verrou est requis avant relance. *(commit GitHub :
`158014e`)*
Le verrou post-correctif, hash
`f9e5738eef35c9a4b9c636cf810a87ed8eb412077f7ac6bcb48c90ae02f8d189`,
franchit `GO_LOCK_CWD_PATCH` et `GO_LOCK_CWD_2` avec 7 900 et 325
contrôles indépendants. Il autorise la seconde tentative complète du même
Gate A, toujours sous sandbox et sans oracle. *(commit GitHub : `e759492`)*
La seconde tentative a terminé le worker complet puis s'est arrêtée avant
publication : APFS `noowners` refuse le renommage d'une racine déjà en
`0555`. Le PoC SSD reproduit l'écart. La promotion conserve maintenant
uniquement la racine en `0700` pendant `rename`, via un FD ancré, puis la
repasse en `0555` dans un `finally`, vérifie l'inode et synchronise les deux
parents. La recovery gèle les états transitoires avant validation.
`GO_CODE_APFS_PATCH`, 58 tests ciblés, 728 tests complets et le smoke réel
sont verts. Le lock `e759492` est révoqué ; aucun artefact incomplet n'a été
publié. *(commit GitHub : `809bb7e`)*
Le verrou APFS corrigé, hash
`265bc418d95a1de1902773b7f5548b5607a2b1360722192658dbddb544a0630d`,
franchit `GO_LOCK_APFS_PATCH` et `GO_LOCK_APFS_2` avec 7 900 et 325
contrôles indépendants. Il autorise la troisième tentative complète du Gate
A, sans changement du worker de matching ni accès à l'oracle. *(commit
GitHub : `7443ef4`)*
La troisième tentative franchit **`GO_V412_STRICT_STORES_SANDBOX`**. Les
648 partitions et 648 caches nécessaires aux 1 456 requêtes dev sont
accessibles sous sandbox, sans cache manquant ou reconstruit ; le lookup
retourne exactement les 10 000 SIRET contrôlés. Le worker n'a ouvert ni
oracle, label, modèle ou résultat historique, et les refus oracle/audit,
écriture, réseau et fork sont effectifs. Son pic RSS est de 1,9568 Go.
Deux audits indépendants rendent également `GO`, avec 20 848 et 11 898
contrôles réussis et le rehash complet des 1 954 entrées du ledger. Artefacts :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/certifications/v4_12_strict_stores/9a99cd246d6d1a118dea064ab1458afe7c3bcb8a9bb28a1da6009d6bc42b4ee4`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_strict_stores/9a99cd246d6d1a118dea064ab1458afe7c3bcb8a9bb28a1da6009d6bc42b4ee4`.
Rapport : `reports/v9/v4_12_strict_stores_results.md`. Ce GO certifie les
stores et l'isolement, pas encore le Recall@100 ni la latence par requête.
Il autorise uniquement le contrat du moteur unitaire et de sa parité ; les
modèles et le test final restent fermés. *(commit GitHub : `614efc2`)*
Le contrat du moteur unitaire retrieval et de sa parité est maintenant
préenregistré et reçu **`GO_CONTRACT_FINAL`**, `GO_CONTRACT_SECURITY` et
`GO_CONTRACT_INDEPENDENT`. Le worker aveugle reproduira localement le sparse
V4.11 exact — nom, adresse, rescues simples, RRF et padding — puis publiera
uniquement `query_id`, rang et SIRET, avec un plafond strict de 100. Il ne
peut ouvrir ni oracle, historique, modèle ou réseau. Après sa terminaison, un
contrôleur séparé, lui-même sans accès au Parquet historique contenant la
vérité, comparera les deux payloads canoniques aux hashes préenregistrés :
145 236 candidats, pools 46–100, hash candidat `1689a2...ab00` et statut
`65e662...5518`. Les blobs du commit reproduisent exactement le contrat
`007ada2f...fe33` et le plan `7eff59a9...180d`. Ce GO autorise uniquement
l'implémentation et son audit de code, pas encore le run, le Recall, les
modèles ou l'oracle. *(commit GitHub : `370a3aa`)*
Le moteur unitaire, son orchestrateur sandbox, le contrôleur de parité et
leur contre-audit indépendant sont implémentés et reçus
**`GO_V412_UNIT_RETRIEVAL_INDEPENDENT_AUDIT`**. Le cœur reproduit le sparse
historique gelé sans importer son pipeline ; le worker ne publie que les
statuts et les listes ordonnées de 100 SIRET au plus. Les deux profils
Seatbelt ont été exécutés réellement sur le Mac avec un Python privé. Les
lectures sensibles descendent par `openat` et `O_NOFOLLOW` sur chaque
composant, les mêmes FDs sont recontrôlés avant/après, et les promotions
utilisent un renommage atomique sans remplacement. Les tests adversariaux
rejettent faux exécutables, rapport de recovery mensonger, substitution du
pending et collision de destination. L'audit final vérifie 13 sources, 12
contrôles statiques et 13 contrôles synthétiques ; 133 tests ciblés et les
803 tests du dépôt passent. Aucun dev réel, store réel, oracle, historique
ou modèle n'a été ouvert par ce jalon. Le prochain geste autorisé est la
création puis les deux contre-audits du verrou d'exécution ; le run worker,
la parité, le Recall et les modèles restent interdits jusque-là. *(commit
GitHub : `6726a95`)*
Un audit d'exécutabilité post-commit a ensuite refusé le passage au verrou :
le worker publié n'avait aucun producteur de run-spec de parité en production.
Le runner enchaîne désormais obligatoirement worker, revalidation, construction
canonique du run-spec, contrôleur Seatbelt et validation complète du `GO`.
La source contrôleur et le run-spec sont consommés depuis des FDs ancrés ; le
Python parent est copié depuis les octets verrouillés dans un runtime privé
scellé ; plan, lock, sources et inputs sont recontrôlés après la parité. Les
PoC faux `GO`, spec mutée, publication substituée, binaire remplacé et cleanup
TOCTOU sont bloqués. `GO_EXECUTION_PATH`, 151 tests ciblés et 821 tests
complets valident le raccord, sans dev réel. Le lock candidat antérieur est
révoqué et doit être régénéré sur ce commit avant toute exécution. *(commit
GitHub : `f4a5309`)*
Le verrou d'exécution final, hash
`0852ee260af4dd66976adaea9831204c1ad968dbdbca6241f07e5f2964b27caf`,
franchit **`GO_LOCK_1`** et **`GO_LOCK_2`**. Il épingle le commit code
`f4a53096338ec5bf2cb3237b5361c6e0e513eedf`, 13 sources, 16 entrées,
les exécutables, le runtime, les quatre racines et les deux projections
worker. Le premier audit a exécuté 15 104 contrôles et rehashé physiquement
7 225 618 142 octets ; le second a recalculé sans importer le runner les
1 945 entrées Gate A contre le ledger de 1 954 lignes. Aucun chemin oracle,
historique ou modèle n'entre dans le worker. Ce GO autorise désormais
l'unique exécution end-to-end worker puis parité ; il n'autorise toujours ni
l'ouverture de l'oracle, ni le Recall, ni les modèles. *(commit GitHub :
`bf82e74`)*
La première tentative sous ce verrou s'est arrêtée en environ cinq secondes,
avant le démarrage du module, toute requête et toute publication :
`ModuleNotFoundError: No module named 'xgb_matcher'`. Le paquet privé était
présent, mais Seatbelt empêchait Python d'en découvrir le répertoire. Le
verrou `bf82e74` est donc révoqué. Le correctif limite la nouvelle lecture au
seul paquet privé scellé, fixe `PYTHONPATH` sur le staging, désactive les
chemins Python implicites et nettoie uniquement le staging courant sur échec
du worker. Deux audits rendent `GO_IMPORT_PATCH` et `GO_IMPORT_PATCH_2`; 153
tests ciblés, deux intégrations macOS natives et les 823 tests du dépôt
passent. Aucun résultat dev, worker ou parité n'a été produit ; oracle,
historique et modèles sont restés fermés. Rapport :
`reports/v9/v4_12_unit_retrieval_launch_failure.md`. Une relance exige un
nouveau verrou doublement audité. *(commit GitHub : correctif `58fabf3`)*
Le verrou de remplacement, hash
`097b65ea73578f2993ffedb133878a7708138a1ab7fa3acc16dfc7b102861359`,
épingle le commit corrigé `58fabf3b42540d1862d1ef3d12cf7cd2f22a2fd4`
et franchit **`GO_LOCK_IMPORT_1`** et **`GO_LOCK_IMPORT_2`**. Les audits
confirment 13 sources, 16 entrées, les exécutables et le runtime exacts. Le
second audit, sans importer le runner, a rehashé les 1 945 fichiers Gate A
(6,73 Gio) et les a rapprochés des 1 954 lignes du ledger. Les projections
worker restent sans oracle, historique, dataset ou modèle. Ce verrou autorise
une nouvelle tentative end-to-end worker puis parité ; il n'autorise toujours
pas l'ouverture de l'oracle, le calcul du Recall ou le dégel des modèles.
*(commit GitHub : `a1c1db8`)*
La tentative autorisée par `a1c1db8` a franchi l'import puis s'est arrêtée
avant toute requête : sous Seatbelt, `platform.platform()` omettait le
processeur et `Mach-O`, ce qui faisait diverger le nom de plateforme du plan
alors que Python et les huit bibliothèques étaient identiques. La valeur
Darwin est désormais reconstruite sans sous-processus à partir de la version
macOS, de la machine et de la taille de pointeur. Le test natif compare le
dictionnaire runtime complet sous la sandbox réelle. Deux audits rendent
`GO_RUNTIME_PATCH_1` et `GO_RUNTIME_PATCH_2`; 153 tests ciblés et les 823
tests du dépôt passent. Le verrou `a1c1db8` est révoqué. Aucun candidat,
manifeste worker ou résultat de parité n'a été produit. Rapport d'incident
mis à jour : `reports/v9/v4_12_unit_retrieval_launch_failure.md`. Une
nouvelle relance exige encore un nouveau verrou doublement audité. *(commit
GitHub : correctif `a0a0e37`)*
Le verrou runtime corrigé, hash
`778946fae29fb427318c29eee4fba71dea60f1b1d6ea67906caab872441d1def`,
épingle `a0a0e3795948d92c5c41e65cfd3998d8e21781ab` et franchit
**`GO_LOCK_RUNTIME_1`** et **`GO_LOCK_RUNTIME_2`**. Les 13 sources, 16
entrées, quatre exécutables et le runtime sont exacts. L'audit indépendant
sans import du runner a de nouveau rehashé les 1 945 fichiers Gate A contre
les 1 954 lignes du ledger. Aucun chemin oracle, historique, dataset ou
modèle n'est exposé au worker. Une nouvelle tentative worker puis parité est
autorisée ; l'oracle et les modèles restent fermés. *(commit GitHub :
`662d555`)*
La troisième tentative termine en 1 030,16 secondes et franchit
**`GO_V412_UNIT_RETRIEVAL_PARITY`**, confirmé indépendamment par
`GO_ARTIFACTS_1` et `GO_ARTIFACTS_2`. Les 1 456 requêtes produisent 145 236
candidats, avec des pools de 46 à 100, 13 pools sous le plafond, aucun pool
vide et aucun lookup manquant. Les payloads candidats
`1689a2f3...ab00` et statuts `65e662c0...5518` égalent exactement les
valeurs préenregistrées. Le ledger couvre 1 980 entrées inchangées ; oracle,
labels, historique, modèles et réseau sont restés fermés. Pic mémoire :
3,39 Gio. Artefacts worker `d2915fe7...dd1a` et parité
`d587937b...05f5`. Rapport :
`reports/v9/v4_12_unit_retrieval_parity_results.md`. Ce GO autorise seulement
le contrat puis l'audit d'un évaluateur oracle séparé ; il ne republie pas
encore le Recall et ne dégèle aucun modèle.
Le contrat de l'évaluateur oracle séparé est préenregistré et doublement
audité **`GO_EVALUATOR_CONTRACT_1`** et
**`GO_EVALUATOR_CONTRACT_2`**. Il gèle la jointure worker
`d2915fe7...dd1a` / oracle `c4045da8...fa70`, les 1 456 requêtes dont
1 217 `MATCH_EXACT`, Recall@1/10/50/100, les Wilson 95/99 et les gates
observés couverture ≥ 80 % / Recall@100 ≥ 99 %. Les références historique,
V2 et V3 seront republiées ensemble mais distinguées de la mesure V4.12.
Un reçu et un journal parent-only sont synchronisés avant toute ouverture
oracle ; la publication audit puis évaluation est non-clobber et sa reprise
est limitée à la promotion d'arbres déjà complets. Tous les payloads,
keysets, schémas et états sont déterministes. Aucun oracle, historique,
modèle ou test final n'a été ouvert pendant ce jalon. Ce GO autorise
uniquement l'implémentation et l'audit de l'évaluateur, pas encore la mesure.
*(commit GitHub : `fe266bd`)*
L'audit de code a ensuite montré qu'une vraie reprise après ouverture oracle
était impossible avec une preuve conservée seulement en mémoire. Le contrat
et le plan sont amendés et doublement reçus
`GO_EVALUATOR_CONTRACT_AMEND_1` / `GO_EVALUATOR_CONTRACT_AMEND_2` :
`computed_attestation.json` scelle désormais les 16 entrées et les deux
arbres validés, puis son hash devient monotone dans le journal v2. Une
reprise post-oracle peut ainsi valider et promouvoir les octets déjà calculés
sans rouvrir la vérité. Les arbres, manifests, rôles 12 data + 4 runtime,
temporaires d'état hors slot et verrou parent durable sont définis
exactement. Aucun oracle ni résultat réel n'a été ouvert pendant
l'amendement. *(commit GitHub : `9e25ebf`)*
L'évaluateur scellé, son parent, son profil Seatbelt, son audit indépendant
et leurs tests sont implémentés et doublement reçus
**`GO_EVALUATOR_CODE_1`** / **`GO_EVALUATOR_CODE_2`**. Le worker est chargé
et attesté avant le commit oracle, puis reçoit les quatre FDs oracle dans
l'ordre contractuel unique via `SCM_RIGHTS`. L'attestation calculée, le
journal v2, le verrou de slot, la reprise sans réouverture oracle, les
manifests, le ledger, la provenance, le plafond RSS et la publication
exclusive sont testés, y compris via le vrai orchestrateur. Les falsifications
coordonnées, symlinks, IDs/rangs invalides et fenêtres de crash sont rejetés.
67 tests evaluator et les 890 tests du dépôt passent ; smokes et audit
statique sont `GO`. Aucun input réel n'a été ouvert. Le prochain geste
autorisé est la création puis le double contre-audit du verrou evaluator, pas
encore l'ouverture oracle. *(commit GitHub : `3ebddc9`)*
Le verrou evaluator, hash
`bcda9024258031ca10e00313443e75ddb5f5650d599e310c4e7eafd27b1e6b4f`,
épingle le commit code `3ebddc9e977151c91d827a783d9996c642e04a58`
et franchit **`GO_EVALUATOR_LOCK_1`** /
**`GO_EVALUATOR_LOCK_2`**. Les 7 sources correspondent au worktree et aux
blobs Git ; les 12 entrées non-oracle ont été rehashées physiquement. Les
quatre engagements oracle ont uniquement été comparés entre plan et verrou,
sans accès filesystem. Runtime, sandbox, racines, RSS et identités build
`50cbc46e...32e7c`, slot `9cf7f6d3...21b7` et attempt
`01260473...c2ed` sont exacts ; aucune destination n'existait au gel. Ce
verrou autorise désormais l'unique évaluation oracle officielle. *(commit
GitHub : `d886ee9`)*
L'évaluation officielle termine `FINAL` et franchit
**`GO_V412_UNIT_RETRIEVAL_EVALUATION`**, confirmé par
`GO_EVALUATOR_ARTIFACTS_1` / `GO_EVALUATOR_ARTIFACTS_2`. Sur 1 456
requêtes, 1 217 sont `MATCH_EXACT` et 239 `AMBIGUOUS` : couverture
identifiable **83,585 %**. Le Recall exact vaut 1 075/1 217 à @1
(88,332 %), 1 211/1 217 à @10 (99,507 %) et 1 217/1 217 à @50/@100
(100 %), avec zéro vérité absente. La borne Wilson bilatérale 99 % de
Recall@100 est 99,458–100 %. Les 145 236 candidats respectent tous le
plafond 100. La chaîne officielle possède sept événements, 16 entrées
conformes et termine `FINAL`; aucun modèle ni test final n'a été ouvert.
Rapport : `reports/v9/v4_12_unit_retrieval_evaluation_results.md`. Ce GO est
un gate développement historique ; il autorise le contrat de l'unique test
retrieval final, pas encore le dégel du ranker/accepteur.
L'audit de transférabilité conclut ensuite
**`PIVOT_NEW_HOLDOUT_REQUIRED`**. Le test final sélectif consommé mesurait
une admission multicanal différente : elle obtenait 2 116/2 128 = 99,436 %,
alors que son sparse seul — correspondant à la famille V4.12 — obtenait
2 059/2 128 = 96,758 %. Le résultat final ne peut donc pas être hérité.
L'inventaire confirme que les 23 609 lignes CRM locales sont toutes
consommées : 23 384 par historique/V4-Fresh puis les 225 restantes par le
challenge V4.11. Aucun nouvel export CRM local n'existe depuis le registre du
28 juillet. Rapport :
`reports/v9/v4_12_retrieval_final_evidence_decision.md`. V4.12 reste
candidat grâce à son GO dev, mais toute certification finale exige un nouvel
export CRM indépendant ; ranker et accepteur restent gelés.

Le contrat V4.11-B est préenregistré avant tout nouveau dataset ou fit.
Il corrige la frontière produit : le SIRET/SIREN historique du CRM devient
une étiquette cachée et ne peut plus alimenter le retrieval, le ranker ou
l'accepteur. Un diagnostic de sensibilité montre que 5 882/5 883 exacts
restent dans le sparse et au top-1 après masquage des signaux directs. V4.11
reconstruira donc un vrai top-100 sparse sans branche identifiant, entraînera
un ranker C de 45 features sur ces pools, puis comparera exactement deux
accepteurs sur une scène compacte de 80 features. Le dev historique reste
développement uniquement ; les 225 lignes inédites restent fermées jusqu'au
gel du candidat. *(commit GitHub : `ca83603`)*
L'audit indépendant pré-fit a ensuite fermé les ambiguïtés restantes :
`UNRESOLVED` est exclu des cibles, les 80 formules/types/contraintes sont
définis sans score absolu inter-fold, les deux moitiés dev ont des volumes
attendus, les baselines sont épinglées et les seuils utilisent une règle
entière déterministe. *(commit GitHub : `399252f`)*
Le calcul de scène V4.11 partagé train/serve est implémenté et testé : ordre
exact de 80 features, 34 binaires et 46 standardisées, vecteur monotone
`49/+`, `6/-`, `25/0`, normalisation des scores par requête, tie-break et cas
0/1 candidat. La suite complète passe 395 tests. Aucun modèle n'a été
entraîné. *(commit GitHub : `c7075ac`)*
L'audit d'intégration des scènes a fermé un échec silencieux avant tout fit :
les cinq champs SIRENE nécessaires aux rôles/NAF sont maintenant obligatoires,
un NAF inconnu n'est plus un faux accord, et plafond/rangs/SIRET sont validés
strictement. La suite passe 402 tests avec le builder retrieval en cours.
*(commit GitHub : `58c70f4`)*
Le contrat transporte désormais explicitement les cinq champs SIRENE bruts
nécessaires aux rôles/NAF *(commit GitHub : `f1bdcdd`)*. Le builder V4.11
input-blind est implémenté et audité `GO build` : vraie requête sparse sans
argument SIRET/SIREN/vérité, top-100 actif, vérité jointe seulement après
fermeture de tous les pools, 45 features ranker et cinq champs de rôle issus
du snapshot. Un smoke réel produit 100 candidats et un NAF réel sans colonne
interdite ; 402 tests passent. Aucun fit n'a encore eu lieu.
*(commit GitHub : `3149d04`)*
Le premier lancement complet s'est arrêté en préflight avant toute requête :
les splits V4.6 attribuent un numéro de pli aux lignes dev aussi. Le
garde-fou a été corrigé pour valider les cinq plis gelés et l'unicité du pli
par composante sur les 7 003 lignes ; 403 tests passent. Aucun pool, label ou
résultat n'a été produit par cette tentative. *(commit GitHub : `734dc24`)*
Le lancement suivant a été interrompu proprement avant toute ouverture des
labels lorsque le RSS a dépassé 7,6 Go : le builder rescannait le snapshot
SIRENE à chaque requête et gardait un cache TF-IDF non borné. Le chemin
corrigé écrit d'abord les pools bruts aveugles, borne le cache RAM à 20,
hydrate état et rôles par une jointure bulk unique, puis recalcule les 45
features avant de fermer le top-100 final. Les identifiants CRM sont exclus
dès la projection Parquet ; les caches disque sont liés aux partitions et
vérifiés par SHA-256 avant désérialisation ; labels et baseline ne sont
hashés/lus qu'après fermeture du pool. Un oracle indépendant confirme la
parité exacte avec l'algorithme précédent, y compris IDF par défaut, fermés,
tie-break et enseignes divergentes. Deux audits ont conclu `GO`, un smoke
réel produit 100 candidats/59 colonnes avec un seul scan, et 437 tests
passent. Aucun résultat retrieval n'a encore été produit. *(commit GitHub :
`fc8c848`)*
Le build complet V4.11 franchit désormais
**`GO_TRAIN_INPUT_BLIND_RANKER`**. Sur le dev historique, le bon SIRET est
présent dans 1 217/1 217 pools à 100 ; sur le fit, dans 4 665/4 666
(99,9786 %), avec l'unique miss `6818`. Les 7 003 requêtes ont toutes un
pool, le plafond maximal est exactement 100 et les 698 892 candidats sont
actifs, uniques et sans injection. Une recomputation indépendante confirme
les hashes, les rangs, les cibles, l'absence des identifiants CRM et le blob
Git du builder. Le dev reste un jeu de développement consommé ; ce `GO`
autorise le ranker C, pas la preuve produit finale. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_input_blind/ec4326ec57e4411d`.
Rapport : `reports/v9/v4_11_input_blind_retrieval_results.md`.
*(commit GitHub : `db5d233`)*
Le runner du ranker C est maintenant implémenté, sans l'avoir encore
exécuté : cinq modèles OOF scorent les scènes fit, un modèle complet score le
dev, les misses retrieval restent des erreurs end-to-end et chaque fit est
rejoué deux fois avec égalité exacte exigée. Le diagnostic ranker B masqué
reconstruit correctement le canal sparse unique (`channel_count=1`) et son
score RRF à partir du rang. Le runner ne sera lancé que si le build
input-blind franchit d'abord le gate Recall@100. *(commit GitHub :
`b6a2332`)*
Le ranker C franchit maintenant **`GO_RANKER_C`** : 4 661/4 666
(99,8928 %) au top-1 SIRET OOF fit et 1 216/1 217 (99,9178 %) sur le dev,
avec modèles, scores et rangs reproduits bit à bit. L'unique vérité absente
du pool, `6818`, reste une erreur end-to-end ; l'unique erreur dev est
`13958`. Un premier artefact correct sur les scores a été superseded avant
promotion parce que son compteur `retrieval_miss` désignait les pools vides
et que la description du canal sparse ne correspondait pas à la matrice. Le
runner corrigé distingue pool vide et vérité absente, puis documente
`channel_count=1`; l'audit indépendant conclut `GO` vers l'accepteur.
Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_ranker_c/e13eb3ac7498256e`.
Rapport : `reports/v9/v4_11_ranker_c_results.md`.
*(commits GitHub : correctif `d2a6f5b`, résultats `c9f16c4`)*
Le dataset et le runner de l'accepteur compact sont aussi implémentés, mais
restent inactifs jusqu'aux gates précédents. Ils imposent le rattachement à
un artefact ranker C `GO` réellement OOF, l'étanchéité des composantes
train/dev, les 5 547 scènes fit et les volumes dev préenregistrés. Les
`UNRESOLVED` sont exclus du fit et matérialisés en `REVIEW`; un éventuel
bundle `GO` épinglera par hash le retrieval, le ranker C, la taxonomie, le
contrat, le calcul de scène et l'accepteur. Le contrôle indépendant a trouvé
puis fait fermer cinq manques de gouvernance avant tout fit ; la suite
complète passe 428 tests. *(commit GitHub : `2a9f51f`)*

Le dataset de scènes de l'accepteur V4.11 franchit maintenant
**`GO_FREEZE_PLAN`**. Il contient 5 547 scènes fit produites par prédictions
OOF et 1 456 scènes dev hors échantillon, soit 7 003 requêtes, 80 features et
5 877 cibles positives. Les cinq erreurs fit, l'erreur dev et les 1 120 cas
`AMBIGUOUS` restent explicitement négatifs ; aucun cas n'a été retiré. Un
premier manifeste a été supersédé avant tout fit car il n'épinglait pas le
code transitive de fonction de site. Le build corrigé verrouille retrieval,
ranker, prédictions, contrat, taxonomie, calcul de scène et fonction de site.
Son parquet est bit à bit identique au premier et le contre-audit conclut
`GO_FREEZE_PLAN`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_acceptor/52ea3faba9a56aff`.
Rapport : `reports/v9/v4_11_acceptor_scene_dataset_results.md`.
*(commits GitHub : correctif `c462a21`, résultats `19f1169`)*

Le plan d'entraînement de l'accepteur V4.11 est gelé avant tout fit. Il
autorise exactement une logistique compacte et un XGBoost monotone peu
profond, avec leurs hyperparamètres fixes, les 80 features dans leur ordre,
les trois populations étanches et la sélection à 99,8 % de précision,
80 % de couverture et zéro `AMBIGUOUS` automatisé. Le verrou d'exécution lie
ce plan au runner et aux sources commités ; préflight et contre-audit
concluent `GO`. Aucun challenge, holdout, unseen ou test final n'a été ouvert.
*(commits GitHub : plan `8033934`, verrou `fd70a64`)*

Le développement de l'accepteur V4.11 conclut
**`GO_FREEZE_V411_CANDIDATE`**. La logistique compacte gagne au seuil
`0.8720916706888049` : 614/746 AUTO, 614 corrects, zéro ambigu automatisé,
soit 82,306 % de couverture et 100 % de précision observée sur
`comparison_dev`. Le XGBoost monotone obtient 612/746 sans erreur. La
baseline obtient 618/746 avec une erreur. Deux audits recomputent à
l'identique modèles, seuils, métriques, familles et bundle. Ce GO autorise
uniquement le gel et le challenge descriptif des 225 cas : le ranker était
déjà exact sur les 634 labels exacts de comparaison, la borne basse Wilson
95 % du 614/614 n'est que 99,378 %, et le dev est historique. Il ne s'agit
donc ni d'une certification 99,8 % ni d'une promotion produit. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_acceptor/9d23bf3deb6b63de`.
Rapport : `reports/v9/v4_11_acceptor_development_results.md`.
*(commit GitHub : `f99c1d1`)*

Le challenge descriptif V4.11 est préenregistré avant qualification et
inférence. Une inspection du registre a exposé à l'orchestrateur le SIRET CRM
de trois lignes ; elles restent dans l'unique run mais sont exclues de la
métrique aveugle principale, qui portera sur 222 lignes, et publiées dans une
cohorte `EXPOSED_3`. Le CRM sera projeté physiquement sans identifiant, les
labels seront produits mécaniquement par la politique V4 gelée et hashés
avant toute inférence, puis les prédictions seront scellées avant ouverture
des labels. Le challenge reste descriptif et ne constitue aucun gate produit.
*(commit GitHub : `30fa0b8`)*

Les builders du challenge V4.11 sont implémentés et audités avant ouverture :
projection CRM physique, mapping scellé, qualification mécanique V4, preuves
et labels immuables avec validateurs fail-closed. La suite passe 451 tests.
Le docket assaini est maintenant construit avec 225 lignes et exactement
sept colonnes CRM ; aucun SIRET/SIREN, fingerprint, label, candidat ou score
n'est présent. Les cohortes contiennent 222 lignes aveugles et trois lignes
exposées. Le contre-audit conclut `GO_QUALIFY`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_sanitized/1c994c852c10acaf`.
*(commit GitHub : `1fc058f`)*

La qualification mécanique du challenge est gelée avant toute prédiction :
74 `MATCH_EXACT`, 17 `AMBIGUOUS` et 134 `UNRESOLVED`. La cohorte aveugle
principale contient 73 exacts sur 222 lignes ; la cohorte exposée un exact
sur trois. Les 138 preuves actives, leurs cardinalités et les identifiants
exacts sont cohérents ; aucun `NO_MATCH`, secours web, modèle ou score n'a
été utilisé. Le contre-audit conclut `GO_FREEZE_LABELS`. Cette population
n'est identifiable qu'à 32,889 % et reste descriptive. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_qualification/4f9ef46516b89ab8`.
Rapport : `reports/v9/v4_11_unseen_qualification_results.md`.
*(commit GitHub : `6b84597`)*

Le runner du challenge descriptif unique est maintenant commité et verrouillé,
sans avoir ouvert ni scoré les 225 cas. Il impose un ledger global indépendant
du répertoire de sortie, scelle et re-hashe les 225 prédictions avant toute
désérialisation des labels, contrôle exactement les populations 222/3 et
sépare erreurs confirmées, AUTO invérifiables et couverture des seuls
`MATCH_EXACT`. Les cinq scripts d'orchestration, tous les modules
`src/xgb_matcher`, les modèles, données et versions runtime sont épinglés par
hash et commit. Deux audits concluent `GO_COMMIT_RUNNER`; 462 tests passent et
la parité historique est bit-exacte sur 1 456 requêtes et 145 236 candidats,
avec cinq contrôles exacts. Le prochain acte autorisé est l'unique exécution
descriptive sous ce verrou. *(commits GitHub : runner `cd1cab5`, verrou
`da6924a`)*

L'unique challenge descriptif V4.11 est terminé avec une intégrité
contre-auditée, et conclut **`PIVOT_ACCEPTOR_EVIDENCE_GATE`**. Sur les 222
lignes aveugles, le retrieval et le ranker réussissent les 73/73 cas exacts,
mais l'accepteur automatise un cas `AMBIGUOUS` : 73/74 décisions AUTO
évaluables sont correctes, soit 98,649 %. Les 72 autres AUTO aveugles sont
`UNRESOLVED` et restent invérifiables. L'erreur contient deux candidats forts
de deux SIREN différents, tous deux dans le top 2 ; la scène ne compte pas
explicitement les identités fortes inter-SIREN. La prochaine variante doit
donc tester une garde déterministe « plusieurs SIREN forts → REVIEW » et les
features correspondantes sur les anciennes populations, puis être gelée
avant un nouvel export. Il est interdit de régler le seuil sur ce challenge
consommé. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_execution/ddb7336e8c2e042d`.
Rapport : `reports/v9/v4_11_unseen_execution_results.md`.
*(commit GitHub : `62e9741`)*

Le contrat V4.12-G est préenregistré avant tout nouveau build. Retrieval,
Ranker C, accepteur V4.11 et seuil restent gelés ; une garde déterministe
n'autorise AUTO que si l'univers géographique actif contient exactement un
candidat direct fort, égal au top-1 déjà accepté. La garde est un veto pur et
ne peut ni choisir ni injecter un candidat. Une allowlist par chemin, hash,
phase et projection limite le développement aux artefacts historiques ; les
trois racines du challenge consommé et tous leurs outputs sont interdits par
hash. Avant le seal des preuves, seules les queries et partitions sont
ouvrables. Le contrat distingue les gates retrieval (couverture exacte et
Recall@100) des gates de décision (couverture AUTO et précision), documente
la circularité des labels mécaniques et exige un nouvel export indépendant.
Deux audits concluent `GO_CONTRACT`. Aucun build V4.12 n'a encore été lancé.
Le gate de performance hors-ligne est clarifié : le temps moyen par requête
sur un batch n'est pas assimilé à une latence de service. Le constructeur
label-free est désormais implémenté et contre-audité : il parcourt l'univers
géographique actif complet, produit une preuve par requête et par candidat,
refuse tout artefact non autorisé et scelle ses sorties de façon atomique.
Les 50 tests V4.12 ciblés et les 512 tests complets passent. Son exécution est
gelée par un verrou audité `GO_COMMIT_LOCK`, qui fixe le commit, les 53
sources, les entrées et le runtime. Le calcul sur les 7 003 requêtes n'a
été autorisé qu'après ce gel. *(commits GitHub : contrat `66e7b9c`,
clarification `31f2721`, constructeur `e822136`, verrou `11c5de9`)*

Le build label-free V4.12 est désormais scellé et contre-audité
**`GO_SEALED_EVIDENCE`**. Sur 7 003 requêtes, 5 883 (84,007 %) ont exactement
un candidat direct actif et 1 120 (15,993 %) en ont plusieurs : 977 collisions
inter-SIREN et 143 cas multisites intra-SIREN. Il n'existe aucun cas sans
preuve directe. Les 10 275 preuves candidates sont actives, uniques par
requête/SIRET et reliées bijectivement aux agrégats. Le pic RSS est de
2,99 Go. Aucun label, challenge, pool ranker, scène ou modèle n'a été ouvert
avant le seal. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_direct_evidence/10f16403795ccee6`.
Rapport : `reports/v9/v4_12_direct_evidence_build_results.md`. La prochaine
étape est le runner post-seal audité, puis l'unique gate historique
`comparison_dev`; aucune promotion n'est encore autorisée.
*(commit GitHub : `3aff8d9`)*

Le runner post-seal V4.12-G est implémenté et contre-audité
**`GO_COMMIT_EVALUATOR`**. Il recalcule les trois populations, reproduit la
baseline V4.11 `614 AUTO / 0 erreur / 0 ambiguïté`, applique un veto pur et
publie les gates entiers et segmentaires. Sa publication est fermée par Git,
hashes, TOCTOU, RSS, fsync et validation sémantique ligne à ligne. Les 23
tests ciblés et les 535 tests complets passent. Le verrou externe audité fixe
11 sources, 13 entrées, le seal V4.12, le modèle, le seuil et le runtime.
*(commits GitHub : évaluateur `37f1476`, fermeture source `0182248`, verrou
`a99dd31`)*

Le gate historique V4.12-G est exécuté, validé et contre-audité
**`GO_V412_HISTORICAL_GATE`**. Sur `comparison_dev`, V4.11 et V4.12-G
conservent 614/746 AUTO, tous exacts, sans ambiguïté automatisée et sans perte
sur aucun des onze segments publiés. Hors gate, la garde retire trois erreurs
`AMBIGUOUS` du fit et une du threshold dev, toutes dues à deux preuves fortes
inter-SIREN, sans retirer d'AUTO exact. Deux préflights se sont arrêtés avant
scoring et sans artefact sur des conventions valides du fit/dev ; les
correctifs et reverrouillages ont été audités avant l'exécution publiée.
Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/evaluations/v4_12_guard_historical/fedcd1d512bfd269`.
Rapport : `reports/v9/v4_12_guard_historical_results.md`. Ce GO reste un
contrôle de cohérence sur des labels circulaires :
`latency_gate_evaluated=false` et `production_certified=false`. La suite est
le gel du bundle, la parité batch/inférence et la latence appariée, puis un
nouvel export indépendant. *(commits GitHub : correctifs `7d70249`,
`e8b052f`, reverrouillages `9452d0f`, `3e3dabc`, résultat `94414af`)*

Le préalable d'inférence V4.12 est préenregistré et contre-audité
**`GO_CONTRACT_LOOKUP`**. V4.11 hydrate aujourd'hui le snapshot SIRENE en
bulk ; aucune p95 par requête honnête n'est donc encore mesurable. La brique
suivante est un DuckDB local en lecture seule contenant exactement les sept
colonnes utilisées par l'hydratation historique, indexées par SIRET. Le
contrat fixe le snapshot de 42 322 035 SIRET uniques, les 698 892 candidats
historiques (508 081 SIRET uniques), un contrôle indépendant de 10 000 SIRET,
les limites Mac/SSD et l'API fail-closed. Aucun lookup n'a encore été
construit. Après parité exacte seulement, un moteur persistant pourra mesurer
la p95 appariée sur les 1 456 requêtes dev.
*(commit GitHub : `00ce1c3`)*

Le builder et le store lookup V4.12 sont implémentés et contre-audités
**`GO_COMMIT_LOOKUP_BUILDER_FINAL`**. Une mini-publication exerce réellement
DuckDB, l'index, la référence/parité, RSS/disque, WAL, fsync, renommage
atomique et revalidation ; les tests de falsification ferment verrou Git,
provenance, TOCTOU et schémas imbriqués. Les 60 tests ciblés et 595 tests
complets passent. Le verrou réel audité fixe sept sources, quatre entrées,
DuckDB 1.4.3 et la racine SSD ; environ 1 049 Gio sont libres et la cible est
absente. Le build réel de 42 322 035 lignes n'a pas encore été lancé.
*(commits GitHub : builder/store `a06cf00`, verrou `591f339`)*

Le build réel du lookup franchit **`GO_V412_SNAPSHOT_LOOKUP`** :
42 322 035 SIRET uniques, zéro invalide, index et lecture seule conformes,
zéro écart sur les 508 081 candidats V4.11, sample indépendant conforme et
pic RSS de 7,8004 Gio. Un premier contre-audit avait publié à tort
`STOP_V412_LOOKUP_PARITY` : ses trois commandes hashaient les deux caractères
littéraux antislash et `n` (`160 000` octets), produisant `72f43460...`.
Avec le véritable octet LF demandé par le contrat, le payload fait
`150 000` octets et reproduit bien `58c970...`. L'incident reste documenté
et un contre-validateur distinguant explicitement les deux encodages sera
ajouté avant le benchmark. L'artefact
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/v4_12_snapshot_lookup/ff0f33ad10803cfb`
autorise désormais la construction du moteur d'inférence, pas la production.
Rapport :
`reports/v9/v4_12_snapshot_lookup_results.md`.
*(commits GitHub : faux STOP conservé `880e57c`, correction `00d71c4`)*

Le contre-validateur indépendant du lookup est préenregistré
**`GO_CONTRACT_INDEPENDENT_AUDIT`** avant son implémentation. Il refait la
sélection SIRET depuis le snapshot sans importer le builder, construit le
payload avec `byte 0x0A`, vérifie explicitement le contre-exemple `5C 6E`,
rejoint séparément les six valeurs métier et compare le store par lots de
100. Schéma, cardinalité, index, quatre fichiers, ressources, sources
transitives et publication sont gelés. Un test devra modifier puis resceller
une valeur DuckDB : le validateur historique peut l'accepter, le nouveau doit
la refuser. Aucun audit formel n'a encore été exécuté.
*(commit GitHub : `5234084`)*

Le contre-audit formel franchit
**`GO_V412_LOOKUP_INDEPENDENT_AUDIT`**. Le runner, ses 23 tests ciblés et la
suite complète de 618 tests ont été contre-audités avant verrouillage. Le
run `4055be6e7a11b003` recalcule 10 000 SIRET depuis le snapshot :
vrai LF `150 000` octets/hash `58c970...`, contre-exemple `5C 6E`
`160 000` octets/hash `72f434...`. La comparaison fraîche donne zéro absent
et zéro écart de valeur/nullité ; le pic RSS est de 4 365 549 568 octets.
Le test hostile confirme qu'une valeur métier modifiée puis rescellée,
acceptée par le validateur historique, est refusée par le nouveau. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_snapshot_lookup/4055be6e7a11b003`.
Rapport :
`reports/v9/v4_12_snapshot_lookup_independent_audit_results.md`. Le lookup
autorise désormais le contrat du moteur requête par requête, pas la
production. *(commits GitHub : runner `3de02dc`, verrou `dd696de`, résultat
`1653175`)*

Le premier jalon du moteur unitaire est préenregistré
**`GO_CONTRACT_INPUTS`** après refus d'un contrat monolithique insuffisamment
séparé. Ce jalon ne calcule aucun match : il doit publier, dans une racine
runtime sans chemin sensible, les six champs CRM sûrs pour 7 003 requêtes et
les 1 456 dev, plus les inventaires cryptographiques des 4 119 partitions et
1 454 paires cache TF-IDF. Les hashes de contenu attendus sont gelés à
`680f1884...5463` et `589360b1...83ce`. Une racine d'audit distincte scellera
les 7 029 fichiers ouverts ; elle ne sera jamais transmise au worker. Ce GO
autorise seulement le builder/tests des inputs, pas l'oracle, le store, le
worker ou le benchmark. *(commit GitHub : `5ebd9de`)*

Le registre V4.11-A des populations consommées est construit et franchit
**`PASS_REGISTRY`**. Le benchmark fermé historique couvre 17 054 lignes
source et le pool V4-Fresh 6 330, sans recouvrement : leur union a déjà
consommé 23 384 des 23 609 lignes de `data/entrainements.csv`. Les 225 lignes
restantes ont toutes un `SERVICE ID` absent et ne peuvent pas constituer une
validation représentative. Elles sont réservées à un challenge descriptif
après gel de V4.11 ; une preuve finale exige un nouvel export CRM indépendant.
Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/registries/v4_11_consumed_population/fd25d1922040d585`.
Rapport : `reports/v9/v4_11_consumed_population_registry_results.md`.
*(commits GitHub : contrat `d0eb5f3`, builder `0aa8ad2`)*

Le développement V4.10b se termine par
**`PIVOT_STRUCTURED_FEATURES`**. Aucune des six variantes structurées ne
franchit le gate. Les logits refusent seulement 16 à 18 des 25 mauvais cas et
automatisent l'unique ambigu ; les XGBoost en refusent 20 à 21, mais perdent
des bons cas ou la non-infériorité historique. `CURRENT80_W1` reste le plus
proche avec 23/25 mauvais refusés, sous le minimum de 24. Les 54 fits logiques
ont été répétés sans aucun écart, les deux audits indépendants confirment
hashes, seuils, gates et populations. Aucun bundle ni fresh dev n'est
autorisé. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_10b_structured_acceptor_development/71e067f75536180b`.
Rapport : `reports/v9/v4_10b_acceptor_development_results.md`.
*(commits GitHub : plan `5ed1ba3`, runner `6ae4cf7`, verrou `fb33c76`)*

Le dataset corrigé V4.10b est construit et franchit
**`GO_FREEZE_TRAINING_PLAN_V410B`**. Le nouveau catalogue autorise 641
features structurées : 157 continues/comptages à standardiser et 484
binaires non standardisées. Les 58 alias sont vérifiés ligne par ligne, les
16 signaux d'instrumentation retrieval et les 75 signaux de provenance sont
hors modèle. Les trois parquets restent identiques au build V4.10 et
`CURRENT80` est bit à bit inchangé. Aucun fit ni seuil n'a été produit.
Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_10b_structured_acceptor/3ad8e97ce0118e8c`.
Rapport : `reports/v9/v4_10b_structured_dataset_results.md`.
*(commits GitHub : politique `eb85597`, clarification `d500fe2`, builder
`f78a9ba`)*

Le plan d'entraînement V4.10b est désormais gelé avant tout fit. Il conserve
trois variantes `CURRENT80` comme contrôles non promouvables et compare six
variantes structurées. Les facteurs de classe, seuils par pli, gates entiers,
54 fits logiques rejoués deux fois, lectures filtrées et bundles multiples
sont préenregistrés. Un verrou externe devra encore épingler le runner
commité avant le premier entraînement. *(commit GitHub : `5ed1ba3`)*

L'audit statistique pré-fit a invalidé l'ordre structuré V4.10 avant tout
entraînement. Il contenait 58 copies sémantiques et 16 signaux résiduels
capables de distinguer l'instrumentation retrieval V4.1 de V4.2-B. La
politique V4.10b, préenregistrée sans utiliser les labels ni les splits,
conserve `CURRENT80` bit à bit, ramène l'ordre structuré de 715 à 641
features, standardise aussi les compteurs pour la logistique et précise les
gates en arithmétique entière. Le build `0d6b87fd50fb550c` et son ancien plan
sont `superseded`; aucun fit ne les a consommés. *(commit GitHub : `eb85597`)*

Le plan d'entraînement V4.10 est gelé avant le premier fit dans
`config/v4_10_training_plan.json`. Il autorise exactement `BASE_FROZEN` et
neuf variantes appariées (`CURRENT80`, `STRUCTURED_LOGIT`,
`STRUCTURED_XGB`, poids difficiles 1/2/4), cinq plis difficiles group-OOF et
une sélection de seuil uniquement sur les 1 452 scènes du dev historique
effectif. Les cas random, frais, descriptifs verrouillés et le test final
restent interdits au fit, au seuil et au gate. Un audit indépendant pré-fit a
ensuite fermé les ambiguïtés de sélection et de reproductibilité sans ouvrir
ni scorer aucune donnée : `BASE_FROZEN` est seulement comparateur, le parquet
verrouillé est hashé mais jamais chargé sémantiquement, et chaque modèle
complet ou de pli doit être reproduit. *(commits GitHub : `47ff289`,
amendement pré-fit `dd0e3c8`)*

Le dataset V4.10 de l'accepteur structuré est construit et franchit
**`GO_TRAIN_V410`**. Il contient 7 003 scènes historiques, 94 cas difficiles
`hard_oof` et quatre cas descriptifs verrouillés. Les 80 features baseline
sont identiques bit à bit aux sources ; 715 features structurées sont
autorisées au modèle et 75 features de provenance restent audit-only. Les
698 428 paires prédiction/candidat V4.1 se joignent exactement, les jointures
CRM et SIRENE utiles sont à 100 %, les 20 supports de composantes sont
préservés et aucun ID random V4.8 n'entre dans les sorties. Aucun modèle ni
seuil n'a encore été entraîné. Rapport :
`reports/v9/v4_10_structured_dataset_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_10_structured_acceptor/0d6b87fd50fb550c`.
*(commits GitHub : contrat `bc2384c`, `8ab9b01`, `1401269`, `99b2438`,
`2d86b5c`, `b19abed`; builder `2966d2b`; correctif `e10e9af`)*

L'audit V4.10 distingue désormais les 31 labels négatifs non détectés par le
garde lexical des véritables faux AUTO `HARD_W1` : sur 26 cas hors pli,
`HARD_W1` n'en automatise que deux ; les trois cas random de ce sous-ensemble
sont tous refusés. Les 31 se répartissent en 14 mauvais sites au sein du même
SIREN, 14 autres personnes morales à la même adresse, deux acteurs affiliés
ou support et un CRM composite. Le flux explique ces erreurs : l'accepteur
perd 47 des 64 features candidat du ranker, ne voit pas l'activité SIRENE et
réduit les frères d'un même SIREN à quelques agrégats. La prochaine
architecture sera donc un accepteur query-level unique enrichi, sans nouveau
veto ni modification retrieval/ranker. Rapport :
`reports/v9/v4_10_error_and_feature_flow_audit.md`.

La V4.9 se termine avec **`STOP_SITE_FUNCTION_GUARD`**. La taxonomie
déterministe, gelée avant mesure, refuse les trois erreurs random V4.8
mairie/école, maternelle/primaire et FAM/MAS, sans refuser aucun des 116
top-1 corrects. Elle ne couvre cependant que 3 des 34 top-1 faux ou ambigus
fiables, sous le minimum préenregistré de cinq. Aucune cohorte fraîche n'est
donc ouverte et la taxonomie ne sera pas enrichie après observation. Aucun
modèle ni seuil n'a changé et le test final reste fermé. Rapport :
`reports/v9/v4_9_site_function_guard_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_9_site_function_retrospective/30e22eae11620538`.
*(commits GitHub : contrat `169d9cf`; taxonomie `a311306`; entrées épinglées
`49832e4`; évaluateur `67b2cb5`)*

La V4.8 se termine avec le verdict final **`STOP_RETRAIN`**. L'ouverture
random unique invalide `HARD_W1` : 47/52 AUTO, mais seulement 44 corrects,
soit trois erreurs et 93,617 % de précision observée. Le baseline gelé fait
43/45 = 95,556 % avec deux erreurs. Le winner automatise deux
`TOP1_WRONG` et l'unique `AMBIGUOUS`; les deux gates de sécurité échouent.
Les trois faux AUTO ont des scores de 0,980 à 0,999 et confondent la fonction
exacte de sites très proches : mairie/école, maternelle/primaire et FAM/MAS.
Ce n'est donc pas un simple problème de seuil. Le registre global empêche
toute réouverture du random ; le test final reste fermé et aucun modèle n'est
promu. Rapport : `reports/v9/v4_8_random_holdout_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_8_random_holdout/f1ac35f4f7450b6a`.
*(commits GitHub : contrat `b738ec5`; ouvreur `685ebae`; préflight
`ba4377f`)*

Le développement V4.8 retient **`HARD_W1`** et autorise l'ouverture unique
du random avec le statut **`GO_RANDOM_OPEN_V48`**. Sur 94 cas difficiles hors
pli, il rejette 23/25 mauvais top-1 contre 13/25 pour `BASE_REFIT`, soit dix
erreurs supplémentaires, tout en gardant 58/68 bons AUTO contre 61/68
(-4,412 points, limite -5). Sur le dev historique effectif, il produit
1 184/1 186 AUTO corrects = 99,831 % observés et 81,680 % de couverture,
sans erreur supplémentaire. Le modèle complet et son seuil
`0.3617231974526733` sont gelés. Aucun random n'a été lu ou scoré et le test
final est resté fermé. Rapport :
`reports/v9/v4_8_acceptor_development_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_8_acceptor_development/f2ea5be7c1a40647`.
*(commits GitHub : contrat/partitions `a15dd07`; runner `3f4671b`;
correctif de lecture `dab961d`)*

La V4.8 a préenregistré puis gelé ses partitions avant tout score accepteur.
Sur 98 labels ciblés fiables, 94 restent évaluables hors pli : 68 top-1
corrects, 25 mauvais et un ambigu. Quatre autres cas fiables sont
`hard_dev_locked` et seront seulement descriptifs. Les 57 cas random sont
tous scellés, leurs cibles sont absentes de l'artefact de partition et 48
scènes historiques reliées ont été exclues. Le fit V4.1 réellement éligible
est bien de 5 545 scènes, pas 5 547 ; le dev effectif futur en contient
1 452 après isolement random. Aucun modèle n'a été chargé, scoré ou entraîné
et le test final est resté fermé. Rapport :
`reports/v9/v4_8_acceptor_partition_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_8_acceptor_partitions/1c78764d5263afca`.
*(commits GitHub : contrats `f56472b`, `1ca9648`, `b63f383`; constructeur
`6bb8518`; correctifs de préflight `08018f9`, `eedac96`)*

La V4.7 a réadjudiqué les 37 top-1 ayant dérivé entre V4.4 et le stack
courant V4.2-B + ranker A, sans transporter l'ancien verdict. Chaque preuve
publique a été téléchargée, archivée et contrôlée par des faits
préenregistrés ; une décision fiable exige toujours au moins deux groupes
indépendants, dont le registre officiel. Vingt-trois nouveaux verdicts sont
fiables (huit `TOP1_CORRECT`, quatorze `TOP1_WRONG`, une `AMBIGUOUS`) et
quatorze restent `UNRESOLVED`. Le corpus courant atteint exactement 150/172
labels fiables, dont 52/57 aléatoires, 28 négatifs ciblés et six négatifs
aléatoires. Tous les gates préenregistrés passent. Verdict :
**`GO_ACCEPTOR_FEASIBILITY`**. Il autorise une expérience V4.8 hors test, pas
un déploiement ni une revendication à 99,8 %. Aucun modèle n'a été entraîné et
le test final est resté fermé. Rapport :
`reports/v9/v4_7_current_top1_adjudication_results.md`. Artefact :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_7_current_adjudications/4cc5420fb5da0683`.
*(commits GitHub : contrat `25b881b`, docket `6af0e45`, registre `b85daf7`,
adjudication `bdfbadc`; 324 tests passants)*

La V4.6 a reconstruit deux fois, avec caches séparés, les pools V4.2-B des
7 003 requêtes historiques puis comparé le ranker A gelé à un ranker B
réentraîné sur ces pools. Les deux datasets contiennent exactement 698 991
candidats et partagent le même hash de contenu ; Recall@100 vaut 100 % sur fit
(4 666/4 666) et dev (1 217/1 217), sans doublon, candidat fermé, pool >100 ou
positif injecté. B atteint 1 216/1 217 = 99,918 % Hit@1 SIRET, contre
1 213/1 217 = 99,671 % pour A : trois corrections, zéro régression. Le gain
n'atteint toutefois ni les quatre corrections minimales, ni une borne
bootstrap strictement positive, ni McNemar `p<0,05` (`p=0,25`). Verdict
contractuel : **`KEEP_RANKER_A`**. B n'est pas promu. Aucun accepteur, seuil,
label V4.4/random ou test final n'a été utilisé. Rapport :
`reports/v9/v4_6_aligned_ranker_results.md`. Artefacts :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_6_aligned_a/301b24f47820f992`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_6_aligned_b/301b24f47820f992`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_6_aligned_ranker/421f2cd0cc436af7`.
*(commits GitHub : contrat `acfd4d2` et `70df9c9`, builder `a9439ed`,
correctif instrumentation `458dd97`, évaluateur `f2b6b9c`, `c94d100` et
`8b835aa`, rapport `67e9e76`; 316 tests passants)*

La V4.5 a vérifié si les labels V4.4 pouvaient être transportés vers les
scènes réellement produites par le retrieval V4.2-B et le ranker V4.1 gelé.
Verdict : **`PIVOT_SCENE_DRIFT`** et `training_authorized=false`. Sur 172
dossiers, 135 seulement conservent le même top-1 et 37 dérivent. Les gates
échouent avec 46/53 labels aléatoires fiables compatibles, 2/6 négatifs
aléatoires, 16/37 `TOP1_WRONG` ciblés et 1/5 `AMBIGUOUS` ciblé. Seul le
minimum des `TOP1_CORRECT` ciblés passe avec 64/67. Aucun accepteur n'a été
chargé, aucun seuil calculé, aucun modèle entraîné et le test final est resté
fermé. Rapport : `reports/v9/v4_5_scene_compatibility_results.md`. Artefacts :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_5_hard_scenes/21f8c0b0b172b907`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/gates/v4_5_scene_compatibility/5c8b87fd8e226157`.
*(commit GitHub : `5c1343e`; 296 tests passants)*

La V4.4 d'adjudication autonome est terminée. Les lots A–R et les
contradictions connues couvrent exactement les 172 `AUTO_MATCH` V4.3 : 162
décisions sont validées par au moins deux groupes de preuves indépendants,
dont 114 `TOP1_CORRECT`, 42 `TOP1_WRONG` et six `AMBIGUOUS`; dix restent
`UNRESOLVED`. Cinquante-trois décisions validées appartiennent au tirage
aléatoire. Verdict contractuel : **`STOP_AUTONOMOUS_LABELING`**. Les seuils
correct et random sont franchis, mais la population entière ne contient que 42
erreurs prouvées pour un minimum préenregistré de 50. Il est impossible de
combler les huit manquantes sans fabriquer des erreurs, abaisser le seuil
après observation ou ouvrir prématurément les `REVIEW`. Aucun modèle n'est
modifié. L'adaptateur recalcule les pools top-10
figés, refuse les hashes de preuve inexacts, interdit l'injection d'un positif
et reproduit les tables canoniques à partir des JSON revus. Les lots ont été
corrigés contre les archives réelles, notamment leurs hashes et les groupes
d'indépendance. La première passe est maintenant bloquée en code aux seuls
`AUTO_MATCH` V4.3 ; une tentative de sélectionner prématurément des `REVIEW`
est refusée. Rapport :
`reports/v9/v4_4_adjudication_gate_results.md`. Artefacts :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_official_evidence/87983e83c11f5284`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_evidence_facts/7ec4f63e1a22b082`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_sector_evidence/3149124f69dd7b1f`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_sector_facts/6a08bff403154884`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/320fe62322e14d25`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/70c65679dfb2c82d`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/1e2c68337408c453`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/925ef3f8ef3f3a4a`,
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_adjudications/2bfdc46480e52784`
et
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_gate/9fb43b4f7bb0919a`.
*(commits GitHub : lots vérifiés `f2aeec0`, adaptateur canonique `c6cb686`,
rapport `a66499e`, garde AUTO `4930521`, invariance `12a088b`, lots F/H
`d644e54`, G `1509841`, I `0e04eb3`, L `617c73c`, J/K `62669ac`, M
`d72c7d8`, N `30617b5`, O/P `65feba0`, Q/R `22f8ba2`; 283 tests passants)*

La V4.3 a transformé les 542 cas non résolus en une file d'adjudication
complète : 172 AUTO et 370 REVIEW, dont 144 cas du tirage aléatoire. La
première population AUTO est maintenant entièrement consommée par V4.4. Les
370 `REVIEW` restent fermés : ils ne peuvent pas être utilisés pour contourner
le gate AUTO. Aucun signal seul n'est converti artificiellement en vérité.

Le « gold standard » historique ne résout pas le problème. Il a été construit
en gardant le SIRET CRM si sa commune ou son CP correspondait à SIRENE, sans
validation humaine. Il recouvre 313/542 cas difficiles. Sur les 172 AUTO,
116 y figurent et 40 top-1 contredisent ce SIRET historique. Verdict
**`PIVOT_VALIDATION`**, statuts **`NO_RETRAIN`** et **`STOP_DEPLOYMENT`**.
Un premier lot de 250 dossiers avec preuves complètes est prêt dans l'artefact
V4.3. Rapport : `reports/v9/v4_3_hard_labels_results.md`.

Le correctif retrieval V4.2 franchit le gate représentatif :
**242/242 = 100 % Recall@100** sur les `MATCH_EXACT` provisoires figés, dont
91/91 dans le tirage aléatoire. Aucun pool ne dépasse 100, aucun candidat fermé
n'est produit, aucune vérité n'est injectée et aucun des 237 anciens succès ne
régrèsse. Les cinq pertes V4.1 sont récupérées aux rangs 1, 2, 1, 3 et 1.

Le correctif ne change ni TF-IDF, ni RRF, ni ranker, ni accepteur. La variante
B exploite le SIRET/SIREN d'entrée comme indice et la barrière finale prend
désormais l'état administratif dans le snapshot SIRENE complet de 42 322 035
établissements, au lieu du magasin rapide incomplet de 14 378 332 candidats.
Verdict **`GO_HARD_LABELS`** : le prochain travail est la constitution de
vrais cas difficiles représentatifs avant tout réentraînement. Le statut
production reste **`STOP_DEPLOYMENT`**. Rapport :
`reports/v9/v4_2_retrieval_integrity_results.md`.

L'audit représentatif V4.1 invalide désormais l'extrapolation des scores dev
au CRM réel. Sur 250 lignes tirées aléatoirement, la preuve déterministe ne
conclut que 106 cas = 42,4 % (91 exacts, 15 ambigus, 144 non résolus), alors
que V4.1 automatise 147 cas. Cinq contradictions AUTO manifestes ont été
documentées ; même en supposant tous les autres AUTO corrects, elles bornent
provisoirement la précision à 142/147 = 96,60 %. Ce n'est pas une estimation
certifiée : labels et contradictions restent `AI_PROVISIONAL`.

Le retrieval A ne conserve que 237/242 = 97,934 % des vérités exactes
provisoires de l'audit. B et C remontent à 240/242 = 99,174 % sans dépasser
100 candidats : elles récupèrent trois lignes grâce au SIRET/SIREN d'entrée.
Les deux pertes restantes sont trouvées par le sparse puis supprimées par le
magasin global d'état incomplet. Décision contractuelle **`PIVOT_LABELS`**,
décision opérationnelle **`STOP_DEPLOYMENT`**. Prochaine action : index
SIRET→état complet depuis le snapshot autoritaire, retrieval B, puis vrais
labels sur les cas difficiles avant tout réentraînement. Rapport :
`reports/v9/v4_1_representative_audit_results.md`.

La V4.1 actif-courant est maintenant implémentée et exécutée entièrement en
local. Le gate retrieval dev retient la variante A avec 305/305 bons SIRET à
Top-100, zéro candidat fermé et 872,6 ms de latence p95. Le dataset canonique
contient 7 003 requêtes et 698 428 paires, sans positif injecté et sans aucun
ID des tests consommés. Le ranker R1 atteint 99,918 % Hit@1 exact sur le dev ;
l'accepteur brut atteint 99,832 % de précision observée et 81,593 % de
couverture sur ce même dev.

Le shadow complet a ensuite scoré exactement 19 025 lignes autorisées et
produit 10 292 `AUTO_MATCH` = 54,097 % et 8 733 `REVIEW`, sans modifier le
CRM. Sa latence p95 est de 982,0 ms. Comme ce corpus n'est pas un test
indépendant et ne porte pas de nouveaux labels, aucune précision shadow n'est
publiée. Verdict de phase : **`PIVOT_CERTIFICATION`**. L'architecture est
techniquement validée, mais un `GO` production exige un prochain snapshot CRM
réellement nouveau et audité sans retuning. Rapport :
`reports/v9/v4_1_shadow_results.md`.

Le chantier **Retrieval sélectif SIRET Recall@100** est terminé avec une
décision contractuelle **`PIVOT`**. Sur le test final gelé, la qualification
V3 conserve 2 128/2 652 dossiers exacts, soit 80,241 % de couverture, et
l'admission gelée place le bon SIRET dans 100 candidats pour 2 116/2 128,
soit **99,436 % de Recall@100**. Tous les gates globaux passent. Le `PIVOT`
vient uniquement de deux gates de stabilité de couverture : établissements
fermés et mégapoles. Le test est désormais définitivement fermé à tout tuning.
Contrat : `docs/retrieval_selective_recall100_contract.md`. Rapport :
`reports/recall100/selective_test_certification.md`.

L'experience V9 sans GPU precedente est terminee avec une decision `PIVOT`.
Ses pools denses multicanaux ne sont pas promus. Le rapport reste
`reports/v9/v9_go_pivot_stop.md`.

V7/V8b et Route B restent physiquement disponibles comme baselines legacy.
Le gate retrieval n'a modifié aucun modèle aval. La phase aval est désormais
ouverte sous le contrat séparé
`docs/downstream_selective_matching_contract.md`, sans réutiliser le test
sélectif consommé.

La qualification V4 « SIRET actif au snapshot » est maintenant exécutée sans
score modèle et sans test. Elle corrige les cinq conflits bloquant E2b et
publie 4 060 exacts train / 872 exacts dev, tous actifs et soutenus par une
correspondance directe unique. Leur compatibilité avec le top-100 actuel est
excellente : 4 058/4 060 = 99,951 % train et 872/872 = 100 % dev. Le ranker E1
atteint 96,034 % / 94,954 % Hit@1 sur ce noyau. Mais V4 échoue à son gate
pré-enregistré : couverture ~34 % au lieu de 50 %, moins de 5 000 exacts train
et 14 SIREN actuels partagés entre les anciens splits. Verdict :
`STOP_V4` sur ce corpus seul, sans assouplissement post-hoc.

Ce blocage est désormais levé par **V4-Fresh** : 6 330 lignes CRM absentes du
benchmark ont été qualifiées avec la même règle, puis séparées par hash SIREN.
Elles ajoutent 819 exacts au fit, 305 au nouveau dev et 302 au holdout scellé.
Le fit combiné atteint **5 751 exacts**, avec zéro SIREN exact partagé entre
fit/dev/holdout. Verdict : **`PASS_V4_FRESH`**. Le holdout n'a reçu aucune
prédiction modèle. Rapport : `reports/v9/v4_fresh_expansion_results.md`.

Le gate retrieval V4 est maintenant franchi : l'admission déterministe gelée
conserve 5 749/5 751 = **99,965 %** des vérités du fit combiné et 305/305 =
**100 % observé** sur le nouveau dev indépendant, toujours avec 100 candidats
maximum. Les 1 124 cas frais ont tous leur vérité visible et conservée. Verdict
de phase : **`GO_RANKER_V4`**. Ce résultat autorise l'entraînement aval ; avec
305 cas dev, il ne constitue pas une garantie statistique de 99 % en
production. Rapport : `reports/v9/v4_retrieval_gate_results.md`.

Le ranker V4 est également validé sur ce nouveau dev : **299/305 = 98,033 %**
Hit@1 SIRET, contre 290/305 = 95,082 % pour l'ancien ranker compatible.
Il corrige dix erreurs et dégrade un ancien succès. Verdict de phase :
**`GO_ACCEPTEUR_V4`**. Le holdout reste fermé. Rapport :
`reports/v9/v4_ranker_e1_results.md`.

L'accepteur V4 franchit ensuite son gate : sur les 189 scènes de la moitié
`threshold`, il automatise **149/189 = 78,836 %** avec **149/149 correctes**.
Il rejette les 31 ambiguës, les quatre erreurs du ranker et cinq bons cas
incertains. Verdict : **`GO_HOLDOUT_V4`**. Les six variantes testées sont à
égalité ; le winner logistique + isotonic vient du tie-break déterministe, pas
d'une supériorité démontrée. Le holdout n'a toujours pas été lu. Rapport :
`reports/v9/v4_acceptor_e2_results.md`.

Le holdout V4-Fresh a ensuite été ouvert une seule fois après gel complet.
Le retrieval atteint **302/302 = 100 % Recall@100 exact**, et le ranker
**296/302 = 98,013 % Hit@1 exact**. L'accepteur automatise 282/354 scènes,
mais commet deux erreurs : précision **280/282 = 99,291 %**, sous le gate de
99,8 %. La qualification stricte ne couvre par ailleurs que 302/1 345 =
22,454 % de la source. Verdict final : **`PIVOT`**, sous-verdict
**`TECHNICAL_PIVOT`**. Le retrieval et le ranker sont validés ; la correction
porte sur le routage des scènes ambiguës, l'état actif/fermé du top1 et la
calibration saturante. Le holdout est désormais consommé. Rapport :
`reports/v9/v4_final_holdout_results.md`.

## Actions terminees (fenetre recente)
- **V4.8 arrêtée par le random unique** : `HARD_W1` commet trois erreurs sur
  47 AUTO, contre deux sur 45 pour le baseline. Verdict `STOP_RETRAIN`; aucun
  shadow ni déploiement. Les erreurs sont des confusions de fonction de site
  malgré des scores 0,98–0,999. Le random est définitivement consommé et le
  test final est resté fermé. Rapport :
  `reports/v9/v4_8_random_holdout_results.md`. *(commits GitHub :
  `685ebae`, `ba4377f`)*
- **Winner accepteur V4.8 gelé avant random** : `HARD_W1` rejette 23/25
  erreurs difficiles hors pli contre 13/25 pour le refit de base, en perdant
  trois bons AUTO. Le dev historique reste à deux erreurs et gagne deux bons
  AUTO. Statut `GO_RANDOM_OPEN_V48`; seuil gelé
  `0.3617231974526733`. Le random et le test final restent fermés. Rapport :
  `reports/v9/v4_8_acceptor_development_results.md`. *(commits GitHub :
  `3f4671b`, `dab961d`)*
- **Partitions V4.8 gelées avant modélisation** : 94 ciblés fiables sont
  disponibles en cinq folds groupés, avec exactement 25 erreurs et une
  ambiguïté. Les 57 random sont scellés sans cible exposée ; 48 scènes
  historiques reliées sont exclues. Le prochain gate compare uniquement des
  accepteurs logistiques à 80 features, avec seuil propre à chaque modèle
  OOF. Aucun score random ni test final n'a été consulté. Rapport :
  `reports/v9/v4_8_acceptor_partition_results.md`. *(commits GitHub :
  `b63f383`, `6bb8518`, `08018f9`, `eedac96`)*
- **Gate V4.7 franchi sur les scènes courantes** : 37/37 top-1 dérivés ont
  été traités ; 23 portent désormais un label fiable et 14 restent
  `UNRESOLVED`. Le corpus agrégé atteint 150/172 labels fiables, 52/57
  aléatoires, 28 négatifs ciblés et six négatifs aléatoires. Zéro ancien
  verdict a été transporté vers un autre SIRET. Verdict
  `GO_ACCEPTOR_FEASIBILITY`; V4.8 doit être préenregistrée avant tout
  entraînement et le test final reste fermé. Rapport :
  `reports/v9/v4_7_current_top1_adjudication_results.md`. *(commit GitHub :
  `bdfbadc`; 324 tests passants)*
- **Population AUTO V4.4 épuisée sans quota fabriqué** : les 172/172
  `AUTO_MATCH` ont été audités. Bilan : 114 `TOP1_CORRECT`, 42
  `TOP1_WRONG`, six `AMBIGUOUS`, dix `UNRESOLVED`, 162 labels acceptor et 53
  cas random validés. Les minima correct/random passent, mais le minimum de 50
  erreurs est impossible puisque la population entière n'en contient que 42
  prouvées. Verdict `STOP_AUTONOMOUS_LABELING`; aucun réentraînement sous le
  contrat V4.4 et aucun `REVIEW` ouvert. Le contrat expérimental V4.5 a été
  préenregistré avant entraînement et reste bloqué tant qu'un pivot explicite
  n'est pas adopté. Rapport :
  `reports/v9/v4_4_adjudication_gate_results.md`. *(commits GitHub : I
  `0e04eb3`, L `617c73c`, J/K `62669ac`, M `d72c7d8`, N `30617b5`, O/P
  `65feba0`, Q/R `22f8ba2`, contrat V4.5 `70cf70f`; 283 tests passants)*
- **Sous-gate random V4.4 franchi avec les lots F–H** : les 36 nouveaux cas
  sont tous des `AUTO_MATCH` V4.3 et ne chevauchent aucun dossier antérieur.
  Ils ajoutent 20 `TOP1_CORRECT`, 14 `TOP1_WRONG`, 13 random validés et deux
  `UNRESOLVED`. Le corpus atteint 89 cas, dont 81 validés : 55 corrects, 25
  incorrects et 32 random. Le gate reste `PIVOT_MORE_EVIDENCE`, avec déficits
  réduits à 20 corrects et 25 incorrects. Aucun SIRET alternatif non doublement
  prouvé n'a été créé. *(commits GitHub : F/H `d644e54`, G `1509841`,
  invariance des lots `12a088b`; 282 tests passants)*
- **Lots V4.4 A–E rendus canoniques et gate recalculé** : 48 dossiers
  sectoriels ont été reliés à la queue V4.3 et aux vrais pools top-10 du shadow,
  puis combinés aux cinq contradictions déjà canoniques. Les erreurs de hashes
  et de taxonomie de sources ont été détectées par recomputation puis corrigées
  contre les archives. Corpus final à ce stade : 53 cas, 47 validés, 35
  `TOP1_CORRECT`, 11 `TOP1_WRONG`, une `AMBIGUOUS`, six `UNRESOLVED` et 19
  random validés. Verdict `PIVOT_MORE_EVIDENCE`; aucun réentraînement.
  La passe initiale refuse désormais en code toute ligne V4.3 `REVIEW`.
  Rapport : `reports/v9/v4_4_adjudication_gate_results.md`.
  *(commits GitHub : données `f2aeec0`, adaptateur `c6cb686`, rapport
  `a66499e`, garde AUTO `4930521`; 282 tests passants)*
- **Premier gate V4.4 canonique publié** : les cinq contradictions ont été
  reliées aux pools top-10 réellement archivés dans le shadow, au bundle et à
  la signature retrieval gelés, puis validées par recomputation. Résultat :
  quatre `TOP1_WRONG` éligibles accepteur, zéro cible ranker et un
  `UNRESOLVED`. Les cinq appartiennent au tirage aléatoire V4.3 ; une première
  perte de cette provenance a été détectée, corrigée et les artefacts ont été
  reconstruits. Verdict partiel `PIVOT_MORE_EVIDENCE`, avec déficits 75
  corrects, 46 incorrects et 26 random. *(commits GitHub : dossiers
  `7c1a6fd`, validateur `edbfbfe`, adaptateur `ef2df25`, correction provenance
  `3a74b8a`, gate `17bf904` ; 271 tests passants)*
- **Preuves sectorielles V4.4 collectées sans dépense** : 117 observations sur
  52 dossiers, auprès des producteurs UAI, FINESS, Agence Bio et ADEME.
  UAI retrouve 27/29 identifiants, FINESS 33/33, Bio 10/10 et RGE 45/45
  couples qualification/SIRET. Les 115 réponses positives portent toutes le
  même SIRET explicite que l'observation ; les deux UAI absents restent sans
  interprétation. La file de priorité contient 14 dossiers avec signal
  sectoriel non attaché au top-1, 37 attachés au top-1, un identifiant non
  résolu et 120 sans signal sectoriel. Zéro label créé automatiquement.
  *(commits GitHub : collecte `332094d`, faits `428942b`)*
- **Faits V4.4 dérivés sans faux consensus** : les 440 vues de l'API
  officielle ont été ramenées à une seule famille de source. Les 172 dossiers
  AUTO portent désormais 53 faits auditables, mais zéro conclusion de
  correction et zéro label entraînable. Un accord SIRET direct, une recherche
  nom + géographie, le score ou l'adresse ne peuvent pas créer une vérité.
  L'ordre de réentraînement est également figé : accepteur logistique d'abord,
  avec retrieval et ranker gelés ; ranker ensuite uniquement pour les erreurs
  ayant un SIRET alternatif exact prouvé et naturellement présent dans le
  pool. Rapport :
  `reports/v9/v4_4_evidence_validated_retraining_design.md`. *(commits GitHub :
  faits `9274399`, design `1d4d0f1` ; 6 tests ciblés passants)*
- **Preuves officielles V4.4 collectées pour les 172 AUTO difficiles** :
  politique autonome gelée, sans validation demandée à l'utilisateur. API
  Recherche d'entreprises interrogée à débit limité : 440/440 réponses HTTP
  200, 325 requêtes avec résultat. Les réponses brutes, URLs et dates sont
  conservées sur le SSD ; elles ne valent pas encore adjudication et aucun
  entraînement n'est ouvert. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_4_official_evidence/87983e83c11f5284`.
  *(commits GitHub : contrat `ede441b`, collecteur `341acf2` ; 223 tests
  passants)*
- **File de labels difficiles V4.3 construite** : population figée de 542
  `UNRESOLVED`, sans suppression ; 172 AUTO, 370 REVIEW, 144 random. Priorités :
  cinq contradictions connues, 35 AUTO adresse-seule, 28 AUTO en désaccord
  avec un input actif, 104 autres AUTO, puis les REVIEW. Un lot opérationnel
  de 250 dossiers réunit CRM, top-1 et preuves SIRENE. Zéro nouveau label
  entraînable : verdict `PIVOT_VALIDATION`, aucun réentraînement autorisé.
  Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_3_hard_labels/0f832305ab199267`.
  Le disque interne étant saturé, seuls les caches Python/pytest régénérables
  ont été supprimés ; aucun artefact métier n'a été touché. *(commits GitHub :
  contrat `a2232bf`, builder `c3c5944`, correction normalisation `3388649`,
  lot 250 `b16ce8b`, rapport `c420c26` ; 221 tests passants)*
- **Intégrité retrieval V4.2 validée sans GPU** : contrat figé avant le
  correctif ; variante B et état autoritaire lu dans le snapshot complet.
  Résultat 242/242 = 100 % Recall@100, random exact 91/91, zéro fermé, zéro
  injection, zéro vérité absente du snapshot et zéro régression sur les 237
  anciens succès. Les cinq misses sont tous récupérés. Latence p50 455,1 ms,
  p95 2 878,6 ms sur cet échantillon difficile. Verdict `GO_HARD_LABELS`,
  sans autorisation de réentraînement ou de déploiement. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_2_retrieval_integrity_7c4b957`.
  *(commits GitHub : contrat `c33d3e0`, source d'état `48ed90b`, évaluateur
  `7c4b957`, rapport `2d7070e` ; 218 tests passants)*
- **Audit représentatif V4.1 exécuté en aveugle** : échantillon figé de 800
  lignes dont 250 aléatoires ; preuves construites sans décision, score,
  prédiction ni rang modèle ; 242 `MATCH_EXACT`, 16 `AMBIGUOUS` et 542
  `UNRESOLVED` provisoires. Le random n'est mécaniquement conclusif qu'à
  42,4 %. Retrieval A : 237/242 ; B/C : 240/242. Deux pertes restantes
  viennent du magasin d'état incomplet. Cinq contradictions AUTO nettes
  réfutent la sécurité extrapolée depuis le dev. Verdict `PIVOT_LABELS` /
  `STOP_DEPLOYMENT`; aucun modèle ni seuil modifié. Artefacts :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative/e06cf0d79849aad4`,
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative_evidence/e696f22d68c0210f`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative_summary/2d18ef172f32aefc`.
  *(commits GitHub : contrat/échantillon `015d718`/`5f8ea00`, preuves
  `361c138`/`edf0858`, synthèse `771be6b`, rapport `17465cd` ; 211 tests
  passants)*
- **V4.1 actif-courant exécutée en shadow local** : gate retrieval A à 305/305
  sur dev avec 100 candidats maximum et zéro fermé ; dataset de 7 003 requêtes
  et 698 428 paires ; ranker R1 à 99,918 % Hit@1 dev ; accepteur brut à
  99,832 % de précision observée et 81,593 % de couverture dev. Le shadow
  atomique contient 19 025 décisions, 10 292 AUTO et 8 733 REVIEW, zéro exclu
  scoré, zéro écriture CRM et aucune revendication de précision. Artefacts :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/retrieval_v41_dev_feede27`,
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_1/f938abf6b8a87155`,
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_1/f938abf6b8a87155`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/shadow/v4_1/runs/v41_shadow_f1058826_20260727_v3`.
  Verdict `PIVOT_CERTIFICATION`; 206 tests passants. *(commits GitHub :
  contrat `eea75f2`, retrieval `a599e4a`/`85f7674`/`993e088`, modèles
  `f158da2`/`942a443`/`c4ffb2a`/`d86f6f6`, inventaire et dataset
  `af18779`/`ab13fb4`/`9fd30d8`/`feede27`, runner
  `41cbc0e`/`8e96961`/`cc5dec1`/`9a322bc`)*
- **Évaluation finale V4 exécutée une seule fois** : autorisation gelée
  `7dbd5527374ca0d4`, zéro chevauchement SIREN, zéro injection et 100 candidats
  maximum. Résultats : Recall@100 302/302 = 100 %, Hit@1 exact 296/302 =
  98,013 %, AUTO 282/354 = 79,661 %, précision AUTO 280/282 = 99,291 %.
  Deux causes simples : un ancien SIRET PALAFIS fermé mais textuellement
  parfait est automatisé à la place du SIRET actif ; une scène ELGEA déjà
  qualifiée `AMBIGUOUS` avec 80 SIRET actifs est automatisée. La calibration
  isotonic attribue exactement 1,0 aux 282 AUTO. Verdict `PIVOT` /
  `TECHNICAL_PIVOT`. Le premier rapport `STOP` est conservé : il inversait
  deux booléens d'intégrité. Sa correction n'a relu ni le holdout ni les
  modèles. Suite complète à 148 tests passants. *(commits GitHub : contrat
  `fb6a20c`, runner `8cc9bfa`, correction instrumentale `aead6f5`, rapport
  `4aade83`)*
- **Accepteur V4 validé avant holdout** : dataset de 7 215 scènes
  (6 054 exactes, 1 161 ambiguës), 721 007 paires et zéro `UNRESOLVED`.
  Les exactes train utilisent les prédictions OOF ; les ambiguës étaient
  entièrement absentes du fit ranker. Sur le demi-dev `threshold`, le bundle
  retenu produit 149/189 AUTO = 78,836 %, zéro erreur observée, contre un gate
  de 25 % à 99,8 %. Les six variantes arrivent au même point. Verdict
  `GO_HOLDOUT_V4`, sans lecture du holdout. Artefacts :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/acceptor_v4/2b8a9c994e0944be`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/acceptor_v4/acceptor_2b8a9c994e0944be_9ec88c8`.
  Suite complète à 145 tests passants. *(commits GitHub : contrat `9a22fd8`,
  préparation `af5ce0b`, builder/train `9ec88c8`, rapport `ff1eea4`)*
- **Ranker V4 validé sur le nouveau dev** : dataset de 604 938 paires,
  5 749 requêtes fit et 305 dev, 55 features déterministes, exactement un
  positif réel par requête, aucune injection et aucun SIREN exact partagé.
  Le nouveau XGBRanker atteint 299/305 = 98,033 % Hit@1 SIRET contre
  290/305 = 95,082 % pour l'ancien ranker épinglé ; comparaison appariée :
  dix erreurs corrigées, une créée. Verdict `GO_ACCEPTEUR_V4`. Artefacts :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/ranker_v4/1aebeada820d92a7`,
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/ranker_v4/ranker_1aebeada820d92a7_6236365`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/ranker_v4_e1_250a05f`.
  Suite complète à 140 tests passants. *(commits GitHub : contrat `0c90c25`,
  builder `6236365`, évaluateur `250a05f`, rapport `13e3547`)*
- **Gate retrieval V4 franchi sans GPU** : contrat gelé avant calcul, reprise
  des 4 932 anciennes listes et reconstruction des seuls 1 124 cas frais.
  Recall@100 : noyau historique 4 930/4 932 = 99,959 %, ajout fit
  819/819 = 100 %, fit combiné 5 749/5 751 = 99,965 %, nouveau dev
  305/305 = 100 %. Zéro dépassement de 100, zéro positif injecté, zéro SIREN
  exact partagé fit/dev, holdout et ancien test non lus. Les deux misses sont
  des scènes fit historiques visibles dans les canaux mais éliminées par
  l'ancienne admission. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/retrieval_v4/ddefe3daaacdf5ef`.
  Suite complète à 136 tests passants. *(commits GitHub : contrat `510868b`,
  builders/tests `e566c25`, rapport `6948aa1`)*
- **Expansion V4-Fresh passée sans réutiliser le benchmark** : les 6 330
  `SERVICE ID` absents du benchmark ont fourni 1 426 SIRET actifs uniques,
  247 ambigus et 4 657 non résolus. Séparation gelée : `fit_addition`
  819 exacts, `dev_new` 305, `holdout_sealed` 302. Le fit combiné avec le noyau
  V4 contient 5 751 exacts. Zéro chevauchement de SIREN exact entre les trois
  rôles, zéro identifiant déjà présent dans le benchmark, zéro SIRET fermé.
  Le holdout est hashé mais non évalué. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/v4_fresh_expansion/14047b719ef90f6f`.
  Suite complète à 132 tests passants. *(commits GitHub : contrat `1c2e84c`,
  builder/tests `613cf7d`, rapport `d8d36b9`)*
- **Qualification V4 actuelle pré-enregistrée puis exécutée** : examen de
  toutes les lignes actives de la partition géographique, sans top-k, rang,
  score ni décision modèle. V4 produit 4 060/11 837 = 34,299 % exacts train et
  872/2 565 = 33,996 % exacts dev ; 759 SIRET et 351 SIREN changent face à
  l'historique, et les cinq conflits E2b sont corrigés. Chaque exact a une
  preuve active unique. Gate `STOP_V4` : couverture <50 %, train <5 000 exacts
  et 14 SIREN V4 traversent l'ancien split. Diagnostic post-qualification :
  Recall@100 99,951 % train / 100 % dev et ranker E1 Hit@1 96,034 % /
  94,954 %. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v4/0b333d33a56ed759`.
  Suite complète à 129 tests passants. *(commits GitHub : contrat `ce82b01`,
  builder/tests `799c32d`, rapport `299bc8a`)*
- **E2b pré-enregistré puis exécuté sans test** : comparaison fermée de la
  régression logistique standardisée et de XGBoost avec score brut, sigmoid ou
  isotonic. Le brut logistique gagne : 85/1 280 = 6,641 % AUTO à 100 % observé
  contre 33/1 280 = 2,578 % pour XGBoost isotonic, mais le gate de 25 % échoue.
  Les 320 premiers scores ne comportent que cinq erreurs formelles ; les cinq
  prédisent un SIRET dont le nom/adresse SIRENE correspondent directement au
  CRM, tandis que le label désigne une autre entité, une ancienne entité ou
  `UNRESOLVED`. Exemples : VISSELECT actif contre AVENIS fermé, PGDIS contre
  OFFICE DEPOT, LMP SANTE actif contre LMP SANTE fermé. Verdict formel
  `STOP_E2B`, lecture architecturale `PIVOT_DATASET_AVAL`. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/acceptor_e2b_3171ef5020c0f068_070c123`.
  Suite complète à 123 tests passants. *(commits GitHub : contrat `cf91432`,
  code `070c123`, rapport `ebb4bf2`)*
- **Expérience aval E1/E2 exécutée sur train/dev** : dataset immuable de
  1 438 845 paires, 100 candidats maximum, zéro doublon, zéro détail manquant
  et Recall@100 V3 de 99,162 % train / 99,572 % dev. Le ranker final atteint
  1 754/2 104 = 83,365 % Hit@1, soit +2,804 points sur l'ancien ranker, avec
  gains actifs, fermés et multi-sites. L'accepteur XGBoost calibré ne couvre
  que 33/1 280 = 2,578 % à 100 % observé ; le palier suivant tombe déjà à
  98,507 %. Verdict `PIVOT_ACCEPTEUR`, sans retour au risk model historique,
  sans dense, sans GPU et sans ouverture du test. Artefacts :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/downstream/3171ef5020c0f068`,
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/ranker_3171ef5020c0f068_fc9cb1b`
  et
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/acceptor_3171ef5020c0f068_fc9cb1b`.
  Suite complète à 121 tests passants. *(commits GitHub : correction builder
  `fc9cb1b`, rapport `9ab7f6a`)*
- **Socle de l'expérience aval E1/E2** : builder train/dev alimenté par les
  listes top-100 gelées, provenance alignée sur les sept canaux sparse
  réellement certifiés, ranker déterministe par défaut, folds OOF groupés par
  SIREN et courbes accepteur aux points 99,0/99,5/99,8 %. Le canal dense
  abandonné n'est plus utilisé comme faux signal de provenance. Le canal
  `current_sparse` audité peut servir directement de baseline train, sans
  recalcul redondant. Suite complète à 120 tests passants. *(commits GitHub :
  `0a75b73`, `dbd8906`, `2c24052`)*
- **Contrats aval exact-SIRET renforcés** : déduplication obligatoire des SIRET
  avant toute scène, plafond absolu 100, preuves top-1/top-2 et deltas
  nom/adresse transportés jusqu'à l'accepteur, folds OOF par SIREN et
  évaluation du holdout final désactivée par défaut. Une autorisation liée à
  un nouveau dataset est obligatoire pour l'ouvrir; le test sélectif consommé
  est explicitement refusé. Suite complète à 119 tests passants. *(commit
  GitHub : `aeeaf0f`)*
- **Contrat matching aval gelé** : trajectoire `top-100 gelé → ranker SIRET
  unique → accepteur exact-SIRET`, première expérience sans sémantique ni GPU,
  vraies scènes OOF, publication end-to-end et nouveau holdout indépendant
  obligatoire. *(commit GitHub : `c18bf28`)*
- **Audit reproductible de l'architecture aval** : la référence historique
  contient 1 428/2 512 scènes avec le même SIRET en top-1 et top-2. Toutes sont
  AUTO. Sur les scènes réellement distinctes, la couverture tombe à 40,959 %
  et la précision brute à 98,649 %. Le fichier versionné donne 1 866/1 872 =
  99,679 %, pas 99,84 %. Le decider garde toutefois un signal utile de +3,03 à
  +4,66 points Hit@1 sur les mêmes scènes. Les risk models ciblent le SIREN et
  1 539/16 621 lignes V7 ont le bon SIREN mais le mauvais SIRET. Verdict :
  `PIVOT AVAL`, conserver les preuves dans `ranker final + accepteur
  exact-SIRET`. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/downstream_architecture_audit_a59fb0f`.
  *(commits GitHub : runner/tests `a59fb0f`, rapport `e186439`)*
- **Audit de stabilité V3 limité à train/dev** : décomposition reproductible
  des pertes entre contradictions structurelles V2 et absence de preuve V3,
  avec refus explicite du split test. La couverture V3 vaut 79,632 % sur train
  et 82,027 % sur dev. Pour les fermés, elle vaut déjà 65,055 % sur train et
  69,405 % sur dev : la difficulté n'est pas créée par le test. Parmi les
  V2 exacts écartés faute de preuve, le nom et l'adresse sont tous deux
  éloignés dans 774/1 569 cas train et 131/296 cas dev; un simple assouplissement
  des seuils ne traite donc pas la cause dominante. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/qualification_stability_train_dev_111b07c`.
  Suite complète à 109 tests passants. *(commit GitHub : `111b07c`)*
- **Certification finale selective sur test** : qualification V2/V3 produite
  avant tout retrieval, puis exécution unique de la configuration gelée.
  Couverture V3 2 128/2 652 = 80,241 %, Recall@100 V3
  2 116/2 128 = 99,436 %, oracle interne 2 128/2 128, maximum 100 candidats et
  zéro dépassement. Les gates globaux passent. Les segments fermés
  (62,633 % de couverture) et mégapoles (77,586 %) ratent uniquement leurs
  planchers de stabilité; leurs recalls atteignent 98,305 % et 99,259 %.
  Verdict pré-enregistré `PIVOT`; aucune nouvelle variante ne doit être testée
  sur ce test. Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/certification/selective_test_c33b80855f560074_6fab035`.
  Suite complète à 105 tests passants. *(commits GitHub : `d1c0fc9`,
  `6fab035`, rapport `41ff2e1`)*
- **Contrat final pré-enregistré avant ouverture du test** : double gate
  couverture V3 ≥80 % et Recall@100 exact ≥99 %, oracle sans vérité invisible,
  plafond strict 100, publication historique/V2/V3 et stabilité segmentaire.
  L'admission, les seuils, les hashes de snapshots et le runner de qualification
  ont été gelés avant l'évaluation. *(commits GitHub : `4f6e317`,
  `eb0e6a3`)*
- **Qualification V3 par preuve directe** : politique indépendante du
  retrieval séparant `NAME_AND_ADDRESS`, `NAME_ONLY`, `ADDRESS_ONLY` et
  `NO_DIRECT_EVIDENCE`. Un label V2 exact sans preuve directe devient
  `UNRESOLVED`, sans promotion automatique d'un autre SIRET. Dev :
  2 104 exacts, 81 ambigus, 380 non résolus, couverture 82,027 % et
  Recall@100 gelé 99,572 %. Test qualifié avant retrieval : 2 128 exacts,
  105 ambigus, 419 non résolus et couverture 80,241 %. *(commits GitHub :
  `09b9d46`, `cf7133c`, `c6c8186`)*
- **Qualification V2 train/dev sans suppression ni relabel automatique**:
  politique indépendante des résultats du retrieval, builder immuable et
  double publication des métriques historique/V2. Sur dev, 2 400/2 565
  restent `MATCH_EXACT`, 81 deviennent `AMBIGUOUS` et 84 `UNRESOLVED`.
  L'admission passe seulement de 2 495/2 565 = 97,271 % à
  2 343/2 400 = 97,625 % sur le périmètre exact: il manque encore 33 succès au
  gate. L'oracle interne atteint 2 394/2 400 = 99,750 %, avec 6 vérités non
  vues et 51 vues puis éliminées. Sur train, 10 995 labels restent exacts,
  440 deviennent ambigus et 402 non résolus. Aucun SIRET alternatif n'est
  promu; le benchmark original et le test restent inchangés. Artefacts:
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v2/522351669d5313dc`
  (dev) et `.../f8af7e1da18fa94a` (train). Rapport:
  `reports/recall100/benchmark_v2_qualification.md`. Suite complète à 99 tests
  passants. *(commits GitHub: `16e657e`, `a68f679`)*
- **Audit global de non-unicité des labels exacts**: nouveau runner immuable
  comparant, pour chaque requête, le label aux autres SIRET du même SIREN via
  la clé d'adresse canonique. Sur les 2 565 requêtes dev, 231 ont un autre
  sibling à l'adresse exacte, 165 au moins un sibling actif, 87 un label fermé
  avec sibling actif exact et 29 plusieurs siblings actifs exacts. Ce volume
  dépasse très largement les 25 erreurs tolérées à 99 % et prouve que le SIRET
  exact n'est pas toujours identifiable avec les champs CRM. Artefact:
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/site_label_audit_dev_c33b80855f560074_ac971e0`.
  Suite complète à 94 tests passants. *(commits GitHub: `638d093`,
  `ac971e0`)*
- **Second passage autonome sur les 63 prunings**: comparaison de tous les
  établissements du SIREN historique, vérification de relations opaques par
  sources publiques et application diagnostique du ranker historique gelé.
  28/63 ont un sibling SIRET dont l'adresse correspond mieux au CRM, 23 ont un
  meilleur sibling actif, et 11 un sibling actif à adresse pratiquement exacte.
  Dix de ces alternatives cohérentes sont déjà dans le top-100 actuel. Le
  ranker historique ne récupère que 22/63; même sans aucune nouvelle perte, son
  plafond optimiste serait 98,13 %. Plusieurs labels sont confirmés comme alias
  métier, tandis que Mercure/Oceania et Globecast/Kinepolis sont contredits par
  les adresses et activités publiques. La recommandation devient la création
  d'un benchmark versionné avec politique `actif à l'adresse`,
  `AMBIGUOUS_SITE` et alias historiques avant tout nouveau modèle. Rapport:
  `reports/recall100/pruned_63_audit.md`. *(commit GitHub: `f3bd0b1`)*
- **Audit des 63 vérités trouvées puis éliminées**: 13 ne sont présentes que
  dans l'overlay fermé et ne reçoivent pas de score complet; parmi les 50
  présentes dans V7, une seule reste dans le top-100 de la fusion, 17 sont
  classées 101–200, 17 entre 201–500 et 15 après 500. L'examen métier sépare
  12 preuves d'adresse, 8 preuves de nom, 13 choix du mauvais établissement
  d'un bon SIREN, 12 équipements publics reliés à leur propriétaire
  administratif et 18 relations historiques faibles ou opaques à valider
  humainement. Les petites règles testées plafonnent à 97,35 %; scorer tout
  l'overlay dégrade à 96,41 %. Rapport:
  `reports/recall100/pruned_63_audit.md`. *(commit GitHub: `58d8b31`)*
- **Décision Recall@100 = PIVOT**: sur les 2 565 requêtes dev, le sparse gelé
  atteint 2 379/2 565 = 92,75 %, la meilleure admission déterministe observée
  2 495/2 565 = 97,27 %, et l'oracle des canaux internes à K=5 000
  2 558/2 565 = 99,73 %. Le plafond de sortie 100 est strictement respecté,
  mais il manque 45 succès au gate; 7 vérités ne sont vues par aucun canal et
  63 sont vues puis éliminées par l'admission. Aucune nouvelle variante n'a été
  exécutée sur le test. Le pivot proposé est une tête d'admission apprise,
  distincte du ranker aval, sous un nouveau contrat. Rapport:
  `reports/recall100/final_go_pivot_stop.md`. *(commit GitHub: `ccb3689`)*
- **Évaluateur d'admission reproductible**: validation des manifests et hashes,
  RRF pondéré déterministe, quotas overlay, plafond strict, oracle interne,
  attribution `unseen`/`pruned`, segments et latence de sélection. L'artefact
  immuable dev est
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/admission_diagnostic_dev_c33b80855f560074_5a0e67f`.
  Suite complète à 90 tests passants. *(commit GitHub: `5a0e67f`)*
- **Audit profond des canaux à K=5 000**: le pool V7 et l'overlay fermé ont été
  audités séparément sans positif injecté. Leur oracle combiné voit
  2 558/2 565 SIRET exacts, contre 2 540 requis, établissant que le sourcing
  peut théoriquement dépasser 99 % mais ne constitue pas une sortie éligible à
  100. *(commit GitHub: `d4255de`)*
- **Canaux SIREN locaux audités**: ajout de `siren_head` et `siren_sites` pour
  regrouper les candidats par SIREN puis ordonner/étaler leurs établissements.
  À K=100, `siren_head` récupère 47 misses du sparse courant; l'oracle V7 passe
  à 96,18 %. *(commit GitHub: `d4255de`)*
- **Overlay fermé construit et audité**: le builder limité en mémoire a publié
  8 230 664 lignes physiques INSEE et 8 286 671 lignes CP sur 52 408 fichiers,
  pour environ 872 Mo. Les 62 vérités absentes du store V7 sont toutes présentes
  dans l'overlay; son sparse courant en récupère 52 à K=100 et son oracle 60 à
  K=5 000. *(commits GitHub: `e39fddd`, `d71d3cb`)*
- **Builder overlay des fermés legacy**: construction immuable et sans lecture
  des labels d'un canal contenant uniquement les SIRET fermés exclus par le
  filtre V7 `dateDebut >= 2016`. Le périmètre géographique est dérivé des seuls
  champs INSEE/CP du benchmark, les snapshots et le benchmark sont contrôlés
  par hash, le build est atomique, manifeste et compatible avec le store
  partitionné. *(commit GitHub: `601eee5`)*
- **Audit unitaire sparse publié**: sur dev, caractères et mots récupèrent
  respectivement 65 et 59 misses du sparse@100; adresse TF-IDF 6, nom exact 15,
  adresse exacte 2 et numérique 0. L'oracle des canaux à leur propre top-100
  atteint 95,59 %, encore sous le plafond du store à 97,58 % et sous la cible.
  Rapport et lecture architecturale dans
  `reports/recall100/channel_audit_dev.md`. *(commit GitHub: `d070db8`)*
- **Audit unitaire des canaux sparse instrumenté**: runner immuable séparant
  TF-IDF nom mots, TF-IDF nom caractères, TF-IDF adresse, nom normalisé exact,
  adresse exacte et rescue numérique. Il conserve les listes/rangs par requête,
  mesure la complémentarité appariée, le SIREN et la géographie, et refuse le
  run si `current_sparse` ne reproduit pas exactement l'artefact baseline gelé.
  Smoke réel sans divergence; suite complète à 83 tests passants. *(commit
  GitHub: `218c22c`)*
- **Baseline sparse Recall@K dev publiée**: préfixe @50 identique sur les
  2 565 requêtes à la baseline historique; Recall SIRET @50/@100/@200/@500 =
  90,33/92,75/94,15/95,79 %. Le store V7 plafonne à 97,58 %: 62 SIRET, tous
  fermés, sont absents des 14,3 M candidats mais présents dans le snapshot brut
  StockEtablissement. Zéro perte filtre ou déduplication; 124 autres vérités
  sont classées après 100. *(commit GitHub: `67f1a9c`)*
- **Préfixes Recall@K stabilisés et cache mutualisé**: un passage max-K ne
  classait pas les partitions de taille comprise entre 51 et K, rendant son
  préfixe @50 différent de la baseline. Ajout d'un seuil de déclenchement du
  ranking indépendant du budget final; le smoke reproduit désormais exactement
  les dix préfixes @50 historiques. Les matrices TF-IDF utilisent un hash
  d'artefact indépendant du cutoff avec fallback vers les 7,7 Go de cache
  legacy. Le premier run `..._963160b` est conservé mais déclaré diagnostic
  invalide pour les préfixes @50/@100/@200. Suite complète à 81 tests passants.
  *(commit GitHub: `bdc7ad4`)*
- **Instrumentation Recall@K et causes de perte**: séparation explicite des
  états avant filtre, après filtre et après déduplication dans le retrieval
  partagé; nouveau runner immuable calculant en un passage les préfixes
  @50/@100/@200/@500, intervalles, segments, latence, cardinalités et buckets
  de perte mutuellement exclusifs. Smoke réel et suite complète à 78 tests
  passants. *(commit GitHub: `5e3fd5f`)*
- **Contrat Recall@100 pre-enregistre**: cible SIRET exacte >=99,0 %, plafond
  absolu 100, courbes diagnostiques @50/@100/@200/@500, attribution obligatoire
  partition/filtre/deduplication/pruning, audit canal par canal, tuning
  train/dev et unique evaluation de la variante gelee sur test. `AGENTS.md`
  pointe desormais vers ce goal actif. *(commit GitHub: `8b77af3`)*
- **Decision finale V9 = PIVOT**: sparse + dense local perd 1,83 point de
  Recall@50 SIRET et sparse + dense global SIREN perd 2,61 points. Les deux
  regressions sont statistiquement nettes et violent les gates segmentaires.
  En revanche, leur Hit@1 SIRET brut gagne respectivement 7,33 et 11,31 points,
  avec des IC95 strictement positifs. Gate 3 ranker/accepteur, Gate 4 open-set
  et le cross-encoder ne sont pas ouverts sur le pool rejete. Le pivot propose
  une nouvelle ablation: pool sparse fixe + dense comme feature de scoring.
  Le Mac a execute l'ensemble sans GPU ni depense cloud. *(commit GitHub:
  `59ec78c`; lecture STOP preliminaire supersedee: `53ad3b3`)*
- **Hit@1 SIRET/SIREN publie par le runner**: les comparaisons appariées
  incluent désormais les Hit@1, leurs recuperations/deplacements, IC95
  bootstrap et tests de McNemar. Sur global, Hit@1 SIRET passe de 36,22 % a
  47,52 % et Hit@1 SIREN de 41,91 % a 53,92 %. Suite complete a 68 tests
  passants. *(commit GitHub: `de0079a`)*
- **Gate 2 dense global SIREN echouee sur dev**: sparse atteint 2 317/2 565 =
  90,33 % contre 2 250/2 565 = 87,72 % pour l'hybride global. Delta apparie
  −2,61 points, IC95 [−3,51; −1,72], McNemar p=1,47e-8; 37 misses recuperes
  mais 104 hits deplaces. Actifs −3,27 points et multi-sites −2,28 points.
  La latence p95 passe a 1,079x et le budget 50 est respecte. *(commit GitHub:
  `53ad3b3`)*
- **Audit de budget multicanal corrige**: deux sorties globales de 50 candidats
  etaient faussement marquees non conformes car leur seul pool local contenait
  41 ou 18 lignes. Le controle refuse maintenant les depassements de K et les
  underfills locaux, sans rejeter les candidats qui completent un pool court
  via un nouveau canal. Rapport global regenere avec zero violation; suite
  complete a 68 tests passants. *(commit GitHub: `bc49918`)*
- **Expansion globale SIREN bornee avant materialisation**: le store candidat
  v2 calcule la densite par zone et applique dans DuckDB un top-K par SIREN,
  ordonne par correspondance INSEE/CP puis densite locale, avant tout transfert
  vers Python. Le cap SQL est 40 et le cap metier final reste 20 SIRET par
  SIREN; le lecteur reste compatible avec les stores v1. Suite complete a
  66 tests passants. *(commit GitHub: `43f2c64`)*
- **Manifeste d'expérience dense fermé**: chaque run V9 référence désormais le
  hash du contrat de son store local, ANN global, géographie mmap et store
  candidat SIREN; les stores partitionnés sans manifeste racine sont liés par
  un hash agrégé déterministe de leurs  manifestes. Suite complète à 66 tests
  passants. *(commit GitHub: `fef3658`)*
- **Expansion globale SIREN rendue exécutable**: le smoke historique rechargeait
  jusqu'à des dizaines de partitions aléatoires par requête (p95 17,5 s sur
  cinq cas, contre 0,67 s sparse). Ajout d'un store DuckDB read-only indexé par
  SIREN qui récupère les 50 groupes de candidats en une requête, tout en
  conservant la priorité géographique et la limite de 20 SIRET par SIREN.
  La lecture Arrow conserve les `None`/listes des partitions, sans conversion
  pandas en `NaN`. Suite complète à 65 tests passants. *(commits GitHub:
  `13d66d2`, `654413f`)*
- **Lookup géographique SIREN compatible 24 Go**: remplacement optionnel du
  chargement legacy de 37,8 M lignes dans pandas/dict par un artefact trié,
  quatre tableaux NumPy mmap et une recherche binaire SIREN. Le builder DuckDB
  travaille sur SSD avec limite mémoire, publie hashes et cardinalité; le
  lecteur conserve la compatibilité avec le parquet historique. Suite complète
  à 64 tests passants. *(commit GitHub: `7781d31`)*
- **Index dense global SIREN construit sur Mac**: 28 982 797 unités légales
  encodées en CPU avec le modèle générique épinglé, puis indexées en IVFPQ
  4096/48; manifeste avec hash source, fingerprint modèle et hashes FAISS/IDs.
  Un contrôle reproductible échantillonne les row groups du parquet, vérifie
  l'intégrité des sorties et mesure self-recall@1/@50 et latence avant toute
  évaluation métier. Suite complète à 63 tests passants. *(commits GitHub:
  `2d74b2b`, `6718d8b`)*
- **Contrat dense global SIREN renforcé**: le builder publie désormais la
  progression d'encodage, le fingerprint intégral du modèle et les hashes des
  fichiers FAISS/IDs. Le benchmark et l'inférence V9 refusent un index construit
  avec un autre modèle avant même de charger FAISS. Suite complète à 62 tests
  passants. *(commit GitHub: `2d74b2b`)*
- **Gate 2 dense local échouée sur dev**: sparse atteint 90,33 % Recall@50
  SIRET contre 88,50 % pour sparse+dense local et 70,29 % pour dense seul.
  L'hybride récupère 45 misses mais déplace 92 hits: delta apparié −1,83 point,
  IC95 [−2,73; −0,94], p exact 0,000073. Budget et latence passent, mais actifs
  (−2,26), mégapoles (−3,03) et multi-sites (−2,28 points) violent le gate
  segmentaire. Les 168 misses sparse au niveau SIREN et 25 récupérations SIREN
  uniques par le dense justifient la dernière expérience globale SIREN, sans
  tuning opportuniste de RRF. *(commit GitHub: `71c68ef`)*
- **Store dense local dev complet**: les 871 partitions INSEE et 14 partitions
  CP du plan gelé ont été encodées sur CPU avec le MiniLM générique épinglé,
  soit 10 216 448 candidats dans 885 paires index/manifeste (3,0 Go sur SSD).
  La vérification exhaustive confirme un unique fingerprint modèle, le hash
  exact du plan, zéro fichier manquant/temporaire et 61 tests passants. Le
  builder cherchait initialement `cp_codes` au lieu du champ canonique
  `postcode_codes`; le défaut est corrigé et couvert par régression. *(commit
  GitHub: `8ec1881`)*
- **Comparateur apparié Gate 2**: validation des hashes de l'expérience et de
  l'alignement exact des requêtes, décompte des misses récupérés et hits
  déplacés, IC95 bootstrap apparié, test exact de McNemar, deltas par segment,
  ratio de latence p95 et refus explicite de toute violation du budget fixe.
  Le rapport JSON/Markdown produit est immuable et lié au manifeste de
  l'expérience; suite complète à 60 tests passants. *(commit GitHub:
  `86dea2c`)*
- **Dense local non contamine prepare**: fingerprint integral du modele
  semantique impose entre build et inference, revision generique MiniLM
  `86741b4e` copiee sans telechargement sur le SSD, reparation du tokenizer
  Unigram et plan de partitions immuable. Le plan dev couvre 871 partitions
  INSEE et 14 CP, environ 10,2 M de lignes physiques; aucune requete dev sans
  partition planifiable. *(commit GitHub: `10dd990`)*
- **Baseline sparse-50 V9 mesuree**: sur les 2 652 requetes test gelees,
  Recall@50 SIRET 88,54 % (2 348 hits, IC95 87,27–89,69), Recall@50 SIREN
  92,16 %, recall du pool geographique 98,00 %, zero violation de budget.
  Les 304 erreurs comprennent 53 absences de partition et 251 prunings; 96
  erreurs conservent le bon SIREN. Segments critiques: fermes 67,09 %,
  megapoles 77,01 %. Artefacts bruts hashes sur le SSD et rapport dans
  `reports/v9/retrieval_baseline_sparse50.md`. *(commit GitHub: `8adc5f3`)*
- **Runner retrieval V9 immuable**: execution sparse, hybride local, dense-only
  et hybride global SIREN avec budget final strict, preuves par requete,
  Recall SIRET/SIREN et Wilson 95/99 %, segments, latences p50/p95/p99, cache
  SSD borne en RAM et manifeste lie au commit. Le benchmark segmente v2
  `c33b80855f560074` remplace le build v1 pour les experiences; le v1 reste
  conserve. *(commit GitHub: `771beb6`)*
- **Benchmark ferme V9 gele**: reconstruction exacte du split V7 historique
  par SIREN (seed 42), validation contre les scenes positives V7, ajout des 692
  requetes historiquement absentes des scenes afin de compter les misses
  end-to-end, hash integral des 4 119 fichiers de partitions et des snapshots
  SIRENE. Build initial immuable `8967e72e07c9f4bf` puis revision segmentee
  `c33b80855f560074` sur le SSD externe: 11 837 train,
  2 565 dev, 2 652 test, zero SIREN partage. Les labels restent des verites CRM
  historiques non reaudites et le modele dense fine-tune local est declare
  contamine pour toute revendication finale sur ce corpus. *(commit GitHub:
  `b384509`)*
- **Gate 0 V9 sans GPU franchie**: cles d'index dense alignees sur les vraies
  partitions, refus des subsets mega-communes incompatibles, manifeste de
  cardinalite et d'ordre SIRET, isolation stricte de PyTorch et FAISS dans
  deux sous-processus persistants sans `KMP_DUPLICATE_LIB_OK`, builders local
  et global SIREN corriges, mode dense-only repare et entrypoints V9
  executables directement. Validation: 52 tests passants, smoke 512 lignes,
  index local reel de 17 462 candidats et index global SIREN de 1 000 entites
  construits/interroges avec succes sur CPU. *(commit GitHub: `88e97e0`)*
- **Contrat d'execution V9 sans GPU**: directive active `GO/PIVOT/STOP`
  placee en tete de `AGENTS.md`, ressources locales autorisees, ordre des
  experiences, gates et regles d'arret formalises dans
  `docs/v9_execution_contract.md`. Les descriptions V6/V7/V8 sont explicitement
  historiques et ne pilotent plus les travaux. *(commit GitHub: `72d2749`)*
- **Benchmark open-set, ablation cross-encoder et gates V9**: feuille
  d'adjudication stratifiee, validation humaine/evidence/snapshot obligatoire,
  gel adresse par hash, cross-encoder top-20 avec revision epinglee, gates
  retrieval/segments/latence/deploiement et guide d'execution. Les trois
  variantes cross-encoder produisent des predictions OOF compatibles avec le
  meme accepteur. *(commits GitHub: `c4cf99f`, `b82271e`)*
- **Ranker unique + accepteur selectif V9**: 54 features brutes partagees
  train/serve puis sous-ensemble manifeste, features retrieval/SIREN, ranker
  XGBoost avec predictions OOF, misses conserves, correction stricte SIRET,
  calibration et selection de seuil sur deux moities dev distinctes, comparaison
  logistique/XGBoost, moteur d'inference `AUTO_MATCH|REVIEW` compatible
  `routing_status`. L'injection de positifs est autorisee uniquement dans le fit
  ranker train et interdite dans les scenes/evaluations. *(commit GitHub:
  `db4ab27`)*
- **Retrieval hybride V9 a budget fixe**: RRF sparse/dense/rescue, vrais scores
  TF-IDF ordonnes, provenance/rangs par canal, configurations 50 et ablation 100,
  index dense global SIREN streaming avec manifeste/tokenizer, expansion limitee
  SIRET et benchmark p50/p95. *(commit GitHub: `36404ae`)*
- **Contrats et dataset canonique V9**: ajout du contrat public `AUTO_MATCH/REVIEW` avec mapping legacy, labels `MATCH_EXACT/NO_MATCH/AMBIGUOUS/UNRESOLVED`, split deterministe SIREN-disjoint, bundle parquet immuable adresse par hash, manifeste de provenance/config/tokenizer/features et registre explicite des artefacts legacy interdits aux entrypoints V9. *(commit GitHub: `afb0f3d`)*
- **Socle V9 semantique + prediction selective**: chargement lazy de SentenceTransformer, reparation runtime du tokenizer Unigram exporte comme BertTokenizer, healthcheck anti-`<unk>`, injection semantique partagee train/serve, remise en service du mining d'homonymes geographiques et primitives testees de courbe risque-couverture/certification binomiale. Suite de tests retablie a 20 tests passants. *(commit GitHub: `fcfc33f`)*
- **Spikes architecture neurale (cross-encoder + dual-encoder)**: benchmark reproductible sur un holdout SIREN-disjoint de 400 requetes. Le cross-encoder court ne remplace pas XGBoost (51,75% vs 85,25% Hit@1 sur les memes scenes). Le dual-encoder structure atteint 74,50% Recall@1 et 96,00% Recall@50; l'union TF-IDF top-50 + dense top-50 atteint 99,25% Recall@50 (8 des 11 misses TF-IDF recuperes). Le modele semantic exporte declare a tort `BertTokenizer`; le chargement actuel via SentenceTransformer produit excessivement des tokens `<unk>`, donc les anciens benchmarks semantiques doivent etre revalides apres correction. *(commit GitHub: `7640772`)*
- **V8 features + hard negatives + hyperparams decider**: ajout de 7 features d'interaction, extension des hard negatives colocataires/homonymes/siblings, tuning decider (`lr=0.05`, `max_depth=7`, `400 rounds`). *(commit GitHub: `35fb441`)*
- **Route B (SIREN-first) implementee**: nouvel index global SIREN, nouveau module de retrieval SIREN, branchement conditionnel dans l'inference profile/engine. *(commit GitHub: `3e090b7`)*
- **Correctifs bloquants Route B**: fix DuckDB `:memory:`, fix champ CRM nom, fix filtre closed/open, ajout CLI `--siren-index` dans le generateur de samples. *(commit GitHub: `c356923`)*
- **Branchement Route B dans le retrieval partage (training)**: `build_candidate_pool()` supporte Route B via indices SIREN, propagation sequentielle + multiprocess dans `generate_training_samples_v5fast.py`. *(commit GitHub: `1305012`)*
- **Implementation V8b SIREN expansion (V7 + local + cross-partition)**: ajout Step 5 d'expansion apres prefilter, feature flag, cap pool dedie et telemetrie d'expansion. *(commit GitHub: `9c0e806`)*
- **Correctifs critiques V8b**: exclusion explicite Route B quand expansion activee, filtres metier expansion, recalc GT coverage/loss reason post-expansion. *(commit GitHub: `f1fbbb8`)*
- **Fix expansion SIREN en mode geo-only**: chargement des index dissocie (global vs geo) dans le generateur de samples pour eviter la desactivation silencieuse de l'expansion quand seul `siren_to_geo.parquet` est present. *(commit GitHub: `c961371`)*

## Historique structurant (deja en place)
- **Retrieval hybride sparse+dense + cache TF-IDF persistant + timing**: integration du socle P0/P1. *(commit GitHub: `9ab297e`)*
- **Ablation dense-only corrigee + flag sparse explicite**: alignement des modes retrieval et signature de config. *(commit GitHub: `35fc3a3`)*
- **Defaults partitions V7 + manifest INSEE O(1)**: bascule des chemins/scripts vers `data/candidates_v7_all`. *(commit GitHub: `a309a7c`)*
- **Priorisation mega-communes embeddings**: orchestration dense amelioree pour runs longs. *(commit GitHub: `66b5b87`)*

## Fichiers modifies recemment
- `docs/benchmark_v3_evidence_policy.md` *(commit GitHub : `09b9d46`)*
- `docs/retrieval_selective_recall100_contract.md` *(commit GitHub :
  `4f6e317`)*
- `scripts/build_benchmark_v3_evidence.py` *(commits GitHub : `cf7133c`,
  `c6c8186`, `eb0e6a3`)*
- `scripts/certify_selective_retrieval_test.py` *(commits GitHub : `d1c0fc9`,
  `6fab035`)*
- `scripts/audit_v3_qualification_stability.py`,
  `tests/test_v3_qualification_stability.py` *(commit GitHub : `111b07c`)*
- `reports/recall100/selective_test_certification.md` *(commit GitHub :
  `41ff2e1`)*
- `reports/recall100/final_go_pivot_stop.md` *(rapport dev historique marqué
  supersédé, commit GitHub : `50e804b`)*
- `scripts/audit_downstream_architecture.py`,
  `tests/test_downstream_architecture_audit.py` *(commits GitHub :
  `a59fb0f`, `aeeaf0f`)*
- `reports/v9/downstream_architecture_audit.md` *(commit GitHub :
  `e186439`)*
- `docs/downstream_selective_matching_contract.md` *(commit GitHub :
  `c18bf28`)*
- `docs/downstream_acceptor_e2b_contract.md` *(commit GitHub : `cf91432`)*
- `docs/benchmark_v4_current_snapshot_policy.md` *(commit GitHub :
  `ce82b01`)*
- `docs/v4_fresh_expansion_contract.md` *(commit GitHub : `1c2e84c`)*
- `docs/v4_retrieval_reconstruction_contract.md` *(commit GitHub :
  `510868b`)*
- `docs/v4_ranker_e1_contract.md` *(commit GitHub : `0c90c25`)*
- `docs/v4_acceptor_e2_contract.md` *(commit GitHub : `9a22fd8`)*
- `docs/v4_final_holdout_contract.md` *(commit GitHub : `fb6a20c`)*
- `scripts/build_benchmark_v4_current_snapshot.py`,
  `tests/test_benchmark_v4_current_snapshot.py` *(commit GitHub :
  `799c32d`)*
- `scripts/build_v4_fresh_expansion.py`,
  `tests/test_v4_fresh_expansion.py` *(commit GitHub : `613cf7d`)*
- `scripts/prepare_v4_retrieval_inputs.py`,
  `scripts/finalize_v4_retrieval_gate.py`,
  `tests/test_v4_retrieval_gate.py` *(commit GitHub : `e566c25`)*
- `scripts/build_v4_ranker_dataset.py`,
  `tests/test_v4_ranker_dataset.py` *(commit GitHub : `6236365`)*
- `scripts/evaluate_v4_ranker_e1.py`,
  `tests/test_v4_ranker_e1.py` *(commit GitHub : `250a05f`)*
- `scripts/prepare_v4_ambiguous_retrieval.py`,
  `tests/test_v4_ambiguous_retrieval.py` *(commit GitHub : `af5ce0b`)*
- `scripts/build_v4_acceptor_dataset.py`,
  `tests/test_v4_acceptor_dataset.py`, `scripts/train_v9_acceptor.py`,
  `src/xgb_matcher/v9_scene.py` *(commit GitHub : `9ec88c8`)*
- `scripts/freeze_v4_final_holdout.py`,
  `scripts/prepare_v4_final_holdout.py`,
  `scripts/evaluate_v4_final_holdout.py`,
  `tests/test_v4_final_holdout.py` *(commit GitHub : `8cc9bfa`)*
- `scripts/repair_v4_final_verdict.py` et correction des booléens du runner
  *(commit GitHub : `aead6f5`)*
- `scripts/build_downstream_selective_dataset.py`,
  `tests/test_downstream_selective_dataset.py` *(commits GitHub :
  `0a75b73`, correction des partitions overlay `fc9cb1b`)*
- `scripts/train_v9_ranker.py`, `src/xgb_matcher/v9_scene.py`,
  `scripts/train_v9_acceptor.py`, `src/xgb_matcher/v9_acceptor.py`
  *(commits GitHub : `aeeaf0f`, `0a75b73`, `dbd8906`)*
- `reports/v9/downstream_e1_e2_results.md` *(commit GitHub : `9ab7f6a`)*
- `reports/v9/downstream_e2b_results.md` *(commit GitHub : `ebb4bf2`)*
- `reports/v9/benchmark_v4_current_snapshot_results.md` *(commit GitHub :
  `299bc8a`)*
- `reports/v9/v4_fresh_expansion_results.md` *(commit GitHub :
  `d8d36b9`)*
- `reports/v9/v4_retrieval_gate_results.md` *(commit GitHub : `6948aa1`)*
- `reports/v9/v4_ranker_e1_results.md` *(commit GitHub : `13e3547`)*
- `reports/v9/v4_acceptor_e2_results.md` *(commit GitHub : `ff1eea4`)*
- `reports/v9/v4_final_holdout_results.md` *(commit GitHub : `4aade83`)*
- `src/xgb_matcher/features.py` *(commits GitHub: `35fb441`, `fcfc33f`, `db4ab27`)*
- `scripts/generate_training_samples_v5fast.py` *(commits GitHub: `35fb441`, `c356923`, `1305012`, `c961371`, `fcfc33f`, `db4ab27`)*
- `scripts/train_xgb_decider.py` *(commit GitHub: `35fb441`)*
- `scripts/build_siren_global_index.py` *(commits GitHub: `3e090b7`, `c356923`)*
- `src/xgb_matcher/siren_retrieval.py` *(commit GitHub: `3e090b7`)*
- `src/xgb_matcher/infer.py` *(commits GitHub: `3e090b7`, `c356923`, `36404ae`)*
- `src/xgb_matcher/retrieval.py` *(commits GitHub: `1305012`, `9c0e806`, `f1fbbb8`, `36404ae`)*
- `src/xgb_matcher/retrieval_config.py` *(commits GitHub: `3e090b7`, `9c0e806`, `36404ae`)*
- `src/xgb_matcher/profile.py` *(commit GitHub: `3e090b7`)*
- `src/xgb_matcher/v9_dataset.py` *(commits GitHub: `afb0f3d`, `db4ab27`)*
- `src/xgb_matcher/v9_scene.py`, `v9_acceptor.py`, `v9_infer.py`
  *(commit GitHub: `db4ab27`)*
- `src/xgb_matcher/fusion.py`, `v9_features.py` *(commit GitHub: `36404ae`)*
- `src/xgb_matcher/v9_adjudication.py`, `v9_cross_encoder.py`,
  `v9_evaluation.py` *(commit GitHub: `c4cf99f`)*

## Travail en cours
- Aucun run long n'est en cours. Les canaux train, le dataset aval, le ranker
  E1, les accepteurs E2/E2b, V4, V4-Fresh et le gate retrieval V4 sont publiés
  sur le SSD.
- V4.10b est close sans modèle promu. Le prochain travail autorisé est un
  nouveau contrat d'architecture alignant de façon homogène retrieval,
  prédictions ranker hors échantillon et scènes accepteur. Les 94 cas
  difficiles V4.10b sont consommés et ne peuvent plus valider cette
  architecture. Ni random V4.8, ni locked, ni test final ne doivent être
  rouverts.
- Le registre V4.11-A est gelé. Les 225 lignes `UNSEEN` ne doivent pas être
  ouvertes avant gel du candidat V4.11 et ne peuvent servir qu'à un challenge
  descriptif, pas à une preuve représentative.
- Le contrat V4.11-B est gelé. Toute implémentation doit rester aveugle au
  SIRET/SIREN CRM, reconstruire le retrieval avant les labels et respecter
  les ordres ranker 45 / accepteur 80.
- Le garde-fou V4.9 de fonction de site est clos par
  `STOP_SITE_FUNCTION_GUARD`. Il ne doit pas être retouché sur les 172 cas
  consommés.
- Le test final historique et le holdout V4-Fresh ont chacun été lus une fois
  et sont maintenant définitivement fermés à toute nouvelle variante, règle
  ou seuil.
- E1 historique est conservé comme baseline. Le nouveau ranker V4 est validé
  sur `dev_new`, mais aucun modèle produit n'est déployé.
- V4-Fresh a validé définitivement le retrieval V4 et le ranker. V4.8 a
  invalidé le nouvel accepteur sur le random et V4.9 a invalidé le garde-fou
  lexical comme piste assez large. La prochaine étape est un diagnostic
  structuré des 31 erreurs/ambiguïtés non interceptées, sans tuning.

## Points d'attention
- **Plafond absolu 100**: les mesures @200/@500 sont diagnostiques et ne
  constituent jamais une configuration eligible.
- **Test final consommé** : ne plus lancer de variante, analyser de miss pour
  choisir une règle, changer de seuil ou modifier la qualification sur ce
  split. Toute évolution nécessite un nouveau holdout indépendant.
- **Portée du 99,436 %** : Recall candidat sur les 80,241 % de dossiers V3
  exacts, pas précision `AUTO_MATCH` et pas taux d'automatisation global.
- **Modèles historiques gelés** : aucun modèle legacy n'est modifié. La phase
  aval E1/E2 est ouverte sous
  `docs/downstream_selective_matching_contract.md`.
- **Decision PIVOT scopee**: elle invalide l'admission/fusion des candidats
  denses V9 testes, pas leur signal de scoring ni le pipeline sparse/XGBoost.
- **Comparaison retrieval uniquement a budget constant**: un gain avec 100
  candidats ne justifie pas la promotion de la variante 50.
- **Precision strictement SIRET**: un bon SIREN mais mauvais etablissement est
  une erreur pour l'accepteur.
- **UNRESOLVED n'est pas un négatif prouvé** : le traiter comme faux match
  pendant l'apprentissage crée du bruit de cible. Il doit rester hors du fit
  tant qu'une validation indépendante ne lui attribue pas `NO_MATCH`,
  `AMBIGUOUS` ou `MATCH_EXACT`.
- **Date de vérité absente** : V4 fixe désormais explicitement la politique
  « actif au snapshot ». Elle ne peut pas servir à reconstruire un exploitant
  historique sans date CRM.
- **Split historique invalidé par les nouvelles vérités** : 14 SIREN V4 exacts
  étaient partagés entre train et dev. L'ancien dev est abandonné pour la
  nouvelle cible ; V4-Fresh fournit un nouveau dev et un holdout sans SIREN
  exact partagé avec le fit.
- **Holdout Fresh scellé** : aucune génération de candidats, prédiction ou
  mesure nouvelle ne doit désormais réutiliser `holdout_sealed`, consommé par
  l'évaluation finale `7dbd5527374ca0d4`.
- **Deux erreurs finales explicables** : `AMBIGUOUS` doit être routé `REVIEW`
  avant l'accepteur ; un top1 fermé ne doit pas être automatisé dans la cible
  V4 actif-courant. Ces règles sont des hypothèses post-holdout et exigent une
  nouvelle validation indépendante.
- **Calibration saturante** : l'isotonic retenu par tie-break donne 1,0 aux
  282 AUTO finaux. Ne pas présenter ce score comme une probabilité fiable.
- **NO_MATCH temporel**: toujours rattache au snapshot SIRENE et a la date de
  reference.
- **Cross-encoder conditionnel**: aucune promotion sans +1 point de couverture
  a precision cible et gates segments/latence. Il reste hors chemin critique et
  aucune location de GPU n'est autorisee.
- **Certification**: avant environ 2 300 AUTO independants audites sans erreur,
  publier une estimation observee, jamais une garantie a 99,8 %.
- **Governance docs**: garder `handover.md` comme journal de commits (regle AGENTS).

## Artefacts cibles (V9)
| Artefact | Chemin |
|----------|--------|
| Partitions candidates | `data/candidates_v7_all/` |
| Bundle canonique | `data/v9/<build_id>/{queries,labels,candidates}.parquet` |
| Manifeste dataset | `data/v9/<build_id>/manifest.json` |
| Mapping geo SIREN | `data/siren_index/siren_to_geo.parquet` |
| Index dense SIREN | `data/v9_indices/siren_dense_<snapshot>/` |
| Ranker + predictions OOF | `models/v9/ranker_<build_id>/` |
| Accepteur + calibration | `models/v9/acceptor_<build_id>/` |
| Benchmark open-set gele | `data/v9_open_set/<benchmark_id>/` |
| Dataset aval E1/E2 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/downstream/3171ef5020c0f068/` |
| Ranker E1 expérimental | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/ranker_3171ef5020c0f068_fc9cb1b/` |
| Accepteur E2 refusé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/acceptor_3171ef5020c0f068_fc9cb1b/` |
| Accepteur E2b refusé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/acceptor_e2b_3171ef5020c0f068_070c123/` |
| Qualification V4 refusée | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v4/0b333d33a56ed759/` |
| Expansion V4-Fresh passée | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/v4_fresh_expansion/14047b719ef90f6f/` |
| Gate retrieval V4 passé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/retrieval_v4/ddefe3daaacdf5ef/` |
| Dataset ranker V4 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/ranker_v4/1aebeada820d92a7/` |
| Ranker V4 validé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/ranker_v4/ranker_1aebeada820d92a7_6236365/` |
| Dataset accepteur V4 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/acceptor_v4/2b8a9c994e0944be/` |
| Accepteur V4 validé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/acceptor_v4/acceptor_2b8a9c994e0944be_9ec88c8/` |
| Autorisation finale V4 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/releases/v4_final/7dbd5527374ca0d4/authorization.json` |
| Première évaluation finale V4 | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/final_evaluations/v4/7dbd5527374ca0d4/` |
| Verdict final V4 corrigé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/final_evaluations/v4/7dbd5527374ca0d4_verdict_repair/` |
| Dataset V4.11 input-blind | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_input_blind/ec4326ec57e4411d/` |
| Ranker C V4.11 validé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_ranker_c/e13eb3ac7498256e/` |
| Scènes accepteur V4.11 validées | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_11_acceptor/52ea3faba9a56aff/` |
| Candidat accepteur V4.11 gelé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/v4_11_acceptor/9d23bf3deb6b63de/` |
| CRM challenge V4.11 assaini | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_sanitized/1c994c852c10acaf/` |
| Labels challenge V4.11 gelés | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/v4_11_unseen_qualification/4f9ef46516b89ab8/` |

## Prochaines etapes
1. Ne plus réutiliser le test historique, le holdout V4-Fresh, le random V4.8
   ni les 172 cas V4.9 pour sélectionner ou valider une variante.
2. Produire une taxonomie descriptive, sans règle de décision, des 31 erreurs
   ou ambiguïtés fiables que V4.9 ne refuse pas. **Terminé en V4.10.**
3. Préenregistrer puis construire une matrice accepteur unique qui conserve
   les relations entrée/candidat, l'état, la provenance, la forme juridique,
   l'activité/fonction, les interactions nom/adresse et la concurrence
   intra-SIREN complète. **Terminé : `GO_TRAIN_V410`.**
4. Comparer une régression logistique et un XGBoost peu profond sans modifier
   le retrieval V4.2-B ni le ranker A, avec OOF par composante et reproduction
   exacte du baseline. **Terminé : `PIVOT_STRUCTURED_FEATURES`.**
5. Toute décision de promotion exigera une population fraîche,
   indépendante et disjointe ; le test final reste fermé.
6. Conserver séparément le chantier qualification/réparation CRM : la
   couverture source 22,454 % reste très loin du gate de 80 % et ne se corrige
   pas par le retrieval.
7. Le challenge descriptif V4.11 est consommé et conclut
   `PIVOT_ACCEPTOR_EVIDENCE_GATE`. Préenregistrer V4.12 sur les anciennes
   populations uniquement : garde multi-SIREN forte, nouvelles features
   d'unicité, comparaison garde seule puis accepteur. Geler le candidat avant
   tout nouvel export CRM indépendant, indispensable à une décision produit.

---
*Regle projet: chaque modification de code/metier doit citer son commit GitHub dans ce document.*
