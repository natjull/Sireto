# V4.12 — Pré-vol lecture seule du locator producteur S1

## Portée

Le pré-vol prouve uniquement que le locator Keychain fermé par le plan ne
contient aucun item avant l'autorisation one-shot. Il ne crée, ne modifie et
ne supprime aucun item Keychain. Il ne lit aucun CRM, label, candidat, modèle
ou preuve. Ses deux seules écritures possibles sont le claim de réservation
et, sur `-25300`, le reçu non secret fermés par le plan.

## Requête

La requête `SecItemCopyMatching` contient exactement :

```text
kSecClass = kSecClassGenericPassword
kSecAttrService = com.sireto.v412.fresh-s1-producer-ed25519
kSecAttrAccount = SIRETO
kSecAttrSynchronizable = false
kSecUseDataProtectionKeychain = true
kSecUseAuthenticationUI = kSecUseAuthenticationUIFail
kSecMatchLimit = kSecMatchLimitOne
```

Elle n'inclut ni `kSecReturnData`, ni `kSecReturnAttributes`, ni buffer de
sortie. Le seul succès est l'OSStatus numérique `-25300`
(`errSecItemNotFound`). Tout autre statut, y compris item présent, produit
`STOP` sans sérialiser de métadonnée.

La requête est sérialisée en JSON UTF-8 canonique avec clés triées,
séparateurs `,` et `:`, valeurs non-ASCII conservées, nombres non finis
interdits et un unique LF final. Son SHA-256 attendu est
`0d5d2fe817391a4d91e51a57b3eaa447cad932c8c081b64a8b630bdc566fb96f`.

## Autorités et résultat

Le code et ses tests ne peuvent être exécutés qu'après deux audits
`GO_PREFLIGHT_IMPLEMENTATION_NEXT_LOCK`. Un lock d'exécution propre au
pré-vol épingle ensuite le commit, les quatre blobs code/tests/sealer/tests,
le plan, le contrat, le lock producteur, le runtime, l'UID, les chemins et le
hash de requête. Deux audits matériels
`GO_PREFLIGHT_LOCK_MATERIAL_NEXT_RUN` sont requis avant l'appel réel.

Avant toute réservation, le script valide ces autorités byte-for-byte et
exige l'absence de l'autorisation producteur, de la racine producteur et de
son claim. `ABSENT` signifie exclusivement qu'un parcours ancré sur des
descripteurs, composant par composant avec `openat(O_DIRECTORY|O_NOFOLLOW)`
puis `fstatat(..., follow_symlinks=false)`, rencontre `ENOENT`. Tous les
parents ouverts doivent être de vrais répertoires et leur identité doit rester
stable. Une entrée existante, un lien symbolique valide ou pendant, un parent
symbolique ou remplacé, une erreur de permission ou tout autre errno produit
`STOP`.

Le lock de pré-vol ferme les 15 champs racine par type et constante/égalité,
les 9 champs d'implémentation et les 9 champs du runtime. Le runtime doit être
identique à celui du lock producteur déjà scellé. Tout champ supplémentaire,
manquant, mal typé ou divergent produit `STOP` avant l'appel natif. Le
plan préenregistre une matrice obligatoire couvrant chaque garde sous forme de
fichier, répertoire, lien valide ou pendant et erreur de parcours, chaque état de
reprise, et les mutations extra/manquante/type/valeur du claim et du lock
imbriqué. Chaque cas de garde et de cycle de vie, y compris le replay valide
claim+reçu, possède une attente explicite de zéro appel natif. Le
résultat canonique contient exactement :

```text
schema_version
verdict
authority_execution_lock_sha256
query_sha256
osstatus
logical_time_utc
preflight_plan_sha256
preflight_execution_lock_sha256
implementation_commit
implementation_sha256
implementation_tests_sha256
```

Le verdict unique est `KEYCHAIN_LOCATOR_ABSENT`, `osstatus` vaut `-25300`.
Les champs stables sont fermés par type et constante dans le plan :

```text
schema_version = sireto-v4.12-fresh-s1-local-producer-keychain-preflight-result-2
verdict = KEYCHAIN_LOCATOR_ABSENT
authority_execution_lock_sha256 = 78665f07bdcee12cfdd3989c7e7c55dd3ac625571181b1b2b6a52ea98f54954b
query_sha256 = 0d5d2fe817391a4d91e51a57b3eaa447cad932c8c081b64a8b630bdc566fb96f
osstatus = -25300
logical_time_utc = 2026-07-31T00:00:00Z
```

Les cinq champs d'identité restants sont fermés par égalité au plan canonique
ou au lock de pré-vol : hash du plan, hash du lock de pré-vol, commit, hash du
programme et hash de ses tests.

## Cycle one-shot et crash

Après validation, un claim canonique `KEYCHAIN_QUERY_RESERVED` est écrit en
`0600`, `O_EXCL`, `O_NOFOLLOW`, avec `fsync`, `F_FULLFSYNC`,
synchronisation du parent et relecture exacte. L'appel natif n'a lieu qu'après
ce claim durable et une seule fois. Sur `-25300`, le reçu est écrit avec les
mêmes garanties.

Un claim valide accompagné d'un reçu valide retourne uniquement le reçu
stocké, sans nouvel appel Keychain. Tout claim seul, reçu seul, contenu
partiel, non canonique, corrompu ou divergent produit `STOP` sans nouvel
appel, réécriture ni suppression. Un crash après réservation est donc classé
indéterminé et ne peut jamais provoquer une seconde requête.

Ce pré-vol n'autorise pas le provisionnement. Après son audit matériel, une
autorisation canonique distincte devra encore être committée et deux audits
`GO_S1_LOCAL_PRODUCER_PROVISION` devront être obtenus.
