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
`GO_PREFLIGHT_RACE_AMENDMENT_NEXT_IMPLEMENTATION_AUDIT`, puis deux audits
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
claim+reçu, possède une borne explicite d'appels natifs. Cette borne est zéro,
sauf pour les deux fenêtres concurrentes irréductibles définies ci-dessous. Le
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

## Limite concurrente explicite

Aucune suite finie de contrôles de chemin en espace utilisateur ne peut
empêcher un autre processus du même UID de renommer un parent exactement
entre la dernière instruction de contrôle et l'instruction suivante. Le plan
sépare donc les deux garanties réellement démontrables :

| Cas | Appels natifs maximum | Reçu |
|---|---:|---|
| `STATE_NAMESPACE_REPLACEMENT_IN_FINAL_INSTRUCTION_WINDOW` | 1 | interdit |
| `GUARD_NAMESPACE_REPLACEMENT_AFTER_FINAL_REVALIDATION` | 1 | peut exister |

Dans le premier cas, la revalidation du magasin d'état ancré détecte le
remplacement après l'appel : l'exécution produit `STOP`, sans reçu, et laisse
au plus un claim seul.

Dans le second cas, un processus non coopératif du même UID modifie une garde
après sa toute dernière revalidation. Cette modification peut être
inobservable avant l'écriture du reçu. Le reçu prouve alors uniquement que le
locator était absent au moment précis de la requête status-only ; il ne prouve
pas que les gardes sont restées absentes ensuite et n'autorise jamais le
provisionnement. Toute étape ultérieure doit revalider l'autorisation, la
racine, le claim et l'ordre des autorités sous son propre protocole.

Dans les deux cas, il y a au plus un appel `SecItemCopyMatching`, toujours
avec un pointeur résultat nul : aucune donnée ni aucun attribut n'est
retourné. Les écrivains coopératifs devront respecter le futur protocole
commun d'autorisation et de verrouillage. La manipulation concurrente du
namespace par un processus non coopératif du même UID est explicitement hors
de la garantie d'ordonnancement séquentiel.

Tous les remplacements observables avant l'appel restent bornés à zéro appel.
Le déplacement ou la suppression volontaire des autorités persistantes par
un processus non coopératif du même UID entre deux exécutions est hors du
modèle ; le code n'en déduit jamais un reçu valide pendant l'exécution
courante.

Ce pré-vol n'autorise pas le provisionnement. Après son audit matériel, une
autorisation canonique distincte devra encore être committée et deux audits
`GO_S1_LOCAL_PRODUCER_PROVISION` devront être obtenus.
