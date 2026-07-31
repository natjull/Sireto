# V4.12 — Pré-vol lecture seule du locator producteur S1

## Portée

Le pré-vol prouve uniquement que le locator Keychain fermé par le plan ne
contient aucun item avant l'autorisation one-shot. Il ne crée, ne modifie et
ne supprime rien. Il ne lit aucun CRM, label, candidat, modèle ou preuve.

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

## Autorités et résultat

Avant la requête, le script reconstruit le lock attendu avec le sealer audité
et exige une égalité byte-for-byte avec le lock scellé. Le résultat canonique
contient exactement :

```text
schema_version
verdict
execution_lock_sha256
query_sha256
osstatus
logical_time_utc
```

Le verdict unique est `KEYCHAIN_LOCATOR_ABSENT`, `osstatus` vaut `-25300`.
Le reçu est écrit en `0600`, `O_EXCL`, `O_NOFOLLOW`, avec `fsync`,
`F_FULLFSYNC`, synchronisation du parent et relecture exacte. Aucun rerun
divergent, overwrite ou suppression n'est autorisé.

Ce pré-vol n'autorise pas le provisionnement. Après son audit matériel, une
autorisation canonique distincte devra encore être committée et deux audits
`GO_S1_LOCAL_PRODUCER_PROVISION` devront être obtenus.
