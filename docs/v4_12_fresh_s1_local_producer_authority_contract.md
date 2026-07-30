# V4.12 — Autorité locale du producteur S1

## 1. Portée

Cette autorité fournit au futur exporteur local S1 une identité, une clé de
signature Ed25519 et un ledger d’exports monotone. Elle est créée avant toute
inbox et sans lire aucun CRM, label, preuve, candidat, modèle ou résultat.

Elle ne crée aucune vérité métier. Plus tard, l’exporteur pourra signer un
mapping contractuel réellement présent dans l’export source ; en son absence,
la ligne restera `UNRESOLVED` et la couverture pourra conduire à `PIVOT`.

Le succès de ce milestone autorise uniquement la construction des catalogues
S1. Il n’autorise ni ouverture CRM, ni retrieval, ni modèle.

## 2. Identité fixe

- `producer_id` : `SIRETO_LOCAL_CRM_EXPORT_PRODUCER_V1` ;
- `source_system` : `LOCAL_CRM_EXPORT_V1` ;
- `portfolio_id` : `SIRETO_FRESH_HOLDOUT_V1` ;
- sémantique record :
  `V411_SERVICE_ID_NORM_EQUIVALENT_SOURCE_RECORD_ID_V1` ;
- `producer_key_id` :
  `SIRETO_V412_FRESH_S1_LOCAL_PRODUCER_ED25519_V1` ;
- `producer_export_ledger_id` :
  `SIRETO_V412_FRESH_S1_LOCAL_EXPORT_LEDGER_V1` ;
- prochaine séquence après genesis : `1`.

La future entrée CRM ne pourra pas remplacer ces valeurs.

## 3. Clé privée Keychain

La clé privée est une graine Ed25519 aléatoire de 32 octets produite par
`os.urandom`. Elle est stockée comme generic password dans le Keychain macOS :

- service : `com.sireto.v412.fresh-s1-producer-ed25519` ;
- account : `SIRETO` ;
- lecture via `SecItemCopyMatching` ;
- création via `SecItemAdd` ;
- UI forcée à `kSecUseAuthenticationUIFail`.

Le binaire n’utilise jamais la commande `security`, un argument, une variable
d’environnement, stdin, un fichier temporaire ou un log pour le secret.
Seuls la clé publique brute de 32 octets, son Base64 canonique et ses hashes
sortent du processus. La copie mutable de la graine est écrasée après usage.

Un item préexistant sans claim/receipt correspondant produit
`STOP_FOREIGN_KEYCHAIN_ITEM`. Un item créé après le claim mais avant un crash
est relu pour terminer le même attempt.

## 4. One-shot et reprise

Le provisioner n’accepte aucun argument. Il charge le plan fixe, vérifie tous
ses pins, puis crée avec `O_EXCL` le claim fixe :

```text
<root>/claims/provision.claim.json
```

Le claim est `fsync`/`F_FULLFSYNC`, puis son répertoire est synchronisé avant
tout accès Keychain. Deux processus concurrents ne peuvent donc pas créer deux
autorités.

- receipt valide présent : retour idempotent, sans nouvel accès secret ;
- claim présent, receipt absent, item présent : reprise du même attempt ;
- claim présent, receipt absent, item absent : création de la clé puis suite ;
- item présent sans claim : `STOP_FOREIGN_KEYCHAIN_ITEM` ;
- divergence entre item, clé publique, genesis, seal ou receipt : `STOP`.

Aucun artefact durable n’est écrasé ou supprimé.

## 5. Genesis signé

L’entrée genesis contient exactement :

```text
schema_version
producer_id
producer_export_ledger_id
producer_export_sequence
producer_export_previous_entry_sha256
entry_kind
producer_key_id
public_key_sha256
logical_time_utc
signature_base64
```

La séquence vaut `0`, le précédent vaut `null`, `entry_kind` vaut `GENESIS`.
La signature Ed25519 porte sur le JSON canonique sans `signature_base64`. Le
hash des octets canoniques signés devient le head du ledger ; la prochaine
séquence vaut `1`.

## 6. Autorité publique et seal

`producer_authority_payload.json` contient identité producteur, clé publique,
locator Keychain non secret, head du ledger, prochaine séquence, autorités
S1, commit/blobs d’implémentation et runtime exacts. Il ne contient ni clé
privée, ni hash de clé privée, ni son propre hash.

`producer_authority_seal.json` contient exactement :

```text
schema_version
authority_id
payload_size_bytes
payload_sha256
ledger_genesis_size_bytes
ledger_genesis_sha256
```

`authority_id` est dérivé par SHA-256 domain-separated sur producteur, key id,
hash de clé publique et ledger id, puis mappé vers 64 lettres `[a-p]`.

Toutes les écritures sont privées `0700/0600`, `umask 0077`, `O_EXCL`,
`fsync`, `F_FULLFSYNC`, synchronisation du répertoire parent et vérification
par FD ancré avant transition.

## 7. Gate d’implémentation

Avant provisionnement :

- plan et contrat canoniques et cross-pinnés ;
- deux audits indépendants `GO_S1_LOCAL_PRODUCER_IMPLEMENTATION` ;
- code/test committés et leurs blobs épinglés dans un lock séparé ;
- tests Keychain mockés, signature golden vector, crash/reprise, concurrence,
  item étranger, idempotence, permissions, liens et mutations ;
- gate réel en lecture seule prouvant que le locator Keychain est absent ;
- suite complète verte.

Après code et lock, deux audits doivent rendre
`GO_S1_LOCAL_PRODUCER_PROVISION`. Une unique autorisation canonique est ensuite
committée avant l’unique run réel.
