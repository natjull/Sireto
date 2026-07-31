# V4.12 — Arrêt du producteur local S1 V1

## Verdict

`PIVOT_FILE_BASED_KEYCHAIN_V2`

Le run V1 autorisé s'est arrêté sur `SecItemAdd` avec l'OSStatus observé
`-34018` (`errSecMissingEntitlement`). Aucun second essai V1 n'est autorisé
sans changement externe des entitlements ; aucune suppression ou réécriture
des artefacts V1 n'est permise.

## État matériel durable

- root V1 : présent, mode `0700`, UID 501 ;
- claim V1 : présent, canonique, mode `0600`,
  SHA-256 `63970c7048cb7411f23200c11e4a353c5350f80c94f87da21f4abda133b983f3` ;
- état du claim : `CLAIMED_BEFORE_KEYCHAIN` ;
- répertoire `authorities` : vide ;
- receipt, payload, seal et genesis : absents.

Le statut `-34018` provient du transcript terminal du run. Il n'est pas
présent dans un receipt V1, car le schéma V1 ne préenregistrait qu'un receipt
de succès. La preuve durable établit donc strictement « claim créé avant
Keychain, aucune autorité publique produite ».

## Cause

V1 impose `kSecUseDataProtectionKeychain=true` et
`kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`. Le processus hôte Python
est signé ad hoc, sans Team ID ni entitlement. Apple documente que le
Data Protection Keychain macOS tire ses groupes d'accès des entitlements du
processus principal et que ces entitlements doivent être autorisés par un
profil de provisioning. Un simple wrapper lançant ce Python ne transmettrait
pas ses entitlements au processus enfant.

Références officielles :

- [TN3137 — On Mac keychain APIs and implementations](https://developer.apple.com/documentation/Technotes/tn3137-on-mac-keychains)
- [Troubleshooting -34018 Keychain Errors](https://developer.apple.com/forums/thread/114456)
- [kSecAttrAccessible](https://developer.apple.com/documentation/security/ksecattraccessible)

## Décision V2

V2 utilisera le Keychain macOS traditionnel via `SecItem`, sans
`kSecUseDataProtectionKeychain`, sans `kSecAttrAccessible` et sans
synchronisation. Elle conservera le SHA-256 du claim dans `kSecAttrGeneric`
et fermera explicitement le Keychain d'ajout, la search list et l'ACL
`SecAccess`.

V2 aura un service, un key id, un root, un claim, un plan, un lock, un
pré-vol et une autorisation distincts. Elle référencera le hash du claim V1
comme provenance de supersession, sans le reprendre, le déplacer ni
l'effacer.

Le threat model V2 exclut un processus arbitraire déjà maître du même UID.
Cette limite est explicite : le Keychain traditionnel utilise des ACL, pas
l'isolation `ThisDeviceOnly` du Data Protection Keychain. Aucun fallback vers
une graine privée en fichier `0600` n'est autorisé.

Avant tout nouvel accès Keychain :

1. préenregistrer le contrat et le plan V2 ;
2. obtenir deux audits indépendants ;
3. adapter le backend et ses faux frameworks ;
4. tester crash, reprise, concurrence, ACL et projection fermée ;
5. sceller et auditer un nouveau lock ;
6. effectuer un pré-vol status-only sur le locator V2 ;
7. committer une autorisation V2 distincte ;
8. obtenir deux `GO` avant l'unique run V2.
