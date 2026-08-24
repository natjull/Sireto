# Tombstone terminal du benchmark physique retrieval v13

Date du constat : 24 août 2026

Verdict terminal : `STOP_V13_EXECUTION_EXCEPTION`
Statut de la tentative : `SEALED_TERMINAL_STOP_V13_NO_RETRY`

Ce document enregistre le résultat réellement produit par l'unique tentative
W3 v13. Il ne corrige, ne requalifie et ne rend pas réutilisable cette
tentative. Le tree externe v13, son marqueur one-shot et son manifeste `STOP`
restent inchangés. Aucun benchmark, entraînement, OOF national ou fold
supplémentaire n'a été lancé pour établir ce tombstone.

## Conclusion opposable

La tentative W3 v13 a consommé son droit one-shot avant le callback, puis a
échoué pendant `MEASURED_W3_CALLBACK_OR_POSTCHECK` avec :

```text
FoldFusedInvariantError: compact output logical content drift
```

Par conséquent :

- aucun `PASS`, aucune autorisation d'exécution et aucun résultat retrieval ne
  peuvent être déduits de v13 ;
- aucune relance v13 n'est permise ;
- les outputs partiels v13 ne peuvent pas devenir les outputs d'une autre
  tentative ;
- fold 0 et fold 1 n'ont pas été ouverts ;
- aucune injection positive n'a eu lieu ;
- l'OOF national n'a pas été lancé.

Le runtime transactionnel préenregistré est celui du commit `41b8eed`; le
handover qui l'a gelé avant exécution est le commit `91a18cf`. Les jalons
antérieurs v13 restent `7e3370f`, `77838a1`, `ee30390` et `db1df27`. Ces
commits sont des autorités historiques, pas une permission de rejouer v13.

## Autorités immuables constatées

| Autorité | Chemin | Octets | Mode | SHA-256 |
|---|---|---:|---:|---|
| Marqueur one-shot consommé | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/.retrieval_v13_w3_attempt_consumed.json` | 466 | `0444` | `927707c05c82d3f775d887ddf44f7f018f62bce3b5463e6830f0b64878b5b028` |
| Manifeste terminal STOP | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_physical_benchmark_v13_w3/manifest.json` | 4 084 | `0644` | `4bc15ca54f5bc8aeda211c4aa1c5982c5db94fe0b4bf13a9823a6125a6fba4b9` |
| Configuration physique v13 | `config/retrieval_all_human_v13_physical_execution.json` | 26 332 | `0644` | `40c75e669289147ecf3b3c89ad37f9ef0f8e289cc092868ec26de10d0e15bf0d` |
| Agrégat des 15 fichiers d'implémentation | inventaire incorporé au manifeste STOP | — | modes du tree Git `100644` | `24841ade82512f1ca8966e8c680b7a0fe2034005c8e2805215fed77c3768a02a` |

Le marqueur porte le statut
`ATTEMPT_IRREVERSIBLY_CONSUMED_BEFORE_CALLBACK`, `retry_allowed=false` et le
chemin canonique exact du tree W3. Le manifeste terminal porte
`automatic_retry_allowed=false`, les folds `[2,3,4]`, et les quatre drapeaux
`fold0_opened`, `fold1_opened`, `positive_injection` et
`execution_authorization_published` à `false`.

## Inventaire exhaustif du tree STOP

L'audit read-only du tree externe donne :

| Propriété | Valeur |
|---|---:|
| Fichiers réguliers | 47 |
| Taille exacte cumulée | 155 819 481 octets |
| Mode de la racine | `0755` |
| Fichiers JSON relus | 23 |
| Manifestes portant les drapeaux protégés | 20 |
| SHA-256 canonique de l'inventaire | `4b45d3e0793b23a43ae9419cdb669bf9ce96b11828b6e9fc5f51f80f70d1a7e2` |
| Racine Merkle de l'inventaire | `34c87618e17fafe3347ebf840a059d4100107090c925c14da03fc66cc1effad9` |

Chaque entrée du receipt contient exactement son `relative_path`, sa taille,
son SHA-256 et son mode POSIX. L'ordre est le tri lexicographique des chemins
POSIX relatifs. Les symlinks, hardlinks, fichiers spéciaux, doublons, chemins
absolus ou échappant à la racine sont refusés. Le fichier est staté avant et
après son hash pour refuser une modification concurrente.

Les ledgers de lecture réellement écrits dans ce tree se réconcilient ainsi :

| Acteur build | Row-groups distincts | Octets compressés projetés |
|---|---:|---:|
| fold 2 | 126 | 921 440 430 |
| fold 3 | 117 | 857 370 091 |
| fold 4 | 125 | 917 265 578 |
| Union, overlap nul | 368 | 2 696 076 099 |

Ces chiffres décrivent les lectures constatées ; ils ne transforment pas le
STOP en benchmark réussi.

## Cause technique, sans requalification scientifique

Le diagnostic post-mortem a été réalisé en lecture seule. Les top100 et
top500 des trois folds ainsi que les outcomes folds 3 et 4 se rechargent avec
leur contenu logique attendu. L'écart est borné aux outcomes fold 2 : deux
slices nullable, pour `query_id=4276` et `query_id=51`, changent de hash quand
le hash porte sur les buffers Arrow physiques après lecture Parquet. La
reconstruction depuis les valeurs logiques retrouve la racine attendue
`6ca48f145b5f4c1df7929db1bf889629bf4166fa6b7fa8b89b9edacd654c569e`,
alors que le slice direct donne
`ce6f0ee9610a1c332b7f3ff27e7766f5aee1185d71f868ba593dc6a4efebb864`.

La cause observée est donc le hash de buffers physiques tenant compte de
l'offset/bitmap de validité d'un slice nullable, et non une preuve de dérive
des candidats ou des métriques. Cela justifie une future version distincte et
préenregistrée ; cela n'autorise ni à modifier le verdict v13, ni à relancer la
tentative consommée.

## Format du receipt content-addressé

L'outil dédié est `scripts/build_retrieval_v13_stop_receipt.py`. Sa commande
`audit` est strictement read-only. Elle :

1. vérifie les SHA, chemins, tailles et modes du marqueur, du STOP et de la
   configuration ;
2. valide l'agrégat scellé des 15 fichiers d'implémentation ;
3. inventorie les 47 fichiers et relit tous les JSON afin de prouver l'absence
   de fold 0/1, d'injection et d'autorisation ;
4. calcule un SHA canonique de l'inventaire et un Merkle séparé par domaine :
   feuille `SHA256(0x00 || JSON_canonique(entrée))`, nœud
   `SHA256(0x01 || gauche || droite)`, dernière feuille dupliquée si nécessaire ;
5. calcule `receipt_id = SHA256(JSON_canonique(corps_sans_receipt_id))`.

L'audit final produit :

- builder SHA-256 :
  `9ace1c727e142512a22ec3e2ee45f0212b98c5f940bfa366770f86838eae7e5f` ;
- receipt ID :
  `bdcba90d29bd7da0e84fb805936fe0f99f8518ce9661a30751412c005f6d6799`.

Un premier receipt `005fb19823721480ca0b23205f041fed04f23d03577e3344e071cf44e6fdc6ac`
a été publié pendant le durcissement du builder. Il inventorie les mêmes 47
fichiers, mais sa version du builder ne pinait pas encore comme constantes
l'identité complète de l'inventaire, le Merkle et les modes des autorités.
Il est conservé sans mutation sous le statut de gouvernance
`DRAFT_SUPERSEDED_NON_AUTHORITY`. Il ne doit être référencé ni par une future
configuration ni comme permission d'exécuter.

| Receipt externe | Statut de gouvernance | Chemin | Mode | SHA-256 du manifeste |
|---|---|---|---:|---|
| `005fb19823721480…` | `DRAFT_SUPERSEDED_NON_AUTHORITY` | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v13_stop_receipt/005fb19823721480/manifest.json` | `0444` | `821c30e289cc79ac8cf89499d82e530ead22b333942153498a36ede951aaccec` |
| `bdcba90d29bd7da0…` | `FINAL_V13_TERMINAL_AUTHORITY` | `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v13_stop_receipt/bdcba90d29bd7da0/manifest.json` | `0444` | `5cb886338693e7198b34fbed7ea3987e6fff68fdb2b1abee9a4091ea6d8269ca` |

La publication est une opération séparée, atomique et idempotente. Elle écrit
uniquement un `manifest.json` en `0444` dans un répertoire `0555` nommé avec
les 16 premiers hexadécimaux du receipt ID, à l'extérieur du tree historique.
Elle refuse toute collision de contenu et ne modifie aucun octet ni mode v13.

Commande de publication externe exécutée séparément après autorisation ; elle
n'est jamais exécutée par l'outil en mode `audit` :

```bash
python scripts/build_retrieval_v13_stop_receipt.py publish \
  --output-root /Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v13_stop_receipt
```

Avant autorisation explicite de publication, seule la commande suivante est
permise :

```bash
python scripts/build_retrieval_v13_stop_receipt.py audit
```

Le receipt final ci-dessus est l'autorité terminale publiée après contrôle de
son chemin, de son mode et de son SHA-256. Sa présence n'est pas une
autorisation de relancer v13. Le draft superseded ne peut en aucun cas remplir
cette fonction.
