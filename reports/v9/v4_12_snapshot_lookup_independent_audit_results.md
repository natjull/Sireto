# V4.12 — Contre-audit indépendant du lookup

## Verdict

`GO_V412_LOOKUP_INDEPENDENT_AUDIT`

Le lookup V4.12 est confirmé indépendamment depuis le snapshot SIRENE. Le
premier faux `STOP` provenait bien de l'écriture des deux caractères
antislash et `n` à la place d'un véritable octet LF ; aucune divergence du
lookup n'est observée.

Ce verdict maintient `GO_V412_SNAPSHOT_LOOKUP` et autorise le contrat du
moteur requête par requête. Il ne certifie ni sa parité complète, ni sa
latence, ni la production.

## Artefact

- chemin :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_snapshot_lookup/4055be6e7a11b003`
- build id : `4055be6e7a11b003`
- manifeste :
  `a3b5a6b57ddacb2f8f11bd2c232b318e0411ce636f1fb575c7255948ebe4576e`
- résultat :
  `8f71cc23b5b6f65ecaecb3a07653451b383ce2899d8d7bd435f3a5c8fbf0dddf`
- verrou :
  `fa3bd3db2e26a1a9456e3f4f571b9035f354be4abd59dfccb7a5d3a677f2a1f8`

## Résultats

| Contrôle | Résultat |
|---|---:|
| SIRET sélectionnés | 10 000 |
| Taille payload vrai LF | 150 000 octets |
| Hash vrai LF | `58c9700d…f945` |
| Taille contre-exemple `\` + `n` | 160 000 octets |
| Hash contre-exemple | `72f43460…4960` |
| SIRET absents du snapshot | 0 |
| SIRET absents du lookup | 0 |
| Écarts de valeur ou nullité | 0 |
| Pic mémoire | 4 365 549 568 octets |
| Plafond mémoire | 8 589 934 592 octets |

Premiers SIRET :

```text
94410569100017
92883024900019
53539062900017
```

Derniers SIRET :

```text
44801807700025
75288994900018
41494554300034
```

## Indépendance

Le contre-validateur :

- sélectionne les SIRET sans importer le builder historique ;
- projette ensuite les six valeurs métier directement depuis le Parquet
  SIRENE ;
- ouvre le DuckDB uniquement en lecture seule ;
- compare par lots plafonnés à 100 ;
- exécute le validateur officiel dans un subprocess séparé ;
- vérifie indépendamment schéma, cardinalité, index, fichiers, hashes,
  sources Git, verrou, runtime, RSS et publication atomique ;
- n'ouvre aucun label, challenge, candidat, score ou décision.

Le test de falsification modifie une valeur métier dans une copie du DuckDB,
exécute `CHECKPOINT`, puis rescelle hashes, tailles et manifeste. Le
validateur historique accepte cette copie, tandis que le nouveau
contre-validateur la refuse par comparaison directe avec le snapshot.

## Validation du code

- 23 tests ciblés réussis sans variable d'environnement spéciale ;
- 618 tests complets réussis ;
- mini-audit complet exercé avec dépôt Git, verrou, subprocess officiel,
  phases A/B, store, RSS, staging, `fsync`, renommage et postvalidation ;
- contre-audit final de l'artefact : `GO_ARTIFACT_V412_INDEPENDENT_AUDIT`.

## Suite

Préenregistrer le moteur persistant et le benchmark apparié sur les
1 456 requêtes dev, sans labels :

1. parité exacte retrieval, 45 features, ranker, scène, accepteur et garde ;
2. zéro miss du cache TF-IDF en lecture seule ;
3. p95 par requête, worker persistant, V4.11 et V4.12-G appariés ;
4. RSS maximal de 8 Gio ;
5. test final toujours fermé.
