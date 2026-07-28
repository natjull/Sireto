# V4.12 — Contrat de contre-validation indépendante du lookup

## 1. Objet

Le lookup publié
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/v4_12_snapshot_lookup/ff0f33ad10803cfb`
a franchi le build et les parités prévues. Son validateur de réception
contrôle toutefois la déclaration du hash de l'échantillon sans refaire la
sélection depuis le snapshot.

Un premier contre-audit a aussi montré qu'une commande imbriquée pouvait
confondre un véritable octet LF avec les deux caractères littéraux `\` et
`n`. Le présent contrat ferme ces deux faiblesses avant le benchmark
requête par requête.

Il ne modifie ni le lookup, ni le retrieval, ni le ranker, ni l'accepteur.

## 2. Entrées gelées

Les seules données métier ouvertes sont :

- le snapshot
  `data/StockEtablissement_utf8.parquet`, SHA-256
  `c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845` ;
- l'artefact lookup `ff0f33ad10803cfb`, avec exactement les quatre fichiers,
  hashes et tailles du plan ;
- le plan `config/v4_12_snapshot_lookup_independent_audit_plan.json` ;
- un verrou d'exécution créé seulement après commit et audit du runner.

Les entrées de provenance sont énumérées dans le plan : présent contrat,
plans, runner, tests, store V4.12 autorisé, validateur officiel, ses sources
transitives et son verrou. Le verrou externe indépendant épinglera leur
contenu et le commit exact ; il est traité séparément et ne tente pas de
s'auto-hasher.

Tout challenge, label, vérité CRM, candidat, score ou décision est interdit.
Le réseau, les API, les LLM, les GPU loués et toute dépense externe sont
interdits.

## 3. Sélection indépendante

DuckDB 1.4.3 est utilisé avec un worker, `threads=1`, une limite mémoire de
6 GB et un répertoire temporaire sur `/Volumes/CATNAT_DATA`.

À partir du snapshot, le runner exécute deux phases séparées.

### Phase A — sélection SIRET

1. projeter uniquement `DISTINCT CAST(siret AS VARCHAR)` ;
2. conserver uniquement les SIRET correspondant à `[0-9]{14}` avec
   `regexp_full_match` ;
3. trier par
   `sha256('v412-lookup-parity:' || siret), siret` ;
4. prendre exactement les 10 000 premiers SIRET.

### Phase B — valeurs métier

Joindre les 10 000 SIRET sélectionnés au snapshot et calculer exactement les
six expressions métier indépendantes gelées dans le plan. La sélection et la
projection ne doivent importer aucun helper du builder historique. Le
validateur officiel est exécuté dans un subprocess séparé. Seul
`V412SnapshotLookup` peut être importé dans le runner indépendant afin de
tester l'API de lecture réellement destinée au service.

Les trois premiers SIRET attendus sont :

```text
94410569100017
92883024900019
53539062900017
```

Les trois derniers sont :

```text
44801807700025
75288994900018
41494554300034
```

Le code de hash n'utilise aucune chaîne représentant un saut de ligne. Pour
chaque SIRET ASCII de 14 octets, il ajoute explicitement `bytes([10])`.

Les assertions bloquantes sont :

- `10 000` SIRET ;
- payload LF de `150 000` octets ;
- SHA-256 LF
  `58c9700d2a1ed2bb433e4f7a25a845ba236d63cfe633dcd64f9156469777f945` ;
- contre-exemple construit avec `bytes([92, 110])` de `160 000` octets ;
- SHA-256 du contre-exemple
  `72f43460bb0e5047186fb4226147f1bf3022ceb8692164e5a8c57d9432a54960` ;
- les deux hashes doivent être différents.

Le plan gèle aussi le namespace, la regex, l'ordre, les sept expressions SQL,
le lot maximal de 100, l'encodage `ASCII_SIRET14_PLUS_BYTE_0A` et le
contre-exemple `ASCII_SIRET14_PLUS_BYTES_5C_6E`.

## 4. Comparaison au lookup

Le lookup est ouvert par `V412SnapshotLookup`, donc en lecture seule. Les
10 000 SIRET sont demandés par lots de 100 au maximum.

Pour chaque ligne, présence, sept valeurs, types et nullité doivent être
identiques à la projection fraîche du snapshot. Aucun SIRET supplémentaire
ne peut être renvoyé.

Le runner doit aussi :

- exécuter le validateur officiel avant le contrôle indépendant ;
- vérifier les hashes et tailles des quatre fichiers publiés ;
- recalculer indépendamment la table unique, les sept colonnes `VARCHAR` dans
  leur ordre, les `42 322 035` lignes/SIRET uniques, zéro invalide et l'index
  unique `candidate_details_siret_uidx` sur `siret` ;
- refuser tout WAL, symlink ou fichier supplémentaire ;
- vérifier snapshot, lookup, plan, verrou et sources avant et après lecture ;
- rester sous 8 Gio de RSS ;
- publier par staging, `fsync` et renommage atomique.

La racine de l'artefact doit être un répertoire réel non symbolique avant
toute résolution de chemin. Le répertoire temporaire est gelé sur le SSD.

## 5. Sortie

La racine de publication est :

```text
/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_snapshot_lookup/
```

Un nouvel identifiant immuable dépend des hashes des entrées, du verrou, des
sources et du runtime. L'artefact contient :

- `audit.json` : compte, premiers/derniers SIRET, tailles et hashes des deux
  payloads, nombre d'écarts, RSS et verdict ;
- `manifest.json` : identité, provenance et hashes de sortie.

Le détail des 10 000 SIRET n'est pas publié : il est recalculé depuis la
source à chaque exécution formelle.

## 6. Tests et ordre d'autorisation

Ordre obligatoire :

1. commit isolé du contrat et du plan ;
2. commit isolé du runner et de ses tests ;
3. audit indépendant du runner sans exécution réelle ;
4. création et commit d'un verrou externe sans option de contournement ;
5. audit du verrou ;
6. une exécution formelle et un contre-audit de l'artefact.

Les tests doivent inclure :

- la distinction exacte entre `bytes([10])` et `bytes([92, 110])`, avec
  longueurs et hashes attendus ;
- les bornes 0, 1, 100 et 101 SIRET ;
- une mini-publication complète ;
- un snapshot ou lookup altéré ;
- une fixture où une valeur métier du DuckDB est modifiée, `CHECKPOINT` est
  exécuté, puis hash, taille et manifeste sont rescellés sans changer la
  cardinalité : le validateur officiel seul peut l'accepter, mais le nouveau
  contre-validateur doit la refuser par comparaison snapshot/lookup ;
- fichier supplémentaire, symlink, WAL, hash, runtime, source, verrou,
  TOCTOU, RSS, staging et publication falsifiés.

Il n'existe aucun flag public permettant d'ignorer le verrou, les hashes, les
tests de valeurs ou la limite mémoire. Toutes les entrées sont rehashées
avant et après lecture.

## 7. Verdicts

- `GO_V412_LOOKUP_INDEPENDENT_AUDIT` : toutes les assertions sont exactes ;
- `STOP_V412_LOOKUP_SAMPLE` : sélection, encodage, hash ou valeurs divergent ;
- `STOP_V412_LOOKUP_AUDIT` : provenance, intégrité, ressources ou publication
  divergent.

Seul le premier verdict maintient `GO_V412_SNAPSHOT_LOOKUP` et autorise le
contrat du moteur requête par requête. Ce contrôle ne certifie toujours pas
la latence ni la production.
