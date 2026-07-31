# V4.12-R30 — Contrat d'exécution de la collecte

Statut : préenregistré après scellement du docket et avant le premier accès
réseau.

Ce contrat spécialise l'exécution des sections 4 à 6.1 de
`v4_12_review_adjudication_pilot_contract.md`. Il conserve la sélection, la
sémantique prudente des preuves, la table d'adjudication et les gates, mais
remplace les règles opérationnelles ambiguës du contrat principal par les
règles fermées ci-dessous. En cas de conflit d'implémentation sur la collecte,
la reconstruction des faits ou l'agrégation des preuves, le présent contrat
prévaut.

## 1. Entrée canonique

Le collecteur accepte exclusivement le dossier immuable :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_12_review_adjudication_pilot/c7a9feecaf2d3c2a`

| Fichier | SHA-256 |
|---|---|
| `manifest.json` | `12c964159a028c9c25d940b6bc156af2af55202c892f531c0ae9c9b34eed97eb` |
| `seal.json` | `4806fe5ad59315c3d170788ecbd6da8f2e50373f53a342151501f4d6dcd9f248` |
| `identity/identity_discovery.parquet` | `e5b62b5a614420d3d8260c2c0daa744043f956cd3aedcb05ca16301ea816b872` |
| `identity/collection_plan.parquet` | `d52ed433ee00a70bd95b5b453558d4432e83be4099f551e7d68c84602b0bbfe0` |
| `comparison/docket.parquet` | `71db76376d2af26de4aba4143bf54032d3c58d64f884022da9c746417e2c1cf1` |
| `comparison/candidate_context.parquet` | `c79592d500f24ca0bbddfc5f1608b4294da87783bb84c44a6cd5cb7e2d57bb93` |

Le sceau d'arbre attendu est
`8a9aade72e741e393bdd5647ae440f38793da879462b185640dbf8ac6cf02df0`.
Les 30 dossiers, les 90 requêtes et les quotas doivent être revalidés avant
tout réseau.

La politique réseau et de domaines est
`config/v4_12_review_collection_policy.json`, SHA-256
`1238eb957f84c811ac64375c66a0d62e1bef977a139c0a685e669a5d18c63b88`.
Toute modification de ce fichier exige un nouveau contrat et un nouveau
contre-audit avant réseau.

## 2. Moteur et accès réseau

Le moteur gratuit épinglé est la page HTML DuckDuckGo :

`GET https://html.duckduckgo.com/html/?q=<requête UTF-8 encodée>`

Les seules requêtes sont les 90 chaînes déjà scellées. Elles sont tentées une
fois, dans l'ordre `(selection_ordinal, query_ordinal)`, sans pagination,
reformulation ni retry. Le moteur n'est jamais une preuve et ses pages de
résultats ne produisent aucun fait.

L'URL est sérialisée par
`urllib.parse.urlencode([("q", query)], quote_via=quote_plus, safe="",
encoding="utf-8", errors="strict")`, sans autre paramètre. Les headers exacts
sont `User-Agent`, `Accept: text/html`, `Accept-Language: fr-FR,fr;q=0.9`,
`Accept-Encoding: identity`, `Connection: close` et `Host`. Tout header
supplémentaire est interdit.

Le parseur `DDG_HTML_ORGANIC_RESULT_V1` utilise `lxml 6.0.2` et applique :

1. paramètre charset HTTP extrait sans tenir compte de la casse, espaces et
   guillemets ; `utf-8|utf8` devient UTF-8,
   `iso-8859-1|latin1|latin-1` devient ISO-8859-1, toute valeur absente ou
   autre devient UTF-8 avec remplacement ;
2. parcours DOM des `div` dont la classe contient le token exact `result`,
   hors token `result--ad` et attribut `data-testid=ad` ;
3. premier descendant `a` de classe-token `result__a` ayant `href`, puis
   premier descendant de classe-token `result__snippet` ;
4. espaces du titre/snippet réduits par `split()` puis jointure par un espace ;
5. lien absolu conservé ; wrapper DuckDuckGo absolu, protocol-relative ou
   `/l/` accepté seulement avec exactement un paramètre `uddg`, décodé une
   fois en UTF-8 strict ; tout autre relatif ignoré ;
6. déduplication sur l'URL résolue octet pour octet, première occurrence
   conservée ; les cinq premières occurrences uniques sont rangées 1 à 5.

La fixture `config/v4_12_review_ddg_parser_fixture.html`, SHA-256
`08b450a351b0856c7ec0cb9958c03137068a50c6a5477ec4c86114c31d27be69`,
doit produire exactement
`config/v4_12_review_ddg_parser_expected.json`, SHA-256
`78c6ad766b24c617dd3b03c30216f263678ac44acb387f4ad4d7d983f2ff5491`.
Elle est rejouée avec le charset HTTP explicite `utf-8`.
Les cas UTF-8, ISO-8859-1 et charset inconnu sont épinglés dans
`config/v4_12_review_ddg_charset_vectors.json`, SHA-256
`f7b4ef1702f2c33698c38d141c2475f206c57a9b8741d57b5e04b959b9ca4b0a`.
Le launcher refuse le réseau si ce golden test échoue. Les octets live sont
archivés avant parsing ; seule leur relecture, jamais une répétition du web,
est exigée reproductible.

Le collecteur journalise et `fsync` chaque URL avant résolution DNS ou
connexion. Il conserve au plus les cinq premiers résultats. Une erreur de
recherche consomme cette tentative et le dossier continue avec la prochaine
requête planifiée, jamais avec une requête de remplacement.

Une page de résultat n'est tentée que si :

- l'URL est HTTPS, sans userinfo, IP littérale, fragment, port autre que 443,
  authentification, cookie requis ou coût variable ;
- son URL apparaît dans l'un des cinq résultats archivés ;
- sa famille présumée est admissible selon la politique épinglée ;
- son domaine enregistrable n'a pas déjà été tenté pour ce dossier ;
- la requête possède encore un de ses deux slots et le dossier un de ses six
  slots.

Une tentative consomme son slot même en cas d'erreur. Un doublon de domaine
ne consomme pas de slot. Les redirections ne sont jamais suivies. Un statut
3xx est archivé comme `HTTP_ERROR/REDIRECT_FORBIDDEN`, consomme le slot et ne
permet aucun remplacement. Zéro retry est autorisé.

`Accept-Encoding` vaut `identity`; toute autre `Content-Encoding` est refusée
avant parsing. Les plafonds s'appliquent séparément aux octets transportés et
décodés : 2 Mio/2 Mio pour le moteur, 10 Mio/10 Mio pour une page. La lecture
streaming s'arrête avant dépassement ; aucune réponse partielle ou tronquée
n'est parsée. Seuls `text/html`, `text/plain` et `application/pdf` sont
éligibles.

### 2.1 Hôtes, domaines et SSRF

Le hostname est normalisé par IDNA 2008 via `idna.encode(..., uts46=True,
std3_rules=True)`, après NFC, suppression d'au plus un point terminal, puis
ASCII lowercase. Label vide, `localhost`, suffixe `.local`, userinfo, IP
littérale et port différent de 443 sont refusés.

Le domaine enregistrable utilise la liste restreinte
`config/v4_12_review_public_suffixes.txt`, SHA-256
`10fe038631c2a3dd619370e368be3dbd9b6cb8daf2bd4203ced236cf6226c823`.
Le plus long suffixe correspondant sur une frontière de label est choisi ; le
domaine est le label immédiatement précédent plus ce suffixe. Un suffixe
inconnu ou un hostname égal au suffixe est inadmissible. Toutes les listes de
domaines utilisent `host == suffix` ou `host.endswith("." + suffix)` ; une
simple sous-chaîne est interdite.

Les golden vectors IDNA, suffixes multi-label, trailing dot, frontière de
suffixe et suffixe inconnu sont
`config/v4_12_review_domain_vectors.json`, SHA-256
`2adca818a3196049caf574e2252ece41963f7cd26a0069cf4044a61b74b18917`.

La précédence de classification est :

```text
UNSAFE_URL
→ SIRENE_COPY
→ ALWAYS_INADMISSIBLE
→ PUBLIC_ADMINISTRATION
→ OFFICIAL_SECTOR_DIRECTORY
→ DATED_PUBLIC_DOCUMENT_CANDIDATE
→ ENTITY_OFFICIAL_SITE_CANDIDATE
→ INADMISSIBLE
```

Les tokens de nom sont obtenus par NFKD, suppression des marques Unicode,
`casefold`, remplacement de chaque suite non alphanumérique par un espace,
`split`, suppression des stopwords épinglés et des tokens de moins de quatre
caractères. `ENTITY_OFFICIAL_SITE_CANDIDATE` exige une intersection non vide
entre les tokens CRM et ceux du hostname, titre ou snippet exacts archivés.
Un chemin finissant par `.pdf` devient
`DATED_PUBLIC_DOCUMENT_CANDIDATE` seulement si son domaine est public/sectoriel
ou passe cette même intersection.

Le token CRM sert uniquement à décider une ouverture. La validation
post-ouverture `ARCHIVE_DIRECT_TRIPLE_V1` est volontairement conservatrice :

1. le texte UTF-8 contient au moins un SIRET extrait par la règle ci-dessous ;
2. les tokens normalisés du nom CRM, après retrait des stopwords, apparaissent
   comme une sous-séquence contiguë dans les tokens du texte ;
3. pour au moins une occurrence exacte du code postal CRM, la fenêtre de 192
   octets UTF-8 avant et après contient le premier numéro de voie CRM comme
   token numérique lorsqu'il existe, et au moins un token de voie CRM de
   longueur trois ou plus après retrait du numéro et de
   `rue|avenue|av|boulevard|bd|chemin|route|place|allee|impasse`.
   Sans numéro CRM, la fenêtre doit contenir tous les tokens de voie s'il y en
   a moins de deux, sinon au moins deux.

Si le nom CRM ne conserve aucun token après filtrage, le triple est faux.
Le premier numéro de voie CRM est le premier token ASCII qui matche
`[0-9]{1,5}` dans l'ordre de l'adresse ; les suffixes `bis|ter` ne changent pas
sa valeur. Les occurrences SIRET, nom et code postal sont ordonnées par
`(byte_start,byte_end)`.

La tokenisation NFKD/casefold est celle définie plus haut et conserve pour
chaque token son span dans les octets UTF-8 originaux. Le nom, l'adresse et le
SIRET qui rendent la page éligible produisent chacun un fait et une ligne
`fact_provenance` avec leurs spans exacts.

Le triple est évalué séparément pour chaque occurrence SIRET dans une fenêtre
de 512 octets avant/après son span. Une page est éligible si au moins un SIRET
possède dans cette fenêtre une sous-séquence nom valide et une occurrence
code-postal/adresse valide. Pour chacune, le choix minimise
`(distance_octets_au_span_siret,byte_start,byte_end)`, où la distance vaut zéro
en cas de chevauchement, sinon le nombre d'octets entre les deux spans. Seuls
ces SIRET qualifiés produisent des faits ; un
autre SIRET présent ailleurs dans la page n'est jamais relié par propagation.
Pour un même SIRET, l'occurrence qualifiée de plus petit
`(byte_start,byte_end)` est retenue. Le slice adresse est le plus petit span
contenant l'occurrence CP, le numéro éventuel et les tokens voie requis ;
les égalités sont départagées par `(longueur,byte_start,byte_end)`.

Une page préouverte `PUBLIC_ADMINISTRATION` ou
`OFFICIAL_SECTOR_DIRECTORY` conserve cette famille seulement si son suffixe
épinglé et le triple sont valides. Une page
`ENTITY_OFFICIAL_SITE_CANDIDATE` devient `ENTITY_OFFICIAL_SITE` uniquement
avec le triple. Une page `DATED_PUBLIC_DOCUMENT_CANDIDATE` devient
`DATED_PUBLIC_DOCUMENT` avec le triple et au moins une date calendrier valide
`YYYY-MM-DD` ou `DD/MM/YYYY`/`DD-MM-YYYY`. Toute autre page devient
`INADMISSIBLE_AFTER_OPEN`.
Les captures de date sont converties en entiers puis validées par
`datetime.date(year, month, day)` ; une date impossible est ignorée.

L'extracteur `ASCII_DIGIT_WITH_OPTIONAL_SPACE_DOT_HYPHEN_LUHN_V1` travaille
sur les octets UTF-8 : motif borné par des non-chiffres, exactement 14 chiffres
ASCII séparés éventuellement chacun par un espace, point ou tiret, puis
suppression des séparateurs et validation Luhn. Un SIREN est soit les neuf
premiers chiffres d'un SIRET valide, avec le span correspondant, soit neuf
chiffres ASCII contigus bornés et Luhn-valides hors span SIRET. Le Luhn double
un chiffre sur deux depuis l'avant-dernier chiffre à droite, soustrait neuf si
le produit dépasse neuf et exige une somme multiple de dix.

Les golden vectors identifiants sont
`config/v4_12_review_identifier_vectors.json`, SHA-256
`2e3f87d909c02f8ca50eaf7687358f0d67f7dab9e002d5ef9ecfe3efa52557b9`.

Les golden vectors post-ouverture sont
`config/v4_12_review_postopen_validation_vectors.json`, SHA-256
`97f56cf2c747293c8d3a7cfd6d19d7b32d074c5f14c2c16a9057f3f751b5e231`.
Les quatre familles ont chacune un positif et un négatif. Un écart interdit
le réseau.

Le texte HTML est produit par `lxml 6.0.2` via `text_content()`, puis espaces
réduits et UTF-8. Le texte PDF utilise `pypdfium2 4.30.0`, pages 1 à N dans
l'ordre, au plus 50 pages. Tout texte extrait dépassant 2 Mio rend la page
inéligible. `idna 3.11` est la seule implémentation IDNA admise.

Avant connexion, toutes les réponses A/AAAA sont journalisées et doivent être
absentes de la table CIDR interdite définie ci-dessous ; une seule adresse
interdite refuse la tentative. Cette décision ne dépend jamais de
`ipaddress.is_global`. Les adresses valides sont triées par
`(version, packed)` et la première est épinglée. La socket se connecte à cette
IP ; TLS utilise le hostname normalisé comme SNI et vérifie le certificat pour
ce hostname ; le header `Host` porte ce même hostname. Aucune seconde
résolution n'est autorisée.

La table interdite est la liste `forbidden_cidrs` épinglée dans
`config/v4_12_review_dns_security_vectors.json`, SHA-256
`a1f460b1c8e51a2e9bcf33e06c512cba7d1ab4854f34321d2429cff887b1fb61`.
L'implémentation la charge
avec `ipaddress.ip_network(..., strict=True)` et décide uniquement par
appartenance. Une adresse IPv4-mappée IPv6 est interdite si son IPv6 ou son
IPv4 mappée appartient à un réseau interdit. Le même fichier contient les
cas unitaires IPv4/IPv6 et les réponses DNS mixtes ; toute divergence au
replay interdit le réseau.

Proxy d'environnement, `netrc`, cookies, `Authorization` et authentification
sont interdits. TLS minimum est 1.2. Le bundle CA exact est
`/opt/homebrew/etc/ca-certificates/cert.pem`, SHA-256
`a4b2e4b2a34ad1a99d9c426261858f5c31afadfd07a88cca4af173399249fbc5`.
Les seules capacités resolver supplémentaires sont `/private/etc/hosts`,
SHA-256 `c7dd0e2ed261ce76d76f852596c5b54026b9a894fa481381ffd399b556c0e2da`,
et `/private/var/run/resolv.conf`, SHA-256
`a1ff91810c50ace9d6802e96e2b8cfc133a9a337480d33676e74623ff31cea77`.
Elles sont revalidées same-FD avant et après le run.

## 3. Frontière de processus

Le launcher parent est le seul composant autorisé à :

- résoudre les chemins canoniques et conserver leurs descripteurs ;
- créer exactement deux workers séquentiels via `/usr/bin/sandbox-exec` ;
- effectuer les accès réseau au travers d'un broker fermé ;
- servir les lookups SIRENE exacts ;
- écrire le journal pré-ouverture et publier les arbres scellés.

Chaque worker est lancé sous un profil `deny default`, sans réseau, fork,
exec, subprocessus ni ouverture locale libre. Il reçoit des capacités par
descripteurs et un socket Unix local :

1. le worker `IDENTITY_DISCOVERY` reçoit uniquement les deux Parquets
   `identity`, jamais le docket, top-1, candidats, rangs, scores ou snapshot ;
2. il se termine définitivement ; le parent scelle et valide le dossier
   d'identité, ferme le broker, passe l'état monotone de
   `IDENTITY_NETWORK_OPEN` à `IDENTITY_SEALED_NETWORK_REVOKED` et refuse dès
   lors toute opération `SEARCH_REQUEST`, `PAGE_REQUEST`, DNS ou socket ;
3. un nouveau worker `FROZEN_CANDIDATE_COMPARISON` reçoit les archives
   scellées et les deux Parquets `comparison`, sans aucune capacité réseau.

Le changement d'état est irréversible et journalisé avant le lancement du
second worker. Un crash du premier worker ne permet pas le second. Toute
demande réseau après révocation donne `STOP_INTEGRITY`.

Le snapshot SIRENE n'est jamais scanné par le worker. Le parent accepte un
lookup exact uniquement pour un SIRET syntaxiquement extrait d'une archive
éligible, une seule fois par `(query_id, siret)`.

Le parent installe un hook d'audit fail-closed et journalise toute tentative
avant action. Les ouvertures métier utilisent une allowlist de fichiers
exacts, jamais une racine. Les dépendances système nécessaires au runtime
Python sont une capacité technique séparée et ne deviennent pas des entrées
métier.

## 4. Journal et chemins

Le journal primaire est un répertoire de records O_EXCL
`journal_events/<event_ordinal_20_digits>.json`. Le JSONL est une projection
finale, jamais la source de vérité. Le record genesis ordinal zéro est créé et
`fsync` avant toute autre action. Chaque record contient exactement :

```text
schema_version: "sireto-v4.12-r30-access-event-1"
event_ordinal: uint64, contigu à partir de 0
event_kind: GENESIS | INTENT | RESULT | STATE_TRANSITION
attempt_id: 64 hex | null
parent_intent_ordinal: uint64 | null
phase: PREFLIGHT | IDENTITY_DISCOVERY | IDENTITY_SEAL |
       COMPARISON | PUBLICATION
operation:
  GENESIS | OPEN_LOCAL | WRITE_LOCAL | SEARCH_REQUEST | PAGE_REQUEST |
  DNS_RESOLUTION | SIRENE_LOOKUP | STATE_TRANSITION
target_kind: PATH | URL | HOSTNAME | SIRET | STATE | NONE
target_canonical: string | null
query_id: string | null
query_ordinal: uint8 | null
result_rank: uint8 | null
outcome:
  PLANNED | SUCCESS | DENIED | NETWORK_ERROR | TIMEOUT | HTTP_ERROR |
  PARSE_ERROR | IO_ERROR | STOP_INTEGRITY | NONE
error_type: enum de collection_errors | null
http_status: uint16 | null
byte_count: uint64 | null
content_sha256: 64 hex | null
previous_event_sha256: 64 hex, zéros pour genesis
event_sha256: 64 hex
```

Pour un `INTENT`, `attempt_id` vaut
`SHA256("SIRETO-V412-R30-ATTEMPT\0" || canonical_json([event_ordinal, phase,
operation, target_kind, target_canonical, query_id, query_ordinal,
result_rank]))`.
Le JSON canonique est exactement
`json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",",":"),
allow_nan=False).encode("utf-8")`, sans LF et sans normalisation Unicode
implicite. `event_sha256` vaut
`SHA256("SIRETO-V412-R30-EVENT\0" || canonical_json(record sans
event_sha256))`. Le fichier contient ce JSON plus un LF, est `fsync`, puis le
répertoire est `fsync`.

Chaque `INTENT` a exactement un `RESULT` portant le même `attempt_id` et
`RESULT.parent_intent_ordinal = INTENT.event_ordinal`; l'INTENT porte
`parent_intent_ordinal=null`. Aucun résultat ne peut exister sans intention.
L'intention est persistée avant l'action. Après crash, une intention terminale
sans résultat est conservée et force `STOP_INTEGRITY`; elle n'est jamais
rejouée. Une rupture de chaîne, ordinal manquant, event partiel, accès sans
intention, URL non planifiée ou ouverture hors allowlist donne
`STOP_INTEGRITY`. `access_journal.jsonl` concatène ensuite les records dans
l'ordre ordinal et son hash est publié.

La matrice de nullabilité est fermée :

- `GENESIS` : ordinal 0, phase `PREFLIGHT`, operation `GENESIS`,
  target_kind `NONE`, target/query/ordinals/attempt/parent nuls,
  outcome `NONE`, tous les champs résultat nuls ;
- `INTENT` : operation parmi `OPEN_LOCAL|WRITE_LOCAL|SEARCH_REQUEST|
  PAGE_REQUEST|DNS_RESOLUTION|SIRENE_LOOKUP`, attempt et target non nuls,
  parent nul, outcome `PLANNED`, champs error/http/bytes/content nuls ;
- `RESULT` : attempt et parent non nuls, outcome différent de
  `PLANNED|NONE`; phase/operation/target/query/ordinals identiques à
  l'INTENT parent ; champs résultat nuls sauf ceux permis par l'outcome ;
- `STATE_TRANSITION` : attempt/parent/query/ordinals et champs résultat nuls,
  operation `STATE_TRANSITION`, target_kind `STATE`, outcome `SUCCESS`.

Pour `SUCCESS`, `byte_count` et `content_sha256` sont simultanément nuls
(action sans payload) ou non nuls (payload archivé). `HTTP_ERROR` exige
`http_status`; `NETWORK_ERROR|TIMEOUT|PARSE_ERROR|IO_ERROR|STOP_INTEGRITY`
exige `error_type`; `DENIED` exige `error_type=IO_INTEGRITY`. Toute autre
combinaison est refusée.

Chaque composant de chemin est ouvert depuis `/` avec `openat` et
`O_NOFOLLOW`; les descripteurs d'ancêtres sont retenus. Symlink, hardlink
multiple, FIFO, device, `..`, mutation ou substitution d'ancêtre sont
interdits.

Après NFKC et `casefold`, tout composant contenant `test`, `final`,
`holdout`, `challenge`, `unseen`, `fresh`, `random` ou `locked` est refusé
avant lecture, sauf égalité exacte avec l'un des fichiers épinglés par le
contrat principal. Le préfixe de valeur `fresh:` n'est jamais interprété comme
un chemin.

## 5. Sorties de la passe identité

Le collecteur ne conclut jamais. Il publie uniquement :

- `search_attempts.parquet` : exactement 90 tentatives ou un arrêt global ;
- `search_results.parquet` : rangs 1 à 5 maximum par tentative ;
- `page_decisions.parquet` : une décision déterministe par résultat ;
- `page_attempts.parquet` : une ligne par ouverture tentée, avant réseau ;
- `page_archives.parquet` : une ligne par page effectivement archivée ;
- `collection_errors.parquet` : erreurs typées sans message libre canonique ;
- `dns_resolutions.parquet` : toutes les réponses A/AAAA et l'IP épinglée ;
- `identifier_occurrences.parquet` : occurrences syntaxiques SIRET/SIREN,
  jamais des preuves ;
- `sirene_lookup_plan.parquet` : une ligne par SIRET découvert et dossier ;
- `sirene_records.parquet` : résultat exact de chaque lookup planifié ;
- `raw/search/`, `raw/dns/`, `raw/pages/`, `raw/text/`, `raw/excerpts/`,
  `raw/sirene/`, `raw/errors/` :
  octets immuables ;
- `access_journal.jsonl`, `manifest.json` et `seal.json`.

Tous les Parquets ont un schéma Arrow exact, sans metadata applicative. Les
champs sont non nuls sauf suffixe `?`, et les enums sont stockés comme
`string` mais refusent toute valeur hors liste.

### 5.1 Schémas fermés

`search_attempts.parquet`, PK `(query_id, query_ordinal)` :

```text
query_id:string
selection_ordinal:uint8
query_ordinal:uint8
search_attempt_id:string[64hex]
dns_attempt_id:string[64hex]
search_query:string
search_url:string
planned_max_results:uint8
attempted_at_utc:timestamp[us,UTC]
status:string enum SUCCESS|NETWORK_ERROR|TIMEOUT|HTTP_ERROR|PARSE_ERROR
http_status:uint16?
connected_ip:string?
request_archive_sha256:string[64hex]
request_archive_relative_path:string
response_archive_sha256:string[64hex]?
response_archive_relative_path:string?
response_headers_sha256:string[64hex]?
response_headers_relative_path:string?
response_wire_bytes:uint64?
response_decoded_bytes:uint64?
mime_type:string?
content_encoding:string?
text_decoder:string? enum UTF8_STRICT|ISO_8859_1|UTF8_REPLACE
truncated:bool
returned_result_count:uint32?
logged_result_count:uint8
error_id:string[64hex]?
```

`search_attempt_id` est le SHA-256 du domaine
`SIRETO-V412-R30-SEARCH\0`, du `query_id`, de l'ordinal et de la requête,
sérialisés en JSON canonique. Il y a exactement une ligne par requête
planifiée, même en erreur.

Les invariants de présence sont exhaustifs. `error_id` est nul si et seulement
si `status=SUCCESS`; autrement il référence exactement une
`collection_errors` de la tentative. `http_status`, les champs headers,
`mime_type` et `content_encoding` sont présents si et seulement si des headers
HTTP valides ont été reçus, sauf `mime_type=null` lorsque `Content-Type` est
absent. Les champs archive body et compteurs d'octets sont présents si et
seulement si le body complet a été archivé. `text_decoder` est présent si et
seulement si un body textuel admis a été décodé. `returned_result_count` est
présent si et seulement si le parser a terminé ; `logged_result_count` vaut
alors `min(returned_result_count,5)`, sinon zéro. `truncated` vaut toujours
false : un dépassement n'archive ni ne parse de body partiel.
`status=SUCCESS` exige headers et body complets, statut 2xx, MIME HTML,
encoding `identity`, decoder présent, parsing terminé et zéro erreur.
`NETWORK_ERROR`, `TIMEOUT`, `HTTP_ERROR` et `PARSE_ERROR` correspondent
respectivement aux types d'erreur
`DNS|PRIVATE_ADDRESS|NETWORK|TLS`,
`CONNECT_TIMEOUT|READ_TIMEOUT`,
`HTTP_STATUS|REDIRECT_FORBIDDEN|TOO_LARGE|CONTENT_ENCODING|
UNSUPPORTED_MIME|MALFORMED_RESPONSE`, et `PARSE`.

`search_results.parquet`, PK
`(query_id, query_ordinal, result_rank)`, FK vers `search_attempts` :

```text
query_id:string
query_ordinal:uint8
result_rank:uint8
search_attempt_id:string[64hex]
title:string
snippet:string
observed_href:string
resolved_url:string
normalized_hostname:string?
registrable_domain:string?
result_payload_sha256:string[64hex]
preopen_family:string enum PUBLIC_ADMINISTRATION|
  ENTITY_OFFICIAL_SITE_CANDIDATE|OFFICIAL_SECTOR_DIRECTORY|
  DATED_PUBLIC_DOCUMENT_CANDIDATE|INADMISSIBLE
preopen_rule_id:string
inadmissible_reason:string enum UNSAFE_URL|SEARCH_ENGINE|
  COMMERCIAL_AGGREGATOR|SIRENE_COPY|AUTH_REQUIRED|
  PAID_OR_VARIABLE_COST|UNRECOGNIZED_PUBLIC_SUFFIX|
  DOMAIN_NOT_ALLOWLISTED|NONE
```

Les rangs sont contigus de 1 à `logged_result_count`, maximum cinq.
`result_payload_sha256` est le SHA-256 du domaine
`SIRETO-V412-R30-SEARCH-RESULT\0` suivi du JSON canonique
`[query_id,query_ordinal,result_rank,title,snippet,observed_href,resolved_url]`.
`preopen_rule_id` vaut toujours
`PREOPEN_CLASSIFICATION_PINNED_DOMAINS_V1`.

`page_decisions.parquet`, exactement une ligne par `search_results`, même PK :

```text
query_id:string
query_ordinal:uint8
result_rank:uint8
decision:string enum OPEN_ATTEMPT|SKIP_INADMISSIBLE|
  SKIP_DUPLICATE_DOMAIN|SKIP_QUERY_QUOTA|SKIP_DOSSIER_QUOTA
decision_rule_id:string
domain_first_seen_query_ordinal:uint8?
domain_first_seen_result_rank:uint8?
query_open_slot:uint8?
dossier_open_ordinal:uint8?
page_attempt_id:string[64hex]?
```

Les quatre champs optionnels de slot/attempt sont non nuls si et seulement si
`decision=OPEN_ATTEMPT`. Les références first-seen sont non nulles uniquement
pour `SKIP_DUPLICATE_DOMAIN`.
`decision_rule_id` vaut toujours `BOUNDED_DOMAIN_QUOTA_V1`.

`page_attempts.parquet`, PK `page_attempt_id`, en bijection exacte avec les
`OPEN_ATTEMPT`. Son identité est fixée avant DNS ; ses champs de résolution et
d'issue réseau sont finalisés après la tentative :

```text
page_attempt_id:string[64hex]
dns_attempt_id:string[64hex]
query_id:string
query_ordinal:uint8
result_rank:uint8
query_open_slot:uint8
dossier_open_ordinal:uint8
requested_url:string
request_archive_sha256:string[64hex]
request_archive_relative_path:string
normalized_hostname:string
registrable_domain:string
connected_ip:string?
preconnect_disposition:string enum CONNECT_PERMITTED|DNS_EMPTY|
  DNS_PRIVATE_ADDRESS|DNS_ERROR
network_outcome:string enum NOT_ATTEMPTED|CONNECT_ERROR|TLS_ERROR|
  WRITE_ERROR|NO_RESPONSE_HEADERS|RESPONSE_HEADERS_RECEIVED
attempted_at_utc:timestamp[us,UTC]
```

La décision et l'identifiant de tentative sont fixés avant DNS. Après la
résolution, mais avant toute socket HTTP, le request JSON est écrit une seule
fois avec `connected_ip` éventuellement nul et `preconnect_disposition`. Il ne
contient aucun champ `sent` ni aucune issue future. Si aucune connexion n'est
permise, `network_outcome=NOT_ATTEMPTED`; sinon l'issue finale est déterminée
exhaustivement par le premier événement terminal : échec TCP, échec TLS, échec
d'écriture, fermeture/erreur avant headers, ou headers reçus. L'IP/SNI/Host
archivés sont ceux réellement utilisés.
`CONNECT_ERROR` produit exactement une erreur `NETWORK` ou
`CONNECT_TIMEOUT`; `TLS_ERROR` une erreur `TLS`; `WRITE_ERROR` une erreur
`NETWORK`; `NO_RESPONSE_HEADERS` une erreur `NETWORK` ou `READ_TIMEOUT`.
`RESPONSE_HEADERS_RECEIVED` produit exactement une `page_archives` et aucune
erreur d'ouverture réseau. `NOT_ATTEMPTED` produit exactement l'erreur
`DNS|PRIVATE_ADDRESS` déjà portée par la résolution. Ces ensembles sont
disjoints et exhaustifs.

`page_archives.parquet`, PK `page_attempt_id`, une ligne uniquement si des
headers HTTP ont été reçus pour un `OPEN_ATTEMPT` :

```text
page_attempt_id:string[64hex]
query_id:string
query_ordinal:uint8
result_rank:uint8
dossier_open_ordinal:uint8
normalized_hostname:string
connected_ip:string
registrable_domain:string
http_status:uint16
mime_type:string?
content_encoding:string
text_decoder:string? enum UTF8_STRICT|ISO_8859_1|UTF8_REPLACE
collected_at_utc:timestamp[us,UTC]
raw_headers_sha256:string[64hex]
headers_archive_relative_path:string
raw_content_sha256:string[64hex]?
archive_relative_path:string?
wire_bytes:uint64
decoded_bytes:uint64
truncated:bool
extracted_text_sha256:string[64hex]?
extracted_text_relative_path:string?
validated_family:string enum PUBLIC_ADMINISTRATION|ENTITY_OFFICIAL_SITE|
  OFFICIAL_SECTOR_DIRECTORY|DATED_PUBLIC_DOCUMENT|
  INADMISSIBLE_AFTER_OPEN
family_rule_id:string
family_validation_reason:string enum EXACT_ENTITY_IDENTIFIER_AND_SITE|
  PUBLIC_ENTITY_RECORD|SECTOR_AUTHORITY_RECORD|DATED_MATCHING_ISSUER|
  REDIRECT_FORBIDDEN|HTTP_STATUS|MISSING_CONTENT_TYPE|MIME_FORBIDDEN|
  CONTENT_ENCODING|TOO_LARGE|READ_TIMEOUT|NETWORK_READ_ERROR|
  PARSE_ERROR|NO_EXTRACTED_TEXT|
  ENTITY_RELATION_NOT_PROVEN
facts_eligible:bool
```

`facts_eligible=true` si et seulement si : statut 2xx, contenu complet,
encoding/MIME admis, texte extrait archivé et
`ARCHIVE_DIRECT_TRIPLE_V1=true`; la famille validée doit être celle déterminée
exhaustivement en section 2.1, avec date valide supplémentaire uniquement pour
`DATED_PUBLIC_DOCUMENT`. Aucun autre matcher de nom/adresse, notamment
`active-direct-current-v4.0`, n'est appelé.
`family_rule_id` vaut toujours `POSTOPEN_FAMILY_DIRECT_TRIPLE_V1`.

`mime_type` est nul si `Content-Type` est absent ; sinon il est l'essence
ASCII lowercase avant le premier `;`, espaces externes retirés.
`content_encoding` absent est canonisé à `identity`; toute valeur autre que
le token ASCII case-insensitive `identity` donne `CONTENT_ENCODING`.
`text_decoder` est non nul pour `text/html|text/plain` selon la table charset
de section 2 et nul pour PDF ou contenu non textuel. Pour une réponse moteur,
`status=SUCCESS` exige 2xx, MIME `text/html`, encoding `identity`, body complet,
decoder non nul et parse réussi ; sinon `logged_result_count=0`.
Une ligne page ayant une raison technique
`REDIRECT_FORBIDDEN|HTTP_STATUS|MISSING_CONTENT_TYPE|MIME_FORBIDDEN|
CONTENT_ENCODING|TOO_LARGE|READ_TIMEOUT|NETWORK_READ_ERROR|
PARSE_ERROR|NO_EXTRACTED_TEXT`
possède exactement une ligne `collection_errors` de même `page_attempt_id`.
`ENTITY_RELATION_NOT_PROVEN` n'est pas une erreur technique.
Le mapping technique est exhaustif :

```text
REDIRECT_FORBIDDEN -> REDIRECT_FORBIDDEN
HTTP_STATUS -> HTTP_STATUS
MISSING_CONTENT_TYPE|MIME_FORBIDDEN -> UNSUPPORTED_MIME
CONTENT_ENCODING -> CONTENT_ENCODING
TOO_LARGE -> TOO_LARGE
READ_TIMEOUT -> READ_TIMEOUT
NETWORK_READ_ERROR -> NETWORK
PARSE_ERROR|NO_EXTRACTED_TEXT -> PARSE
```

`dns_resolutions.parquet`, PK `dns_attempt_id` :

```text
dns_attempt_id:string[64hex]
parent_attempt_id:string[64hex]
request_kind:string enum SEARCH|PAGE
query_id:string
query_ordinal:uint8
result_rank:uint8?
normalized_hostname:string
port:uint16
resolved_addresses_json:string
all_addresses_permitted:bool
chosen_ip:string?
resolver_hosts_sha256:string[64hex]
resolver_config_sha256:string[64hex]
dns_archive_sha256:string[64hex]
dns_archive_relative_path:string
resolved_at_utc:timestamp[us,UTC]
```

`resolved_addresses_json` est le tableau JSON compact, unique et trié par
`(IP.version, IP.packed)` de toutes les adresses retournées. Il est archivé
également sous `raw/dns/<dns_attempt_id>.json`, qui contient exactement le
hostname, port, tableau et hashes resolver. `chosen_ip` est non nul si et
seulement si le tableau est non vide et `all_addresses_permitted=true`, et vaut
son premier élément. Chaque tentative SEARCH/PAGE a exactement une ligne DNS
avant toute connexion. Un tableau vide ou une adresse interdite produit en
plus exactement une erreur DNS, aucune connexion et `chosen_ip=null`. Un
tableau vide donne `error_type=DNS`; la présence d'au moins une adresse
interdite, même mêlée à une adresse autorisée, donne
`error_type=PRIVATE_ADDRESS`. Une erreur du resolver avant obtention d'un
tableau donne `error_type=DNS`.

`collection_errors.parquet`, PK `error_id` :

```text
error_id:string[64hex]
query_id:string
stage:string enum SEARCH|PAGE_OPEN|DNS|TLS|HTTP|ARCHIVE|PARSE
query_ordinal:uint8
result_rank:uint8?
page_attempt_id:string[64hex]?
error_type:string enum DNS|PRIVATE_ADDRESS|NETWORK|CONNECT_TIMEOUT|READ_TIMEOUT|
  TLS|HTTP_STATUS|REDIRECT_FORBIDDEN|TOO_LARGE|CONTENT_ENCODING|
  UNSUPPORTED_MIME|MALFORMED_RESPONSE|PARSE|IO_INTEGRITY
occurred_at_utc:timestamp[us,UTC]
error_payload_sha256:string[64hex]
error_archive_relative_path:string
```

Le payload d'erreur est un JSON fermé contenant uniquement code, errno
numérique éventuel, statut éventuel et étape ; aucun message d'exception,
HTML ou texte libre n'est canonique.

`identifier_occurrences.parquet`, PK `occurrence_id` :

```text
occurrence_id:string[64hex]
query_id:string
page_attempt_id:string[64hex]
raw_content_sha256:string[64hex]
extracted_text_sha256:string[64hex]
identifier_type:string enum SIRET|SIREN
identifier_value:string
source_excerpt_sha256:string[64hex]
text_byte_start:uint64
text_byte_end:uint64
extractor_rule_id:string
```

Les offsets portent sur les octets de `raw/text/<page_attempt_id>.utf8`; le
slice exact doit avoir `source_excerpt_sha256`. Les occurrences ne sont
produites que depuis une page `facts_eligible`.
`extractor_rule_id` vaut toujours
`ASCII_DIGIT_WITH_OPTIONAL_SPACE_DOT_HYPHEN_LUHN_V1`.

`sirene_lookup_plan.parquet`, PK `(query_id, siret)` :

```text
query_id:string
siret:string[14digits]
first_occurrence_id:string[64hex]
snapshot_ref:string
snapshot_sha256:string[64hex]
lookup_ordinal:uint16
```

Son ensemble de clés est exactement égal, et non simplement inclus, à
l'ensemble dédupliqué des occurrences `SIRET` issues de pages
`facts_eligible`. Les `lookup_ordinal` sont attribués aux SIRET distincts
triés, et la même valeur est réutilisée si un SIRET apparaît dans plusieurs
dossiers. Aucun SIRET candidat/top-1 absent des occurrences ne peut être
ajouté.

`sirene_records.parquet`, PK `siret`, exactement une ligne par SIRET distinct
de `sirene_lookup_plan` :

```text
siret:string[14digits]
sirene_record_id:string[64hex]
siren:string[9digits]
found_exactly_once:bool
etat_administratif:string?
enseigne_1:string?
enseigne_2:string?
enseigne_3:string?
denomination_usuelle:string?
numero_voie:string?
type_voie:string?
libelle_voie:string?
code_postal:string?
libelle_commune:string?
code_commune:string?
snapshot_sha256:string[64hex]
lookup_ordinal:uint16
looked_up_at_utc:timestamp[us,UTC]
record_archive_sha256:string[64hex]
record_archive_relative_path:string
```

Si le SIRET n'existe pas exactement une fois, seuls les identifiants, le
booléen, le hash snapshot, l'ordinal, l'heure de lookup et les deux champs
d'archive record sont non nuls. Ce résultat ne crée aucun second groupe de
preuve.
L'archive SIRENE est le JSON canonique de la ligne sans ses deux champs
`record_archive_*`; ses valeurs scalaires servent de slices d'extrait aux
faits SIRENE.

### 5.2 Relations fermées

- `search_attempts.(query_id,query_ordinal)` est en égalité exacte avec
  `collection_plan` et `search_query` est identique octet pour octet ;
- chaque `search_results.search_attempt_id` référence l'unique tentative
  portant les mêmes query/ordinal ;
- `page_decisions` est en bijection exacte avec `search_results` ;
- chaque `OPEN_ATTEMPT.page_attempt_id` possède exactement une issue :
  une page row, ou une erreur sans page row si aucun header n'a été reçu ;
- chaque `page_archives.page_attempt_id` référence un unique
  `OPEN_ATTEMPT` de mêmes query/ordinal/rank/slots ;
- chaque `dns_resolutions.parent_attempt_id` référence exactement un
  `search_attempt_id` ou `page_attempt_id`; réciproquement chaque tentative
  réseau porte exactement un `dns_attempt_id` ;
- chaque erreur SEARCH référence une tentative search en erreur ; chaque
  erreur PAGE/DNS/TLS/HTTP/ARCHIVE/PARSE référence l'OPEN_ATTEMPT concerné ;
- chaque occurrence référence une page `facts_eligible=true`, le même
  `raw_content_sha256` et le même texte extrait ;
- chaque `sirene_lookup_plan.first_occurrence_id` référence la première
  occurrence de ce `(query_id,siret)` dans l'ordre
  `(dossier_open_ordinal,text_byte_start)` ;
- `sirene_records.siret` est en égalité exacte avec les SIRET distincts du
  plan ; chaque SIRET est consulté une seule fois globalement.

`facts.parquet` a pour PK
`(query_id,proof_id,fact_type,fact_value_normalized,related_siret,
source_excerpt_sha256)` ;
les doublons exacts sont supprimés avant écriture. Cette PK est celle de
`fact_provenance`. Toute FK manquante, supplémentaire ou contradictoire donne
`STOP_INTEGRITY`.

### 5.3 Noms d'archives

Les chemins relatifs sont dérivés uniquement des IDs hexadécimaux :

```text
raw/search/<search_attempt_id>.request.json
raw/search/<search_attempt_id>.headers.json
raw/search/<search_attempt_id>.response.bin
raw/dns/<dns_attempt_id>.json
raw/pages/<page_attempt_id>.request.json
raw/pages/<page_attempt_id>.headers.json
raw/pages/<page_attempt_id>.content.bin
raw/text/<page_attempt_id>.utf8
raw/excerpts/<source_excerpt_sha256>.utf8
raw/sirene/<sirene_record_id>.json
raw/errors/<error_id>.json
```

Tous les chemins sauf `raw/excerpts/` apparaissent dans une seule ligne de
table. Les excerpts sont dédupliqués par contenu : plusieurs faits peuvent
référencer le même chemin si et seulement si leur
`source_excerpt_sha256` est identique. Chaque fichier apparaît dans le
manifest et aucun fichier orphelin n'est permis. Les JSON de requête
contiennent exactement méthode, URL et headers ordonnés. Les headers de
réponse sont une liste ordonnée de paires après suppression obligatoire de
`set-cookie`; aucune valeur d'authentification n'est archivée.

Chaque ID est le SHA-256 du domaine littéral suivi du JSON canonique de la
liste ordonnée indiquée :

```text
search_attempt_id:
  "SIRETO-V412-R30-SEARCH\0" +
  [query_id, query_ordinal, search_query]
page_attempt_id:
  "SIRETO-V412-R30-PAGE\0" +
  [query_id, query_ordinal, result_rank, resolved_url,
   query_open_slot, dossier_open_ordinal]
dns_attempt_id:
  "SIRETO-V412-R30-DNS\0" +
  [parent_attempt_id, normalized_hostname, 443]
occurrence_id:
  "SIRETO-V412-R30-OCCURRENCE\0" +
  [query_id, page_attempt_id, extracted_text_sha256, identifier_type,
   identifier_value, text_byte_start, text_byte_end]
error_id:
  "SIRETO-V412-R30-ERROR\0" +
  [query_id, stage, query_ordinal, result_rank, page_attempt_id, error_type]
```

Le JSON canonique suit les règles du journal. Aucun ID ne dépend d'un
timestamp. `proof_id` vaut `page_attempt_id` pour une archive web et
`SHA256("SIRETO-V412-R30-SIRENE-PROOF\0" +
canonical_json([query_id,siret,snapshot_sha256]))` pour SIRENE.
`sirene_record_id` vaut
`SHA256("SIRETO-V412-R30-SIRENE-RECORD\0" +
canonical_json([siret,snapshot_sha256]))`.

La requête page archivée contient exactement `method=GET`, URL, hostname
normalisé, IP épinglée nullable, `preconnect_disposition`, SNI, port 443 et la
liste ordonnée des headers :
`Host`, `User-Agent`, `Accept:
text/html,text/plain,application/pdf`, `Accept-Language: fr-FR,fr;q=0.9`,
`Accept-Encoding: identity`, `Connection: close`. Aucun redirect handler,
proxy, cookie jar, `netrc` ou header supplémentaire n'existe dans le broker.

`wire_bytes` désigne les octets du body après retrait du framing HTTP
(status-line, headers et chunks) mais avant tout décodage de contenu ;
`decoded_bytes` désigne les octets après `Content-Encoding`. Puisque le seul
encoding admis est `identity`, les deux flux doivent être octet pour octet
égaux. Le body est lu en chunks d'au plus 64 Kio, avec refus dès que le chunk
suivant dépasserait le plafond, et n'est écrit dans l'archive finale qu'après
lecture complète.

Le parcours des résultats est strictement `(query_ordinal, result_rank)`.
`page_decisions.parquet` utilise seulement :

```text
OPEN_ATTEMPT
SKIP_INADMISSIBLE
SKIP_DUPLICATE_DOMAIN
SKIP_QUERY_QUOTA
SKIP_DOSSIER_QUOTA
```

Une erreur d'ouverture consomme le slot et n'autorise pas le résultat suivant
comme remplacement. Un domaine est consommé dès `OPEN_ATTEMPT`.

Les occurrences portent les offsets, le hash de l'archive et le hash de
l'extrait. Elles ne contiennent aucun top-1, candidat, rang, score, verdict,
champ `correct`, `wrong`, `target` ou synonyme. Une occurrence ne signifie
jamais `supports(siret)`.

### 5.4 Reconstruction hors réseau

Après le seal identité, le comparateur relit uniquement les archives, tables
identité et records SIRENE scellés. Il reconstruit `facts.parquet` avec les
treize colonnes exactes du contrat principal, sans extension.
Dans cette exécution, `collected_at_utc` a le type Arrow exact
`timestamp[us,UTC]` et `extractor_rule_id` vaut toujours
`FACT_RECONSTRUCTION_DIRECT_TRIPLE_V1`.
`fact_provenance.parquet` porte séparément le lien technique, avec la même clé
logique que le fait :

```text
query_id:string
proof_id:string
fact_type:string enum identique à facts
fact_value_normalized:string
related_siret:string[14digits]
source_excerpt_sha256:string[64hex]
excerpt_archive_relative_path:string
provenance_kind:string enum PAGE_ARCHIVE|SIRENE_RECORD
page_attempt_id:string[64hex]?
extracted_text_sha256:string[64hex]?
text_byte_start:uint64?
text_byte_end:uint64?
lookup_siret:string[14digits]?
sirene_record_sha256:string[64hex]?
sirene_field_names_json:string?
```

Sa PK est les six premiers champs. Elle est en bijection exacte avec les
lignes de `facts.parquet`. Pour `PAGE_ARCHIVE`, page, texte et offsets sont
non nuls, le slice a `source_excerpt_sha256` et contient la valeur brute. Pour
`SIRENE_RECORD`, `lookup_siret` et `sirene_record_sha256` sont non nuls et
référencent exactement l'archive canonique de `sirene_records`;
`sirene_field_names_json` est le tableau trié unique des clés JSON utilisées.
Chaque excerpt, web ou SIRENE, est archivé exactement sous
`raw/excerpts/<source_excerpt_sha256>.utf8`. Pour un fait SIRENE composite,
l'excerpt est le JSON canonique de l'objet des champs source indiqués, jamais
un faux slice contigu.

Pour chaque SIRET web qualifié, le comparateur produit exactement cinq faits
de même `proof_id=page_attempt_id` :

| `fact_type` | valeur normalisée | excerpt | `related_*` | `site_specific` |
|---|---|---|---|---|
| `SIRET_IDENTIFIER` | 14 chiffres | occurrence SIRET choisie | SIRET et ses 9 premiers chiffres | true |
| `SIREN_IDENTIFIER` | 9 chiffres | span des 9 premiers chiffres | même SIRET/SIREN | false |
| `SITE_NAME` | normalisation NFKD/casefold du slice nom choisi | du premier au dernier token nom | même SIRET/SIREN | false |
| `SITE_ADDRESS` | normalisation NFKD/casefold du plus petit slice contenant numéro éventuel, token(s) voie et CP | ce slice minimal | même SIRET/SIREN | true |
| `ENTITY_SITE_RELATION` | normalisation du slice relation | du début minimal au terme maximal des quatre preuves | même SIRET/SIREN | true |

La fonction unique de normalisation des noms, adresses et relations est :
`unicodedata.normalize("NFKD", value)` ; suppression de chaque caractère de
catégorie Unicode `Mn` ; `casefold()` ; remplacement de chaque suite de
caractères non alphanumériques Unicode par un espace ASCII ; `split()` puis
jointure par un espace ASCII. Les identifiants restent des chiffres. La valeur
de `ENTITY_SITE_RELATION` est donc cette normalisation appliquée à son excerpt
exact, et non un JSON construit absent de la source. Si deux occurrences
donnent la même PK fact, la première par span est gardée.

Le replay de cette transformation utilise
`config/v4_12_review_fact_reconstruction_vectors.json`, SHA-256
`b50924bbd301a0fb7a4b970faf9a8b97da27195b227329906fc28aee48b48325`.
Chaque segment `literal` est copié tel quel ; un segment `repeat` est répété
exactement `count` fois ; leur concaténation doit avoir
`expanded_text_sha256`. Chaque tuple `expected_facts` contient, dans l'ordre,
`related_siret`, `fact_type`, `fact_value_normalized`, `text_byte_start`,
`text_byte_end`, `source_excerpt_sha256`. Les lignes `facts` et
`fact_provenance` reconstruites doivent correspondre exactement, y compris
les cinq faits par SIRET, les spans, les SIRET non qualifiés et la
déduplication d'excerpts.

Pour chaque ligne `(query_id,siret)` de `sirene_lookup_plan` dont le
`sirene_records` global est trouvé exactement une fois, le comparateur produit
un jeu de faits SIRENE propre à ce `query_id`. Plusieurs dossiers réutilisent
donc la même archive globale et le même record, mais jamais le même
`proof_id`, celui-ci incluant `query_id`. Les faits sont reconstruits depuis
le JSON record : SIRET, SIREN, chaque nom de site non nul, l'adresse composée
`numero_voie type_voie libelle_voie code_postal libelle_commune`, puis une
relation dont l'excerpt SIRENE est le JSON canonique de l'objet contenant
`siret`, `noms_tries_uniques` et `adresse`, et dont la valeur est la
normalisation de cet excerpt. Le nom porte
`site_specific=false`; identifiants, adresse et relation portent true.
`source_url_or_snapshot_ref` est le chemin+hash du snapshot, `content_sha256`
est `record_archive_sha256`, et l'excerpt est la valeur JSON scalaire ou
structure exacte. `SIRENE_REGISTRY` considère actif uniquement
`etat_administratif == "A"`. `collected_at_utc` des faits SIRENE vaut
`looked_up_at_utc`.

`evidence.parquet` est une dérivation déterministe de `facts.parquet`,
`sirene_lookup_plan.parquet` et `sirene_records.parquet`, avec PK
`(query_id, related_siret, independence_group, proof_id)` et schéma exact :

```text
query_id:string
related_siret:string[14digits]
independence_group:string enum des cinq familles du contrat principal
proof_id:string
evidence_ref_id:string[64hex]
archive_content_sha256:string[64hex]
has_siret_identifier:bool
name_rule_pass:bool
address_rule_pass:bool
site_specific_pass:bool
within_group_contradiction:bool
group_supports:bool
evidence_rule_id:string
```

Deux URLs de même famille ne créent qu'un groupe. `SIRENE_REGISTRY` ne
supporte qu'un SIRET actif dont nom et site reproduisent une archive
indépendante. Aucune colonne libre, aucun texte généré et aucun verdict ne
sont des entrées.

`evidence_ref_id` vaut
`SHA256("SIRETO-V412-R30-EVIDENCE\0" +
canonical_json([query_id,related_siret,independence_group,proof_id]))`.
Pour une preuve web, `archive_content_sha256` est le body archivé ; pour
SIRENE, c'est `record_archive_sha256`.
`evidence_rule_id` vaut toujours `TWO_FAMILIES_INCLUDING_SIRENE_V1`.

Une preuve web a `has_siret_identifier`, `name_rule_pass`,
`address_rule_pass` et `site_specific_pass` vrais si et seulement si ses cinq
faits qualifiés existent pour ce SIRET ; `site_specific_pass` exige en
particulier `SITE_ADDRESS` et `ENTITY_SITE_RELATION` marqués true. Pour un
groupe web non SIRENE et une requête, `within_group_contradiction=true` sur
toutes ses lignes si l'union des relations qualifiées contient plus d'un
SIRET ; sinon false. Une ligne web a `group_supports=true` si les quatre
passes sont vraies et le groupe sans contradiction.

Chaque ligne `(query_id,siret)` du plan SIRENE produit exactement une ligne
`SIRENE_REGISTRY`, même si le record n'est pas unique et ne produit donc aucun
fait SIRENE. La ligne SIRENE d'un SIRET a les quatre passes vraies si et seulement si le
record est unique, actif `A`, et si au moins une preuve web indépendante
possède pour ce même SIRET : un `SITE_NAME` égal après la normalisation
NFKD/casefold à l'un des noms SIRENE, et un `SITE_ADDRESS` dont code postal,
numéro éventuel et tokens voie passent exactement la règle de fenêtre contre
l'adresse SIRENE. Sinon ses passes sont fausses. Le groupe SIRENE est en
contradiction si le record n'est pas trouvé exactement une fois ; il supporte seulement avec
quatre passes vraies et aucune contradiction.

L'ensemble supporté contient les SIRET ayant `group_supports=true` dans au
moins deux familles distinctes, dont `SIRENE_REGISTRY`. Plusieurs preuves
d'une même famille comptent une seule fois.

Le replay `facts → evidence` utilise
`config/v4_12_review_evidence_vectors.json`, SHA-256
`691d434591a03a9b75c59beac968921e43c38881632ef6ef4123e4212bb37b8f`.
Les tuples suivent exclusivement `tuple_schema`; chaque tuple
`expected_evidence`, complété par le `query_id` du cas et
`evidence_rule_id=TWO_FAMILIES_INCLUDING_SIRENE_V1`, est une ligne complète
attendue. Aucun support implicite ou ligne supplémentaire n'est admis.

`adjudications.parquet`, PK `query_id`, est ensuite produit uniquement par la
table exhaustive du contrat principal :

```text
query_id:string
stratum:string enum SAME_SIREN_MULTISITE|CROSS_SIREN_COLLISION|OTHER_REVIEW
status:string enum TOP1_CORRECT|TOP1_WRONG|AMBIGUOUS|UNRESOLVED
reliable:bool
top1_siret:string[14digits]
supported_sirets_json:string
independent_group_count:uint8
evidence_ref_ids_json:string
exact_alternative_known:bool
alternative_siret:string[14digits]?
alternative_naturally_in_top100:bool
decision_rule_id:string
```

Les deux JSON sont des tableaux triés, uniques, UTF-8 compacts. `reliable`
vaut vrai uniquement pour `TOP1_CORRECT`, `TOP1_WRONG` ou `AMBIGUOUS` avec
au moins deux groupes dont SIRENE et les conditions site-spécifiques.
`alternative_siret` est non nul si et seulement si
`status=TOP1_WRONG` et `exact_alternative_known=true`. Le top-1 et le top-100
sont ouverts pour la première fois dans le worker comparaison. Un replay
local à partir du seal identité doit reproduire octet pour octet les tables
dérivées, hors timestamps déjà contenus dans les archives.
`decision_rule_id` vaut toujours `SUPPORTED_SET_EXACT_TOP1_V1`.

`evidence_ref_ids_json` est la liste triée des `evidence_ref_id` dont
`group_supports=true` pour les SIRET supportés. Pour zéro support, elle est
vide. `independent_group_count` vaut zéro sans support, le nombre de familles
du SIRET unique avec un support, et le minimum des nombres de familles des
SIRET supportés en cas `AMBIGUOUS`.

La table de décision s'applique exactement :

- ensemble vide : `UNRESOLVED`, `reliable=false` ;
- singleton égal top-1 : `TOP1_CORRECT`, fiable si le compte est au moins 2 ;
- singleton différent : `TOP1_WRONG`, fiable si le compte est au moins 2,
  `exact_alternative_known=true`, `alternative_siret` vaut ce singleton,
  `alternative_naturally_in_top100` vaut l'appartenance exacte au pool ;
- au moins deux : `AMBIGUOUS`, fiable si le compte minimum est au moins 2,
  champs alternative faux/nul.

Une alternative hors top-100 reste un label de refus pour l'accepteur et
n'est jamais un positif ranker.

La table de décision est testée par replay de
`config/v4_12_review_adjudication_vectors.json`, SHA-256
`c5b9bab97770a6f0a17ea735cc42b554febc3f585e94199158b9bd5cd934dfa3`.
Pour `TOP1_CORRECT`, `AMBIGUOUS` et `UNRESOLVED`,
`exact_alternative_known=false`, `alternative_siret=null` et
`alternative_naturally_in_top100=false`.

## 6. Validation avant comparaison

Le validateur hors réseau recalcule :

- les 30 × 3 tentatives dans l'ordre et l'égalité exacte des requêtes ;
- les plafonds cinq résultats, deux tentatives par requête et six par dossier ;
- la déduplication de domaine dès la première tentative ;
- l'absence de retry, pagination, remplacement ou URL non planifiée ;
- la chaîne complète du journal et les hashes de toutes les archives ;
- zéro colonne interdite et zéro ouverture locale hors allowlist ;
- l'égalité exacte entre les SIRET du plan SIRENE et les occurrences SIRET
  éligibles dédupliquées ;
- les schémas, PK/FK, enums, nullabilité et cardinalités exacts ;
- la bijection entre lignes et archives, les slices d'extrait et l'absence
  d'archive orpheline ;
- la révocation réseau avant comparaison et zéro événement réseau après
  `IDENTITY_SEALED_NETWORK_REVOKED`.

Toute divergence donne `STOP_INTEGRITY`. Un défaut de réseau ou des preuves
insuffisantes n'est pas une divergence et conduira plus tard à `UNRESOLVED`.

Le passage à l'implémentation exige deux contre-audits `GO_COLLECTION_CONTRACT`.
Le passage au réseau exige ensuite deux contre-audits
`GO_IDENTITY_BROKER_WORKER_PHASE` sur le parent broker, le worker identité,
le profil effectif, les tests et un run synthétique sans réseau. Aucun de ces
GO n'autorise le test final ou l'entraînement.
