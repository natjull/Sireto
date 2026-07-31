# V4.13 — Contrat exécutable minimal de labels CRM frais

## 1. Objet et frontière de confiance

V4.13 vise uniquement la North Star SIRETO : mesurer puis améliorer le
matching CRM vers SIRET sur une population réellement nouvelle. Les artefacts
V1/S1 sont conservés comme historique, mais Ed25519, le producteur Keychain et
la PKI locale sortent du chemin critique.

Le modèle de menace est un opérateur local coopératif sous un UID unique. Les
hashes, manifests, écritures exclusives et séparations physiques empêchent les
erreurs, la dérive, la réutilisation accidentelle, la fuite de vérité,
l'optional stopping et le rejeu. Ils ne prétendent pas résister à un processus
hostile déjà maître du même UID.

Une exception étroite est autorisée : l'auditeur anti-chevauchement peut lire,
sans interface utilisateur, la clé HMAC historique déjà existante
`SIRETO_V412_COMPATIBILITY_LINEAGE_HMAC_V1`. Il utilise uniquement
`SecItemCopyMatching`, ne crée, ne modifie, ne supprime et n'exporte aucune
clé. Cette lecture ne constitue pas une PKI et ne peut créer un label.

## 2. Séquence sans circularité

L'ordre suivant est obligatoire :

1. préenregistrer ce contrat, le plan, les schémas et leurs hashes ;
2. obtenir deux audits indépendants `GO_V413_PREREGISTRATION` ;
3. implémenter sur fixtures synthétiques les composants listés dans le plan ;
4. geler un execution lock qui pinne commit, blobs, tests, runtime et
   préenregistrement, puis obtenir deux audits `GO_V413_IMPLEMENTATION` ;
5. attendre une collection sans ouvrir ses payloads ;
6. Gate 0A : sélectionner une seule collection à partir des manifests
   uniquement et créer une claim durable `O_EXCL` ;
7. Gate 0B : créer le marqueur d'ouverture, ouvrir les payloads une seule
   fois, qualifier toutes les lignes et sceller queries/oracle/audit ;
8. geler les splits, exécuter le retrieval sur fit/dev et franchir le gate
   dev ;
9. seulement alors entraîner ranker et accepteur ;
10. geler tout le bundle puis ouvrir et évaluer test une seule fois ;
11. conclure exactement `GO`, `PIVOT` ou `STOP`.

Le Gate 0A ne lit que `collection_manifest.json`. Le Gate 0B n'est autorisé
que si le préenregistrement et l'implémentation ont chacun deux GO et si la
claim 0A existe. Aucune source réelle n'est nécessaire pour coder ou auditer
les composants génériques.

## 3. Première collection : sélection déterministe

L'inbox fixe ne contient que des répertoires enfants directs nommés
`<20 chiffres UTC epoch-ns>_<64 hex du SHA-256 du manifeste>`. Chaque enfant
contient exactement :

```text
collection_manifest.json
crm_source.csv | crm_source.parquet
authoritative_mapping.csv | authoritative_mapping.parquet
```

L'auditeur énumère tous les enfants directs, rejette liens symboliques,
hardlinks, fichiers supplémentaires, noms non conformes et manifests non
canoniques. Il observe chaque manifeste deux fois à au moins 60 secondes
d'intervalle ; inode, device, uid, nlink, mode, taille, mtime, ctime et hash
doivent rester identiques.

Sont admissibles les manifests qui :

- lient le SHA du plan, le commit et le SHA de l'execution lock ;
- déclarent une population exhaustive et aucune exclusion issue d'un matching ;
- ont `period_start_utc`, `export_cutoff_utc` et `created_at_utc` strictement
  postérieurs à `preregistration_lock.created_at_utc` ;
- utilisent les schémas exacts et le catalogue de preuves préenregistrés ;
- déclarent hashes, tailles et nombres de lignes non nuls.

Le gagnant est le tuple lexicographiquement minimal
`(arrival_epoch_ns, collection_manifest_sha256)`. L'auditeur écrit d'abord un
ledger exhaustif de tous les manifests observés, puis une claim globale
`O_EXCL`; une autre collection ne peut jamais être sélectionnée. Une claim
valide incomplète reprend seulement la même collection. Une claim incohérente
ou un crash après marqueur d'ouverture produit `STOP`, jamais une seconde
sélection.

## 4. Schémas source et preuve indépendante

Les schémas fermés sont dans le fichier pinné par le plan. La frame CRM porte
un `source_record_id` stable, un éventuel `source_group_id`, une date de
création source et uniquement les champs CRM bruts autorisés. Elle ne porte
jamais SIRET, SIREN, label, candidat, rang, score ou prédiction.

Le mapping est joint uniquement par égalité de `source_record_id`. Il porte
un type de preuve, un issuer, un identifiant autoritatif, sa temporalité et le
hash d'un payload de preuve. Il ne porte aucun nom, adresse ou champ utilisable
pour une similarité.

Le catalogue fermé peut admettre uniquement :

- `SOURCE_SYSTEM_OFFICIAL_SIRET` : SIRET déjà présent dans le système de
  référence avant l'export, sans production par SIRETO ;
- `CONTRACT_OR_BILLING_SIRET` : identifiant tiré d'un système contractuel ou
  de facturation distinct du matching ;
- `SEALED_ADMINISTRATIVE_DOCUMENT` : document administratif antérieur au
  cutoff, référencé et hashé.

Au présent préenregistrement, l'allowlist réelle est volontairement vide :
aucun issuer et aucun système de preuve réel n'est encore connu. Cela permet
d'implémenter et d'auditer les composants sur fixtures synthétiques, mais
interdit Gate 0A et tout `MATCH_EXACT` réel. Avant qu'une collection soit
déposée, un amendement séparé devra pinner pour chaque couple
`(authority_type, authority_issuer_id, authority_system_id)` le root, le
manifest, le schéma, le mécanisme de lookup, le cutoff et les payloads de
preuve rehashables, puis obtenir deux GO. Une chaîne libre ou une simple
attestation n'est jamais admissible.

Chaque future entrée autorisée atteste `matching_pipeline_used=false`, nomme
`authority_issuer_id` et `authority_system_id`, et doit avoir été créée avant
le cutoff du CRM. Une ressemblance nom/adresse, SIRENE seul, le retrieval, un
candidat, un hit, un rang, un score, une prédiction, un LLM ou une validation
utilisateur ne peut jamais créer la vérité.

Après contrôle de syntaxe et de temporalité :

- un unique SIRET autoritatif valide donne `MATCH_EXACT` ;
- plusieurs SIRET, un SIREN seul ou des preuves contradictoires donnent
  `AMBIGUOUS` ;
- aucune preuve admissible donne `UNRESOLVED`.

SIRENE peut uniquement contrôler un SIRET déjà fourni. Toutes les lignes
source restent au dénominateur.

## 5. Anti-chevauchement exact

Avant le split, chaque ligne est comparée aux trois keysets historiques dont
la projection reste sémantiquement applicable : `service_id`,
`siret_masked` et `fuzzy_historical`. `source_record_id_equivalence_attested`
doit être vrai pour toutes les lignes ; sinon la collection est refusée, de
sorte que `service_id` est toujours comparable. Les projections et
normalisations sont exactement celles du plan V4.12 pinné. La clé HMAC
historique est lue en lecture seule et son identifiant ainsi que son SHA sont
vérifiés avant usage.

`input_siret_lineage` reste matériellement pinné mais est explicitement exclu
du claim : il exigeait le SIRET brut du CRM historique, champ volontairement
interdit dans le nouveau CRM. Le SIRET autoritatif du mapping n'a pas la même
sémantique et ne lui est jamais substitué. Il est donc interdit de publier
« zéro hit sur quatre keysets » ; la publication dit « trois keysets
applicables, input_siret_lineage non applicable par conception ».

Le rapport publie, pour chaque keyset applicable puis pour leur union, le nombre de lignes
comparables, de hits et de lignes uniques en collision. Le seuil autorisé est
zéro sur chaque keyset et sur l'union. Chaque SIREN autoritativement connu,
y compris dans les lignes ambiguës ou contradictoires, est aussi comparé au
registre gelé des 19 754 SIREN ; l'intersection autorisée est zéro.

Une copie ancienne dotée d'un nouvel `export_id` échoue donc. Une ligne non
comparable à l'un des trois keysets applicables provoque `STOP` ; elle ne
devient jamais un non-hit silencieux.

## 6. Qualification, IDs et splits

L'ID opaque est le SHA-256 de :

```text
UTF8("SIRETO-V413-OPAQUE-QUERY-ID\0")
+ canonical_json([collection_manifest_sha256,
                  source_file_sha256,
                  source_row_ordinal_1_based,
                  source_record_id])
```

Chaque nibble hexadécimal `0..f` est ensuite transcodé `a..p`. Le builder
n'importe aucun module de retrieval ou modèle et produit des arbres séparés
`queries`, `oracle` et `audit`. Le scanner refuse les champs interdits et
toute séquence Unicode autonome de 9 ou 14 chiffres après NFKC.

Le split utilise un union-find sur toutes les lignes. Deux lignes sont reliées
si elles partagent un `source_group_id` non vide ou un SIREN autoritativement
connu, y compris ambigu ou contradictoire. Une ligne sans aucun lien forme une
composante singleton. La clé de composante est le JSON canonique du tableau
trié de ses IDs opaques.

On calcule :

```text
digest = SHA256(UTF8("SIRETO-V413-FRESH-SPLIT\0") + component_key)
u = unsigned_big_endian_uint64(digest[0:8])
fit  si u < 12912720851596686131
dev  si 12912720851596686131 <= u < 15679732462653118873
test sinon
```

Les manifests fit/dev/test sont physiquement distincts, scellés avant tout
retrieval et leur intersection de composantes doit être vide.

## 7. Gate source et retrieval gelé

Gate 0B est calculé exhaustivement :

- au moins 657 `MATCH_EXACT` ;
- couverture `MATCH_EXACT / toutes les lignes source ≥ 80,0 %` ;
- zéro fuite et zéro chevauchement interdit.

Sans manifest admissible : `WAITING_FOR_NEW_SOURCE`. Intégrité saine mais
volume/couverture insuffisant : `PIVOT_SOURCE_EVIDENCE`. Fuite, collision,
dérive ou protocole violé : `STOP`.

Le retrieval est exactement celui du contrat actif pinné : mêmes sept canaux,
poids RRF, quotas, profondeur 5 000, tie-break SIRET croissant et plafond
absolu de 100. Aucune variante n'est autorisée. Fit et dev sont exécutés une
fois ; chaque exécution est inscrite dans un ledger.

Le gate dev exige simultanément :

- couverture identifiable ≥ 80,0 % ;
- Recall SIRET exact @100 ≥ 99,0 % ;
- vérité absente du pool comptée comme erreur ;
- zéro liste de plus de 100 candidats ;
- métriques historique, V2, V3 et V4.13 publiées ensemble.

Ranker, decider, risk model et accepteur restent gelés jusque-là.

## 8. Ranker et accepteur

Le présent contrat ne choisit pas encore leurs familles, features ou
hyperparamètres. Avant la première exécution retrieval dev, un contrat modèle
séparé doit pinner les builders de dataset, trainer ranker, générateur OOF,
trainer accepteur, sélecteur de seuil, sealer/publisher, tests, features,
familles, hyperparamètres, seeds et protocole de sélection borné. Il doit
obtenir deux audits indépendants. Sans ce lock, le retrieval dev et toute
phase modèle sont interdits ; aucun choix de méthode ne peut donc être fait
après observation de dev.

Après le GO retrieval seulement, le ranker candidat apprend sur les pools
réels de fit. Ses données pour l'accepteur sont produites en cinq folds OOF
par composante, avec :

```text
fold = uint64_be(SHA256(
  UTF8("SIRETO-V413-RANKER-OOF-FOLD\0") + component_key)[0:8]) mod 5
```

Chaque scène d'accepteur provient exclusivement d'une prédiction OOF. Les
misses retrieval/ranker restent présents et incorrects. Le seuil est choisi
une seule fois sur dev parmi les scores uniques : maximiser le nombre d'AUTO,
sous précision SIRET exacte observée ≥ 99,8 %, zéro `AMBIGUOUS` ou
`UNRESOLVED` en AUTO et au moins `max(50, ceil(25 % des lignes dev))` AUTO.
Les égalités préfèrent moins d'erreurs, puis le seuil le plus élevé. Si aucun
seuil n'est admissible, le produit reste intégralement en REVIEW et le verdict
qualité est `PIVOT`.

## 9. Test fermé et verdict terminal

Les queries test, l'oracle test et les arbres fit/dev sont physiquement
séparés. Aucun entrypoint fit/dev ne peut recevoir un path, un hash ou un
handle de l'oracle test. Après gel du code, des modèles, du seuil, du runtime,
du retrieval et des manifests :

1. créer `test/opening.json` en `O_EXCL` avant le premier FD query test ;
2. scorer une fois puis sceller candidats et décisions ;
3. créer l'événement `results-sealed` avant le premier FD oracle test ;
4. créer l'événement `oracle-open`, calculer les métriques sans rescoring ;
5. sceller métriques et receipt terminal.

Crash avant scellement des résultats : `STOP_NO_RESCORING`. Crash après
scellement mais avant oracle : reprise uniquement depuis les résultats
scellés. Crash après oracle : recalcul des métriques autorisé, rescoring
interdit. Une receipt valide rend toute relance idempotente sans réouverture.

`GO` exige sur test : couverture identifiable ≥ 80 %, Recall exact @100
≥ 99 %, plafond 100 respecté, précision AUTO exacte observée ≥ 99,8 %, zéro
AUTO ambigu/non résolu et volume AUTO minimal identique au dev. `PIVOT`
signifie intégrité saine mais au moins un gate qualité manqué. `STOP` signifie
fuite, corruption, contamination, rerun ou violation de protocole.

Les nombres bruts, Wilson bilatéral 95/99 % et segments sont toujours publiés.
Une garantie 99,8 % n'est revendiquée que si la borne inférieure
Clopper-Pearson unilatérale 99 % est ≥ 99,8 %, avec décisions indépendantes
agrégées au niveau composante. Avec zéro erreur, cela exige au moins 2 301
composantes AUTO indépendantes ; avant cela, on publie « estimation observée ».

## 10. État au préenregistrement

Les 23 609 lignes locales sont consommées et aucune collection admissible
n'est actuellement présente. Cet état est une observation de handover, pas
une constante normative du plan. V4.13 peut donc avancer jusqu'au verrouillage
de l'implémentation ; l'ouverture réelle attendra une nouvelle matière
première et ne sera jamais simulée par des labels reconstruits.
