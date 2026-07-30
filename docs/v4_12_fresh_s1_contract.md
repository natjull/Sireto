# V4.12 — Contrat S1 d’intake CRM frais

## 1. Objet et autorité

S1 transforme le contrat métier V4.12 en protocole exécutable pour une seule
collection CRM fraîche. Il succède au sas synthétique S0-R3 certifié, mais ne
réutilise ni sa fixture ni son worker comme composant métier.

Le présent préenregistrement n’autorise aucune ouverture de CRM réel. Cette
ouverture reste interdite jusqu’à ce que les catalogues, le code, les tests,
les runtimes, les profils sandbox et le verrou statique S1 soient construits,
scellés et audités, puis qu’une autorisation one-shot soit committée.

S1 ne lance aucun retrieval et ne dégèle aucun ranker, decider, risk model ou
accepteur. Son seul résultat métier est un holdout `READY`, `PIVOT` ou `STOP`.

## 2. Architecture minimale et séparation

S1 comporte quatre frontières, sans framework générique supplémentaire :

1. **Admission manifest-only** : choisit la première collection complète
   admissible, crée un claim exclusif et un receipt d’arrivée avant toute
   ouverture du payload, puis dérive le verrou de collection à partir des
   seuls manifests signés et stables.
2. **Worker Q** : reçoit le CRM, ses manifests et le registre de compatibilité.
   Il contrôle schéma, fuite, provenance et collisions, puis produit les
   requêtes sûres et un bridge privé. Il ne reçoit aucune preuve, aucun oracle
   et aucun SIRET de vérité.
3. **Worker E** : reçoit uniquement le bridge
   `query_id/source_batch_id/source_record_id/reference_date`, le paquet de
   preuves, le catalogue de preuves et le registre minimal des SIREN
   consommés. Il ne reçoit aucun nom, adresse, commune ou code postal CRM.
4. **Scorer ultérieur** : reçoit uniquement les requêtes scellées après
   `READY`. Il ne peut ouvrir ni bridge, ni preuve, ni oracle.

Le broker parent ne désérialise pas les lignes CRM. Il ouvre et rehash les
artefacts par descripteurs ancrés, transmet seulement les rôles autorisés,
valide les arbres exacts et effectue les promotions non-clobber.

## 3. Vérité indépendante

Une correspondance `MATCH_EXACT` ne peut venir que d’un mapping contractuel
autoritatif `source_record_id → SIRET`, d’un identifiant officiel
préenregistré ou d’un document administratif scellé. Cette preuve doit être
antérieure à la qualification, traçable, temporellement compatible avec
`reference_date` et indépendante des sorties SIRETO.

La ressemblance nom/adresse, les candidats, les rangs, les scores, les
prédictions et les sorties de modèle sont interdits. Un snapshot SIRENE peut
seulement confirmer l’éligibilité ou révéler une contradiction pour un SIRET
déjà autorisé par une preuve indépendante.

- un seul SIRET autorisé, éligible et cohérent produit `MATCH_EXACT` ;
- plusieurs SIRET compatibles ou un SIREN seul produisent `AMBIGUOUS` ;
- l’absence, l’expiration ou la contradiction des preuves produit
  `UNRESOLVED`.

Si le producteur ne fournit pas de mapping autoritatif suffisant, S1 publie la
faible couverture et termine `PIVOT`. Il n’invente pas de labels et ne demande
pas à l’utilisateur de les valider.

## 4. Autorités pré-CRM

Le verrou statique S1 épingle byte-for-byte :

- le contrat et le plan d’intake V4.12 ;
- le receipt terminal S0-R3 certifié ;
- les catalogues développement et artefacts interdits ;
- le registre des SIREN consommés ;
- le registre de compatibilité, ses quatre keysets et l’identité/hash de sa
  clé HMAC Keychain ;
- le catalogue source et le catalogue de preuves ;
- le commit d’implémentation et chaque blob exécutable/test ;
- les deux profils sandbox, les closures runtime et leurs smokes ;
- l’UID, le device, l’UUID du volume, le binaire `sandbox-exec`, les golden
  vectors et les manifests de canaris.

Il est interdit de résoudre un chemin `latest`. Toute dérive de hash, de
schéma, de type, de nullabilité, de rôle FD ou d’identité produit `STOP`.

Chaque catalogue est composé de deux fichiers distincts. Le payload canonique
contient les règles, les chemins et hashes des configurations/snapshots, mais
jamais son propre hash. Un seal canonique séparé contient exactement
`schema_version`, `catalog_id`, `payload_size_bytes` et le SHA-256 des octets
exacts du payload. Cette convention non autoréférentielle est obligatoire.

Le payload du catalogue source ferme les producteurs et clés de signature
autorisés, les couples `source_system/portfolio_id`, les schémas exacts, les
versions, la sémantique `source_record_id`, la preuve de lignée et la règle
`FIRST_COMPLETE_ADMISSIBLE_COLLECTION`.

Le payload du catalogue de preuves ferme les types de preuve, codes de
provenance, priorités, temporalité, snapshots et hashes, schémas
d’entrée/oracle, builder, tests, runtime et configuration. Les snapshots fixes
peuvent être vides ; dans ce cas seul le paquet de preuve producteur signé
peut créer une vérité.

Les manifests collection, source et evidence contiennent `producer_id` et
`producer_key_id` et sont signés en Ed25519 par l’unique clé active
correspondant à ce couple dans le catalogue source. La signature est le
Base64 RFC 4648 canonique des octets canoniques du manifeste dont seul
`producer_signature` est exclu. Une clé inconnue, révoquée ou une signature
invalide produit `STOP`.

## 5. Verrous, identité et one-shot

Le verrou statique est scellé avant l’existence de l’inbox réelle. Après
réception, l’admission ne lit que les manifests. Elle vérifie signature,
exhaustivité, stabilité d’au moins 60 secondes et catalogues, puis crée avec
`O_EXCL` :

1. le claim global de sélection ;
2. le receipt d’arrivée manifest-only ;
3. le verrou dynamique de collection.

`collection_id` est dérivé du hash du manifeste collection, du hash du
catalogue source et de la séquence d’arrivée exclusive. `attempt_id` est
dérivé du verrou statique, du verrou de collection et du temps logique
annoncé. Les domaines et projections sont fermés par le plan.

S1 n’admet qu’un producteur actif. Le catalogue source épingle son ledger
d’exports, son hash de tête et le prochain numéro attendu. Le manifeste
collection signé porte `producer_export_sequence`, l’identifiant du ledger et
le hash de l’entrée précédente. Le broker n’accepte aucun chemin cible en
argument ou variable d’environnement : il énumère tous les manifests enfants
directs de l’inbox, enregistre leur état, et ne peut sélectionner que la
collection complète portant exactement le prochain numéro attendu et le bon
lien de chaîne. Un numéro supérieur avec un trou attend sans ouvrir de
payload ; deux manifests au numéro attendu produisent `STOP`.

La séquence technique d’arrivée est ensuite allouée sous mutex sur le
répertoire ancré et
produit un enregistrement monotone `O_EXCL` qui ferme chemins et hashes de
tous les manifests. Le mutex est conservé jusqu’au `F_FULLFSYNC` du claim :
la plus petite séquence durable gagne sans course entre allocation et claim.
Cet enregistrement constitue l’autorité de reprise avant payload sans
reparcourir une inbox non receipted. Claim, receipt, lock dynamique, marqueur,
checkpoints, événements, manifests d’événements, seals et receipts terminaux
suivent tous : octets canoniques dans un fichier `O_EXCL`, `fsync`,
`F_FULLFSYNC`, synchronisation du répertoire parent, vérification par FD
ancré, puis seulement transition suivante. Un marqueur exclusif
`PAYLOAD_OPEN_POSSIBLE` est écrit avant le premier FD de payload. Le lock
dynamique exclut son propre hash et `attempt_id` de sa projection ; son temps
logique vient du manifeste collection signé.

Un claim sans receipt après ouverture potentielle produit
`STOP_NO_RERUN`. Avec receipts et checkpoints valides, la reprise conserve le
même attempt et repart uniquement des arbres scellés, jamais de l’inbox. Un
receipt terminal rend tout lancement ultérieur idempotent et interdit un
nouveau worker.

L’automate exact est :
`UNCLAIMED → MANIFESTS_CLAIMED → ARRIVAL_RECEIPTED →
COLLECTION_LOCKED → PAYLOAD_OPEN_POSSIBLE → QUERY_SEALED →
EVIDENCE_QUALIFIED → FINALIZED → READY|PIVOT`, avec `STOP` depuis tout état
non terminal. La reprise suit receipts, dernier manifeste d’événements
complet, puis checkpoints.

## 6. Schémas et diagnostics

Les manifests collection, source et evidence utilisent des ensembles exacts
de champs, types et nullabilités définis par le plan : chaque champ possède
exactement un type et aucun champ supplémentaire n’est admis. Les contraintes
croisées définissent aussi le paquet evidence absent. Le champ
`v411_service_id_equivalence_attested` est un booléen exact. Une valeur vraie
exige une référence de lignée non vide et vérifiable ; une valeur fausse
termine `STOP_UNPROVABLE_LINEAGE`.

Les diagnostics publics sont limités à `phase` et `reason_code` dans des
énumérations fermées. Sur échec, un seul JSON canonique de 512 octets maximum
est admis ; exit code, signal et présence du diagnostic suivent la matrice
fermée du plan. stdout et stderr sont vides sur succès. Aucun nom, adresse,
SIRET, SIREN, `source_record_id`, chemin privé, traceback ou représentation
d’exception ne peut être journalisé.

Tous les outputs privés sont `0700/0600` sous `umask 0077`. Les requêtes
scorer ne contiennent que les IDs opaques, la date et les champs CRM utiles ;
le bridge et la table inverse restent dans l’audit privé.

## 7. Sandbox et tests préalables

Worker Q et Worker E ont des profils deny-by-default distincts, sans réseau,
fork, modèle, worktree, cache retrieval ni répertoire de développement. Les
entrées arrivent uniquement par FDs scellés. Worker Q peut effectuer le calcul
HMAC via la fiche Keychain épinglée, sans UI ni sérialisation de clé. Worker E
n’a pas accès au Keychain.

Avant toute ouverture réelle, un gate entièrement synthétique multi-batch
doit couvrir :

- trois batches, deux portfolios, ordre, doublon, batch manquant/extra et
  cutoff ;
- schémas fermés, mutations, symlink/hardlink, stabilité et lecture à EOF ;
- vecteurs Unicode 9/14 chiffres et absence de fuite ;
- collisions des quatre keysets et attestation de lignée ;
- preuves unique, multiple, absente, contradictoire et expirée ;
- frame de plus de 657 lignes avec une erreur tardive pour interdire l’arrêt
  anticipé ;
- crashes à chaque frontière durable, reprise, course de deux launchers et
  anti-rerun ;
- canaris réseau, modèles, données historiques, preuves depuis Q et CRM depuis
  E ;
- arbres séparés queries/evidence/oracle/audit, permissions et diagnostics.

Deux gates sont distincts. Deux audits du présent préenregistrement rendent
`GO_S1_IMPLEMENTATION` et n’autorisent que les catalogues et l’implémentation
synthétique. Après la suite complète et le gate multi-batch, deux nouveaux
audits doivent rendre `GO_S1_REAL_CRM_OPEN` sur les mêmes code, catalogues,
runtimes, profils et hashes. Seul ce second verdict autorise le verrou puis
l’autorisation one-shot réels.

## 8. Gates métier

Toutes les lignes de la frame immuable restent au dénominateur. `READY` exige :

- qualification exhaustive et indépendance valide ;
- couverture `MATCH_EXACT / toutes lignes source >= 80,0 %` ;
- au moins 657 `MATCH_EXACT` ;
- zéro fuite SIRET/SIREN dans les requêtes ;
- manifests, receipts, ledgers et arbres valides.

Un échec de volume ou de couverture avec intégrité saine produit `PIVOT`.
Une fuite, une dérive, un optional stopping, une lignée improuvable ou une
violation one-shot produit `STOP`.

Le retrieval frais n’est exécuté qu’après `READY`, une seule fois, avec un
plafond absolu de 100 candidats. La vérité absente du pool compte comme miss.
Les métriques historique, V2, V3 et holdout frais sont publiées ensemble.
Recall SIRET exact observé à 100 doit atteindre 99,0 % pour `GO`; le ranker et
l’accepteur restent gelés jusqu’à ce gate.

Avant toute ouverture des requêtes, un scoring freeze scelle build, commit,
sources, plan, lock, runtime, inputs et politique du retrieval, ainsi que les
références historique/V2/V3. Le scorer crée `OPENING.json` avec `O_EXCL`, puis
un événement `SCORING_OPEN_COMMITTED`, avant le premier FD query. Il ne reçoit
ni oracle, ni evidence, ni bridge, ni ranker/decider/risk/accepteur. Résultats
et candidats sont scellés avant `ORACLE_OPEN_COMMITTED`; l’évaluateur seul
ouvre ensuite l’oracle. Une reprise rattache des résultats scellés et
n’exécute jamais un second scoring.

Les événements d’évaluation sont des JSON canoniques `O_EXCL`, nommés par leur
SHA-256, chaînés par `previous_event_sha256` et durablement synchronisés. Un
receipt terminal ferme opening, manifeste d’événements, seals résultats,
candidats et métriques. Un crash avant le scellement des résultats produit
`STOP_NO_RESCORING`; après scellement, seuls les résultats peuvent être
rattachés et les métriques recalculées.

## 9. Séquence autorisée

1. préenregistrer ce contrat et son plan canonique ;
2. obtenir deux audits `GO_S1_IMPLEMENTATION` ;
3. construire et sceller les catalogues ;
4. implémenter et tester uniquement sur fixtures synthétiques ;
5. auditer code, runtimes, profils et gate synthétique ;
6. sceller le verrou statique S1 ;
7. committer l’autorisation one-shot ;
8. seulement alors accepter et ouvrir la première collection CRM réelle.
