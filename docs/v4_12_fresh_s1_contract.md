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

Le catalogue source ferme les producteurs et clés de signature autorisés, les
couples `source_system/portfolio_id`, les schémas exacts, les versions, la
sémantique `source_record_id`, la preuve de lignée et la règle
`FIRST_COMPLETE_ADMISSIBLE_COLLECTION`.

Le catalogue de preuves ferme les types de preuve, codes de provenance,
priorités, temporalité, snapshots et hashes, schémas d’entrée/oracle,
builder, tests, runtime et configuration. Les snapshots fixes peuvent être
vides ; dans ce cas seul le paquet de preuve producteur signé peut créer une
vérité.

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

Un claim sans receipt après ouverture potentielle produit
`STOP_NO_RERUN`. Avec receipts et checkpoints valides, la reprise conserve le
même attempt et repart uniquement des arbres scellés, jamais de l’inbox. Un
receipt terminal rend tout lancement ultérieur idempotent et interdit un
nouveau worker.

## 6. Schémas et diagnostics

Les manifests collection, source et evidence utilisent des ensembles exacts
de champs, types et nullabilités définis par le plan ; aucun champ
supplémentaire n’est admis. `v411_service_id_equivalence_attested` est un
booléen exact. Une valeur vraie exige une référence de lignée non vide et
vérifiable ; une valeur fausse termine `STOP_UNPROVABLE_LINEAGE`.

Les diagnostics publics sont limités à `phase` et `reason_code` dans des
énumérations fermées. stdout et stderr des workers sont vides sur succès et
bornés sur échec. Aucun nom, adresse, SIRET, SIREN, `source_record_id`, chemin
privé, traceback ou représentation d’exception ne peut être journalisé.

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

La suite complète doit être verte et deux audits indépendants doivent rendre
`GO_S1_IMPLEMENTATION` sur le même commit et les mêmes hashes.

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

## 9. Séquence autorisée

1. préenregistrer ce contrat et son plan canonique ;
2. obtenir deux audits `GO_S1_IMPLEMENTATION` ;
3. construire et sceller les catalogues ;
4. implémenter et tester uniquement sur fixtures synthétiques ;
5. auditer code, runtimes, profils et gate synthétique ;
6. sceller le verrou statique S1 ;
7. committer l’autorisation one-shot ;
8. seulement alors accepter et ouvrir la première collection CRM réelle.
