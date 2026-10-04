# V169 — toutes les preuves de noms, tableaux temporaires paginés

Protocole fixé avant mesure nationale, 4 octobre 2026. V38 INITIAL100 actif,
aval gelé ; aucune métrique après ranker n'est une validation de retrieval.
Le coût des noms150/151 (~2,90Gio dernière sonde151 sous ANCIEN budget3Gio)
ne passe pas la politique V1531Gio actuelle. Le prototype156 uint8 seul n'est
pas une preuve nationale1Gio. Aucune ancienne mesure n'est réinterprétée.

## Invariants

- Toutes les57530682 variantes officielles, tous105564761 liens alias-owner,
  toutes les patches149 et toutes9730972 observations de succession restent
  accessibles. Fermés/historique/UNKNOWN conservés ; aucun token, owner, site,
  relation ou candidat tronqué. Les scores restent float64, pas quantifiés.
- Couverture138 : maximum du taux de tokens distincts par canal pour CHAQUE
  alias entier ; normalisation/channels inchangés. Compteurs156 représentent
  exactement la borne des tokens, sans saturation. Postings triés uniques
  contrôlés par chunk, frontières comprises. Exact alias-current inchangé.
- Scatter des seuls alias démontrablement zéro évité, sinon TOUS leurs faits
  sont parcourus, même fanout supérieur au quantum physique. Patches149 sur
  rows exactes ; MISSING UL reste UNKNOWN, MISSING site fatal. Aucun lift d'un
  nom de site au parent/frères ; parents officiels applicables à leurs sites.
- Graphe137 : snapshot des endpoints directs figé AVANT toutes les relations,
  deux directions, grade0,5 si continuité expliciteTRUE sinon0,25. Selflinks
  sans expansion, maximum des doublons, pas de deux sauts ni de flag exact
  hérité. Appended-sites utilisent leurs parent-proofs explicitement joints.

## Mémoire et intégrité

V169 conserve la validation source complète150 à l'ouverture et au rehash
final. Rotation uniquement des mmap readonly possédées par CE lecteur avec
copie avant fermeture, inode/taille/mtime et dtype/shape inchangés. Pas de
purge système ou d'intervention sur d'autres applications. Les clés opaque
SIREN/SIRET restent des joins, jamais features de similarité.
Après chaque posting entièrement consommé, seule sa partition physique
de quatre arrays est remappée. Les255 autres partitions restent identiques
et accessibles ; pas1024 réouvertures inutiles par token. Hash DuckDB piné,
aucune sélection de noms ou de candidats. Source fullhash avant/après intact.

Les vectors query-owned sont .npy frais, tailles complètes du namespace,
init zéro filesystem ; chaque read/take retourne une copie possédée. Les
updates max/OR ferment leurs propres pages avant l'opération suivante.
Sources ne sont jamais écrites. Nettoyage au succès uniquement des fichiers
temporaires explicitement créés par la requête ; un échec conserve les
fichiers pour diagnostic et ne publie aucun snapshot/certificat complet.

Tests oracle138/149/137/150 bit-exacts, plusieurs quanta et cas de doublons,
fanout, patches, scopes, exact/direct, chaîne/selflinks/UNKNOWN. Fixtures
ne prouvent ni coût ni recall national ; première mesure nationale requise.

## Essai borné

Une seule première requête de l'ordre128 query-only V115 préexistant,
fold2/3/4 seulement, aucun GT, anciens hits/candidats ou label-side ouverts.
Corpus national complet57530682 noms, parents30027199/sites44403034,
graph/raw/namespace149 protégés. Fullhash de toutes les sources à l'ouverture
et après résultat complet. Consommer TOUT le namespace site par gather en
chunks et publier digest/compteurs d'intégrité, pas des candidats ni recall.

Quiet30s/normal pressure/free>=50, worker+parent sous même politique153 :
1Gio RSS,80s, global swap seulement télémétrie ; supervision164 avec exit
propre wait/reaped0 nécessaire. Queryguard20s, scratch plafond3Gio, chunk
131072 uniquement physique (pas cap métier). Arrêt au premier échec, pas de
retry ni hausse de budget ; code/config après mesure restent immuables.

Seals/pack de l'adresse168 progressent séparément. Ce test de noms ne suffit
pas pour INITIAL100 national ou SLO. Dernier exact V101114/128 (89,0625%) sur
pilote réutilisé reste inchangé. Objectif99,2% NON démontré, noREADY.
