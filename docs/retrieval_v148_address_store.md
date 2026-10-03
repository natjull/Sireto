# V148 — driver complet du dictionnaire d'adresses et accès inverse

Nouveau driver non mesuré nationalement. Aucun lancement national, recall
ou SLO revendiqué. Il refuse une projection partielle, même si ses lots
réussis sont hashés. V144 n'a pas de manifest global à ce stade.

## Prérequis et phases

Un manifest V143/V144 final complet, son SHA256 exact et sa namespace V141
sont requis explicitement. Le driver contrôle toutes les sources originales,
roles et parts en ordre,312 parts dans le plan national,185993425 lignes
source et317683879 observations. Chaque part importée/nouvelle passe le
validateur V143 INCHANGÉ, audits de tous champs/scope et budgets inclus.
Les identifiants observation restent les positions globales du plan, sans
re-indexer les seuls lots disponibles. Une absence du manifest complet STOP
AVANT de créer un root ou lancer un worker. Folds0/1/CRM/labels/fit fermés.

Quatre phases séquentielles :

1. tous les lots projetés sont partitionnés V145 par texte d'adresse ;
2. tous les buckets produisent dictionnaires/faits/liens d'observations ;
3. tous les faits sont repartitionnés V147 par intervalle physique de site ;
4. TOUS les intervalles, y compris vides, produisent leur CSR inverse.

Chaque seal exige l'ensemble de sa phase, cardinalités/namespace/provenance
exactes, hashes d'artefacts et sources avant/après. Le manifest final dit
explicitement `lexical_address_access_built=false` : il ne prétend pas avoir
encore construit le lexique d'adresses ou le moteur national initial100.
Les adresses parent/masquées restent dans les sources originales scellées,
sans propagation vers des sites ; toutes observations admissibles sont
liées aux faits, jamais seulement aux derniers enregistrements.

## Exécution et intégrité

Le root est immuable ; une phase présente sans manifest terminé est
STOP_OR_LIVE, jamais relancée automatiquement. `remaining` ne saute que des
réussites validées et suit les mêmes phases/seals. Worker80s internes/90s
superviseur/3GioRSS/ZÉRO swapout global restent INCHANGÉS. Gate système30s
init/1s phase, min50% mémoire disponible et attente55s max, compteur stable
jusqu'au début du Budget. DuckDB1.4.3/NumPy2.4.1 fixés ; threads1,
256MiB mémoire DuckDB et3GiB spill. Seuils stockage bornent la construction,
pas les preuves ou candidats ; tout dépassement STOP sans tronquer.

Les seals eux-mêmes sont sous le même Budget et gate, pas seulement les
workers. Leur JSON de succès n'est écrit qu'APRÈS validation des ressources ;
un STOP ressource de seal est conservé et interdit de retenter ce root.
Les sorties nationales sont limitées au root audits CATNAT configuré.

La politique worker1Go/pression mémoire avec swap publié séparément demandée
à l'utilisateur n'est PAS implémentée ni implicitement autorisée ici. Un
futur changement exigera un nouveau protocole/version/root et validation
explicite des sources mesurées sous sa politique ; ce driver ne réinterprète
pas les anciens STOP ou les budgets V143. Aucune dépense/cloud/GPU.

## Ce qui reste pour l'objectif

Il faut d'abord terminer/sceller la projection réelle. Puis mesurer cette
construction sans garanties de temps/RSS non mesurées, bâtir les canaux
lexicaux d'adresses et brancher query-side noms V138/relations/attributs/XY
dans V146. Le parcours complet sert d'oracle aux bornes sûres, pas encore
de promesse SLO. Sceller128 sorties avant labels, puis OOF all-human,
exact/op séparés, Wilson/généralisation/SLO et objectif99,2% inchangé.
Aucun positif injecté/aliasCRM/filtre géographique dur ou modèle aval appelé.
