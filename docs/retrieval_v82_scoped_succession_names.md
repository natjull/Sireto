# V82 — Noms officiels scopés, factorisés autour des successions

Préenregistrement source-only avant mesure. Objectif recherche initiale
100 exact99,2% inchangé ; aval gelé. V81 fournit7512820 observations entre
deux endpoints officiels, avec date/direction/graduation de site, pas des
alias légitimes transférés. Le dernier vrai recall reste V78,89,0625% pilote.

Lire les162537492 observations nationales de noms de la source canonique,
sans CRM/labels/teacher/anciens candidats. Garder TOUT nom nonvide des
endpoints réellement présents dans V81 (SIRET ET SIREN officiels), et de
leurs personnes morales (SIREN officiel). Aucun site frère dont le SIRET
n'est pas un endpoint ne devient source de nom. Les noms SIREN ne sont
pas fabriqués comme alias d'établissement : grain et portée restent distincts.

Conserver source,kind,evidence_id,source_record_id,dates validité/observation,
statut/priorité. Aucune coupe par récence, état, popularité, cap d'alias,
préfixe CRM ou erreurs de recall. Toutes observations répétées conservées,
pas autant de faits lexicaux uniques revendiqués. Enseigne déclarée au grain
SIREN reste CONTEXT_ONLY_SITE_SCOPE_UNVERIFIED ; ne pas supposer identité ou
enseigne universelle de tous sites. Noms légaux/usages/commerciaux au grain
SIREN sont preuves de personne morale, enseignes au grain SIRET preuves
propres à cet endpoint. Kinds inconnus contexte uniquement, conservés.

Représentation FACTORISÉE : noms par propriétaire officiel + références vers
la table des successions/date/sites V81. Pas une jointure explosant chaque
nom sur tous les liens, pas de nouveau nom actuel inventé pour le target.
Cette étape n'est pas un index ni une recherche ; les IDs servent aux
jointures/provenance uniquement. La future ingestion lexicale devra appliquer
le masque d'identifiants V56, et vérifier visibilité temporelle si date CRM
connue ; pas de revendication de co-validité entre nom et succession ici.

Sortie Parquet complète des noms pertinents et scopes d'endpoints (pas un
corpus choisi autour du GT). Gardes80s/4Gio/zéro swapout,DuckDB1GB/2threads,
spill1GB, output+spill≤2Gio, supervision90s. Sources complets entièrement
hashés avant/après, mêmes162,5m/7,5m, aucun externe/payé/fold0/1. STOP sans
élargissement si la garde échoue. Prochain collecteur devra interroger
directement les chemins avec score initial/certificat jusqu'à100, sans seed
parent déjà retrouvé, pool fixe ou modèle aval. Aucun recall/SLO validé ici.
