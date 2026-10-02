# V73 — Preuve explicite de propriétaire légal RNE dans la même formalité

Préenregistré après audit STRUCTUREL V72, avant V73. Le prefix RNE32768
observations contient14627 parents et18141 sites ; la plupart des sites
n'ont pas de nom propre, les noms légaux sont portés par les parents.
Les prefixes BODACC audités sont tous SIREN : ne pas les transformer librement
en SIRET ni assimiler ASSET_TRANSFER/succession à identité de site.

Composer UNIQUEMENT LEGAL de parent RNE SIREN avec adresse d'un site RNE
SIRET si SIREN égaux ET le site est EXPLICITEMENT enfant de LA MÊME formalité :
site.source_record_id==parent.source_record_id+':site:'+site.siret.
IDs servent à jointure et provenance, pas feature de similarité. Pas de join
par nom ou voisinage, pas de propagation sibling→sibling, pas d'autre source
ou formalité au seul motif d'un SIREN égal. Parent dupliqué ambigu impose STOP.

Il s'agit d'une preuve de LEGAL_OWNER_SAME_FORMALITY, **pas de preuve de
validité temporelle commune** : valid_from manquant sur ce prefix, sites
fermés inclus. Publier explicitement UNKNOWN, dates parent/site distinctes et
compteurs fermés/manquants. L'observation composée est DERIVÉE, pas une
réécriture d'OfficialEvidence canonique ; sidecar de liaison contient les deux
evidence_id et la règle. Aucune nouvelle adresse copiée depuis le parent,
aucun nom TRADE transféré, alias locaux restent séparés, aucun alias fabriqué.

Même premier prefix RNE32768 que V72, indépendant CRM/labels, pas de sélection
sur pertes GT. Autorité V72 hashée + source size/mtime/inode/footer avant/après,
digest du prefix lu. Même CRM8, et8 sentinelles de coût parmi mots officiels
de site APRES liaison, non données CRM synthétiques/qualité/train. Index brut
et composé interrogés séparément, chacun oracle sur ENTIER cohort,32 parités.
Collector100 SIRET DISTINCTS, aucune fusion/rerank ou pool2000/500.

Publier couverture nominale sites avant/après, liaisons, ambiguïtés/UNKNOWN,
sites distincts, budgets et timings, parité ordre/score. Pas de candidats
écrits, aucun label lu ni fit/modèle aval, aucune mesure recall humain.
80s/4Gio/zéro swapout,CPU2,superviseur90s,STOP immuable. Ce n'est ni un test
national ni une validation SLO. Gros fichiers non rehashés entièrement.

Suite distincte nécessaire : BODACC légal→site par preuve officielle
d'adresse/site, joins typés temporels, relations/filiations pertinentes,
apprentissage OOF, XY/BAN et exécuteur NATIONAL. Ne pas annoncer leur usage
comme réalisé par cette seule composition RNE. Gate99,2% exact reste ouvert.
