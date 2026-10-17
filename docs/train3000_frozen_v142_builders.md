# Train3000 — constructeurs retracés et blocage d'exécution, 17 octobre 2026

Suite autorisée `sireto-site-target-8.md`, après tirage commité `d54cec2c`.
**Jalon partiel, pas un run3000 ni une fidélité quatre familles validée.**

## Chemin de construction historique à conserver

| Famille | Constructeurs réellement producteurs des entrées526 | Sources/paramètres figés |
|---|---|---|
| Numérique51 | V14 `prepare_zone` + `dense_zone(v3_official512)` ; V18 `prepare`/banque d'identité et V19 `generate` pour les cinq scores propres ; V23 `make_features`/`prepare` pour22 preuves ; V32 `prepare` conserve celles-ci ; V57 nullsafe délègue V56 `prepare`, V68 `prepare`, V71 `prepare`, puis **V75 `prepare`** est le dernier producteur51. V81 et les couches suivantes transmettent ces mêmes matrices. | Même corpus national/commune complet, dictionnaire noms officiels, Gemma best_adapter step1000, vues512, float32, banques propre/parent séparées, sources de rôles corporate, atlas public, groupe historique et section clients publiée. Aucun V173/site-target. |
| Activité | **V112 `main`**, puis V114 `prepare` et **V115 `prepare`**, projection sans perte en refs S9/confiance float64 par requête. | Toutes les années BODACC déjà archivées, objet établissement unique/sujet corporate public, cutoff2026-08-01, tokens V49, IDF propriétaire, plafond0,5 et max, pas somme. Pas de remplacement par un shard526 vide. |
| BANO | **V141 quarantined `main`** exécute la source nationale immuable avec la politique uniforme de quarantaine ; boucle query-context de `v141_bano_national_source.main` ; **V142 `prepare`** calcule ensuite les deltas pondérés. | Même archive nationale `full.csv.gz`, mêmes lignes source OSM uniquement, SQLite national de la génération `_night_d`, mêmes communes/déléguées, rue normalisée exacte, ensemble des points maison sinon rue, max géo0,5/(1+d/300), coefficients58 inchangés. |
| OSM | **V129 `osm_disk_category_source.main`**, continuation liée de `osm_disk_source.main` (catégories séparées), puis projection query_context/shapes ; **V143 `prepare`**, continuation vide seulement de V132, calcule geometry/named. | Même PBF national `france-261012.osm.pbf` (5 097 546 976 octets, présent), rayons contexte2000m/buffer3500m, catégories/rues/POI/adresses inchangées, géométrie d'objet propre, pas de serveur Nominatim ni téléchargement. |

Les sources nationales sont locales ; l'autorisation de matérialisation est
acceptée. **Le précédent STOP « mêmes caches » n'est plus le blocage.** Les
modèles ne doivent pas être édités pour ajouter artificiellement des heldout IDs :
l'affectation des nouveaux dossiers est prouvée hors entreprise dans le tirage,
les dix fichiers modèles/coefficient restent ceux du gel.

Les quatre familles ne suffisent pas à elles seules aux arguments du collecteur :
le replay V173 `prepare_queries` expose aussi geometry, context_expansion,
initial_alignment, supplier, address, RNA et BANATIC. Ces contextes sont eux aussi
liés aux query_id du panneau ; leurs préparateurs historiques doivent être
appliqués aux nouvelles requêtes sans labels, pas remplacés par des listes vides.
Les sources/builders incluent V17/BAN local, V88/V98, V133, V137 et V130.
Ce sont des dépendances de la même recette, pas des sources ou features nouvelles.

## Adaptateur BANO livré, périmètre volontairement explicite

`scripts/train3000_frozen_v142_bano.py` reprend **sans changer son corps** la
boucle query-context historique. Il réutilise le SQLite national en
`mode=ro&immutable=1` (pas un cache limité aux526) et les mêmes références
communes/déléguées. Aucun label ni modèle en entrée du constructeur.

- Source originale SHA : `bbc819028db034bbbec59a4510be2b0853beb1cf16f960867e87abacbf5a5d14`.
- Corps extrait SHA : `02038f338ea091bfa292481008b8ca24a5bedc50fad18aa6029fb725dd43f87a`.
- Quatre helpers (préparation partagée, parseur numéro, normalisation, L93)
  revérifiés contre leurs octets au commit historique `ae7c3e05`, puis épinglés.
  Avant toute tentative worker, le premier contrôleur61686 a été arrêté proprement
  pour renforcer ces seules vérifications ; source originale et chaîne de
  provenance archivées dans `bano/code_revisions/source_guard/`, corps inchangé.
- Le vrai corps a été vérifié/compilé localement ; ce n'est PAS une preuve
  numérique ou une fidélité20 exécutée.
- À l'admission VM arrêtée : reconstruire les20 contextes pilotes sélectionnés,
  comparer les octets JSON canoniques aux contextes526, STOP au moindre écart ;
  puis matérialiser les3000 nouveaux contextes dansv174. Les CRM sont revérifiés
  contre les fichiers source projetés/scellés, jamais contre des labels.
- Ce stade **ne calcule pas les deltas BANO pondérés**, ne reconstruit pas les
  matrices51, activité ou OSM, ne réalise aucune recherche ni smoke natif.
- Il réemploie `retrieval_site_target_full.supervised` et ses sondes disque/VM/RSS
  **sans changer leur code**, seulement des bindings volatils privés pour le
  namespace, le point d'entrée worker et l'artefact attendu. Aucun fichier V173
  n'est changé. Reçus de ses propres enfants, locks, attente60s, RSS8Gio et
  disque10Gio ; kill uniquement d'un enfant possédé si la VM reprend/limite atteinte.
  Aucune nouvelle lecture/écriture de pending préexistant.

## État / limites réels de ce jalon

À l'observation du17/10 : **hydraulic Running**. Aucun worker lourd admis.
Le census de tirage (8,30s/422,4Mio, comptage seulement) est le seul calcul mesuré.
Les trois autres adaptateurs de reconstruction et les contextes associés ne sont
pas encore implémentés pourv174 ; fidélité complète0/20, smoke natif0/20,
génération0/3000, aucune ETA préparation/génération mesurée. Il serait faux
d'extrapoler les temps du census ou de lancer le full sous le gate30h.

La reprise BANO est détachée, **PID61903**, état observé `WAITING_HYDRAULIC`,
aucun enfant lourd, mais **ce n'est pas un superviseur autonome de tout le brief**.
L'état daté et les mesures exactes sont dans
`docs/train3000_frozen_v142_extension_status.json`. Le générateur complet, le gate ETA,
l'évaluation post-sceau et le commit automatique final ne sont pas réalisés ici.
Les rapports préflight `train3000_frozen_v142_result.*` restent historiques,
non remplacés par un faux résultat final. Les étapes restantes exigent une
reprise de travail sur les adaptateurs sous admission VM arrêtée.

Suivi BANO : `.venv/bin/python scripts/train3000_frozen_v142_bano.py --status`.
Tirage : `.venv/bin/python scripts/train3000_frozen_v142_cohort.py --status`.
Logs : `H/v174/train3000_frozen_v142_20261017/extension_20261017/bano_launch.log`.
