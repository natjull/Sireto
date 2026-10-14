# Mesure supplémentaire « site opérationnel » — contrat du 14 octobre 2026

Contrat figé avant calcul. Analyse secondaire en lecture seule du run
`/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2/v142/bano_native_20261013_night/all_new_sources_max_bano`,
variante `learned_native_ooc`, toutes ses 526 requêtes de développement,
folds 2/3/4 uniquement. Aucun nouveau retrieval, rescoring, modèle, label,
candidat injecté, accès web/API, jeu fermé ou promotion produit.
Le recall SIRET humain exact reste principal ; cette mesure n'est ni
l'opérationnel historique (qui accepte toujours l'exact), ni un nouveau gate.

## Population, preuves et vérifications

Les candidats sont exclusivement les sorties scellées `outputs.json`, au
plus 100 par requête. Le panel CRM et la géométrie CRM sont ceux référencés
par `v142/bano_native_20261013_night/launch.json` (`panel`, `geometry`).
Les labels sont lus dans le ledger officiel référencé par
`config/retrieval_v358_postseal_evaluation.json`, avec filtres fold 2/3/4 ET
les 526 query_id avant matérialisation. L'opérationnel existant est recalculé
avec l'exact et les acceptables du sidecar officiel scellé de cette évaluation,
avec les mêmes filtres. Les vues exact/SIREN/opérationnel doivent reproduire
507/511/508 et les booléens existants par requête : sinon STOP avant toute
publication de la nouvelle mesure. Aucun statut ne filtre le dénominateur.
Les petits bindings et le sidecar utilisés sont vérifiés par SHA256 ; les
gros parquets conservent leurs références héritées, taille et mtime avant/après
(pas de nouveau hash complet). Les sources restent immuables.

## Succès « site opérationnel »

Il faut au moins un candidat émis du **même SIREN que le label humain** ET
une preuve par rapport à **l'adresse/position CRM**, jamais à celle du label :

- **Adresse** : même INSEE non vide, même numéro strictement positif et même
  suffixe, même voie non vide normalisée. On réutilise `normalize(...,
  street=True)` de `scripts/embeddinggemma2_v17_structural_retrieval.py`
  (majuscules, accents/ponctuation retirés, abréviations AV/BD/BLD/R/RTE/CHE/
  CH/IMP/PL/ALL/ST/STE développées). Le numéro est parsé séparément : entier
  et suffixe BIS/TER/QUATER ou lettre ; zéros initiaux ignorés, aucun numéro
  inventé. Une plage explicite de numéros SIRENE n'est pas une concordance
  de numéro unique. Pas de fallback CP, de fuzzy-match ou de règle entreprise.
  Les compléments bâtiment/étage et l'état administratif ne changent pas
  cette mesure, conformément à la définition demandée.
- **Distance** : Euclidienne Lambert-93 EPSG:2154 entre les points CRM et
  candidat, sans arrondi avant comparaison. T=200 m si les deux points sont
  classés numéro ; T=500 m si au moins un est rue/imprécis. Commune, qualité
  inconnue ou coordonnées absentes/non compatibles sur un des deux points
  interdisent la preuve par distance dans la règle principale.

L'exact n'est **pas** un succès site automatique. Toutes les comparaisons
utilisent les adresses courantes du stock SIRENE du 2026-08-01, référencé par
`launch.stock_source`, et le CRM original du panel.

## Sources de points et correspondance de précision figées

L'inspection locale a identifié les coordonnées SIRENE courantes v142
`site_xy.npy` sans score/type, les points BAN/BANO et l'observation SIRENE
statistique typée utilisée séparément par v171. Pour ne pas transférer une
qualité entre coordonnées différentes, la présente mesure choisit **une
seule source candidat typée**, sans sélection après résultats :
`/Volumes/CATNAT_DATA/oubusiness/raw/sirene/geolocalisation_etablissements_2026-07-01.parquet`
(champs siret, x, y, qualite_xy, epsg, distance_precision). Pas de réparation
ou substitution par site_xy, BANO, adresse du label, ni de nouveau géocodage.
Une absence de cette source conserve la voie adresse, sinon c'est un échec.

Correspondance, indépendante des labels et hits :

| Source | Métadonnées | Classe de la règle principale |
|---|---|---|
| SIRENE étude juillet | qualite_xy 11 ou 21 | numéro (équivalent le plus proche) |
| SIRENE étude juillet | 12 ou 22 | rue/imprécis (imputation dans la rue) |
| SIRENE étude juillet | 33 | commune (imputation dans la commune, pas forcément centroïde) |
| SIRENE étude juillet | autre/absent | inconnue, distance principale interdite |
| BAN CRM v17 | `STREET_FOOTPRINT_NOT_HOUSE` | rue/imprécis, centroïde de la rue conservé |
| BAN CRM v17 | entrée/bâtiment/logement, sans `ban_declared_weak` | numéro |
| BAN CRM v17 | `ban_declared_weak`, parcelle/segment/autre/inconnue/délivrance postale/service technique/type vide | rue/imprécis si résolu et compatible |
| BAN CRM | type explicitement municipality/commune/city | commune |
| Toutes | absence, non fini, CRS incompatible | coordonnées absentes |

`ban_source_position=commune` est une provenance municipale, **pas** un
niveau commune. Les coordonnées doivent être finies et dans le domaine
Lambert-93 utilisé par v142/v171 : 0<=x<=1300000, 6000000<=y<=7300000.
Pour l'étude, epsg doit valoir strictement `2154` ; pour le CRM,
`ban_resolved` et `spatial_crs_compatible` doivent être vrais.
Aucun score de confiance probabiliste n'existe dans ces sources choisies :
`ban_street_similarity`, `ban_uncertainty_m` et `distance_precision` ne sont
pas un score >=0,7. On conserve ces métadonnées sans créer un score.
Les codes 11/21 identifient un numéro, **ne certifient pas** universellement
une entrée précise actuelle ; notamment le CFE peut mesurer la cohérence
fiscale/SIRENE. Ce mapping est une approximation explicite de la règle métier.
`distance_precision=0` ne certifie pas un point parfait. Les dates juillet
(points statistiques) / août (adresses) et la géométrie BAN existante peuvent
différer ; l'analyse ne certifie pas le site fibre ni l'actualité du point.

## Sensibilité préenregistrée et détails par requête

Publier simultanément `même SIREN ET (adresse OU distance<=100/200/500 m)`.
Ces trois seuils plats ignorent la précision, **y compris commune/inconnue**,
mais jamais les coordonnées absentes/non finies/CRS incompatible. Ils ne
remplacent pas la règle principale ; aucun choix du meilleur seuil après coup.

Pour chaque query_id : fold, poids, exact/SIREN/opérationnel existant, site,
trois sensibilités, preuve principale adresse/distance/aucune, précision CRM
et candidat témoin, seuil, distance minimale des candidats du bon SIREN et
leur précision. Pour audit, conserver aussi les preuves de tous les candidats
du bon SIREN (aucun nom/adresse CRM brute exporté). Le témoin est le premier
candidat concordant par adresse dans l'ordre émis, sinon le plus proche qui
valide la distance, sinon le plus proche disponible (égalité : ordre émis).
La distance minimale peut être celle d'un point commune, ce qui ne valide pas
la règle principale. Distance/point absent : null, jamais zéro.

Parmi les 19 échecs exacts, compter les succès site. Parmi les 507 succès
exacts, comparer le **point du SIRET label exact** au CRM avec le même mapping
et T : compter ceux à distance strictement >T, publier aussi le nombre de
comparaisons admissibles et non mesurables. Aucun point du label n'est utilisé
pour chercher un candidat. Ce compteur est un indicateur, pas une correction
ou preuve qu'un label est erroné. Pas de commentaire individuel de cas.

## Agrégats et ressources

Publier les sept vues en comptes/526, pourcentages et Wilson bilatéral 95 %
(z=1,959963984540054), ainsi que les mêmes statistiques par fold 2/3/4.
Pondération : reproduire l'estimateur Hájek existant
`sum(calibrated_query_weight * hit) / sum(calibrated_query_weight)` des
outcomes, vérifier sa reproduction exacte pour les trois vues existantes.
Ces poids sont ceux du tirage poststratifié du report. Les Wilson portent
uniquement sur les effectifs bruts ; ne pas les attribuer au pondéré, ni
inventer une certification nationale/OOF/99,2 % ou un nouvel IC pondéré.

Script nouveau `scripts/retrieval_site_operational_metric.py`, sources ouvertes
en lecture seule ; scans filtrés/projetés DuckDB/Arrow, 1 thread, plafond
mémoire DuckDB 512 Mio, garde RSS <3,5 Gio (<4 Gio). Gros calcul détaché avec
log si nécessaire. Sorties dédiées JSON (totaux et 526 lignes) et Markdown
court dans docs ; pas de test unitaire ni run/evaluate v1xx. Commit du présent
contrat seul avant exécution ; puis commit isolé script/résultats/handover.
