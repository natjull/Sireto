# V26 — noms officiels et proximité réelle : résultats

10 septembre 2026. Verdict **`PIVOT_GENERATION`**. Les deux audits sont terminés;
aucun nouveau canal, modèle, candidat ou gain de Recall n'est revendiqué.
Préenregistrement/code/tests : commit `f6cb9919`.

## Périmètre et résultat utile

Les 1034 requêtes scellées des folds 2/3/4 comprennent 522 difficultés
historiques et 512 contrôles. Ce panel réutilisé est diagnostique, pas un
échantillon national représentatif ni une validation inter-CRM. Les vérités
servent exclusivement aux jointures d'audit. Aucun fold 0/1 ouvert.

Deux hypothèses vérifiées : les noms utiles disparaissent-ils entre portfolio
officiel et index, et un autre établissement du parent humain se trouve-t-il
près du point CRM malgré une adresse différente ?

- **Noms : aucune perte globale trouvée dans les quatre rôles examinés.**
  Les ensembles locaux matérialisés concordent avec l'index effectivement lu.
- **Géographie : signal réel, mais plafond limité sur les parents manquants.**
  Parmi 132 parents absents des preuves, 12 ont un site à 500 m ou moins,
  aucun à 100 m ou moins. Pour 61 autres, aucune distance n'est exploitable.
  Ces derniers ne sont pas classés comme éloignés.

Les compteurs Master antérieurs v24 sont seulement recopiés : **823 exacts,
825 opérationnels**, dont 311/313 sur les 522 difficultés et 512/512 contrôles.
Pas de nouveau Master2000, PreRank500, Recall100, Wilson ou SLO. V24 reste
non promue; v23 reste la référence de recherche. Labels et politique
d'équivalence inchangés; aucun dégel du ranker/decider/risk/accepteur.

## 1. Noms : portfolio → documents → index

Comparaison de LEGAL_CURRENT, TRADE_CURRENT, SITE_CURRENT et HISTORICAL,
parent et tous ses sites. SUPPORTING exclu : sa provenance officielle ne
constitue pas une autorisation d'utiliser un nom de personne comme feature.

| Mesure par requête | Portfolio | Documents globaux | Documents locaux |
|---|---:|---:|---:|
| Au moins un nom | 1033 | 1033 | 1032 |
| Nom CRM normalisé identique à un nom | 215 | 215 | 214 |
| Au moins un token commun | 658 | 658 | 655 |

Sur **10 018 documents réellement lus dans Tantivy**, égalité des ensembles
normalisés par requête, type de document et rôle avec la matérialisation
locale. Aucun nom du portfolio des quatre rôles n'est absent de tous les
champs globaux examinés. Ce test global ne garantit pas qu'un alias figure
sur chaque document/site, ni qu'il reçoit un score ou un rang suffisant.

La géographie locale retire une correspondance nominale exacte dans un cas
`SITE_NOT_EXPANDED`; trois cas perdent tout token commun (un HIT, une éviction
Master, un site non développé). **Aucun de ces cas parmi les 132 parents
absents.** Chez ces 132 : tous ont des noms globaux et locaux, zéro égalité
nominale exacte avec le CRM, seulement **3** avec un token commun. Un token
peut être générique; il ne prouve pas une identité. Les 31 correspondances
nom-caractères v25 ne contredisent pas ce constat : ce n'est pas un accord de mots.

Les lignes du portfolio pour ces parents portent effectivement les provenances
SIRENE_CURRENT et/ou RNE. Par rôle, les nombres de lignes sont 114 légales,
52 commerciales, 749 de site et 17 historiques (ces dernières RNE). Ce ne
sont ni des nombres d'entreprises distinctes ni des gains possibles.

**Limite importante :** le départ de cet audit est le portfolio officiel déjà
construit, pas tous les fichiers officiels bruts. Il ne prouve pas l'absence
de pertes pendant leur ingestion, leur normalisation ou leur classification.
Le constructeur `src/xgb_matcher/siren_dossier.py` classe notamment BODACC en
SUPPORTING et applique ensuite des budgets par rôle aux documents. Une
nouvelle investigation amont doit comparer les champs bruts autorisés et
leurs preuves au portfolio avant de conclure qu'un nom est absent des sources.
La concordance des champs stockés ne garantit pas non plus une bonne analyse
textuelle, pondération ou position dans les résultats libres du moteur.

## 2. X/Y : tous les sites du bon SIREN

Résolution BAN v16 inchangée, à partir du CRM seulement; puis jointure du
parent humain vers **tous ses établissements du stock officiel**, sans limite
de commune ni plafond de sites. **60 507 paires requête/site** examinées.
Distances euclidiennes Lambert93; coordonnées absentes/hors-emprise laissées
inconnues. Aucun X/Y brut exporté. Le stock est le snapshot local scellé,
pas une affirmation sur la situation actuelle de chaque établissement.

BAN résolue : 866/1034, dont 550 non déclarées faibles et 316 faibles;
168 non résolues. « Non faible » n'est pas une garantie de précision.
Distance vers au moins un site : 734; vers le SIRET exact : 664. Pour 608
requêtes, au moins un établissement du parent n'a pas de coordonnées valides.
Dans **231 cas**, un autre site est strictement plus proche que le SIRET exact
(parmi les coordonnées disponibles). Cela ne rend pas le label erroné.

| Distance maximale | Parent : site le plus proche | SIRET exact | Site actif | Voie normalisée différente |
|---|---:|---:|---:|---:|
| 50 m | 323 | 258 | 276 | 50 |
| 100 m | 372 | 282 | 322 | 84 |
| 250 m | 435 | 344 | 376 | 168 |
| 500 m | 506 | 399 | 435 | 265 |

Colonnes chevauchantes, seuils cumulatifs préenregistrés; ce ne sont pas des
récupérations nouvelles. Le parent figurait déjà dans l'ancien canal géo
pour 457 requêtes. Une voie normalisée différente peut être une autre rue
ou une variation textuelle; ni perpendicularité, ni entrée commune, ni
identité du bâtiment ne sont établies.

### Zoom sur les 132 parents absents des preuves

- BAN résolue : 102, dont 68 non déclarées faibles; 30 non résolues.
- Distance parent disponible : 71; distance SIRET exact : 63.
- À 50/100/250/500 m : **0/0/3/12** parents. Les mêmes comptes valent pour
  le SIRET exact à ces seuils : élargir à tous les sites n'ajoute donc pas
  de cas à ces quatre seuils dans ce segment.
- À 500 m : 9 cas avec BAN non déclarée faible, 9 avec un site actif,
  11 avec une voie normalisée différente; sous-ensembles non additifs.
- Aucun des 132 n'était présent dans les parents voisins v16, y compris
  ces 12 proches. Un autre site est strictement plus proche dans 19 cas,
  sans récupération supplémentaire aux seuils ci-dessus.

L'ancien canal géographique utilisait déjà les X/Y, avec géographie locale
et budgets de voisins. L'audit v26 ne les introduit pas : il examine leur
plafond en levant ces contraintes côté vérité uniquement. Il ne permet pas
encore d'attribuer chaque absence à la commune, au rang spatial, au budget
ou au snapshot de coordonnées.

## 3. Validation et ressources

**71 tests passent**, incluant données manquantes, BAN faible, absence de
relabellisation et rôles exclus. `scripts/audit_retrieval_v26_results.py`
réconcilie en lecture seule les hashes de résultats, les 1034 labels du
ledger bornés aux folds 2/3/4, les ensembles complets de sites contre le
stock, les agrégats de distances et les témoins du minimum. Les résultats
des 16 requêtes de préflight concordent avec le complet; l'identité du
témoin le plus proche est contrôlée par sa distance, sans imposer l'ordre
entre ex aequo. Verdict `PASS_V26_OUTPUT_RECONCILIATION`.

Ce contrôle n'est pas une seconde implémentation du géocodeur BAN ni un
recalcul indépendant de toutes les distances depuis les coordonnées brutes.

Temps du corps d'audit, hors rehash préalable des grandes sources : géo
9,376 s, RSS 628 260 864 octets; noms 22,487 s, RSS 941 408 256 octets.
Zéro swapout, calcul Mac/CATNAT, aucune dépense externe. Indexmeta et
inventaire tailles/mtime stables avant/après; pas de rehash intégral de
tous les octets Tantivy. Les deux premiers préflights noms incomplets
(`preflight16`, `preflight16-asfix`) ne sont pas admissibles; le préflight
réussi est `preflight16-sqlfix`, après corrections SQL avant le gel du code.

Les manifestes publient tous les segments historique/cohorte, V2/V3, folds,
origines, statuts, états et étapes de perte, sans modifier le dénominateur.

## 4. Suite proposée, non exécutée

1. Pour les 12 parents spatialement proches mais absents, attribuer le
   blocage exact du canal géo existant : frontières de commune, rang de
   voisinage, troncature ou différence de référentiel. Mesurer aussi les
   contrôles; ne pas choisir un rayon sur ces douze labels.
2. Auditer **l'amont du portfolio** : présence des champs nominatifs
   autorisés dans les sources locales SIRENE/RNE, extraction, classification
   et normalisation CRM. Ne pas reconstruire un dictionnaire d'alias CRM.
3. Ne préenregistrer un canal naturel supplémentaire que si ces audits
   révèlent une preuve exploitable manquante ou mal accessible. Les proches
   doivent être concurrencés par les entreprises voisines, sans connaître
   le parent humain à la génération. Préserver les labels exacts et publier
   l'opérationnel séparément. Sinon changer d'hypothèse, pas les labels.

## Autorités scellées

Configuration `config/retrieval_v26_evidence_audit.json`, SHA256
`5c845b51f346e54c624140b5b2357b2244a223136f7e7d6f62a9eb9779401737`.

Racine `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/` :

- `retrieval_v26_geo/5c845b51f346e54c/manifest.json`, SHA256
  `0e50b9b49267bf2712368b0d6ccdf0439e47969e7bbe82643ca63abce0b4e3d8`.
- `retrieval_v26_names/5c845b51f346e54c/manifest.json`, SHA256
  `1a6e7ea9522e0e09ca9d1a9012e24abf2ea842a2f84ae3896768e33430b3883e`.

Ces manifestes scellent les sources, l'implémentation et les Parquet détaillés.
Le verdict scientifique reste **`PIVOT_GENERATION`**, pas `READY_FOR_FOLD1`.
