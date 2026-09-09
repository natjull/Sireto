# V25 — découverte : audit précis, canal numéro écarté

9 septembre 2026. **`PIVOT_GENERATION`**. Aucun gain de retrieval revendiqué.
Le nouveau canal ne retrouve **aucun parent ni SIRET exact supplémentaire**.
Il n'est pas intégré au Master. V23 demeure la référence de recherche,
non une version produit validée; v24 n'est pas promue non plus.

## 1. Ce que l'audit des absences a établi

Le commit `80c7db77` préenregistre un audit **label-side**, distinct de toute
génération. La correction technique `a77cd307` remplace l'export Arrow non
supporté par DuckDB; le premier essai avait échoué avant diagnostic final.
Le répertoire incomplet `retrieval_v25_discovery_audit/protocol-80c7db77`
n'est pas admissible. Aucune règle de recherche n'a été changée par ce correctif.

Les 1034 requêtes sont auditées, pas seulement les échecs. Parmi les **132
parents absents des preuves de retrieval, hors l'unique absence catalogue** :

| Diagnostic sur l'index local | Requêtes |
|---|---:|
| Parent indexé dans la zone; au moins un canal textuel v18 correspond, mais le parent n'est pas dans les signaux | **88** |
| Parent indexé dans la zone; aucun des dix canaux textuels SIRET v18 ne correspond | **44** |
| Aucun document SIRET indexé du parent | **0** |
| Parent indexé seulement hors de la géographie effective | **0** |

Pour ces 132 requêtes, le SIRET exact lui-même est présent parmi les documents
locaux du parent. Il existe des noms officiels sélectionnés dans le portfolio,
et aucun des parents n'a uniquement des documents locaux sans noms/adresses.
**Existence de noms ne prouve pas que chaque alias utile a été indexé** :
la comparaison fine du contenu source/alias/index reste à faire.

Parmi les 88 atteignables, 82 correspondent au canal adresse-caractères,
50 à adresse-mots, 31 à nom-courant-caractères, 22 à adresse de support,
14 à adresse historique, 3 au BM25 nominal fieldé, 2 au nom de site,
1 au nom de support et 1 au nom historique. Ces comptes **se recouvrent**.
Les rangs libres n'ont pas été mesurés dans cet audit : « atteignable » est
une correspondance obtenue avec le parent humain comme filtre d'audit, pas
un candidat découvert naturellement. La profondeur/troncature reste une
hypothèse à examiner, non la promesse de 88 rescues.

| Population | Atteignables mais absents | Aucun match textuel v18 | Autres diagnostics |
|---|---:|---:|---|
| Fold2, 353 lignes | 24 | 14 | 315 déjà retrouvés |
| Fold3, 346 lignes | 29 | 21 | 295 retrouvés, 1 hors géographie |
| Fold4, 335 lignes | 35 | 21 | 278 retrouvés, 1 absent officiel |
| CRM récent, 672 lignes | 55 | 48 | 568 retrouvés, 1 absent officiel |
| CRM historique, 362 lignes | 33 | 8 | 320 retrouvés, 1 hors géographie |
| Contrôles, 512 lignes | 0 | 2 | 510 déjà retrouvés |

La colonne « aucun match textuel » contient aussi des parents retrouvés par
d'autres canaux : elle totalise 56 sur tout le panel, dont 44 parmi les 132
absents. L'audit utilise les dix définitions SIRET v18 reconstituées et scellées
sur l'index local; il ne reconstitue pas tous les rangs historiques v18.

Contrôle supplémentaire de la projection SceneIR : sur les 1034 requêtes,
les champs géographiques priorisés par l'adaptateur historique et les champs
CRM donnent les mêmes INSEE et CP; aucune variation intra-requête de ces
champs n'a été observée. Pas de correction de géographie justifiée ici.

## 2. Hypothèse testée, sans nouvel ajustement

Le commit **`2318898f`** fige la recette additive suivante, après l'audit et
avant le run complet : numéro exact **ET** adresse-caractères, géographie
inchangée, maximum512 SIRET. Le générateur ne lit ni labels, ni sidecar de
diagnostic. Numéro extrait par l'adaptateur CRM historique, sans remplissage
depuis la vérité. Aucun nouveau modèle, alias, source, géocodage ou quota.

935 requêtes possèdent numéro et adresse dans cette représentation; 99 ne
sont pas éligibles au canal et restent au dénominateur. 857 ont des résultats.
Le canal produit 258 113 références SIRET, dont **40 431 paires requête/SIRET
nouvelles** par rapport au réservoir. Elles ne contiennent cependant aucune
vérité exacte ni aucun parent humain nouveau. Le nombre de candidats nouveaux
n'est donc pas une mesure de progrès.

| Présence naturelle, tous les 1034 labels | Réservoir v21 | Canal seul | Union diagnostique |
|---|---:|---:|---:|
| Bon parent SIREN | 900 | 360 | **900** |
| SIRET humain exact | 898 | 280 | **898** |
| Exact ou équivalent opérationnel | 898 | 324 | **898** |

**0 gain /0 perte** en couverture de l'union. La monotonicité est garantie par
l'ajout, pas par un modèle; elle ne constitue pas une amélioration de la sortie
limitée. L'union contient 3 940 575 références exploratoires : **ce n'est pas
un Master@2000**, aucun modèle SIRET ne la score, aucun top500/top100 n'est émis.

| Segment | N | Parent ancien = union | Exact ancien = union | Parent canal | Exact canal | Opérationnel canal |
|---|---:|---:|---:|---:|---:|---:|
| Échecs initiaux | 522 | 388 | 386 | 54 | 20 | 32 |
| Contrôles | 512 | 512 | 512 | 306 | 260 | 292 |
| Fold2 | 353 | 319 | 318 | 120 | 92 | 111 |
| Fold3 | 346 | 300 | 299 | 128 | 100 | 114 |
| Fold4 | 335 | 281 | 281 | 112 | 88 | 99 |
| CRM récent | 672 | 578 | 578 | 201 | 145 | 177 |
| CRM historique | 362 | 322 | 320 | 159 | 135 | 147 |
| MATCH_EXACT | 874 | 780 | 778 | 326 | 268 | 300 |
| AMBIGUOUS | 7 | 7 | 7 | 6 | 4 | 6 |
| UNRESOLVED | 153 | 113 | 113 | 28 | 8 | 18 |

Les vues historique all-human, V2, V3 et états administratifs sont scellées
ensemble. V2/V3 sont vides pour toutes les lignes; aucune qualification n'est
inventée ni exclue. La vue opérationnelle garde le sidecar same-site précédent,
ses limites sur les compléments d'adresse et la distinction présence/top1.
Le canal seul n'est pas une nouvelle sortie produit autonome.

## 3. Pourquoi la contrainte numéro ne résout pas ce blocage

Recalcul indépendant dans les documents officiels effectivement utilisés pour
construire l'index, sur **tous les sites du parent**, pas seulement le label :

| Parmi les 132 parents absents | N | Numéro+adresse disponibles | Un site local du parent au numéro demandé | SIRET exact au numéro demandé |
|---|---:|---:|---:|---:|
| Texte atteignable mais parent non retrouvé | 88 | 79 | **2** | **1** |
| Aucun match textuel v18 | 44 | 40 | **0** | **0** |

Ainsi, parmi les 119 requêtes éligibles, seuls deux parents possèdent un site
local au numéro demandé. Les deux conditions restent nécessaires, pas
suffisantes : ces deux parents n'ont pas été récupérés par le canal complet.
Un filtre numéro obligatoire ne pouvait avoir qu'un plafond très faible ici.

Sur l'ensemble des 522 difficiles, 477 sont éligibles, 67 ont un site local
du parent au numéro demandé et 24 ont le SIRET exact à ce numéro. Pour les
512 contrôles : 458 éligibles, 310 parents avec accord local et 260 accords
exacts. Le désaccord est donc particulièrement marqué dans le segment difficile.
Cela **ne prouve pas que les labels sont faux** : les coordonnées CRM, l'adresse
du site commercial, l'adresse légale, les périodes et les représentations
peuvent diverger. Aucune vérité n'est corrigée, retirée ou neutralisée.

Décision : écarter ce canal pour la découverte. Garder numéro/géographie comme
preuves potentiellement utiles mais ne pas durcir leur accord pour retrouver
ces absents. Pas d'élargissement post-hoc du canal à d'autres seuils ou variantes.
Il aurait été préférable de calculer ce plafond de compatibilité du numéro
avant le pilote; ce contrôle préalable est à reprendre pour les prochaines
hypothèses, sans le transformer en oracle de génération.

## 4. Intégrité et attribution

57 tests passent sur les suites v19 fusion et v20–v25. Tests réels sur petit
index Tantivy : numéro obligatoire, garde géographique, clés officielles non
dérivées d'un préfixe, rang dense ne dépendant pas d'identifiants, cap512,
valeurs manquantes et refus de changement des paramètres figés.

Préflight de16 requêtes sans labels, puis égalité complète de ses requêtes et
résultats avec le run complet. Audit indépendant : tous les SIRET/SIREN émis
existent dans les documents sources scellés et respectent numéro/géographie;
recalcul des présences depuis le ledger, de l'opérationnel depuis le sidecar,
de l'union naturelle et de son attribution. Aucun positif injecté.

Attribution **du réservoir**, pas du pipeline final : 898 exacts présents,
133 parents absents du réservoir, 2 sites absents malgré parent dans le
réservoir, 1 absent du catalogue. La différence avec l'attribution officielle
amont **132 parents absents des preuves /3 sites non développés /1 catalogue**
vient d'un parent présent dans les signaux mais sans site matérialisé dans le
réservoir. Ce changement de niveau d'observation n'est pas un gain ni une
rétractation de l'attribution pipeline. Les pertes de préclassement/admission
ne sont pas mesurées ici.

| Phase | Calcul chronométré | Pic RSS | Swapout |
|---|---:|---:|---:|
| Audit source/index | 188,291 s | 8,62 Gio | 0 |
| Préflight, 16 requêtes | 8,238 s | 1,27 Gio | 0 |
| Canal complet | 121,132 s | 7,59 Gio | 0 |
| Évaluation | 0,895 s | 1,15 Gio | 0 |
| Audit indépendant de résultats | 1,448 s | 1,56 Gio | 0 |

Hashes préalables exclus des durées; pas de SLO produit déduit. Mac/CATNAT,
aucun coût externe, aucun nettoyage utilisateur. Manifestes, sources et résultats
vérifiés par SHA. meta.json Tantivy et inventaire tailles/mtime restent identiques
avant/après; **les octets de l'index national complet n'ont pas été rehashés**.

## 5. Suite et scellement

Prochaine priorité, non exécutée : comparer le **contenu** des noms officiels
du bon parent et de tous ses établissements à la représentation indexée,
avec contrôles et séparation current/historical. Chercher une perte de preuve
due à la sélection des alias, ou constater qu'aucune preuve nominale exploitable
ne relie le CRM au parent. Ne pas réannoncer les noms historiques comme une
nouveauté : v18 les utilise déjà, et les documents SIREN agrègent déjà des noms
de sites sélectionnés globalement. Mesurer le plafond informationnel avant de
construire un nouveau canal ou un modèle. Aucun alias appris du CRM.

Configuration SHA : `2c11d2553743ea1e43f7c71b6a56f242c6018626e3818096fcf3391ab163ddf6`.
Racine : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/`.

- `retrieval_v25_discovery_audit/arrowfix/manifest.json` :
  `e6b7b842b6e77cffa90e7e7f2b2b0190c119158424167405e181c046bf2adb83`.
- `retrieval_v25_conjunctive_preflight/fixed512/manifest.json` :
  `5024bbb2f538f2f4b1f57d41e0a52b25a6c12a853af8264520c3c02f59bb8c80`.
- `retrieval_v25_conjunctive_generation/2c11d2553743ea1e/manifest.json` :
  `a9c1fd746977ed3294761660aa8d29c8058f5479c90af1ee171cd9e49517597a`.
- `retrieval_v25_conjunctive_evaluation/2c11d2553743ea1e/manifest.json` :
  `d1421100dbac579c4f2e76de297c033b22cf5cf8cadd5ad059614268b47b78d6`.
- `retrieval_v25_result_audit/2c11d2553743ea1e/manifest.json` :
  `5b460fdfe6b0e6c39b8111dde7cc6469b2308fd6f3ff2d8308619cb87fa08baf`.

Les scripts de génération/évaluation et d'audit sont versionnés séparément.
Le premier audit prend v24 comme état diagnostique (823 exacts/825 opérationnels
au Master); l'évaluation du canal compare le réservoir v21 avec v23 comme
autorité des labels. Ni v23 ni v24 n'ont été recalculées ou modifiées.

Pas de national, de projection nationale, de gate Master/500/100/Wilson/SLO,
de dégel aval ou d'ouverture fold0/fold1. Le panel reste biaisé et réutilisé;
aucune preuve cross-CRM. Verdict **`PIVOT_GENERATION`**, pas de promesse 99,2 %.
