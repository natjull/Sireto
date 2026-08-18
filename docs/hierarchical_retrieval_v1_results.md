# Retrieval hiérarchique sparse — résultats dev

Date : 18 août 2026  
Décision : **PIVOT**  
Fold test : **fermé**

## Artefact national

- Index : `hierarchical_retrieval_v1/096dd81d4102bdcd`
- 43 896 818 documents SIRET et 39 167 442 documents géographiques SIREN
- 83 064 260 documents au total, environ 98 Gio
- SIRENE courant, historiques établissement/unité légale et successions
  officielles présents ; `temporal_complete=true`
- Aucun label ni alias CRM dans l'index ; `contains_crm_labels=false`
- Plafond de sortie : 100 candidats, sans injection du positif
- Runtime évalué : commit `0638c3b`

## Résultats

### Population commerciale humaine prospective

Les 3 510 relations CRM→SIRET sont conservées comme labels humains, conformément
à la décision métier. Une erreur de label éventuelle compte donc comme une erreur
end-to-end du retrieval exact ; elle n'est pas corrigée à partir des scores du
moteur.

| Mesure | Baseline multicanal | Hiérarchique V1 |
|---|---:|---:|
| Requêtes | 3 510 | 3 510 |
| Recall@100 SIRET exact | 3 279 (93,419 %) | **3 301 (94,046 %)** |
| Recall@100 opérationnel publié | 3 279 (93,419 %) | 3 301 (94,046 %) |
| Candidats maximum | 100 | 100 |
| Latence p50 | 150 ms | **166 ms** |
| Latence p95 | 1 410 ms | **635 ms** |
| Latence p99 | 5 197 ms | **1 020 ms** |

La vue opérationnelle de ce benchmark est encore presque entièrement singleton ;
elle ne matérialise donc pas correctement tous les SIRET alternatifs du même
SIREN et du même site physique. Elle est publiée séparément mais ne doit pas être
interprétée comme une mesure complète de l'équivalence opérationnelle.

### Sous-population prospective certifiée identité + site

| Mesure | Résultat |
|---|---:|
| Requêtes | 751 |
| Recall@100 SIRET exact | **748 (99,601 %)** |
| Recall@100 opérationnel publié | 748 (99,601 %) |
| Candidats maximum | 100 |
| Latence p50 / p95 / p99 | 220 / 499 / 1 092 ms |

Cette mesure montre que le retrieval franchit 99 % lorsque les champs CRM et le
SIRET sont directement soutenus par l'identité et le site officiels. Elle ne
remplace pas la mesure sur tous les labels humains et ne justifie pas l'ouverture
du test.

### Références historique / V2 / V3 déjà gelées

Ces références restent inchangées et sont rappelées ensemble comme l'exige le
contrat : historique 2 495/2 565 (97,271 %), V2 exact 2 343/2 400
(97,625 %), V3 exact identifiable 2 095/2 104 (99,572 %) avec couverture V3
2 104/2 565 (82,027 %).

## Diagnostic du delta

- La V1 récupère 121 des 231 échecs du top-100 historique, mais perd 99 anciens
  succès : le gain net de 22 masque un défaut important de fusion/admission.
- L'oracle profond historique contient 169/231 de ces vérités. L'union interne
  de la V1 en contient 139/231, dont 25 absentes de l'oracle historique.
- L'union parfaite des deux générateurs contient 194/231 échecs, soit une borne
  globale de 3 473/3 510 = **98,946 %** : encore deux vérités sous le minimum
  de 3 475 nécessaire au gate de 99 %, avant même la sélection à 100.
- Étendre les SIREN lexicaux de top-5 à top-20 ne sauve aucun des 37 cas absents
  des deux générateurs.

Un diagnostic dense générique, exécuté uniquement sur ces 37 cas avec l'index
MiniLM global V9 déjà existant, retrouve le bon SIREN dans 5 cas. Pour chacun,
le store officiel contient bien le SIRET exact dans les 2 à 30 sites déroulés.
L'oracle de génération combiné atteint donc au moins 3 478/3 510 = **99,088 %**.
La fusion RRF V9 historique ne conserve cependant aucun de ces cinq SIRET dans
son top-100 et sa latence p95 atteint 9,35 s sur ce sous-ensemble volontairement
difficile. Cela valide le dense comme rescue conditionnel, pas comme fusion
symétrique ni comme chemin systématique.

Le problème restant n'est donc ni un alias à mémoriser ni un simple changement
de quota. Il faut :

1. conserver l'union sparse historique + historique officiel hiérarchique ;
2. exécuter le dense global uniquement sur les scènes sparse incertaines et
   dérouler de façon bornée les sites locaux de ses quelques SIREN de tête ;
3. apprendre ou calibrer sur train/dev une fusion de retrieval qui sélectionne
   100 candidats dans cette union sans évincer les succès sparse ;
4. republier exact et opérationnel, puis ouvrir le test une seule fois seulement
   si couverture et Recall@100 exact franchissent leurs gates.

## Artefacts scellés

- Commercial :
  `evaluations/hierarchical_retrieval_v1/commercial_dev_3510_result.json`, seal
  `be798569f022135e64e0badcc7c0be9578c4bb18a58d048cb62956908706226a`
- Certifié :
  `evaluations/hierarchical_retrieval_v1/certified_dev_751_result.json`, seal
  `90a61d4f264a26ff876cb330a8b7579e556ac9c0cd35551630efee12e0f6d5b1`
- Diagnostic des 231 échecs historiques :
  `evaluations/hierarchical_retrieval_v1/diagnostics/legacy_misses_231_internal_oracle.json`
- Diagnostic dense des 37 invisibles :
  `evaluations/hierarchical_retrieval_v1/diagnostics/dense_rescue_37/global_siren_hybrid100`
