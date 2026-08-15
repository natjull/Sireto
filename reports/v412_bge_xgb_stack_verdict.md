# Verdict V4.12 — BGE fine-tuné et stack XGBoost

Date : 15 août 2026  
Verdict préenregistré : `STOP_RANKER_GATE`  
Fold 1 : fermé  
Test final : fermé  
Google Maps : aucun appel

## Résultat direct

Le fine-tuning de `BAAI/bge-reranker-v2-m3` a produit un signal complémentaire,
mais le stack déterministe préenregistré ne dépasse pas
`BUSINESS_LEARNED` :

| Système | SIRET exacts fold 0 | Hit@1 exact |
|---|---:|---:|
| BGE zéro-shot, référence déjà lue | 2 171 / 2 797 | 77,62 % |
| CamemBERT fine-tuné, référence déjà lue | 2 353 / 2 797 | 84,13 % |
| BGE fine-tuné groupwise | 2 400 / 2 797 | 85,81 % |
| `BUSINESS_LEARNED` | **2 437 / 2 797** | **87,13 %** |
| Stack XGBoost + BGE cross-fitté | 2 436 / 2 797 | 87,09 % |

Le stack corrige 41 erreurs de `BUSINESS_LEARNED`, mais régresse sur 42 cas :

| BUSINESS | Stack | Nombre |
|---|---|---:|
| correct | correct | 2 395 |
| correct | faux | 42 |
| faux | correct | 41 |
| faux | faux | 319 |

Le BGE seul corrige 79 erreurs XGBoost, mais perd 116 réponses que XGBoost
avait justes. La complémentarité existe donc réellement ; le méta-ranker
candidat préenregistré ne sait simplement pas l'exploiter sans perte nette.

## Gate fold 0

| Condition gelée | Résultat | Statut |
|---|---:|---|
| Exact >= 2 452 / 2 797 | 2 436 / 2 797 | échec |
| Difficiles >= 33 / 38 | 32 / 38 | échec |
| Actifs >= 2 164 / 2 391 | 2 180 / 2 391 | passe |
| Fermés >= 246 / 406 | 256 / 406 | passe |
| Toutes les requêtes scorées | 2 797 / 2 797 | passe |
| Plafond stack | 10 candidats | passe |
| Scores BGE cross-fittés | oui | passe |
| Injection du positif | non | passe |

Par rapport à `BUSINESS_LEARNED`, le stack perd 7 actifs (2 187 -> 2 180),
gagne 6 fermés (250 -> 256) et perd un des 38 cas difficiles (33 -> 32).
L'échec n'est donc pas seulement une différence globale d'une ligne : la
protection du segment difficile échoue elle aussi.

La branche CamemBERT conditionnelle n'est pas autorisée par le contrat : elle
n'était ouverte que pour un stack gagnant d'au moins dix réponses et situé
entre 2 447 et 2 451. Le résultat 2 436 ne satisfait aucune de ces conditions.

## Vue opérationnelle secondaire même site

Cette vue est une analyse rétrospective secondaire conforme à
`docs/siret_operational_equivalence_policy.md`. Elle ne modifie, ne réentraîne
et ne rescored aucun résultat primaire. La preuve est volontairement
conservatrice : même SIREN, numéro et suffixe exacts, voie exactement égale
après normalisation déterministe, puis code postal exact ou INSEE exact si un
code postal manque. Les équivalents sont cherchés uniquement dans le top 100
gelé ; la vue peut donc sous-compter les équivalents externes au pool.

| Système | Exact | Opérationnel | Promotions même site | Successeurs actifs |
|---|---:|---:|---:|---:|
| BGE fine-tuné | 2 400 / 2 797 | 2 418 / 2 797 | 18 | 5 |
| `BUSINESS_LEARNED` | **2 437 / 2 797** | **2 454 / 2 797** | 17 | 5 |
| Stack XGBoost + BGE | 2 436 / 2 797 | 2 453 / 2 797 | 17 | 5 |

La nouvelle politique métier ne renverse donc pas le verdict : le stack reste
une réponse derrière `BUSINESS_LEARNED`, aussi bien juridiquement
qu'opérationnellement. Dix cas observés possèdent plusieurs équivalents actifs
dans le pool candidat ; ils sont signalés, jamais départagés silencieusement.

Artefact secondaire :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_bge_operational_secondary/cccd9f1b99877848`.

## Ressources et reproductibilité

- modèle BGE et tokenizer épinglés à la révision
  `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` ;
- quatre fits BGE : folds cibles 0, 2, 3 et 4, chacun hors apprentissage ;
- 31 454 s d'entraînement et 16 976 s de scoring, soit 13,45 h cumulées ;
- pic RSS maximal observé : 5 389 762 560 octets ;
- zéro OOM, zéro GPU loué, zéro service payant ;
- stack XGBoost : 3,72 s de fit, paramètres préenregistrés inchangés ;
- tous les fichiers déclarés par les manifests ont été rehashés sans écart.

Artefacts principaux :

- BGE fold 0 top 100 :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_bge_groupwise/01e1049c16af2600` ;
- BGE OOF fold 2 : `.../v4_12_bge_groupwise/2b424777fbf2f02e` ;
- BGE OOF fold 3 : `.../v4_12_bge_groupwise/9c5091071d727cb6` ;
- BGE OOF fold 4 : `.../v4_12_bge_groupwise/a79c8c3adb3ca3bc` ;
- stack fold 0 :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_bge_xgb_stack/8c1bce0bbf9593b5`.

## Conséquence produit

Le gate ranker était une précondition obligatoire de l'accepteur pré-Maps. Il
n'est pas franchi. Par conséquent :

- le fold 1 n'est pas ouvert ;
- aucun accepteur n'est entraîné sur ce stack ;
- aucun taux AUTO, appel Maps évité ou rematching post-Maps n'est revendiqué ;
- le test final reste fermé.

Le verdict `STOP_RANKER_GATE` concerne cette configuration précise : BGE
groupwise exact mono-positif puis méta-ranker candidat XGBoost top 10. Un futur
cycle opérationnel devra être préenregistré séparément avec
`acceptable_sirets_operational`, une loss multi-positive ou l'exclusion des
siblings même-site des négatifs. Il ne doit pas réutiliser ce fold 0 pour
retuner rétrospectivement le cycle clos.
