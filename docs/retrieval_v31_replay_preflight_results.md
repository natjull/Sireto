# V31 — préflight du rejeu complet : STOP sur la reproductibilité

11 septembre 2026. Autorisation utilisateur « Vas-y » pour le contrôle puis
le rejeu v23/v30 si ce contrôle passe. **Le contrôle ne passe pas : aucun
rejeu complet ni nouveau recall n'est publié.** Les recettes v23/v30 restent
inchangées. Aucun entraînement ni modèle aval activé.

## Population complète et entrées : contrôles passés

Le ledger scellé et sa projection d'entrée donnent exactement :

| Fold | Labels humains exploitables | Absences de label publiées |
|---|---:|---:|
| 2 | 7 767 | 8 |
| 3 | 7 967 | 3 |
| 4 | 7 837 | 5 |
| Total | 23 571 | 16 |

Aucune exclusion par qualification, présence catalogue ou résultat de
retrieval. Les 16 absences sont dans le sidecar officiel des exclusions.
Les 23 571 requêtes reconstruites depuis les champs d'entrée concordent avec
la projection de recherche utilisée historiquement. Aucun SIREN humain ne
chevauche les folds 2/3/4, et aucune entreprise d'entraînement du modèle
outer concerné n'est dans son nouveau périmètre d'évaluation. Les modèles
avaient respectivement 581, 600 et 619 requêtes effectivement entraînées.
Ce contrôle d'isolation reste label-side, jamais une feature de recherche.

Folds 0 et 1 non matérialisés. Projection des colonnes autorisées et prédicat
de folds au niveau du scanner Parquet. Le contrôle ne démontre ni une
validation indépendante ni la performance sur de futurs CRM.

## Premier écart corrigé dans le runner de contrôle

La composition d'index reprise de v15 n'était pas celle des sorties legacy
v11. La provenance des candidats historiques mentionne un second index
sectoriel et l'overlay officiel principal contient BODACC. La restauration
du branchement officiel complet comprend l'overlay courant v2, le complément
temporel RNE/BODACC et Data ES, aux côtés des autres sources déjà présentes.

Sans cette restauration : 0/16 ordres de candidats reproduits. Après :
**16/16 identiques, SIRET, SIREN et rang**, depuis les champs CRM uniquement.
Il s'agit du rétablissement du branchement legacy, pas d'un canal nouveau
ajouté à v23/v30 pour gagner du recall. Les index ne sont pas reconstruits.

## Deuxième écart : témoins de succession non reproductibles

Les mêmes 16 entrées sont ensuite rejouées jusqu'aux deux Masters.
La recherche parent native, l'union des parents, les dix canaux SIRET directs
et le canal conjoint nom/adresse concordent avec les artefacts scellés.

La différence est dans `sirene_economic_continuity` : **231 témoins SIRET
différents sur les 16 requêtes**, à SIREN, rang de canal et rang de seed
inchangés. Parmi eux, 222 changent aussi de direction prédécesseur/successeur.

Dans `_write_structured_signals`, les agrégations de succession utilisent
`arg_min(..., seed_rank)` sans départage des témoins à seed égal, notamment
après union des deux directions. Le diagnostic retrouve, pour les 231
différences, plusieurs SIRET admissibles à ce même rang minimal. Les anciens
et nouveaux témoins sont tous appuyés par les relations officielles : aucun
n'est un positif inventé. Mais le choix entre preuves égales n'est pas gelé.

Les **100 fichiers de successions sont vérifiés par leur hash**, et leur
ensemble correspond exactement au manifeste, sans fichier supplémentaire
dans le glob. Ce n'est donc pas une nouvelle source apparue dans ce répertoire.

Deux processus d'audit reproduisent chacun les appels avec 1, 2 et 8 threads.
Dans un processus les trois résultats concordent entre eux; dans l'autre
aussi, mais **les deux sorties à un thread diffèrent de 216 témoins**, sur
les mêmes 16 requêtes et les mêmes sources. Fixer seulement le nombre de
threads n'est donc pas une réparation suffisante. La comparaison interprocessus
est nécessaire : la deuxième exécution seule publie prudemment
`STOP_PARITY_CAUSE_NOT_FULLY_ESTABLISHED`, parce qu'elle reproduit le premier
préflight. Comparée au premier audit, la dépendance au contexte d'exécution
est néanmoins constatée. Aucun de ces rapports intermédiaires n'est masqué.

### Conséquence mesurée sur les candidats, pas sur les labels

| Master, 16 requêtes | Paires requête/SIRET retirées | Ajoutées | Rangs modifiés parmi les paires communes |
|---|---:|---:|---:|
| V23 reconstruit vs fichier v23 gelé | 209 | 208 | 18 365 |
| V30 reconstruit vs fichier v30 gelé | 212 | 211 | 18 612 |

Ces nombres ne sont **pas des pertes/gains de vrais positifs** : aucune
vérité humaine n'est lue pendant ces replays. La différence de témoin change
le soutien direct et le réservoir, puis le classement sous le plafond2000.
Les anciens 822 et829 hits demeurent les mesures de leurs fichiers gelés;
ils ne sont pas une garantie de reproduction de bout en bout.

## Tests, coûts et traces

Code population/rejeu initial : `2af594b9`. Branchement restauré, contrôle
des canaux/Masters, diagnostic de succession et tests : `a036fde5`.
**19 tests passent** (préflight v31 et canal conjoint v30), dont projection
sans labels/folds interdits, indépendance aux identifiants/champs GT,
conservation des requêtes vides et comparaison sensible aux rangs.

Population : 1,93 s; ancien branchement 16 : 75,29 s; branchement restauré
16 : 40,08 s; autres canaux et deux Masters : 35,80 s; audits 7,33 et6,86 s.
Pics RSS inférieurs à4,5 Gio, aucun swapout, gardes14 Gio/30min respectées.
Ces temps de préflight ne sont ni un SLO produit ni une extrapolation fiable
du temps complet. Le dimensionnement complet reste à faire après stabilisation.

Racine des artefacts :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v31_replay_preflight/`.

- `population/manifest.json`, SHA
  `f2232c960a67334dd96b518b6a0e0abbe27915773915f3532aae7d195a3e316d`.
- `legacy16_restored_sources/manifest.json`, SHA
  `63ce3a61de0c21bd44f768f48610672633b8e32121da1aee4a824f627ffdd78c`.
- `channels16/manifest.json`, SHA
  `9e29f79b10fb9a353b0fccc5ca1168f33e89a829e0ff12ab3d7f08a660bfb4ee`.
- `succession_audit/manifest.json`, SHA
  `34ea8ecc3f8566449d016497a49631e4a206256358ba3067e02fc0a43b8412ef`.
- `succession_audit_confirmed/manifest.json`, SHA
  `73035ebb3fad09cd82034579583e5932c2aa0d2ad6bee645a16fac8db1da2ed6`.
- Sorties comparées à un thread : SHA
  `f2b2b4df3aa694c83c28fd2c7e439425e04e8eea727e7a83dcd13cf18a7e15b1`
  et `5af7d3f1f8f599a9de4af7a9b91b08ab6df89b627f9eec33aacb0963bb2ede2c`.

## Décision et suite nécessitant un nouveau gel

**STOP : parité d'exécution non acquise.** Ne pas lancer les23 571 requêtes,
ne pas simuler la fidélité en recopiant les témoins précalculés par query_id.

Réparation recommandée : définir un départage total des preuves de succession
à seed égal, sélectionner une même ligne de preuve pour le SIRET et la
direction, puis vérifier parité interprocessus, ordre, threads et shards.
Cette règle doit être commune aux deux branches et indépendante des labels;
aucune protection de cas ni pondération apprise. Elle modifie potentiellement
les sorties gelées : les nouvelles branches devront être nommées et scellées
comme des versions stabilisées, pas annoncées identiques à v23/v30 historiques.
Cette modification de recette n'est pas exécutée dans le présent contrôle.

Après ce nouveau gel seulement : dimensionnement puis rejeu apparié complet,
exact/opérationnel séparés, tous labels conservés. Pas de national mesuré,
de500/100, de garantie99,2% ou de READY_FOR_FOLD1. Aucun calcul en cours.
