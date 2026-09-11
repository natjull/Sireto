# V32 — réparation de succession validée, rejeu complet restant

11 septembre 2026. Autorisation utilisateur explicite de stabiliser la recette.
Implémentation et contrat préenregistrés dans `54156809`, avant les trois
exécutions comparées. **PASS_STABLE_WITNESSES_AND_MASTERS**, limité au
contrôle de reproductibilité des 16 entrées v31. Aucun nouveau recall.

## Réparation commune aux deux branches

Conserver le meilleur rang de seed. À rang égal, choisir la relation officielle
dont la date effective est la plus récente, dates inconnues après les dates
connues. Sélectionner une ligne entière : SIRET lié, seed, direction, date et
relation restent cohérents. Les identifiants ne servent qu'aux jointures et
au départage terminal, jamais de features de pertinence.

Pas de préférence systématique pour un successeur ou un établissement actif :
une vérité humaine exacte peut être historique et fermée. Ni nouvelle feature
géographique, ni réglage sur les labels. Le classement des parents conserve
sa date agrégée historique, distincte de la date de la preuve choisie.

Les deux sorties sont nommées `v23_stable` et `v30_stable`. Les anciennes
recettes et leurs fichiers restent conservés; la réparation peut changer leur
membership SIRET et ne doit pas être annoncée identique aux résultats historiques.
Les modèles parent, autres canaux, poids et plafonds restent inchangés.

## Vérifications effectuées

- 28 tests passent : neuf cas v32 et dix-neuf contrôles v31/v30. Cohérence de
  ligne, priorité seed/date, dates manquantes, doublons, directions, caps,
  permutations, threads, shards et rejet de fold interdit couverts.
- Les 100 fichiers officiels de succession sont vérifiés par hash. Aucun
  nouveau fichier ni aucune source ajoutée pour gagner du recall.
- Référence Python indépendante égale au SQL sur les **1 494 preuves** des
  16 requêtes réelles. Même ensemble de parents, rang de seed et rang de canal
  que la sortie historique contrôlée.
- Trois processus distincts font varier simultanément nombre de threads,
  taille des lots, ordre des requêtes, ordre des relations et hash seed Python.
- Comparaison exhaustive bidirectionnelle de toutes les colonnes, doublons
  compris : **zéro différence** pour preuves, signaux et chaque Master,
  scores et rangs inclus. Les quatre fichiers ont aussi des SHA identiques.

| Exécution | Threads | Lot | Ordre | Hash seed | Temps | Pic RSS processus |
|---|---:|---:|---|---:|---:|---:|
| A | 1 | 16 | croissant | 13 | 31,72 s | 0,63 Gio |
| B | 8 | 1 | inversé | 71 | 63,72 s | 1,07 Gio |
| C | 2 | 4 | inversé | 109 | 61,91 s | 0,63 Gio |

Préparation : 18,81 s, RSS 0,60 Gio. Comparaison finale : 0,19 s, RSS
0,43 Gio. Zéro swapout pour chaque phase. Les exécutions se sont partiellement
chevauchées : ces temps ne constituent ni un benchmark de parallélisme ni
un SLO produit. Le budget vérifie le RSS du processus, pas la RAM machine.

## Limites et étape suivante

Ce test isole la réparation à partir des canaux v31 déjà reconstruits sans
labels. Il **ne régénère pas toute la chaîne** dans chaque processus. Il ne
mesure ni un gain de recall, ni la stabilité sur toutes les entrées possibles,
ni la performance de production. Aucune vérité humaine ouverte dans v32.

Le défaut identifié est levé sur le contrôle prescrit. Le runner apparié
complet doit encore être intégré : entrées CRM originales vers tous les
canaux, réparation commune, modèles outer gelés et deux Masters naturels.
La garde historique de prédiction limitée au panel ne doit pas être simplement
supprimée : une autorité d'inférence explicite doit reprendre l'isolation OOF
v31 pour les 23 571 entrées, sans faire entrer les labels dans la génération.

Avant un long run : mesurer sur ce chemin intégré le coût de lots bornés,
partager les calculs communs sans cache de candidats par query_id, sceller
les sources/code et assurer la reprise sans omission ni doublon. Les contrôles
de 16 entrées ne donnent pas une estimation fiable du coût complet.

L'évaluation doit rester séparée, après scellement des candidats, avec un
sidecar officiel label-side couvrant toute la population et les vues exacte
et opérationnelle distinctes. Aucun recyclage du sidecar limité aux 1 034
requêtes pour prétendre couvrir tout le CRM. Absences catalogue comptées
comme erreurs, dénominateur all-human inchangé.

**Pas de READY_FOR_FOLD1 : STOP pour toute conclusion de gate complet tant
que le rejeu et son audit exhaustif restent absents.** Aucun fit, admission
500/100 ou modèle aval lancé; folds 0/1 fermés. Aucun calcul restant en cours
à la clôture de ce contrôle.

## Traces scellées

Racine :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v32_stability/`.

- `prepared/manifest.json` :
  `d324d6ccf76cf21b5544362ceb0af75ac3f481fcef69399958fb90e7dbe31b58`.
- `verification/manifest.json` :
  `d5c25efeb10a92930683b96b3110a6527a2f08d2994165da6f2f20641ac76422`.
- `proof.parquet`, identique A/B/C :
  `6f1c15fe7616b18f180cf0f3f02d9d466e15a3c0f67e2482db9651f1b50e6144`.
- `signals.parquet`, identique A/B/C :
  `7bbd6ae314543426da8f663ef3de59944c7c6b4d3ecb1d5dcbb3c7338f203618`.
- `v23_stable.parquet`, identique A/B/C :
  `51ba8b7b7314954fcd52cfc50c47558621f2faefc5142e5ee2d4997913a1402c`.
- `v30_stable.parquet`, identique A/B/C :
  `47197d4b6b1b63729347dd43b3c54729c09f0a0a5efb31d728ffdc5a87b5ccad`.
