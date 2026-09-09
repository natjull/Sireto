# V23 — noms officiels, preuve de site et parent appris OOF

9 septembre 2026. Recette figée dans `7ba9b699` avant extraction complète,
entraînements et résultat externe. Verdict : **`PIVOT_GENERATION`**.

## Résultat : +49 exacts nets, avec 29 régressions à publier

| Mesure, même panel de 1034 dossiers | V22 | V23 |
|---|---:|---:|
| Exacts difficiles conservés au Master2000 | 262 / 522 | 310 / 522 |
| Contrôles exacts conservés | 511 / 512 | 512 / 512 |
| Total exact | 773 / 1034 | 822 / 1034 |
| Total opérationnel | 773 / 1034 | 824 / 1034 |

**78 gains exacts et 29 pertes, soit +49 nets.** Parmi les difficiles :
77 nouveaux hits et 29 anciens hits perdus, soit +48. Le contrôle perdu en
v22 est récupéré. La conservation de tous les contrôles ne masque donc pas
les régressions sur d'autres requêtes. Aucun statut ne réduit le dénominateur.

Le taux exact 79,4971 % est celui du panel enrichi en erreurs, pas du national.
Le préfixe n'est toujours pas protégé : aucune projection nationale et aucun
gate ne sont déduits de ces chiffres, même avec 512/512 contrôles.

| Segment | Requêtes | Exact v23 | Opérationnel v23 | Gains / pertes exacts |
|---|---:|---:|---:|---:|
| Fold 2 | 353 | 296 | 296 | 44 / 11 |
| Fold 3 | 346 | 268 | 268 | 20 / 10 |
| Fold 4 | 335 | 258 | 260 | 14 / 8 |
| CRM récent | 672 | 520 | 521 | 55 / 21 |
| CRM historique | 362 | 302 | 303 | 23 / 8 |
| MATCH_EXACT | 874 | 719 | 720 | 55 / 21 |
| AMBIGUOUS | 7 | 7 | 7 | 0 / 0 |
| UNRESOLVED | 153 | 96 | 97 | 23 / 8 |
| V2 non renseignée | 1034 | 822 | 824 | 78 / 29 |
| V3 non renseignée | 1034 | 822 | 824 | 78 / 29 |

Les vues historique all-human et états sont aussi publiées dans le manifeste.
L'opérationnel ajoute deux hits; 25 requêtes ont un successeur actif acceptable
de vérité officiellement fermée dans le Master, et 22 ont plusieurs actifs
acceptables conservés. Ce sont des mesures de présence, pas de précision top1.
L'ensemble acceptable est celui scellé en v20/v21; ses limites, notamment les
compléments d'adresse non mesurés, restent inchangées.

## Ce qui a été construit

Extraction indépendante des labels : noms légaux/commerciaux actuels officiels
attachés au SIREN, noms propres actuels attachés au SIRET; comparaison au nom
CRM par Jaro-Winkler et Jaccard de tokens. La preuve conjointe est calculée
sur chaque site avec sa voie et son numéro/suffixe, puis agrégée au parent.
Le nom d'un établissement ne peut pas être combiné avec l'adresse d'un sibling
pour créer une preuve inexistante. Le nom légal du parent est, lui, applicable
à ses établissements. Les adresses manquantes restent des manques explicites.

Les sources, 2 960 092 noms officiels pertinents et 16 features numériques
produisent 2 558 996 couples requête/parent et 3 900 144 références site. Ce
dernier ensemble est exactement le réservoir antérieur : aucun positif ni
candidat nouveau ajouté à l'aide de la vérité.

Six LambdaMART **parent**, paramètres fixes et négatifs représentatifs :

| Modèle | Folds d'apprentissage | Folds prédits | Groupes positifs naturels entraînables | Lignes de fit |
|---|---|---|---:|---:|
| outer2 | 3, 4 | 2 | 581 | 296 577 |
| outer3 | 2, 4 | 3 | 600 | 306 186 |
| outer4 | 2, 3 | 4 | 619 | 315 427 |
| inner_train2 | 2 | 3, 4 | 319 | 162 518 |
| inner_train3 | 3 | 2, 4 | 300 | 152 909 |
| inner_train4 | 4 | 2, 3 | 281 | 143 668 |

Les trois modèles externes fournissent seuls les scores parent utilisés pour
le Master. Les trois autres produisent les prédictions inner-OOF réutilisables
pour les folds d'apprentissage de chaque contexte externe; aucune comparaison
de leurs performances n'a sélectionné un modèle ou un seuil.

Les groupes sans positif naturel ou sans négatif ne sont pas entraînables
en pairwise (100/88/80 groupes pour les trois fits externes, respectivement).
Ils restent intégralement dans l'évaluation. Aucun positif n'est injecté.
Séparation vérifiée des requêtes et de leurs SIREN humains entre train/cible.
Cela ne prétend pas que les ensembles de tous les candidats négatifs sont
disjoints : une entreprise peut apparaître comme candidate d'un autre dossier.
Les identifiants et noms bruts ne sont pas des features du modèle.

L'admission utilise le rang parent appris hors fold et le rang site par preuve
conjointe, dans la forme de score gelée
`direct_rrf + 1/(60 + rang_parent + rang_site)`, puis le top2000 global.
**Aucun LambdaMART SIRET n'a scoré le grand réservoir** et aucun modèle aval
n'a été dégelé. La sélection conserve 1 973 172 références, cap absolu 2000.

Cette recette change ensemble les preuves nominales, l'apprentissage parent
et le rang site. Le +49 mesure ce paquet complet; il ne permet pas d'attribuer
causalement tout le gain au seul nom, au seul modèle ou au seul ordre site.

## Attribution et diagnostic secondaire, sans retuning

Les 212 pertes exactes sont exhaustivement attribuées par le sidecar officiel :
**132 parents absents, 3 sites non développés, 76 évictions vers 2000 et un
SIRET absent du catalogue**. Le réservoir reste le même : changer le classement
ne résout pas les 132 absences de parent.

Le bon parent est présent dans les features pour 900/1034 requêtes. En OOF,
il est dans le top10 pour 587, top100 pour 746, top500 pour 845 et top2000
pour 897. Ce sont des rangs **parent**, pas des Recall@100 SIRET produit.

Parmi les 76 exacts présents mais évincés, le parent est classé 11–100 pour
8 dossiers, 101–500 pour 35 et au-delà de 500 pour 33. Aucun de ces 76 parents
n'est dans le top10. La capacité de classer les bons parents reste donc une
limite concrète; cela ne prouve pas qu'un nouveau quota résoudrait les pertes.

Les 29 régressions ont toutes un nom CRM et au moins un nom officiel disponible.
Pour 18, aucun token du nom CRM n'est commun aux noms actuels parent/site
comparés. Ce diagnostic suggère d'examiner les preuves nominales faibles ou
historiques et le désaccord avec le retrieval précédent. Il ne démontre ni
une erreur de label, ni qu'ajouter des noms historiques sauverait ces cas.
Aucun correctif particulier n'a été appliqué à ces 18/29 dossiers.

## Intégrité, coûts et portée

43 tests passent, dont 8 v23 : projection des rôles officiels, impossibilité
de fabriquer une preuve entre siblings, nom légal partagé, nom manquant,
invariance aux identifiants/labels, sampling hors fold avec positifs naturels,
refus de SIREN humain partagé entre folds, cap et témoin hors géographie.

Recalcul indépendant depuis le ledger : 822 exacts, naturalité du pool,
cap/unicité, features strictement identiques entre préflight et mêmes requêtes
du run complet. Les six splits et fichiers de prédiction ont été revérifiés :
aucune requête ni SIREN humain cible dans les groupes de fit correspondants.
Les scores parent du Master correspondent aux sorties des seuls modèles
externes. Tous les hashes des sources déclarées à l'exécution et des fichiers
de résultat ont été vérifiés.

Le premier préflight a échoué sur un alias SQL réservé, avant résultat. Il
reste non admissible sous `retrieval_v23_preflight/bb12058773f80d6f`. Correction
et test de régression ont précédé le commit scientifique et toutes les phases
complètes. Aucun paramètre n'a changé après un résultat de qualité.

| Phase réussie | Calcul chronométré | Pic RSS | Swapout |
|---|---:|---:|---:|
| Préflight, 16 requêtes sans labels | 3,197 s | 0,50 Gio | 0 |
| Extraction complète sans labels | 12,386 s | 4,39 Gio | 0 |
| Six fits, prédictions OOF et Master | 27,669 s | 4,09 Gio | 0 |
| Évaluation | 0,191 s | 0,74 Gio | 0 |

Ces temps excluent la validation préalable des hashes; ce ne sont pas les
SLO end-to-end. Mac et CATNAT seuls, temporaires externes, aucun coût externe.
Versions scellées : XGBoost3.1.3, DuckDB1.4.3, NumPy2.4.1, pandas3.0.0,
PyArrow23.0.0. Code et versions sont contrôlés entre les phases.

OOF protège les fits du label de leur cible; il n'efface pas la réutilisation
de ce panel pour choisir des stratégies. Pas de test CRM indépendant, de
projection nationale, de PreRank500, Recall100, Wilson ou latence validés.
Les 29 régressions ne sont pas effacées par le +49 net ou les 512 contrôles.

## Scellement et suite

Configuration SHA-256 :
`bb12058773f80d6f5d7ff3ba7b0fc03d1b2a4eb5e95911b5deb7b33580746d69`.
Racine : `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/`.

| Manifeste relatif à la racine | SHA-256 |
|---|---|
| `retrieval_v23_features/bb12058773f80d6f/manifest.json` | `dcfffcdcc9f5d34db0d88c5039674090a2ed8911b8242b38d31f39d5c887adbe` |
| `retrieval_v23_oof_master/bb12058773f80d6f/manifest.json` | `ac4b279be0f30fbee5cfe7fc0c205c31b9cf50fcc23351795ceb55423e05de81` |
| `retrieval_v23_evaluation/bb12058773f80d6f/manifest.json` | `ca3400c50e3520d6974ac0abab49d80e17b27832bbbbaafaeb447abb5902010e` |
| `retrieval_v23_preflight/bb12058773f80d6f-sqlfix/manifest.json` | `373a729bbb4efe66d74a55d9a3c6d8bb9249f1e328898ab0c184736388c1879f` |

Prochaine orientation : mieux traiter les preuves de nom faibles et le
désaccord parent appris/retrieval, avec règles générales ou apprentissage
uniquement interne et recette préenregistrée. Ne pas choisir rétrospectivement
« v22 ou v23 » selon leur hit, ni bricoler une liste de 29 exceptions. Les
132 parents introuvables nécessitent un travail de découverte distinct.

**`PIVOT_GENERATION`**. Version expérimentale, aucune promotion, aucun national,
fold0/fold1 fermés. Un CRM réellement nouveau et scellé reste indispensable
avant une revendication de généralisation.
