# Retrieval all-human v12 — résultat du benchmark physique shardé

Date : 24 août 2026
Verdict : `STOP_NO_PARALLEL_WORKER_AUTHORIZED`

Ce document enregistre le premier benchmark physique réel de l'exécuteur
fold-fused shardé. Il ne modifie ni la recette scientifique, ni les labels, ni
les modèles. Il ne constitue pas une évaluation nationale du Recall.

## Artefact et population

- code gelé : commit `ca4c4f3` ;
- sample runtime scellé : 512 requêtes folds 2/3/4 ;
- ordre exécuté : W3, puis W1, puis W2 conditionnel ;
- manifeste :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_physical_benchmark_v12_ca4c4f3/manifest.json` ;
- SHA-256 du manifeste :
  `2269b8925a9ca5aa3b4215fc3f3ec30fccc6c59d8ace1d4165ba12daa0bb5242` ;
- fold 0 ouvert : non ;
- fold 1 ouvert : non ;
- positif injecté : non.

Le benchmark n'a publié aucun `execution_authorization.json`. L'OOF national
n'a donc pas été lancé.

## Résultats physiques

| Exécution | Temps mur | Speed-up vs W1 | RAM machine pic | Swapout | Parité complète | Verdict |
|---|---:|---:|---:|---:|---:|---|
| W1 | 1 918,99 s | 1,000x | 11,04 Go | 0 | référence | référence |
| W2 | 1 330,46 s | 1,442x | 13,15 Go | 0 | oui | rejet vitesse |
| W3 | 679,15 s | 2,826x | 13,42 Go | 0 | oui | rejet I/O |

La parité est exacte pour les huit digests scellés : heldout spool, matrices
d'entraînement, groupes, trois modèles miniatures, PreRank500, top100,
outcomes et bridge PREFIT. Le plafond RAM machine de 14 Gio et l'absence de
swapout sont respectés par W2 et W3.

W2 échoue uniquement au seuil de speed-up préenregistré : 1,442x pour un
minimum de 1,7x. Avec trois shards de taille voisine exécutés en deux vagues,
ce résultat est cohérent avec le plafond d'utilisation imposé par la
topologie physique actuelle.

W3 passe la vitesse, la RAM, le swap et la parité. Il échoue au ratio I/O du
process tree : 5,411 pour un maximum de 1,15. Les autres observations scellées
ne racontent pas la même histoire :

| Mesure | W1 | W3 | Ratio W3/W1 |
|---|---:|---:|---:|
| lectures process tree | 724 160 512 | 5 011 951 616 | 6,921 |
| écritures process tree | 250 548 224 | 262 520 832 | 1,048 |
| lectures disque machine | 144 789 475 328 | 56 854 482 944 | 0,393 |
| artefacts matérialisés par le core | 464 378 945 | 464 746 996 | 1,001 |

W3 a été exécuté avant W1. Le compteur process reflète donc aussi la différence
de cache froid/chaud, tandis que le compteur machine intègre l'activité de
toute la machine et la durée très différente des essais. Ces chiffres ne
permettent pas de requalifier rétroactivement le gate : le verdict du protocole
préenregistré reste `STOP`.

## Stockage projeté

Le benchmark projette 21 339 984 795 octets pour le national, sous le plafond
de 21 474 836 480 octets. Le gate stockage passe, mais sa marge n'est que
d'environ 135 Mo ; une prochaine architecture doit supprimer les copies
physiques redondantes plutôt que consommer cette marge.

## Suite autorisée

Avant toute nouvelle mesure, il faut préenregistrer un protocole physique qui
ne change aucun candidat, feature, modèle, fold ni seuil de Recall et qui :

1. mesure l'amplification I/O indépendamment de l'ordre froid/chaud ;
2. conserve une preuve directe de couverture disjointe des row groups source ;
3. évite les doubles copies worker + merged lorsque le lecteur peut consommer
   une vue ordonnée des shards ;
4. améliore l'occupation de W2 au-delà des deux vagues de trois shards, ou
   confirme W3 avec une métrique I/O cache-neutre ;
5. refait un benchmark scellé avant toute autorisation nationale.

Aucun résultat de qualité modèle ne doit être utilisé pour choisir ce nouveau
protocole physique.
