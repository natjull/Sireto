# Retrieval all-human v14 — protocole physique NULL-canonique

Statut : **préenregistré, non exécuté**. Ce document ne constitue ni un
`PASS`, ni une autorisation nationale, ni une nouvelle mesure scientifique.

## 1. Motif et périmètre

Le one-shot v13 est terminé par `STOP` et reste immuable. Son reçu terminal
final est :

- commit de tombstone : `914c475` ;
- manifeste :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v13_stop_receipt/bdcba90d29bd7da0/manifest.json` ;
- SHA-256 :
  `5cb886338693e7198b34fbed7ea3987e6fff68fdb2b1abee9a4091ea6d8269ca` ;
- receipt id :
  `bdcba90d29bd7da0e84fb805936fe0f99f8518ce9661a30751412c005f6d6799`.

Le reçu antérieur `005fb…` est un brouillon superseded, sans autorité. Aucun
artefact v13 n'est récupéré, continué, corrigé sur place ou compté comme essai
v14.

V14 ne change que le protocole physique de hash logique des tables Arrow. La
cause v13 était un hash dépendant des octets masqués sous des valeurs NULL :
le contenu fonctionnel W1/W3 était identique, mais deux buffers physiques
différaient. V14 canonicalise les NULL par `take(identity)` avant sérialisation
IPC :

`SIRETO_ARROW_IDENTITY_TAKE_IPC_SHA256_NULL_CANONICAL_V14_1`.

Les candidats, features (319), labels, modèles, hyperparamètres, folds,
budgets, caps, rangs, tie-breaks et gates qualité restent strictement ceux du
contrat v12. Il n'est permis ni d'ajuster un seuil ni de relire fold0/fold1.

## 2. Référence et expérience uniques

La seule référence est le W1 v12 immuable :

- benchmark SHA :
  `2269b8925a9ca5aa3b4215fc3f3ec30fccc6c59d8ace1d4165ba12daa0bb5242` ;
- trial W1 SHA :
  `67d37417237d0e93b933c95759699e871d5b3a825cb4a03deeed470391a36a74` ;
- 512 requêtes, folds 2/3/4 ;
- wall : `1918.9878759580315 s` ;
- octets matérialisés : `464378945`.

Il sera exécuté exactement un W3 v14 frais de 512 requêtes. W1 n'est pas
réexécuté, W2 est interdit, le cache machine n'est pas purgé et aucun artefact
W3 v13 ne peut servir de point de départ.

Chemins v14 uniques :

- projections :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_v14_immutable_projections` ;
- benchmark :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_physical_benchmark_v14_w3` ;
- marker benchmark :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/.retrieval_v14_w3_attempt_consumed.json` ;
- autorisation : `execution_authorization_v14.json` sous le benchmark ;
- national :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_national_oof_v14` ;
- marker national :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/.retrieval_v14_national_attempt_consumed.json`.

## 3. Projection W1 v14 non circulaire

Avant le marker irréversible, le runner officiel :

1. ouvre uniquement les métadonnées SceneIR avec `verify_source_hash=false` ;
2. ouvre le W1 v12 et PREFIT scellés ;
3. recalcule les seize racines fonctionnelles avec le provider v14 ;
4. scelle un manifeste natif
   `sireto-retrieval-v14-immutable-w1-functional-projection-1` ;
5. y lie le provider (id, record schema, SHA, fingerprint et vecteur
   self-test), les bytes du
   moteur v13 réutilisé, l'implémentation v14 complète et tous les inputs ;
6. utilise le SHA complet comme dossier content-addressé ;
7. recalcule les racines même si ce dossier existe déjà.

Un manifeste de projection v13 n'est jamais admissible. Un cache préexistant
forgé, une collision de préfixe, un provider clone, un changement de code ou
un input modifié imposent un arrêt avant consommation du marker.

Les seize racines couvrent train outer2/3/4, groupes, heldout, modèles,
PREFIT, 319 features replay, PreRank500 compact et payload complet réhydraté,
top100 et outcomes.

## 4. Provider explicite et compatibilité v13

Le moteur fragmenté v13 reste réutilisé avec une injection explicite du
provider sur toutes ses surfaces : writer, seal/load train et heldout, fit,
replay, vues de parité, zero-copy, réhydratation et loaders stricts.

Le provider v13 demeure le défaut historique. Cette compatibilité doit être
prouvée byte-for-byte par les tests : sans injection, les manifests, racines
et sorties restent ceux de v13. En v14, seul le singleton v14 scellé est
accepté. Sont refusés : clone `dataclasses.replace`, constructeur manuel,
rebind de l'alias exporté, mauvais id/révision/SHA/callable/capability et
mutation accidentelle du code détectée par fingerprint/self-test.

Le threat model n'affirme pas résister à un attaquant capable de réécrire
arbitrairement toute la mémoire du processus Python. Toute dérive persistante
des fichiers reste couverte par le hash d'implémentation v14.

## 5. Architecture W3 et gate I/O

W3 conserve trois acteurs build persistants, un par fold, concurrence 3. Ils
lisent des row-groups disjoints et publient les ledgers réels. Ils quittent et
leur RSS est libérée avant les fits LambdaMART séquentiels 2, 3, 4. Trois
nouveaux acteurs replay ouvrent chacun exactement son modèle outer et jamais
SceneIR, sidecars ou parents.

Les sept familles physiques restent : `train_outer2`, `train_outer3`,
`train_outer4`, `heldout_spool`, `prerank500_compact`, `top100_full`,
`outcomes`. Aucun merge. Top500 conserve sept champs compacts ; les payloads
riches ne sont persistés que pour top100. La preuve payload top500 est scellée
avant compaction puis reconstruite hors timer.

Le gate I/O utilise la projection exacte de 116 colonnes, SHA
`c47aa0d64d96ad756df1a0b333328df69c13071d55861f85ba5b808e6813e9e8`.
Les trois ledgers doivent être disjoints et leur union égale au W1 logique :
368 row-groups et `2696076099` octets compressés projetés pour le benchmark.
Le hash source complet n'est effectué qu'après la mesure W3.

## 6. Gates physiques gelés

Tous les gates doivent passer ensemble :

- parité exacte des seize racines W1 v12 ↔ W3 v14 ;
- réhydratation indépendante exacte, une seule fois hors timer ;
- zéro merge et protocole payload compact respecté ;
- amplification matérialisée `<= 1,15` ;
- wall W3 `<= 872.2672163445598 s` ;
- speedup W1/W3 `>= 2,2` ;
- pic mémoire machine `<= 14 GiB` ;
- delta swapout `= 0` ;
- projection nationale `<= 20 GiB`.

Les métriques process/machine I/O restent diagnostiques. Une erreur de parité
fonctionnelle produit `STOP_V14_PARITY`; elle n'est pas une exception à
retenter. Toute exception/intégrité produit un STOP durable et consomme
l'essai.

## 7. One-shot, PASS et autorisation

Le runner officiel possède le preflight, les projections, la mesure et les
comparaisons. Il n'accepte aucun callback, `Prepared` ou produit injecté pour
publier un PASS. Le marker est créé par `O_EXCL`, fsyncé avant le callback. Un
PASS/STOP/exception ou une faute après `O_EXCL` consomme définitivement v14.

Seul `PASS_V14_PHYSICAL_BENCHMARK` peut publier
`sireto-retrieval-v14-national-authorization-1`. Le loader recalcule les gates
depuis les faits scellés, rouvre marker, benchmark, W3, logical-I/O, projection
v14 et implementation lineage. Une autorisation v13 est refusée.

## 8. National v14 (verrouillé avant PASS)

Le chemin national existe avant le benchmark pour que ses bytes soient inclus
dans l'implementation hash. Il ne peut s'exécuter qu'avec l'autorisation v14
exacte. Il utilise 23 571 requêtes, trois shards fold-owned, zéro merge, fits
séquentiels et replay v14.

Un marker national distinct est consommé avant stage/worker. Le travail se
fait dans un stage content-addressé, strictement rouvert, puis promu par rename
atomique même filesystem. Une exception conserve un STOP durable ; aucun
retry/salvage v13 ou stage précédent n'est permis. Le manifeste national reste
`EXECUTED_NO_READY_CLAIM` : il ne constitue ni un GO fold1 ni un résultat de
production.

## 9. Décision scientifique inchangée

Après une autorisation physique, le national republiera les gates v12 sans
ajustement : Master@2000 `>=99,5 %`, PreRank@500 `>=99,4 %`, Recall@100 exact
`>=99,2 %`, Wilson95 `>=99,0 %`, chaque outer `>=98,8 %`, maximum 100. Fold0
reste contaminé, fold1 fermé, et aucun positif ne peut être injecté.

Ce préenregistrement n'autorise aucun calcul. Le benchmark v14 ne peut être
lancé qu'après revue, gel des SHA et validation des tests contractuels.
