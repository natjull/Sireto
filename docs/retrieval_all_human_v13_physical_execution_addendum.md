# Addendum physique v13 — acteurs persistants et artefacts sans fusion

Statut : **PRÉENREGISTRÉ — BENCHMARK V13 NON EXÉCUTÉ — OOF NATIONAL
INTERDIT**.

Ce document est une révision **strictement physique** de l’exécution all-human
v12. Il ne crée pas une nouvelle recette scientifique. Population, folds,
candidats, ordre des candidats, features, labels, sampling, modèles,
hyperparamètres, scores, tie-breaks, métriques, plafonds et seuils restent ceux
du contrat v12 et de son exécuteur fold-fused.

Le benchmark v12 du 24 août 2026 conserve irrévocablement son verdict
`STOP_NO_PARALLEL_WORKER_AUTHORIZED`. Son rapport, son manifeste et son hash ne
sont ni remplacés ni réinterprétés. Le fait que W3 y ait été exact et rapide ne
constitue aucune autorisation v13. La présente révision devra produire une
nouvelle preuve scellée ; aucun run réel n’a encore été exécuté.

La configuration normative associée est
`config/retrieval_all_human_v13_physical_execution.json`.

## 1. Périmètre scientifique inchangé

Les autorités scientifiques v12 restent content-addressées. En particulier :

- seuls les folds 2, 3 et 4 sont ouverts ;
- fold 0 reste contaminé et fold 1 fermé ;
- aucun positif n’est injecté ;
- le Master complet reste borné à 2 000 références légères ;
- le fit reste borné à 128 candidats par contexte train ;
- PreRank reste borné à 500 SIRET ;
- la sortie produit reste bornée à 100 SIRET ;
- les 319 features, leurs bits `float32`, les poids, les labels, les trois
  LambdaMART site et les six modèles parent restent identiques ;
- les mêmes candidats sont scorés et le tie-break reste score décroissant puis
  SIRET croissant ;
- Master@2000 reste diagnostique, PreRank@500 doit atteindre 99,4 %, le Recall
  exact@100 99,2 %, sa borne Wilson 95 % 99,0 %, et chaque outer fold 98,8 % ;
- les SLO produit restent p95 chaud au plus 1 s et p99 au plus 2 s.

Toute modification d’une de ces propriétés invalide la preuve physique et
impose `STOP_V13_SCIENTIFIC_DRIFT`. Les métriques de qualité observées pendant
le benchmark physique ne peuvent servir à choisir ou modifier cette
architecture.

## 2. Architecture W3 gelée

Le chemin candidat v13 utilise exactement trois acteurs `spawn` persistants
pendant chaque phase parallèle, un par fold SceneIR : `fold2`, `fold3` et
`fold4`. Chaque acteur de build :

1. ouvre lui-même ses handles read-only après le `spawn` ;
2. possède exclusivement les requêtes et row-groups de son fold ;
3. compile chaque scène une seule fois et produit les trois contextes outer ;
4. écrit ses fragments canoniques train et heldout ;
5. ne termine qu’après scellement et acquittement de son manifeste de build.

Les trois acteurs de build doivent ensuite être arrêtés, attendus et leur RSS
libérée **avant** le premier fit site. Le coordinateur ajuste les trois modèles
site séquentiellement, dans l’ordre outer2, outer3, outer4, à partir des vues de
fragments. Il ne peut fitter tant qu’un acteur de build ou un de ses handles
reste vivant.

Après scellement des modèles, trois nouveaux acteurs `spawn` de replay sont
créés, un par fold. Ils restent persistants pendant toute leur phase de replay,
reçoivent uniquement le modèle outer correspondant, lisent le spool heldout et
écrivent leurs fragments PreRank500 compact, top100 complet et outcomes. Ils
n’ouvrent jamais SceneIR, les sidecars officiels ou les modèles parent et
terminent après scellement de leur manifeste final.

Il n’existe ni processus par requête, ni pool de tâches dynamiques, ni work
stealing, ni migration d’une requête, ni retry automatique dans un autre
acteur. La persistance est bornée à la phase et au fold : conserver les acteurs
de build pendant les fits est interdit pour ne pas cumuler trois RSS de
sidecars avec la mémoire d’entraînement. Le coordinateur ne compile aucune
scène et ne fusionne aucune table de données. Il vérifie les manifests,
présente aux fits des vues ordonnées de fragments, distribue les modèles puis
calcule les gates à partir de la vue globale ordonnée.

La partition nationale reste exactement celle de v12 : fold2 `[0, 7 767)`,
fold3 `[7 767, 15 734)`, fold4 `[15 734, 23 571)`, avec respectivement les
row-groups source `[0, 210)`, `[210, 423)` et `[423, 635)`. Les trois ensembles
sont disjoints, contigus et exhaustifs.

## 3. Référence W1 immuable et parité indépendante de la concurrence

Le benchmark scellé utilise le même sample immuable de 512 requêtes. Il ne
réexécute pas W1 : sa référence est le trial W1 déjà scellé dans le manifeste
v12 SHA-256
`2269b8925a9ca5aa3b4215fc3f3ec30fccc6c59d8ace1d4165ba12daa0bb5242`,
dont le sous-manifeste W1 a le SHA-256
`67d37417237d0e93b933c95759699e871d5b3a825cb4a03deeed470391a36a4`.
Son temps mur immuable est 1 918,9878759580315 s.

W3 exécute une seule fois les trois partitions simultanément avec les acteurs
persistants par phase décrits ci-dessus. Avant de mesurer W3, le certifieur lit
le W1 content-addressé et en produit en mémoire les projections logiques v13
gelées. Il ne modifie aucun artefact W1 et ne réutilise aucune métrique de
qualité pour décider.

Cette comparaison est le **bridge non circulaire v12 W1 → v13 W3**. La parité
est byte-for-byte entre les flux de lignes canoniques ordonnés W3 et les mêmes
projections de l’ancien W1, et bit-for-bit pour toute valeur `float32`. Les
racines Merkle v13 certifient les nouveaux fragments, mais ne sont pas
comparées aux racines physiques de l’ancien format avec fichiers fusionnés. La
projection couvre au minimum : population et ordre global, parent scores,
Master et rangs, statistiques Master, lignes et groupes de fit, 319 features,
labels et poids, spools heldout, trois modèles miniatures, scores site, champs
fonctionnels communs du PreRank500, contenu complet top100, raisons de
sélection, outcomes et agrégats. La suppression du payload redondant du
PreRank500 n’est autorisée que hors projection fonctionnelle. La parité entre
deux exécutions du seul nouveau format ne pourrait pas remplacer ce bridge.

Une différence donne `STOP_V13_PARITY`. Il n’y a ni tolérance numérique, ni
tri réparateur, ni relance automatique. L’usage d’une baseline cross-run est
explicitement accepté ici parce que le sample, le code W1, ses entrées, ses
artefacts et son temps sont tous content-addressés ; le temps maximal W3 est
aussi gelé en valeur absolue afin qu’aucun choix ultérieur de baseline ne soit
possible.

## 4. Fragments Parquet canoniques et vues Merkle

Les anciennes copies `worker + merged` sont supprimées du chemin v13. Les sept
vues physiques sont des ensembles ordonnés de fragments Parquet possédés par
un fold : `train_outer2`, `train_outer3`, `train_outer4`, `heldout_spool`,
`prerank500_compact`, `top100_full` et `outcomes`. `query_groups` n’est pas une
huitième famille physique : son digest ordonné est une métadonnée obligatoire
de chacune des trois vues train.

Un fragment est canonique : schéma et ordre de colonnes gelés, groupes de
requêtes contigus, ordre SceneIR puis ordre candidat inchangé, compression et
options Parquet gelées, aucune réécriture après scellement. Sa feuille Merkle
lie la famille, le fold, l’ordinal du fragment, les bornes de requêtes, le
nombre de lignes, le SHA-256 du schéma, la taille et le SHA-256 exact du fichier.
Les feuilles sont triées uniquement par `(fold_order, fragment_ordinal)` et la
racine est calculée avec séparation binaire de domaines SHA-256 : préfixe
`0x00` pour une feuille et `0x01` pour un nœud. À chaque niveau, un dernier
élément impair est dupliqué avant le hash parent (`DUPLICATE_LAST`). Un dataset
vide est refusé. Le manifeste lie aussi la liste ordonnée complète des
feuilles : une racine seule n’est jamais une preuve suffisante.

Pour le fit outer2, la vue est `fold3 ⊕ fold4`; pour outer3,
`fold2 ⊕ fold4`; pour outer4, `fold2 ⊕ fold3`. Pour les sorties, la vue
est `fold2 ⊕ fold3 ⊕ fold4`. `⊕` est une lecture séquentielle des
fragments existants, jamais une nouvelle concaténation physique. Le lecteur
refuse fragment manquant, dupliqué, chevauchant, hors ordre, hashé
différemment, ou groupe coupé entre deux fragments.

Un fragment train d’un fold peut être référencé par deux vues train. Il ne
figure néanmoins qu’une fois dans l’inventaire des fichiers et ses octets ne
sont comptés qu’une fois. Les hardlinks sont interdits : l’unicité est définie
par le chemin canonique relatif et le SHA-256 du fichier.

Aucun fichier `merged_*` n’est produit. Les modèles miniatures ou nationaux
sont de petits artefacts indépendants, écrits une seule fois.

## 5. Payload compact sans perte de preuve

La génération et tous les calculs de features restent inchangés. Les preuves
riches nécessaires au classement des 500 sont disponibles en mémoire durant
le calcul, mais ne sont plus répétées dans un payload JSON persistant pour
chaque ligne du PreRank500.

Le fragment PreRank500 compact ne duplique pas les 319 features. Il conserve
uniquement l’identité et le fold de la requête, le SIRET, son parent, le rang,
le score site `float32`, la raison de sélection et une join key
content-addressée vers la ligne correspondante du spool heldout. Les 319
features sont rejouables depuis ce spool et la configuration scellée ; leur
digest logique complet est conservé par le benchmark. Le payload officiel
complet, avec détail des preuves et provenances, n’est sérialisé que pour le
top100.

Cette compaction est post-score et ne peut donc changer un score, un rang, une
raison ou un candidat. Une réhydratation d’audit du top500 à partir de la join
key, du spool et de la configuration scellés doit reproduire exactement les
319 features et les mêmes preuves ; son contrôle fait partie des tests
négatifs du benchmark.

## 6. Gate I/O cache-neutre

Les compteurs `proc_pid_rusage`, `psutil` et disque machine dépendent de l’état
du cache et de l’activité extérieure. Ils restent publiés comme diagnostics,
mais ne peuvent ni autoriser ni rejeter W3.

Le gate dur utilise deux quantités déterministes :

1. **lecture logique source** : chaque acteur publie les extents du Parquet
   candidat réellement nécessaires, identifiés par
   `(source_sha256, row_group, projection_sha256)`. Un extent est compté une
   seule fois par acteur, avec les octets compressés des column chunks projetés
   lus dans les métadonnées Parquet ;
2. **octets matérialisés** : somme des tailles exactes des feuilles Parquet
   canoniques, manifests et modèles écrits une seule fois. Les répertoires
   temporaires, logs et fichiers non déclarés sont également comptés, afin
   d’empêcher une écriture hors bilan.

Les trois ledgers W3 doivent être deux à deux disjoints sur la clé source
`(source_sha256, row_group)` et leur union doit être exactement égale à la
projection du ledger W1 immuable. Le nombre d’extents, leur projection et le
total d’octets logiques doivent être identiques entre W1 et W3 : amplification
logique maximale 1,000.

La projection source est la liste ordonnée exacte des 116 colonnes produites
par `retrieval_v12_scene_ir.production_projection` sur le schéma scellé de 121
colonnes. Sa liste complète est dans la configuration et son SHA-256 canonique
est `c47aa0d64d96ad756df1a0b333328df69c13071d55861f85ba5b808e6813e9e8`.
Le ledger refuse une colonne, un ordre ou un hash différent avant W3. Les
2 696 076 099 octets attendus sont bien la somme des column chunks de cette
projection, et non les 2 696 633 226 octets de toutes les colonnes.

Le ratio `octets_matérialisés_W3 / octets_matérialisés_W1_scellé` doit être au
plus 1,15. Le dénominateur est fixé à 464 378 945 octets par le trial W1 v12 ;
il inclut ses anciennes copies fusionnées et rend donc le gate v13
conservateur. Toute lecture source chevauchante, extent absent/supplémentaire,
projection différente, fichier écrit non déclaré ou ratio supérieur à 1,15
donne `STOP_V13_IO`.

Pour le sample scellé déjà audité, l’attente preregistrée est une union de 368
row-groups et 2 696 076 099 octets compressés du Parquet candidat. Cette valeur
est une vérification d’identité du sample, pas une mesure de performance et pas
un résultat v13.

## 7. Benchmark physique unique et gates

Un seul benchmark W3 est autorisé sur les 512 requêtes scellées, contre la
baseline W1 immuable décrite en section 3. Aucun nouveau W1, W2, ordre alterné,
purge du cache, canary ou essai supplémentaire n’est prévu.
Le résultat n’autorise l’OOF national que si tous les contrôles suivants
passent simultanément :

- sample exact de 512 requêtes, folds 2/3/4 uniquement ;
- parité des projections logiques W1/W3 byte-for-byte et `float32` bit-for-bit ;
- extents W3 disjoints, union W3 égale à W1, 368 row-groups et
  2 696 076 099 octets logiques attendus ;
- ratio d’octets matérialisés W3/W1 au plus 1,15 ;
- pic de RAM machine W3 au plus 14 Gio ;
- delta de swapout W3 nul ;
- temps mur W3 au plus 872,2672163445598 s, exactement équivalent à un
  speed-up `1 918,9878759580315 / wall_W3` au moins égal à 2,2 ;
- projection nationale de stockage au plus 20 Gio ;
- fold 0 fermé, fold 1 fermé, injection positive absente ;
- aucun fichier fusionné et aucun payload complet hors top100.

Les lectures/écritures process et machine, CPU et page faults sont enregistrés
pour diagnostic seulement. Ils n’apparaissent dans aucune expression de gate.
Les métriques de Recall éventuellement calculées pour vérifier la parité sont
non décisionnelles et ne peuvent être utilisées pour modifier la recette.

Le benchmark doit publier soit `PASS_V13_PHYSICAL_BENCHMARK`, soit un seul
verdict `STOP_V13_*` explicite. Même en cas de PASS, l’OOF national ne peut être
lancé que par une autorisation distincte content-addressée. Avant ce nouveau
manifeste et cette autorisation, l’état reste `NO_READY`.

## 8. Tests négatifs obligatoires

Le certifieur doit notamment rejeter : acteur non persistant ou mauvais fold,
requête migrée, fragment manquant/dupliqué/chevauchant/hors ordre, schéma ou
compression divergente, racine Merkle sans inventaire de feuilles, inversion
d’une vue train, fichier `merged_*`, payload riche hors top100, perte de preuve
compacte, extent source partagé entre acteurs, union W3 différente de W1,
double comptage d’un extent dans un acteur, projection de colonnes différente,
écriture non déclarée, ratio matérialisé supérieur à 1,15, parité altérée,
RAM ou swap dépassés, speed-up insuffisant, stockage supérieur à 20 Gio,
fold 0/1 ouvert, injection positive et dérive d’une autorité scientifique.
