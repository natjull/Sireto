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
`67d37417237d0e93b933c95759699e871d5b3a825cb4a03deeed470391a36a74`.
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
fonctionnels communs du PreRank500, payload et preuves complets du PreRank500
avant compaction, contenu complet top100, raisons de sélection, outcomes et
agrégats. La suppression du payload redondant du PreRank500 n’est autorisée
que dans sa représentation physique compacte : une racine logique séparée
`prerank500_full_payload` doit rester identique au W1 immuable. La parité entre
deux exécutions du seul nouveau format ne pourrait pas remplacer ce bridge.

L’ordre canonique des 512 requêtes est celui des outputs W1/SceneIR scellés,
cross-validé entre `heldout_spool`, `top500`, `top100` et `outcomes` : fold2,
puis fold3, puis fold4 dans leur ordre de scène. Le fichier sample ne sert qu’à vérifier l’égalité de la
population ; son éventuel ordre de hash ne peut jamais ordonner une racine.
Cette même séquence canonique ordonne notamment `prefit_bridge_logical`,
`site_feature_values` et les projections de sortie. Une permutation change la
racine et impose `STOP_V13_PARITY`.

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
exactement les sept colonnes historiques nécessaires au résultat :
`query_id`, `fold`, `candidate_siret`, `candidate_parent`, `rank`,
`site_score` (`float64`, afin de préserver les bytes de la référence) et
`selection_reason`. Il ne répète aucun hash ni pointeur de spool sur les
millions de lignes. La clé naturelle `(query_id, candidate_siret)` doit être
unique dans une requête et rejoint le dataset heldout dont la racine Merkle
content-addressée est scellée une seule fois dans le manifeste. Les 319
features sont rejouables depuis ce dataset et la configuration scellée ; leur
digest logique complet est conservé par le benchmark. Avant suppression du
payload PreRank500 de la représentation physique, chaque acteur scelle une
feuille par requête sur la table Arrow complète de schéma top100 : les sept
colonnes communes plus le `payload_json` canonique. La racine globale
`prerank500_full_payload` doit égaler celle extraite du top500 W1 immuable. Le
payload officiel complet, avec détail des preuves et provenances, n’est
sérialisé que pour le top100.

Cette compaction est post-score et ne peut donc changer un score, un rang, une
raison ou un candidat. Pendant le callback chronométré, le replay scelle les
digests des 319 features et du payload complet déjà présents en mémoire ; il
ne les recalcule pas. Après l’arrêt du chronomètre W3, le certifieur effectue
exactement une réhydratation indépendante du top500 à partir de la clé
naturelle, du dataset heldout scellé et de la configuration scellée. Elle doit
reproduire bit pour bit les 319 features, le payload et les mêmes preuves ; sa
racine doit être égale à la fois à la racine W3 scellée avant compaction et à
la racine W1 immuable. Une reconstruction dans le callback ou une deuxième
reconstruction fausserait le gate de temps et est interdite.

## 6. Gate I/O cache-neutre

Les compteurs `proc_pid_rusage`, `psutil` et disque machine dépendent de l’état
du cache et de l’activité extérieure. Ils restent publiés comme diagnostics,
mais ne peuvent ni autoriser ni rejeter W3.

Le preflight v13 précédant le chronomètre ne lit et ne hashe jamais les 4,7 Gio
du Parquet candidat. Il contrôle néanmoins le hash du manifeste source, le
hash, le schéma et la cardinalité de l’index SceneIR, l’existence et la taille
non nulle du Parquet, son footer, son nombre de lignes et la présence des
colonnes requête/fold. Le SHA-256 intégral du Parquet candidat est recalculé
obligatoirement par la comparaison I/O **après** l’arrêt du chronomètre W3.
Cette séparation empêche le preflight de réchauffer artificiellement le cache
de la source ; une validation intégrale avant W3 invalide le benchmark.

L’entrypoint capable de publier une autorisation possède lui-même ce preflight
et appelle l’API scellée avec `verify_source_hash=False`. Il n’accepte ni objet
`Prepared` construit par l’appelant, ni projection W1 fournie de l’extérieur.
Il construit puis recharge strictement, avant le marqueur one-shot, les
projections fonctionnelle et I/O W1 dans leurs répertoires content-addressés.
Les callbacks injectables restent réservés au harness de test : même lorsque
tous leurs contrôles sont vrais, ils publient
`STOP_V13_UNTRUSTED_EXECUTION_HOOKS` et ne peuvent créer aucune autorisation.

Le gate dur utilise deux quantités déterministes :

1. **lecture logique source** : chaque acteur publie les row-groups du Parquet
   candidat qu'il a effectivement décodés, identifiés par
   `(source_sha256, row_group, projection_sha256)`. Un extent est compté une
   seule fois par acteur, avec les octets compressés des column chunks projetés
   lus dans les métadonnées Parquet ;
2. **octets matérialisés** : somme des tailles exactes des feuilles Parquet
   canoniques, manifests et modèles écrits une seule fois. Les répertoires
   temporaires, logs et fichiers non déclarés sont également comptés, afin
   d’empêcher une écriture hors bilan.

Cette preuve n'est jamais reconstruite depuis les seules plages du shard.
`SceneIRCache.iter_selected_scenes` émet un événement uniquement après le
succès d'un vrai `read_row_group` et uniquement lors d'un cache-miss. Chaque
acteur scelle l'ordre de lecture, l'ordinal, les bornes et le nombre de lignes,
les 116 colonnes et leur hash de projection, les octets compressés, le hash
source, le hash du manifeste source, le build SceneIR, le shard et la lignée
d'exécution. Son manifeste et son Parquet de lignes sont tous deux requis dans
l'inventaire. Un ledger absent, supplémentaire, dupliqué, hors ordre ou dont
la projection dérive impose `STOP_V13_IO`.

Les trois ledgers W3 effectivement observés doivent être deux à deux disjoints sur la clé source
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

Le run national publie et inventorie les mêmes trois ledgers réels. Ses shards
exhaustifs doivent couvrir exactement les 635 row-groups de la source, sans
chevauchement, pour 4 646 932 145 octets compressés sur les mêmes 116 colonnes.
La validation de ces ledgers intervient après la sortie des trois acteurs build
et avant tout scellement de dataset, fit LambdaMART ou replay.

## 7. Benchmark physique unique et gates

Un seul benchmark W3 est autorisé sur les 512 requêtes scellées, contre la
baseline W1 immuable décrite en section 3. Aucun nouveau W1, W2, ordre alterné,
purge du cache, canary ou essai supplémentaire n’est prévu.
L’unique destination admissible est
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_physical_benchmark_v13_w3`.
Avant d’entrer dans le callback chronométré, le lanceur crée atomiquement avec
`O_EXCL` le marqueur sibling
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/.retrieval_v13_w3_attempt_consumed.json`.
Un marqueur déjà présent ou toute autre destination sont refusés avant le
calcul. Une fois le marqueur créé, un PASS, un STOP ou une exception consomme
irrévocablement l’essai : aucune relance automatique ou manuelle sous une autre
destination n’est admissible dans ce contrat.
Le hash de l’implémentation et la liste de ses fichiers sont figés avant la
création du marqueur. Une exception utilise cette preuve pré-marquage pour
publier son STOP durable, même si des fichiers dérivent pendant le callback.
L’API de build W3 refuse également tout appel direct : elle exige le chemin,
le hash et le payload exacts de ce marqueur déjà consommé, ainsi que le chemin
canonique `.../retrieval_fold_fused_physical_benchmark_v13_w3/trial_w3`.
Une exception après consommation doit laisser dans la destination canonique
un manifeste durable `STOP_V13_EXECUTION_EXCEPTION` liant le chemin et le hash
du marqueur, la configuration, le hash d’implémentation et la phase échouée ;
aucune autorisation nationale ne peut alors être publiée.
Cette obligation commence dès que le processus a obtenu `O_EXCL` : une panne
d’écriture, de `fsync`, de hash ou de contrôle du marqueur après sa création
publie elle aussi le STOP durable et interdit toute nouvelle tentative.
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
L’empreinte d’environnement héritée de v12 scelle l’implémentation et la
version Python, le système et l’architecture génériques, ainsi que les versions
des dépendances critiques. Elle ne scelle pas la release/build exacte de macOS
ni les variables de threads BLAS/OpenMP (`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`). Le
benchmark ne constitue donc pas une preuve de reproductibilité entre machines
ou environnements différents ; cette limite est explicite et non décisionnelle.

Le benchmark doit publier soit `PASS_V13_PHYSICAL_BENCHMARK`, soit un seul
verdict `STOP_V13_*` explicite. Même en cas de PASS, l’OOF national ne peut être
lancé que par une autorisation distincte content-addressée. Avant ce nouveau
manifeste et cette autorisation, l’état reste `NO_READY`.

Le loader de cette autorisation ne croit jamais les booléens `checks` du
rapport. Il rouvre le W1 immuable, le produit W3, la preuve I/O et les métriques
de ressources, recalcule le speed-up, les seuils RAM/swap/wall/stockage, les
ratios I/O et le verdict avec la même précédence, puis exige l’égalité exacte
avec le rapport. Toute substitution rehashée d’un temps, d’un pic RAM, du
swap, du stockage ou d’un ratio logique est donc rejetée.

### Publication nationale transactionnelle

L’autorisation nationale distincte ne peut viser qu’une destination :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_national_oof_v13`.
Avant tout worker, le lanceur consomme avec `O_EXCL` le marqueur sibling
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/.retrieval_v13_national_attempt_consumed.json`.
Son identité lie configuration, autorisation, benchmark physique,
implémentation, source, plan sharded, PREFIT et destination. Le calcul écrit
ensuite dans un stage sibling content-addressé
`.retrieval_fold_fused_national_oof_v13.stage-<SHA256>`. Une autre destination,
un marqueur ou un stage antérieur interdisent toute relance automatique.

Tous les chemins entre fragments, spools, modèles et manifests internes sont
relatifs à ce même arbre et vérifiés sous sa racine ; les autorités externes
restent des records absolus dédiés du manifeste national. Après scellement,
inventaire récursif et relecture, le stage complet est promu par un unique
`rename` atomique sur le même filesystem. Une erreur, y compris pendant la
promotion ou la relecture finale, remet l’arbre au stage et publie
`STOP_NATIONAL_EXECUTION_EXCEPTION` avec la preuve d’implémentation capturée
avant le marqueur. Le stage n’est ni supprimé ni réessayé.

Après extinction et `join` des trois acteurs build, immédiatement avant le
premier fit, une seconde garde scellée vérifie zéro acteur build vivant, un
fit à la fois dans l’ordre 2/3/4, collecte mémoire entre les outer fits, delta
de swapout nul et RAM disponible suffisante. Cette garde photographie l’état
pré-fit ; elle ne prétend pas mesurer le pic mémoire pendant les fits. Aucun
seuil nouveau n’est ajouté : sa marge minimale disponible vaut
`RAM_machine_totale - 14 Gio`, dérivée du plafond du benchmark. Le pic réel du
run national restera une métrique du profil runtime scellé. Un échec produit
`STOP_V13_NATIONAL_PREFIT_RESOURCE_GUARD`, sans fallback W1/W2 ni relance.

## 8. Tests négatifs obligatoires

Le certifieur doit notamment rejeter : acteur non persistant ou mauvais fold,
requête migrée, fragment manquant/dupliqué/chevauchant/hors ordre, schéma ou
compression divergente, racine Merkle sans inventaire de feuilles, inversion
d’une vue train, fichier `merged_*`, payload riche hors top100, perte de preuve
compacte, extent source partagé entre acteurs, union W3 différente de W1,
double comptage d’un extent dans un acteur, projection de colonnes différente,
ledger réel absent, row-group observé supplémentaire, doublon ou ordre de
lecture altéré,
écriture non déclarée, ratio matérialisé supérieur à 1,15, parité altérée,
RAM ou swap dépassés, speed-up insuffisant, stockage supérieur à 20 Gio,
fold 0/1 ouvert, injection positive, plus d’un modèle outer ouvert par acteur
de replay, réhydratation dans le chronomètre ou répétée, racine payload
PreRank500 différente, autorisation/lineage national substitué et dérive d’une
autorité scientifique. S’ajoutent pour le run national : sortie alternative,
marqueur ou stage déjà présent, chemin interne absolu ou échappant à la racine,
mémoire/swap pré-fit insuffisant, exception sans STOP durable, promotion d’un
stage en échec et arbre impossible à rouvrir après le `rename` atomique.
Une faute postérieure à l’obtention de `O_EXCL`, mais antérieure au retour du
writer de marqueur, doit également laisser le marqueur et un STOP durable, tant
pour le benchmark W3 que pour le run national.
