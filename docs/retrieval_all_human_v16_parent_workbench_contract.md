# Contrat du workbench parent SIREN v16

Statut : préenregistré, préflight validé, exécution officielle non lancée.

## Question testée

Le pivot v15 n'a retrouvé le bon parent que pour 103/522 misses top1750,
alors que l'expansion a ensuite émis 98/103 bons sites. V16 teste si des
documents **SIREN natifs** peuvent porter le recall parent au niveau requis,
sans tronquer un SIRET et sans ouvrir la vérité pendant la génération.

Le pool est exactement celui de v15 : 522 misses et 512 contrôles
déterministes, folds 2/3/4 seulement. Le générateur reçoit uniquement les
`query_id`, les folds et les scènes CRM label-blind scellées. Les folds 0/1,
les SIRET humains, le type de cohorte et tout hit/rang vérité lui sont
inaccessibles.

## Autorités et canaux

Les manifests et chemins sont figés dans
`config/retrieval_all_human_v16_parent_workbench.json`. Les six backends sont :

- dossier SIREN national v3 ;
- overlays officiels RNE et officiel v2 ;
- historique SIRENE natif ;
- rôles d'entités légales RNE ;
- preuves sectorielles officielles.

Chaque recherche force `document_type=siren`. Les 26 canaux conservent
séparément backend, famille, mode exact/word/char/adresse, rang et score. Aucun
champ SIRET n'est consulté ou publié. Les champs autorisés et les types de
manifest sont contrôlés avant la première recherche ; toute dérive échoue
fermée.

## Génération X et évaluation Y

La génération publie uniquement `parent_signals.parquet`, puis le scelle avec
son hash. L'évaluateur truth-side est un autre module et un autre processus. Il
ouvre ensuite `cohort_labels.parquet` et compare les parents à
`human_ground_truth_siret[:9]`. Ce calcul truth-side ne modifie jamais X.

Les ablations préenregistrées isolent : équivalent lexical v15, base natif
multicanal, overlays officiels, temporel, rôles, secteur, lexical global et
portefeuille natif complet. La fusion est un RRF fixe `k=60`, avec poids
déclarés avant exécution. Sont publiés Recall parent aux rangs
8/16/32/64/128/250/384/512 et MRR pour le pool candidat équilibré, le RRF et
chaque canal, ainsi que l'oracle de l'union brute et le gain par rapport aux
parents v15 scellés. Le pool candidat alterne les canaux dans l'ordre gelé et
se ferme à 512 parents uniques, sans consulter Y.

## Gates et décisions

Le workbench s'arrête avant toute expansion site si, à parent@512, un fold est
sous 155/148/138. Le niveau de confiance exige 160/158/146, soit 464 parents
au total. Les 512 contrôles v11 restent immuablement dans le préfixe base ; le
workbench ne remplace aucune ligne Master.

- `PASS_PARENT_V16_FEASIBILITY` : niveau de confiance parent franchi ; seul un
  pilote parent→site séparé peut alors être préenregistré ;
- `PROMISING_PARENT_V16` : au moins +100 parents versus v15 sans hard kill ;
- `PIVOT_PARENT_INDEX` : hard kill ou gain insuffisant.

Un PASS parent ne vaut pas PASS Master. L'étape suivante doit encore produire
les rescues exacts 155/148/138, Master@2000 >=99,5 % global et par fold, et
préserver 512/512 contrôles avant tout run national.

## Ressources et séquence

Préflight : 1 à 16 requêtes, sortie `/tmp`, non décisionnel. Exécution officielle
unique : 1 034 requêtes, <=30 minutes, RSS <=14 Gio, zéro swapout, artefacts
neufs sur `/Volumes/CATNAT_DATA`. Toute modification de canal, poids, limite,
gate ou autorité exige une nouvelle version ; aucune reprise/touche post-hoc
n'est jointe au résultat v16.
