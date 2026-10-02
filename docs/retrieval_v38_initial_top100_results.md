# Résultat du premier pilote de recherche initiale à 100

Le 2 octobre 2026, la recette v38 retrouve105 des128 SIRET humains exacts,
soit **82,03125% de recall@100 avant tout modèle aval**. Le comparateur
direct fixé retrouve103/128, soit80,46875%. Le primaire a9 gains et7 pertes.
La recette est **rejetée** : elle reste très loin de99,2% et régresse sur
une origine CRM. Verdict **PIVOT_GENERATION**, aucun déploiement ni promotion.

Ce résultat concerne ce premier candidat de recherche initiale, pas toute
l'ancienne recherche multicanal. Il ne remplace pas le96,97% du pilote v37,
qui mesurait un classement appris différent sur1024 requêtes. Ces chiffres
ne sont pas directement comparables.

## Ce qui a réellement été exécuté

Préenregistrement, code et tests : commit `0a8bae16`, avec mise à jour de
la directive active dans `AGENTS.md` selon la clarification utilisateur.
Le [contrat](retrieval_v38_initial_top100_contract.md) fixe les128 premières
requêtes des quatre lots hashés v34, sans sélection par hit/label/statut.
Ce sous-ensemble appartient au panel de développement déjà réutilisé ; il
n'est pas un holdout indépendant.

Chaque requête interroge le vrai index national :82772691 documents, dont
43896818 établissements. Le moteur recherche seulement les documents SIRET
et retourne directement au maximum100 références. Aucun pool intermédiaire,
parent appris, modèle XGBoost, reranker ou sélection de vérité n'intervient.
Seule la coupure native du moteur définit la sortie.

Le primaire combine les vues BM25 nom courant, enseigne, nom de site,
historique/supporting et adresses, mots et ngrams. Nom et adresse peuvent
apporter chacun une preuve sans être tous deux obligatoires. Commune/CP
apportent un bonus constant au lieu d'une exclusion dure. Les poids restent
ceux fixés avant mesure, sans ajustement sur ces erreurs.

Le comparateur utilise la requête conjointe v30 nom courant ET adresse
courante, avec son filtre de localité, mais retourne lui aussi100 références
directes sans la suite v23/v30. Ce n'est donc PAS une reproduction de tout
le pipeline historique. Les deux recherches alternent leur ordre par requête.

Les16 premières requêtes ont passé un gate de coût sans labels :3,64510s
en moyenne pour la paire de recherches, sous le plafond expérimental4s.
Les128 sorties ont ensuite été terminées et scellées avant ouverture des
labels dans un processus d'évaluation séparé.

## Résultats et segments

| Mesure | Primaire | Comparateur direct |
|---|---:|---:|
| Exact@100 | 105/128 =82,03125% | 103/128 =80,46875% |
| Opérationnel@100 | 108/128 =84,37500% | 106/128 =82,81250% |
| Wilson95 borne basse exacte | 74,47837% | 72,75593% |
| Fold2 | 26/38 | 24/38 |
| Fold3 | 46/51 | 45/51 |
| Fold4 | 33/39 | 34/39 |
| CRM20260817 | 60/80 =75% | 64/80 =80% |
| CRM historique | 45/48 =93,75% | 39/48 =81,25% |

Les deux recettes réussissent ensemble96 requêtes et échouent toutes deux
sur16. Les9 gains et7 régressions sont tous conservés dans le rapport.
Le gain global de2 ne justifie pas de masquer les4 hits nets perdus sur
CRM20260817, ni de promouvoir cette recette sur le seul CRM historique.

Les128 labels exploitables restent au dénominateur. Les16 absences humaines
de la population source sont publiées séparément (folds2/3/4 :8/3/5).
Le manifeste publie aussi label_kind, qualificationsV2/V3, état, origine,
unseen_siren, new_site_known_siren et mega_base_pool ; les valeurs vides sont
UNKNOWN, pas des qualifications inventées. Aucune de ces catégories n'entre
dans la recherche ou ne filtre la métrique.

Le sidecar officiel label-side scellé est vérifié et repris pour l'exact et
l'opérationnel. Les23 pertes primaires sont NOT_RETRIEVED_AT100 ; aucune
n'est attribuée à une absence du catalogue officiel. Cette classe ne prouve
pas une absence dans l'index : l'attribution fine score/représentation/index
reste explicitement non mesurée. Aucune vérité n'est recherchée dans le
moteur pour produire ou corriger un résultat.

## Diagnostic sans nouvelle recherche

Audit postérieur `24fdb1a1`, réalisé uniquement sur les références déjà
retournées et la localité officielle, en1,09s sous garde, environ205Mio,
zéro nouveau swapout. Aucun fit ou nouvel appel à l'index.

Sur12800 références primaires,12792 ont une localité dans le sidecar
officiel utilisé. Parmi celles-ci,7704, soit**60,23%**, sont hors de la
commune ET du code postal CRM. Les8 non jointes restent inconnues, pas
comptées hors zone. Pour les7 régressions,485/700 références, soit69,29%,
sont hors de ces deux clés. Le comparateur est entièrement local par
construction.

C'est compatible avec un soutien géographique trop faible face aux
ressemblances textuelles nationales. Ce n'est pas une preuve causale que
la géographie explique seule les7 pertes, ni que tous les candidats hors
zone sont faux. Augmenter arbitrairement le poids jusqu'à récupérer ces7
cas serait du réglage sur le résultat de développement, pas une validation.

Un fait structurel est néanmoins acquis : les7 vérités perdues existaient
dans le top100 du comparateur et satisfont aussi le prédicat textuel plus
large du primaire. Elles ne sont donc pas perdues par absence du corpus
ou nouveau filtre dur ; le nouveau score les évince de son top100.

Le code de matérialisation confirme par ailleurs que les documents SIRET
incluent déjà les noms légaux et usuels de leur parent officiel. Il ne faut
pas présenter cette jointure comme une source nouvelle à ajouter. En
revanche, l'index ne possède aucun champ XY et ne porte pas automatiquement
toutes les sources temporelles ou relations conservées séparément.

## Coût et limites de la mesure

Génération primaire ET comparateur :**296,86s**, soit environ4min57 pour
128 requêtes. Pic worker2,12GioRSS, aucun nouveau swapout sur les16 workers,
tous les budgets passés. Processus séquentiels avecRAYON_NUM_THREADS=2.
Aucune réindexation nationale, aucun téléchargement de modèle, aucune dépense.

| Temps recherche et extraction des références | p50 | p95 | p99 |
|---|---:|---:|---:|
| Primaire | 1,557s | 2,761s | 3,299s |
| Comparateur | 0,500s | 1,300s | 1,986s |

Le cache froid/chaud n'est pas contrôlé : ces mesures ne valident pas les
SLO de production. La première recherche primaire a pris10,72s, à publier
comme extrême observé plutôt que le cacher derrière les percentiles.
Les manifestes, métadonnées, inventaires de segments et nombre de documents
de l'index ont été contrôlés avant/après ; ses gros segments n'ont pas été
rehashés intégralement pendant ce pilote. Cette limite est publiée.

30 tests ciblés passent, dont7 pour ce moteur : cap100, filtre document
SIRET, absence de résultat sans texte, recherche hors localité, historique,
adresse sans accord de nom, bonus géographique borné et comparateur direct.
Folds0/1 non ouverts, aucun modèle aval modifié. Aucun job restant en cours.

## Orientation conservée après ce rejet

Le contrat de recherche initiale100 reste actif. Ce test rejette la recette
précise « somme de vues lexicales et bonus local constant », pas toute
recherche lexicale ni toute recherche sans filtre communal.

La suite pertinente est une représentation qui permet au moteur de chercher
ensemble les preuves de nom, de site physique et de temporalité, avec des
coordonnées et une incertitude BAN réellement indexables. Le coût de sa
construction doit être mesuré avant toute réindexation nationale. Les
relations officielles doivent conserver leur provenance et leur portée ;
pas de propagation d'une enseigne de site à tous les établissements.

Le challenger neuronal de recherche initiale reste non exécuté. Il devra
affronter le même corpus national et le même cap100, avec un préflight de
coût Mac, sans réutiliser le résultat v9 dense comme garantie ni renommer
un reranker en moteur de recherche. Aucun de ces travaux n'est annoncé comme
déjà améliorant le recall. Pas de grille de poids ni de nouveau run national
inchangé lancé à la clôture.

## Artefacts

Racine CATNAT :
`SIRETO_RECALL100/audits/retrieval_v38_initial_top100/pilot_20261002_a/`.

| Fichier | SHA256 |
|---|---|
| `launch.json` | `a515bcd6add4d8b69ae32d26ffde0c372751408959150971ec51a46e0ff61de6` |
| `manifest.json` | `e37d9d874a79bf0c05dea1003b120a12fe7451c773134877b64aef847235b7fa` |
| `evaluation.json` | `98b4816d04b6d1b3c7506a466851b825ade0a9257a37e7257c0ca5b61804bec5` |
| `geography_audit.json` | `b45320db1ec88627f53cb75eeecd0b8435e39a66727a5658be164f927e878b43` |

Verdict final : **PIVOT_GENERATION**. Recherche directe effectivement testée,
mais objectif99,2% non atteint, recette rejetée, aucun READY_FOR_FOLD1.
