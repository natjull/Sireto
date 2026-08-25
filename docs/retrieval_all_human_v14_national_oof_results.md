# Retrieval all-human v14 — résultat OOF national

Date d'exécution : 25 août 2026

Verdict projet : **`PIVOT_RECALL_BELOW_GATE`**

Statut machine du manifeste : **`EXECUTED_NO_READY_CLAIM`**

## Portée et intégrité

L'unique tentative nationale v14 a exécuté la recette gelée sur les 23 571
requêtes des folds 2/3/4. Les trois acteurs build ont produit les fragments
zero-copy, les trois LambdaMART ont été entraînés séquentiellement, puis les
trois acteurs replay ont produit le top500 compact, le top100 complet et les
outcomes. Le stage a été promu atomiquement vers le répertoire canonique après
validation.

Le runner officiel a terminé avec le code 0. Une seconde invocation du loader
strict a rouvert la source, l'autorisation, les trois modèles, les ledgers, les
sept datasets canoniques, l'inventaire et les gates sans divergence. Fold0 et
fold1 sont restés fermés, aucun positif n'a été injecté et aucun fichier
`merged` n'a été créé.

## Résultats principaux

| Surface | Succès / total | Mesure | Seuil gelé | Verdict |
|---|---:|---:|---:|---|
| Master@2000 | 22 826 / 23 571 | 96,8393 % | >= 99,5 % | FAIL |
| PreRank@500 | 22 394 / 23 571 | 95,0066 % | >= 99,4 % | FAIL |
| Recall@100 SIRET exact | 21 974 / 23 571 | **93,2247 %** | >= 99,2 % | **FAIL** |
| Borne Wilson95 basse Recall@100 | — | 92,8968 % | >= 99,0 % | FAIL |
| Recall@100 opérationnel | 21 974 / 23 571 | 93,2247 % | vue séparée | — |

L'intervalle Wilson95 du Recall@100 exact est
`[92,8968 % ; 93,5386 %]`. Atteindre 99,2 % sur ce dénominateur exige au moins
23 383 succès : il manque donc 1 409 hits. `recall_gates_pass=false`.

Le manifeste conserve `gate_eligible=false` et `NO_READY_DECISION`, car la
parité POSTFIT est encore `PENDING` et la latence `NOT_EVALUATED`. Ces étapes
ne peuvent toutefois pas inverser l'échec des gates de Recall déjà observé.
Le verdict projet est donc un PIVOT, jamais un GO vers fold0/fold1.

## Stabilité par outer fold

| Outer fold | Succès / total | Recall@100 exact | Wilson95 |
|---|---:|---:|---:|
| 2 | 7 242 / 7 767 | 93,2406 % | [92,6607 % ; 93,7778 %] |
| 3 | 7 393 / 7 967 | 92,7953 % | [92,2066 % ; 93,3427 %] |
| 4 | 7 339 / 7 837 | 93,6455 % | [93,0838 % ; 94,1645 %] |

Les trois folds échouent le minimum gelé de 98,8 %. Leur proximité montre que
le résultat global n'est pas causé par un seul split accidentel.

## Anatomie exacte des 1 597 pertes

| Famille de perte | Nombre | Part des requêtes |
|---|---:|---:|
| `SITE_NOT_EXPANDED` | 525 | 2,2273 % |
| `EVICTED_MASTER_TO_500` | 432 | 1,8328 % |
| `EVICTED_500_TO_100` | 420 | 1,7819 % |
| `PARENT_NOT_RETRIEVED` | 219 | 0,9291 % |
| `OFFICIAL_CATALOG_ABSENT` | 1 | 0,0042 % |

Le plafond principal est en amont du LambdaMART : 745 vérités sont déjà
absentes de Master@2000 (`SITE_NOT_EXPANDED`, `PARENT_NOT_RETRIEVED` ou
catalogue absent). Même un ranker parfait ne pourrait donc pas atteindre le
gate avec le pool actuel. Les 852 pertes restantes se répartissent ensuite
presque également entre la réduction 2000→500 et la réduction 500→100.

## Segments les plus informatifs

- établissements actifs : 17 184 / 17 880 = 96,1074 % ;
- établissements fermés : 3 133 / 3 682 = 85,0896 % ;
- nouveau site d'un SIREN connu : 1 657 / 2 009 = 82,4788 % ;
- CRM du 17 août 2026 : 12 218 / 13 310 = 91,7956 % ;
- CRM historique : 9 756 / 10 261 = 95,0785 % ;
- labels `UNRESOLVED` : 1 307 / 1 651 = 79,1641 %.

La vue opérationnelle est identique à la vue exacte dans cet artefact. Elle ne
fournit donc aucun rattrapage au gate SIRET exact.

## Artefacts et autorités

- manifeste national :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/retrieval_fold_fused_national_oof_v14/manifest.json`
- SHA-256 manifeste :
  `c361bdf6f284e9996390341e215cc5bf17e0e78a4b09134db7037071747d1308`
- run ID :
  `131a6c76afa1cac92272e4234720422b867c0b45ce1d522532548ffd39a9f222`
- marker national SHA-256 :
  `449730b42d8c4e7f1fbdcc9535cd813cb44b8fa864ea1ecf5fefb6db5d58e5cb`
- autorisation physique SHA-256 :
  `0684d350afa717a74644647f7ce7e6b43d81943e28e77e5b76d5df0c4539cfe6`
- inventaire SHA-256 :
  `7106712b0f280c1dec5ffd975b8ac33a4f3cba0c126352ec30274f198701f09b`
- volume matérialisé : 6 819 150 349 octets, soit environ 6,35 Gio ;
- garde avant fit : `PASS_PREFIT_RESOURCES`, zéro swapout, fits 2→3→4
  séquentiels.

## Décision et prochaine architecture

Ne pas ouvrir fold0 ni fold1 et ne pas lancer de modèle aval. Une itération
limitée au tuning LambdaMART ne peut pas combler l'écart : le prochain contrat
doit d'abord modifier la génération et l'expansion pour récupérer les 745
vérités absentes de Master, puis traiter séparément les pertes 2000→500 et
500→100. Toute nouvelle recette doit être préenregistrée sur folds 2/3/4 avant
une nouvelle évaluation ; le run v14 reste une autorité terminale, non
réutilisable comme essai ajustable.
