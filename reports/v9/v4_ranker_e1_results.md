# E1 — Ranker V4 exact-SIRET

## Verdict

**`GO_ACCEPTEUR_V4`**

Sur les 305 requêtes du nouveau dev indépendant :

| Ordre candidat | Hit@1 SIRET | Hit@1 SIREN |
|---|---:|---:|
| Admission brute | 8/305 = 2,623 % | 34/305 = 11,148 % |
| Ancien ranker E1 épinglé | 290/305 = 95,082 % | 292/305 = 95,738 % |
| Nouveau ranker V4 | **299/305 = 98,033 %** | **301/305 = 98,689 %** |

Le nouveau ranker gagne **2,951 points absolus** et neuf bonnes décisions
nettes face à l’ancien : il corrige dix erreurs et dégrade un ancien succès.
Son intervalle Wilson à 95 % sur le Hit@1 SIRET est
[95,775 % ; 99,095 %].

## Interprétation simple

Le retrieval a déjà assuré que le bon SIRET figure parmi les 100 candidats.
Le ranker doit ensuite le faire remonter en première position.

Il y parvient dans 299 cas sur 305. Les six erreurs restantes se répartissent
ainsi :

- deux choix du mauvais établissement au sein du bon SIREN ;
- quatre choix d’un mauvais SIREN.

Ce taux de 98,03 % n’est pas encore une précision d’automatisation. La brique
suivante, l’accepteur, doit reconnaître les scènes risquées et les envoyer en
`REVIEW`. Elle devra notamment écarter les six erreurs sans rejeter trop de
bons cas.

## Dataset et contrôles

- 5 749 requêtes fit exactes ;
- 305 requêtes dev exactes ;
- 604 938 paires CRM–SIRET ;
- 55 features déterministes ;
- exactement un positif réel par requête ;
- aucun doublon `(query_id, candidate_siret)` ;
- 100 candidats maximum ;
- zéro SIREN exact partagé entre fit et dev ;
- zéro positif injecté ;
- holdout scellé et ancien test non lus.

Les requêtes fit `6818` et `8109` ont été exclues car leur liste historique ne
contenait pas la vérité V4. Elles n’ont pas été réparées par injection.

## Entraînement

- `XGBRanker` pairwise ;
- cinq folds OOF groupés par SIREN ;
- seed 42 ;
- features sémantiques désactivées ;
- CPU du Mac uniquement ;
- aucun GPU ou service externe.

Sur les prédictions OOF du fit, le ranker atteint 5 616/5 749 =
97,687 % Hit@1 SIRET et 5 632/5 749 = 97,965 % Hit@1 SIREN.

## Artefacts

- Contrat : `docs/v4_ranker_e1_contract.md`, commit `0c90c25`.
- Builder dataset : `scripts/build_v4_ranker_dataset.py`, commit `6236365`.
- Évaluateur : `scripts/evaluate_v4_ranker_e1.py`, commit `250a05f`.
- Dataset :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/ranker_v4/1aebeada820d92a7/`
- Modèle et prédictions :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/ranker_v4/ranker_1aebeada820d92a7_6236365/`
- Comparaison :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/ranker_v4_e1_250a05f/`
- Suite complète : 140 tests passants.

## Étape suivante

Construire les scènes de l’accepteur à partir :

1. des prédictions OOF du fit exact ;
2. des scènes `AMBIGUOUS` de fit, qui doivent toujours cibler `REVIEW` ;
3. des preuves top-1/top-2 et de leurs écarts ;
4. d’un dev séparé en deux parties pour choisir puis calibrer le seuil.

`UNRESOLVED` reste exclu tant qu’il n’est pas validé comme `NO_MATCH`,
`AMBIGUOUS` ou `MATCH_EXACT`. Le holdout reste fermé jusqu’au gel du modèle et
du seuil.
