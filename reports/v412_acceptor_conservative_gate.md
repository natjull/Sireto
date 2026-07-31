# V4.12 — gate dev de l'accepteur conservateur

Artefact local :
`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_acceptor_conservative/88e50a879d7fcc2b`.

La famille XGBoost monotone, le poids trusted `10`, les 80 features et le
ranker sont inchangés. Le seuil final de développement est calibré sur les 279
scores OOF consommés : `0.9886879324913025`.

| Population | AUTO | Corrects | Erreurs | Couverture |
|---|---:|---:|---:|---:|
| 279 trusted OOF consommés | 89 | 89 | 0 | 31,90 % |
| 1 127 contrôles positifs dev non utilisés | 1 077 | 1 077 | 0 | 95,56 % |
| **Projection dev combinée** | **1 166** | **1 166** | **0** | **82,93 %** |

Aucune ambiguïté n'est automatisée. La précision observée est 100 %, mais
la borne basse Wilson 95 % n'est que 99,67 %. Cette mesure ne certifie donc pas
99,8 %.

Les 1 127 contrôles ne contiennent que des positifs exacts : ils prouvent la
couverture et l'absence de régression sur ce segment, pas la capacité à rejeter
de nouveaux négatifs. Les 279 trusted ont servi à calibrer le seuil et sont
explicitement qualifiés de développement consommé.

Verdict : **`GO_FREEZE_FOR_NEW_HOLDOUT`**. Le gate dev de couverture est
franchi, mais aucun holdout local vierge n'existe : les anciens tests et
holdouts sont consommés et interdits. Le test final reste fermé jusqu'à
réception d'un nouvel export CRM indépendant.
