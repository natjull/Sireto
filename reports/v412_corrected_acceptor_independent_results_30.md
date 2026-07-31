# V4.12 — validation indépendante de l'accepteur corrigé

Date : 31 juillet 2026

## Population métier

Les 30 dossiers ont été adjudiqués en aveugle avant ouverture des sorties du
modèle :

| Verdict métier | Nombre |
|---|---:|
| `MATCH_EXACT` | **26** |
| `AMBIGUOUS` | **4** |
| `UNRESOLVED` | **0** |

Vingt-quatre exacts sont `HIGH`; Damartex et Edeis sont `MEDIUM`. Pour
Damartex, le SIRET fermé de l'ancienne adresse n'est pas utilisé comme cible :
la vérité courante retenue est le siège actif `44137831200025`.

Les labels et preuves sont dans
[`v412_corrected_acceptor_independent_labels_30.csv`](v412_corrected_acceptor_independent_labels_30.csv).

## Modèle figé avant l'audit

- ranker : poids difficile `0,5` ;
- accepteur : XGBoost monotone, poids difficile `10` ;
- seuil : `0.8974587321281433` ;
- aucune modification après lecture des 30 labels ;
- aucun accès au test final.

## Résultat

| Mesure | Résultat |
|---|---:|
| Bon top 1 du ranker parmi les exacts | **24/26 (92,31 %)** |
| Erreurs ranker | 2 |
| `AUTO_MATCH` | **0/30** |
| `REVIEW` | **30/30** |
| Couverture AUTO | **0 %** |
| Erreur AUTO | 0 |
| Ambiguïté automatisée | 0 |

Les scores accepteur vont de `0,0308` à `0,8328`, tous sous le seuil figé de
`0,8975`. Le détail est dans
[`v412_corrected_acceptor_independent_results_30.csv`](v412_corrected_acceptor_independent_results_30.csv).

Les deux erreurs du ranker sont :

- `fresh:FR025705` — ANACOURS : vérité `44146037500229`, top 1
  `51164444500039` ;
- `12298` — DAMARTEX GROUP : vérité active courante `44137831200025`, top 1
  `33320208300015`.

## Verdict

**`PIVOT_ACCEPTOR_REDESIGN`**.

Le `GO_NEW_INDEPENDENT_ACCEPTOR_DOCKET` précédent reposait sur seulement trois
AUTO difficiles hors échantillon. Le nouveau lot montre qu'il ne se généralise
pas : l'accepteur actuel obtient sa sécurité par abstention totale sur la
population qui devrait précisément augmenter la couverture.

Le ranker n'est plus le goulot principal sur ce lot. La suite ne consiste ni à
baisser le seuil après coup ni à augmenter encore les poids. Il faut revoir la
cible et les preuves utilisées par l'accepteur, puis valider toute nouvelle
politique sur un autre lot non adjudiqué parmi les 99 REVIEW historiques
restants.
