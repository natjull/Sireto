# V9 sans GPU — décision finale GO/PIVOT/STOP

## Décision

# STOP

Cette décision porte sur la **promotion de l’architecture V9 multicanal dense**
telle qu’elle a été pré-enregistrée. Elle ne signifie ni que SIRETO doit être
abandonné, ni que l’architecture historique sparse/XGBoost était une mauvaise
direction.

Aucun gain fiable n’a franchi Gate 2 :

- sparse + dense local : **−1,83 point** de Recall@50 SIRET ;
- sparse + dense global SIREN : **−2,61 points** ;
- les deux régressions sont statistiquement nettes ;
- plusieurs segments critiques perdent plus de deux points ;
- budget et latence passent, donc l’échec est qualitatif et non matériel.

Selon le contrat, le ranker/accepteur V9, les 500 adjudications open-set et le
cross-encoder ne doivent pas être lancés sur ce retrieval.

## Gates

| Gate | Statut | Preuve |
|---|---|---|
| 0 — Exécutabilité Mac | PASS | Runtime CPU isolé, tokenizer réparé, FAISS et stores mmap/DuckDB opérationnels |
| 1 — Baseline gelée | PASS | Benchmark `c33b80855f560074`, splits SIREN-disjoints, manifestes et hashes |
| 2a — Dense local | FAIL | 88,50 % contre 90,33 %, delta −1,83 pt |
| 2b — Dense global SIREN | FAIL | 87,72 % contre 90,33 %, delta −2,61 pts |
| 3 — Ranker/accepteur | NON OUVERTE | Interdite après échec de Gate 2 |
| 4 — 500 labels open-set | NON OUVERTE | Pas de signal fermé justifiant le coût humain |
| Cross-encoder | NON LANCÉ | Hors chemin critique et aucune dépense GPU autorisée |

Le split test n’a pas été utilisé pour choisir ou régler une variante dense.
La seule mesure test disponible reste la baseline sparse gelée :
2 348/2 652, soit **88,54 % Recall@50 SIRET**.

## Ce que l’expérience tranche

1. **La direction historique n’était pas fondamentalement mauvaise.** Sur les
   données disponibles, le sparse géographique fournit un meilleur pool
   candidat que les deux alternatives denses génériques évaluées.
2. **Le retrieval dense n’est pas une amélioration gratuite.** Il récupère des
   misses, mais évince davantage de bons candidats quand il partage le même
   budget 50.
3. **Un ranker ne doit pas servir à sauver une hypothèse retrieval ayant échoué
   son gate.** Cela mélangerait deux changements et rendrait la conclusion
   impossible à attribuer.
4. **Le Mac n’est pas le facteur limitant.** L’index de près de 29 millions de
   SIREN et l’évaluation complète ont été réalisés en CPU, sans location de GPU.
   L’espace V9 occupe environ 34 Go sur le SSD externe.
5. **Le benchmark reste fermé et historique.** Il permet une comparaison
   reproductible, pas une certification open-set ni une garantie de précision
   en production.

## Ce qui reste valable

- la baseline sparse V7/V9 et l’architecture historique
  ranker/decider/risk-model comme référence reproductible ;
- le dataset SIREN-disjoint et ses manifestes ;
- les correctifs tokenizer, train/serve et anti-fuite ;
- les runners à budget fixe, les comparaisons appariées et les courbes
  risque-couverture déjà implémentées ;
- les index et stores sur SSD comme artefacts d’étude, non comme composants à
  promouvoir.

## Suite recommandée hors de ce goal

Ne pas poursuivre une « V9 dense » par inertie. Si un nouveau cycle est ouvert,
le point de départ rationnel est la baseline sparse, avec une hypothèse unique
et pré-enregistrée, par exemple :

- améliorer le sparse sans changer le budget ;
- auditer complètement les features et les scènes du ranker/decider/risk-model
  historiques ;
- tester un rescue conditionnel uniquement sur les scènes où le local ne
  fournit pas de candidat crédible, sans fusion symétrique avec les hits sparse.

Ces pistes sont un nouveau plan expérimental. Elles ne transforment pas le
résultat actuel en `PIVOT` : dans le périmètre testé, aucune brique dense n’a
apporté de gain net fiable.

## Références d’artefacts

- baseline test :
  `/Volumes/CATNAT_DATA/SIRETO_V9/experiments/sparse50_c33b80855f560074_4e82530`
- dense local dev :
  `/Volumes/CATNAT_DATA/SIRETO_V9/comparisons/dev_local_minilm867_c33b80855f560074_fa19430`
- dense global dev, résultats :
  `/Volumes/CATNAT_DATA/SIRETO_V9/experiments/dev_global_siren_v2_c33b80855f560074_5123516`
- dense global dev, comparaison corrigée :
  `/Volumes/CATNAT_DATA/SIRETO_V9/comparisons/dev_global_siren_v2_c33b80855f560074_bc49918`

