# V9 sans GPU — décision finale GO/PIVOT/STOP

## Décision

# PIVOT

L’architecture V9 multicanal dense telle qu’elle a été pré-enregistrée n’est
pas promue : ses deux pools hybrides échouent Gate 2. En revanche, le dense
apporte un signal de classement Hit@1 fort et statistiquement net. La bonne
suite n’est donc ni un `GO` sur V9, ni un abandon du dense, mais un pivot :
**conserver le pool sparse et tester le dense uniquement comme signal de
scoring, sans éviction de candidats sparse**.

Aucun gain de Recall@50 n’a franchi Gate 2 :

- sparse + dense local : **−1,83 point** de Recall@50 SIRET ;
- sparse + dense global SIREN : **−2,61 points** ;
- les deux régressions sont statistiquement nettes ;
- plusieurs segments critiques perdent plus de deux points ;
- budget et latence passent, donc l’échec est qualitatif et non matériel.

En parallèle, le Hit@1 SIRET brut passe :

- de 36,22 % à 43,55 % avec le dense local, soit **+7,33 points**,
  IC95 [+5,54 ; +9,12] ;
- de 36,22 % à 47,52 % avec le dense global SIREN, soit **+11,31 points**,
  IC95 [+9,67 ; +12,98].

Selon le contrat, le ranker/accepteur V9, les 500 adjudications open-set et le
cross-encoder ne doivent pas être lancés sur le pool hybride rejeté. Le pivot
proposé est une nouvelle expérience et ne fait pas partie de ce goal.

## Gates

| Gate | Statut | Preuve |
|---|---|---|
| 0 — Exécutabilité Mac | PASS | Runtime CPU isolé, tokenizer réparé, FAISS et stores mmap/DuckDB opérationnels |
| 1 — Baseline gelée | PASS | Benchmark `c33b80855f560074`, splits SIREN-disjoints, manifestes et hashes |
| 2a — Dense local | FAIL | 88,50 % contre 90,33 %, delta −1,83 pt |
| 2b — Dense global SIREN | FAIL | 87,72 % contre 90,33 %, delta −2,61 pts |
| 3 — Ranker/accepteur V9 | NON OUVERTE | Interdite sur le pool hybride après échec de Gate 2 |
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
3. **Le dense contient néanmoins un vrai signal d’ordre.** Le Hit@1 progresse
   fortement aux niveaux SIRET et SIREN. Ce signal doit être dissocié de
   l’admission de nouveaux candidats.
4. **Un ranker ne doit pas servir à sauver après coup le pool ayant échoué son
   gate.** Une ablation `pool sparse fixe + feature dense` doit avoir son propre
   contrat pour rendre le gain attribuable.
5. **Le Mac n’est pas le facteur limitant.** L’index de près de 29 millions de
   SIREN et l’évaluation complète ont été réalisés en CPU, sans location de GPU.
   L’espace V9 occupe environ 34 Go sur le SSD externe.
6. **Le benchmark reste fermé et historique.** Il permet une comparaison
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

Ne pas poursuivre la fusion symétrique. Si un nouveau goal est ouvert, la
première ablation rationnelle est :

1. geler exactement les 50 candidats sparse actuels ;
2. calculer pour eux les scores/rangs denses local et global, sans ajouter ni
   retirer aucun candidat ;
3. mesurer ces signaux comme features du ranker avec prédictions OOF ;
4. n’ouvrir l’accepteur que si Hit@1 SIRET end-to-end progresse sans baisse de
   Recall@50 ni régression segmentaire.

L’audit complet des features/scènes du ranker, decider et risk-model
historiques reste nécessaire avant ce nouveau cycle. Un rescue conditionnel
des scènes sans candidat crédible pourra être une seconde hypothèse, jamais
mélangée à la première.

## Références d’artefacts

- baseline test :
  `/Volumes/CATNAT_DATA/SIRETO_V9/experiments/sparse50_c33b80855f560074_4e82530`
- dense local dev :
  `/Volumes/CATNAT_DATA/SIRETO_V9/comparisons/dev_local_minilm867_c33b80855f560074_de0079a`
- dense global dev, résultats :
  `/Volumes/CATNAT_DATA/SIRETO_V9/experiments/dev_global_siren_v2_c33b80855f560074_5123516`
- dense global dev, comparaison corrigée :
  `/Volumes/CATNAT_DATA/SIRETO_V9/comparisons/dev_global_siren_v2_c33b80855f560074_de0079a`
