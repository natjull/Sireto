# Résultats aval E1/E2 — top-100 gelé

## Verdict

- **E1 ranker : PASS**
- **E2 accepteur : `PIVOT_ACCEPTEUR`**
- test sélectif final lu : **non**

Le nouveau ranker améliore réellement le choix du SIRET. En revanche,
l'accepteur actuel ne retrouve pas une couverture AUTO utile à très haute
précision. Il ne faut ni revenir au bundle historique, ni présenter cette
première tête de confiance comme déployable.

## Dataset

Build : `3171ef5020c0f068`.

- 11 837 requêtes train ;
- 2 565 requêtes dev ;
- 1 438 845 paires CRM–SIRET ;
- 100 candidats maximum ;
- zéro doublon `(query_id, candidate_siret)` ;
- zéro détail candidat manquant ;
- zéro test dans le dataset ;
- 44 features déterministes candidat et 11 features du retrieval sélectif ;
- trois features sémantiques exclues.

Recall@100 sur les labels V3 `MATCH_EXACT` :

| Split | Succès | Recall@100 |
|---|---:|---:|
| train | 9 347/9 426 | 99,162 % |
| dev | 2 095/2 104 | 99,572 % |

Le premier build a été refusé parce que deux candidats fermés de l'overlay
étaient chargés depuis une autre partition que le canal V7. Le builder a été
corrigé pour respecter séparément la provenance géographique de chaque
source. L'artefact refusé est conservé dans `downstream_failed`.

## E1 — Ranker final

| Scorer dev | Hit@1 SIRET | Succès |
|---|---:|---:|
| Ordre brut de l'admission | 1,663 % | 35/2 104 |
| Ancien decider calibré | 76,996 % | 1 620/2 104 |
| Ancien decider brut | 77,044 % | 1 621/2 104 |
| Ancien ranker | 80,561 % | 1 695/2 104 |
| **Nouveau ranker** | **83,365 %** | **1 754/2 104** |

Delta nouveau ranker contre ancien ranker : **+2,804 points**.

| Segment V3 exact dev | Volume | Ancien | Nouveau | Delta |
|---|---:|---:|---:|---:|
| tous | 2 104 | 80,561 % | 83,365 % | +2,804 |
| actifs | 1 766 | 84,881 % | 88,052 % | +3,171 |
| fermés | 338 | 57,988 % | 58,876 % | +0,888 |
| mégapoles | 144 | 72,222 % | 72,222 % | 0 |
| multi-sites | 488 | 77,869 % | 81,967 % | +4,098 |
| localisation INSEE | 2 076 | 80,491 % | 83,333 % | +2,842 |
| localisation CP seule | 28 | 85,714 % | 85,714 % | 0 |

E1 passe : le nouveau ranker gagne globalement et aucune famille critique ne
régresse.

## E2 — Accepteur exact-SIRET

Les scènes train proviennent exclusivement des prédictions OOF groupées par
SIREN. Dev est séparé en deux moitiés déterministes : calibration et choix du
seuil. Aucun test n'est disponible ni évalué.

- régression logistique : non éligible et avertissement de non-convergence ;
- XGBoost : seul modèle éligible ;
- seuil isotonic retenu : `1.0` ;
- AUTO : 33/1 280 ;
- couverture : **2,578 %** ;
- erreurs : 0 ;
- précision observée : 100 %.

Le même point est retenu aux cibles 99,0 %, 99,5 % et 99,8 %. Le palier
isotonic suivant contient 134 AUTO et 2 erreurs, soit 98,507 % : il échoue
déjà à 99,0 %.

Le gate pré-enregistré exigeait au moins 25 % de couverture dev à 99,8 %
observé. E2 échoue donc sans ambiguïté.

## Lecture

L'expérience répond à la question d'architecture :

1. conserver le retrieval sélectif est justifié ;
2. remplacer les deux scorers candidat par un ranker final unique est
   bénéfique ;
3. revenir au risk model historique serait injustifié : sa couverture était
   gonflée par les top-1/top-2 dupliqués et sa cible était le SIREN ;
4. l'accepteur V9 actuel reste insuffisant ;
5. la calibration isotonic réduit ici les scores à de gros paliers et détruit
   la finesse nécessaire au choix de couverture.

La suite requiert un nouveau mini-contrat limité à l'accepteur : standardiser
la régression logistique, comparer score brut, calibration sigmoid et isotonic
sans changer le ranker ni le dataset, puis geler à nouveau sur dev. Ce travail
ne justifie ni GPU, ni dense, ni retour au pipe historique.

## Artefacts

- dataset :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/downstream/3171ef5020c0f068` ;
- ranker :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/ranker_3171ef5020c0f068_fc9cb1b` ;
- accepteur :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/downstream/acceptor_3171ef5020c0f068_fc9cb1b` ;
- builder corrigé : commit `fc9cb1b` ;
- tests : 121 passants après exécution.
