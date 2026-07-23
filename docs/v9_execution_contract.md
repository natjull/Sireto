# Contrat d’exécution — Expérience SIRETO V9 sans GPU

## Décision recherchée

L’expérience doit conclure par exactement une décision :

- `GO` : V9 franchit les gates et justifie l’entraînement end-to-end ;
- `PIVOT` : une brique précise apporte un signal, mais l’architecture V9
  complète n’est pas justifiée ;
- `STOP` : aucun gain fiable ne justifie la poursuite.

Une absence de gain est un résultat valide. Le but n’est pas de confirmer V9.

## Ressources autorisées

- MacBook Pro M4 Pro, 12 cœurs et 24 Go de mémoire ;
- environnement Python local isolé du projet ;
- SSD `/Volumes/CATNAT_DATA`, sans effacer les autres projets présents ;
- aucune location de GPU, API payante ou infrastructure cloud.

Les artefacts volumineux V9 doivent être écrits sous un dossier dédié :

```text
/Volumes/CATNAT_DATA/SIRETO_V9/
```

## Séquence et gates

### Gate 0 — Exécutabilité

- suite de tests verte ;
- encodeur et FAISS utilisables sans contournement OpenMP dangereux ;
- index dense et candidats chargés avec le même contrat de partition ;
- smoke test prouvant que le canal dense retourne réellement des candidats.

### Gate 1 — Baseline gelée

- snapshot SIRENE, requêtes, labels et splits identifiés par hash ;
- aucun tuning après lecture du test final ;
- sparse V7/V9 mesuré avec un budget final strict de 50 ;
- nombres bruts, Recall@50, Hit@1 SIRET/SIREN et latence p95 publiés.

### Gate 2 — Retrieval hybride

Comparer sur les mêmes requêtes :

1. sparse local SIRET ;
2. sparse + dense dans les mêmes partitions ;
3. sparse local + dense global SIREN, puis expansion SIRET.

Une variante ne passe que si :

- Recall@50 est strictement supérieur à la baseline ;
- aucune famille critique ne régresse de plus de 2 points ;
- la latence p95 reste inférieure à deux fois la baseline ;
- le budget final reste exactement identique.

### Gate 3 — Ranking et abstention

Cette gate n’est ouverte que si Gate 2 passe.

- ranker candidat unique ;
- scènes train produites out-of-fold ;
- exactitude évaluée au SIRET, jamais au seul SIREN ;
- calibration et choix du seuil sur deux sous-ensembles dev distincts ;
- test final consulté une fois ;
- `AUTO_NO_MATCH` interdit.

### Gate 4 — Travail humain

Les 500 adjudications open-set ne sont lancées qu’après un signal positif aux
gates fermées. Un LLM peut réunir des preuves mais ne valide aucun label.

## Cross-encoder

Le cross-encoder est exclu du chemin critique. Il ne bloque aucune décision
`GO`, `PIVOT` ou `STOP`. Une ablation CPU ultérieure reste possible seulement
si le pipeline sparse/dense/XGBoost atteint un plafond documenté.

## Traçabilité

Chaque expérience conserve :

- commande exacte ;
- versions et hashes des entrées ;
- configuration et seed ;
- durée et latence ;
- résultats bruts et agrégés ;
- erreurs par famille ;
- commit Git correspondant.

`handover.md` est mis à jour à chaque milestone. Aucun ancien artefact n’est
déplacé ou supprimé pour libérer de la place.
