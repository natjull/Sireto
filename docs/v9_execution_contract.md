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

Le runtime de référence est `.venv/bin/python`. Le smoke test obligatoire est :

```bash
.venv/bin/python scripts/check_v9_runtime.py --rows 512
```

PyTorch et FAISS sont chacun isolés dans un sous-processus dédié. Le processus
principal peut donc exécuter PyArrow/XGBoost sans charger leur runtime OpenMP.
Le worker FAISS est persistant à l’inférence et met les index en cache :
l’isolation n’ajoute pas un lancement de Python par requête.
`KMP_DUPLICATE_LIB_OK` est interdit.

Chaque index dense local doit avoir un manifeste voisin contenant au minimum
le nombre de candidats et le hash de l’ordre des SIRET. Une différence de
cardinalité ou d’ordre désactive ce canal : un indice FAISS ne peut jamais être
appliqué silencieusement à une autre scène.

Preuves Gate 0 obtenues le 23 juillet 2026 :

- smoke synthétique 512 lignes : `PASS`, dimension 384, environ 2 024
  encodages/s CPU ;
- partition réelle INSEE `01053` : 17 462 vecteurs et index FAISS produits en
  17 s environ ;
- requête dense réelle : budget strict 50, vérité terrain retournée top-1 ;
- index global SIREN limité à 1 000 entités : build et requête top-1 réussis.

Ces chiffres prouvent l’exécutabilité locale, pas la qualité du retrieval.

Le builder global écrit temporairement les vecteurs float32 sur le volume de
destination, puis construit FAISS dans un processus propre et supprime le
fichier intermédiaire. Prévoir jusqu’à environ 45 Go temporaires pour 29
millions d’entités à 384 dimensions ; le build complet doit donc cibler le SSD
externe, jamais le disque interne.

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

## Résultat de l’exécution

Décision finale : **`STOP`** pour la promotion de l’architecture V9 dense.

- Gate 0 : `PASS` ;
- Gate 1 : `PASS` ;
- Gate 2 dense local : `FAIL`, delta Recall@50 SIRET −1,83 point ;
- Gate 2 dense global SIREN : `FAIL`, delta −2,61 points ;
- Gate 3 et Gate 4 : non ouvertes conformément au présent contrat ;
- aucune dépense GPU, API payante ou cloud.

Le détail, les limites et les chemins d’artefacts sont publiés dans
`reports/v9/v9_go_pivot_stop.md`.
