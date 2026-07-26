# Résultats V4-Fresh — lignes CRM hors benchmark

## Verdict

- verdict : **`PASS_V4_FRESH`** ;
- lignes sources inédites : **6 330** ;
- chevauchement d'identifiant avec le benchmark : **0** ;
- test historique lu : **non** ;
- holdout évalué par un modèle : **non**.

Le dépôt contenait suffisamment de lignes CRM hors benchmark pour lever le
blocage de volume V4 sans demander un nouvel export.

## Qualification

| Label fresh | Volume | Part |
|---|---:|---:|
| `MATCH_EXACT` | 1 426 | 22,528 % |
| `AMBIGUOUS` | 247 | 3,902 % |
| `UNRESOLVED` | 4 657 | 73,570 % |

La couverture est plus faible que sur le benchmark historique parce que ces
lignes avaient justement été écartées du sous-ensemble CRM considéré comme
propre.

Parmi les 1 426 exacts :

- 1 099 conservent le SIREN historique mais changent de SIRET ;
- 325 changent de SIREN ;
- 2 n'ont pas de SIRET historique valide ;
- 636 anciens SIRET sont encore actifs, 587 sont fermés et 203 absents du
  snapshot ;
- les 1 426 nouveaux SIRET V4 sont tous actifs et possèdent une preuve
  nom–adresse unique.

## Séparation gelée

| Rôle | Requêtes | `MATCH_EXACT` | `AMBIGUOUS` | `UNRESOLVED` |
|---|---:|---:|---:|---:|
| `fit_addition` | 3 664 | 819 | 142 | 2 703 |
| `dev_new` | 1 321 | 305 | 53 | 963 |
| `holdout_sealed` | 1 345 | 302 | 52 | 991 |

- exacts du noyau V4 historique : 4 932 ;
- exacts du fit combiné : **5 751** ;
- SIREN exact partagé fit/dev : 0 ;
- SIREN exact partagé fit/holdout : 0 ;
- SIREN exact partagé dev/holdout : 0 ;
- SIREN exact de `dev_new` ou `holdout_sealed` partagé avec le noyau
  historique : 0.

Tous les gates pré-enregistrés passent :

- fit combiné ≥5 000 exacts ;
- nouveau dev ≥300 exacts ;
- holdout ≥300 exacts ;
- zéro fuite SIREN ;
- zéro chevauchement avec le benchmark ;
- zéro SIRET fermé retenu.

## Portée

Le holdout est qualifié et hashé mais ne doit pas être ouvert par les scripts
d'entraînement. Son volume est suffisant pour une évaluation préliminaire, pas
pour garantir statistiquement 99,8 %.

La suite autorisée est désormais :

1. produire les candidats du retrieval gelé pour le noyau V4 +
   `fit_addition` et pour `dev_new` ;
2. entraîner le ranker sur les seuls exacts ;
3. entraîner l'accepteur sans `UNRESOLVED` ;
4. geler le bundle et le seuil ;
5. ouvrir une seule fois `holdout_sealed`.

## Artefact

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/v4_fresh_expansion/14047b719ef90f6f`

- contrat : `1c2e84c` ;
- builder/tests : `613cf7d` ;
- suite complète : 132 tests passants.
