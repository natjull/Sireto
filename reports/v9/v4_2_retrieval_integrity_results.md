# V4.2 — Intégrité du retrieval représentatif

Date : 27 juillet 2026  
Verdict : **`GO_HARD_LABELS`**

## Conclusion directe

Le défaut de retrieval identifié par l'audit V4.1 est corrigé sans nouvelle
architecture, sans GPU et sans réentraînement.

Sur les 242 cas `MATCH_EXACT` provisoires figés avant le correctif, la variante
B avec le snapshot SIRENE complet conserve **242/242 bons SIRET = 100 %**
dans 100 candidats maximum.

Ce résultat franchit le gate technique de 99,0 %. Il ne mesure pas la
précision des décisions AUTO et n'autorise pas le déploiement. Il autorise
uniquement la prochaine étape : constituer des labels représentatifs
difficiles avant de réentraîner les modèles.

## Correctif

Le TF-IDF, RRF, les partitions géographiques et le budget n'ont pas changé.
Le ranker, l'accepteur et leur seuil sont restés gelés.

Deux changements seulement :

1. la variante B utilise le SIRET/SIREN d'entrée comme indice, sans le
   considérer comme vérité ;
2. la dernière barrière vérifie l'état administratif dans le snapshot SIRENE
   complet de 42 322 035 établissements, au lieu de se fier au magasin rapide
   de 14 378 332 candidats.

Le magasin rapide reste utilisé pour enrichir les candidats et développer les
sites d'un SIREN. Une absence dans ce magasin n'est désormais plus assimilée
à une fermeture.

Le snapshot complet est lu en lecture seule et par lots depuis le parquet
local. Aucun second exemplaire massif n'a été créé sur le Mac ou le SSD.

## Résultats

| Mesure | Résultat |
|---|---:|
| Cas exacts provisoires | 242 |
| Recall@100 variante A auditée | 237/242 = 97,934 % |
| Recall@100 variante B avant correction d'état | 240/242 = 99,174 % |
| Recall@100 V4.2 | **242/242 = 100 %** |
| Tirage aléatoire exact | **91/91 = 100 %** |
| Maximum de candidats | 100 |
| Candidats fermés | 0 |
| Vérités absentes du snapshot d'état | 0 |
| Positifs injectés | 0 |
| Régressions sur les 237 anciens succès | 0 |
| Latence p50 | 455,1 ms |
| Latence p95 | 2 878,6 ms |
| Latence maximale | 5 073,8 ms |

Les labels sont `PROVISIONAL`. Le 100 % est donc une validation d'intégrité sur
ce corpus figé, pas une garantie statistique de production.

## Les cinq pertes récupérées

| Service | Bon SIRET | Nouveau rang | Cause corrigée |
|---|---|---:|---|
| `AC004970` | `35404764900018` | 1 | preuve SIRET/SIREN d'entrée |
| `FR035590` | `42094822600097` | 2 | candidat actif auparavant supprimé |
| `FR036649` | `92271723600017` | 1 | candidat actif auparavant supprimé |
| `AC012610` | `39771088000022` | 3 | preuve SIRET/SIREN d'entrée |
| `AC017507` | `81535085500012` | 1 | preuve SIRET/SIREN d'entrée |

Les trois premiers gains de B étaient déjà visibles dans l'autopsie. Les deux
autres confirment le défaut précis : le sparse trouvait le bon SIRET, puis la
barrière finale le supprimait faute de ligne dans le magasin rapide.

## Latence

La latence est mesurée sur les 242 cas exacts de l'audit, qui surreprésentent
volontairement des scènes difficiles. Elle n'est pas directement comparable à
la latence moyenne du shadow.

Le p95 de 2,88 secondes n'était pas un gate de ce correctif et ne bloque pas la
constitution de labels. Avant une nouvelle exécution shadow complète, il
faudra toutefois mesurer séparément le coût :

- du TF-IDF sur les grandes communes ;
- de l'expansion SIREN de B ;
- de la lecture d'état dans le snapshot complet.

Si la lecture d'état devient dominante, un index matérialisé `SIRET → état`
sur le SSD pourra être construit sans modifier le résultat fonctionnel.

## Invariants

- population de 242 cas inchangée ;
- 91 cas exacts du tirage aléatoire inchangés ;
- hashes des entrées contrôlés contre les manifestes gelés ;
- même snapshot SIRENE que celui des preuves aveugles ;
- vérité jamais transmise au retrieval ;
- variante B préexistante, sans réglage après résultat ;
- 100 candidats est un plafond absolu ;
- aucune consultation d'un test ou holdout ;
- 218 tests passent.

## Décision

Tous les gates du contrat
`docs/v4_2_retrieval_integrity_contract.md` passent.

**`GO_HARD_LABELS`** signifie :

- le retrieval n'est plus le blocage immédiat ;
- aucun nouveau modèle ne doit encore être entraîné ;
- la prochaine matière de travail est un corpus de cas difficiles réellement
  représentatifs, notamment les AUTO non résolus, les adresses partagées, les
  équipements publics et les groupes multi-sites.

Le statut production reste **`STOP_DEPLOYMENT`**, car l'audit précédent a
documenté cinq décisions AUTO manifestement incompatibles.

## Artefact

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_2_retrieval_integrity_7c4b957`

## Provenance Git

- contrat : `c33d3e0` ;
- source d'état autoritaire : `48ed90b` ;
- évaluateur figé : `7c4b957`.
