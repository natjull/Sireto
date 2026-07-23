# V9 — Baseline retrieval sparse-50 gelée

## Identité

- benchmark : `c33b80855f560074`, split `test` ;
- commit d’exécution : `4e82530ee62edc4e7dfa1e7b8c7b9e897232174a` ;
- 2 652 requêtes, 2 130 SIREN, zéro SIREN partagé avec train/dev ;
- 500 candidats maximum par sous-canal sparse, fusion RRF et sortie finale
  limitée à 50 ;
- aucune réinjection de vérité terrain ;
- artefacts bruts :
  `/Volumes/CATNAT_DATA/SIRETO_V9/experiments/sparse50_c33b80855f560074_4e82530/`.

## Résultats

| Mesure | Résultat |
|---|---:|
| Recall@50 SIRET | 2 348 / 2 652 = **88,54 %** |
| IC Wilson 95 % | 87,27–89,69 % |
| IC Wilson 99 % | 86,85–90,04 % |
| Recall@50 SIREN | 2 444 / 2 652 = **92,16 %** |
| Vérité présente dans le pool géographique | 2 599 / 2 652 = **98,00 %** |
| Violations du budget | **0** |
| Latence p50 / p95 / p99 | 186 ms / 1 808 ms / 4 270 ms |
| Durée cold-cache | 1 340 s |

Les 304 erreurs SIRET se décomposent en :

- 53 vérités absentes de la partition, toutes sur des établissements fermés ;
- 251 vérités présentes géographiquement mais éliminées par le sparse/RRF ;
- parmi l’ensemble des erreurs, 96 scènes contiennent tout de même le bon
  SIREN mais pas le bon SIRET.

## Segments

| Segment | n | Recall@50 SIRET |
|---|---:|---:|
| Établissement actif | 2 181 | **93,17 %** |
| Établissement fermé | 471 | **67,09 %** |
| Mégapole, pool > 100 000 | 174 | **77,01 %** |
| Hors mégapole | 2 478 | **89,35 %** |
| Multi-site dans le benchmark | 716 | **89,39 %** |
| Localisation INSEE | 2 611 | **88,63 %** |
| Localisation CP seul | 41 | **82,93 %** |

## Lecture

Cette mesure invalide l’usage des anciens recalls de spikes comme baseline V9 :
ils reposaient sur des corpus plus petits et des protocoles différents. Elle ne
condamne pas encore l’architecture hybride, car 251 erreurs restent
récupérables dans le pool géographique et constituent précisément la cible du
canal dense.

Le modèle `siret-bert-deploy` ne peut pas servir à une revendication finale sur
ce benchmark : ses données d’entraînement ou de sélection couvrent tous les
SIREN du corpus historique. L’ablation dense propre utilisera d’abord la
révision locale épinglée du modèle multilingue générique, sans fine-tuning
SIRETO.
