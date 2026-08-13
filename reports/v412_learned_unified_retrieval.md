# V4.12-L — retrieval unifié à 100 candidats

## Verdict

**`GO_RANKER_TRAINING`** sur le développement consommé. Il ne s'agit pas d'une
nouvelle certification indépendante.

Artefact autoritatif :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/evaluations/v4_12_learned_unified_retrieval/cce1bc83f82a1c3f`

| Vue | Couverture | Recall@100 | Succès / exacts |
|---|---:|---:|---:|
| Historique | 100 % | 96,106 % | 16 390 / 17 054 |
| V2 exact | 92,958 % | 96,423 % | 15 286 / 15 853 |
| V3 exact | 80,087 % | 99,268 % | 13 558 / 13 658 |
| V4.12-L corrigé | **80,154 %** | **99,270 %** | **13 604 / 13 704** |

Les trois gates sont franchis : couverture au moins 80 %, Recall@100 au moins
99 % et plafond absolu de 100 candidats. Aucun positif n'a été injecté.

## Segments V4.12-L

| Segment | Exacts | Recall@100 |
|---|---:|---:|
| Actifs | 11 619 | 99,552 % |
| Fermés | 2 085 | 97,698 % |
| Mégapoles | 938 | 98,827 % |
| Multi-sites | 3 019 | 98,874 % |
| Ajouts frais audités | 33 | 100 % |

Les cinq plis OOF groupés par composante SIREN obtiennent respectivement
99,356 %, 99,171 %, 99,158 %, 99,270 % et 99,399 %. Le retrieval ne masque donc
pas un pli structurellement défaillant.

Cent requêtes exactes restent absentes du top 100 : 52 actives et 48 fermées.
Le gate global autorise l'apprentissage du ranker, mais ne justifie pas de
promouvoir les établissements fermés dans le chemin opérationnel. Ils restent
des exemples auxiliaires d'identité, pondérés à 0,5 ; la production préfère les
établissements actifs.

## Replay des 43 ajouts frais

Les anciens pools V4.11 divergent pour les 43 requêtes de la politique
sélective gelée ; ils n'ont donc pas été mélangés à la mesure. Les canaux actif
et overlay ont été rejoués localement avec une profondeur interne de 5 000,
puis fusionnés par la politique gelée et tronqués à 100. Les 33 dossiers frais
`MATCH_EXACT` sont tous retrouvés. Les dix labels ouverts sont conservés dans
les pools pour le futur accepteur mais exclus du Recall.

## Suite autorisée

Matérialiser les lignes candidat et leurs features pour ces pools, puis
entraîner le ranker en cinq plis OOF. Les signaux métier précédemment codés en
règles deviennent des features ; aucune règle de promotion déterministe ne
sera appliquée à la sortie.
