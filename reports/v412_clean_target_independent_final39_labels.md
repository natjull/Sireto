# V4.12 — audit métier aveugle des 39 derniers REVIEW

Les 39 dossiers ont été adjudiqués sans consulter les prédictions, les
scores ou les anciennes vérités terrain.

| Verdict métier | Nombre |
|---|---:|
| `MATCH_EXACT` actif courant | **33** |
| `AMBIGUOUS` | **6** |
| `UNRESOLVED` | **0** |
| Fiabilité `HIGH` | **35** |
| Fiabilité `MEDIUM` | **4** |

Les six ambiguïtés sont structurelles : Securitas, Burger King, BNP Paribas,
Sandvik, King Tony et Hyméo possèdent plusieurs personnes morales actives sous
une marque générique et à la même adresse. Le CRM ne contient pas l'information
qui permettrait de choisir un SIRET exact.

Les 33 autres dossiers fournissent des labels exacts utilisables, dont 29 à
fiabilité haute et quatre à fiabilité moyenne. Les preuves ligne par ligne sont
dans
[`v412_clean_target_independent_final39_labels.csv`](v412_clean_target_independent_final39_labels.csv).

Ces labels ont été scellés avant toute ouverture des scores. Aucun modèle n'a
été réentraîné et le test final reste fermé.
