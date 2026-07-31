# V4.12 — audit métier aveugle B2

Les 30 dossiers ont été adjudiqués sans consulter la prédiction, le score du
ranker, le score de l'accepteur ni l'ancienne vérité terrain.

| Verdict métier | Nombre |
|---|---:|
| `MATCH_EXACT` actif courant | **28** |
| `AMBIGUOUS` | **2** |
| `UNRESOLVED` | **0** |
| Fiabilité `HIGH` | **26** |
| Fiabilité `MEDIUM` | **4** |

Les deux ambiguïtés sont intrinsèques aux informations CRM :

- `AFPA Beziers` : deux personnes morales AFPA possèdent un établissement
  actif au 34 rue de Costesèque ;
- `AFT SERVICES` : plusieurs établissements AFT/AFTRAL actifs sont
  co-localisés, et le libellé historique ne permet pas de choisir.

Les principales causes de difficulté observées sont les transferts de SIRET,
les antennes de groupes multisites, les entités co-localisées, les holdings
portant un nom voisin et les adresses CRM devenues anciennes.

Le tableau ligne par ligne, avec SIRET, fiabilité et preuve consultée, est dans
[`v412_clean_target_independent_b2_labels_30.csv`](v412_clean_target_independent_b2_labels_30.csv).

Ces labels ont été scellés avant l'ouverture des scores du candidat
clean-target. Aucun modèle n'a été réentraîné et le test final reste fermé.
