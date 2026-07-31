# V4.12 — Docket gelé de validation indépendante du ranker

Date de gel : 31 juillet 2026  
Statut : **vérités non consultées au moment de ce commit**.

## Sélection préenregistrée

Population source : 196 dossiers `REVIEW` V4.12 qui n'appartiennent ni aux 30 adjudications R30, ni aux 53 adjudications R53.

Règle de sélection fixée avant toute recherche métier : conserver exhaustivement les dossiers pour lesquels le ranker baseline V4.12 et le candidat pondéré `0,5` ne donnent pas le même top 1. Cette règle produit exactement sept dossiers, tous inclus dans [`v412_ranker_independent_validation_docket.csv`](v412_ranker_independent_validation_docket.csv).

| Requête | CRM | Baseline | Candidat |
|---|---|---:|---:|
| `10395` | PROMOTRANS LYON | `77568013501063` | `77568013501071` |
| `1495` | PHB CREATION | `53465464500025` | `90370122500018` |
| `3165` | Centre Hospitalier Dufresne Sommeil | `26740017400046` | `26740017400012` |
| `4522` | UNIVERSITE D ARTOIS | `19624401600255` | `19624401600016` |
| `5708` | ALMERYS | `88414992300010` | `43270163900069` |
| `fresh:FR031148` | ROCHA SA | `44415907300014` | `30202053200073` |
| `fresh:FR031197` | BIGGIE GROUP | `49283100300057` | `91359061800033` |

## Éléments gelés

- retrieval et features : V4.11/V4.12 build `ec4326ec57e4411d` / `b4b7fef24c5e7036` ;
- baseline : `ranker_reference.parquet` V4.12 ;
- candidat : artefact `bba02575366ebe80`, poids des nouveaux groupes `0,5` ;
- hash du modèle candidat : `45f8735382111ee3dc308926bd4883f2c71601cb9e30be72ebb76eba36fd62cd` ;
- plafond : 100 candidats ;
- aucune vérité réinjectée, aucun test final ouvert.

Après ce gel seulement, chaque dossier sera adjudiqué à partir du snapshot SIRENE local et de preuves externes traçables. Une vérité absente du pool comptera comme erreur end-to-end. Une ambiguïté ne sera attribuée ni au baseline ni au candidat.
