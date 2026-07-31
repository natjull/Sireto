# V4.12 — Contre-audit métier des changements de top 1 (lot 1/53)

Date : 31 juillet 2026  
Périmètre : dix premiers dossiers `REVIEW` historiques dont la correction exploratoire changerait le top 1. Aucun réentraînement, aucun test final et aucune promotion produit.

## Résultat métier

| Requête | CRM | Top 1 actuel | Choix exploratoire | Adjudication | Fiabilité | Effet de la correction | Cause réelle |
|---|---|---:|---:|---:|---|---|---|
| `10319` | OLYMPIQUE LYONNAIS, 10 av. Simone Veil | `38507188100119` | `38507188100093` | `38507188100093` | élevée | correction | mauvais établissement du bon SIREN ; le CRM générique vise le siège |
| `10613` | IDEF 86, 189 rue de la Gibauderie | `26860086300115` | `26860086300016` | `26860086300115` | élevée | **régression** | le signal « siège » contredit ici le site opérationnel explicitement rattaché à l'adresse |
| `12061` | SPB, 71 quai Colbert | `82178484000034` | `30510977900077` | `30510977900077` | élevée | correction | confusion avec une société du groupe à la même adresse |
| `12496` | CARREFOUR MARKET, rue de l'Yser | `44028375206603` | `93263285400026` | `93263285400026` | élevée, état courant | correction | ancien exploitant contre exploitant courant du magasin |
| `13168` | SPK GROUP, 25 rue du Ranzay | `81952269900043` | `89106734000036` | `89106734000036` | élevée | correction | filiale `SPK` contre entité légale `SPK GROUP` |
| `13266` | GROUPE DELAMBRE, 4 bis rue des Chevaliers | `33000259300078` | `97799453200017` | `40535660100022` (fermé après transfert) | élevée sur l'identité ; date CRM inconnue | non exploitable par la correction | le SIRET exact correspondant à l'adresse historique est absent du pool de 100 |
| `13505` | MC3 LA MONTAGNE, 11 av. de la Libération | `44943203800015` | `80835946700020` | `80835946700020` | élevée | correction | société immobilière co-localisée contre établissement `MC3` |
| `13923` | SIX ARES, 11 rue La Fayette | `88119287600020` | `91464067700029` | `84239312600029` (fermé) | élevée sur l'identité ; date CRM inconnue | non exploitable | le SIRET exact de `SIX ARES`, radié après fusion en 2024, est absent du pool de 100 ; les deux choix sont des SCCV co-localisées |
| `13947` | AVOXA, 1 mail du Front Populaire | `42214714000020` | `53856370100019` | `53856370100019` | élevée | correction | structure rennaise/ancienne organisation contre entité nantaise exacte |
| `13958` | AXES - SYGMALAB, 2 bis rue Newton | `34882399800045` | `38884788100058` | `AMBIGUOUS` | élevée sur l'ambiguïté | neutre | deux sociétés distinctes, `SYGMALAB` et `AXES`, sont revendiquées ensemble par le site et co-localisées ; le CRM ne permet pas d'en choisir une |

## Preuves consultées

La première preuve est le snapshot SIRENE local V4.12 et le pool réellement présenté au ranker :

- `candidates_features.parquet` et `ranker_reference.parquet`, build `b4b7fef24c5e7036` ;
- identité CRM issue de `queries.parquet` du même build ;
- absence de `405356601` pour `13266` et de `84239312600029` pour `13923` vérifiée dans les 100 candidats réellement fournis.

Les preuves externes suivantes ont servi à distinguer les entités co-localisées et à contrôler l'état courant :

- OLYMPIQUE LYONNAIS : [registre courant, siège `38507188100093`](https://www.pappers.fr/entreprise/olympique-lyonnais-385071881) ;
- IDEF 86 : [site officiel, siège à l'adresse CRM](https://idef86.fr/contact/) et [avis de situation INSEE de `26860086300115`](https://api-avis-situation-sirene.insee.fr/identification/pdf/26860086300115) ;
- SPB : [mentions légales officielles, SIREN `305109779` et adresse CRM](https://www.spb.eu/fr/mentions-legales/) ;
- CARREFOUR MARKET : [Annuaire des Entreprises, SIRET courant `93263285400026`](https://annuaire-entreprises.data.gouv.fr/etablissement/93263285400026) et [page magasin Carrefour](https://www.carrefour.fr/magasin/market-tourcoing) ;
- SPK GROUP : [registre et établissement `89106734000036`](https://www.pappers.fr/entreprise/spk-group-891067340) et [site officiel à l'adresse CRM](https://www.spk-group.fr/fr/project/puma-future-event-2025/) ;
- GROUPE DELAMBRE : [statuts officiels publiés, nom/SIREN/adresse](https://www.pappers.fr/entreprise/groupe-delambre-405356601/documents/GROUPE%20DELAMBRE%20-%20Statuts%20mis%20%C3%A0%20jour%2016-03-2022.pdf) ;
- SIX ARES : [registre, SIRET exact et radiation/fusion](https://www.pappers.fr/entreprise/six-ares-842393126) ;
- AVOXA : [site officiel du cabinet nantais](https://www.avoxa.fr/avoxa-nantes/) et [registre de `53856370100019`](https://www.pappers.fr/entreprise/538563701) ;
- AXES / SYGMALAB : [site officiel présentant conjointement les deux marques à la même adresse](https://www.axes-44.com/).

## Bilan du lot

| Catégorie | Nombre |
|---|---:|
| Choix exploratoire exact et fiable | 6 |
| Top 1 initial exact ; changement régressif | 1 |
| Vérité exacte absente des 100 candidats | 2 |
| Ambiguïté métier réelle | 1 |
| Labels SIRET exacts utilisables pour apprendre ou évaluer | 9 |

La correction n'est donc pas une règle déployable : elle gagne six dossiers mais en dégrade déjà un. Deux cas sur dix révèlent surtout un défaut de retrieval, invisible dans le spike R30, et un cas ne possède pas de vérité SIRET unique. La suite doit séparer ces trois populations au lieu de les mélanger dans un futur entraînement.
