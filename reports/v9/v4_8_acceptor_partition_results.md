# V4.8 — gel des partitions de faisabilité accepteur

Date : 28 juillet 2026  
Statut : partitions gelées, aucun modèle scoré ou entraîné.

## Résultat

Le protocole V4.8 a été préenregistré puis renforcé après une relecture
indépendante, avant toute exécution modèle. Le constructeur a produit
l'artefact immuable :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_8_acceptor_partitions/1c78764d5263afca`

Le retrieval V4.2-B, le ranker A, les top-1 et les 80 features restent gelés.
Le test final reste fermé.

## Comptes gelés

| Population / rôle | Nombre |
|---|---:|
| scènes historiques | 7 003 |
| fit historique initial | 5 547 |
| fit historiquement éligible | 5 545 |
| dev historique initial | 1 456 |
| cas courants | 172 |
| cas random scellés | 57 |
| cas ciblés fiables `hard_oof` | 94 |
| dont `TOP1_CORRECT` | 68 |
| dont `TOP1_WRONG` | 25 |
| dont `AMBIGUOUS` | 1 |
| ciblés fiables `hard_dev_locked` | 4 |
| ciblés non résolus hors random | 17 |

Les 94 cas `hard_oof` sont répartis par composante en cinq folds. Chacun
contient exactement cinq `TOP1_WRONG`. L'unique `AMBIGUOUS` est dans le fold
0. Les `hard_dev_locked` seront publiés descriptivement mais exclus du choix
des variantes et des gates ciblés.

## Effet des barrières anti-fuite

- Les 57 cas random sont tous dans `random_sealed`.
- Aucune de leurs cibles ni adjudications n'est présente dans
  `partition_assignments.parquet`.
- 48 scènes historiques reliées à une composante random sont exclues, dont
  44 fit éligibles et quatre dev.
- Vingt scènes fit historiques partagent une composante avec un cas
  `hard_oof`; elles suivront le fold de cette composante.
- Le dev historique effectif de comparaison contient 1 452 scènes.
- Les SIREN seulement présents dans les 100 candidats n'ont créé aucune
  arête.

Le graphe conserve chaque composante historique V4.1 comme atome et la relie
aux scènes courantes uniquement par les SIREN exacts d'entrée, de top-1 et,
lorsqu'il existe, du SIRET exact validé.

## Intégrité et hashes

| Artefact | SHA-256 |
|---|---|
| `manifest.json` | `f0e255b891dfb6b24d57f3b7423dd64a227908dbf68559b2da4572ea37791d33` |
| `partition_assignments.parquet` | `f828249172c36ce33a3279d294dfc5030e6d8eeb58baee9cf9e08130f13593b9` |
| `component_edges.parquet` | `fb949f64996659e4109fa3e28999e1411f74312e7576b2ed590a97293e5049cc` |
| `summary.json` | `a9178bcd14993a91cf22108bf53de62d500be938eddda93408b3341a23256e3a` |

Le manifeste atteste `model_loaded=false`,
`model_scoring_performed=false`, `random_targets_exposed=false` et
`test_opened=false`.

## Suite autorisée

La prochaine étape est l'expérience de développement V4.8 :

1. reproduire exactement le baseline gelé sur les 1 456 scènes dev
   originales ;
2. comparer `BASE_REFIT`, `HARD_W1`, `HARD_W2` et `HARD_W4` ;
3. produire des décisions réellement hors pli pour les 94 cas difficiles,
   avec un seuil propre à chaque modèle de fold ;
4. geler un winner uniquement s'il passe le gate historique et améliore au
   moins quatre erreurs difficiles ;
5. n'ouvrir la réserve random qu'après ce gel.

Commits de référence : contrat initial `f56472b`, identités épinglées
`1ca9648`, protocole renforcé `b63f383`, constructeur `6bb8518`, correctifs de
préflight `08018f9` et `eedac96`.
