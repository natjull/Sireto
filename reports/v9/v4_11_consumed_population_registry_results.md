# V4.11-A — registre des populations CRM consommées

## Verdict

**`PASS_REGISTRY`**

Le registre canonique est construit sous :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/registries/v4_11_consumed_population/fd25d1922040d585`

Il ne mesure aucune performance modèle. Il fixe seulement la frontière entre
les dossiers déjà utilisés et ceux qui peuvent encore être considérés comme
inédits.

## Résultat

| Population | Lignes source |
|---|---:|
| CRM source | 23 609 |
| benchmark fermé historique | 17 054 |
| pool V4-Fresh complet | 6 330 |
| recouvrement entre les deux populations | 0 |
| union consommée | 23 384 |
| lignes encore inédites | 225 |

Les 225 lignes inédites ont toutes un `SERVICE ID` absent. Elles constituent
donc un résidu particulier, et non un échantillon représentatif du CRM.

Les artefacts V4.1 à V4.10 audités sont des sous-ensembles des deux
populations primaires. Ils n'ajoutent aucune ligne source à l'union.

## Méthode

Chaque ligne du CRM reçoit :

- un numéro de ligne source ;
- un identifiant de service normalisé lorsqu'il existe ;
- un SIRET CRM normalisé uniquement comme clé de lignée ;
- une empreinte SHA-256 canonique des huit champs ;
- une clé source unique.

L'exclusion utilise l'union des identifiants de service et des SIRET source
des deux populations épinglées. Le SIRET CRM ne devient ni une vérité terrain,
ni une feature modèle, ni une preuve suffisante de matching.

## Intégrité

- 23 609 clés source uniques ;
- 23 609 empreintes uniques ;
- aucune ligne perdue ;
- zéro recouvrement fermé/V4-Fresh ;
- hashes des trois parquets recomputés conformes au manifeste ;
- 386 tests passants.

Hashes de sortie :

| Fichier | SHA-256 |
|---|---|
| `source_registry.parquet` | `3fda773e3712b53aad017c2380471452c91e63fdf8a127a1fa09a46e8575e28b` |
| `consumed.parquet` | `bad97c3769a621a6a32b4c27ce1a0b8c15cd1f3877f2718ab0b3ab6c8759fe32` |
| `unseen.parquet` | `63ff648f6e326721e0646b0101de079f9a6feadb6e02c0474066c1288d8025a3` |

## Conséquence

V4.11 peut poursuivre son développement aligné sur les données déjà
consommées, puisqu'elles sont explicitement marquées comme développement.
Après gel du candidat, les 225 lignes peuvent servir à un challenge
descriptif autonome.

Elles ne suffisent pas à prouver la North Star à 99,8 %. Une validation
représentative exige un nouvel export CRM indépendant des 23 609 lignes
actuelles.

