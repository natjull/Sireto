# V4.12 — Résultat du lookup SIRENE indexé

## Verdict

`STOP_V412_LOOKUP_PARITY`

Le snapshot SIRENE complet a bien été construit sous forme d'un index local,
interrogeable en lecture seule par lots strictement plafonnés à 100 SIRET.
Cependant, un contre-audit indépendant ne reproduit pas le hash contractuel
de l'échantillon déterministe de 10 000 SIRET. Le contrat impose un arrêt au
premier écart.

L'artefact déclare `GO_V412_SNAPSHOT_LOOKUP`, mais cette déclaration est
invalidée. Il n'autorise ni la construction du moteur requête par requête, ni
le benchmark de latence, ni la production.

## Artefact

- chemin :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/indexes/v4_12_snapshot_lookup/ff0f33ad10803cfb`
- build id : `ff0f33ad10803cfb`
- manifeste :
  `04e098952ee4cc7957155623599d3ba35b95f9126932e5a5d420ec02b110b15e`
- verrou d'exécution :
  `a75db01dff045be627816780574bce7c906afbc9e357c17c024404582bcd26c9`
- base DuckDB :
  `5da123bb0dde06d55886dfbc5c36e142c9d528ffec1b6899022e5c7c63bee894`
  (`2 732 863 488` octets)

## Intégrité

| Contrôle | Résultat |
|---|---:|
| Lignes du snapshot indexées | 42 322 035 |
| SIRET uniques | 42 322 035 |
| SIRET invalides | 0 |
| Index unique | `candidate_details_siret_uidx` |
| Ouverture de contrôle | lecture seule |
| Taille maximale d'un appel | 100 SIRET |
| Labels ou challenges ouverts | non |

La table expose uniquement les sept champs gelés nécessaires à l'inférence :
SIRET, état administratif, trois enseignes, dénomination usuelle et code
d'activité.

## Contrôles conformes

- les `508 081` SIRET uniques des `698 892` candidats V4.11 ont été contrôlés
  après la construction ;
- la référence bulk contient `517 963` lignes ;
- zéro écart de présence, valeur, type ou nullité ;
- le snapshot source n'a été scanné qu'une fois pour la référence.

Le contre-audit confirme également zéro écart de valeur sur son nouvel
échantillon indépendant de 10 000 SIRET.

## Écart bloquant

Le contrat demande de prendre les 10 000 premiers SIRET après tri par :

```text
(sha256("v412-lookup-parity:" + siret), siret)
```

puis de hasher les SIRET dans cet ordre, chacun suivi de `\n`.

Trois recalculs indépendants, dont une reproduction exacte du CTE SQL du
builder, donnent :

```text
72f43460bb0e5047186fb4226147f1bf3022ceb8692164e5a8c57d9432a54960
```

Le contrat, le plan, le builder et l'artefact déclarent :

```text
58c9700d2a1ed2bb433e4f7a25a845ba236d63cfe633dcd64f9156469777f945
```

Le même ensemble retrié lexicalement produit encore une troisième valeur,
`399ff013f99f21cfe4bbd29b51c011a6c8cae5a6e2df65a49e9f801bbf7b163b`.
Il ne s'agit donc pas d'une simple confusion entre ordre aléatoire gelé et
ordre lexical.

Le validateur officiel passe à tort parce qu'il vérifie que le JSON contient
la valeur gelée, sans la recalculer depuis le snapshot. La cause exacte de la
valeur `58c970...` et la raison pour laquelle le builder l'a acceptée restent
à établir avant toute correction.

## Ressources

- durée de construction : `81,1504` secondes ;
- pic mémoire : `8 375 615 488` octets, soit `7,8004 Gio` ;
- plafond contractuel : `8 589 934 592` octets, soit `8 Gio`.

Le contrôle mémoire passe avec environ `204 Mio` de marge. Cette marge étroite
interdit d'assimiler le coût du builder au coût du futur service : le
benchmark d'inférence devra utiliser la base déjà construite, en lecture
seule, avec un worker persistant.

## Portée exacte

La base DuckDB paraît matériellement correcte, mais elle reste un artefact
refusé. Elle ne peut pas être recyclée silencieusement comme artefact gelé.

La prochaine étape obligatoire est :

1. expliquer et tester l'écart `58c970...` / `72f434...` ;
2. corriger le contrat, le calcul et le validateur avant un nouveau build ;
3. faire contre-auditer le correctif et produire un nouveau verrou ;
4. reconstruire et revalider sous un nouvel identifiant immuable ;
5. seulement après un vrai `GO`, ouvrir le chantier parité/latence sur les
   `1 456` requêtes dev.

Le test final reste fermé.
