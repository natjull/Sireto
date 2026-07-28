# V4.12 — Résultat du lookup SIRENE indexé

## Verdict

`GO_V412_SNAPSHOT_LOOKUP`

Le snapshot SIRENE complet a bien été construit sous forme d'un index local,
interrogeable en lecture seule par lots strictement plafonnés à 100 SIRET.
L'intégrité, la parité sur les candidats et l'échantillon indépendant sont
conformes. Ce verdict autorise la construction du moteur requête par requête
et son benchmark apparié V4.11/V4.12-G.

Il ne certifie ni la latence d'inférence, ni la parité batch/service du pipe
complet, ni la production.

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
- l'échantillon indépendant de `10 000` SIRET reproduit le hash gelé
  `58c9700d2a1ed2bb433e4f7a25a845ba236d63cfe633dcd64f9156469777f945` ;
- le snapshot source n'a été scanné qu'une fois pour la référence.

Le contre-audit confirme également zéro écart de valeur sur son nouvel
échantillon indépendant de 10 000 SIRET.

## Incident de contre-audit corrigé

Le contrat demande de prendre les 10 000 premiers SIRET après tri par :

```text
(sha256("v412-lookup-parity:" + siret), siret)
```

puis de hasher les SIRET dans cet ordre, chacun suivi d'un véritable octet
LF.

Un premier contre-audit a conclu à tort
`STOP_V412_LOOKUP_PARITY` avec la valeur :

```text
72f43460bb0e5047186fb4226147f1bf3022ceb8692164e5a8c57d9432a54960
```

Ses trois commandes utilisaient toutes les deux caractères littéraux
antislash et `n`, soit un payload de `160 000` octets. Elles répétaient donc
la même erreur d'échappement et n'étaient pas indépendantes sur ce point.

La reproduction corrigée utilise un vrai LF après chacun des 10 000 SIRET,
soit exactement `150 000` octets. Elle donne :

```text
58c9700d2a1ed2bb433e4f7a25a845ba236d63cfe633dcd64f9156469777f945
```

Elle reproduit exactement le contrat, le plan, le code du builder et
l'artefact. Les trois premiers SIRET sont aussi identiques :
`94410569100017`, `92883024900019`, `53539062900017`.

Le `STOP` publié dans le commit `880e57c` est donc annulé par le présent
correctif. L'incident reste documenté. Il révèle néanmoins une faiblesse
réelle : le validateur officiel relit actuellement la déclaration du sample
sans refaire la sélection depuis le snapshot. Un contre-validateur
indépendant devra être ajouté avec un test distinguant explicitement le vrai
LF du texte littéral `\n`.

## Ressources

- durée de construction : `81,1504` secondes ;
- pic mémoire : `8 375 615 488` octets, soit `7,8004 Gio` ;
- plafond contractuel : `8 589 934 592` octets, soit `8 Gio`.

Le contrôle mémoire passe avec environ `204 Mio` de marge. Cette marge étroite
interdit d'assimiler le coût du builder au coût du futur service : le
benchmark d'inférence devra utiliser la base déjà construite, en lecture
seule, avec un worker persistant.

## Portée exacte

Le lookup retire le principal obstacle technique au calcul V4.12-G en
requête unitaire : il n'est plus nécessaire de rescanner les 42,3 millions
d'établissements pour hydrater jusqu'à 100 candidats.

La prochaine preuve reste à produire sur les `1 456` requêtes dev :

1. mêmes candidats, rangs, features, top-1, scène et décision entre batch et
   moteur requête par requête ;
2. mêmes preuves directes et même veto V4.12-G ;
3. latence p95 appariée V4.11/V4.12-G avec worker persistant ;
4. aucun cache manquant et aucune lecture de label.

Le test final reste fermé.
