# V4.8 — ouverture unique de la réserve aléatoire

Date : 28 juillet 2026  
Verdict final V4.8 : **`STOP_RETRAIN`**

## Résultat

La réserve aléatoire a été ouverte une seule fois après gel complet de
`HARD_W1`, de son seuil et de ses partitions. Le gate échoue :

| Variante | AUTO | Corrects AUTO | Erreurs AUTO | Précision observée | Couverture |
|---|---:|---:|---:|---:|---:|
| `BASE_FROZEN` | 45/52 | 43 | 2 | 95,556 % | 86,538 % |
| `HARD_W1` | 47/52 | 44 | 3 | 93,617 % | 90,385 % |

`HARD_W1` automatise un bon cas de plus que le baseline, mais aussi une erreur
de plus. Il automatise deux des cinq `TOP1_WRONG` et l'unique `AMBIGUOUS`.
Les contraintes « zéro négatif AUTO » et « zéro erreur AUTO » échouent.

Les intervalles de Wilson à 95 % de la précision observée sont
82,84–97,81 % pour `HARD_W1` et 85,17–98,77 % pour le baseline gelé. Les
effectifs sont petits, mais les trois erreurs observées suffisent à invalider
le gate sans extrapolation statistique.

## Les trois erreurs du winner

| Dossier | Verdict réel | Score `HARD_W1` | Cause métier |
|---|---|---:|---|
| `008373b595622d22` | `AMBIGUOUS` | 0,992347 | le CRM mélange FAM et MAS d'un même complexe AFAPEI ; le top-1 courant représente un foyer/EAM alors que les preuves ne permettent pas un SIRET unique |
| `00ebcafaaa0a8bf5` | `TOP1_WRONG` | 0,980186 | le CRM demande une mairie ; le top-1 est l'école primaire voisine, du même SIREN communal et presque à la même adresse |
| `01d50f2a608bb3bb` | `TOP1_WRONG` | 0,999190 | le CRM demande l'école maternelle au 46 ; le top-1 courant est l'école primaire au 48 et correspond même au SIRET d'entrée erroné |

Ces scores très élevés montrent que le défaut n'est pas un simple seuil mal
placé. Les 80 features reconnaissent très bien le nom général, l'adresse et le
SIREN, mais ne représentent pas assez la **fonction exacte du site** :
mairie contre école, maternelle contre primaire, FAM contre MAS.

Il est interdit de relever le seuil après observation du random. Une telle
correction serait de plus peu crédible : les trois erreurs ont des scores
compris entre 0,98 et 0,999.

## Intégrité de l'ouverture

Le registre global canonique a été créé par `O_EXCL` avant toute lecture
sémantique et est en lecture seule :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_8_random_holdout/OPENING_LEDGER.json`

SHA-256 :
`0f09e84d05d9b7f790544d4505a6288345bf338a361dd59f8dfa69edf49bb117`.

Il interdit toute seconde ouverture, quel que soit le script ou le dossier de
sortie. Le statut terminal vaut `OPENED_ONCE_COMPLETED`, le test final est
resté fermé et les cinq random `UNRESOLVED` n'ont pas été scorés.

Artefact :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_8_random_holdout/f1ac35f4f7450b6a`

| Fichier | SHA-256 |
|---|---|
| `manifest.json` | `86aa84fe1084fadc416e151a2a46c12bd523aa66cc4d328e944e822ff78a6712` |
| `random_report.json` | `e253fee5a2263f102b2e905345d460f414c032301185ed78f6c83548818cd568` |
| `random_predictions.parquet` | `9d236f7d53c167a2a970c77fac9e34ef9540cd60418e917d5dd4fb468fcc6e79` |
| `random_opening_marker.json` | `dda4e936c89877406fb1382157c8428a8b6196ddad513182bea21e12006f3dd3` |
| `random_opening_status.json` | `329377f0ab24ea7af59a99de70d0be91928ee0f78c6c6c1ad3c0c8449c29221e` |

## Conclusion scientifique

V4.8 démontre deux choses distinctes :

1. enrichir la logistique avec des erreurs ciblées améliore fortement les cas
   proches du seuil ;
2. les 80 features actuelles restent aveugles à certaines confusions
   sémantiques entre établissements d'un même SIREN ou d'une même adresse.

Le winner V4.8 n'est pas promu et aucun shadow frais n'est autorisé sous ce
contrat. Le random consommé ne peut servir à régler un nouveau seuil ou à
choisir une nouvelle variante.

La suite raisonnable est un nouveau contrat V4.9, avec :

- hypothèse préenregistrée sur la fonction précise du site ;
- features déterministes dérivées des libellés CRM et des champs SIRENE
  existants, sans LLM en inférence ;
- entraînement possible sur les labels désormais connus ;
- nouvelle évaluation réellement fraîche, collectée après gel, sans
  réutiliser le random V4.8 comme validation.

Commits de référence : autorisation `b738ec5`, ouvreur irréversible
`685ebae`, correction du hash de préflight `ba4377f`.
