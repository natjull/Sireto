# V4.12 — Résultat du build de preuve directe

## Verdict

`GO_SEALED_EVIDENCE`

Le build label-free V4.12 a été exécuté une fois sous le verrou audité
`11c5de9`. Il ne constitue pas encore l'évaluation historique de la garde et
n'autorise aucune promotion.

## Artefact

- chemin :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/v4_12_direct_evidence/10f16403795ccee6`
- build id : `10f16403795ccee6`
- manifeste :
  `2006184308fc412af944b4752f8fd0dbbc9ea167943681238215b46ea5fbc12a`
- verrou :
  `f5833e8ed65d6f4bca9c2ce20fc0edee4e40f9e54e1c1dbda6d161dba0c8d5f8`
- politique : `active-direct-current-v4.0`

## Résultats label-free

| Mesure | Valeur |
|---|---:|
| Requêtes | 7 003 |
| Preuves candidates actives | 10 275 |
| Un seul candidat direct | 5 883 — 84,007 % |
| Plusieurs candidats directs | 1 120 — 15,993 % |
| Aucun candidat direct | 0 |
| Collision entre plusieurs SIREN | 977 — 13,951 % |
| Plusieurs sites du même SIREN | 143 — 2,042 % |
| Maximum de candidats directs | 277 |

Les 1 120 dossiers à preuves multiples se divisent exactement entre 977
collisions inter-SIREN et 143 cas multisites intra-SIREN. Ces agrégats
décrivent la population complète ; ils ne disent pas encore combien de
décisions `AUTO_MATCH` V4.11 seront refusées.

## Intégrité

- 7 003 `query_id` uniques ;
- aucune preuve candidate dupliquée par requête et SIRET ;
- références query/candidat bijectives ;
- état administratif actif pour les 10 275 candidats ;
- univers géographique actif complet ;
- aucun label, challenge, pool ranker, scène ou modèle ouvert avant le seal ;
- aucune modification du top-100 du retrieval ;
- les 4 175 fichiers d'entrée passent la denylist ;
- pic RSS : 2 991 915 008 octets, inférieur au plafond de 8 Gio ;
- validation officielle et contre-audit indépendant concluants.

Le p95 de 137,331 ms est un temps batch amorti. Il n'est pas une mesure de
latence de service et ne franchit aucun gate de latence.

## Suite autorisée

Construire, tester, auditer et verrouiller le runner post-seal qui :

1. reproduit les décisions V4.11 gelées ;
2. applique uniquement le veto V4.12-G ;
3. mesure une seule fois le gate `comparison_dev` :
   `A >= 600`, `E == 0`, `B == 0` ;
4. publie les pertes appariées par segment ;
5. conclut `GO`, `PIVOT` ou `STOP_V412_GUARD`.

Le challenge V4.11 consommé reste interdit et aucun résultat de ce build ne
certifie encore la North Star produit.
