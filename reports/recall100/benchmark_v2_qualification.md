# Qualification V2 du benchmark Recall@100

## Verdict

La séparation des labels structurellement douteux est utile, mais elle ne
résout pas le problème de retrieval.

Sur le dev historique :

| Périmètre | Succès | Recall@100 |
|---|---:|---:|
| historique, toutes les requêtes | 2 495 / 2 565 | 97,271 % |
| V2, SIRET exact encore évaluable | 2 343 / 2 400 | 97,625 % |
| oracle interne, périmètre V2 exact | 2 394 / 2 400 | 99,750 % |

Le gate V2 à 99 % exige 2 376 succès. Il manque encore **33 requêtes**.

Le nettoyage n'a donc pas créé artificiellement un passage à 99 %. Il confirme
que le signal nécessaire existe en amont, mais que la réduction à 100
candidats reste insuffisante.

## Qualification dev

Les 2 565 lignes sont toutes conservées :

- 2 400 `MATCH_EXACT`, incluses dans la métrique SIRET exacte ;
- 81 `AMBIGUOUS`, exclues de cette métrique et destinées à `REVIEW` ;
- 84 `UNRESOLVED`, exclues de cette métrique et destinées à `REVIEW`.

Les 81 ambiguïtés regroupent :

- 52 labels actifs avec un autre SIRET actif du même SIREN à l'adresse CRM ;
- 29 cas avec plusieurs SIRET actifs alternatifs à cette adresse.

Les 84 non-résolus ont un label fermé et un autre SIRET actif du même SIREN à
l'adresse CRM.

Aucun SIRET alternatif n'est promu automatiquement. Le SIRET historique reste
présent dans les colonnes de provenance.

## Qualification train

Sur 11 837 requêtes :

- 10 995 restent `MATCH_EXACT` ;
- 440 deviennent `AMBIGUOUS` ;
- 402 deviennent `UNRESOLVED`.

Ces 842 lignes ne doivent plus servir comme positifs SIRET exacts à un futur
apprentissage tant qu'elles ne sont pas adjudiquées.

## Pertes restantes sur le dev V2

L'admission à 100 commet encore 57 erreurs sur les 2 400 labels exacts :

- 6 SIRET ne sont vus par aucun canal interne jusqu'à 5 000 :
  `218`, `724`, `725`, `2021`, `11369`, `16995` ;
- 51 sont trouvés en amont puis éliminés lors de la réduction à 100.

Le premier chantier est un problème de sourcing, d'alias ou de localisation.
Le second reste un problème d'admission.

## Portée et limites

Cette V2 est une qualification mécanique rétrospective, pas une vérité terrain
certifiée. Les relations métier opaques non détectables par la seule structure
SIRENE peuvent encore subsister parmi les 2 400 lignes.

Les métriques historique et V2 doivent toujours être publiées ensemble. Le
test reste fermé et le benchmark original conserve son hash
`4c533813218dced6627da238b885db47e45745d784ae9078a4aaa836680308b6`.

## Artefacts

- dev :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v2/522351669d5313dc` ;
- train :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/benchmarks/qualification_v2/f8af7e1da18fa94a` ;
- audit train :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/site_label_audit_train_c33b80855f560074_a68f679`.

La politique est définie dans `docs/benchmark_v2_label_policy.md` et le builder
dans `scripts/build_benchmark_v2_qualification.py`.
