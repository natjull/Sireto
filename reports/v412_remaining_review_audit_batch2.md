# V4.12 — second lot complémentaire de 30 REVIEW

## Bilan métier

| Résultat | Nombre |
|---|---:|
| `MATCH_EXACT` fiable | **29** |
| `AMBIGUOUS` | **1** |
| `UNRESOLVED` | **0** |
| Top 1 ranker exacts parmi les 29 identifiables | **27** |
| Anciennes qualifications à corriger | **27 / 30** |

Les preuves et résultats ligne par ligne sont enregistrés dans
[`v412_remaining_review_audit_batch2.csv`](v412_remaining_review_audit_batch2.csv).

## Les deux erreurs réelles du ranker

- `7373` — **UIMM** : le SIRET officiel de l'UIMM Alsace est
  `78931279000015`; le ranker retient une autre entité juridique.
- `7749` — **Actérim Douai** : le SIRET exact est `87895687900026`; l'ancienne
  vérité et le top 1 du pipeline sont étrangers au CRM.

Les 27 autres dossiers exacts avaient déjà le bon SIRET en tête mais étaient
envoyés en REVIEW. L'unique ambiguïté réelle est Arthaud & Associés : trois
entités actives `Lyon`, `Audit` et `Group` partagent l'adresse, tandis que le
CRM ne contient aucun qualificatif ni date.

## Diagnostic consolidé après 60 nouveaux REVIEW

En ajoutant le premier lot complémentaire :

- **56 `MATCH_EXACT`**, quatre `AMBIGUOUS`, zéro `UNRESOLVED` ;
- **53 anciennes qualifications sur 60** doivent être corrigées ;
- le ranker avait le bon top 1 dans **51/56** cas exacts ;
- cinq erreurs relèvent réellement du ranker ;
- aucune absence générale du retrieval n'est mise en évidence.

La contamination ne ressemble plus à quelques erreurs ponctuelles. Plusieurs
anciens SIRET sont sans rapport avec le CRM par le nom, l'adresse et parfois le
département. L'hypothèse prioritaire devient une ancienne jointure ou un
réindexage décalé lors de la fabrication des vérités historiques.

## Suite

Avant de relire mécaniquement tous les dossiers restants, il faut auditer la
construction des labels historiques afin de déterminer si le décalage suit une
règle reproductible. Si une erreur de jointure est retrouvée, elle permettra de
corriger le corpus entier de manière traçable ; sinon l'adjudication dossier par
dossier continue. Aucun réentraînement n'est autorisé avant ce diagnostic.
