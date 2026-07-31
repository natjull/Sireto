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

La contamination ne ressemble plus à quelques erreurs ponctuelles. L'audit de
provenance distingue finalement deux causes qui ne doivent pas être confondues :

- l'ancien benchmark V6A reconstruisait `query_id` à partir de la position des
  lignes après filtrage ; ses SIRET devenus étrangers au CRM proviennent bien
  d'identifiants positionnels instables ;
- V4.1 ne présente pas ce bug de jointure : il relie le CRM par
  `crm_record_id`/`SERVICE ID` unique. En revanche, sa qualification mécanique
  transforme toute scène comportant plusieurs correspondances directes actives
  en `AMBIGUOUS`, même lorsqu'une preuve métier permet d'identifier exactement
  le SIRET.

Sur les 56 vérités exactes du lot, 50 sont égales au SIRET CRM d'entrée et 51
étaient déjà le top 1 du ranker. Parmi les 50 cas exacts que V4.1 avait marqués
`AMBIGUOUS`, 46 conservaient en réalité le bon SIRET historique et 49 avaient
la vérité exacte dans la liste des correspondances directes. Le défaut dominant
est donc une politique de label trop prudente, non un défaut du retrieval.

## Suite

Le diagnostic de provenance est désormais terminé. Les 56 labels exacts et les
quatre ambiguïtés peuvent être utilisés comme overlay de développement. Le
prochain essai autorisé est un réentraînement local hors échantillon du ranker
et de l'accepteur avec ces corrections, en excluant ces 60 dossiers du choix de
seuil et de la comparaison classique. Le test final reste fermé.
