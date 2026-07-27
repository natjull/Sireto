# Contrat V4.2 — intégrité du retrieval représentatif

Statut : préenregistré avant modification du retrieval et avant nouvelle
évaluation.

## Objectif

Corriger les deux défauts déterministes démontrés par l'audit V4.1 :

1. la variante A n'exploite pas le SIRET/SIREN d'entrée ;
2. la barrière finale considère à tort qu'un SIRET absent du magasin rapide
   de candidats n'est pas actif.

La V4.2 doit conserver le bon SIRET dans au moins 99,0 % des cas
`MATCH_EXACT` provisoires de l'audit représentatif, avec 100 candidats au
maximum.

## Population figée

L'évaluation utilise exclusivement les 242 labels `MATCH_EXACT` déjà publiés
dans :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative_evidence/e696f22d68c0210f/provisional_adjudications.parquet`

Les entrées CRM viennent de l'échantillon aveugle déjà publié :

`/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/v4_1_representative/e06cf0d79849aad4/blind_cases.parquet`

Aucun cas, label ou sous-groupe ne peut être ajouté ou retiré après le calcul.
Les labels restent `PROVISIONAL` : cette expérience mesure l'intégrité
technique du retrieval, pas la précision production.

## Modification autorisée

- utiliser exclusivement la variante B déjà implémentée ;
- ajouter une source autoritaire et complète `SIRET → état administratif`
  construite ou lue depuis le snapshot SIRENE épinglé ;
- faire prévaloir cet état sur celui du magasin rapide incomplet ;
- conserver le magasin rapide pour l'hydratation des détails et l'expansion
  des sites d'un SIREN ;
- ajouter la provenance et le hash de la source d'état au manifeste.

Sont interdits pendant ce milestone :

- modification du sparse retrieval, de RRF ou du budget ;
- nouveau canal dense, web ou LLM ;
- injection du positif ;
- modification ou entraînement du ranker ;
- modification ou entraînement de l'accepteur ;
- réglage d'un seuil après lecture des résultats.

## Source d'état

Le snapshot autoritaire est :

`data/StockEtablissement_utf8.parquet`

Il contient 42 322 035 établissements. La source d'état doit :

- lire `siret` et `etatAdministratifEtablissement` ;
- normaliser uniquement le format, sans inventer de SIRET ;
- distinguer `A`, `F` et l'absence ;
- répondre par lot ;
- être utilisable en lecture seule sur le Mac et le SSD externe ;
- être liée au hash du snapshot dans tout artefact d'évaluation.

Une absence dans le magasin rapide ne peut plus éliminer un candidat déclaré
actif par cette source. Une fermeture déclarée par le snapshot doit toujours
éliminer le candidat, même si une partition le marque actif.

## Mesures

Publier :

- Recall@100 SIRET exact sur 242 cas ;
- résultat séparé sur les 91 cas exacts du tirage aléatoire ;
- nombre maximal de candidats ;
- nombre de candidats fermés dans les sorties ;
- vérités absentes de la source d'état ;
- détail de chaque miss ;
- latence p50 et p95 ;
- hashes des entrées, du snapshot et des sorties.

La baseline figée à rappeler est :

- variante A : 237/242 = 97,934 % ;
- variante B avec magasin incomplet : 240/242 = 99,174 %.

## Gates

`GO_HARD_LABELS` si toutes les conditions sont satisfaites :

- Recall@100 exact supérieur ou égal à 99,0 % ;
- aucun pool supérieur à 100 ;
- aucun candidat final fermé ;
- aucune vérité injectée ;
- aucune vérité exacte absente de la source d'état ;
- aucune régression sur les 237 cas déjà retrouvés par A.

`PIVOT` si la cible de recall échoue mais qu'une cause technique locale et
réparable est démontrée.

`STOP` si le retrieval ne peut pas atteindre la cible sans nouvelle
architecture, dépense externe ou utilisation d'une vérité comme feature.

Un `GO_HARD_LABELS` autorise uniquement la constitution de labels
représentatifs difficiles. Il n'autorise ni réentraînement immédiat ni
déploiement.

## Livrables

- implémentation et tests de la source d'état complète ;
- artefact d'évaluation immuable ;
- rapport `reports/v9/v4_2_retrieval_integrity_results.md` ;
- mise à jour de `handover.md` ;
- commits isolés cités dans le handover.
