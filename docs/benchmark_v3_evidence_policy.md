# Politique V3 de preuve directe CRM → SIRET

## Objet

La V2 détecte les contradictions entre établissements d'un même SIREN. Elle
ne détecte pas un label historique dont le nom et l'adresse sont tous deux
sans rapport direct avec le CRM.

La V3 ajoute donc un audit de la preuve disponible entre la requête CRM et le
SIRET historique. Elle ne cherche pas à déterminer si une relation métier
opaque est vraie ou fausse. Elle répond à une question plus stricte :

> Les champs disponibles à l'inférence permettent-ils de défendre ce SIRET
> exact sans connaissance externe non versionnée ?

## Statut expérimental

Cette politique est définie après observation du dev historique. Son résultat
sur le dev est rétrospectif et ne constitue pas une validation aveugle du gate
à 99 %. Le test reste fermé.

La politique peut être appliquée au train pour préparer les apprentissages.
Une future certification exige une adjudication indépendante ou l'application
inchangée de la politique à un nouveau jeu jamais observé.

## Signaux

Les signaux utilisent les fonctions partagées de normalisation et de features
du pipeline, sans sémantique neuronale.

### Preuve de nom forte

La preuve est forte si au moins une condition est vraie :

- nom normalisé exactement égal à un nom SIRENE disponible ;
- Jaro-Winkler ≥ 0,85 et overlap de tokens ≥ 0,50 ;
- relation de contenu ou d'acronyme, avec Jaro-Winkler ≥ 0,75.

Les noms disponibles sont l'enseigne, la dénomination usuelle établissement,
les noms usuels, sigle et dénomination de l'unité légale, ainsi que le nom de
personne lorsqu'il est autorisé par la catégorie juridique.

### Preuve d'adresse forte

La preuve est forte si au moins une condition est vraie :

- clé canonique `numéro|voie` exactement égale ;
- code postal égal, similarité Jaro-Winkler de voie ≥ 0,90 et numéros égaux ;
- la condition précédente sans contrainte de numéro lorsque l'un des deux
  côtés ne possède pas de numéro exploitable.

## Classes de preuve

| Classe | Nom fort | Adresse forte |
|---|---:|---:|
| `NAME_AND_ADDRESS` | oui | oui |
| `NAME_ONLY` | oui | non |
| `ADDRESS_ONLY` | non | oui |
| `NO_DIRECT_EVIDENCE` | non | non |

Les trois premières classes restent évaluables comme `MATCH_EXACT`, sous
réserve des exclusions structurelles V2.

Une ligne V2 `MATCH_EXACT` classée `NO_DIRECT_EVIDENCE` devient
`UNRESOLVED` dans la vue V3. Son SIRET historique reste conservé comme
provenance. Elle pourra redevenir `MATCH_EXACT` uniquement via :

- une source externe versionnée établissant l'alias, l'exploitant ou le
  propriétaire ;
- une adjudication humaine traçable ;
- une correction du nom ou de l'adresse CRM.

Les labels déjà `AMBIGUOUS` ou `UNRESOLVED` en V2 ne sont jamais réouverts
automatiquement.

## Garde-fous

- la classe ne reçoit jamais le hit/miss du retrieval comme entrée ;
- aucun seuil n'est ajusté requête par requête ;
- aucun SIRET alternatif n'est créé ou promu ;
- toutes les lignes restent présentes dans l'artefact ;
- métriques historique, V2 et V3 sont publiées ensemble ;
- les volumes par classe de preuve sont obligatoires ;
- le routing des lignes ouvertes reste une métrique séparée, non mesurable par
  le retrieval seul.

## Limite connue

Cette politique sélectionne mécaniquement les relations soutenues par les
champs utilisés par le moteur. Elle mesure donc un périmètre « identifiable
avec les données disponibles », pas la totalité des relations commerciales
possibles.

Un bon résultat V3 ne prouve pas que le système sait résoudre les marques,
occupants, équipements publics ou relations de propriété absents de SIRENE.
Ces cas nécessitent un registre d'alias et de preuves séparé.
