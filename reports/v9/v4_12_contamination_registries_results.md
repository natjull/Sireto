# V4.12 — Registres de contamination pré-holdout

## Verdict

`GO_V412_CONTAMINATION_REGISTRIES`

Les deux registres requis avant l'observation d'un nouvel export CRM ont été
construits sur `/Volumes/CATNAT_DATA`, scellés et contrôlés indépendamment.
Aucun futur CRM, modèle, candidat ou résultat de retrieval n'a été ouvert.

## Registre des SIREN consommés

- Build ID : `fbc0b84d9c81b01a`
- Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/registries/v4_12_consumed_sirens/fbc0b84d9c81b01a`
- Manifest SHA-256 :
  `b220efd7c4dc89a980b9d0b5501e16fd286edcafdff61573ae6c5e8d8423c6ff`
- Payload logique :
  `65015c7d82a5765cb304137e82f132ab61b04319330566704eec29a565e2348d`
- `consumed_sirens.parquet` :
  `9d440612a2560fe1a63e7ff5b3b196d33ad615c1d09720dfe05b3d70969851de`
- Volumes : 64 618 observations, 19 754 SIREN uniques, zéro rejet.

Le build ferme uniquement les identités autoritatives déjà consommées. Les
candidats, prédictions, rangs, scores et sondes techniques restent exclus.

## Registre de compatibilité des 23 609 anciennes lignes

- Build ID :
  `48851668dd2f173686f3240ecc62e30fcbfdb96d8abf0ced498eb29891d8a490`
- Artefact :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake/registry/consumed_compatibility/48851668dd2f173686f3240ecc62e30fcbfdb96d8abf0ced498eb29891d8a490`
- Payload manifest :
  `6435f071c027b7458a8c8b3f09dce796cadc8a0fdc03e0413bcecba1fe969f9b`
- Seal :
  `2068a5d18aac189b7bffc0515054fa31166cb5cd9e4d066f143d3c2d5bc3e976`
- Payload logique :
  `2557a27ff9f225649d0746ea236ec405be211c90dbae91bb4ccff785243d5ab3`
- Verrou d'exécution :
  `89e06772ac57c72738ca6bb7cde2c42ce5ee57ac2f5075a1c6e95cdee97ee115`
- Tête de chaîne d'événements :
  `93fcd8850ba9f0d38a11cfe9358cd692f771ebdb2c106938c8fa0d5719e7953d`

Les contrôles indépendants ont recomputé la parité des 23 609 lignes, dont
23 384 historiques et 225 du challenge, les quatre keysets, leurs
multiplicités, les schémas Parquet, la provenance, les permissions et les
11 fichiers des arbres principal et de reproduction. Résultat : zéro écart
et zéro rejet.

La clé HMAC reste dans le Keychain macOS. Le builder la lit avec
`SecItemCopyMatching` dans le même processus, sans UI, argument, variable
d'environnement, fichier temporaire, log ou sortie. Seuls l'identifiant
logique et le hash de la clé sont scellés.

## Tentative arrêtée conservée

La première tentative de compatibilité,
`v412-compat-8c4f31ce-attempt-01`, a été arrêtée avant validation des entrées
car le BOM UTF-8 réel du CSV historique n'était pas contractualisé. Elle
contient uniquement le receipt et `ATTEMPT_RECEIPTED`, sans payload ni arbre
de build. Elle reste immuable comme trace d'audit.

La correction exige exactement un BOM `EF BB BF` au début du fichier, refuse
un second BOM initial et conserve un éventuel `U+FEFF` situé dans une valeur
CRM. Le CSV et ses hashes n'ont pas été modifiés.

## Commits de fermeture

- SIREN : code/tests `3b66fd7`, contrat/plan `9f74c00`, cross-pin intake
  `a20c704`.
- Compatibilité Mac et Keychain : `6de4585`, `4a5ac60`, `38b18d8`,
  `4b8bd2a`.
- Ancien verrou révoqué après le STOP BOM : `213a3b0`.
- Correction BOM : code/tests `47e9772`, contrat/plan `6f9ad7e`,
  cross-pin intake `63e45f1`, verrou final `5516ba6`.

## Autorisation suivante

Le gate autorise l'implémentation et les tests du scanner/sealer d'arrivée sur
des paquets exclusivement synthétiques. Il n'autorise pas encore l'ouverture
d'un nouvel export CRM, le test retrieval one-shot, ni le dégel du ranker ou
de l'accepteur.
