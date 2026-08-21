# Retrieval open data v5 — résultat oracle

Date : 21 août 2026

Fold ouvert : développement `0` seulement. Fold `1` : **fermé**.
Verdict : **`PIVOT_ORACLE_BELOW_99_5`**.

## Résultat

La population développement contient 7 009 scènes, dont 6 307 SIRET exacts
identifiables sans utiliser le retrieval, soit 89,984 % de couverture.

Le socle temporel contenait 6 231 vérités. Sur ses 76 échecs, la recette v5
retrouve 28 vérités avec les canaux officiels et le rescue contrastif gelé en
retrouve 2. Leur union contient donc :

```text
6 231 + 30 = 6 261 / 6 307 = 99,2707 %
```

Le seuil v5 préenregistré à 99,5 % exige 6 276 vérités. Il en manque 15.
Le gain par rapport au plafond v4 publié de 6 258 est de trois vérités.
L'oracle étant insuffisant, LambdaMART n'a pas été entraîné et le top 100
final n'a pas été présenté comme validé. Ranker, decider, risk et fusion
restent gelés.

## Éléments effectivement ajoutés

- expansion consensus SIREN label-blind : au moins deux SIRET distincts
  récupérés pour un parent, puis déroulage borné de ses sites locaux ;
- répartition round-robin des 700 slots consensus entre les parents, pour
  empêcher un gros SIREN d'épuiser le budget ;
- composition de plusieurs overlays RNE/BODACC et de plusieurs référentiels
  sectoriels sans reconstruire l'index national de base ;
- projection complémentaire des preuves RNE/BODACC omises par le portefeuille
  actuel : 230 501 documents (88 528 preuves RNE, 101 017 BODACC et relations
  structurées BODACC), index de 351 Mo ;
- Data ES structurée : 498 020 preuves, 41 062 SIRET et 41 062 documents dans
  un index de 192 Mo ;
- adaptateur Associations de l'État implémenté, mais aucune collecte partielle
  orientée par les labels : faute de snapshot national structuré SIRET/SIREN,
  ce canal n'a pas été matérialisé.

La couche temporelle complémentaire ne porte l'identifiant d'aucune vérité
nouvellement récupérée. Neuf des vérités récupérées existent dans Data ES,
mais cette présence ne constitue pas à elle seule une causalité puisque les
référentiels sectoriels précédents restent simultanément actifs.

## Audit des 46 absents restants

- 31 établissements actifs et 15 fermés ;
- 41/46 n'ont aucun token de nom commun entre CRM et dossier officiel courant ;
- 29/46 n'ont aucun token d'adresse commun ;
- 27/46 n'ont ni token de nom ni token d'adresse commun ;
- seulement 8/46 ont encore un autre candidat du bon SIREN dans l'union ;
- 46/46 sont pourtant dans la commune INSEE attendue.

Les cas dominants ne sont donc plus des fautes lexicales ordinaires. Ils
portent des relations marque-opérateur, équipement-collectivité,
service-organisme ou une adresse devenue obsolète : `FNAC DARTY` face à
`ALESIA`, `CLEOR` face à `CGE BIJOUTERIE`, `CNAM` face à une CPAM, une piscine
face à l'agglomération, etc. Le snapshot officiel courant et les historiques
RNE/BODACC disponibles ne contiennent souvent pas cette relation.

## Latence

Sur la passe bornée des 76 échecs, la latence end-to-end vaut p50 1,018 s,
p95 3,227 s et p99 9,199 s. Elle échoue aussi les seuils 1 s / 2 s. Cette
mesure concentre volontairement les cas les plus difficiles et n'est pas une
estimation de production sur l'ensemble des 7 009 scènes, mais elle prouve que
les gros overlays doivent être préfiltrés davantage avant toute promotion.

## Artefacts

- rapport scellé :
  `/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/retrieval_open_data_v5/oracle_report_v5.json` ;
- union des 76 échecs : SHA-256
  `f2bdc9da035586e36daf3128b55bcfc3560524093d589f920f817ae39ecc3c8b` ;
- audit des 46 absents : SHA-256
  `07ffadc18ed4b4643f70a6f8d842ca3778fd39bfafefed5fefffe001bc7906b6` ;
- overlay temporel RNE/BODACC :
  `dossier_temporal_tail_v1/66f38cc146c51da6` ;
- overlay Data ES : `official_sector_data_es_v1/37ea905013a83660`.

Aucune vérité n'a été injectée, les données synthétiques n'ont pas été
utilisées, et le fold 1 n'a pas été lu.

## Décision

La recette v5 est conservée comme meilleure génération officielle, mais elle
n'est pas promue. La prochaine source open data doit réellement créer
l'information manquante — par exemple un rattachement structuré marque-titulaire
ou équipement-gestionnaire — et non augmenter encore la profondeur lexicale.
Toute nouvelle source doit être construite indépendamment des labels puis
franchir l'oracle avant le moindre entraînement d'admission.
