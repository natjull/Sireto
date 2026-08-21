# Retrieval officiel v4 — résultat jusqu'au rescue contrastif

Date : 21 août 2026

Périmètre ouvert : folds 2/3/4 pour l'apprentissage, fold 0 pour le dev.

Fold 1 : **fermé**.
Verdict : **`PIVOT_ORACLE_BELOW_99_3`**.

## Résultat principal

Le précédent oracle temporel contenait le SIRET humain exact pour
6 231 des 6 307 scènes exactes identifiables, soit 98,795 %.

Les nouvelles preuves déterministes récupèrent 25 des 76 absents :

- expansion d'un SIRET direct vers son SIREN puis vers ses sites locaux ;
- successions SIRENE structurées ;
- FINESS, annuaire UAI et annuaire DINUM structurés ;
- rôles RNE dont le représentant est une personne morale.

Le rescue contrastif SIREN ajoute ensuite 2 vérités réellement nouvelles :

- `GLOBAL DISTRIBUTION`, parent SIREN au rang 1, site au rang 1 ;
- `COLLEGE SAINT LOUIS`, parent SIREN au rang 11, site au rang 2.

La passe exhaustive top-50 SIREN sur les 49 échecs restants n'ajoute aucun
autre hit. Le plafond final observé est donc :

```text
6 231 + 25 + 2 = 6 258 / 6 307 = 99,2231 %
```

Le gate préenregistré de 99,3 % demande au moins 6 263 hits. Il manque encore
5 vérités exactes. Aucune vérité n'a été injectée et les 20 000 lignes
synthétiques n'ont pas servi à l'apprentissage du retrieval.

## Sources nouvelles réellement valorisées

### Référentiels sectoriels

410 603 preuves structurées ont été matérialisées, puis 519 027 documents
géofiltrables ont été indexés :

- FINESS : SIREN et SIRET d'établissements sanitaires et médico-sociaux ;
- UAI : SIRET des établissements d'enseignement ;
- DINUM : SIREN des administrations.

Parquet : `official_sector_evidence.parquet`, SHA-256
`ce446da891b17e89b653d90abea00d55b94050d5f4699a35a53dc44f886a31a1`.

### RNE — personnes morales liées

27 348 473 formalités RNE ont été parcourues. L'extraction conserve
1 754 075 rôles de personnes morales, dont 1 640 743 disposent d'un SIREN lié
structuré. Les personnes physiques, bénéficiaires effectifs et adresses de
dirigeants sont exclus. L'overlay final contient 2 217 800 documents
`(commune, SIREN)`.

Parquet : `legal_entity_roles.parquet`, SHA-256
`e626cdf17395693411375d41c6848fcf6085d62ce3c66d08cf1388a0d42457b1`.

Cette source apporte 2 des 25 rescues officiels sur les 76 échecs. Elle est
donc utile comme canal borné, mais son coût de requête ne justifie pas de la
mettre au cœur du retrieval sans index/pruning supplémentaire.

## Étape 6 contrastive

Le rescue apprend une transformation de similarité, pas un dictionnaire
d'identités :

- 21 262 scènes humaines issues exclusivement des folds 2/3/4 ;
- 13 980 SIREN d'apprentissage distincts ;
- zéro SIREN partagé avec fold 0 ;
- MiniLM multilingue gelé ;
- adaptateur linéaire contrastif appris sur positif et hard negative ;
- aucun SIREN, SIRET ou identifiant d'annonce dans le texte ou les features ;
- perte triplet 0,0700 → 0,0416.

Manifeste modèle SHA-256 :
`17c4ee8207fa900b8e27f82e6dae8795a9c8c59849d13ed4f2e5c06850759703`.

Le canal est validé techniquement mais **non promu** : +2 vérités ne suffit
pas au gate, et l'encodage à la volée des mégapoles est trop lent. Une version
production exigerait des embeddings pré-calculés et un ANN géopartitionné ; ce
travail n'est pas justifié avant qu'une nouvelle source relève l'oracle.

## Décision

- ne pas entraîner LambdaMART sur ce nouvel univers ;
- ne pas ouvrir fold 1 ;
- garder ranker, decider, risk et accepteur gelés ;
- conserver les overlays sectoriels/RNE et le code contrastif comme artefacts
  reproductibles, sans les promouvoir en production ;
- le prochain levier doit ajouter une information absente, pas régler davantage
  la fusion sur fold 0 : preuves publiques first-party/store-locator,
  rattachements administratifs structurés supplémentaires, ou qualification
  sélective des cas non observables.

Le replay complet des 7 009 scènes n'a pas été poursuivi après cette preuve
d'oracle insuffisant. Un Recall@100 à 99,0 % resterait mathématiquement
possible avec seulement 14 évictions au plus, mais le contrat exige d'abord
un oracle à 99,3 % ; poursuivre aurait donc consommé plusieurs heures sans
autoriser le stade suivant.
