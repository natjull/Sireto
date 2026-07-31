# V4.13 — Contrat minimal de labels CRM frais

## 1. Décision d'architecture

Ce contrat succède à la branche S1 pour le seul objectif scientifique actif :
valider puis améliorer le matching CRM vers SIRET. Il ne modifie pas les
artefacts S1 et V1 existants, qui restent immuables et historiques.

L'autorité Ed25519, le Keychain producteur, les manifests signés et les
workers S1 ne sont plus dans le chemin critique. Ils apportaient une preuve
de non-répudiation du producteur, mais ne créaient ni export CRM frais, ni
vérité indépendante, ni couverture identifiable.

Le threat model V4.13 est un opérateur local coopératif sous un UID unique.
Le contrat protège contre la dérive, la réutilisation accidentelle, la fuite
de vérité, l'optional stopping et le rejeu. Il ne prétend pas résister à un
processus hostile déjà maître du même UID.

## 2. Gate zéro : disponibilité de la matière première

Avant toute qualification, il faut une collection nouvelle contenant :

1. une frame CRM exhaustive, postérieure aux 23 609 lignes consommées ;
2. un identifiant source stable et unique par ligne ;
3. une preuve indépendante permettant de relier cet identifiant à un SIRET.

La preuve indépendante peut être :

- un mapping contractuel ou administratif `source_record_id → SIRET` ;
- un identifiant officiel déjà porté par le système source ;
- un document administratif scellé et versionné.

Une ressemblance nom/adresse, le retrieval, un candidat, un rang, un score,
une prédiction ou SIRENE seul ne peut jamais créer `MATCH_EXACT`. SIRENE peut
uniquement contrôler le format, le SIREN, l'état et la cohérence temporelle
d'un SIRET déjà fourni par une preuve indépendante.

Le gate zéro est exécuté sur la collection complète, sans arrêt anticipé :

- au moins 657 lignes `MATCH_EXACT` ;
- couverture `MATCH_EXACT / toutes les lignes source ≥ 80,0 %` ;
- zéro chevauchement interdit avec les registres consommés.

Si l'export ou la preuve n'existe pas, le verdict est
`WAITING_FOR_NEW_SOURCE`. Si l'intégrité est saine mais que le volume ou la
couverture échoue, le verdict est `PIVOT_SOURCE_EVIDENCE`. Aucun modèle n'est
alors entraîné.

## 3. Entrée minimale

La collection est déposée sous une racine SSD fixe et contient exactement :

```text
collection_manifest.json
crm_source.csv | crm_source.parquet
authoritative_mapping.csv | authoritative_mapping.parquet
```

Le manifeste canonique fixe avant ouverture :

- `export_id`, `reference_date`, période et cutoff ;
- définition exhaustive de la population ;
- format, taille, nombre de lignes et SHA-256 des deux fichiers ;
- colonnes et sémantique de `source_record_id` ;
- type, origine, date et portée de la preuve autoritative ;
- absence d'exclusion fondée sur un résultat de matching.

Les fichiers sont réguliers, à lien unique, sans symlink, en `0600` sous des
répertoires `0700`. Deux observations espacées d'au moins 60 secondes doivent
conserver inode, taille, mtime, ctime et hash. Les écritures aval sont
`O_EXCL`, durables et non-clobbering.

## 4. Qualification et séparation

Le builder de qualification n'importe aucun module de retrieval ou de
modèle. Il produit trois arbres physiquement distincts :

- `queries` : ID opaque, date et champs CRM autorisés, sans SIRET/SIREN ;
- `oracle` : ID opaque, label, SIRET/SIREN éventuels, preuves et motifs ;
- `audit` : manifests, hashes, couverture, collisions et anti-chevauchement.

Chaque ligne reçoit exactement un label :

- `MATCH_EXACT` : une preuve indépendante unique, SIRET valide et
  temporellement cohérent ;
- `AMBIGUOUS` : plusieurs SIRET compatibles ou seulement un SIREN ;
- `UNRESOLVED` : preuve absente, invalide, expirée ou contradictoire.

Toutes les lignes source restent au dénominateur. Le scanner de fuite refuse
toute séquence décimale autonome de 9 ou 14 chiffres dans les colonnes
requêtes, y compris Unicode et après projection NFKC.

Les registres gelés sont obligatoires :

- compatibilité des 23 609 anciennes lignes, manifest
  `6435f071c027b7458a8c8b3f09dce796cadc8a0fdc03e0413bcecba1fe969f9b` ;
- 19 754 SIREN consommés, manifest
  `b220efd7c4dc89a980b9d0b5501e16fd286edcafdff61573ae6c5e8d8423c6ff`.

## 5. Splits et ordre ML

Après qualification complète, les composantes partageant un SIREN
autoritatif sont affectées ensemble à `fit`, `dev` ou `test` par une fonction
de hash préenregistrée. Les proportions cibles sont 70/15/15. Une ligne sans
SIREN suit le hash de son ID opaque. L'affectation ne reçoit aucun score ou
résultat modèle.

L'ordre est obligatoire :

1. geler les trois splits et leurs manifests ;
2. exécuter le retrieval gelé sur `fit` et `dev`, jamais encore sur `test` ;
3. franchir couverture identifiable ≥ 80 % et Recall SIRET exact @100 ≥ 99 %
   sur `dev`, vérité absente du pool comptée comme erreur ;
4. entraîner le ranker candidat sur les pools réels `fit` ;
5. produire ses prédictions OOF par composante SIREN ;
6. entraîner l'accepteur query-level sur ces scènes OOF ;
7. choisir le seuil sur `dev` à précision SIRET exacte observée ≥ 99,8 % ;
8. geler code, modèles, seuils et politiques ;
9. ouvrir `test` une seule fois.

Le ranker, le decider, le risk model et l'accepteur historiques restent gelés
jusqu'au gate retrieval de l'étape 3.

## 6. Mesures et interprétation

Le plafond de candidats est 100 par requête, jamais une moyenne. Sont publiés
ensemble :

- couverture identifiable et nombres bruts ;
- Recall SIRET exact @1/@10/@50/@100 ;
- précision et couverture `AUTO_MATCH` ;
- intervalles Wilson 95 % et 99 % ;
- métriques historique, V2, V3 et V4.13 ;
- résultats par actifs, fermés, mégapoles et multi-sites.

Une précision observée ≥ 99,8 % n'est pas une garantie. Une revendication
unilatérale à 99 % avec zéro erreur exige au moins 2 301 décisions AUTO
indépendantes. En dessous, le rapport doit dire « estimation observée ».

## 7. État actuel et séquence

Au 31 juillet 2026 :

- le retrieval V4.12 passe le dev historique à 1 217/1 217 Recall@100 ;
- la garde V4.12-G produit 614/746 AUTO, zéro erreur historique ;
- les 23 609 lignes locales sont toutes consommées ;
- aucune collection CRM fraîche admissible n'est présente dans l'inbox.

La séquence suivante est donc :

1. auditer et geler le présent contrat et son plan ;
2. attendre une nouvelle collection sans ouvrir retrieval ou modèle ;
3. exécuter seulement le gate zéro ;
4. implémenter le builder minimal si et seulement si la source et la preuve
   existent ;
5. poursuivre les gates dans l'ordre défini ci-dessus.
