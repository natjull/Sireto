# Retrieval dossier SIREN v2 — exécution bornée

Ce plan remplace le bag de noms concaténé par les documents typés du dossier
SIREN v3. Il n'ouvre pas le fold 1 et ne déclenche aucun entraînement ranker,
decider, risk ou fusion avant le gate retrieval.

## Entrées requises

1. dossier SIREN v3 construit avec SIRENE courant ;
2. stock RNE formalités national canonicalisé ;
3. historique BODACC 2008–2026 fusionné ;
4. configuration `config/siren_name_portfolio_v1.json` ;
5. configuration retrieval `config/retrieval_hierarchical_v2.json`.

Le manifest des documents doit indiquer `temporal_complete=true`, contenir les
sources `SIRENE_CURRENT`, `RNE` et `BODACC`, interdire la concaténation aveugle
et limiter la sortie finale à 100 candidats.

## Exécution

1. Matérialiser d'abord un périmètre technique de 1 000 SIRET :

   ```bash
   python scripts/materialize_siren_dossier_consumers.py retrieval \
     --dossier <SIREN_DOSSIER_V3> \
     --output-dir <TYPED_RETRIEVAL_SMOKE_DOCUMENTS> \
     --document-limit 1000
   ```

2. Construire l'index Tantivy de smoke avec `--smoke-limit 1000`, puis vérifier
   les champs et le plafond. Si le smoke passe, rematérialiser sans
   `--document-limit` et construire une seule fois l'index national :

   ```bash
   python scripts/build_hierarchical_retrieval_index.py \
     --dossier-documents <TYPED_RETRIEVAL_DOCUMENTS> \
     --retrieval-config config/retrieval_hierarchical_v2.json \
     --output-root /Volumes/CATNAT_DATA/SIRETO_RECALL100/indices/hierarchical_v2
   ```

3. Le smoke vérifie uniquement schéma, champs exacts actuels, rescue historique,
   filtre géographique, provenance et union brute ; il n'est pas un benchmark.

4. Produire une seule union label-blind de maximum 2 000 candidats. Tantivy ne
   produit plus de top100 et aucun RRF ne précède le modèle. Les champs légaux,
   commerciaux et de site fournissent chacun leur score BM25, plus un canal
   field-aware aux boosts gelés. LambdaMART reçoit également exacts, numéro,
   q-grams, historique, provenance et expansion SIREN :

   ```bash
   python scripts/retrieve_official_evidence_union.py \
     --input <QUALIFIED_HUMAN_QUERIES_0_2_3_4> \
     --base-index <CONTENT_ADDRESSED_TANTIVY_V3> \
     --output <RAW_UNION_PARQUET> \
     --config config/retrieval_ltr_admission_dossier_v2.json

   python scripts/run_retrieval_ltr_admission.py build-union \
     --candidates <RAW_UNION_PARQUET> \
     --output-dir <SEALED_DEVELOPMENT_UNION> \
     --scope development
   ```

5. Entraîner une fois LambdaMART sur 2/3/4, puis évaluer une fois sur dev 0.
   Publier ensemble couverture identifiable, Recall@100 exact/opérationnel,
   oracle de l'union, actif/fermé, tailles de communes et p50/p95/p99.

6. Verdict :

   - `GO` si couverture >= 80 %, Recall exact >= 99 %, oracle >= 99,3 %,
     max candidats <= 100 et latence dans le contrat ;
   - `PIVOT` si l'oracle passe mais l'admission échoue ;
   - `STOP/PIVOT DATA` si l'oracle reste sous 99,3 %.

Le fold 1 n'est ouvert qu'après gel du retrieval et de l'admission. Il n'y a ni
grille de modèles ni succession de canaris : un smoke technique, un fit train,
une évaluation dev, puis une décision.

Après un `GO` test, la sous-commande `refit` consomme l'union développement et
l'union test déjà scellée, puis ajuste la même recette sur les cinq folds. Cet
artefact est marqué `PRODUCTION_REFIT_NOT_AN_EVALUATION_RESULT`; il ne remplace
jamais les métriques officielles du modèle pré-refit.

## Décisions d'architecture issues de la revue neuve

- Le moteur ne prétend pas implémenter la formule BM25F au sens strict :
  Tantivy calcule BM25 séparément par champ. Un canal field-aware applique les
  trois boosts préenregistrés, et LambdaMART reçoit aussi chaque score élémentaire.
  Cette solution conserve davantage d'information qu'un champ concaténé.
- Le numéro exact est un signal dédié, mais ne constitue jamais à lui seul une
  preuve exacte protégée.
- Seuls nom/adresse actuels exacts et consensus multicanal reçoivent une
  protection d'admission. L'historique et BODACC restent rescue-only.
- Les relations suivies sont exclusivement des relations BODACC structurées,
  sur un seul saut, puis filtrées par INSEE ou par CP de secours. Aucun texte
  d'annonce ni identifiant d'annonce n'entre dans les features.
- Le Parquet d'union possède un manifeste obligatoire liant son SHA-256, la
  configuration et les manifests content-addressés des index officiels.
- Le chemin production refuse un index dont `temporal_complete` n'est pas vrai.
