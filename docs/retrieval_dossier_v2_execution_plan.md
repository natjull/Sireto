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
   filtre géographique et plafond 100 ; il n'est pas un benchmark.

4. Produire l'union train 2/3/4 et dev 0, maximum interne 2 000, avec les
   canaux déclarés dans `config/retrieval_ltr_admission_dossier_v2.json`.

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
