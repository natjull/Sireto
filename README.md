# Sireto

Pipeline Python pour le rapprochement automatique CRM ↔ SIRENE/SIRET.

## Vue d'ensemble

Sireto V6 prend des entrées CRM (nom client, adresse, code INSEE/JINSEI) et passe par deux phases LLM :

1. **LLM #1** normalise nom/adresse et classe la cible en `PUBLIC`, `PRIVE`, `EQUIPEMENT_URBAIN` ou `INCONNU` via Ollama (`gpt-oss:20b`).
2. **LLM #2** arbitre parmi des candidats collectés (API SIRENE, INPI RNE, DataGouv, recherches Qwant sur pappers/annuaire-entreprises/societe) pour produire un statut `MATCH`, `REVIEW` ou `NO_MATCH`.

Le cache SQLite local agit comme source de vérité pour les données SIRENE et évite les appels redondants.

## Note Phase 4 (V7 / Places)

- Le routing XGBoost sort **uniquement AUTO vs REVIEW** (pas de NO_MATCH avant Places).
- Les cas REVIEW passent par **Places‑as‑CRM** (decider identique).
- **NO_MATCH n’apparaît qu’après Places** si aucune promotion n’est possible.

## Diagramme de flux

![Pipe V6 flowchart](docs/diagrams/pipe_v6_flowchart.svg)

Le diagramme est maintenu sous `docs/diagrams/pipe_v6_flowchart.mmd` (Mermaid) et régénéré via :

```bash
npx -y @mermaid-js/mermaid-cli@10.9.1 \
  -i docs/diagrams/pipe_v6_flowchart.mmd \
  -o docs/diagrams/pipe_v6_flowchart.svg \
  --backgroundColor transparent
```

## Modules cibles (voir `AGENTS.md`)

- `src/pipe_v6/config.py` : dataclass `PipelineConfig`, loader YAML/env.
- `logging_utils.py` : `setup_logging` centralisé.
- `crm_loader.py` + `commune_detection.py` : ingestion CSV et extraction des communes (`CommuneKey`).
- `sirene_client.py` / `sirene_cache.py` : appels API, cache SQLite (`establishments`).
- `llm_normalizer.py` : wrapper Ollama + parsing `NormalizedCRMEntry`.
- `external_sources.py` : clients RNE, DataGouv, Qwant + extraction SIREN/SIRET.
- `candidate_store.py` : `RawCandidate`, `NormalizedCandidate`, enrichissement via cache.
- `category_mapping.py` : mapping nature juridique → catégorie publique/privée/équipement.
- `llm_matcher.py` : génération prompt arbitrage, parsing `LLMMatchDecision`, classification finale.
- `exporter.py` + `pipeline.py` : orchestration complète, exports CSV/JSON et métriques.
- `scripts/run_pipe_v6.py` : CLI (`--config`, `--crm-path`, `--output-path`).

## Roadmap de développement

1. **Structuration** : créer `src/pipe_v6/`, config centralisée, logging.
2. **Ingestion & communes** : mapping CRM → DataFrame standard et extraire les communes uniques.
3. **SIRENE cache** : schéma SQLite, client API, stratégie `get_or_fetch_commune`.
4. **LLM #1** : prompt normalisation + parser robuste.
5. **Sources externes** : clients RNE/DataGouv/Qwant, extraction URL → SIREN/SIRET.
6. **Candidats** : déduplication, enrichissement SIRENE, attribution catégorie.
7. **LLM #2** : filtre par catégorie, prompt arbitrage, seuils `MATCH/REVIEW/NO_MATCH`.
8. **Pipeline** : préchargement par commune, traitement ligne à ligne, accumulation des résultats.
9. **Export & métriques** : CSV/JSON, stats globales.
10. **CLI & tests** : entrée `run_pipe_v6.py`, suites Pytest (`category_mapping`, `extract_siren_siret_from_url`, parsers LLM, cache SQLite) + test d'intégration sur échantillon CRM.

Chaque tâche est détaillée dans `AGENTS.md` avec objectifs, entrées/sorties et points d'implémentation.
