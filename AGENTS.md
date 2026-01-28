# Diagramme de référence Pipe V6

![Pipe V6 flowchart](docs/diagrams/pipe_v6_flowchart.svg)

_Source Mermaid ci-dessous et version éditable dans `docs/diagrams/pipe_v6_flowchart.mmd`._

```mermaid
flowchart TD

    %% Entrée CRM
    A[CSV CRM<br/>(nom client, n° voie, voie, CP, code INSEE/JINSEI...)] --> B[Détection communes<br/>INSEE en priorité, sinon CP+ville]

    %% SIRENE source de vérité
    B --> C[Appels API SIRENE<br/>par commune + pagination]
    C --> D[(Cache SQLite SIRENE<br/>établissements par commune)]

    %% LLM #1
    A --> E[LLM #1 via Ollama<br/>Normalisation nom + adresse<br/>+ tag PUBLIC/PRIVE/EQUIPEMENT]
    E --> E1[CRM enrichi<br/>nom/adresse normalisés<br/>+ catégorie CRM]

    %% Requêtes INPI / DataGouv
    E1 --> F[Construction requêtes<br/>INPI RNE + DataGouv<br/>(nom normalisé + commune)]
    F --> H[API INPI RNE]
    F --> I[API Recherche entreprise<br/>annuaire-entreprises.data.gouv.fr]

    

    %% Collecte candidats
    H --> K[Collecte SIREN/SIRET candidats]
    I --> K
    

    K --> L[Déduplication candidats<br/>(max 10 / source)]
    L --> M[Lookup SIREN/SIRET<br/>dans cache SQLite SIRENE<br/>(ou API si manquant)]
    M --> N[JSON candidats normalisés<br/>+ infos SIRENE<br/>+ provenance + type juridique]

    %% LLM #2 arbitrage
    A --> O[Contexte CRM brut]
    E1 --> O
    N --> P[LLM #2 via Ollama<br/>Comparaison CRM vs candidats<br/>avec filtre public/privé/équipement]

    P --> Q{Candidat fiable ?}

    Q -->|Oui| R[Sortie MATCH<br/>SIRET retenu<br/>+ score confiance<br/>+ sources impliquées]
    Q -->|Non| S[Sortie REVIEW<br/>(NO_MATCH uniquement après Places)]

    R & S --> T[Export final<br/>CSV/JSON pour CRM<br/>+ métriques/logs]
```

---

# ROUTING XGBoost SIRETO v2.0 (Risk Metamodel)

**Date de mise à jour** : 26 janvier 2026  
**Performance** : 74.5% AUTO @ 99.84% Precision (3 vrais FPs sur 1 872 AUTO)

## Architecture 3 Stages

```
CRM Input → [Stage 1: Ranker] → [Stage 2: Decider] → [Stage 3: Risk Model] → AUTO/REVIEW
```

| Stage | Modèle | Rôle |
|-------|--------|------|
| **Stage 1** | `xgbranker_20260124_210313.json` | Sélectionne les top-k candidats SIRENE par requête |
| **Stage 2** | `xgb_decider_20260124_210218.json` | Score chaque candidat (probabilité 0-1) |
| **Stage 3** | `routing_risk_model.pkl` | Décide AUTO vs REVIEW en analysant la "scène" |

## Règles de décision (Risk Model)

Le **Stage 3** analyse 68 features décrivant le contexte de la requête.

```python
# Routing simplifié
if risk_score >= 0.835:
    return "AUTO_RISK"
else:
    return "REVIEW"
```

## Métriques de référence (Test Set, 2 512 requêtes)

| Métrique | Valeur |
|----------|--------|
| AUTO Rate | 74.5% (1 872 / 2 512) |
| Precision réelle | 99.84% (3 vrais FPs après audit) |
| REVIEW Rate | 25.5% (640 cas) |

---

# Diagramme cible Pipe V7 (XGBoost-first + fallback Web déterministe)

Objectif : faire du **XGBoost** le socle et remplacer les LLM par un traitement **100% déterministe**. Les appels “web” ne servent qu’à tenter d’**upgrader** des cas `REVIEW` en `MATCH` **sans jamais créer de faux positif**.

```mermaid
flowchart TD

    %% Entrée CRM
    A[CSV CRM<br/>(nom, n° voie, voie, CP, commune, INSEE...)] --> B[Pré-traitements déterministes<br/>(normalisation light + clés communes)]

    %% Pool candidats SIRENE local
    B --> C[Préchargement / pool candidats SIRENE<br/>(par INSEE, sinon CP+ville)]
    C --> D[(Cache SQLite / parquet SIRENE)]

    %% Scoring ML
    B --> E[Scoring XGBoost top-k<br/>(features nom/adresse + sémantique)]
    E --> F{Routing XGBoost v2.0<br/>Risk Metamodel<br/>(AUTO / REVIEW)}

    %% Sortie directe
    F -->|AUTO| G[Sortie MATCH (XGB)<br/>SIRET=top1 + score + features]

    %% Fallback Places (Simplifié - "Places as CRM Repair")
    F -->|REVIEW| H[Serper Places API<br/>(nom + adresse)]
    H --> I{Dept-guard<br/>places.postcode[:2] == crm.postcode[:2]}
    I -->|Non| M[NO_MATCH]
    I -->|Oui| J[Rerun XGB Pipeline<br/>avec Places top-1 comme CRM]
    J --> K{XGB Status ?}
    K -->|AUTO| L[MATCH_PLACES<br/>SIRET retenu]
    K -->|REVIEW| M

    G & L & M --> N[Export final<br/>CSV/JSON + métriques/logs]
```

### Principes du Pipe V7

- **Plus de LLM** : uniquement des features ML (XGBoost) + des règles déterministes.
- **Sécurité** : 74.5% AUTO @ 99.84% precision.
- **Places-guided simplifié** : "Places as CRM Repair" - Google identifie l'entreprise, XGB identifie le SIRET.
- **Dept-guard** : rejet si le code postal Places ne correspond pas au département CRM.
- **Post-Places binaire** : soit MATCH_PLACES, soit NO_MATCH (pas de REVIEW après Places).

