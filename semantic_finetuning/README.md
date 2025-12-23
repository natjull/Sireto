# SIRET-BERT Semantic Fine-tuning

Ce module contient les scripts pour fine-tuner un modèle d'embeddings sémantiques 
spécialisé pour le matching des dénominations sociales françaises.

## Prérequis

- Python 3.14+ avec l'environnement `.venv-ml`
- PyTorch 2.9+ avec support MPS (Apple Silicon)
- sentence-transformers 5.2+

## Utilisation

### 1. Préparation du dataset

```bash
.venv-ml/bin/python semantic_finetuning/prepare_dataset.py
```

Génère:
- `data/semantic_train.jsonl` : Paires d'entraînement avec hard negatives
- `data/semantic_eval.jsonl` : Paires d'évaluation

### 2. Fine-tuning

```bash
.venv-ml/bin/python semantic_finetuning/train.py
```

Options:
- `--config semantic_finetuning/config.yaml` : Fichier de configuration
- `--resume <checkpoint_path>` : Reprendre un entraînement

### 3. Évaluation

```bash
.venv-ml/bin/python semantic_finetuning/evaluate.py \
    --model models/semantic/siret-bert-v1/run_XXXXXX/final \
    --sample-size 1000
```

Métriques:
- Recall@1, @5, @10
- MRR (Mean Reciprocal Rank)
- Comparaison avec le modèle de base

### 4. Export pour production

```bash
.venv-ml/bin/python semantic_finetuning/export.py \
    --model models/semantic/siret-bert-v1/run_XXXXXX/final \
    --version v1
```

Crée le modèle dans `models/semantic/siret-bert-v1/`

### 5. Utilisation dans le pipeline

```bash
export XGB_SEMANTIC_MODEL=models/semantic/siret-bert-v1
export XGB_SEMANTIC_ENABLED=1

python scripts/infer_xgb_matcher_topk.py
```

## Configuration

Le fichier `config.yaml` contient tous les hyperparamètres:

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `batch_size` | 64 | Optimisé pour 24GB RAM unifiée |
| `epochs` | 3 | Nombre d'époques |
| `learning_rate` | 2e-5 | Taux d'apprentissage |
| `device` | mps | Apple Metal Performance Shaders |

## Architecture

```
semantic_finetuning/
├── config.yaml          # Hyperparamètres
├── prepare_dataset.py   # Génération des paires
├── train.py             # Fine-tuning
├── evaluate.py          # Métriques d'évaluation
├── export.py            # Export pour production
└── README.md            # Ce fichier
```

## Preprocessing Cohérent

**IMPORTANT**: Le preprocessing doit être identique à `src/xgb_matcher/semantic.py`:

```python
from src.xgb_matcher.semantic import _normalize_for_embedding
from src.xgb_matcher.naming import normalize_name

def preprocess_for_embedding(text: str) -> str:
    normalized = normalize_name(text, uppercase=False)
    return _normalize_for_embedding(normalized)
```

## Logs TensorBoard

Les logs sont sauvegardés dans `models/semantic/siret-bert-v1/run_XXXXXX/`:

```bash
tensorboard --logdir models/semantic/siret-bert-v1
```
