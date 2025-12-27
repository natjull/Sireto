# Diagnostic Quantifié - XGBoost Entity Matching

## Résumé Exécutif

| Métrique | Valeur |
|----------|--------|
| **Precision@1** | 91.04% |
| **Taux d'erreur (FP)** | 8.96% (12/134) |
| **Seuil optimal** | **θ = 0.995** (4% erreur, 73% couverture) |

---

## 1. Courbe Risk-Coverage

![Risk-Coverage](diagnostic_plots.png)

### Interprétation

| Seuil θ | Couverture | Erreurs | Taux d'erreur |
|---------|------------|---------|---------------|
| 0.90 | 95.5% | 12 | 9.4% |
| 0.95 | 91.8% | 12 | 9.8% |
| 0.99 | 82.8% | 11 | 9.9% |
| **0.995** | **73.1%** | **4** | **4.1%** ✅ |

> Le seuil **θ = 0.995** est le point d'inflexion où le taux d'erreur passe de ~10% à ~4%.

---

## 2. Analyse de Calibration

Le modèle est **MAL CALIBRÉ** pour les scores élevés :

| Bin de Score | Échantillons | Taux d'erreur réel | Attendu |
|--------------|--------------|-------------------|---------|
| (0.99, 1.00] | 111 | **9.9%** | ~1% |
| (0.98, 0.99] | 8 | 12.5% | ~2% |

**Problème** : Le modèle prédit 99%+ de confiance mais se trompe dans ~10% des cas.

---

## 3. Signature des Erreurs

### Features discriminantes

| Feature | Erreurs | Corrects | Δ |
|---------|---------|----------|---|
| `name_token_overlap` | **0.06** | 0.33 | -0.27 ❌ |
| `addr_token_overlap` | **0.64** | 0.45 | +0.19 ⚠️ |
| `name_semantic_max` | 0.74 | 0.82 | -0.08 |

**Pattern clé** : Les erreurs ont un faible chevauchement de tokens de nom (**0.06 vs 0.33**) mais fort chevauchement d'adresse (**0.64 vs 0.45**).

→ Le modèle se fait piéger par les **établissements co-localisés**.

---

## 4. Seuils Optimaux par Segment

### Par longueur de nom (objectif ≤5% erreur)

| Segment | Seuil θ | Recommandation |
|---------|---------|----------------|
| Court (1-2 mots) | 0.995 | Plus strict |
| Moyen (3-4 mots) | 0.99 | Standard |
| Long (5+ mots) | 0.98 | Plus permissif |

### Par complétude d'adresse

| Segment | Seuil θ | Recommandation |
|---------|---------|----------------|
| Adresse complète | 0.99 | Standard |
| Adresse incomplète | 0.995 | Plus strict |

---

## 5. Règle de Routing Proposée (v2.7)

```python
def route_auto(score, name_token_overlap, addr_token_overlap, name_word_count):
    # Bloquer les matchs par adresse seule
    if name_token_overlap < 0.1 and addr_token_overlap > 0.8:
        return "REVIEW"  # Suspect co-location
    
    # Seuils adaptatifs par segment
    if name_word_count <= 2:
        threshold = 0.995
    elif name_word_count <= 4:
        threshold = 0.99
    else:
        threshold = 0.98
    
    return "AUTO" if score >= threshold else "REVIEW"
```

### Impact estimé

| Métrique | Avant | Après |
|----------|-------|-------|
| Taux d'erreur AUTO | 9.9% | **~4%** |
| Couverture AUTO | 83% | ~73% |

---

## 6. Conclusion

1. **Le modèle actuel est sur-confiant** : Il prédit 99%+ mais a ~10% d'erreur
2. **Les erreurs sont prévisibles** : Faible name overlap + fort addr overlap
3. **Seuil global recommandé** : θ = 0.995 réduit le risque de moitié
4. **Seuils par segment** : Noms courts sont plus risqués → seuil plus strict
