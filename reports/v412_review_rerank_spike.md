# V4.12 — Expérience courte de correction du classement REVIEW

Date : 31 juillet 2026  
Périmètre : données de développement déjà consommées uniquement. Aucun test final, aucun réentraînement et aucune promotion produit.

## Question testée

Les 30 adjudications ont montré que le ranker donne parfois trop de poids à l'adresse et pas assez au nom légal et au rôle de siège. L'expérience teste donc une correction volontairement simple :

```text
score exploratoire = score ranker
                    + 8 × ressemblance avec le nom de l'unité légale
                    + 2 × indicateur siège
```

Cette correction n'est appliquée que lorsque le score du top 1 initial est supérieur ou égal à `2,5`. Les trois cas `AMBIGUOUS` restent obligatoirement en REVIEW, quel que soit le classement produit.

## Résultats

| Mesure | Ranker gelé | Correction exploratoire |
|---|---:|---:|
| Bons top 1 sur les 27 labels exacts R30 | 15 | **25** |
| Mauvais top 1 | 12 | **2** |
| Bons cas dégradés dans R30 | — | **0** |
| Anciens top 1 fiables contrôlés | 116 | 116 |
| Régressions sur ces 116 cas | — | **0** |
| Ambiguïtés transformées en labels exacts | — | **0** |

Les dix dossiers corrigés sont : `fresh:FR028730`, `15204`, `3699`, `8432`, `5691`, `5216`, `15448`, `9835`, `fresh:AC002179` et `15470`.

Les deux erreurs restantes sont :

- `8816`, GHNE Longjumeau : le nom légal est identique pour les établissements concurrents. Il faut comprendre que « site hospitalier » correspond à l'activité hospitalière et non à l'entité administrative GHT ;
- `9675`, INLOG : le nom légal favorise bien IN LOG, mais le top 1 initial est sous le seuil prudent de `2,5`. Le corriger exigerait d'élargir la règle à une zone où les anciens labels montrent davantage de risques.

## Contre-expérience rejetée

La même correction appliquée sans condition atteint 26/27 sur R30, mais dégrade 20 des 116 anciens top 1 fiables. Elle est donc rejetée. Le résultat n'autorise pas une règle globale, ni un changement silencieux du ranker.

## Décision

Verdict : **`GO_EXPAND_LABELS_BEFORE_TRAINING`**.

Le signal « nom légal + rôle de siège » est suffisamment fort pour justifier la suite, mais 27 labels exacts ne suffisent pas pour réentraîner proprement le ranker. La prochaine population à auditer est constituée des REVIEW historiques restants pour lesquels cette correction change effectivement le top 1. Elle fournit les contre-exemples les plus informatifs et permet de décider si le signal doit devenir une feature apprise, une règle REVIEW bornée ou être abandonné.

Reproduction :

```bash
python3 scripts/evaluate_v412_review_rerank_spike.py
```

Le résultat machine doit indiquer `GO_EXPAND_LABELS_BEFORE_TRAINING`, 25 top 1 corrects sur 27, dix corrections, zéro régression R30 et zéro régression sur les 116 scènes historiques fiables.
