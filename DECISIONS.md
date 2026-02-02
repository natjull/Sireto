# Decisions Log

## 2026-01-31 - Orientation cible: retrain + zero skew + commune/CP strict

Decision:
- Reentrainer les modeles (ranker/decider/routing) pour refleter le retrieval reel.
- Le ranker devient le pruning ML principal et doit rester leger/rapide.
- Zero train/serve skew: meme retrieval/pruning/features pour samples, training et inference; aucune injection de GT.
- Aucun fallback departemental: commune/CP strict (si CRM faux, c'est un cas REVIEW/NO_MATCH).

Rationale:
- Les metriques training ne sont pas reproductibles en inference tant que le retrieval/pruning differe.
- L'injection GT dans le training masque des pertes reelles en inference.
- Le besoin metier impose de ne pas depasser la commune/CP.

Consequences:
- Fixer le retrieval commune/CP (pool + pruning non destructif), puis regenerer samples et reentrainer.
- La precision AUTO reste non negociable; le routing sera recalibre sur la distribution d'inference.

## 2026-01-31 - Strategie hard negatives (ranker d'abord)

Decision:
- Le ranker est entraine sans hard negatives generes par un modele.
- Les hard negatives pour le decider et le routing sont generes par le ranker nouvellement entraine (meme retrieval commune/CP).

Rationale:
- Evite de biaiser le ranker par un ancien modele.
- Aligne les hard negatives sur la distribution reelle de Stage 1 en production.

Consequences:
- Generation de samples en deux temps: dataset ranker sans hard negatives, puis dataset decider avec hard negatives issus du nouveau ranker.

## 2026-01-31 - Fix Retrieval "Bag of Names" SOTA

Decision:
- Adoption de la strategie "Bag-of-Names" pour le retrieval (TF-IDF) ET l'entrainement.
- Suppression de la normalisation L2 dans TF-IDF (`norm=None`).
- Activation du "Universal Rescue" (Whitelist Adresse Hash + Numeric Tokens) pour tous les modes (y compris `insee_then_postcode`).

Rationale:
- La normalisation L2 pénalisait les candidats "riches" (filiales avec beaucoup de noms), causant une perte de Recall critique.
- Le mode "insee" strict manquait de robustesse pour les cas "Nom faux / Adresse exacte".
- Cette stratégie permet de "Ringardiser le marché" en trouvant des correspondances structurelles complexes (Siège/Filiale) que les méthodes classiques ratent.

Consequences:
- Modification de `blocking.py` (norm=None) et `infer_xgb_two_stage.py` (Rescue).
- Regeneration des samples avec ces nouveaux parametres pour supprimer le biais train/serve.

## 2026-02-01 - Opération "Ultima" : Double Indexation Retrieval

Decision:
- Implémentation du "Address First Retrieval" : double indexation TF-IDF sur le Nom ET l'Adresse.
- Fusion des résultats au niveau du préfiltre pour maximiser le Recall.

Rationale:
- Atteindre le 100% théorique de Recall en Top-500.
- Assurer que même si le Nom CRM est totalement différent du Nom SIRENE, le candidat est repêché si l'adresse (même approximative) matche.
- Renforcer la robustesse aux changements de dénomination (Siège/Filiale) et aux erreurs de saisie d'adresse.

Consequences:
- Modification de `blocking.py` (Address TF-IDF) et alignement de tous les scripts de génération et d'inférence.
- Nouveau record de Recall Retrieval Relatif à ~97% (hors communes inexistantes).
