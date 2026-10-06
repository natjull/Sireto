# Sidequest EmbeddingGemma 2 — 6 octobre 2026

Entraînement local autonome demandé par l'utilisateur, sans mise en production,
sans modification du retrieval V356/V359 en cours, sans réentraînement aval.

Base officielle Google `google/embeddinggemma-2`, révision
`914f7f89142e33e77833254d9c9b90c3cef7303b`, Apache 2.0. Poids source HF,
pas GGUF. Texte seul : 271 002 624 paramètres réellement chargés ; BF16,
pooling moyen avec préfixes officiels query/document puis normalisation L2.
Les composants audio/vision sont désactivés.

Folds 0/1 fermés. Les 2 000 query_id du run courant sont exclus AVANT
matérialisation des labels. La population restante des folds 2/3/4 comporte
21 571 requêtes. Le découpage déterministe 70/10/20 est groupé par SIREN,
avec strates grossières public présumé (préfixe SIREN 1/2) et multi-lignes.
Les proportions sont celles des groupes, pas une promesse de proportions de
lignes exactes ; EI, statut officiel public et temporalité restent à auditer.
Le test interne est scellé avant minage et jamais lu par le trainer.
Cette population historique a déjà servi au développement : ce n'est PAS
un nouveau test pristine du pipeline complet. Le modèle fine-tuné, lui,
ne voit aucun exemple CRM de ses groupes validation/test pendant le fit.
Les groupes du panel courant peuvent partager une entité avec ce nouveau
train : le modèle expérimental n'est jamais appliqué au panel courant ; ses
métriques éventuelles ne pourront pas utiliser ce panel comme test inédit.

Documents positifs exclusivement officiels du corpus Tantivy national existant
(82 772 691 documents dont 43 896 818 établissements). Aucune variante CRM
ajoutée à l'index. Positifs absents des documents officiels publiés séparément,
pas supprimés du dénominateur d'une future évaluation retrieval.
Négatifs BM25 naturels dans la commune/CP, au plus trois : exclusion du même
SIREN et des SIREN annotés appartenant à un autre split. Aucun GT injecté dans
une recherche évaluée : les lookups de positifs ne servent qu'à l'entraînement.
Les identifiants servent aux joints/splits/masquages, jamais aux textes.

Une recette unique : LoRA rang 8 / alpha 16 sur Q/V et projection embedding,
AdamW LR 5e-5, weight decay .01, contraste softmax température .05 ; 4 queries
et au plus 2 négatifs explicites/query, autres positifs comme négatifs sauf
même SIREN. Batch 4 est le VRAI batch contrastif, pas une accumulation annoncée
comme un plus grand pool. Séquences 160 tokens, deux epochs maximum, budget
global huit heures ; aucune recherche interminable d'hyperparamètres.

Baseline modèle brut conservée. Contrôle validation contrastif fixe de
64 lignes sélectionnées par hash, mesuré avant fit puis tous les 100 pas.
Checkpoint du meilleur contrôle et reprise périodique ; cette loss ne vaut
ni recall, ni mesure contre tout SIRENE, ni choix de fusion top100. Si le contrôle
ne progresse pas, ne pas déployer. Aucune garantie de 99,2 %.

Runtime/cache/poids/artefacts uniquement dans
`/Volumes/CATNAT_DATA/SIRETO_EMBEDDINGGEMMA2`, environnement séparé.
MPS limité à 16 % de la mémoire recommandée ; un thread CPU ; RSS 3 Gio max.
Admission à 72 % libre, yield à 62 %, superviseur libère le child avant 58 %
ou toute pression anormale, largement avant le seuil retrieval 50 %.
Pas de désactivation des watermarks MPS. Reprises resource-only, bornées à 12,
pas de retry aveugle d'erreurs de code/numerics ; import/chargement inclus
dans la surveillance et deadline. Ces gardes limitent le risque, elles ne
garantissent pas une absence totale de contention ou de pic instantané.

Suite seulement si intérêt : comparer brut/tuné sur le même corpus complet
(pas les seules cibles annotées), recall SIRET exact@100 et SIREN, contribution
marginale à l'union géo/lexical/dense, précision opérationnelle séparée,
latence et mémoire. Architecture initiale directement <=100, sans nouveau
pool fixe reranké. Cross-fitting sur folds 2/3/4 avant éventuelle consommation
des scores dense par un modèle aval ; les modèles aval restent gelés jusqu'au
gate retrieval. Fold 1 inchangé, fermé jusqu'à autorisation one-shot explicite.

Sources : [modèle officiel](https://huggingface.co/google/embeddinggemma-2),
[entraînement Google](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-multimodal-embeddinggemma-with-sentence-transformers).
