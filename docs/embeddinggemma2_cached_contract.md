# Reprise contrastive à mémoire bornée — 7 octobre 2026

L'essai nocturne initial reste immuable. Il a effectué un pas, yield normal
à 60 % mémoire libre, repris puis échoué MPS OOM sous la limite .16 (~2.84 Gio).
Il ne fournit aucun modèle validé ni gain de recall.

Nouvelle implémentation `scripts/embeddinggemma2_cached.py`, nouveau namespace
CATNAT `cached_20261007_a`. Reprise du checkpoint autorisé du TRAIN initial,
sans recalcul du dataset, changement de split, lecture test, fold0/fold1,
alias CRM dans l'index ou modèles aval. Hashes inputs/checkpoint/code figés.

La loss conserve quatre queries et tous leurs positifs/négatifs (au plus deux
explicites/query), température .05 et exclusion des faux négatifs même SIREN.
Gradient cache : embeddings sans graphe en micro-lots2, calcul de la loss
globale et gradients des embeddings, puis forwards/backwards micro-lots2.
Ce n'est PAS une accumulation de quatre petites losses ni une réduction du
batch contrastif. L'encodeur doit être déterministe : attention dropout zéro
vérifiée, LoRA sans dropout. Même BF16, tokenisation160, masque pooling moyen,
normalisation et rang8/alpha16/Q-V-projection/LR5e-5/AdamW que l'essai initial.
Le gradient est exact pour les valeurs de forward utilisées ; BF16 et padding
par micro-lot peuvent introduire une petite différence numérique avec le
forward monolithique. Test FP64 de loss ET gradients incluant mêmes SIREN.

MPS fraction .25 (~4.4 Gio recommandés), limite maintenue ; ni watermark0,
ni mémoire illimitée. RSS5Gio max, CPU1thread. Admission65% libres, yield50%,
superviseur libère uniquement son child avant45% libres/pression anormale.
Le retrieval précédent est TERMINÉ avant ce lancement : ne pas transposer
ces guards à son contrat figé3Gio/50%. L'utilisateur autorise une consommation
raisonnable au-delà2Gio ; l'objectif principal de la réparation est de réduire
les activations, pas de consommer la totalité des24Gio.

Checkpoints25pas et après les5 premiers, curseur epoch/offset explicite,
loss/gradient nonfinis fatals, historique sorties enfants/retries conservé.
Reprises bornées12 resource-only, aucune boucle sur bug/OOM inconnu.
Validation64 fixe tous100pas ; meilleur checkpoint seulement si loss baisse.
Deuxepochs et budget8h, test final toujours fermé. Loss != recall ; comparaison
brut/tuné sur corpus complet et contribution marginale à initial100 dues.

Préflight réel5 pas supplémentaires terminé exit0 : step1→6, RSS final
1 073 168 384 bytes, MPS driver1 253 785 600 bytes, libre73%, ~6.04s pour5pas
(chargement exclu), pas une garantie sur le pic d'un epoch entier.

Fondement : [GradCache](https://aclanthology.org/2021.repl4nlp-1.31.pdf),
[loss cached officielle Sentence Transformers](https://sbert.net/docs/package_reference/sentence_transformer/losses.html).
