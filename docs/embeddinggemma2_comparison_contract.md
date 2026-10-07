# Comparaison EmbeddingGemma2 — 7 octobre 2026

Autorisation : utilisateur demande de tester le modèle et d'auditer ses inputs.
Modèle brut versus meilleur adapter choisi par contrôle validation, pas dernier
checkpoint. Aucun fit supplémentaire, changement aval ou ouverture test/fold1.

Premier diagnostic : toutes les 2 212 lignes validation préparées ; positives
officielles contre au plus deux négatifs BM25 locaux du dataset figé. Les lignes
sans négatifs sont publiées séparément. Ce n'est PAS du recall retrieval et ne
contient pas tous les distracteurs. Le contrôle64 ayant choisi le checkpoint
fait partie de validation : résultat développement, pas test indépendant.

Deuxième diagnostic : recherche dense initiale exacte100 sur TOUS les SIRET
indexés dans six communes CRM, actifs ET fermés, sans sous-échantillonnage des
établissements, lookup de GT, présélection lexicale ou reranker. Communes :
ordre SHA256(seed|INSEE), minimum4 requêtes validation et 1 000–12 000 SIRET
indexés, premières6 éligibles. Sélection basée CRM/comptages sans GT/outcomes.
Seed `embeddinggemma2-zone-probe-v1`. Le biais vers communes petites/moyennes
et la taille du panel sont publiés ; pas preuve nationale, OOF ou gate99,2%.
L'intégralité du corpus de chaque commune est matérialisée avant toute
génération. Aucun corpus réduit autour des cibles annotées. Les GT hors de la
partition restent des erreurs et sont publiés, pas retirés du dénominateur.

Textes exactement comme le fit : champs natifs legal/trade/site/historical/
supporting names et addresses, code postal. Identifiants exclus comme features.
Pas de XY/BAN, relations BODACC/successions ni contexte des autres sites du SIREN.
Toutes les chaînes stockées sont concaténées mais tokenizer tronque à160 tokens ;
publier la fraction tronquée. Aucune affirmation de couverture de tous les noms
disponibles en amont. Déduplication de textes identiques pour encoder seulement,
puis projection sur CHAQUE SIRET officiel, top100 exact produit directement sur
les scores du corpus complet, jamais un pool2000/500 puis reclassement.

Processus génération séparé sans lecture labels/dataset supervisé ; panel CRM
sanitisé. Après sortie confirmée exit0, outputs hashés complets pour brut ET
tuned : évaluation exact SIRET, SIREN et opérationnel officiel scellé séparés,
gains/pertes appariés. Ledger/membership/sidecar hashes vérifiés ; folds234
uniquement. Test interne4 323 jamais lu, fold0/1 fermés. Pas de recall union
avec l'ancien pipeline ni de validation SLO. Ancien recall91,15% inchangé.

Mac uniquement : MPS cap.25, CPU1 thread, batch inference16, RSS5Gio,
STOP pression anormale/<50% mémoire libre ou25min par phase, timeout parent26min.
Runtime/cache/outputs CATNAT. Namespace comparaison neuf, entraînements et
résultats historiques immuables. Manifeste/indexmeta full hash et inventaire
tailles/mtimes vérifiés avant/après ; gros segments non entièrement rehashés,
limite explicite. Aucun alias CRM indexé ni dépense externe.
