# V120 — Même mapping, clé nullable et jointure par hachage

Contrat avant mesure. V119 :25,097137s worker, dont24,416852s export
observations, mais0,215670s pour l'audit complet aligné. L'instrumentation
réfute l'attribution du temps restant au seul audit. EXPLAIN du mapping
`name_text = name_text AND lexical_eligible` montre BLOCKWISE_NL_JOIN.

Révision physique unique : remplacer ce prédicat par
`CASE WHEN lexical_eligible THEN name_text END = name_text`.
EXPLAIN montre HASH_JOIN. Pour eligible=false/NULL, cléNULL, aucun match ;
pour eligible=true, égalité inchangée. Conservation des lignes contextuelles
par LEFT JOIN, mêmes noms/owners/scopes/metadata/IDs, aucun filtre additionnel.
Pas de hausse de mémoire, threads, temps ou taille de sortie.

Source identique V118 scellée, même audit V119 indépendant de chaque champ
et chaque owner/référence, mêmes contrôles de dictionnaires complets V118.
Fixtures comparent toutes les tables V118/V119/V120 et refusent corruption
et redistribution d'owner. Aucune modification des artefacts V118/V119.

Toujours préparation d'un sample officiel, pas de retrieval ni de mesure
recall. Aucune compilation nationale, modèle/CRM/GT ou fold0/fold1 ouvert.
Les limites de couverture physique/hash complet des grandes sources restent
héritées. Le sample officiel de noms V118 inclut RNE et SIRENE_CURRENT,
mais pas BODACC ; cela ne prouve pas l'absence de BODACC de la source complète.
Les footers montrent738 groupes purement BODACC (premier1249),468 RNE
(premier1991),1245 SIRENE_CURRENT, plus8 groupes mixtes, total2459.
Le prochain probe de couverture doit inclure explicitement ces familles
par métadonnées source-only, indépendamment de toute query ou vérité.
