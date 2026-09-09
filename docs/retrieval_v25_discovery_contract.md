# V25 — audit des absences de découverte, puis correction distincte

9 septembre 2026. Première phase : **audit label-side**, pas un générateur.
Population entière des 1034 requêtes scellées folds2/3/4, y compris contrôles,
statuts ambigus/non résolus et vérités absentes. Fold0/fold1 jamais ouverts.

Comparer la représentation officielle et indexée des parents humains :
existence de sites officiels, noms disponibles/sélectionnés, présence dans
l'index Tantivy, admissibilité à sa contrainte géographique, correspondance
à au moins un des dix canaux textuels SIRET v18. Une recherche par identifiant
humain est autorisée uniquement dans cet audit de plafond : **aucun résultat
ainsi obtenu ne peut entrer dans un pool candidat ou dans une feature**.

Catégories, dans cet ordre : absence de sites officiels; aucun document SIRET
indexé du parent; parent indexé mais hors géographie effective; parent local
mais aucun des canaux textuels v18 ne correspond; texte correspondant mais
parent absent des signaux (profondeur/troncature à investiguer); déjà retrouvé.
Une catégorie texte/profondeur n'est pas une preuve qu'une nouvelle recherche
non guidée par la vérité retrouvera le parent. Ne pas confondre existence dans
les sources et indexation de chaque alias.

Reconstituer les requêtes v18 exactement depuis SceneIR en ne lisant que les
champs CRM/géographiques autorisés. Vérifier aussi que les champs de géographie
priorisés par l'ancien adaptateur ne divergent pas des champs CRM. Conserver
la variante CRM stricte comme diagnostic, jamais choisir la géographie selon GT.

Publication par perte initiale, contrôles, fold, origine, label_kind, V2/V3,
et exact/opérationnel de la baseline inchangée. Pas de nouvelle métrique
Master/100 revendiquée par cet audit. Les sidecars officiels scellés restent
l'autorité de l'attribution end-to-end.

Mac/CATNAT seulement; 30 minutes, RSS14 Gio, zéro swapout; temporaires externes.
Les sources et manifestes sont vérifiés par SHA; sceller aussi meta.json et
l'inventaire taille/mtime des fichiers Tantivy avant/après, sans prétendre
avoir recalculé tous les octets de cet index national. Arrêter si mutation.

Après cet audit, une éventuelle correction doit avoir **son propre addendum
figé avant mesure**, une génération séparée sans labels et un budget borné.
Aucune grille, aucun alias CRM, aucune injection, aucune exception par dossier.
Pas de national ni dégel aval; OOF n'efface pas la réutilisation du panel.
Verdict maximal `PIVOT_GENERATION`; défaut d'intégrité = `STOP`.
