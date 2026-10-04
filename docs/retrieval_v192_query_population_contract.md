# V192 — généraliser la query, sans réutiliser le vecteur0

Protocole AVANT mesure, 4 octobre2026. V191 a scellé une query initial100
sur tous44403034sites/98804872faits, PAS un recall. Codes/artefacts191gelés.
La population active du pilote reste TOUTES128queries gelées115/78, folds234,
aucun filtre/label/ancienhit. FullOOF et gates99,2%/Wilson/généralisation/SLO
restent dus, aval gelé, fold0contaminé/fold1 fermés.

V192 prépare la génération128 : vérifie TOUTES128typedCRM/uniqueID/folds
AVANT sélection ; alignement exact launchencoder78/CRM115/proofencoded78/
vecteurs128x384, finite/unitaire. Seuls IDs/folds/sha du proofencodé sont
des contrôles ; AUCUN token d'expansion, ancien modèle/candidat/truth suivi.
Queryposition n'est pasfeature. Vecteur propre[pos:pos+1] et head ownfold,
exact30features/poids/sourcecatalog113-114/grades/arithmétique identiques187.
Les champs métier CRM et XY/BAN sont traités par125/139 commeavant, pas
réassignés d'une autre query. Aucun vecteur0 partagé entre les queries.

Les groupes1/2 sont physiques, couvrent0..127unefois sans choix label-side.
Ce module seul ne produitPASdes candidats, ne garantitpas cost/recall/SLO.
Tests prouveront alignement et paritéquery0 avec187 puis différents heads
correctement pour autrespositions ; une lecture future source-only pourra
vérifier sans ouvrirGT ces faits sur vraiepopulation aprèscommit.

Étape de coût suivante séparée : deuxquerynames sous1Gio/80s, contrôlefull
sources aprèsdansworkerindépendant avant toutefeatureadmissible. Ne pas
lancer128clonesde339s+quiet aveuglément (~12hworkers/20h total), paspreuve99%.
Tous afterhash/tous128sorties avantGTséparée, exact/op distinct/allhuman,
pas pool2000/500/reranker/GTinjecté/aliasCRM/IDfeature, Mac/CATNAT sansdépense.
