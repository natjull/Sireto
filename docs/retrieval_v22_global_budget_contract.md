# V22 — concurrence globale des 2000 références

Préenregistrement, 9 septembre 2026. Autorisation utilisateur : « suis ton
feeling », après proposition de remettre les 2000 places en compétition.
Cela ne vaut pas autorisation de national ou de fold1.

## Hypothèse et recette unique

« Ancien » signifie présent dans le raw v11 de recherche initiale, pas ancien
dans le temps, fermé, humainement confirmé ou déjà client. « Nouveau » signifie
ajouté au réservoir par les canaux complémentaires ou l'expansion SIREN→sites.
Ce sont des provenances de génération, jamais des labels de qualité.

V21 protège jusqu'à 1750 SIRET du raw et remplit le reste en rescue. V22
supprime cette priorité d'admission. Sur le **même** réservoir v21, la seule
recette est le top2000 global par `rescue_score`, puis `direct_rrf`, puis
`parent_rrf`, puis SIRET pour départage exact. Le score v21, les preuves et
les rangs restent inchangés, calculés avant lecture de labels. Le rang de
base n'est pas un critère de sélection v22; il reste un champ d'audit.

C'est un test causal de la protection 1750+250, pas encore un nouveau modèle
nom/adresse. Les canaux initiaux peuvent apporter une preuve au score; ils
n'apportent plus une place garantie. Il n'y a pas d'égalité artificielle de
quota entre sources, ni de nombre fixe de sites par entreprise.

Cette étape est choisie avant de développer le parent appris et la preuve
conjointe plus riche, afin de ne pas leur attribuer un effet du seul budget.
Aucun LambdaMART ne score plus de 2000 références; aucun modèle n'est ajusté
dans cette expérience. Pas de nouvelles features synthétiques ou identifiants.

## Évaluation et généralisation

Un préflight label-blind 1..16, puis une génération complète scellée, puis une
évaluation séparée sur les 1034 dossiers. Plafond absolu 2000, aucune injection
de vérité. Reprendre les vérités exactes et les sidecars officiel/opérationnel
scellés de v21. Toutes les qualifications, absences et origines restent dans
le dénominateur. Publier chaque gain ET chaque régression, y compris parmi
les 512 contrôles, sans faire échouer l'évaluateur au seul motif d'une perte.

Les métriques historique, V2, V3, exactes et opérationnelles sont publiées
ensemble. Une présence opérationnelle n'est pas une précision top1. Les
limites de preuve same-site v20/v21 restent inchangées.

**La projection `23049 + rescues` est désormais interdite** : le préfixe
n'est plus garanti. Les 512 contrôles présélectionnés ne permettent pas
d'estimer les pertes sur tous les anciens hits nationaux. Aucun pourcentage
national ni passage de gate ne peut être déduit de ce panel biaisé.

Les folds 2/3/4 ont été réutilisés pour le développement. Pas de grille,
aucun réglage après résultat, aucun nouveau split présenté comme vierge.
Un export réellement nouveau, non sélectionné par le succès du retrieval,
scellé avant son évaluation, demeure requis pour une affirmation cross-CRM.
Une absence de paramètre appris ne supprime pas le biais de choix de stratégie.

## Intégrité et arrêt

Tests : compétition globale même pour un ancien rang1, score inchangé,
invariance des scores/sélections hors départages exacts aux lignes et à la
provenance d'audit, cap, absence de doublons, refus de fold interdit et
publication des régressions de contrôles. Sources, code et sorties hashés.
Mac + CATNAT seulement, temporaires externes, RSS <=14 Gio, zéro swapout,
30 minutes par phase. Aucun coût externe, aucune écriture dans les artefacts
historiques, aucun dégel des modèles aval.

Verdict maximal `PIVOT_GENERATION` : Master seulement, PreRank500/Recall100,
Wilson et SLO non mesurés. `STOP` si intégrité ou attribution manque. En cas
de résultat faible, publier puis arrêter cette recette, sans modifier son
score pour sauver les dossiers examinés.
