# V10 — complémentarité et validation élargie

Autorisation humaine du8octobre2026 après la nuit : poursuivre la complémentarité
et obtenir une mesure représentative avec une incertitude explicite. Aucun nouveau
mandat nocturne implicite ; suivi précédent PAUSED, test4323/folds0/1 fermés.

## Population et représentativité

Recenser TOUTE la validation2212 préexistante, SIREN-disjointe du TRAIN, sans
sélection sur score, présence de cible, taille de commune ou statut humain.
Sources scellées de `night_20261006_a`. Projection query-only avant génération.
Les SIREN ne servent qu'aux vérifications de split et segments, jamais au score.
Les3communes inconnues et341adresses sans numéro restent dans le dénominateur.
Recensement effectif :1430SIREN,937communes,714/757/741queries folds2/3/4.
196queries partagent un SIREN avec le panel80,2016non. Publier ces segments
séparément, sans appeler les2016 un holdout pristine : validation utilisée pour
sélection de checkpoint et historique de développement du pipeline.

Ceci représente la population de validation du CRM disponible, pas tous les CRM
futurs. Le split initial stratifie grossièrement par préfixe administratif SIREN
et multiplicité, pas par toutes les catégories juridiques ou secteurs. Aucun
identifiant ne devient feature. Audit des états/qualifications après scellement,
pas pour choisir les cas. Une certification finale demande une recette figée,
test4323 ouvert uniquement après nouvelle autorisation explicite, et OOF/SLO
exigés par le contrat actif. Fold1 reste fermé indépendamment de cette demande.

## Hypothèse de fusion figée

Chaque canal émet indépendamment initial100. Deux encodeurs denses ne constituent
pas deux preuves indépendantes : contribution dense = MAX des RRF60 de l'ancien
native160 et du nouveau SITE512, puis ajout RRF60 lexical et géo (1fiable/.5faible,
géo nonrésolue vide). Sortie100 au maximum, pas quota fixe, positif injecté,
master2000/500, modèle aval, entraînement d'admission ou règle pour NAOS.
Une hypothèse motivée de famille de preuves corrélées, PAS un optimum/SOTA.

Tout replay sur les80déjà lus est strictement post-hoc diagnostique, jamais une
validation de généralisation. Ne pas modifier les poids après ce diagnostic.
Sur la population élargie : baseline nouveauSITE+lex+géo et challenger famille
dense+lex+géo, mêmes requêtes/corpus, tous sorties et vrais exits scellés avant
labels. Tous sites officiels des partitions CRM déclarées, sans partition autour
de la vérité ; une cible hors partition reste un raté, jamais retirée.

## Coût et précision

Recensement officiel query-only :19 097 808sites dans les937communes (somme par
partition, pas cardinalité unique nationale). Les anciens80couvraient119064sites.
Encoder intégralement les partitions élargies avec les deux modèles représente
plusieurs jours au débit nocturne observé, avant dédup effective/enrichissement.
Ne pas lancer cette charge massive silencieusement. Un pilote probabiliste doit
être tiré sur toute la population avant scores, sans rejeter les grandes villes.
Il ne remplace pas la validation complète ni le test final.

Rapporter exact/SIREN/opérationnel séparément, Wilson95% et bootstrap apparié
par SIREN (sites corrélés), segments folds/zones/origines/états et gains/pertes.
À99,2% observés sur2212, Wilson95 environ[98,72%;99,48%] : pas une certitude
ni même le gate Wilsonbas99,0%. L'intervalle4323 est environ[98,88%;99,42%].
Ces chiffres sont de planification, PAS des scores mesurés. Incertitude sur
échantillon et décalage de distribution à distinguer. Une barre de99,2% seule
ne suffit pas à qualifier le résultat READY_FOR_FOLD1.

Tests CPU : duplication d'un encodeur ne double pas sa contribution, conservation
des deux sources, plafond100/uniqueness, rejet parent incohérent, projection sans
GT et conservation champs absents, refus fold1, intervalle non certain.
Artefacts CATNAT uniquement, historiques et modèles aval immuables.
