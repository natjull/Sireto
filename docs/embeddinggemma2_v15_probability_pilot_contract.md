# V15 — pilote probabiliste du pipeline Gemma2 complet

Le 8 octobre, l'utilisateur remplace explicitement l'encodage exhaustif2212
par « quelque chose de représentatif ». V14 suspendu sans effacer ses caches.
Pas reprise du run7jours, pas nouvelle nuit native ni changement d'aval.

## Tirage sans labels, sans rejeter les communes coûteuses

Population source : toute la validation2212/census937communes V10, folds2/3/4.
Unité primaire : commune. Strates de coût/densité officielles établies avant
labels : <1000,1000–9999,10000–49999,50000–199999,>=200000établissements.
Tirage uniforme sans remise de communes par permutationSHA256, seed immuable
`gemma2-probability-commune-pilot-20261008-v1`. Allocations145(toutes)/100/40/8/2.
Aucune exclusion/repioche après coût constaté, aucun score/GT/proximité de cible
dans la sélection. Allocations choisies avec census query-only pour compromis
coût/précision, même ordre hashé pour toutes les allocations inspectées.

Tous les cas CRM des communes tirées sont évalués, tous les établissements
officiels de chacune sont dans le corpus. Requêtes sans INSEE incluses dans
la strate des petites partitions, tirée exhaustivement ; canal dense vide.
Tirage effectif :526queries,295communes,2 554 848sites. Strates :174/169/111/33/39
queries. Grandes agglomérations : Nantes/Lille, Paris6e/9e, Lyon6e/Villeurbanne,
Marseille8e ; aucun plafond20k comme dans l'ancienpanel80.

Ce n'est PAS un SRS de526requêtes, ni une représentation uniforme de tous les
CRM futurs. Les strates ont des taux d'échantillonnage différents : publier
probabilités d'inclusion des communes, poids inverses et calibration par les
totaux connus de requêtes CRM de chaque strate. Le recall brut526 est descriptif,
PAS le taux représentatif. Taux principal estimé = somme des parts de population
des strates multipliées par leur ratio hits/queries observé (Hájek calibré).
Poids effectifs et taille effectiveKish publiés, pas526observations indépendantes.

## Pipeline scientifique inchangé, comparaison ancienne différée

Seulement Gemma2SITE512best1000 + lexicalnational + BAN/XY, RRF60, géo1/.5faible,
maximum100. Même vuesSITE/packing/tokenizer/batch8/runtime que V8/V14. Ancien
encodeur non réencodé : comparaison familledense différée, pas oubliée ni annoncée
comme mesurée. Aucun entraînement/gradient, modèle aval ni expansionBODACC nouvelle.

Réutiliser lookup V130 complet, géo nationale2212/ownedexit0 et lexicalnational
2212/ownedexit0 de V14/V12. Projection du scope uniquement, pas rescans. Réutiliser
corpus et sortiesSITE entièrement scellés/ownedexit0 des communes compatibles.
Bindings sources/poids/textes inchangés, provenance explicite ; aucun nouveau
ownedexit prétendu pour les sorties réutilisées. Autres communes : workersV14
ORIGINAUX, corpus complet, namespace neuf
`v14/representative_20261008_b`, lancement/supervisionV15. Pas modification du
codeV14 immuable pour adapter sa population. Sources/checkpoints/caches anciens
conservés. Estimation préalable24–36h, à actualiser au débit de ce nouveau scope,
pas promesse8h. Sources/corpus jamais diminués pour respecter cette estimation.

## Évaluation et incertitude

TOUTES les526sorties/canaux/communes et ownedexits0 requis avant ouverture des
labels. Aucun recall partiel ni qualification supprimée. Publier exact/SIREN/
opérationnel, countsbruts/Wilson descriptif séparé, estimation pondérée, segments
folds/rôle/strates/V2/V3/origine/état, gains/pertes sur mêmes526queries.
Sidecar label-side officiel vérifié, absent=STOP. Folds0/1/test4323 fermés.

Bootstrap2000 stratifié par communes, correction de population finie par strate,
intervalle APPROXIMATIF ; compléter par bootstrap pondéré des SIREN pour dépendance
multisite. Le Wilson brut ne couvre PAS le plan de sondage. Deux grandes communes
seulement rendent leur intervalle fragile ; bootstrap dégénéré possible sans
raté observé, ne couvre pas les classes de ratés absentes de l'échantillon.
Ces intervalles ne donnent aucune garantie distribution-free ni preuve99,2%.
Validation déjà utilisée pour checkpoint/développement, pas holdoutpristine.
Recette non retunée selon résultats. Les gates OOF/Wilson/SLO/test final restent dus.

Les protections physiques/reprises atomiquesV14 restent actives. Les erreurs de
données/hashes stoppent, ressources seules retriées, pas de duplication du run
exhaustif. Automation native antérieure toujours en pause, aucune dépense externe.
