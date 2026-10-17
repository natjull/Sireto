# Brief11 — quatre familles fidèles, STOP avant smoke natif

## Fait
Correctif ciblé `3e3c9726` : `mini.read(mini.verified(ref))` au lieu de lire le dictionnaire.
Recette/poids/sorties attendues inchangés ; activité/OSM réutilisés sans nouvelle fidélité.

## Preuve
- BANO20/20 contextes+deltas, activité20/20, OSM20/20 : erreur max0, reçus exit0 vérifiés.
- Numérique51 :20/20,20/20 bit à bit, écart maximal des vecteurs0 ;598,380s,
  pic RSS worker3918954496octets (suivi parent3939860480octets), sortie possédée0.
- Rehash complet :3681 fichiers/148479301179octets conformes,162,057s, sortie possédée0.
- Contrôleur détaché PID80851 attendu UNE fois via child.wait(),764,570s ; sortie possédée1.
- Le log de cette reprise n'admet que numérique51, audit puis préparation ; pas activité/OSM.

## Écart / blocage
`train3000_frozen_v142_pipeline.selected` passe une référence Parquet vérifiée à
`cohort.projected`, parseur JSON uniquement. `IncompleteJSONError` dès l'en-tête `PAR1`.
Préparation arrêtée avant tout candidat natif, pas non-fidélité numérique ni manque de RAM.
Échec et artefacts conservés ; arrêt automatique `4d77e44c`, aucune deuxième relance.

## Reste / risques
Smoke0/20 : s/requête et pic natif NON mesurés ; ETA préparation+génération NON mesurées.
Pas de ready.json ni commande full validée ; aucun run3000 lancé.
Réparation du chargement query-only Parquet nécessaire avant reprise ; ne pas convertir
ni modifier les sources/labels, ne pas remplir les preuves manquantes par des zéros.
Détails/hashes : `docs/train3000_frozen_v142_brief11_result.json` ; handover mis à jour.
Coefficient-OOF seulement, aucun full-OOF encodeur/recall/SLO annoncé.
Aucune action VM, fit, web, push, modification des sources/labels/pending ou travaux étrangers.
