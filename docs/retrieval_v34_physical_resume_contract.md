# V34 — reprise physique, recette v33 inchangée

La demande du 11 septembre 2026 autorise correction, accélération sans impact
métier et relance. Aucun nouveau canal, modèle, seuil, quota ou label en entrée.
Le run reste Master2000 apparié v23/v30, succession v32 commune, sur les mêmes
23 571 CRM folds2/3/4. Aval gelé, folds0/1 fermés. Ce n'est pas un gate top100.

## Correction et transformations permises

- BAN : compléter uniquement le départage auparavant indéfini qui comparait
  deux objets Python. Les sept priorités existantes restent intactes.
- Caches locaux à un lecteur immuable, par adresse physique du document ou
  arguments complets de recherche. Pas de cache par vérité ou identifiant CRM.
- Projection des sidecars : toutes les lignes des communes demandées, ou CP
  lorsque l'INSEE manque. Garder toutes les colonnes et les comptages nationaux
  préexistants. Aucune restriction géographique des témoins du Master global.
- Libérer les index entre étapes; borner caches, futures et mémoire. Aucun
  effet sur quotas, candidats, scores, ordre, folds ou dénominateur.
- Ne pas conserver les projections temporaires après usage. Les sources
  officielles, anciens résultats scellés et preuves restent intacts.

## Gate avant reprise

1. Nouvelle préparation officielle hashée après correction BAN. Sources,
   index et inputs identiques à l'ancienne; seul changement de code métier
   autorisé : `retrieval_geo_v16.py`, correction de départage documentée.
2. Tests unitaires; parité exhaustive multiensemble des sorties communes
   (raw, BAN, preuves, signaux, parents, pools et deux Masters avec scores et
   rangs) sur un lot32 déjà scellé. Pas seulement le nombre de candidats.
3. Rejouer le lot32 ayant échoué et contrôler la reproductibilité en ordre
   inverse/sous-lots8. Toute différence non expliquée impose STOP.
4. Balayer la résolution BAN des 23 571 entrées sans labels et vérifier les
   704 sorties BAN déjà scellées avant toute adoption.
5. Mesurer temps, RAM et swap; conserver le plafond14Gio RSS, aucun nouveau
   swapout,30min/lot et36h/run, disque projeté100Gio max. Pas de double worker
   national sans preuve de budget agrégé; un seul processus lourd par défaut.

## Reprise et évaluation

Les22 anciens lots complets ne sont jamais réécrits. Des enveloppes d'adoption
référencent leurs manifests originaux, la nouvelle préparation et le certificat
de compatibilité. Leurs artefacts restent vérifiés par hash. Le lot incomplet
704 est recalculé; un nouveau dossier conserve toute la nouvelle exécution.
Le code accélérateur et ce contrat sont scellés séparément du code v33.

Checkpoint32 et processus neuf par lot évitent la croissance mémoire. Une
exception publie STOP; pas de retry silencieux ni d'extension de budget. Une
génération23571 entièrement scellée précède l'évaluation officielle séparée,
exacte et opérationnelle, historique/V2/V3, pertes et absences de labels. Aucun
gain de recall annoncé avant celle-ci; conclusion Master-only, pas READY_FOR_FOLD1.
