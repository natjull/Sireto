# V37 transmission des preuves de découverte parent

Préenregistrement du2 octobre2026, avant fit v37. Correction structurelle
identifiée pendant l'audit v36, pas une grille de réglages : les26 canaux
du contrat parent v16 sont calculés dans `parent_signals.parquet`, mais ni
v35 ni v36 ne donnent leurs rangs/scores individuels au LambdaMART SIRET.
Leur contribution arrive surtout via une présence agrégée et le classement
parent historique. Les14 canaux v36 provenaient des signaux SIRET directs et
structurés, et ne rétablissaient pas ces26 canaux de découverte.

Diagnostic label-side sur les28 pertes de classement v36 :15 vérités ont
leur parent dans un de ces canaux, toutes à un rang<=100, dont9<=10 et3 au
rang1. Ce constat ne vaut pas15 récupérations futures et ne sélectionne
aucun canal : TOUS les canaux de la configuration v16 sont inclus, y compris
ceux sans observation dans un lot. Les sources temporelles historiques
existent déjà, comme établi en v27 ; aucune nouveauté de corpus revendiquée.

La recette unique conserve le Master, les131 features v36, les paramètres
de fit complets v36 et le protocole nested OOF. Elle ajoute uniquement le
rang réciproque et le score relatif au maximum de chaque canal/requête pour
les26 canaux parent. Les scores sont joints par SIREN officiel, sans donner
l'identifiant au modèle. Une preuve parent ne devient pas une preuve de
nom conjoint au site. La portée est une inférence sur le dossier entreprise.
Les rôlesRNE proviennent de l'index de personnes morales qui exclut les
personnes physiques ; aucun nom de dirigeant n'est introduit.

Ces52 features sont calculées à partir des sorties scellées label-blind,
avant les fits, et n'utilisent ni vérité, ni ancien hit/rang de vérité, ni
qualification. Le modèle score au plus2 000 références, puis le même ordre
donne500 et100. Source v36 reprise et hachée, folds2/3/4 uniquement,
équivalents opérationnels neutralisés, absences gardées au dénominateur,
fold1 fermé. Tous les résultats, y compris régressions, sont publiés.

Un processus par fold, séquentiel, 2 threads,5GioRSS,300s/fold,zéro nouveau
swapout,900s maximum au total. Aucun nouveau téléchargement, index ou canal
de génération. La comparaison au v36 à features plus pauvres est explicative;
le v37 ne peut sélectionner un modèle par requête ni retoucher ses paramètres
après les résultats. Les hyperparamètres et la cible99,2% restent inchangés.

Ce second diagnostic reste sur le même panel de développement réutilisé.
Il ne prouve pas99,2% complet, ni cross-CRM, ni SLO. Conclusion v12 explicite,
et jamais READY_FOR_FOLD1 sur ce pilote. La correction de génération reste
nécessaire : le plafond du Master complet est98,96059%.
