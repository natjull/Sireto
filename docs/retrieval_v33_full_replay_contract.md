# V33 — rejeu apparié depuis les entrées CRM complètes

11 septembre 2026. Autorisation utilisateur « Go » : intégrer le chemin complet,
contrôle technique court puis lancement complet si conforme. Master2000 seul;
pas d'autorisation de fit, d'admission500/100, d'ouverture fold1 ou de déploiement.

## Science inchangée

Comparer `v23_stable` et `v30_stable` sur les23 571 entrées originales des
folds2/3/4, sans sous-sélection selon les succès, échecs ou qualifications.
Toutes les requêtes restent au dénominateur, même sans aucun candidat.
V30 ajoute seulement son canal conjoint nom/adresse. Réparation v32 commune,
modèles parent v23 gelés, caps et features inchangés, aucun retuning après lecture.
Les recettes historiques demeurent immuables.

Les colonnes d'entrée sont projetées au scanner : query_id, oof_fold, crm_name,
crm_address, postcode, insee. Seuls les champs CRM entrent dans RetrievalQuery;
query_id sert à l'assemblage, fold à l'affectation du modèle OOF. Aucun ledger,
ancien pool, SceneIR ou candidat par query_id n'alimente la génération.

Une autorité explicite d'inférence étend la portée des modèles : le contrôle
v31 scellé prouve l'absence de chevauchement SIREN entraînement/cible sur cette
population. Il doit couvrir exactement les mêmes modèles et inputs. Exporter
uniquement le scope query_id/fold pour l'inférence. Ne pas supprimer la garde
des anciens runners. Modèles ajustés sur le petit panel : limite à publier,
pas une validation indépendante ni une preuve cross-CRM.

## Reconstruction et ressources

Legacyv11 avec les branchements restaurés v31; recherche parent nativev16;
BAN/géographie/adressev17; relations RNE/BODACC/SIRENE; dix canaux directsv18;
canal conjointv30. Calculs de génération communs partagés avant deux fusions
et inférences distinctes. Aucun nouveau canal, seuil ou source.

Le helper géographique historique déduit un identifiant parent : le runner
v33 vérifie ce résultat contre le SIREN **stocké officiellement pour chaque
voisin considéré**, avant de consommer sa présence. Un écart arrête le run.
Les témoins nearest_siret géographiques sont diagnostics, pas des features;
seul l'ordre des parents géographiques alimente cette recette.

La succession instable éventuellement produite dans un intermédiaire par
le helper v18 est systématiquement jetée avant scoring, remplacée par v32.
Son intermédiaire est conservé pour audit, jamais injecté dans un Master.

Lots indépendants de32 requêtes, processus successifs pour borner caches/RSS,
sorties sur CATNAT et modèles CPU seuls. Gardes par lot :30min,14Gio de RSS,
zéro swapout machine. Ne pas annoncer ce RSS comme une mesure RAM machine.
Plafond orchestrateur36h, espace disque restant au moins30Gio. Aucun retry
silencieux d'une erreur; STOP et conservation des lots déjà scellés.

Manifestes, meta index et inventaires de fichiers scellés; sources tabulaires
et100 fichiers temporels intégralement hashés à la préparation. Stat/inode/
taille/mtime vérifiés ensuite avant et après chaque lot; code/config hashés.
Les gros segments Tantivy ne sont pas intégralement rehashés : limite explicite.

Amendement physique après le premier pilote sans labels :224,18s/32, projection
45,87h, donc pas de lancement sous la garde36h. EXPLAIN révèle un
BLOCKWISE_NL_JOIN pour la restriction des relations sur predecessor OR successor.
Remplacer uniquement cet opérateur par deux semi-jointures d'égalité et UNION
distinct : même ensemble de lignes officielles, NULL/doublons inclus. Référence
exhaustive contre les relations matérialisées du premier pilote et tests
unitaires avant nouveau gel. Nouvelle préparation/essais dans une racine
distincte; le pilote lent et ses sorties sont conservés. Aucun seuil de coût,
de pertinence, aucune source, feature ou métrique modifiés.

## Contrôle avant long run

1. Rejouer les16 entrées de parité v32 depuis le CRM jusqu'aux Masters.
2. Tirer32 entrées par tri SHA256("v33-cost|" + query_id) sur **tout** le CRM,
   sans lire les labels. C'est un essai de coût, pas de qualité ni le profil
   SLO512 du contratv12.
3. Régénérer ces mêmes32 dans un autre processus, ordre inversé et lots8.
   Comparer exactement raw/proof/structured/direct/joint/pools/Masters.
   Aucun réglage de pertinence ni nouveau score autorisé pendant ce contrôle.
4. Vérifier la parité16 avec les Masters stables v32. Estimer le coût complet
   par extrapolation linéaire du premier essai32 seulement; publier la forte
   incertitude. Refuser si projection>36h ou>100Gio ou espace libre inférieur
   à deux fois la projection (minimum50Gio). Ce n'est pas une garantie de durée.
5. Sceller le gate avant lancement et vérifier ses hashes. Le complet
   régénère aussi les requêtes des essais; leurs candidats ne sont pas recyclés.

## Évaluation séparée et verdict

La génération complète se scelle sans lire les labels. Ensuite seulement,
reconstruire l'audit officiel label-side de toute la population; publier
catalogue/parent/site/éviction2000, exact et opérationnel séparés, les vues
historique/V2/V3 avec valeurs manquantes explicites, et les courbes de couverture.
La proximité seule ne prouve pas l'équivalence opérationnelle.

Le présent runner livre génération et preuves, pas une métrique automatique.
Un état SEALED_FULL_PAIRED_GENERATION implique encore évaluation et réconciliation
exhaustive; il ne signifie pas un gain de recall. Aucun READY_FOR_FOLD1.
Une preuve d'intégrité ou d'attribution manquante impose STOP. Les résultats
Master ne franchissent pas les gates PreRank500/Recall100 ni les SLO.
