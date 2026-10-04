# V236/V237 : accès aux variantes officielles entières dans le texte CRM

Recette figée avant émission,5 octobre2026. Objectif99,2% exact initial100.
V234/V235 a régressé117/128 vs120/128, malgré un meilleur contrôle TRAIN
censuré. Pas de promotion, pas d'union oracle des succès des deux recettes.
V229/V230 reste référence de développement. Onze poids77 sauf place228,
alpha/vocab/projections/expansions et composants natifs217 inchangés.

Nouvelle hypothèse SOURCE, pas nouveau fit : toutes variantes officielles
entières du catalogue national134 peuvent être contenues dans le nom CRM.
La recherche full-query exact/all-word210 demeure. Les spans contigus217
éligibles sont interrogés en EXACT sur les chaînes officielles134, incluant
leurs noms historiques/sigles préexistants, via postings134/owners132/149.
Pas de regex préfixe, d'acronyme inventé ni de nom extrait du sidecar label.
Pas d'import des308 noms103 sélectionnés par les GT.

Une variante UL se propage à ses établissements via le seul SIREN officiel ;
une variante de site se limite à CE SIRET, pas ses frères. Tous propriétaires
et liens de succession un hop parcourus, aucun cap/fanout/topk intermédiaire.
Le singleton contenu dans un nom plus long a confiance1/nombre tokens de
contenu hors particules/formes légales. Les spans multiword ont1. Il s'agit
d'une règle générale de confiance, PAS d'un dictionnaire d'alias CRM.
MAX des preuves d'un propriétaire (anciens ET nouveaux), jamais bonus
additionnels pour plusieurs alias. Les successions gardent demi-confiance.
Bonus propriétaire conditionné par MAX(adresse,place), comme217 ; il ne
devient pas identité confirmée. Hypothèse source fixe, pas apprise sur128.

ONE score natif global avant collecteur100, pas reservoir fixe/reranker.
XY/BAN faible et géographie administrative restent soft, fermés/hors-zone
et coordonnées absentes accessibles. Native43896818 feuilles, pas append
44403034 annoncé. Les identifiants sont uniquement jointures/sortie.

128 requêtes CRM originales234, aucun GT/ancien hit/candidat en génération.
16 workers8 frais, tous actualexit0/PASS214/scellés avant évaluateur238.
Gardes214 inchangées2Gio/20s query ouverture incluse/80s worker/un thread,
quiet30/pression1/free>=50. Source trop coûteuse = STOP complet, ni hausse
des ressources ni recyclage partiel. Sources utilisées FULL SHA/epochs
pré/post ; natif inventaire/méta, segments gros non entièrement rehashés.

238 publie exact/op séparés, même dénominateur all-human/cibles/sources,
Wilson/folds/qualifications/absences et comparaison appariée230. Aucun
label corrigé/écarté, aucun positif injecté. Pilote128 réutilisé, donc pas
preuve fullOOF/généralisation/SLO ni READY même100%. Fold0 interdit,fold1
fermé, modèles aval gelés, Mac/CATNAT seulement, zéro dépense externe.
