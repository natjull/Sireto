# V6 — Finalisation sans étape de training supplémentaire

À 06:43 le superviseur V4 a envoyé SIGTERM pour échéance, puis SIGKILL après
45 secondes pendant la validation finale. Owned exit -9, pression1/free65%,
RSS1,75Go : ce n'est pas une preuve d'OOM. Le dernier checkpoint durable est
conservé ; la comparaison n'a pas commencé et aucun nouveau label n'a été lu.

V6 exige cette interruption réellement confirmée et l'absence de processus
d'entraînement original. Il copie poids/optimizer/cursor et meilleur checkpoint
dans un namespace neuf, rebind le launch, préserve les deux échéances, puis
appelle le worker original avec son échéance de fit déjà passée. Les boucles
de training sont donc sautées ; seule la validation finale originale s'exécute.
Le numéro d'étape et l'époque doivent rester exactement ceux du checkpoint.
La couverture réelle et le statut TIME_BUDGET_CHECKPOINTED sont publiés.

Les 80 requêtes, 119064 établissements complets, textes officiels, trois
ablations, lexical national et recettes RRF restent identiques. Aucun corpus
ou minage n'est relancé. Chaque génération doit être complète, scellée avec
owned exit0 avant labels. Les sorties géographiques V5 déjà scellées et leur
vrai reçu d'exit sont réutilisés explicitement, sans regénération.

Tous les anciens fichiers/code/config restent immuables. V6 importe les
implémentations originales hashées et ajoute son propre binding d'exécution.
Échéance absolue toujours 08:43, pas nouvelle nuit ni nouvelles étapes de fit.
Test4323/folds0/1 et modèles aval restent fermés/gelés. Les métriques sur80
requêtes sont diagnostiques, jamais une validation nationale99,2%.
