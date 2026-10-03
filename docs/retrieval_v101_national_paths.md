# V101 — plan physique V99 inchangé

V99 national complet s'est arrêté mémoire au second lot : RSS6628098048
pour limite6442450944,52,767823s,zéro swap.15 requêtes certifiées dans logs
mais seulement8 dans lot scellé ; aucun recall/labels ouverts.8 premiers
temps moyenne6,231237s non gate16. Les appels natifs ponctuels répétés
coûtent7,940397s sur la première requête du deuxième lot (14 appels).

V101 garde EXACTEMENT les preuves/champs/poids64/grades/corpus V99 décrits
dans son contrat. Même métrique, nouveau plan physique uniquement :
dictionnaire de mots projeté sur les tokens CRM nécessaires au lot en cours,
NON projection des faits/sites/DF. Toutes vocabulaires3,6M mots sont lues et
comptées, tous109M postings restent accessibles sur tout57804962 faits.
Les tokens CRM ne deviennent pas alias de source ni feature apprise.
Appel avec token non préparé STOP, jamais score0 silencieux. Arithmétique
et traces des scores identiques au lecteur global pour tout token demandé.
Chargement de requête prod peut préparer ses mots de la même manière ; le
panel n'est pas un index alias ni une source conditionnée au GT.

Completion physique8192 et flush doublons65536, contre512/4096 : aucun
plafond de fanout ni pool fini, tous endpoints de chaque page consommés
avant avance de frontière, sortie heap100 certifiée inchangée. Aucun
nouveau score, budget élargi, label lu, modèle aval ou artefact ancien modifié.

Mêmes guards80s/6Gio/zéro swap/90s superviseur/900s contrôleur/20s requête,
préflight16 moyenne<=4s puis128. Comparateur V91 en évaluation uniquement
après128 scellées. Folds2/3/4, all-human, exact/op, zéro injection, fold0/1
fermés. Pas SOTA garanti ; fullOOF/Wilson/généralisation/SLO restent requis.
