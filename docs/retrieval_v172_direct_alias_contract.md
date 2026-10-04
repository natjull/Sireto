# V172 — coût et équivalence d'une exécution directe des textes

Protocole fixé avant mesure, 4 octobre 2026. Ce test porte sur le lot physique
49 déjà certifié de V168, pas sur un sous-ensemble de candidats. Il ne mesure
ni recall, ni SLO national, ni un retrieval complet. Les autres lots continuent
à être compilés, sans redémarrage ni modification de leur implémentation.

Les feature_text exportés V166 sont égaux au normaliseur scalaire, et chaque
ensemble de postings a été vérifié exhaustivement. Pour chaque alias entier,
V172 compte la présence de chaque token distinct de la requête : mots avec
frontières d'espace, char3/char4 avec espaces remplacés par underscores,
max des trois fractions en float64 et flag exact du texte normalisé non vide.
Il ne change ni score, masque identifiant, texte, ni scope d'adresse. Les
IDs opaques servent exclusivement à joindre les preuves, jamais comme feature.

La première requête query-only V115, ordre antérieur, fournit sa rue. Tous ses
128 enregistrements sont contrôlés folds234 sansGT avant sélection de cette
première requête. Aucun candidat ancien, label, fold0/1 ou modèle aval ouvert.

Oracle indépendant : parcourir TOUS les fichiers de postings du lot par groupes
physiques de quatre lecteurs maximum et sommer les compteurs de tokens par
alias/channel avant division. Comparer chaque float64 et bool, pas un simple
compteur global. Vérifier IDs contigus, toutes les lignes, hashes de chaque
artifact avant et après, même autorité complète INIT V168/V163 et epochs.
Ne pas présenter ce lot comme corpus national complet.

Worker et parent supervisés V153/V164 : 1Gio/80s, pression réelle1/free>=50%,
swap global télémétrie non attribuée et non bloquante, préflight30s. DuckDB
un thread/256Mio/spill3Gio. Root fresh immutable, vrai exit0 nécessaire.
Mesurer direct et oracle séparément, toute lecture/validation dans budget.
Un échec conserve ses fichiers et n'est jamais recyclé en succès.

Si le coût est intéressant, une prochaine version devra certifier TOUS les
576 lots et le namespace complet avant toute sortie initial100/évaluation.
Pas d'extrapolation linéaire présentée comme temps national mesuré. Dernier
recall exact V101114/128 inchangé, objectif99,2% non prouvé, noREADY.
