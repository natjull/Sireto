# V199 — toutes les 128 recherches initiales, calcul de noms RAM exact

Protocole avant mesure. Nouvelle racine indépendante, aucune preuve, capsule
ou sortie réutilisée des STOP V197/V198. La tentative V198 a passé son coût
q4 et son hash FULL, mais a STOP sur une erreur de comparaison de formats
de référence. Son admission n'est pas réparée rétrospectivement.

La génération complète V197 est conservée : ALL128 requêtes figées V192,
folds2/3/4, leurs propres vecteurs/heads/preuves, ALL44403034sites et
98804872faits, même formule V182 et collecteur exact V183 <=100 sans pool
fixe ni re-score. Modèles aval gelés. Historiques, coordonnées/BAN et preuves
parent/site/succession restent exactement identiques ; pas nouvelles
absorptions/TUP BODACC inventées. Pas alias CRM comme source officielle,
identifiants comme features ni GT, anciennes sorties ou folds0/1 ouverts.

Seul calcul de noms changé physiquement : RamCountersNames V198 mesuré,
compteur intégral <=64Mio, mêmes tokens/postings/aliases et grades, sinon
fallback disque V170 sans troncature. Corpus complet57530682aliases.
Float64/bool et preuves parent/site/one-hop inchangées. Les tests indépendants
vérifient les valeurs bit-exactes ; ils ne prédisent pas le recall.

Ordre physique fixe sans labels : names000, addresses000, score000-000 créent
la capsule commune ; puis names001..063, addresses001..063 et tous les
scorers restants. Exactement384 producteurs une fois chacun. Ce changement
détecte un nouveau STOP de noms avant les longs scorings suivants, pas une
sélection de clients. Ordre canonique du plan/audit reste celui V197 pour
conserver tous les contrôles. Capsule/audit utilise le helper V197 IMMUTABLE
(tags internes historiques conservés), explicitement hashé dans ce launch.

Chaque preuve doit appartenir à l'inventaire FULL et aux epochs pré-lecture.
Comparaison canonique path+SHA256 pour formats avec/sans size_bytes ; la
taille lorsqu'elle existe reste contrôlée contre l'epoch original et verify
re-hashe entièrement la preuve. Une référence absente ou un mauvais hash,
size/epoch ne passe pas. Test explicite des DEUX formats, pas seulement des
refs déjà normalisées comme dans les tests V198.

TOUTES les sources externes exactes, preuves512vecteurs, producteurs et
certificats sont scellés dans le plan après TOUS les calculs. Hash FULL par
chunks <=5Gio/4096fichiers, puis64merges sans re-score et ledger93507 final.
Chaque source/epoch original reste conservé ; capsule seule, stat ou sample
ne vaut pas audit. Aucun initial100 admis avant l'ensemble128 terminé.

Budgets réels inchangés : query de preuve et flatten20s chacun, scoring
30s/partition, worker80s/1Gio RSS, quiet30, pression1/libre>=50%, swap GLOBAL
observé non attribué/non bloquant. Correction explicite de l'erreur de texte
30s de preuves dans le contrat historique V197, laissé intact. Mono-thread
séquentiel sur Mac/CATNAT ; pas cloud, achat, purge ou application arrêtée.

Code/config/contrat/tests commités et handover avant mesure. STOP immuable,
sans relance élargie ni réemploi de fragments. Plusieurs heures possibles,
cache non contrôlé : aucun SLO validé. Les labels ne sont ouverts dans un
évaluateur séparé QU'APRÈS SEALED des128 sorties. Publier ensuite exact/op,
denominateur all-human, absences/V2/V3/segments et sidecar officiel actuel.
Pilote réutilisé128 n'autorise jamais READY même à100%; objectif99,2%, Wilson,
OOF complet, généralisation et SLO restent à prouver. Aucun claim SOTA.
